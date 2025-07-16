import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import base64
import tempfile
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import os


st.title("Опън в покритието")

# Зареждане на данните
@st.cache_data
def load_data():
    return pd.read_csv("sigma_data.csv")

data = load_data()

# Функция за изчисляване на σR
def compute_sigma_R(H, D, Esr, Ed):
    hD = H / D
    Esr_Ed = Esr / Ed
    tol = 1e-3
    iso_levels = sorted(data['Esr_over_Ed'].unique())

    for low, high in zip(iso_levels, iso_levels[1:]):
        if not (low - tol <= Esr_Ed <= high + tol):
            continue

        grp_low = data[data['Esr_over_Ed'] == low].sort_values('H_over_D')
        grp_high = data[data['Esr_over_Ed'] == high].sort_values('H_over_D')

        h_min = max(grp_low['H_over_D'].min(), grp_high['H_over_D'].min())
        h_max = min(grp_low['H_over_D'].max(), grp_high['H_over_D'].max())
        if not (h_min - tol <= hD <= h_max + tol):
            continue

        y_low = np.interp(hD, grp_low['H_over_D'], grp_low['sigma_R'])
        y_high = np.interp(hD, grp_high['H_over_D'], grp_high['sigma_R'])

        frac = 0 if np.isclose(high, low) else (Esr_Ed - low) / (high - low)
        sigma = y_low + frac * (y_high - y_low)

        return sigma, hD, y_low, y_high, low, high

    return None, None, None, None, None, None

# Заглавна част
st.title("Определяне опънното напрежение в долния пласт на покритието фиг.9.2")

# Въвеждане на параметри
st.markdown("### Въвеждане на параметри на пластове")

# Зареждане на стойности от session_state или задаване на дефолтни
D_default = st.session_state.get("final_D", 34.0)
Ei_list_full = st.session_state.get("Ei_list", [])
hi_list_full = st.session_state.get("hi_list", [])
Ei_list_default = Ei_list_full[:2] if len(Ei_list_full) >= 2 else [1000.0, 1000.0]
hi_list_default = hi_list_full[:2] if len(hi_list_full) >= 2 else [10.0, 10.0]

if len(Ei_list_full) > 2:
    st.info("ℹ️ Използват се само първите два пласта от въведените на предишната страница.")

# Избор на диаметър
D = st.selectbox(
    "Диаметър на отпечатъка на колело D (см)",
    options=[34.0, 32.04, 33.0],
    index=[34.0, 32.04, 33.0].index(D_default) if D_default in [34.0, 32.04, 33.0] else 0
)

# Брой пластове (фиксиран на 2)
st.markdown(f"**Брой пластове:** 2 (фиксиран за това изчисление)")
n = 2

# Въвеждане на параметри за двата пласта
Ei_list = []
hi_list = []

st.markdown("#### Въвеждане на Eᵢ и hᵢ за всеки пласт:")
for i in range(1, n + 1):
    col1, col2 = st.columns(2)
    with col1:
        Ei = st.number_input(
            f"E{i} (MPa)",
            key=f"Ei_{i}",
            value=Ei_list_default[i - 1] if i - 1 < len(Ei_list_default) else 1000.0
        )
    with col2:
        hi = st.number_input(
            f"h{i} (см)",
            key=f"hi_{i}",
            value=hi_list_default[i - 1] if i - 1 < len(hi_list_default) else 10.0
        )
    Ei_list.append(Ei)
    hi_list.append(hi)

# Запазване на параметрите
st.session_state["final_D"] = D
st.session_state["Ei_list"] = Ei_list
st.session_state["hi_list"] = hi_list

# Вземане на Ed от първата страница
st.markdown("---")
if "final_Ed_list" not in st.session_state:
    st.error("⚠️ Липсва final_Ed_list в session_state!")
    st.info("Моля, върнете се на първата страница и изчислете всички пластове")
    st.stop()

# Автоматично определяне на Ed (модул на следващия пласт)
n_layers = len(Ei_list)
if len(st.session_state.final_Ed_list) <= n_layers:
    st.error(f"⚠️ Недостатъчно пластове в final_Ed_list (изисква се поне {n_layers+1})!")
    st.stop()

Ed = st.session_state.final_Ed_list[n_layers-1]
st.session_state["final_Ed"] = Ed

st.markdown(f"""
#### 🟢 Стойност за Ed (модул на деформация на земното основание)
- Взета от пласт {n_layers} 
- Ed = {Ed:.0f} MPa
""")

# Изчисляване на Esr и H
numerator = sum(Ei * hi for Ei, hi in zip(Ei_list, hi_list))
denominator = sum(hi_list)
Esr = numerator / denominator if denominator != 0 else 0
H = denominator

# Показване на формулите
st.markdown("### ℹ️ Формули за изчисление")
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

# Показване на заместени стойности
numerator_str = " + ".join([f"{Ei}×{hi}" for Ei, hi in zip(Ei_list, hi_list)])
denominator_str = " + ".join([f"{hi}" for hi in hi_list])
st.latex(fr"Esr = \frac{{{numerator_str}}}{{{denominator_str}}} = {Esr:.2f} \text{{ MPa}}")
st.latex(fr"H = {denominator_str} = {H:.2f} \text{{ см}}")

# Автоматично изчисляване на σR
if denominator != 0:
    sigma, hD, y_low, y_high, low, high = compute_sigma_R(H, D, Esr, Ed)
    
    if sigma is not None:
        # Запазване на резултатите
        st.session_state["final_sigma"] = sigma
        st.session_state["final_hD"] = hD
        st.session_state["final_y_low"] = y_low
        st.session_state["final_y_high"] = y_high
        st.session_state["final_low"] = low
        st.session_state["final_high"] = high
        
        # Показване на резултатите
        st.markdown("## 📋 Резултати от изчисленията")
        st.markdown(f"""
        **Изчислено:**
        - $Esr / Ed = {Esr:.2f} / {Ed:.0f} = {Esr / Ed:.3f}$
        - $H / D = {H:.2f} / {D:.2f} = {H / D:.3f}$
        """)
        st.success(f"✅ σR = {sigma:.3f}")
        st.info(f"Интерполация между изолинии: Esr/Ed = {low:.2f} и {high:.2f}")

        # Графика
        fig = go.Figure()
        for val, group in data.groupby("Esr_over_Ed"):
            fig.add_trace(go.Scatter(
                x=group["H_over_D"],
                y=group["sigma_R"],
                mode='lines',
                name=f"Esr/Ed = {val:.1f}"
            ))
        fig.add_trace(go.Scatter(
            x=[H / D], y=[sigma],
            mode='markers',
            marker=dict(size=8, color='red'),
            name="Твоята точка"
        ))
        fig.update_layout(
            title="Номограма: σR срещу H/D",
            xaxis_title="H / D",
            yaxis_title="σR",
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❗ Точката е извън диапазона на наличните данни.")
        for key in ["final_sigma", "final_hD", "final_y_low", "final_y_high", "final_low", "final_high"]:
            if key in st.session_state:
                del st.session_state[key]
else:
    st.error("Сумата на hᵢ не може да бъде 0.")

# Изображение на допустимите напрежения
st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=800)

# Автоматично изчисляване на крайното σR
axle_load = st.session_state.get("axle_load", 100)
p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else None

if p is not None:
    st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
    st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")
    
    sigma = st.session_state.get("final_sigma", None)
    
    if sigma is not None:
        sigma_final = 1.15 * p * sigma
        st.markdown("### Формула за изчисление на крайното напрежение σR:")
        st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
        st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
        st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
        
        # Запазване на крайната стойност
        st.session_state["final_sigma_R"] = sigma_final
    else:
        st.warning("❗ Липсва σR от номограмата за изчисление.")
else:
    st.warning("❗ Не е зададен валиден осов товар. Не може да се изчисли p.")

# Секция за ръчно въвеждане
st.markdown(
    """
    <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
        <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Инициализиране на ръчната стойност
if 'manual_sigma_value' not in st.session_state:
    st.session_state.manual_sigma_value = 1.2

# Поле за ръчно въвеждане
manual_value = st.number_input(
    label="Въведете допустимо опънно напрежение σR [MPa] (от таблица 9.7)",
    min_value=0.0,
    max_value=20.0,
    value=st.session_state.manual_sigma_value,
    step=0.1,
    key="manual_sigma_input",
    label_visibility="visible"
)

# Запазване на въведената стойност
st.session_state.manual_sigma_value = manual_value

# Автоматична проверка на условието
sigma_to_compare = st.session_state.get("final_sigma_R", None)

if sigma_to_compare is not None:
    check_passed = sigma_to_compare <= manual_value
    if check_passed:
        st.success(
            f"✅ Проверката е удовлетворена: "
            f"изчисленото σR = {sigma_to_compare:.3f} MPa ≤ {manual_value:.3f} MPa (допустимото σR)"
        )
    else:
        st.error(
            f"❌ Проверката НЕ е удовлетворена: "
            f"изчисленото σR = {sigma_to_compare:.3f} MPa > {manual_value:.3f} MPa (допустимото σR)"
        )
else:
    st.warning("❗ Няма изчислена стойност σR (след коефициенти) за проверка.")

# Линк към предишната страница
st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")

# В края на файла Опън в покритието.py (след секцията за ръчно въвеждане)

# =====================================================================
# ДОБАВЕН КОД ЗА PDF ГЕНЕРАЦИЯ
# =====================================================================

# Клас за персонализиран PDF с български шрифтове
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_font_files = []
        
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
        
    def add_font_from_bytes(self, family, style, font_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
            tmp_file.write(font_bytes)
            tmp_file_path = tmp_file.name
            self.temp_font_files.append(tmp_file_path)
            self.add_font(family, style, tmp_file_path)
            
    def cleanup_fonts(self):
        for file_path in self.temp_font_files:
            try:
                os.unlink(file_path)
            except Exception:
                pass

# Функция за генериране на PDF отчет
def generate_tension_report():
    # Създаване на PDF документ
    pdf = PDF()
    
    # Зареждане на шрифтове (примерни, трябва да ги имате в папка fonts)
    try:
        # Зареждане на шрифтове от локална папка
        font_dir = "fonts"
        dejavu_sans = open(os.path.join(font_dir, "DejaVuSans.ttf"), "rb").read()
        dejavu_bold = open(os.path.join(font_dir, "DejaVuSans-Bold.ttf"), "rb").read()
        dejavu_italic = open(os.path.join(font_dir, "DejaVuSans-Oblique.ttf"), "rb").read()
        
        pdf.add_font_from_bytes('DejaVu', '', dejavu_sans)
        pdf.add_font_from_bytes('DejaVu', 'B', dejavu_bold)
        pdf.add_font_from_bytes('DejaVu', 'I', dejavu_italic)
    except Exception as e:
        st.error(f"Грешка при зареждане на шрифтове: {str(e)}")
        return None

    pdf.set_font('DejaVu', '', 12)
    pdf.add_page()
    
    # Заглавие
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'Изчисление на опън в покритието', 0, 1, 'C')
    pdf.ln(5)
    
    # Дата
    pdf.set_font('DejaVu', '', 12)
    today = datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.cell(0, 8, f'Дата: {today}', 0, 1)
    pdf.ln(10)
    
    # Параметри на пластовете
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 8, 'Параметри на пластовете', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    
    # Извличане на данни от session state
    D = st.session_state.get("final_D", 34.0)
    Ei_list = st.session_state.get("Ei_list", [1000, 1000])
    hi_list = st.session_state.get("hi_list", [10, 10])
    Ed = st.session_state.get("final_Ed", 100)
    
    # Таблица с параметрите
    col_widths = [40, 40, 40, 40]
    headers = ["Пласт", "Ei (MPa)", "hi (cm)", "Ed (MPa)"]
    
    # Хедъри на таблицата
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()
    
    # Данни за пластовете
    for i in range(len(Ei_list)):
        pdf.cell(col_widths[0], 10, str(i+1), 1, 0, 'C')
        pdf.cell(col_widths[1], 10, str(Ei_list[i]), 1, 0, 'C')
        pdf.cell(col_widths[2], 10, str(hi_list[i]), 1, 0, 'C')
        # Показва Ed само за последния пласт
        if i == len(Ei_list) - 1:
            pdf.cell(col_widths[3], 10, str(round(Ed)), 1, 0, 'C')
        else:
            pdf.cell(col_widths[3], 10, "-", 1, 0, 'C')
        pdf.ln()
    
    pdf.ln(10)
    
    # Изчислени параметри
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 8, 'Изчислени величини', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    
    H = sum(hi_list)
    Esr = sum([Ei * hi for Ei, hi in zip(Ei_list, hi_list)]) / H
    hD = H / D
    Esr_Ed = Esr / Ed
    
    pdf.cell(0, 8, f'Диаметър (D): {D} cm', 0, 1)
    pdf.cell(0, 8, f'Сума на дебелините (H): {H:.2f} cm', 0, 1)
    pdf.cell(0, 8, f'Еквивалентен модул (Esr): {Esr:.2f} MPa', 0, 1)
    pdf.cell(0, 8, f'Модул на основание (Ed): {Ed} MPa', 0, 1)
    pdf.cell(0, 8, f'H/D: {hD:.4f}', 0, 1)
    pdf.cell(0, 8, f'Esr/Ed: {Esr_Ed:.4f}', 0, 1)
    
    # Резултати от номограмата
    if "final_sigma" in st.session_state:
        sigma_nomogram = st.session_state["final_sigma"]
        axle_load = st.session_state.get("axle_load", 100)
        p = 0.620 if axle_load == 100 else 0.633
        
        pdf.ln(10)
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 8, 'Резултати от номограмата', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        pdf.cell(0, 8, f'σR (номограма): {sigma_nomogram:.4f}', 0, 1)
        pdf.cell(0, 8, f'Осова тежест: {axle_load} kN → p = {p}', 0, 1)
        pdf.cell(0, 8, f'Крайно σR = 1.15 * p * σR = 1.15 * {p} * {sigma_nomogram:.4f} = {1.15*p*sigma_nomogram:.4f} MPa', 0, 1)
        
        # Ръчно въведена стойност и проверка
        manual_sigma = st.session_state.get("manual_sigma_value", 1.2)
        sigma_final = 1.15 * p * sigma_nomogram
        check_status = "УДОВЛЕТВОРЯВА" if sigma_final <= manual_sigma else "НЕ УДОВЛЕТВОРЯВА"
        
        pdf.ln(10)
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 8, 'Проверка спрямо допустимо напрежение', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        pdf.cell(0, 8, f'Ръчно зададено допустимо напрежение: {manual_sigma} MPa', 0, 1)
        pdf.cell(0, 8, f'Сравнение: {sigma_final:.4f} MPa ≤ {manual_sigma} MPa → {check_status}', 0, 1)
    
    # Добавяне на изображение (ако съществува)
    img_path = "Допустими опънни напрежения.png"
    if os.path.exists(img_path):
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 8, 'Допустими опънни напрежения', 0, 1)
        pdf.image(img_path, x=10, w=190)
    
    pdf.cleanup_fonts()
    return pdf.output(dest='S').encode('latin1')

# Добавяне на бутон за генериране на PDF
st.markdown("---")
st.subheader("Генериране на отчет")

if st.button("📊 Генерирай PDF отчет", key="generate_pdf_button"):
    with st.spinner('Генерира се PDF отчет...'):
        pdf_bytes = generate_tension_report()
        
        if pdf_bytes:
            # Сваляне на файла
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(pdf_bytes)
            
            with open(tmpfile.name, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="opyn_v_pokritieto_report.pdf">⬇️ Свали PDF отчета</a>'
                st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.error("Грешка при генериране на PDF отчета")
