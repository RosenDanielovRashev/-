import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from fpdf import FPDF
from PIL import Image
import plotly.io as pio
import os
import tempfile
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import mathtext
from datetime import datetime

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
- Ed = {Ed:.2f} MPa
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
numerator_str = " + ".join([f"{Ei:.2f}×{hi:.2f}" for Ei, hi in zip(Ei_list, hi_list)])
denominator_str = " + ".join([f"{hi:.2f}" for hi in hi_list])
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
        - $Esr / Ed = {Esr:.2f} / {Ed:.2f} = {Esr / Ed:.3f}$
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
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Запазване на фигурата в session_state
        st.session_state["fig"] = fig
        
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
    st.session_state.manual_sigma_value = 1.20

# Поле за ръчно въвеждане
manual_value = st.number_input(
    label="Въведете допустимо опънно напрежение σR [MPa] (от таблица 9.7)",
    min_value=0.0,
    max_value=20.0,
    value=st.session_state.manual_sigma_value,
    step=0.01,
    key="manual_sigma_input",
    format="%.2f",
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
            f"изчисленото σR = {sigma_to_compare:.3f} MPa ≤ {manual_value:.2f} MPa (допустимото σR)"
        )
    else:
        st.error(
            f"❌ Проверката НЕ е удовлетворена: "
            f"изчисленото σR = {sigma_to_compare:.3f} MPa > {manual_value:.2f} MPa (допустимото σR)"
        )
else:
    st.warning("❗ Няма изчислена стойност σR (след коефициенти) за проверка.")

def render_formula_to_image(formula, fontsize=12, dpi=200):
    """Render LaTeX formula to image with left alignment"""
    # Изчисляване на ширината въз основа на дължината на формулата
    width = max(8, min(12, len(formula) * 0.15))  # Динамична ширина
    
    fig = plt.figure(figsize=(width, 0.8))
    # Промяна: подравняване в ляво
    fig.text(0.02, 0.5, f'${formula}$', fontsize=fontsize, 
             ha='left', va='center', usetex=False)
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)
    return buf

class EnhancedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_font_files = []
        self.temp_image_files = []
        
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
            
    def add_external_image(self, image_path, width=180):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            img = Image.open(image_path)
            img.save(tmp_file, format='PNG')
            tmp_file_path = tmp_file.name
            self.temp_image_files.append(tmp_file_path)
        self.image(tmp_file_path, x=10, w=width)
        self.ln(10)
        
    def add_latex_formula(self, formula_text):
        try:
            img_buf = render_formula_to_image(formula_text)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_buf.read())
                tmp_file_path = tmp_file.name
                self.temp_image_files.append(tmp_file_path)
            
            self.image(tmp_file_path, x=10, w=180)
            self.ln(10)
        except Exception as e:
            self.set_font('DejaVu', 'I', 12)
            self.cell(0, 10, formula_text, 0, 1)
            self.ln(5)
            
    def add_plotly_figure(self, fig, width=180):
        try:
            img_bytes = pio.to_image(
                fig, 
                format="png", 
                width=1200, 
                height=900, 
                scale=3,
                engine="kaleido"
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file_path = tmp_file.name
                self.temp_image_files.append(tmp_file_path)
            
            self.image(tmp_file_path, x=10, w=width)
            self.ln(10)
            return True
        except Exception as e:
            print(f"Грешка при добавяне на Plotly фигура: {e}")
            return False
            
    def cleanup_temp_files(self):
        for file_path in self.temp_font_files + self.temp_image_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Грешка при изтриване на временен файл: {e}")
def generate_pdf_report():
    pdf = EnhancedPDF()
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_dir = os.path.join(base_dir, "fonts")
        
        if not os.path.exists(font_dir):
            os.makedirs(font_dir, exist_ok=True)
        
        sans_path = os.path.join(font_dir, "DejaVuSans.ttf")
        bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
        italic_path = os.path.join(font_dir, "DejaVuSans-Oblique.ttf")
        
        if not all(os.path.exists(p) for p in [sans_path, bold_path, italic_path]):
            from fpdf.fonts import FontsByFPDF
            fonts = FontsByFPDF()
            for style, data in [('', fonts.helvetica), ('B', fonts.helvetica_bold), ('I', fonts.helvetica_oblique)]:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                    tmp_file.write(data)
                    tmp_file_path = tmp_file.name
                    pdf.add_font('DejaVu', style, tmp_file_path)
        else:
            with open(sans_path, "rb") as f:
                dejavu_sans = BytesIO(f.read())
            with open(bold_path, "rb") as f:
                dejavu_bold = BytesIO(f.read())
            with open(italic_path, "rb") as f:
                dejavu_italic = BytesIO(f.read())
            
            pdf.add_font_from_bytes('DejaVu', '', dejavu_sans.getvalue())
            pdf.add_font_from_bytes('DejaVu', 'B', dejavu_bold.getvalue())
            pdf.add_font_from_bytes('DejaVu', 'I', dejavu_italic.getvalue())
    except Exception as e:
        st.error(f"Грешка при зареждане на шрифтове: {e}")
        return b""

    pdf.set_font('DejaVu', '', 12)
    pdf.add_page()
    
    # Заглавие
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'ОПЪН В ПОКРИТИЕТО - ОТЧЕТ', 0, 1, 'C')
    
    # Дата
    today = datetime.today().strftime("%d.%m.%Y")
    pdf.set_font('DejaVu', 'I', 10)
    pdf.cell(0, 10, f'Дата: {today}', 0, 1, 'R')
    pdf.ln(10)
    
    # Основна информация
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '1. Входни параметри', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    
    # Таблица за входни параметри
    col_width = 45
    row_height = 8
    
    # Заглавен ред
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font('DejaVu', 'B', 10)
    pdf.cell(col_width, row_height, 'Параметър', 1, 0, 'C', 1)
    pdf.cell(col_width, row_height, 'Стойност', 1, 0, 'C', 1)
    pdf.cell(col_width, row_height, 'Мерна единица', 1, 1, 'C', 1)
    
    # Данни
    pdf.set_font('DejaVu', '', 10)
    pdf.cell(col_width, row_height, 'Диаметър D', 1)
    pdf.cell(col_width, row_height, f"{st.session_state.final_D:.2f}", 1)
    pdf.cell(col_width, row_height, 'cm', 1, 1)
    
    pdf.cell(col_width, row_height, 'Брой пластове', 1)
    pdf.cell(col_width, row_height, '2', 1)
    pdf.cell(col_width, row_height, '', 1, 1)
    
    for i in range(2):
        pdf.cell(col_width, row_height, f'Пласт {i+1} - Ei', 1)
        pdf.cell(col_width, row_height, f"{st.session_state.Ei_list[i]:.2f}", 1)
        pdf.cell(col_width, row_height, 'MPa', 1, 1)
        
        pdf.cell(col_width, row_height, f'Пласт {i+1} - hi', 1)
        pdf.cell(col_width, row_height, f"{st.session_state.hi_list[i]:.2f}", 1)
        pdf.cell(col_width, row_height, 'cm', 1, 1)
    
    pdf.cell(col_width, row_height, 'Ed', 1)
    pdf.cell(col_width, row_height, f"{st.session_state.final_Ed:.2f}", 1)
    pdf.cell(col_width, row_height, 'MPa', 1, 1)
    
    axle_load = st.session_state.get("axle_load", 100)
    pdf.cell(col_width, row_height, 'Осова тежест', 1)
    pdf.cell(col_width, row_height, f"{axle_load}", 1)
    pdf.cell(col_width, row_height, 'kN', 1, 1)
    
    pdf.ln(10)
    
    # Формули
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '2. Формули за изчисление', 0, 1)
    
    formulas = [
        r"E_{sr} = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}",
        r"H = \sum_{i=1}^{n} h_i",
        r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}"
    ]
    
    for formula in formulas:
        pdf.add_latex_formula(formula)
    
    # Изчисления
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '3. Изчисления', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    
    # Изчисление на Esr и H
    numerator = sum(Ei * hi for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list))
    denominator = sum(st.session_state.hi_list)
    Esr = numerator / denominator if denominator != 0 else 0
    H = denominator
    
    numerator_str = " + ".join(
        [f"{Ei:.2f} \\times {hi:.2f}" 
         for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list)]
    )
    denominator_str = " + ".join(
        [f"{hi:.2f}" for hi in st.session_state.hi_list]
    )
    
    pdf.add_latex_formula(fr"E_{{sr}} = \frac{{{numerator_str}}}{{{denominator_str}}} = {Esr:.2f} \, \text{{MPa}}")
    pdf.add_latex_formula(fr"H = {denominator_str} = {H:.2f} \, \text{{cm}}")
    
    if 'final_sigma' in st.session_state:
        pdf.add_latex_formula(fr"\frac{{E_{{sr}}}}{{E_d}} = \frac{{{Esr:.2f}}}{{{st.session_state.final_Ed:.2f}}} = {Esr/st.session_state.final_Ed:.3f}")
        pdf.add_latex_formula(fr"\frac{{H}}{{D}} = \frac{{{H:.2f}}}{{{st.session_state.final_D:.2f}}} = {H/st.session_state.final_D:.3f}")
        pdf.add_latex_formula(fr"\sigma_R^{{nom}} = {st.session_state.final_sigma:.3f} \, \text{{MPa}}")
    
    # Изчисление на крайното σR
    p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
    if p and 'final_sigma' in st.session_state:
        sigma_final = 1.15 * p * st.session_state.final_sigma
        pdf.add_latex_formula(fr"p = {p:.3f} \, \text{{ (за осов товар {axle_load} kN)}}")
        pdf.add_latex_formula(fr"\sigma_R = 1.15 \times {p:.3f} \times {st.session_state.final_sigma:.3f} = {sigma_final:.3f} \, \text{{MPa}}")
    
    # Графика
    if "fig" in st.session_state:
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '4. Графика на номограмата', 0, 1)
        pdf.add_plotly_figure(st.session_state["fig"], width=160)
    
    # Допустими напрежения
    try:
        image_path = "Допустими опънни напрежения.png"
        if os.path.exists(image_path):
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, '5. Допустими опънни напрежения', 0, 1)
            pdf.add_external_image(image_path, width=160)
    except Exception as e:
        st.error(f"Грешка при добавяне на изображение: {e}")
    
    # Резултати и проверка
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '6. Резултати и проверка', 0, 1)
    pdf.set_font('DejaVu', '', 10)
    
    if 'final_sigma_R' in st.session_state and 'manual_sigma_value' in st.session_state:
        check_passed = st.session_state.final_sigma_R <= st.session_state.manual_sigma_value
        
        # Резултати в таблица
        pdf.set_fill_color(230, 240, 255)
        pdf.set_font('DejaVu', 'B', 10)
        pdf.cell(90, 8, 'Параметър', 1, 0, 'C', 1)
        pdf.cell(90, 8, 'Стойност', 1, 1, 'C', 1)
        
        pdf.set_font('DejaVu', '', 10)
        pdf.cell(90, 8, 'Изчислено σR', 1)
        pdf.cell(90, 8, f"{st.session_state.final_sigma_R:.3f} MPa", 1, 1)
        
        pdf.cell(90, 8, 'Допустимо σR (ръчно)', 1)
        pdf.cell(90, 8, f"{st.session_state.manual_sigma_value:.2f} MPa", 1, 1)
        
        pdf.ln(5)
        
        # Проверка
        pdf.set_font('DejaVu', 'B', 12)
        pdf.set_text_color(0, 100, 0 if check_passed else 150, 0)
        result = "УДОВЛЕТВОРЕНА ✅" if check_passed else "НЕУДОВЛЕТВОРЕНА ❌"
        pdf.cell(0, 10, f"Проверка: {result}", 0, 1)
        
        if check_passed:
            pdf.set_text_color(0, 100, 0)
            pdf.multi_cell(0, 8, 
                f"Изчисленото напрежение σR = {st.session_state.final_sigma_R:.3f} MPa "
                f"е по-малко или равно на допустимото напрежение {st.session_state.manual_sigma_value:.2f} MPa. "
                "Конструкцията отговаря на изискванията.")
        else:
            pdf.set_text_color(150, 0, 0)
            pdf.multi_cell(0, 8, 
                f"Изчисленото напрежение σR = {st.session_state.final_sigma_R:.3f} MPa "
                f"надвишава допустимото напрежение {st.session_state.manual_sigma_value:.2f} MPa. "
                "Конструкцията НЕ отговаря на изискванията.")
    
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('DejaVu', 'I', 8)
    pdf.cell(0, 10, 'Съставено със система за автоматизирано изчисление на пътни конструкции', 0, 1, 'C')
    
    pdf.cleanup_temp_files()
    return pdf.output(dest='S').encode('latin-1')

# Бутон за генериране на PDF
st.markdown("---")
st.subheader("Генериране на PDF отчет")
if st.button("📄 Генерирай PDF отчет"):
    with st.spinner('Генериране на PDF отчет...'):
        try:
            pdf_bytes = generate_pdf_report()
            if pdf_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(pdf_bytes)
                    tmpfile.flush()
                with open(tmpfile.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    download_link = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="open_v_pokritieto_report.pdf">Свали PDF отчет</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.success("✅ PDF отчетът е успешно генериран!")
            else:
                st.error("Неуспешно генериране на PDF. Моля, проверете грешките по-горе.")
        except Exception as e:
            st.error(f"Грешка при генериране на PDF: {str(e)}")
