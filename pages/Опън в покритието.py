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
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage

# Зареждане на данните
@st.cache_data
def load_data():
    try:
        return pd.read_csv("sigma_data.csv")
    except FileNotFoundError:
        st.error("Файлът sigma_data.csv не е намерен!")
        return None

data = load_data()
if data is None:
    st.stop()

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
img_path = "Допустими опънни напрежения.png"
if os.path.exists(img_path):
    st.image(img_path, caption="Допустими опънни напрежения", width=800)
else:
    st.warning("⚠️ Изображението 'Допустими опънни напрежения.png' не е намерено!")

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

# Коригиран PDF клас и функция за генериране на отчет
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(10, 10, 10)  # Задаване на маргини
        self.temp_font_files = []
        
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, align='C')
        
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

def split_formula(formula_str, max_len=40):  # Намалена максимална дължина
    parts = formula_str.split(" + ")
    lines = []
    current_line = ""
    for part in parts:
        if len(current_line) + len(part) + 3 > max_len:
            lines.append(current_line.rstrip(" + "))
            current_line = ""
        current_line += part + " + "
    if current_line:
        lines.append(current_line.rstrip(" + "))
    return lines

def generate_tension_report():
    # Създаване на буфер за PDF
    buffer = io.BytesIO()
    
    # Инициализация на документа
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    # Стилове
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1))
    styles.add(ParagraphStyle(name='Right', alignment=2))
    
    # Съдържание на документа
    story = []
    
    # Заглавие
    title = Paragraph("Изчисление на опън в покритието", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Дата
    today = datetime.now().strftime("%d.%m.%Y %H:%M")
    date_text = Paragraph(f"Дата: {today}", styles['Normal'])
    story.append(date_text)
    story.append(Spacer(1, 24))
    
    # Параметри на пластовете
    section_title = Paragraph("1. Параметри на пластовете", styles['Heading2'])
    story.append(section_title)
    story.append(Spacer(1, 12))
    
    # Извличане на данни от session state
    D = st.session_state.get("final_D", 34.0)
    Ei_list = st.session_state.get("Ei_list", [1000, 1000])
    hi_list = st.session_state.get("hi_list", [10, 10])
    Ed = st.session_state.get("final_Ed", 100)
    axle_load = st.session_state.get("axle_load", 100)
    
    # Таблица с параметрите
    data = [
        ["Пласт", "Ei (MPa)", "hi (cm)", "Ed (MPa)"],
        *[
            [str(i+1), f"{Ei:.1f}", f"{hi:.1f}", f"{Ed:.1f}" if i == len(Ei_list)-1 else "-"]
            for i, (Ei, hi) in enumerate(zip(Ei_list, hi_list))
        ]
    ]
    
    table = Table(data, colWidths=[1*inch, 1.5*inch, 1*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4B6A88")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 24))
    
    # Изчислени величини
    section_title = Paragraph("2. Изчислени величини", styles['Heading2'])
    story.append(section_title)
    story.append(Spacer(1, 12))
    
    H = sum(hi_list)
    Esr = sum([Ei * hi for Ei, hi in zip(Ei_list, hi_list)]) / H if H != 0 else 0
    hD = H / D if D != 0 else 0
    Esr_Ed = Esr / Ed if Ed != 0 else 0
    
    calculations = [
        f"Диаметър (D): {D:.2f} cm",
        f"Сума на дебелините (H): {H:.2f} cm",
        f"Еквивалентен модул (Esr): {Esr:.2f} MPa",
        f"Модул на основание (Ed): {Ed:.2f} MPa",
        f"H/D: {hD:.4f}",
        f"Esr/Ed: {Esr_Ed:.4f}",
    ]
    
    for calc in calculations:
        story.append(Paragraph(calc, styles['Normal']))
        story.append(Spacer(1, 8))
    
    # Резултати от номограмата
    if "final_sigma" in st.session_state:
        sigma_nomogram = st.session_state["final_sigma"]
        p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
        
        section_title = Paragraph("3. Резултати от номограмата", styles['Heading2'])
        story.append(section_title)
        story.append(Spacer(1, 12))
        
        results = [
            f"σR (номограма): {sigma_nomogram:.4f}",
            f"Осова тежест: {axle_load} kN → p = {p:.3f}",
            f"σR = 1.15 × {p:.3f} × {sigma_nomogram:.4f} = {1.15*p*sigma_nomogram:.4f} MPa"
        ]
        
        for res in results:
            story.append(Paragraph(res, styles['Normal']))
            story.append(Spacer(1, 8))
        
        # Добавяне на графиката от Plotly
        try:
            fig = st.session_state.get("plotly_fig")
            if fig:
                # Запазване на графиката като временен файл
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name, format="png", width=700, height=500)
                    img_path = tmpfile.name
                
                # Добавяне на изображението в PDF
                story.append(Spacer(1, 12))
                story.append(Paragraph("Номограма: σR срещу H/D", styles['Heading3']))
                
                # Ресайз на изображението да се събира в страницата
                pil_img = PILImage.open(img_path)
                img_width, img_height = pil_img.size
                aspect = img_height / float(img_width)
                max_width = 6 * inch  # Максимална ширина 6 инча
                max_height = 8 * inch  # Максимална височина 8 инча
                
                if aspect * max_width > max_height:
                    width = max_height / aspect
                    height = max_height
                else:
                    width = max_width
                    height = aspect * max_width
                
                story.append(Image(img_path, width=width, height=height))
                story.append(Spacer(1, 12))
                
                # Изтриване на временния файл
                os.unlink(img_path)
        except Exception as e:
            st.error(f"Грешка при добавяне на графиката: {str(e)}")
        
        # Проверка спрямо допустимо напрежение
        manual_sigma = st.session_state.get("manual_sigma_value", 1.2)
        sigma_final = 1.15 * p * sigma_nomogram
        check_status = "УДОВЛЕТВОРЯВА" if sigma_final <= manual_sigma else "НЕ УДОВЛЕТВОРЯВА"
        
        section_title = Paragraph("4. Проверка спрямо допустимо напрежение", styles['Heading2'])
        story.append(section_title)
        story.append(Spacer(1, 12))
        
        check_text = [
            f"Ръчно зададено допустимо напрежение: {manual_sigma:.2f} MPa",
            f"Сравнение: {sigma_final:.4f} MPa ≤ {manual_sigma:.2f} MPa → {check_status}"
        ]
        
        for text in check_text:
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 8))
    
    # Добавяне на изображението с допустими напрежения
    img_path = "Допустими опънни напрежения.png"
    if os.path.exists(img_path):
        try:
            story.append(Spacer(1, 24))
            story.append(Paragraph("5. Допустими опънни напрежения", styles['Heading2']))
            
            # Ресайз на изображението
            pil_img = PILImage.open(img_path)
            img_width, img_height = pil_img.size
            aspect = img_height / float(img_width)
            max_width = 6 * inch
            max_height = 8 * inch
            
            if aspect * max_width > max_height:
                width = max_height / aspect
                height = max_height
            else:
                width = max_width
                height = aspect * max_width
            
            story.append(Image(img_path, width=width, height=height))
        except Exception as e:
            st.error(f"Грешка при добавяне на изображението: {str(e)}")
    else:
        st.warning("Изображението 'Допустими опънни напрежения.png' не е намерено!")
    
    # Генериране на PDF
    doc.build(story)
    
    # Връщане на PDF като bytes
    buffer.seek(0)
    return buffer.read()

# В Streamlit частта, преди да генерирате PDF, запазете фигурата в session_state:
if 'plotly_fig' not in st.session_state and 'final_sigma' in st.session_state:
    # Създаване на фигурата (както е в оригиналния ви код)
    fig = go.Figure()
    for val, group in data.groupby("Esr_over_Ed"):
        fig.add_trace(go.Scatter(
            x=group["H_over_D"],
            y=group["sigma_R"],
            mode='lines',
            name=f"Esr/Ed = {val:.1f}"
        ))
    fig.add_trace(go.Scatter(
        x=[H / D], y=[sigma_nomogram],
        mode='markers',
        marker=dict(size=8, color='red'),
        name="Твоята точка"
    ))
    fig.update_layout(
        title="Номограма: σR срещу H/D",
        xaxis_title="H / D",
        yaxis_title="σR",
        height=500
    )
    st.session_state['plotly_fig'] = fig

# Бутон за генериране на PDF
if st.button("📊 Генерирай PDF отчет", key="generate_pdf_button"):
    with st.spinner('Генерира се PDF отчет...'):
        try:
            pdf_bytes = generate_tension_report()
            
            if pdf_bytes:
                # Създаване на линк за сваляне
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="opyn_v_pokritieto_report.pdf">⬇️ Свали PDF отчета</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("PDF отчетът е генериран успешно!")
            else:
                st.error("Грешка при генериране на PDF отчета")
        except Exception as e:
            st.error(f"Възникна грешка: {str(e)}")
