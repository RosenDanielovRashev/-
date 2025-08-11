import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from fpdf import FPDF
from PIL import Image
import plotly.io as pio
import os
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# Настройки за висококачествени формули
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.formatter.use_mathtext'] = True

st.title("Опън в покритието")

@st.cache_data
def load_data():
    return pd.read_csv("sigma_data.csv")

data = load_data()

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

# UI част
st.title("Определяне опънното напрежение в долния пласт на покритието фиг.9.2")

# Въвеждане на параметри
D_default = st.session_state.get("final_D", 34.0)
Ei_list_full = st.session_state.get("Ei_list", [])
hi_list_full = st.session_state.get("hi_list", [])
Ei_list_default = Ei_list_full[:2] if len(Ei_list_full) >= 2 else [1000.0, 1000.0]
hi_list_default = hi_list_full[:2] if len(hi_list_full) >= 2 else [10.0, 10.0]

if len(Ei_list_full) > 2:
    st.info("ℹ️ Използват се само първите два пласта от въведените на предишната страница.")

D = st.selectbox(
    "Диаметър на отпечатъка на колело D (см)",
    options=[34.0, 32.04, 33.0],
    index=[34.0, 32.04, 33.0].index(D_default) if D_default in [34.0, 32.04, 33.0] else 0
)

st.markdown(f"**Брой пластове:** 2 (фиксиран за това изчисление)")
n = 2

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

st.session_state["final_D"] = D
st.session_state["Ei_list"] = Ei_list
st.session_state["hi_list"] = hi_list

# Проверка на final_Ed_list
if "final_Ed_list" not in st.session_state:
    st.error("⚠️ Липсва final_Ed_list в session_state!")
    st.info("Моля, върнете се на първата страница и изчислете всички пластове")
    st.stop()

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

numerator = sum(Ei * hi for Ei, hi in zip(Ei_list, hi_list))
denominator = sum(hi_list)
Esr = numerator / denominator if denominator != 0 else 0
H = denominator

# Показване на формулите
st.markdown("### ℹ️ Формули за изчисление")
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

numerator_str = " + ".join([f"{Ei:.2f}×{hi:.2f}" for Ei, hi in zip(Ei_list, hi_list)])
denominator_str = " + ".join([f"{hi:.2f}" for hi in hi_list])
st.latex(fr"Esr = \frac{{{numerator_str}}}{{{denominator_str}}} = {Esr:.2f} \text{{ MPa}}")
st.latex(fr"H = {denominator_str} = {H:.2f} \text{{ см}}")

# Изчисления и визуализация
if denominator != 0:
    sigma, hD, y_low, y_high, low, high = compute_sigma_R(H, D, Esr, Ed)
    
    if sigma is not None:
        st.session_state["final_sigma"] = sigma
        st.session_state["final_hD"] = hD
        st.session_state["final_y_low"] = y_low
        st.session_state["final_y_high"] = y_high
        st.session_state["final_low"] = low
        st.session_state["final_high"] = high
        
        st.markdown("## 📋 Резултати от изчисленията")
        st.markdown(f"""
        **Изчислено:**
        - $Esr / Ed = {Esr:.2f} / {Ed:.2f} = {Esr / Ed:.3f}$
        - $H / D = {H:.2f} / {D:.2f} = {H / D:.3f}$
        """)
        st.success(f"✅ σR = {sigma:.3f}")
        st.info(f"Интерполация между изолинии: Esr/Ed = {low:.2f} и {high:.2f}")

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
        st.session_state["fig"] = fig
        
    else:
        st.warning("❗ Точката е извън диапазона на наличните данни.")
else:
    st.error("Сумата на hᵢ не може да бъде 0.")

st.image("Допустими опънни напрежения.png", caption="Таблица 9.2", use_column_width=True)

# ----- PDF част с поправени шрифтове -----

# Трябва да имате локални файлове с DejaVu шрифтове или изтеглете:
DEJAVU_REGULAR = "DejaVuSans.ttf"
DEJAVU_BOLD = "DejaVuSans-Bold.ttf"
DEJAVU_ITALIC = "DejaVuSans-Oblique.ttf"

class EnhancedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_files = []
        # Добавяне на шрифтовете
        self.add_font('DejaVu', '', DEJAVU_REGULAR, uni=True)
        self.add_font('DejaVu', 'B', DEJAVU_BOLD, uni=True)
        self.add_font('DejaVu', 'I', DEJAVU_ITALIC, uni=True)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

    def add_highres_image(self, image_data, width=180):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_data)
            self.temp_files.append(tmp_file.name)
            self.image(tmp_file.name, x=10, w=width)
            self.ln(10)

    def add_latex_formula(self, formula_text):
        try:
            img_buf = render_formula_to_image(formula_text)
            self.add_highres_image(img_buf.getvalue(), width=160)
        except Exception as e:
            self.set_font('DejaVu', '', 12)
            self.cell(0, 10, f'Формула: {formula_text}', 0, 1)
            self.ln(5)

    def cleanup_temp_files(self):
        for file_path in self.temp_files:
            try:
                os.unlink(file_path)
            except Exception:
                pass

def render_formula_to_image(formula, fontsize=14, dpi=300):
    fig = plt.figure(figsize=(8, 0.5))
    fig.text(0.02, 0.5, f'${formula}$', fontsize=fontsize, ha='left', va='center', color='black')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_pdf_report():
    pdf = EnhancedPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    try:
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 18)
        pdf.cell(0, 15, 'ОПЪН В ПОКРИТИЕТО - ОТЧЕТ', ln=True, align='C')
        pdf.set_font('DejaVu', 'I', 12)
        pdf.cell(0, 10, f"Дата на генериране: {datetime.today().strftime('%d.%m.%Y')}", ln=True, align='C')
        pdf.ln(15)

        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Входни данни:', ln=True)
        pdf.set_font('DejaVu', '', 12)
        pdf.cell(0, 10, f'Диаметър на отпечатъка D: {st.session_state["final_D"]:.2f} см', ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, 'Модул на деформация и дебелина на пластовете:', ln=True)
        for i, (Ei, hi) in enumerate(zip(st.session_state["Ei_list"], st.session_state["hi_list"]), start=1):
            pdf.cell(0, 10, f'Пласт {i}: E{i} = {Ei:.2f} MPa, h{i} = {hi:.2f} см', ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, f'Ed (модул на деформация на основата): {st.session_state["final_Ed"]:.2f} MPa', ln=True)
        pdf.ln(10)

        pdf.cell(0, 10, 'Изчисления:', ln=True)
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Esr:', ln=True)
        pdf.set_font('DejaVu', '', 12)

        numerator_str = " + ".join([f"{Ei:.2f}×{hi:.2f}" for Ei, hi in zip(st.session_state["Ei_list"], st.session_state["hi_list"])])
        denominator_str = " + ".join([f"{hi:.2f}" for hi in st.session_state["hi_list"]])
        formula1 = fr"Esr = \frac{{{numerator_str}}}{{{denominator_str}}} = {numerator:.2f} / {denominator:.2f} = {numerator/denominator:.2f} \text{{ MPa}}"
        pdf.add_latex_formula(formula1)
        pdf.ln(5)
        pdf.cell(0, 10, f'H = {denominator_str} = {denominator:.2f} см', ln=True)
        pdf.ln(5)

        pdf.cell(0, 10, 'Резултат:', ln=True)
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(0, 15, f'σR = {st.session_state["final_sigma"]:.3f}', ln=True)
        pdf.ln(10)

        # Вмъкваме графиката
        if "fig" in st.session_state:
            img_bytes = pio.to_image(st.session_state["fig"], format='png', width=600, height=400, scale=2)
            pdf.add_highres_image(img_bytes, width=180)

        pdf.cleanup_temp_files()
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {str(e)}")
        return b""

if st.button("Генерирай PDF отчет"):
    pdf_data = generate_pdf_report()
    if pdf_data:
        st.download_button(
            label="Свали PDF",
            data=pdf_data,
            file_name="Otchet_Opan_v_Pokritieto.pdf",
            mime="application/pdf"
        )
