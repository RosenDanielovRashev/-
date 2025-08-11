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
import matplotlib as mpl

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

# Изчисления
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

st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=800)

axle_load = st.session_state.get("axle_load", 100)
p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else None

if p is not None:
    st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
    st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")
    
    sigma = st.session_state.get("final_sigma", None)
    
    if sigma is not None:
        sigma_final = 1.15 * p * sigma
        st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
        st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
        st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
        st.session_state["final_sigma_R"] = sigma_final

# Ръчно въвеждане
if 'manual_sigma_value' not in st.session_state:
    st.session_state.manual_sigma_value = 1.20

manual_value = st.number_input(
    label="Въведете допустимо опънно напрежение σR [MPa] (от таблица 9.7)",
    min_value=0.0,
    max_value=20.0,
    value=st.session_state.manual_sigma_value,
    step=0.01,
    key="manual_sigma_input",
    format="%.2f"
)

st.session_state.manual_sigma_value = manual_value

# PDF генериране
def render_formula_to_image(formula, fontsize=14, dpi=300):
    """Render LaTeX formula to high-quality image"""
    fig = plt.figure(figsize=(8, 0.5))
    fig.text(0.02, 0.5, f'${formula}$', 
             fontsize=fontsize, 
             ha='left', 
             va='center',
             color='black')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return buf

class EnhancedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_files = []
        
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
        
    def add_font_from_bytes(self, family, style, font_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
            tmp_file.write(font_bytes)
            self.temp_files.append(tmp_file.name)
            self.add_font(family, style, tmp_file.name)
    
    def add_highres_image(self, image_data, width=180):
        """Add high resolution image"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_data)
            self.temp_files.append(tmp_file.name)
        self.image(tmp_file.name, x=10, w=width)
        self.ln(10)
        
    def add_latex_formula(self, formula_text):
        """Add LaTeX formula as image"""
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
            except:
                pass

def generate_pdf_report():
    pdf = EnhancedPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Зареждане на шрифтове
    try:
        # Добавяне на шрифтове
        pdf.add_page()
        pdf.set_font('DejaVu', 'B', 18)
        pdf.cell(0, 15, 'ОПЪН В ПОКРИТИЕТО - ОТЧЕТ', ln=True, align='C')
        pdf.set_font('DejaVu', 'I', 12)
        pdf.cell(0, 10, f"Дата на генериране: {datetime.today().strftime('%d.%m.%Y')}", ln=True, align='C')
        pdf.ln(15)

        # 1. Входни параметри
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '1. Входни параметри', ln=True)
        
        # Таблица с параметри
        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(60, 8, 'Параметър', 1, 0, 'C', True)
        pdf.cell(60, 8, 'Стойност', 1, 0, 'C', True)
        pdf.cell(60, 8, 'Мерна единица', 1, 1, 'C', True)
        
        pdf.set_font('DejaVu', '', 10)
        params = [
            ("Диаметър D", f"{st.session_state.final_D:.2f}", "cm"),
            ("Брой пластове", "2", ""),
            ("Пласт 1 - Ei", f"{st.session_state.Ei_list[0]:.2f}", "MPa"),
            ("Пласт 1 - hi", f"{st.session_state.hi_list[0]:.2f}", "cm"),
            ("Пласт 2 - Ei", f"{st.session_state.Ei_list[1]:.2f}", "MPa"),
            ("Пласт 2 - hi", f"{st.session_state.hi_list[1]:.2f}", "cm"),
            ("Ed", f"{st.session_state.final_Ed:.2f}", "MPa"),
            ("Осова тежест", f"{st.session_state.get('axle_load', 100)}", "kN")
        ]
        
        fill = False
        for p_name, p_val, p_unit in params:
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(60, 8, p_name, 1, 0, 'L', fill)
            pdf.cell(60, 8, p_val, 1, 0, 'C', fill)
            pdf.cell(60, 8, p_unit, 1, 1, 'C', fill)
            fill = not fill
        
        pdf.ln(10)

        # 2. Формули
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '2. Формули за изчисление', ln=True)
        
        formulas = [
            r"E_{sr} = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}",
            r"H = \sum_{i=1}^{n} h_i",
            r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}"
        ]
        
        for formula in formulas:
            pdf.add_latex_formula(formula)
        
        pdf.ln(5)

        # 3. Изчисления
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '3. Изчисления', ln=True)
        
        # Изчислителни формули
        numerator = sum(Ei * hi for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list))
        denominator = sum(st.session_state.hi_list)
        Esr = numerator / denominator if denominator else 0
        H = denominator
        
        num_str = " + ".join([f"{Ei:.2f} \\times {hi:.2f}" for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list)])
        den_str = " + ".join([f"{hi:.2f}" for hi in st.session_state.hi_list])
        
        pdf.add_latex_formula(fr"E_{{sr}} = \frac{{{num_str}}}{{{den_str}}} = {Esr:.2f} \, \text{{MPa}}")
        pdf.add_latex_formula(fr"H = {den_str} = {H:.2f} \, \text{{cm}}")
        
        if 'final_sigma' in st.session_state:
            pdf.add_latex_formula(fr"\frac{{E_{{sr}}}}{{E_d}} = \frac{{{Esr:.2f}}}{{{st.session_state.final_Ed:.2f}}} = {Esr/st.session_state.final_Ed:.3f}")
            pdf.add_latex_formula(fr"\frac{{H}}{{D}} = \frac{{{H:.2f}}}{{{st.session_state.final_D:.2f}}} = {H/st.session_state.final_D:.3f}")
            pdf.add_latex_formula(fr"\sigma_R^{{nom}} = {st.session_state.final_sigma:.3f} \, \text{{MPa}}")
            
            axle_load = st.session_state.get("axle_load", 100)
            p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
            if p:
                sigma_final = 1.15 * p * st.session_state.final_sigma
                pdf.add_latex_formula(fr"p = {p:.3f} \, \text{{ (за осов товар {axle_load} kN)}}")
                pdf.add_latex_formula(fr"\sigma_R = 1.15 \times {p:.3f} \times {st.session_state.final_sigma:.3f} = {sigma_final:.3f} \, \text{{MPa}}")
        
        pdf.ln(10)

        # 4. Графика
        if "fig" in st.session_state:
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, '4. Графика на номограмата', ln=True)
            
            img_bytes = pio.to_image(
                st.session_state["fig"], 
                format="png", 
                width=1600,
                height=1200,
                scale=3,
                engine="kaleido"
            )
            pdf.add_highres_image(img_bytes, width=160)

        # 5. Допустими напрежения
        img_path = "Допустими опънни напрежения.png"
        if os.path.exists(img_path):
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, '5. Допустими опънни напрежения', ln=True)
            
            with open(img_path, "rb") as f:
                pdf.add_highres_image(f.read(), width=160)

        # 6. Резултати
        if 'final_sigma_R' in st.session_state and 'manual_sigma_value' in st.session_state:
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, '6. Резултати и проверка', ln=True)
            
            check_passed = st.session_state.final_sigma_R <= st.session_state.manual_sigma_value
            
            pdf.set_font('DejaVu', 'B', 10)
            pdf.set_fill_color(200, 220, 255)
            pdf.cell(90, 8, 'Параметър', 1, 0, 'C', True)
            pdf.cell(90, 8, 'Стойност', 1, 1, 'C', True)
            
            pdf.set_font('DejaVu', '', 10)
            pdf.cell(90, 8, 'Изчислено σR', 1, 0, 'L')
            pdf.cell(90, 8, f"{st.session_state.final_sigma_R:.3f} MPa", 1, 1, 'C')
            
            pdf.cell(90, 8, 'Допустимо σR', 1, 0, 'L')
            pdf.cell(90, 8, f"{st.session_state.manual_sigma_value:.2f} MPa", 1, 1, 'C')
            
            pdf.ln(5)
            pdf.set_font('DejaVu', 'B', 12)
            if check_passed:
                pdf.set_text_color(0, 100, 0)
                pdf.cell(0, 10, "✅ Проверка: УДОВЛЕТВОРЕНА", ln=True)
            else:
                pdf.set_text_color(150, 0, 0)
                pdf.cell(0, 10, "❌ Проверка: НЕУДОВЛЕТВОРЕНА", ln=True)
            
            pdf.set_text_color(0, 0, 0)

        pdf.ln(10)
        pdf.set_font('DejaVu', 'I', 8)
        pdf.cell(0, 10, 'Съставено със система за автоматизирано изчисление', align='C')

        pdf.cleanup_temp_files()
        return pdf.output(dest='S').encode('latin-1')
    
    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {str(e)}")
        return b""

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
                
                with open(tmpfile.name, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    download_link = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="open_v_pokritieto_report.pdf">Свали PDF отчет</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.success("✅ PDF отчетът е успешно генериран!")
            else:
                st.error("Неуспешно генериране на PDF.")
        except Exception as e:
            st.error(f"Грешка: {str(e)}")
