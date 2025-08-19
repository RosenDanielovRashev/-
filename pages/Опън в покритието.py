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

# Опит за импорт на cairosvg (за векторни формули)
try:
    import cairosvg  # pip install cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

st.title("Опън в покритието")

# -----------------------------
# Зареждане на данните
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("sigma_data.csv")

data = load_data()

# -----------------------------
# Функция за изчисляване на σR
# -----------------------------
def compute_sigma_R(H, D, Esr, Ed):
    hD = H / D if D else 0
    Esr_Ed = Esr / Ed if Ed else 0
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

# -----------------------------
# UI: Заглавие и входове
# -----------------------------
st.title("Определяне опънното напрежение в долния пласт на покритието фиг.9.2")
st.markdown("### Въвеждане на параметри на пластове")

# Вземаме дефолтни стойности от session_state (ако има)
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

# Фиксиран брой пластове (2)
st.markdown(f"**Брой пластове:** 2 (фиксиран за това изчисление)")
n = 2

# Въвеждане на Ei и hi
Ei_list, hi_list = [], []
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

# Запазване в session_state
st.session_state["final_D"] = D
st.session_state["Ei_list"] = Ei_list
st.session_state["hi_list"] = hi_list

# Проверка за Ed
st.markdown("---")
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

# Esr и H
numerator = sum(Ei * hi for Ei, hi in zip(Ei_list, hi_list))
denominator = sum(hi_list)
Esr = numerator / denominator if denominator != 0 else 0
H = denominator

# Показване на формули (Streamlit)
st.markdown("### ℹ️ Формули за изчисление")
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

numerator_str = " + ".join([f"{Ei:.2f}×{hi:.2f}" for Ei, hi in zip(Ei_list, hi_list)])
denominator_str = " + ".join([f"{hi:.2f}" for hi in hi_list])
st.latex(fr"Esr = \frac{{{numerator_str}}}{{{denominator_str}}} = {Esr:.2f} \text{{ MPa}}")
st.latex(fr"H = {denominator_str} = {H:.2f} \text{{ см}}")

# Автоматично изчисляване на σR
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

# Крайно σR (спрямо осов товар)
axle_load = st.session_state.get("axle_load", 100)
p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else None

if p is not None:
    st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
    st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")

    sigma_nom = st.session_state.get("final_sigma", None)
    if sigma_nom is not None:
        sigma_final = 1.15 * p * sigma_nom
        st.markdown("### Формула за изчисление на крайното напрежение σR:")
        st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
        st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma_nom:.3f} = {sigma_final:.3f} \text{{ MPa}}")
        st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
        st.session_state["final_sigma_R"] = sigma_final
    else:
        st.warning("❗ Липсва σR от номограмата за изчисление.")
else:
    st.warning("❗ Не е зададен валиден осов товар. Не може да се изчисли p.")

# Ръчно въвеждане
st.markdown(
    """
    <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
        <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
    </div>
    """,
    unsafe_allow_html=True
)
if 'manual_sigma_value' not in st.session_state:
    st.session_state.manual_sigma_value = 1.20

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
st.session_state.manual_sigma_value = manual_value

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

# -------------------------------------------------
# Векторен рендер на формули: SVG -> PNG (или fallback)
# -------------------------------------------------
def render_formula_to_svg(formula, output_path):
    """
    Рендва формула като SVG чрез matplotlib.mathtext.
    """
    parser = mathtext.MathTextParser("path")
    parser.to_svg(f"${formula}$", output_path)
    return output_path

def svg_to_png(svg_path, png_path=None, dpi=300):
    """
    Конвертира SVG към PNG с висока резолюция. Изисква cairosvg.
    """
    if not _HAS_CAIROSVG:
        raise RuntimeError("cairosvg не е наличен")
    if png_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
            png_path = tmp_png.name
    cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)
    return png_path

def render_formula_to_image_fallback(formula, fontsize=22, dpi=450):
    """
    Fallback: рендва формула директно в PNG чрез matplotlib (растерно, но висок DPI).
    """
    fig = plt.figure(figsize=(8, 2.5))
    fig.text(0.05, 0.5, f'${formula}$', fontsize=fontsize, ha='left', va='center', usetex=False)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    buf.seek(0)
    return buf

# -------------------------------------------------
# PDF клас с подобрено управление на формули (без сив фон)
# -------------------------------------------------
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

    def _formula_png_from_svg_or_fallback(self, formula_text, dpi=300):
        """
        Прави PNG път от формула чрез SVG→PNG, а ако няма cairosvg → fallback PNG буфер.
        Връща път към PNG файл, добавен към temp списъка.
        """
        try:
            # SVG временен файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
                render_formula_to_svg(formula_text, tmp_svg.name)
                # PNG от SVG
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
                    svg_to_png(tmp_svg.name, tmp_png.name, dpi=dpi)
                    png_path = tmp_png.name
            self.temp_image_files.append(png_path)
            return png_path
        except Exception:
            # Fallback: директно PNG от matplotlib
            buf = render_formula_to_image_fallback(formula_text, fontsize=22, dpi=450)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(buf.read())
                png_path = tmp_file.name
            self.temp_image_files.append(png_path)
            return png_path

    def add_latex_formula(self, formula_text, width=100, line_gap=12):
        """
        Добавя ЕДНА формула като изображение (векторен рендер до PNG), без фонови плочи.
        """
        try:
            png_path = self._formula_png_from_svg_or_fallback(formula_text)
            # Вмъкване с фиксирана ширина → еднакъв визуален размер
            self.image(png_path, x=self.get_x(), y=self.get_y(), w=width)
            # Приблизителен вертикален интервал
            self.ln(line_gap + width * 0.22)
        except Exception:
            self.set_font('DejaVu', 'I', 12)
            self.multi_cell(0, 8, formula_text)
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

    def add_formula_section(self, title, formulas, columns=2, col_width=95, img_width=85, row_gap=8):
        """
        Секция с формули, подредени по колони, без фон и с еднакво мащабиране.
        - img_width контролира реалната ширина на всяка формула.
        """
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

        # Групираме по броя колони
        rows = [formulas[i:i+columns] for i in range(0, len(formulas), columns)]

        for row in rows:
            # Начална X позиция
            start_x = 10
            self.set_x(start_x)
            max_row_height = 0  # при нужда може да се развие за още по-точен вертикален интервал

            for idx, formula in enumerate(row):
                try:
                    png_path = self._formula_png_from_svg_or_fallback(formula)
                    # Картинка с фиксиран img_width за еднакъв размер
                    self.image(png_path, x=self.get_x(), y=self.get_y(), w=img_width)
                except Exception:
                    # Текстов fallback
                    self.set_font('DejaVu', '', 11)
                    self.multi_cell(col_width, 6, formula)
                # Преместваме в следващата колона
                self.set_x(start_x + col_width * (idx + 1))
                max_row_height = max(max_row_height, img_width * 0.28)

            # Нов ред с малък промеждутък
            self.ln(max(18, int(max_row_height)) + row_gap)

        self.ln(4)

# -------------------------------------------------
# Генерация на PDF
# -------------------------------------------------
def generate_pdf_report():
    pdf = EnhancedPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Шрифтове
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    font_dir = os.path.join(base_dir, "fonts")
    os.makedirs(font_dir, exist_ok=True)

    sans_path = os.path.join(font_dir, "DejaVuSans.ttf")
    bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
    italic_path = os.path.join(font_dir, "DejaVuSans-Oblique.ttf")

    try:
        if all(os.path.exists(p) for p in [sans_path, bold_path, italic_path]):
            with open(sans_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', '', f.read())
            with open(bold_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', 'B', f.read())
            with open(italic_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', 'I', f.read())
        else:
            from fpdf.fonts import FontsByFPDF
            fonts = FontsByFPDF()
            for style, data in [('', fonts.helvetica),
                                ('B', fonts.helvetica_bold),
                                ('I', fonts.helvetica_oblique)]:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                    tmp_file.write(data)
                    pdf.add_font('DejaVu', style, tmp_file.name)
    except Exception as e:
        st.error(f"Грешка при зареждане на шрифтове: {e}")
        return b""

    # Заглавна страница
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 18)
    pdf.cell(0, 15, 'ОПЪН В ПОКРИТИЕТО', ln=True, align='C')
    pdf.set_font('DejaVu', 'I', 12)
    pdf.ln(6)

    # 1. Входни параметри
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '1. Входни параметри', ln=True)

    col_width = 60
    row_height = 8

    pdf.set_font('DejaVu', 'B', 11)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(col_width, row_height, 'Параметър', border=1, align='C', fill=True)
    pdf.cell(col_width, row_height, 'Стойност', border=1, align='C', fill=True)
    pdf.cell(col_width, row_height, 'Мерна единица', border=1, align='C', fill=True)
    pdf.ln(row_height)

    pdf.set_font('DejaVu', '', 10)
    params = [
        ("Диаметър D", f"{st.session_state.final_D:.2f}", "cm"),
        ("Брой пластове", "2", ""),
    ]
    for i in range(2):
        params.append((f"Пласт {i+1} - Ei", f"{st.session_state.Ei_list[i]:.2f}", "MPa"))
        params.append((f"Пласт {i+1} - hi", f"{st.session_state.hi_list[i]:.2f}", "cm"))
    params.extend([
        ("Ed", f"{st.session_state.final_Ed:.2f}", "MPa"),
        ("Осова тежест", f"{st.session_state.get('axle_load', 100)}", "kN")
    ])

    fill = False
    for p_name, p_val, p_unit in params:
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_width, row_height, p_name, border=1, fill=True)
        pdf.cell(col_width, row_height, p_val, border=1, align='C', fill=True)
        pdf.cell(col_width, row_height, p_unit, border=1, align='C', fill=True)
        pdf.ln(row_height)
        fill = not fill

    pdf.ln(5)

    # 2. Формули за изчисление (векторен рендер)
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '2. Формули за изчисление', ln=True)

    formulas_section2 = [
        r"E_{sr} = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}",
        r"H = \sum_{i=1}^{n} h_i",
        r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}",
    ]
    pdf.add_formula_section("Основни формули за изчисление:", formulas_section2, columns=2, col_width=95, img_width=85, row_gap=-3)

    # 3. Изчисления (с числени замествания)
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '3. Изчисления', ln=True)

    num = sum(Ei * hi for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list))
    den = sum(st.session_state.hi_list)
    Esr_val = num / den if den else 0
    H_val = den
    num_str = " + ".join([f"{Ei:.2f} \\times {hi:.2f}" for Ei, hi in zip(st.session_state.Ei_list, st.session_state.hi_list)])
    den_str = " + ".join([f"{hi:.2f}" for hi in st.session_state.hi_list])

    formulas_section3 = [
        fr"E_{{sr}} = \frac{{{num_str}}}{{{den_str}}} = {Esr_val:.2f} \, \text{{MPa}}",
        fr"H = {den_str} = {H_val:.2f} \, \text{{cm}}"
    ]
    if 'final_sigma' in st.session_state:
        formulas_section3.append(fr"\frac{{E_{{sr}}}}{{E_d}} = \frac{{{Esr_val:.2f}}}{{{st.session_state.final_Ed:.2f}}} = {Esr_val/st.session_state.final_Ed:.3f}")
        formulas_section3.append(fr"\frac{{H}}{{D}} = \frac{{{H_val:.2f}}}{{{st.session_state.final_D:.2f}}} = {H_val/st.session_state.final_D:.3f}")
        formulas_section3.append(fr"\sigma_R^{{nom}} = {st.session_state.final_sigma:.3f} \, \text{{MPa}}")

    axle_load = st.session_state.get("axle_load", 100)
    p_loc = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
    if p_loc and 'final_sigma' in st.session_state:
        sigma_final_loc = 1.15 * p_loc * st.session_state.final_sigma
        formulas_section3.append(fr"p = {p_loc:.3f} \, \text{{ (за осов товар {axle_load} kN)}}")
        formulas_section3.append(fr"\sigma_R = 1.15 \times {p_loc:.3f} \times {st.session_state.final_sigma:.3f} = {sigma_final_loc:.3f} \, \text{{MPa}}")

    pdf.add_formula_section("Изчислителни формули:", formulas_section3, columns=2, col_width=95, img_width=85, row_gap=-3)

    pdf.ln(5)

    # 4. Графика
    if "fig" in st.session_state:
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '4. Графика на номограмата', ln=True)
        pdf.add_plotly_figure(st.session_state["fig"], width=160)

    # 5. Допустими напрежения
    img_path = "Допустими опънни напрежения.png"
    if os.path.exists(img_path):
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '5. Допустими опънни напрежения', ln=True)
        pdf.add_external_image(img_path, width=160)

    # 6. Резултати и проверка
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '6. Резултати и проверка', ln=True)

    if 'final_sigma_R' in st.session_state and 'manual_sigma_value' in st.session_state:
        check_passed = st.session_state.final_sigma_R <= st.session_state.manual_sigma_value

        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(90, 8, 'Параметър', border=1, align='C', fill=True)
        pdf.cell(90, 8, 'Стойност', border=1, align='C', fill=True)
        pdf.ln(8)

        pdf.set_font('DejaVu', '', 10)
        for label, val in [
            ('Изчислено σR', f"{st.session_state.final_sigma_R:.3f} MPa"),
            ('Допустимо σR (ръчно)', f"{st.session_state.manual_sigma_value:.2f} MPa")
        ]:
            pdf.set_fill_color(245, 245, 245) if label.startswith('Изчислено') else pdf.set_fill_color(255, 255, 255)
            pdf.cell(90, 8, label, border=1, fill=True)
            pdf.cell(90, 8, val, border=1, align='C', fill=True)
            pdf.ln(8)

        pdf.ln(5)
        if check_passed:
            pdf.set_text_color(0, 100, 0)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, "✅ Проверка: УДОВЛЕТВОРЕНА", ln=True)
        else:
            pdf.set_text_color(150, 0, 0)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, "❌ Проверка: НЕУДОВЛЕТВОРЕНА", ln=True)

        pdf.set_text_color(0, 0, 0)

    # Footer
    pdf.ln(10)
    pdf.set_font('DejaVu', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Съставено със система за автоматизирано изчисление на пътни конструкции', align='C')

    pdf.cleanup_temp_files()
    return pdf.output(dest='S')

# -----------------------------
# Бутон за генериране на PDF
# -----------------------------
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
                st.error("Неуспешно генериране на PDF. Моля, проверете грешките по-горе.")
        except Exception as e:
            st.error(f"Грешка при генериране на PDF: {str(e)}")
