import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from fpdf import FPDF
import base64
from io import BytesIO
import tempfile
import os
from PIL import Image
import plotly.io as pio
from matplotlib import mathtext
import matplotlib.pyplot as plt

# Опит за импорт на cairosvg (за векторни формули)
try:
    import cairosvg  # pip install cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 18px !important;
        }
        .block-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .css-1lcbmi9 {
            max-width: 800px !important;
            margin: 0 auto !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Initialize session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}
if 'manual_sigma_values' not in st.session_state:
    st.session_state.manual_sigma_values = {}
if 'check_results' not in st.session_state:
    st.session_state.check_results = {}
if 'pdf_exported' not in st.session_state:
    st.session_state.pdf_exported = False
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'x_intercept' not in st.session_state:
    st.session_state.x_intercept = None
if 'y_at_ratio' not in st.session_state:
    st.session_state.y_at_ratio = None
if 'has_intercept' not in st.session_state:
    st.session_state.has_intercept = False
if 'sigma_r' not in st.session_state:
    st.session_state.sigma_r = None

# Проверка за данни от główny файл
use_auto_data = False
if 'layers_data_all' in st.session_state and 'final_D_all' in st.session_state:
    layers_data = st.session_state.layers_data_all
    D_auto = st.session_state.final_D_all
    n_auto = len(layers_data)
    h_values_auto = [layer.get('h', 4.0) for layer in layers_data]
    E_values_auto = [layer.get('Ei', 400.0) for layer in layers_data]
    Ed_values_auto = [layer.get('Ed', 30.0) for layer in layers_data]
    use_auto_data = True

# Input parameters
if use_auto_data:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=n_auto)
    D = st.selectbox("Избери D", options=[32.04, 34.0], index=0 if D_auto == 32.04 else 1)
else:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=4)
    D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Input data for all layers
st.markdown("### Въведи стойности за всички пластове")
h_values = []
E_values = []
Ed_values = []

cols = st.columns(3)
for i in range(n):
    # Автоматично попълване ако има данни
    h_default = h_values_auto[i] if use_auto_data and i < len(h_values_auto) else 4.0
    E_default = E_values_auto[i] if use_auto_data and i < len(E_values_auto) else [1200.0, 1000.0, 800.0, 400.0][i] if i < 4 else 400.0
    Ed_default = Ed_values_auto[i] if use_auto_data and i < len(Ed_values_auto) else 30.0
    
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=h_default, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E{to_subscript(i+1)}", value=E_default, step=0.1, key=f"E_{i}")
        E_values.append(E)
    with cols[2]:
        Ed = st.number_input(f"Ed{to_subscript(i+1)}", value=round(Ed_default), step=1, key=f"Ed_{i}")
        Ed_values.append(Ed)

# Layer selection
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(2, n)], index=n-3 if n > 2 else 0)
layer_idx = int(selected_layer.split()[-1]) - 1

# Calculation function
def calculate_layer(layer_index):
    h_array = np.array(h_values[:layer_index+1])
    E_array = np.array(E_values[:layer_index+1])
    current_Ed = Ed_values[layer_index]
    
    sum_h_n_1 = h_array[:-1].sum() if layer_index > 0 else 0
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1]) if layer_index > 0 else 0
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0
    
    H_n = h_array.sum()
    H_n_1 = sum_h_n_1
    
    results = {
        'H_n_1_r': round(H_n_1, 3),
        'H_n_r': round(H_n, 3),
        'Esr_r': round(Esr, 3),
        'ratio_r': round(H_n / D, 3) if D != 0 else 0,
        'En_r': round(E_values[layer_index], 3),
        'Ed_r': round(current_Ed, 3),
        'Esr_over_En_r': round(Esr / E_values[layer_index], 3) if E_values[layer_index] != 0 else 0,
        'En_over_Ed_r': round(E_values[layer_index] / current_Ed, 3) if current_Ed != 0 else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

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

# Коригиран код за функцията create_nomogram_plot
def create_nomogram_plot(results, layer_idx):
    try:
        df_original = pd.read_csv("danni_1.csv")
        df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D_1.csv")
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

        fig = go.Figure()

        # Add isolines from original df
        if 'Ei/Ed' in df_original.columns:
            for level in sorted(df_original['Ei/Ed'].unique()):
                df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Ei/Ed = {round(level,3)}',
                    line=dict(width=2)
                ))

        # Add isolines from new df
        if 'sr_Ei' in df_new.columns:
            for sr_Ei in sorted(df_new['sr_Ei'].unique()):
                df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Esr/Ei = {round(sr_Ei,3)}',
                    line=dict(width=2)
                ))

        # Interpolation and marking points
        if layer_idx > 0:
            sr_Ei_values = sorted(df_new['sr_Ei'].unique())
            target_sr_Ei = results['Esr_over_En_r']
            target_Hn_D = results['ratio_r']

            y_at_ratio = None
            if min(sr_Ei_values) <= target_sr_Ei <= max(sr_Ei_values):
                if target_sr_Ei in sr_Ei_values:
                    df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
                    y_at_ratio = np.interp(target_Hn_D, df_target['H/D'], df_target['y'])
                else:
                    for i in range(len(sr_Ei_values)-1):
                        if sr_Ei_values[i] < target_sr_Ei < sr_Ei_values[i+1]:
                            df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                            df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[i+1]].sort_values(by='H/D')
                            
                            y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
                            y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])
                            
                            y_at_ratio = y_lower + (y_upper - y_lower) * (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i+1] - sr_Ei_values[i])
                            break

            if y_at_ratio is not None:
                # Вертикална линия (синя)
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                    mode='lines', line=dict(color='blue', dash='dash'),
                    name='Вертикална линия'
                ))

                # Червена точка
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D], y=[y_at_ratio],
                    mode='markers', marker=dict(color='red', size=10),
                    name='Точка на интерполация'
                ))

                # Пресечна точка (оранжева)
                Ei_Ed_target = results['En_over_Ed_r']
                if 'Ei/Ed' in df_original.columns:
                    Ei_Ed_values = sorted(df_original['Ei/Ed'].unique())
                    if min(Ei_Ed_values) <= Ei_Ed_target <= max(Ei_Ed_values):
                        x_intercept = None
                        if Ei_Ed_target in Ei_Ed_values:
                            df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                            x_intercept = np.interp(y_at_ratio, df_level['y'], df_level['H/D'])
                        else:
                            for i in range(len(Ei_Ed_values)-1):
                                if Ei_Ed_values[i] < Ei_Ed_target < Ei_Ed_values[i+1]:
                                    df_lower = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i]].sort_values(by='H/D')
                                    df_upper = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i+1]].sort_values(by='H/D')
                                    
                                    x_lower = np.interp(y_at_ratio, df_lower['y'], df_lower['H/D'])
                                    x_upper = np.interp(y_at_ratio, df_upper['y'], df_upper['H/D'])
                                    
                                    x_intercept = x_lower + (x_upper - x_lower) * (Ei_Ed_target - Ei_Ed_values[i]) / (Ei_Ed_values[i+1] - Ei_Ed_values[i])
                                    break

                        if x_intercept is not None:
                            fig.add_trace(go.Scatter(
                                x=[x_intercept], y=[y_at_ratio],
                                mode='markers', marker=dict(color='orange', size=12),
                                name='Пресечна точка'
                            ))
                            # Хоризонтална линия между червената и оранжевата точка
                            fig.add_trace(go.Scatter(
                                x=[target_Hn_D, x_intercept],
                                y=[y_at_ratio, y_at_ratio],
                                mode='lines',
                                line=dict(color='green', dash='dash'),
                                name='Линия между червена и оранжева точка'
                            ))

                            # Вертикална линия от оранжева точка до y=2.5
                            fig.add_trace(go.Scatter(
                                x=[x_intercept, x_intercept],
                                y=[y_at_ratio, 2.5],
                                mode='lines',
                                line=dict(color='purple', dash='dash'),
                                name='Вертикална линия до y=2.5'
                            ))

                            # Запазване на стойностите за експорт
                            st.session_state.x_intercept = x_intercept
                            st.session_state.y_at_ratio = y_at_ratio
                            st.session_state.has_intercept = True
                            st.session_state.sigma_r = round(x_intercept / 2, 3)

        # --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[None, None],  # y не влияе
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            xaxis='x2'  # Свързваме с втората ос
        ))

        fig.update_layout(
            title='Графика на изолинии',
            xaxis=dict(
                title='H/D',
                showgrid=True,
                zeroline=False,
                range=[0, 1]  # Явно задаване на диапазон
            ),
            xaxis2=dict(
                overlaying='x',
                side='top',
                range=[0, 1],  # Явно задаване на диапазон
                showgrid=False,
                zeroline=False,
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1'],
                title='σr'
            ),
            yaxis=dict(
                title='y',
                range=[0, 3]  # Явно задаване на диапазон
            ),
            showlegend=False,
            width=800,  # Добавяне на фиксирана ширина
            height=600   # Добавяне на фиксирана височина
        )

        return fig

    except Exception as e:
        st.error(f"Грешка при създаване на графиката: {e}")
        return None

# PDF Generation function
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
        return None

    # Заглавна страница
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 18)
    pdf.cell(0, 15, 'ОПЪН В МЕЖДИНЕН ПЛАСТ', ln=True, align='C')
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
        ("Диаметър D", f"{D:.2f}", "cm"),
        ("Брой пластове", f"{n}", ""),
    ]
    for i in range(n):
        params.append((f"Пласт {i+1} - Ei", f"{E_values[i]:.2f}", "MPa"))
        params.append((f"Пласт {i+1} - hi", f"{h_values[i]:.2f}", "cm"))
        params.append((f"Пласт {i+1} - Ed", f"{Ed_values[i]:.2f}", "MPa"))
    params.extend([
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
        r"H_{n-1} = \sum_{i=1}^{n-1} h_i",
        r"H_n = \sum_{i=1}^n h_i",
        r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}",
        r"\frac{H_n}{D}",
        r"\frac{Esr}{E_n}",
        r"\frac{E_n}{Ed_n}",
        r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}",
    ]
    pdf.add_formula_section("Основни формули за изчисление:", formulas_section2, columns=2, col_width=95, img_width=85, row_gap=-3)

    # 3. Изчисления (с числени замествания)
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '3. Изчисления', ln=True)

    if layer_idx in st.session_state.layer_results:
        results = st.session_state.layer_results[layer_idx]
        
        # H_n-1
        if layer_idx > 0:
            h_terms_n1 = " + ".join([f"{h_values[i]}" for i in range(layer_idx)])
            formulas_section3 = [
                fr"H_{{{layer_idx}}} = {h_terms_n1} = {results['H_n_1_r']} \, \text{{cm}}"
            ]
        else:
            formulas_section3 = [r"H_{0} = 0 \, \text{cm}"]
        
        # H_n
        h_terms_n = " + ".join([f"{h_values[i]}" for i in range(layer_idx+1)])
        formulas_section3.append(fr"H_{{{layer_idx+1}}} = {h_terms_n} = {results['H_n_r']} \, \text{{cm}}")
        
        # Esr
        if layer_idx > 0:
            numerator = " + ".join([f"{E_values[i]} \times {h_values[i]}" for i in range(layer_idx)])
            denominator = " + ".join([f"{h_values[i]}" for i in range(layer_idx)])
            formulas_section3.append(fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']} \, \text{{MPa}}")
        else:
            formulas_section3.append("Esr = 0 \, \text{MPa} (няма предишни пластове)")
        
        # H_n/D
        formulas_section3.append(fr"\frac{{H_{{{layer_idx+1}}}}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
        
        # E_n
        formulas_section3.append(fr"E_{{{layer_idx+1}}} = {results['En_r']} \, \text{{MPa}}")
        
        # Esr/E_n
        formulas_section3.append(fr"\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = \frac{{{results['Esr_r']}}}{{{results['En_r']}}} = {results['Esr_over_En_r']}")
        
        # E_n/Ed_n
        formulas_section3.append(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")
        
        # σR от номограма
        if st.session_state.sigma_r is not None:
            formulas_section3.append(fr"\sigma_R^{{\mathrm{{номограма}}}} = {st.session_state.sigma_r:.3f} \, \text{{MPa}}")
            
            # Крайно σR
            axle_load = st.session_state.get("axle_load", 100)
            p_loc = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
            if p_loc:
                sigma_final_loc = 1.15 * p_loc * st.session_state.sigma_r
                formulas_section3.append(fr"p = {p_loc:.3f} \, \text{{ (за осов товар {axle_load} kN)}}")
                formulas_section3.append(fr"\sigma_R = 1.15 \times {p_loc:.3f} \times {st.session_state.sigma_r:.3f} = {sigma_final_loc:.3f} \, \text{{MPa}}")

        pdf.add_formula_section("Изчислителни формули:", formulas_section3, columns=2, col_width=95, img_width=85, row_gap=-3)

    pdf.ln(5)

    # 4. Графика
    if st.session_state.fig is not None:
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '4. Графика на номограмата', ln=True)
        pdf.add_plotly_figure(st.session_state.fig, width=160)

    # 5. Допустими напрежения
    img_path = "Допустими опънни напрежения.png"
    if os.path.exists(img_path):
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '5. Допустими опънни напрежения', ln=True)
        pdf.add_external_image(img_path, width=160)
        
    # 6. Резултати и проверка
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '6. Резултати и проверка', ln=True)

    if 'final_sigma_R' in st.session_state and f'manual_sigma_{layer_idx}' in st.session_state.manual_sigma_values:
        manual_sigma = st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}']
        check_passed = st.session_state.final_sigma_R <= manual_sigma

        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(90, 8, 'Параметър', border=1, align='C', fill=True)
        pdf.cell(90, 8, 'Стойност', border=1, align='C', fill=True)
        pdf.ln(8)

        pdf.set_font('DejaVu', '', 10)
        for label, val in [
            ('Изчислено σR', f"{st.session_state.final_sigma_R:.3f} MPa"),
            ('Допустимо σR (ръчно)', f"{manual_sigma:.2f} MPa")
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

    # Save the PDF to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    pdf.cleanup_temp_files()
    
    return temp_file.name

# Calculate button
if st.button(f"Изчисли за пласт {layer_idx+1}"):
    results = calculate_layer(layer_idx)
    st.success(f"Изчисленията за пласт {layer_idx+1} са запазени!")

# Display results
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    if layer_idx > 0:
        h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx)])
        st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(layer_idx)} = {results['H_n_1_r']}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(results['n_for_calc'])])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(results['n_for_calc'])} = {results['H_n_r']}")

    if layer_idx > 0:
        st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
        numerator = " + ".join([f"{results['E_values'][i]} \cdot {results['h_values'][i]}" for i in range(layer_idx)])
        denominator = " + ".join([f"{results['h_values'][i]}" for i in range(layer_idx)])
        st.latex(fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {round(results['Esr_r'])}")
    else:
        st.write("Esr = 0 (няма предишни пластове)")

    st.latex(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
    st.latex(fr"E_{{{layer_idx+1}}} = {results['En_r']}")
    st.latex(fr"\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
    st.latex(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")

    # Visualization
    try:
        df_original = pd.read_csv("danni_1.csv")
        df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D_1.csv")
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

        fig = go.Figure()

        # Add isolines from original df
        if 'Ei/Ed' in df_original.columns:
            for level in sorted(df_original['Ei/Ed'].unique()):
                df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Ei/Ed = {round(level,3)}',
                    line=dict(width=2)
                ))

        # Add isolines from new df
        if 'sr_Ei' in df_new.columns:
            for sr_Ei in sorted(df_new['sr_Ei'].unique()):
                df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Esr/Ei = {round(sr_Ei,3)}',
                    line=dict(width=2)
                ))

        # Interpolation and marking points
        if layer_idx > 0:
            sr_Ei_values = sorted(df_new['sr_Ei'].unique())
            target_sr_Ei = results['Esr_over_En_r']
            target_Hn_D = results['ratio_r']

            y_at_ratio = None
            if min(sr_Ei_values) <= target_sr_Ei <= max(sr_Ei_values):
                if target_sr_Ei in sr_Ei_values:
                    df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
                    y_at_ratio = np.interp(target_Hn_D, df_target['H/D'], df_target['y'])
                else:
                    for i in range(len(sr_Ei_values)-1):
                        if sr_Ei_values[i] < target_sr_Ei < sr_Ei_values[i+1]:
                            df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                            df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[i+1]].sort_values(by='H/D')
                            
                            y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
                            y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])
                            
                            y_at_ratio = y_lower + (y_upper - y_lower) * (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i+1] - sr_Ei_values[i])
                            break

            if y_at_ratio is not None:
                # Вертикална линия (синя)
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                    mode='lines', line=dict(color='blue', dash='dash'),
                    name='Вертикална линия'
                ))

                # Червена точка
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D], y=[y_at_ratio],
                    mode='markers', marker=dict(color='red', size=10),
                    name='Точка на интерполация'
                ))

                # Пресечна точка (оранжева)
                Ei_Ed_target = results['En_over_Ed_r']
                if 'Ei/Ed' in df_original.columns:
                    Ei_Ed_values = sorted(df_original['Ei/Ed'].unique())
                    if min(Ei_Ed_values) <= Ei_Ed_target <= max(Ei_Ed_values):
                        x_intercept = None
                        if Ei_Ed_target in Ei_Ed_values:
                            df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                            x_intercept = np.interp(y_at_ratio, df_level['y'], df_level['H/D'])
                        else:
                            for i in range(len(Ei_Ed_values)-1):
                                if Ei_Ed_values[i] < Ei_Ed_target < Ei_Ed_values[i+1]:
                                    df_lower = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i]].sort_values(by='H/D')
                                    df_upper = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i+1]].sort_values(by='H/D')
                                    
                                    x_lower = np.interp(y_at_ratio, df_lower['y'], df_lower['H/D'])
                                    x_upper = np.interp(y_at_ratio, df_upper['y'], df_upper['H/D'])
                                    
                                    x_intercept = x_lower + (x_upper - x_lower) * (Ei_Ed_target - Ei_Ed_values[i]) / (Ei_Ed_values[i+1] - Ei_Ed_values[i])
                                    break

                        if x_intercept is not None:
                            fig.add_trace(go.Scatter(
                                x=[x_intercept], y=[y_at_ratio],
                                mode='markers', marker=dict(color='orange', size=12),
                                name='Пресечна точка'
                            ))
                            # Хоризонтална линия между червената и оранжевата точка
                            fig.add_trace(go.Scatter(
                                x=[target_Hn_D, x_intercept],
                                y=[y_at_ratio, y_at_ratio],
                                mode='lines',
                                line=dict(color='green', dash='dash'),
                                name='Линия между червена и оранжева точка'
                            ))

                            # Вертикална линия от оранжева точка до y=2.5
                            fig.add_trace(go.Scatter(
                                x=[x_intercept, x_intercept],
                                y=[y_at_ratio, 2.5],
                                mode='lines',
                                line=dict(color='purple', dash='dash'),
                                name='Вертикална линия до y=2.5'
                            ))

        # --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[None, None],  # y не влияе
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            xaxis='x2'  # Свързваме с втората ос
        ))

        fig.update_layout(
            title='Графика на изолинии',
            xaxis=dict(
                title='H/D',
                showgrid=True,
                zeroline=False,
            ),
            xaxis2=dict(
                overlaying='x',
                side='top',
                range=[fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 0, 1],
                showgrid=False,
                zeroline=False,
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1'],
                title='σr'
            ),
            yaxis=dict(
                title='y',
                range=[0, 3]
            ),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=600)
        
        # Проверка дали x_intercept е дефинирана и не е None
        if ('x_intercept' in locals()) and (x_intercept is not None):
            sigma_r = round(x_intercept / 2, 3)
            st.markdown(f"**Изчислено σr = {sigma_r}**")
            
            # Запазваме стойността в session_state
            st.session_state.final_sigma = sigma_r

            # Вземане на осов товар от първата страница
            axle_load = st.session_state.get("axle_load", 100)
            
            # Определяне на p според осовия товар
            if axle_load == 100:
                p = 0.620
            elif axle_load == 115:
                p = 0.633
            else:
                p = None
            st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
            if p is not None:
                st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")
            else:
                st.warning("❗ Не е зададен валиден осов товар. Не може да се изчисли p.")
                
            # Вземаме sigma от session_state, ако има
            sigma = st.session_state.get("final_sigma", None)

            # Променлива за крайното σR
            sigma_final = None
            
            if p is not None and sigma is not None:
                sigma_final = 1.15 * p * sigma
                st.markdown("### Формула за изчисление на крайното напрежение σR:")
                st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
                st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
                st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
                
                # Запазваме крайната стойност за проверката
                st.session_state["final_sigma_R"] = sigma_final
            else:
                st.warning("❗ Липсва p или σR от номограмата за изчисление.")


    
            # Секция за ръчно въвеждане
            st.markdown(
                """
                <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
                    <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Инициализираме ръчната стойност за този пласт, ако не съществува
            if f'manual_sigma_{layer_idx}' not in st.session_state.manual_sigma_values:
                st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = sigma_r

            # Поле за ръчно въвеждане
            manual_value = st.number_input(
                label="Въведете ръчно отчетена стойност σR [MPa]",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', sigma_r),
                step=0.1,
                key=f"manual_sigma_input_{layer_idx}",
                label_visibility="visible"
            )
            
            # Запазваме ръчно въведената стойност
            st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = manual_value
            
            # Проверка на условието (без бутон, автоматично при промяна)
            sigma_to_compare = st.session_state.get("final_sigma_R", None)
            
            if sigma_to_compare is not None:
                # Проверяваме дали вече имаме резултат за този пласт
                if f'check_result_{layer_idx}' not in st.session_state.check_results:
                    st.session_state.check_results[f'check_result_{layer_idx}'] = None
                
                # Проверка на условието
                check_passed = sigma_to_compare <= manual_value
                st.session_state.check_results[f'check_result_{layer_idx}'] = check_passed
                
                # Показваме резултата
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
        else:
            st.markdown("**σr = -** (Няма изчислена стойност)")

    except Exception as e:
        st.error(f"Грешка при визуализацията: {e}")

    # PDF Export Button
    st.markdown("---")
    st.markdown("### Експорт на отчет")
    
    if st.button("Генерирай PDF отчет"):
        with st.spinner("Генериране на PDF..."):
            pdf_path = generate_pdf_report()
            
            # Read the PDF file
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Create download button
            st.download_button(
                label="Свали PDF отчета",
                data=pdf_bytes,
                file_name="opyn_v_mejdinen_report.pdf",
                mime="application/pdf"
            )
            
            # Clean up temporary file
            os.unlink(pdf_path)
            
            st.session_state.pdf_exported = True

    # Линк към предишната страница
    st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")
