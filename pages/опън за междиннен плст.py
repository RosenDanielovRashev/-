import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from fpdf import FPDF
import base64
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import mathtext
import cairosvg
from io import BytesIO

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

# Функции за рендиране на математически формули
def render_formula_to_svg(formula, output_path):
    """Рендира формула като SVG чрез matplotlib.mathtext"""
    try:
        parser = mathtext.MathTextParser("path")
        parser.to_svg(f"${formula}$", output_path)
        return output_path
    except Exception as e:
        print(f"Грешка при рендиране на SVG: {e}")
        raise

def svg_to_png(svg_path, png_path=None, dpi=300):
    """Конвертира SVG към PNG с висока резолюция"""
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)
        return png_path
    except Exception as e:
        print(f"Грешка при конвертиране SVG към PNG: {e}")
        raise

def render_formula_to_image_fallback(formula, fontsize=22, dpi=450):
    """Fallback: рендва формула директно в PNG чрез matplotlib"""
    try:
        fig = plt.figure(figsize=(8, 2.5))
        fig.text(0.05, 0.5, f'${formula}$', fontsize=fontsize)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Грешка при fallback рендиране: {e}")
        raise

def formula_png_from_svg_or_fallback(formula_text, dpi=300):
    """Създава PNG от формула чрез SVG→PNG или fallback директно към PNG"""
    try:
        # Опит за векторно рендиране (SVG → PNG)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
            render_formula_to_svg(formula_text, tmp_svg.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
                svg_to_png(tmp_svg.name, tmp_png.name, dpi=dpi)
                return tmp_png.name
    except Exception as e:
        print(f"SVG метод се провали, опитвам fallback: {e}")
        # Fallback: директно PNG от matplotlib
        try:
            buf = render_formula_to_image_fallback(formula_text, dpi=450)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(buf.getvalue())
                return tmp_file.name
        except Exception as e2:
            print(f"И двата метода се провалиха: {e2}")
            return None
            
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
        # Try to find the CSV files in different locations
        csv_paths = [
            "danni_1.csv",
            "./danni_1.csv",
            "pages/danni_1.csv",
            "../danni_1.csv"
        ]
        
        df_original = None
        for path in csv_paths:
            try:
                df_original = pd.read_csv(path)
                break
            except:
                continue
                
        if df_original is None:
            st.error("Файлът 'danni_1.csv' не е намерен. Моля, уверете се, че файлът съществува.")
            # Remove the return statement here
        else:
            csv_paths2 = [
                "Оразмеряване на опън за междиннен плстH_D_1.csv",
                "./Оразмеряване на опън за междиннен плстH_D_1.csv",
                "pages/Оразмеряване на опън за междиннен плстH_D_1.csv",
                "../Оразмеряване на опън за междиннен плстH_D_1.csv"
            ]
            
            df_new = None
            for path in csv_paths2:
                try:
                    df_new = pd.read_csv(path)
                    break
                except:
                    continue
                    
            if df_new is None:
                st.error("Файлът 'Оразмеряване на опън за междиннен плстH_D_1.csv' не е намерен.")
            else:
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
                x_intercept = None  # Initialize x_intercept
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
                                            
                                            # ПОПРАВЕНО: Правилно отстъпване и изчисление
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

                                    # Calculate sigma_r
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

                # Try to find the image in different locations
                image_paths = [
                    "Допустими опънни напрежения.png",
                    "./Допустими опънни напрежения.png",
                    "pages/Допустими опънни напрежения.png",
                    "../Допустими опънни напрежения.png"
                ]
                
                img_found = False
                for path in image_paths:
                    try:
                        st.image(path, caption="Допустими опънни напрежения", width=600)
                        img_found = True
                        break
                    except:
                        continue
                        
                if not img_found:
                    st.warning("Изображението 'Допустими опънни напрежения.png' не е намерено.")

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
                    st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = sigma_r if 'sigma_r' in locals() else 0.0

                # Поле за ръчно въвеждане
                manual_value = st.number_input(
                    label="Въведете ръчно отчетена стойност σR [MPa]",
                    min_value=0.0,
                    max_value=20.0,
                    value=st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', sigma_r if 'sigma_r' in locals() else 0.0),
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

    except Exception as e:
        st.error(f"Грешка при визуализацията: {e}")

    def generate_pdf_report(layer_idx, results, D, sigma_r=None, sigma_final=None, manual_value=None, check_passed=None):
        # Създаваме PDF клас с разширена функционалност
        class EnhancedPDF(FPDF):
            def add_latex_formula(self, formula_text, width=100, line_gap=12, align='L'):
                """Добавя формула като изображение в PDF"""
                try:
                    png_path = formula_png_from_svg_or_fallback(formula_text)
                    if png_path and os.path.exists(png_path):
                        # Запазваме текущата позиция
                        x = self.get_x()
                        y = self.get_y()
                        
                        # Центриране ако е необходимо
                        if align == 'C':
                            x = (210 - width) / 2  # 210mm е ширината на A4
                        
                        self.image(png_path, x=x, y=y, w=width)
                        self.ln(line_gap + width * 0.22)
                        
                        # Изтриване на временния файл
                        try:
                            os.unlink(png_path)
                        except:
                            pass
                    else:
                        # Fallback: показване като чист текст
                        self.set_font("DejaVu", "I", 10)
                        self.multi_cell(0, 8, formula_text)
                        self.ln(5)
                except Exception as e:
                    print(f"Грешка при добавяне на формула: {e}")
                    # Fallback
                    self.set_font("DejaVu", "I", 10)
                    self.multi_cell(0, 8, formula_text)
                    self.ln(5)
        
        pdf = EnhancedPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Добавяне на шрифтове DejaVu
        try:
            font_path = os.path.join("pages", "fonts", "DejaVuSans.ttf")
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.add_font("DejaVu", "B", font_path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), uni=True)
            pdf.set_font("DejaVu", "", 12)
        except:
            # Fallback към стандартни шрифтове ако DejaVu не е наличен
            pdf.set_font("Arial", "", 12)
        
        pdf.add_page()
        
        # Заглавие
        pdf.set_font("DejaVu", "B", 16)
        pdf.cell(0, 10, "ОПЪННО НАПРЕЖЕНИЕ В МЕЖДИНЕН ПЛАСТ", 0, 1, 'C')
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 8, "ОТ ПЪТНАТА КОНСТРУКЦИЯ - ФИГ. 9.3", 0, 1, 'C')
        pdf.ln(5)
        
        # Хоризонтална линия
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)
        
        # 1. Входни параметри
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "1. ВХОДНИ ПАРАМЕТРИ", 0, 1)
        pdf.set_font("DejaVu", "", 10)
        
        # Таблица с входни параметри (форматирана като в снимката)
        col_widths = [60, 40, 40]
        
        # Заглавия на колоните
        pdf.set_font("DejaVu", "B", 10)
        pdf.cell(col_widths[0], 8, "Параметър", 1, 0, 'C')
        pdf.cell(col_widths[1], 8, "Стойност", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, "Мерна единица", 1, 1, 'C')
        
        # Данни в таблицата
        pdf.set_font("DejaVu", "", 10)
        
        # Диаметър D
        pdf.cell(col_widths[0], 8, "Диаметър D", 1, 0)
        pdf.cell(col_widths[1], 8, f"{D}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, "cm", 1, 1, 'C')
        
        # Брой пластове
        pdf.cell(col_widths[0], 8, "Брой пластове", 1, 0)
        pdf.cell(col_widths[1], 8, f"{len(h_values)}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, "", 1, 1, 'C')
        
        # Данни за всеки пласт
        for i in range(len(h_values)):
            # Ei
            pdf.cell(col_widths[0], 8, f"Пласт {i+1} - Ei", 1, 0)
            pdf.cell(col_widths[1], 8, f"{E_values[i]}", 1, 0, 'C')
            pdf.cell(col_widths[2], 8, "MPa", 1, 1, 'C')
            
            # hi (закръглено до 2 знака)
            pdf.cell(col_widths[0], 8, f"Пласт {i+1} - hi", 1, 0)
            pdf.cell(col_widths[1], 8, f"{round(h_values[i], 2)}", 1, 0, 'C')
            pdf.cell(col_widths[2], 8, "cm", 1, 1, 'C')
        
        # Ed
        pdf.cell(col_widths[0], 8, "Ed", 1, 0)
        pdf.cell(col_widths[1], 8, f"{Ed_values[layer_idx]}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, "MPa", 1, 1, 'C')
        
        # Осова тежест
        axle_load = st.session_state.get("axle_load", 100)
        pdf.cell(col_widths[0], 8, "Осова тежест", 1, 0)
        pdf.cell(col_widths[1], 8, f"{axle_load}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, "kN", 1, 1, 'C')
        
        pdf.ln(8)
        
        # 2. Формули за изчисление
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "2. ФОРМУЛИ ЗА ИЗЧИСЛЕНИЕ", 0, 1)
        pdf.set_font("DejaVu", "", 10)
        
        # По-добро форматиране на формулите
        formulas = [
            r"H_{n-1} = \sum_{i=1}^{n-1} h_i",
            r"H_n = \sum_{i=1}^n h_i",
            r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}",
            r"\frac{H_n}{D}",
            r"\frac{Esr}{E_n}",
            r"\frac{E_n}{Ed_n}",
            r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}"
        ]
        
        for formula in formulas:
            pdf.cell(5, 8, "", 0, 0)  # Отстъп
            pdf.add_latex_formula(formula, width=180, line_gap=8, align='L')
        
        pdf.ln(5)
        
        # 3. Изчисления
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, f"3. ИЗЧИСЛЕНИЯ ЗА ПЛАСТ {layer_idx+1}", 0, 1)
        pdf.set_font("DejaVu", "", 10)
        
        # Създаване на таблица за резултатите
        col_widths_calc = [70, 50]
        
        pdf.cell(col_widths_calc[0], 8, f"H{to_subscript(layer_idx)}:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['H_n_1_r']} cm", 0, 1)
        
        pdf.cell(col_widths_calc[0], 8, f"H{to_subscript(results['n_for_calc'])}:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['H_n_r']} cm", 0, 1)
        
        if layer_idx > 0:
            pdf.cell(col_widths_calc[0], 8, "Esr:", 0, 0)
            pdf.cell(col_widths_calc[1], 8, f"{results['Esr_r']} MPa", 0, 1)
            
            # Показване на формулата за Esr
            numerator = " + ".join([f"{results['E_values'][i]} \cdot {results['h_values'][i]}" for i in range(layer_idx)])
            denominator = " + ".join([f"{results['h_values'][i]}" for i in range(layer_idx)])
            esr_formula = fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {round(results['Esr_r'])}"
            pdf.add_latex_formula(esr_formula, width=180, line_gap=6, align='L')
        else:
            pdf.cell(col_widths_calc[0], 8, "Esr:", 0, 0)
            pdf.cell(col_widths_calc[1], 8, "0 (няма предишни пластове)", 0, 1)
        
        pdf.cell(col_widths_calc[0], 8, f"H{to_subscript(results['n_for_calc'])}/D:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['ratio_r']}", 0, 1)
        
        pdf.cell(col_widths_calc[0], 8, f"E{to_subscript(layer_idx+1)}:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['En_r']} MPa", 0, 1)
        
        pdf.cell(col_widths_calc[0], 8, f"Esr/E{to_subscript(layer_idx+1)}:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['Esr_over_En_r']}", 0, 1)
        
        pdf.cell(col_widths_calc[0], 8, f"E{to_subscript(layer_idx+1)}/Ed{to_subscript(layer_idx+1)}:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{results['En_over_Ed_r']}", 0, 1)
        
        if sigma_r is not None:
            pdf.cell(col_widths_calc[0], 8, "σr (от номограма):", 0, 0)
            pdf.cell(col_widths_calc[1], 8, f"{sigma_r} MPa", 0, 1)
        
        # Информация за осов товар
        axle_load = st.session_state.get("axle_load", 100)
        pdf.cell(col_widths_calc[0], 8, "Осов товар:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{axle_load} kN", 0, 1)
        
        # Определяне на p според осовия товар
        if axle_load == 100:
            p = 0.620
        elif axle_load == 115:
            p = 0.633
        else:
            p = "неизвестен"
        
        pdf.cell(col_widths_calc[0], 8, "Коефициент p:", 0, 0)
        pdf.cell(col_widths_calc[1], 8, f"{p} MPa", 0, 1)
        
        if sigma_final is not None:
            pdf.set_font("DejaVu", "B", 10)
            pdf.cell(col_widths_calc[0], 10, "Крайно σR:", 0, 0)
            pdf.cell(col_widths_calc[1], 10, f"{sigma_final:.3f} MPa", 0, 1)
            
            # Показване на формулата за σR
            if sigma_r is not None:
                sigma_formula = fr"\sigma_R = 1.15 \cdot {p} \cdot {sigma_r} = {sigma_final:.3f}  \text{{MPa}}"
                pdf.add_latex_formula(sigma_formula, width=180, line_gap=6, align='L')
        
        pdf.ln(5)
        
        # ... останалата част от функцията остава непроменена
        # (секции 4, 5, 6 и т.н.)
        
        # 4. Графика на номограмата
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "4. ГРАФИКА НА НОМОГРАМАТА", 0, 1)
        
        # Запазване на графиката като временно изображение с по-висока резолюция
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                # Експорт във формат SVG за по-добро качество
                fig.write_image(tmpfile.name, format="png", width=1000, height=700, scale=2)
                pdf.image(tmpfile.name, x=10, y=None, w=190)
                os.unlink(tmpfile.name)
        except Exception as e:
            # Ако SVG не се поддържа, опитайте с PNG
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.write_image(tmpfile.name, width=1000, height=700, scale=2)
                    pdf.image(tmpfile.name, x=10, y=None, w=190)
                    os.unlink(tmpfile.name)
            except Exception as e2:
                pdf.set_font("DejaVu", "", 10)
                pdf.cell(0, 6, f"Грешка при добавяне на графиката: {e2}", 0, 1)
        
        pdf.ln(5)
        
        # 5. Допустими опънни напрежения
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "5. ДОПУСТИМИ ОПЪННИ НАПРЕЖЕНИЯ", 0, 1)
        
        try:
            # Try to find the image
            image_paths = [
                "Допустими опънни напрежения.png",
                "./Допустими опънни напрежения.png",
                "pages/Допустими опънни напрежения.png",
                "../Допустими опънни напрежения.png"
            ]
            
            img_found = False
            for path in image_paths:
                try:
                    # Опит за зареждане на изображението с по-висока резолюция
                    pdf.image(path, x=10, y=None, w=190)
                    img_found = True
                    break
                except:
                    continue
                    
            if not img_found:
                pdf.set_font("DejaVu", "", 10)
                pdf.cell(0, 6, "Изображението не е намерено", 0, 1)
        except Exception as e:
            pdf.set_font("DejaVu", "", 10)
            pdf.cell(0, 6, f"Грешка при добавяне на изображението: {e}", 0, 1)
        
        pdf.ln(8)
        
        # 6. Резултати и проверка
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, "6. РЕЗУЛТАТИ И ПРОВЕРКА", 0, 1)
        pdf.set_font("DejaVu", "", 10)
        
        if manual_value is not None:
            pdf.cell(70, 8, "Ръчно отчетена стойност σR:", 0, 0)
            pdf.cell(0, 8, f"{manual_value} MPa", 0, 1)
        
        if sigma_final is not None:
            pdf.cell(70, 8, "Изчислена стойност σR:", 0, 0)
            pdf.cell(0, 8, f"{sigma_final:.3f} MPa", 0, 1)
        
        if check_passed is not None:
            pdf.ln(3)
            if check_passed:
                pdf.set_fill_color(220, 255, 220)
                pdf.cell(0, 8, "✓ ПРОВЕРКАТА Е УДОВЛЕТВОРЕНА", 1, 1, 'C', True)
                pdf.cell(0, 6, f"Изчисленото σR = {sigma_final:.3f} MPa ≤ {manual_value} MPa (допустимото σR)", 0, 1)
            else:
                pdf.set_fill_color(255, 220, 220)
                pdf.cell(0, 8, "✗ ПРОВЕРКАТА НЕ Е УДОВЛЕТВОРЕНА", 1, 1, 'C', True)
                pdf.cell(0, 6, f"Изчисленото σR = {sigma_final:.3f} MPa > {manual_value} MPa (допустимото σR)", 0, 1)
        
        # Добавяне на дата и час на генериране
        pdf.ln(10)
        pdf.set_font("DejaVu", "", 8)
        from datetime import datetime
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 5, f"Генерирано на: {generated_at}", 0, 0, 'R')
        
        # Запазване на PDF във временен файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            
            # Четене на файла и връщане като base64
            with open(tmpfile.name, "rb") as f:
                pdf_bytes = f.read()
            
            os.unlink(tmpfile.name)
            return pdf_bytes
        
    # Добавяне на бутон за генериране на PDF отчет
    if st.button("Генерирай PDF отчет"):
        with st.spinner("Генериране на PDF..."):
            # Вземете необходимите данни за отчета
            sigma_r = st.session_state.get("final_sigma", None)
            sigma_final = st.session_state.get("final_sigma_R", None)
            manual_value = st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', None)
            check_passed = st.session_state.check_results.get(f'check_result_{layer_idx}', None)
            
            # Генериране на PDF
            pdf_bytes = generate_pdf_report(
                layer_idx, results, D, sigma_r, sigma_final, manual_value, check_passed
            )
            
            # Показване на линк за изтегляне
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="опън_за_междинен_пласт_отчет.pdf">Изтегли PDF отчет</a>'
            st.markdown(href, unsafe_allow_html=True)

# Линк към предишната страница
st.markdown('[Към Оразмеряване на пътна конструкция](orazmeriavane_patna_konstrukcia.py)')
