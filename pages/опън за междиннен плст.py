import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import base64
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import mathtext
from io import BytesIO
import io
from datetime import datetime
import plotly.io as pio

# ReportLab импорти за новия стил
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage

# Опит за импорт на cairosvg
try:
    import cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# ПОДОБРЕНО ОФОРМЛЕНИЕ
st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 18px !important;
        }
        .main .block-container {
            max-width: 900px;          /* намалена от 1000px */
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stPlotlyChart {
            width: 100% !important;
        }
        .stNumberInput > label {
            font-size: 14px !important;
        }
        .stSelectbox > label {
            font-size: 14px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция фиг.9.3")

# Функции за рендиране на математически формули
def render_formula_to_svg(formula, output_path):
    try:
        parser = mathtext.MathTextParser("path")
        parser.to_svg(f"${formula}$", output_path)
        return output_path
    except Exception as e:
        print(f"Грешка при рендиране на SVG: {e}")
        raise

def svg_to_png(svg_path, png_path=None, dpi=300):
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)
        return png_path
    except Exception as e:
        print(f"Грешка при конвертиране SVG към PNG: {e}")
        raise

def render_formula_to_image_fallback(formula, fontsize=22, dpi=450):
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

def render_formula_to_image(formula_text, fontsize=26, dpi=150):
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = fontsize
    
    fig = plt.figure(figsize=(10.56, 1.58))
    plt.text(0.5, 0.5, f'${formula_text}$', 
             horizontalalignment='center', 
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=fontsize)
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

def to_superscript(number):
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(number).translate(superscripts)

# Initialize session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}
if 'manual_sigma_values' not in st.session_state:
    st.session_state.manual_sigma_values = {}
if 'check_results' not in st.session_state:
    st.session_state.check_results = {}

# Проверка за данни от главния файл
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

# Функция за оптимизирана графика за PDF
def create_optimized_pdf_figure_intermediate():
    try:
        csv_paths = ["danni_1.csv", "./danni_1.csv", "pages/danni_1.csv", "../danni_1.csv"]
        df_original = None
        for path in csv_paths:
            if os.path.exists(path):
                df_original = pd.read_csv(path)
                break
        
        if df_original is None:
            return None
        
        csv_paths2 = ["Оразмеряване на опън за междиннен плстH_D_1.csv", "./Оразмеряване на опън за междиннен плстH_D_1.csv",
                      "pages/Оразмеряване на опън за междиннен плстH_D_1.csv", "../Оразмеряване на опън за междиннен плстH_D_1.csv"]
        df_new = None
        for path in csv_paths2:
            if os.path.exists(path):
                df_new = pd.read_csv(path)
                break
        
        if df_new is None:
            return None
        
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
        
        fig = go.Figure()
        colors_isolines = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        if 'Ei/Ed' in df_original.columns:
            levels = sorted(df_original['Ei/Ed'].unique())
            for i, level in enumerate(levels):
                df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines',
                                         name=f'Ei/Ed = {round(level,2)}',
                                         line=dict(color=colors_isolines[i % len(colors_isolines)], width=1.5)))
        
        if 'sr_Ei' in df_new.columns:
            sr_Ei_levels = sorted(df_new['sr_Ei'].unique())
            for i, sr_Ei in enumerate(sr_Ei_levels):
                df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                color = colors_isolines[(len(levels if 'levels' in locals() else 0) + i) % len(colors_isolines)]
                fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines',
                                         name=f'Esr/Ei = {round(sr_Ei,2)}',
                                         line=dict(color=color, width=1.5, dash='dash')))
        
        # ПОДОБРЕНО ОФОРМЛЕНИЕ ЗА PDF
        fig.update_layout(
            title=dict(text='Номограма: σR в междинен пласт', font=dict(size=16)),
            xaxis=dict(title='H/D', range=[0,1], title_font=dict(size=14), tickfont=dict(size=12)),
            xaxis2=dict(overlaying='x', side='top', range=[0,1], title='σr', title_font=dict(size=14), tickfont=dict(size=12),
                        tickvals=[0,0.25,0.5,0.75,1], ticktext=['0','0.25','0.5','0.75','1']),
            yaxis=dict(title='y', range=[0,2.7], title_font=dict(size=14), tickfont=dict(size=12)),
            legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle',
                        bgcolor='rgba(255,255,255,0.9)', font=dict(size=10)),
            margin=dict(l=60, r=180, t=60, b=60),
            width=850,
            height=650,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    except Exception as e:
        print(f"Грешка при създаване на PDF графика: {e}")
        return None

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
        st.latex(r"E_{sr} = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
        numerator = " + ".join([f"{results['E_values'][i]} \cdot {results['h_values'][i]}" for i in range(layer_idx)])
        denominator = " + ".join([f"{results['h_values'][i]}" for i in range(layer_idx)])
        st.latex(fr"E_{{sr}} = \frac{{{numerator}}}{{{denominator}}} = {round(results['Esr_r'])}")
    else:
        st.write("Esr = 0 (няма предишни пластове)")

    st.latex(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
    st.latex(fr"E_{{{layer_idx+1}}} = {results['En_r']}")
    st.latex(fr"\frac{{E_{{sr}}}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
    st.latex(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")

    # Visualization
    try:
        csv_paths = ["danni_1.csv", "./danni_1.csv", "pages/danni_1.csv", "../danni_1.csv"]
        df_original = None
        for path in csv_paths:
            if os.path.exists(path):
                df_original = pd.read_csv(path)
                break
        
        if df_original is None:
            st.error("Файлът 'danni_1.csv' не е намерен.")
        else:
            csv_paths2 = ["Оразмеряване на опън за междиннен плстH_D_1.csv", "./Оразмеряване на опън за междиннен плстH_D_1.csv",
                          "pages/Оразмеряване на опън за междиннен плстH_D_1.csv", "../Оразмеряване на опън за междиннен плстH_D_1.csv"]
            df_new = None
            for path in csv_paths2:
                if os.path.exists(path):
                    df_new = pd.read_csv(path)
                    break
            
            if df_new is None:
                st.error("Файлът 'Оразмеряване на опън за междиннен плстH_D_1.csv' не е намерен.")
            else:
                df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

                fig = go.Figure()
                colors_isolines = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                   '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

                # Изолинии Ei/Ed
                if 'Ei/Ed' in df_original.columns:
                    levels = sorted(df_original['Ei/Ed'].unique())
                    for i, level in enumerate(levels):
                        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                        y_positions = np.linspace(0.5, 2.5, len(levels))
                        target_y = y_positions[i]
                        closest_idx = (df_level['y'] - target_y).abs().idxmin()
                        x_mid = df_level.loc[closest_idx, 'H/D']
                        y_mid = df_level.loc[closest_idx, 'y']
                        
                        fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines',
                                                 name=f'Ei/Ed = {round(level,2)}',
                                                 line=dict(color=colors_isolines[i % len(colors_isolines)], width=2)))
                        fig.add_trace(go.Scatter(x=[x_mid], y=[y_mid], mode='text',
                                                 text=[f'{round(level,2)}'],
                                                 textposition='middle right',
                                                 textfont=dict(size=9, color=colors_isolines[i % len(colors_isolines)]),
                                                 showlegend=False, hoverinfo='skip'))

                # Изолинии Esr/Ei
                if 'sr_Ei' in df_new.columns:
                    sr_Ei_levels = sorted(df_new['sr_Ei'].unique())
                    offset = len(levels) if 'levels' in locals() else 0
                    for i, sr_Ei in enumerate(sr_Ei_levels):
                        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                        mid_idx = 2 * len(df_level) // 3 if len(df_level) > 10 else len(df_level) // 2
                        x_mid = df_level['H/D'].iloc[mid_idx]
                        y_mid = df_level['y'].iloc[mid_idx]
                        color = colors_isolines[(offset + i) % len(colors_isolines)]
                        
                        fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines',
                                                 name=f'Esr/Ei = {round(sr_Ei,2)}',
                                                 line=dict(color=color, width=2, dash='dash')))
                        fig.add_trace(go.Scatter(x=[x_mid], y=[y_mid], mode='text',
                                                 text=[f'{round(sr_Ei,2)}'],
                                                 textposition='middle left',
                                                 textfont=dict(size=9, color=color),
                                                 showlegend=False, hoverinfo='skip'))

                # Интерполация и маркиране
                x_intercept = None
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
                        fig.add_trace(go.Scatter(x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                                                 mode='lines', line=dict(color='blue', dash='dash', width=2),
                                                 name='Вертикална линия', showlegend=True))
                        fig.add_trace(go.Scatter(x=[target_Hn_D], y=[y_at_ratio],
                                                 mode='markers', marker=dict(color='red', size=12, symbol='circle',
                                                                            line=dict(color='darkred', width=2)),
                                                 name='Точка на интерполация', showlegend=True))

                        Ei_Ed_target = results['En_over_Ed_r']
                        if 'Ei/Ed' in df_original.columns:
                            Ei_Ed_values = sorted(df_original['Ei/Ed'].unique())
                            if min(Ei_Ed_values) <= Ei_Ed_target <= max(Ei_Ed_values):
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
                                    fig.add_trace(go.Scatter(x=[x_intercept], y=[y_at_ratio],
                                                             mode='markers', marker=dict(color='orange', size=14, symbol='diamond',
                                                                                        line=dict(color='darkorange', width=2)),
                                                             name='Пресечна точка', showlegend=True))
                                    fig.add_trace(go.Scatter(x=[target_Hn_D, x_intercept], y=[y_at_ratio, y_at_ratio],
                                                             mode='lines', line=dict(color='green', dash='dash', width=2),
                                                             name='Хоризонтална линия', showlegend=True))
                                    fig.add_trace(go.Scatter(x=[x_intercept, x_intercept], y=[y_at_ratio, 2.5],
                                                             mode='lines', line=dict(color='purple', dash='dash', width=2),
                                                             name='Вертикална линия до σr', showlegend=True))

                                    sigma_r = round(x_intercept / 2, 3)
                                    st.markdown(f"**Изчислено σr = {sigma_r}**")
                                    st.session_state.final_sigma = sigma_r

                                    axle_load = st.session_state.get("axle_load", 100)
                                    p = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else None
                                    st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
                                    if p is not None:
                                        st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")
                                    else:
                                        st.warning("❗ Не е зададен валиден осов товар.")

                                    sigma = st.session_state.get("final_sigma", None)
                                    if p is not None and sigma is not None:
                                        sigma_final = 1.15 * p * sigma
                                        st.markdown("### Формула за изчисление на крайното напрежение σR:")
                                        st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
                                        st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
                                        st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
                                        st.session_state["final_sigma_R"] = sigma_final

                # ПОДОБРЕНО ОФОРМЛЕНИЕ ЗА STREAMLIT
                fig.update_layout(
                    title=dict(text='Графика на изолинии', font=dict(size=16, color='black')),
                    xaxis=dict(title='H/D', title_font=dict(size=14), tickfont=dict(size=12),
                               range=[0, 1], gridcolor='lightgray', linecolor='black', mirror=True),
                    xaxis2=dict(overlaying='x', side='top', range=[0, 1], title='σr',
                                title_font=dict(size=14), tickfont=dict(size=12),
                                tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=['0', '0.25', '0.5', '0.75', '1'],
                                showgrid=False),
                    yaxis=dict(title='y', title_font=dict(size=14), tickfont=dict(size=12),
                               range=[0, 2.7], gridcolor='lightgray', linecolor='black', mirror=True),
                    legend=dict(orientation='v', x=1.02, y=0.5, xanchor='left', yanchor='middle',
                                bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='black', borderwidth=1,
                                font=dict(size=11)),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=60, r=180, t=60, b=60),
                    height=700,
                    autosize=True
                )

                st.plotly_chart(fig, use_container_width=True, config={'responsive': True, 'scrollZoom': True})

                # Изображение с допустими напрежения
                image_paths = ["Допустими опънни напрежения.png", "./Допустими опънни напрежения.png",
                               "pages/Допустими опънни напрежения.png", "../Допустими опънни напрежения.png"]
                img_found = False
                for path in image_paths:
                    if os.path.exists(path):
                        st.image(path, caption="Допустими опънни напрежения", use_container_width=True)
                        img_found = True
                        break
                if not img_found:
                    st.warning("Изображението 'Допустими опънни напрежения.png' не е намерено.")

                # Ръчно въвеждане
                st.markdown("""
                    <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
                        <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
                    </div>
                """, unsafe_allow_html=True)

                if f'manual_sigma_{layer_idx}' not in st.session_state.manual_sigma_values:
                    st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = sigma_r if 'sigma_r' in locals() else 0.0

                manual_value = st.number_input(
                    label="Въведете ръчно отчетена стойност σR [MPa]",
                    min_value=0.0, max_value=20.0, step=0.1,
                    value=st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', sigma_r if 'sigma_r' in locals() else 0.0),
                    key=f"manual_sigma_input_{layer_idx}"
                )
                st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = manual_value

                sigma_to_compare = st.session_state.get("final_sigma_R", None)
                if sigma_to_compare is not None:
                    check_passed = sigma_to_compare <= manual_value
                    st.session_state.check_results[f'check_result_{layer_idx}'] = check_passed
                    if check_passed:
                        st.success(f"✅ Проверката е удовлетворена: {sigma_to_compare:.3f} MPa ≤ {manual_value:.3f} MPa")
                    else:
                        st.error(f"❌ Проверката НЕ е удовлетворена: {sigma_to_compare:.3f} MPa > {manual_value:.3f} MPa")
                else:
                    st.warning("❗ Няма изчислена стойност σR за проверка.")

    except Exception as e:
        st.error(f"Грешка при визуализацията: {e}")
        import traceback
        st.error(traceback.format_exc())

# PDF генератор (без промяна в логиката, само подобрена графика чрез create_optimized_pdf_figure_intermediate)
class NumberedDocTemplate(SimpleDocTemplate):
    def __init__(self, filename, start_page=1, **kwargs):
        self.start_page = start_page
        super().__init__(filename, **kwargs)
        
    def afterPage(self):
        self._pageNumber = self.start_page + self.page - 1
        super().afterPage()

def generate_pdf_report(layer_idx, results, D, sigma_r=None, sigma_final=None, manual_value=None, check_passed=None):
    # Кодът остава същият както преди, но с подобрена графика (използва create_optimized_pdf_figure_intermediate)
    # ... (пълният код от предишната версия – не го променям тук за краткост, но в реалния файл го запази)
    # В частта с графиката:
    #   pdf_fig = create_optimized_pdf_figure_intermediate()
    #   img_bytes = pio.to_image(pdf_fig, format="png", width=850, height=650, scale=2.5, engine="kaleido")

# Бутон за PDF
st.markdown("---")
st.subheader("Генериране на PDF отчет")

if st.button("📄 Генерирай PDF отчет", type="primary"):
    if layer_idx in st.session_state.layer_results:
        with st.spinner("Генериране на PDF отчет..."):
            results = st.session_state.layer_results[layer_idx]
            sigma_r = st.session_state.get("final_sigma", None)
            sigma_final = st.session_state.get("final_sigma_R", None)
            manual_value = st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', None)
            check_passed = st.session_state.check_results.get(f'check_result_{layer_idx}', None)
            
            pdf_buffer = generate_pdf_report(layer_idx, results, D, sigma_r, sigma_final, manual_value, check_passed)
            
            if pdf_buffer:
                st.success("✅ PDF отчетът е готов!")
                st.download_button("📥 Изтегли PDF отчет",
                                   pdf_buffer,
                                   file_name=f"Опън_в_междинен_пласт_Пласт_{layer_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                   mime="application/pdf")
            else:
                st.error("❌ Грешка при генериране на PDF.")
    else:
        st.warning("Моля, изчислете първо пласта.")

st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")
