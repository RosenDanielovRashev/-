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

def render_formula_to_image(formula_text, fontsize=26, dpi=150):
    """Рендва формула като изображение чрез matplotlib mathtext (за ReportLab)"""
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

# Глобална променлива за фигурата
fig = None

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
        # КОРИГИРАНА ФОРМУЛА ЗА Esr - вместо точки използваме \cdot
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

                # Цветова палитра за изолиниите
                colors_isolines = [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
                ]
                
                # Add isolines from original df (Ei/Ed) - СИНИ изолинии
                if 'Ei/Ed' in df_original.columns:
                    levels = sorted(df_original['Ei/Ed'].unique())
                    for i, level in enumerate(levels):
                        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                        
                        # Разпределяме етикетите равномерно по вертикала
                        y_positions = np.linspace(0.5, 2.5, len(levels))
                        target_y = y_positions[i]
                        
                        # Намираме най-близката точка до целевата y позиция
                        closest_idx = (df_level['y'] - target_y).abs().idxmin()
                        x_mid = df_level.loc[closest_idx, 'H/D']
                        y_mid = df_level.loc[closest_idx, 'y']
                        
                        fig.add_trace(go.Scatter(
                            x=df_level['H/D'], y=df_level['y'],
                            mode='lines', 
                            name=f'Ei/Ed = {round(level,2)}',
                            line=dict(color=colors_isolines[i % len(colors_isolines)], width=2),
                            showlegend=False
                        ))
                        # Добавяме етикет за изолинията
                        fig.add_trace(go.Scatter(
                            x=[x_mid], y=[y_mid],
                            mode='text',
                            text=[f'{round(level,2)}'],
                            textposition='middle right',
                            textfont=dict(size=14, color='black'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # Add isolines from new df (Esr/Ei) - ЧЕРВЕНИ изолинии
                if 'sr_Ei' in df_new.columns:
                    sr_Ei_levels = sorted(df_new['sr_Ei'].unique())
                    offset = len(levels) if 'levels' in locals() else 0
                    
                    for i, sr_Ei in enumerate(sr_Ei_levels):
                        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                        
                        # Избираме различна позиция за всяка линия
                        if len(df_level) > 10:
                            mid_idx = 2 * len(df_level) // 3  # 2/3 от пътя
                        else:
                            mid_idx = len(df_level) // 2
                            
                        x_mid = df_level['H/D'].iloc[mid_idx]
                        y_mid = df_level['y'].iloc[mid_idx]
                        
                        # Използваме различна цветова палитра за втория тип изолинии
                        color_idx = (offset + i) % len(colors_isolines)
                        # Използваме по-тъмни цветове за втория тип
                        color = colors_isolines[color_idx]
                        
                        fig.add_trace(go.Scatter(
                            x=df_level['H/D'], y=df_level['y'],
                            mode='lines', 
                            name=f'Esr/Ei = {round(sr_Ei,2)}',
                            line=dict(color=color, width=2, dash='dash'),
                            showlegend=False
                        ))
                        # Добавяме етикет за изолинията
                        fig.add_trace(go.Scatter(
                            x=[x_mid], y=[y_mid],
                            mode='text',
                            text=[f'{round(sr_Ei,2)}'],
                            textposition='middle left',
                            textfont=dict(size=14, color='black'),
                            showlegend=False,
                            hoverinfo='skip'
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
                            mode='lines', 
                            line=dict(color='blue', dash='dash', width=2),
                            name='Вертикална линия',
                            showlegend=True
                        ))

                        # Червена точка
                        fig.add_trace(go.Scatter(
                            x=[target_Hn_D], y=[y_at_ratio],
                            mode='markers', 
                            marker=dict(color='red', size=12, symbol='circle', line=dict(color='darkred', width=2)),
                            name='Точка на интерполация',
                            showlegend=True
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
                                        mode='markers', 
                                        marker=dict(color='orange', size=14, symbol='diamond', line=dict(color='darkorange', width=2)),
                                        name='Пресечна точка',
                                        showlegend=True
                                    ))
                                    # Хоризонтална линия между червената и оранжевата точка
                                    fig.add_trace(go.Scatter(
                                        x=[target_Hn_D, x_intercept],
                                        y=[y_at_ratio, y_at_ratio],
                                        mode='lines',
                                        line=dict(color='green', dash='dash', width=2),
                                        name='Хоризонтална линия',
                                        showlegend=True
                                    ))

                                    # Вертикална линия от оранжева точка до y=2.5
                                    fig.add_trace(go.Scatter(
                                        x=[x_intercept, x_intercept],
                                        y=[y_at_ratio, 2.5],
                                        mode='lines',
                                        line=dict(color='purple', dash='dash', width=2),
                                        name='Вертикална линия до σr',
                                        showlegend=True
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
                # Добавяне на два специални елемента само за легендата
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],  # Невидими точки
                    mode='lines',
                    name='Ei/Ed - плътна линия',
                    line=dict(color='black', width=2, dash='solid'),
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],  # Невидими точки
                    mode='lines',
                    name='Esr/Ei - пунктирана линия',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=True
                ))
                # Обновяване на оформлението с цветна легенда
                fig.update_layout(
                    title=dict(
                        text='Графика на изолинии',
                        font=dict(size=16, color='black')
                    ),
                    xaxis=dict(
                        title='H/D',
                        title_font=dict(size=12, color='black'),
                        tickfont=dict(size=10, color='black'),
                        linecolor='black',
                        gridcolor='lightgray',
                        mirror=True,
                        showgrid=True,
                        range=[0, 2.1]
                    ),
                    xaxis2=dict(
                        overlaying='x',
                        side='top',
                        range=[0, 2.1],
                        showgrid=False,
                        zeroline=False,
                        tickvals=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.50, 1.75, 2],
                        ticktext=['0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0'], 
                        title='σr',
                        title_font=dict(size=12, color='black'),
                        tickfont=dict(size=10, color='black')
                    ),
                    yaxis=dict(
                        title='y',
                        title_font=dict(size=12, color='black'),
                        tickfont=dict(size=10, color='black'),
                        linecolor='black',
                        gridcolor='lightgray',
                        mirror=True,
                        showgrid=True,
                        range=[0, 2.7]
                    ),
                    legend=dict(
                        title=dict(
                            text='Легенда:',
                            font=dict(size=10, color='black')
                        ),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='black',
                        borderwidth=1,
                        font=dict(size=8, color='black'),
                        x=1.02,
                        y=1.0,
                        xanchor='left',
                        yanchor='top',
                        traceorder='normal',
                        itemsizing='constant',
                        orientation='v'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=800,
                    height=500,
                    margin=dict(l=50, r=150, t=50, b=50),
                    autosize=True
                )

                st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

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
        import traceback
        st.error(traceback.format_exc())

# КОРИГИРАН NumberedDocTemplate клас
class NumberedDocTemplate(SimpleDocTemplate):
    def __init__(self, filename, start_page=1, **kwargs):
        self.start_page = start_page
        super().__init__(filename, **kwargs)
        
    def afterPage(self):
        """Override to add page numbers with offset"""
        self._pageNumber = self.start_page + self.page - 1
        super().afterPage()

def generate_pdf_report(layer_idx, results, D, sigma_r=None, sigma_final=None, manual_value=None, check_passed=None):
    try:
        buffer = io.BytesIO()
        
        # UI за начален номер на страница - ДОБАВЕНО ТУК
        start_page_number = st.session_state.get("pdf_start_page", 1)
        
        doc = NumberedDocTemplate(
            buffer,
            start_page=start_page_number,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm
        )
        story = []
        styles = getSampleStyleSheet()

        # Зареждане на шрифт
        try:
            pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
            font_name = 'DejaVuSans-Bold'
        except:
            font_name = 'Helvetica-Bold'

        # ЗАГЛАВИЕ
        title_style = ParagraphStyle(
            'CustomTitle',
            fontSize=20,
            spaceAfter=5,
            alignment=1,
            textColor=colors.HexColor('#006064'),
            fontName=font_name,
            leading=20,
        )
        
        story.append(Paragraph("ОПЪН В МЕЖДИНЕН ПЛАСТ", title_style))
        story.append(Spacer(1, 16.5))

                # ДОБАВЕТЕ ТОВА: СВОБОДЕН ТЕКСТ ОТ ПОТРЕБИТЕЛЯ
        # Проверка дали има текст и ако има - го добавете
        free_text_content = st.session_state.get('pdf_comments', '')
        if free_text_content and free_text_content.strip():
            # Стил за коментарите
            comment_style = ParagraphStyle(
                'CommentStyle',
                parent=styles['Normal'],
                fontName='DejaVuSans',
                fontSize=10,
                textColor=colors.HexColor('#5D4037'),
                alignment=0,  # подравняване отляво
                spaceBefore=8,
                spaceAfter=12,
                leftIndent=10,
                rightIndent=10,
                borderPadding=5,
                borderWidth=1,
                borderColor=colors.HexColor('#BDBDBD'),
                backColor=colors.HexColor('#FFF3E0')
            )
            
            # Добавяне на коментара с рамка
            story.append(Paragraph("", ParagraphStyle(
                'CommentTitle',
                fontName=font_name,
                fontSize=11,
                textColor=colors.HexColor('#2C5530'),
                spaceBefore=15,
                spaceAfter=5,
                alignment=0
            )))
            
            # Разделяне на текста на редове за по-добро форматиране
            lines = free_text_content.strip().split('\n')
            for line in lines:
                if line.strip():  # Добавя само непразни редове
                    story.append(Paragraph(line.strip(), comment_style))
            
            story.append(Spacer(1, 8))
        
        story.append(Spacer(1, 16.5))

        # ИНФОРМАЦИЯ ЗА ПАРАМЕТРИ
        axle_load = st.session_state.get("axle_load", 100)
        table_data = [
            ["ПАРАМЕТЪР", "СТОЙНОСТ", "ЕДИНИЦА"],
            ["Диаметър D", f"{D:.2f}", "cm"],
            ["Брой пластове", f"{len(h_values)}", ""],
            ["Осова тежест", f"{axle_load}", "kN"],
            ["Пласт за проверка", f"{layer_idx+1}", ""],
        ]

        # Добавяне на данни за всеки пласт
        for i in range(len(h_values)):
            table_data.append([f"Пласт {i+1} - Ei", f"{E_values[i]:.2f}", "MPa"])
            table_data.append([f"Пласт {i+1} - hi", f"{h_values[i]:.2f}", "cm"])
            table_data.append([f"Пласт {i+1} - Edi", f"{Ed_values[i]:.2f}", "MPa"])

        info_table = Table(table_data, colWidths=[66*mm, 55*mm, 33*mm], hAlign='LEFT')
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A7C59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 9.9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5.5),
            ('TOPPADDING', (0, 0), (-1, 0), 5.5),
            
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
            ('FONTSIZE', (0, 1), (-1, -1), 8.8),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3.3),
            ('TOPPADDING', (0, 1), (-1, -1), 3.3),
            
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 15))

        # 2. ФОРМУЛИ ЗА ИЗЧИСЛЕНИЕ
        formulas_title_style = ParagraphStyle(
            'FormulasTitle',
            fontName=font_name,
            fontSize=14.08,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=11,
            alignment=0
        )
        story.append(Paragraph("2. Формули за изчисление", formulas_title_style))

        # Основни формули в две колони
        formulas = [
            r"H_{n-1} = \sum_{i=1}^{n-1} h_i",
            r"H_n = \sum_{i=1}^n h_i", 
            r"E_{sr} = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}",
            r"\sigma_R = 1.15 p \sigma_R^{nom}"
        ]

        # Създаваме таблица с две колони за формулите
        formula_table_data = []
        for i in range(0, len(formulas), 2):
            row = []
            # Първа колона
            if i < len(formulas):
                try:
                    img_buf1 = render_formula_to_image(formulas[i], fontsize=23.76, dpi=150)
                    row.append(RLImage(img_buf1, width=99*mm, height=19.8*mm))
                except:
                    row.append(Paragraph(formulas[i].replace('_', '').replace('^', ''), formulas_title_style))
            else:
                row.append('')
            
            # Втора колона
            if i + 1 < len(formulas):
                try:
                    img_buf2 = render_formula_to_image(formulas[i + 1], fontsize=23.76, dpi=150)
                    row.append(RLImage(img_buf2, width=99*mm, height=19.8*mm))
                except:
                    row.append(Paragraph(formulas[i + 1].replace('_', '').replace('^', ''), formulas_title_style))
            else:
                row.append('')
            
            formula_table_data.append(row)

        formula_table = Table(formula_table_data, colWidths=[105.6*mm, 105.6*mm])
        formula_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8.8),
            ('TOPPADDING', (0, 0), (-1, -1), 8.8),
        ]))
        
        story.append(formula_table)
        story.append(Spacer(1, 15))

        # 3. ИЗЧИСЛЕНИЯ
        calculations_title_style = ParagraphStyle(
            'CalculationsTitle',
            fontName=font_name,
            fontSize=14.08,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=11,
            alignment=0
        )
        story.append(Paragraph(f"3. Изчисления за пласт {layer_idx+1}", calculations_title_style))

        # Изчисление на H_{n-1}
        calculation_formulas = []
        
        if layer_idx > 0:
            h_terms_n1 = " + ".join([f"{h_values[i]:.0f}" for i in range(layer_idx)])
            calculation_formulas.append(fr"H_{{{layer_idx}}} = {h_terms_n1} = {results['H_n_1_r']:.0f} \ \mathrm{{cm}}")
        else:
            calculation_formulas.append(fr"H_{{{layer_idx}}} = 0 \ \mathrm{{cm}}")
        
        # H_n
        h_terms_n = " + ".join([f"{h_values[i]:.0f}" for i in range(results['n_for_calc'])])
        calculation_formulas.append(fr"H_{{{results['n_for_calc']}}} = {h_terms_n} = {results['H_n_r']:.0f} \ \mathrm{{cm}}")
        
        # Esr
        if layer_idx > 0:
            numerator = " + ".join([f"{E_values[i]:.0f} \cdot {h_values[i]:.0f}" for i in range(layer_idx)])
            denominator = " + ".join([f"{h_values[i]:.0f}" for i in range(layer_idx)])
            calculation_formulas.append(fr"E_{{sr}} = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']:.0f} \ \mathrm{{MPa}}")
        else:
            calculation_formulas.append(r"E_{sr} = 0 \ \mathrm{MPa}")
        
        # Други изчисления
        calculation_formulas.append(fr"\frac{{H_{{{results['n_for_calc']}}}}}{{D}} = \frac{{{results['H_n_r']:.0f}}}{{{D:.0f}}} = {results['ratio_r']:.3f}")
        calculation_formulas.append(fr"E_{{{layer_idx+1}}} = {results['En_r']:.0f} \ \mathrm{{MPa}}")
        calculation_formulas.append(fr"\frac{{E_{{sr}}}}{{E_{{{layer_idx+1}}}}} = \frac{{{results['Esr_r']:.0f}}}{{{results['En_r']:.0f}}} = {results['Esr_over_En_r']:.3f}")
        calculation_formulas.append(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']:.0f}}}{{{results['Ed_r']:.0f}}} = {results['En_over_Ed_r']:.3f}")
        
        # Информация за осов товар и σR
        if axle_load == 100:
            p = 0.620
        elif axle_load == 115:
            p = 0.633
        else:
            p = 0.0
            
        if sigma_r is not None:
            calculation_formulas.append(fr"\sigma_R^{{nom}} = {sigma_r:.3f} \ \mathrm{{MPa}}")
            
        if p and sigma_r is not None and sigma_final is not None:
            calculation_formulas.append(fr"p = {p:.3f} \ \mathrm{{({axle_load} \ kN)}}")
            calculation_formulas.append(fr"\sigma_R = 1.15 \times {p:.3f} \times {sigma_r:.3f} = {sigma_final:.3f} \ \mathrm{{MPa}}")

        # Изчисления в две колони
        calc_table_data = []
        for i in range(0, len(calculation_formulas), 2):
            row = []
            # Първа колона
            if i < len(calculation_formulas):
                try:
                    img_buf1 = render_formula_to_image(calculation_formulas[i], fontsize=21.12, dpi=150)
                    row.append(RLImage(img_buf1, width=99*mm, height=18.48*mm))
                except:
                    simple_text = calculation_formulas[i].replace('{', '').replace('}', '').replace('\\', '')
                    row.append(Paragraph(simple_text, calculations_title_style))
            else:
                row.append('')
            
            # Втора колона
            if i + 1 < len(calculation_formulas):
                try:
                    img_buf2 = render_formula_to_image(calculation_formulas[i + 1], fontsize=21.12, dpi=150)
                    row.append(RLImage(img_buf2, width=99*mm, height=18.48*mm))
                except:
                    simple_text = calculation_formulas[i + 1].replace('{', '').replace('}', '').replace('\\', '')
                    row.append(Paragraph(simple_text, calculations_title_style))
            else:
                row.append('')
            
            calc_table_data.append(row)

        calc_table = Table(calc_table_data, colWidths=[105.6*mm, 105.6*mm])
        calc_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6.6),
            ('TOPPADDING', (0, 0), (-1, -1), 6.6),
        ]))
        
        story.append(calc_table)
        story.append(Spacer(1, 15))

        # НОВ ЛИСТ ЗА ГРАФИКАТА
        

        # ГРАФИКА НА НОМОГРАМАТА
        graph_title_style = ParagraphStyle(
            'GraphTitle',
            fontName=font_name,
            fontSize=17.6,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=16.5,
            alignment=1
        )
        story.append(Paragraph("ГРАФИКА НА НОМОГРАМАТА", graph_title_style))
        
        try:
            if fig is not None:
                # Обновяване на оформлението за PDF с по-добра легенда
                fig_pdf = go.Figure(fig)
                
                # Настройки за PDF
                fig_pdf.update_layout(
                    title=dict(
                        text='',
                        font=dict(size=18, color='black', family="Arial")
                    ),
                    xaxis=dict(
                        title='H/D',
                        title_font=dict(size=14, color='black'),
                        tickfont=dict(size=12, color='black'),
                        linecolor='black',
                        gridcolor='lightgray',
                        mirror=True,
                        showgrid=True,
                        range=[0, 2.1]
                    ),
                    xaxis2=dict(
                        overlaying='x',
                        side='top',
                        range=[0, 2.1],
                        showgrid=False,
                        zeroline=False,
                        tickvals=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.50, 1.75, 2],
                        ticktext=['0', '0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0'],
                        title='σr',
                        title_font=dict(size=14, color='black'),
                        tickfont=dict(size=12, color='black')
                    ),
                    yaxis=dict(
                        title='y',
                        title_font=dict(size=14, color='black'),
                        tickfont=dict(size=12, color='black'),
                        linecolor='black',
                        gridcolor='lightgray',
                        mirror=True,
                        showgrid=True,
                        range=[0, 2.7]
                    ),
                    legend=dict(
                        title=dict(
                            text='Легенда:',
                            font=dict(size=14, color='black')
                        ),
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='black',
                        borderwidth=1,
                        font=dict(size=12, color='black'),
                        x=0.5,
                        y=-0.3,
                        xanchor='center',  # ЗАКРЕПЕНЕ В ЦЕНТЪРА
                        yanchor='top',     # ЗАКРЕПЕНЕ ОТГОРЕ
                        traceorder='normal',
                        itemsizing='constant',
                        orientation='h'  # ПРОМЕНЕТЕ НА ХОРИЗОНТАЛНА ОРИЕНТАЦИЯ
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=800,
                    height=600,
                    margin=dict(l=50, r=50, t=50, b=150)
                )
                # Експортиране на фигурата с висока резолюция
                img_bytes = pio.to_image(fig_pdf, format="png", width=1200, height=800, scale=4, engine="kaleido")
                
                pil_img = PILImage.open(BytesIO(img_bytes))
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG", dpi=(300, 300))
                img_buffer.seek(0)
                
                story.append(RLImage(img_buffer, width=170 * mm, height=130 * mm))
                story.append(Spacer(1, 15))
                
        except Exception as e:
            error_style = ParagraphStyle(
                'ErrorStyle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=5.5,
                fontName=font_name,
                textColor=colors.HexColor('#d32f2f'),
                alignment=1
            )
            story.append(Paragraph(f"Грешка при генериране на графика: {e}", error_style))

        # ДОПУСТИМИ НАПРЕЖЕНИЯ
        img_path = "Допустими опънни напрежения.png"
        if os.path.exists(img_path):
            allowable_title_style = ParagraphStyle(
                'AllowableTitle',
                fontName=font_name,
                fontSize=15.4,
                textColor=colors.HexColor('#2C5530'),
                spaceAfter=11,
                alignment=1
            )
            story.append(Spacer(1, 22))
            story.append(PageBreak())
            story.append(Paragraph("ДОПУСТИМИ ОПЪННИ НАПРЕЖЕНИЯ", allowable_title_style))
            
            try:
                pil_img = PILImage.open(img_path)
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                story.append(RLImage(img_buffer, width=170 * mm, height=130 * mm))
                story.append(Spacer(1, 15))
            except Exception as e:
                error_style = ParagraphStyle(
                    'ErrorStyle',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=5.5,
                    fontName=font_name,
                    textColor=colors.HexColor('#d32f2f'),
                    alignment=1
                )
                story.append(Paragraph("Грешка при зареждане на изображение", error_style))

        # РЕЗУЛТАТИ И ПРОВЕРКА
    
        results_title_style = ParagraphStyle(
            'ResultsTitle',
            fontName=font_name,
            fontSize=17.6,
            textColor=colors.HexColor('#006064'),
            spaceAfter=16.5,
            alignment=1
        )
        story.append(Paragraph("РЕЗУЛТАТИ И ПРОВЕРКА", results_title_style))

        if sigma_final is not None and manual_value is not None:
            check_passed = sigma_final <= manual_value

            # Таблица с резултати
            results_data = [
                ["ПАРАМЕТЪР", "СТОЙНОСТ"],
                ["Изчислено σR", f"{sigma_final:.3f} MPa"],
                ["Допустимо σR", f"{manual_value:.2f} MPa"]
            ]

            results_table = Table(results_data, colWidths=[88*mm, 66*mm], hAlign='CENTER')
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A7C59')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6.6),
                ('TOPPADDING', (0, 0), (-1, 0), 6.6),
                
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 1), (-1, -1), 9.9),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4.4),
                ('TOPPADDING', (0, 1), (-1, -1), 4.4),
                
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
            ]))

            story.append(results_table)
            story.append(Spacer(1, 16.5))

            # Съобщение за проверка
            if check_passed:
                status_style = ParagraphStyle(
                    'StatusOK',
                    fontName=font_name,
                    fontSize=13.2,
                    textColor=colors.HexColor('#2e7d32'),
                    spaceAfter=13.2,
                    alignment=1,
                    backColor=colors.HexColor('#e8f5e9')
                )
                story.append(Paragraph("ПРОВЕРКАТА Е УДОВЛЕТВОРЕНА", status_style))
                subtitle_style = ParagraphStyle(
                    'SubtitleStyle',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=5.5,
                    fontName=font_name,
                    textColor=colors.HexColor('#5D4037'),
                    alignment=1
                )
                story.append(Paragraph("Изчисленото σR е по-малко или равно на допустимото σR", subtitle_style))
            else:
                status_style = ParagraphStyle(
                    'StatusFail',
                    fontName=font_name,
                    fontSize=13.2,
                    textColor=colors.HexColor('#c62828'),
                    spaceAfter=13.2,
                    alignment=1,
                    backColor=colors.HexColor('#ffebee')
                )
                story.append(Paragraph("ПРОВЕРКАТА НЕ Е УДОВЛЕТВОРЕНА", status_style))
                subtitle_style = ParagraphStyle(
                    'SubtitleStyle',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=5.5,
                    fontName=font_name,
                    textColor=colors.HexColor('#5D4037'),
                    alignment=1
                )
                story.append(Paragraph("Изчисленото σR е по-голямо от допустимото σR", subtitle_style))

        # ДАТА И ПОДПИС
        story.append(Spacer(1, 22))
        current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
        date_style = ParagraphStyle(
            'DateStyle',
            fontName=font_name,
            fontSize=9.9,
            alignment=2,
            textColor=colors.HexColor('#666666')
        )
        story.append(Paragraph(f"Генерирано на: {current_date}", date_style))
        
        # Добавяне на номера на страниците
        def add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont('DejaVuSans', 8)
            page_num = doc.start_page + canvas.getPageNumber() - 1
            canvas.drawString(190*mm, 15*mm, f"{page_num}")
            canvas.restoreState()
        
        # Финализиране на PDF
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        buffer.seek(0)
        
        return buffer

    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# ... [останалата част на кода остава същата до края] ...

# -------------------------------------------------
# ДОБАВЕНО: Секция за настройки на PDF преди бутона за генериране
# -------------------------------------------------
st.markdown("---")
st.subheader("Настройки за PDF отчет")

# Инициализиране на pdf_start_page в session_state, ако не съществува
if "pdf_start_page" not in st.session_state:
    st.session_state.pdf_start_page = 1

# Поле за въвеждане на начален номер на страница
pdf_start_page = st.number_input(
    "Начален номер на страница:",
    min_value=1,
    max_value=1000,
    value=st.session_state.pdf_start_page,  # Използваме стойността от session_state
    step=1,
    help="Задайте от кой номер да започва номерацията на страниците",
    key="pdf_start_page_input"  # Различен ключ от session_state ключа
)

# Актуализираме session_state при промяна
if pdf_start_page != st.session_state.pdf_start_page:
    st.session_state.pdf_start_page = pdf_start_page


# ДОБАВЕТЕ ТОВА НОВО ТЕКСТОВО ПОЛЕ:
st.markdown("### Бележки и коментари")
free_text = st.text_area(
    "Въведете допълнителни бележки или коментари:",
    value=st.session_state.get('pdf_comments', ''),
    height=150,
    help="Този текст ще се появи в PDF отчета под заглавието",
    key="pdf_comments_input"
)

# Запазване на текста в session state
if 'pdf_comments' not in st.session_state:
    st.session_state.pdf_comments = ''
if free_text != st.session_state.pdf_comments:
    st.session_state.pdf_comments = free_text





# -------------------------------------------------
# ПРОМЕНЕТЕ БУТОНА ЗА ГЕНЕРИРАНЕ НА PDF
# -------------------------------------------------
st.subheader("Генериране на PDF отчет")

if st.button("📄 Генерирай PDF отчет", type="primary"):
    if layer_idx in st.session_state.layer_results:
        with st.spinner("Генериране на PDF отчет..."):
            # Вземете необходимите данни за отчета
            results = st.session_state.layer_results[layer_idx]
            sigma_r = st.session_state.get("final_sigma", None)
            sigma_final = st.session_state.get("final_sigma_R", None)
            manual_value = st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', None)
            check_passed = st.session_state.check_results.get(f'check_result_{layer_idx}', None)
            
            # Генериране на PDF
            pdf_buffer = generate_pdf_report(
                layer_idx, results, D, sigma_r, sigma_final, manual_value, check_passed
            )
            
            if pdf_buffer:
                st.success("✅ PDF отчетът с модерно графично оформление е готов!")
                st.download_button(
                    "📥 Изтегли PDF отчет",
                    pdf_buffer,
                    file_name=f"Опън_в_междинен_пласт_Пласт_{layer_idx+1}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("❌ Неуспешно генериране на PDF. Моля, проверете грешките по-горе.")
    else:
        st.warning("Моля, изчислете първо пласта преди да генерирате отчет.")

# Линк към предишната страница
st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")
