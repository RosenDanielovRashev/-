import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Добавете тези нови импорти:
import os
import tempfile
import base64
from io import BytesIO
from datetime import datetime
from fpdf import FPDF  # Това е ключовият импорт, който липсва
from PIL import Image
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
import io
from PIL import Image as PILImage
from matplotlib import mathtext

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

st.title("Определяне на напреженията на срязване за несързани почви фиг9.6 maxH/D=1.5")

# Зареждане на данните за номограмата τb
@st.cache_data
def load_tau_b_data():
    Fi_data = pd.read_csv('Fi_3.csv')
    H_data = pd.read_csv('H_3.csv')
    
    Fi_data.columns = ['y', 'x', 'Fi']
    
    Fi_data['Fi'] = Fi_data['Fi'].astype(float)
    H_data['H'] = H_data['H'].astype(float)
    
    Fi_data = Fi_data.drop_duplicates(subset=['x', 'y', 'Fi'])
    
    # Подготовка на данните за Fi
    fi_aggregated_groups = {}
    fi_interpolators = {}
    fi_values_available = sorted(Fi_data['Fi'].unique())

    for fi in fi_values_available:
        group = Fi_data[Fi_data['Fi'] == fi].sort_values(by='x')
        fi_aggregated_groups[fi] = group
        
        x = group['x'].values
        y = group['y'].values
        
        if len(x) < 2:
            def constant_func(x_val, y_const=y[0]):
                return np.full_like(x_val, y_const)
            fi_interpolators[fi] = constant_func
        else:
            fi_interpolators[fi] = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Създаване на mapping между x и H
    unique_h = H_data[['x', 'H']].drop_duplicates()
    x_to_h = dict(zip(unique_h['x'], unique_h['H']))
    h_to_x = dict(zip(unique_h['H'], unique_h['x']))
    h_values_available = sorted(h_to_x.keys())
    
    return Fi_data, H_data, fi_aggregated_groups, fi_interpolators, fi_values_available, h_to_x, h_values_available, x_to_h

# Функция за изчисляване и визуализация на τb с билинейна интерполация
def plot_tau_b(fi_value, h_value):
    try:
        # Зареждане на данните
        Fi_data, H_data, fi_aggregated_groups, fi_interpolators, fi_values_available, h_to_x, h_values_available, x_to_h = load_tau_b_data()
        
        h_value = float(h_value)
        fi_value = float(fi_value)
        
        # Намиране на двата най-близки H (долна и горна граница)
        h_val_arr = np.array(h_values_available)
        idx_h = np.searchsorted(h_val_arr, h_value)
        if idx_h == 0:
            h_low = h_high = h_val_arr[0]
        elif idx_h == len(h_val_arr):
            h_low = h_high = h_val_arr[-1]
        else:
            h_low = h_val_arr[idx_h-1]
            h_high = h_val_arr[idx_h]
        
        # Намиране на двата най-близки φ (долна и горна граница)
        fi_val_arr = np.array(fi_values_available)
        idx_fi = np.searchsorted(fi_val_arr, fi_value)
        if idx_fi == 0:
            fi_low = fi_high = fi_val_arr[0]
        elif idx_fi == len(fi_val_arr):
            fi_low = fi_high = fi_val_arr[-1]
        else:
            fi_low = fi_val_arr[idx_fi-1]
            fi_high = fi_val_arr[idx_fi]
        
        # Изчисляване на тегла за интерполация
        t_h = (h_value - h_low) / (h_high - h_low) if h_high != h_low else 0.0
        t_fi = (fi_value - fi_low) / (fi_high - fi_low) if fi_high != fi_low else 0.0
        
        # Функция за получаване на y за дадени H и φ
        def get_y_for_h_fi(h_val, fi_val):
            x_h = h_to_x[h_val]
            if fi_val in fi_interpolators:
                return float(fi_interpolators[fi_val](x_h))
            else:
                closest_fi = min(fi_values_available, key=lambda x: abs(x - fi_val))
                return float(fi_interpolators[closest_fi](x_h))
        
        # Изчисляване на τb с билинейна интерполация
        y_low_low = get_y_for_h_fi(h_low, fi_low)
        y_low_high = get_y_for_h_fi(h_low, fi_high)
        y_high_low = get_y_for_h_fi(h_high, fi_low)
        y_high_high = get_y_for_h_fi(h_high, fi_high)
        
        y_low = y_low_low + t_fi * (y_low_high - y_low_low)
        y_high = y_high_low + t_fi * (y_high_high - y_high_low)
        y_tau = y_low + t_h * (y_high - y_low)
        
        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 7))
        
        x_min = min(Fi_data['x'].min(), min(h_to_x.values()))
        x_max = max(Fi_data['x'].max(), max(h_to_x.values()))
        y_min = min(Fi_data['y'].min(), H_data['y'].min()) - 0.001
        y_max = max(Fi_data['y'].max(), H_data['y'].max()) + 0.001
        
        # Рисуване на всички изолинии (светли)
        for fi_val in fi_values_available:
            group = fi_aggregated_groups[fi_val]
            if len(group) == 1:
                ax.plot([x_min, x_max], [group['y'].iloc[0]]*2, 
                        'b-', linewidth=0.5, alpha=0.3)
                ax.text(x_max, group['y'].iloc[0], f'φ={fi_val}', color='blue', 
                       va='center', ha='left', fontsize=9, alpha=0.7)
            else:
                x_smooth = np.linspace(group['x'].min(), group['x'].max(), 100)
                y_smooth = fi_interpolators[fi_val](x_smooth)
                ax.plot(x_smooth, y_smooth, 'b-', linewidth=0.5, alpha=0.3)
                ax.text(x_smooth[-1], y_smooth[-1], f'φ={fi_val}', color='blue',
                       va='center', ha='left', fontsize=9, alpha=0.7)

        for h_val in h_values_available:
            x_pos = h_to_x[h_val]
            y_min_h = H_data[H_data['H'] == h_val]['y'].min()
            y_max_h = H_data[H_data['H'] == h_val]['y'].max()
            ax.plot([x_pos]*2, [y_min_h, y_max_h], 'r-', linewidth=0.5, alpha=0.3)
        
        # Подчертаване на използваните изолинии (дебели линии)
        for fi_val in [fi_low, fi_high]:
            if fi_val in fi_aggregated_groups:
                group = fi_aggregated_groups[fi_val]
                if len(group) == 1:
                    ax.plot([x_min, x_max], [group['y'].iloc[0]]*2, 
                            'b-', linewidth=2, alpha=0.8)
                else:
                    x_smooth = np.linspace(group['x'].min(), group['x'].max(), 100)
                    y_smooth = fi_interpolators[fi_val](x_smooth)
                    ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, alpha=0.8)
        
        for h_val in [h_low, h_high]:
            if h_val in h_to_x:
                x_pos = h_to_x[h_val]
                y_min_h = H_data[H_data['H'] == h_val]['y'].min()
                y_max_h = H_data[H_data['H'] == h_val]['y'].max()
                ax.plot([x_pos]*2, [y_min_h, y_max_h], 'r-', linewidth=2, alpha=0.8)
        
        # КОРИГИРАНА ЧАСТ: Интерполация на x за h_value
        x_low = h_to_x[h_low]
        x_high = h_to_x[h_high]
        x_value = x_low + t_h * (x_high - x_low)  # Интерполирана x координата
        
        # Маркиране на пресечната точка с интерполирана x координата
        ax.plot(x_value, y_tau, 'ko', markersize=8, 
                label=f'τb = {y_tau:.6f}\nH: {h_low}→{h_value}→{h_high}\nφ: {fi_low}→{fi_value}→{fi_high}')
        
        # Настройки на графиката
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # КОРИГИРАНА ЧАСТ: Подготовка на тикчетата (включвайки h_value)
        h_ticks = sorted(set([h_low, h_value, h_high] + h_values_available))
        x_positions = []
        h_tick_labels = []
        
        for h in h_ticks:
            if h in h_to_x:
                x_positions.append(h_to_x[h])
                h_tick_labels.append(f"{h:.1f}")
            elif h == h_value:
                # Добавяме текущата H стойност като тик
                x_positions.append(x_value)
                h_tick_labels.append(f"{h_value:.1f}")
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(h_tick_labels)
        
        ax.set_xlabel('H', fontsize=8)
        ax.set_ylabel('τb', fontsize=8)
        ax.set_title(f'Номограма за активно напрежение на срязване (τb)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower left')
        
        return fig, y_tau
        
    except Exception as e:
        st.error(f"Грешка при изчисляване на τb: {str(e)}")
        return None, None

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Инициализиране на променливите
h_values = []
Ei_values = []
Ed_values = []
Fi_values = []
n = 3  # Стойност по подразбиране

# Проверка за данни в session_state
session_data_available = all(key in st.session_state for key in ['fig9_6_h', 'fig9_6_fi']) and \
                         'layers_data' in st.session_state and \
                         len(st.session_state.layers_data) > 0

# Автоматично зареждане на данни ако са налични
if session_data_available:
    try:
        n = len(st.session_state.fig9_6_h)
        h_values = [round(float(h), 2) for h in st.session_state.fig9_6_h]
        Ed_values = [round(layer["Ed"]) for layer in st.session_state.layers_data]
        Ei_values = [round(layer["Ei"]) for layer in st.session_state.layers_data]
        Fi_values = st.session_state.fig9_6_fi[:n]  # Взимаме само необходимия брой
        
        D_options = [32.04, 34.0, 33.0]
        
        if 'fig9_6_D' in st.session_state:
            current_d = st.session_state.fig9_6_D
            if current_d not in D_options:
                D_options.insert(0, current_d)
        else:
            current_d = D_options[0]

        selected_d = st.selectbox("Избери D", options=D_options, index=D_options.index(current_d))
        st.session_state.fig9_6_D = selected_d
        D = selected_d
        
        # Добавяне на избор за осов товар
        axle_load_options = [100, 115]
        if 'axle_load' in st.session_state:
            current_axle = st.session_state.axle_load
        else:
            current_axle = 100
        axle_load = st.selectbox("Осова товарност (kN)", options=axle_load_options, index=axle_load_options.index(current_axle))
        st.session_state.axle_load = axle_load
        
        st.markdown("### Автоматично заредени данни за пластовете")
        cols = st.columns(4)  # Променено от 3 на 4 колони
        
        h_values_edited = []
        Ei_values_edited = []
        Ed_values_edited = []
        Fi_values_edited = []
        
        for i in range(n):
            with cols[0]:
                default_h = float(h_values[i]) if i < len(h_values) else 4.0
                h_val = st.number_input(f"h{to_subscript(i+1)}", value=default_h, step=0.1, key=f"auto_h_{i}")
                h_values_edited.append(round(h_val, 2))
            with cols[1]:
                default_ei = int(Ei_values[i]) if i < len(Ei_values) else 1000
                ei_val = st.number_input(f"Ei{to_subscript(i+1)}", value=default_ei, step=1, key=f"auto_Ei_{i}")
                Ei_values_edited.append(ei_val)
            with cols[2]:
                default_ed = int(Ed_values[i]) if i < len(Ed_values) else 1000
                ed_val = st.number_input(f"Ed{to_subscript(i+1)}", value=default_ed, step=1, key=f"auto_Ed_{i}")
                Ed_values_edited.append(ed_val)
            with cols[3]:
                default_fi = Fi_values[i] if i < len(Fi_values) else 15
                fi_val = st.number_input(f"Fi{to_subscript(i+1)}", value=default_fi, step=1, key=f"auto_Fi_{i}")
                Fi_values_edited.append(fi_val)
        
        h_values = h_values_edited
        Ei_values = Ei_values_edited
        Ed_values = Ed_values_edited
        Fi_values = Fi_values_edited
        st.session_state.fig9_6_fi = Fi_values  # Запазване във session state

    except Exception as e:
        st.error(f"Грешка при зареждане на данните: {str(e)}")
        session_data_available = False

# Ръчно въвеждане ако няма данни в сесията или има грешка
if not session_data_available:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
    D_options = [32.04, 34.0, 33.0]
    selected_d = st.selectbox("Избери D", options=D_options, index=0)
    st.session_state.fig9_6_D = selected_d
    D = selected_d
    
    # Добавяне на избор за осов товар
    axle_load_options = [100, 115]
    if 'axle_load' in st.session_state:
        current_axle = st.session_state.axle_load
    else:
        current_axle = 100
    axle_load = st.selectbox("Осова товарност (kN)", options=axle_load_options, index=axle_load_options.index(current_axle))
    st.session_state.axle_load = axle_load
    
    st.markdown("### Въведи стойности за всеки пласт")
    h_values = []
    Ei_values = []
    Ed_values = []
    Fi_values = []
    cols = st.columns(4)  # Променено от 3 на 4 колони
    for i in range(n):
        with cols[0]:
            h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
            h_values.append(round(h, 2))
        with cols[1]:
            Ei_val = st.number_input(f"Ei{to_subscript(i+1)}", value=1000, step=1, key=f"Ei_{i}")
            Ei_values.append(Ei_val)
        with cols[2]:
            Ed_val = st.number_input(f"Ed{to_subscript(i+1)}", value=1000, step=1, key=f"Ed_{i}")
            Ed_values.append(Ed_val)
        with cols[3]:
            Fi_val = st.number_input(f"Fi{to_subscript(i+1)}", value=15, step=1, key=f"Fi_{i}")
            Fi_values.append(Fi_val)
    st.session_state.fig9_6_fi = Fi_values  # Запазване във session state

# Избор на пласт за проверка
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Задаване на Eo = Ed на избрания пласт (с закръгляне)
Eo = round(Ed_values[layer_idx])
st.markdown(f"**Eo = Ed{to_subscript(layer_idx+1)} = {Eo}**")

# Изчисляване на H и Esr за избрания пласт (с закръгляне)
h_array = np.array([round(h, 2) for h in h_values[:layer_idx+1]])
Ei_rounded = [round(val) for val in Ei_values[:layer_idx+1]]  # Закръглени Ei стойности
E_array = np.array(Ei_rounded)

H = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / H if H != 0 else 0
Esr = round(Esr)  # Закръгляне до цяло число

# Формули и резултати
st.latex(r"H = \sum_{i=1}^n h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx+1)])
st.latex(r"H = " + h_terms)
st.write(f"H = {H:.2f}")

st.latex(r"Esr = \frac{\sum_{i=1}^n (E_i \cdot h_i)}{\sum_{i=1}^n h_i}")
numerator = " + ".join([f"{Ei_rounded[i]} \cdot {h_values[i]}" for i in range(layer_idx+1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(layer_idx+1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum:.2f}}}{{{H:.2f}}} = {Esr}"
st.latex(formula_with_values)

ratio = H / D if D != 0 else 0
st.latex(r"\frac{H}{D} = \frac{" + f"{H:.2f}" + "}{" + f"{D}" + "} = " + f"{ratio:.3f}")

st.latex(r"\frac{Esr}{E_o} = \frac{" + f"{Esr}" + "}{" + f"{Eo}" + "} = " + f"{Esr / Eo:.3f}")
Esr_over_Eo = Esr / Eo if Eo != 0 else 0

# Зареждане на данни
df_fi = pd.read_csv("fi_9.6.csv")
df_esr_eo = pd.read_csv("Esr_Eo_9.6.csv")

df_fi.rename(columns={df_fi.columns[2]: 'fi'}, inplace=True)
df_esr_eo.rename(columns={df_esr_eo.columns[2]: 'Esr_Eo'}, inplace=True)

fig = go.Figure()

# Изолинии fi
unique_fi = sorted(df_fi['fi'].unique())
for fi_val in unique_fi:
    df_level = df_fi[df_fi['fi'] == fi_val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'ϕ = {fi_val}',
        line=dict(width=2)
    ))

# Изолинии Esr/Eo
unique_esr_eo = sorted(df_esr_eo['Esr_Eo'].unique())
for val in unique_esr_eo:
    df_level = df_esr_eo[df_esr_eo['Esr_Eo'] == val].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Esr/Eo = {val}',
        line=dict(width=2)
    ))

# Функция за интерполация на точка по H/D
def get_point_on_curve(df, x_target):
    x_vals = df['H/D'].values
    y_vals = df['y'].values
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x_target <= x_vals[i + 1]:
            x1, y1 = x_vals[i], y_vals[i]
            x2, y2 = x_vals[i + 1], y_vals[i + 1]
            t = (x_target - x1) / (x2 - x1)
            y_interp = y1 + t * (y2 - y1)
            return np.array([x_target, y_interp])
    return None

# Интерполация за червената точка между Esr/Eo изолинии
unique_esr_eo_sorted = sorted(df_esr_eo['Esr_Eo'].unique())
lower_vals = [v for v in unique_esr_eo_sorted if v <= Esr_over_Eo]
upper_vals = [v for v in unique_esr_eo_sorted if v >= Esr_over_Eo]

if lower_vals and upper_vals:
    v1 = lower_vals[-1]
    v2 = upper_vals[0]
    
    if v1 == v2:
        df_interp = df_esr_eo[df_esr_eo['Esr_Eo'] == v1]
        point_on_esr_eo = get_point_on_curve(df_interp, ratio)
    else:
        df1 = df_esr_eo[df_esr_eo['Esr_Eo'] == v1].sort_values(by='H/D')
        df2 = df_esr_eo[df_esr_eo['Esr_Eo'] == v2].sort_values(by='H/D')
        p1 = get_point_on_curve(df1, ratio)
        p2 = get_point_on_curve(df2, ratio)

        if p1 is not None and p2 is not None:
            t = (Esr_over_Eo - v1) / (v2 - v1)
            y_interp = p1[1] + t * (p2[1] - p1[1])
            point_on_esr_eo = np.array([ratio, y_interp])
        else:
            point_on_esr_eo = None
else:
    point_on_esr_eo = None

# Функция за интерполация по y за дадена fi изолиния
def interp_x_at_y(df_curve, y_target):
    x_arr = df_curve['H/D'].values
    y_arr = df_curve['y'].values
    for k in range(len(y_arr) - 1):
        y1, y2 = y_arr[k], y_arr[k + 1]
        if (y1 - y_target) * (y2 - y_target) <= 0:
            x1, x2 = x_arr[k], x_arr[k + 1]
            if y2 == y1:
                return x1
            t = (y_target - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None
    

# Интерполация на x (H/D) между fi изолинии
def interp_x_for_fi_interp(df, fi_target, y_target):
    fi_values = sorted(df['fi'].unique())
    lower_fi = [v for v in fi_values if v <= fi_target]
    upper_fi = [v for v in fi_values if v >= fi_target]

    if not lower_fi or not upper_fi:
        return None

    fi1 = lower_fi[-1]
    fi2 = upper_fi[0]

    if fi1 == fi2:
        df1 = df[df['fi'] == fi1].sort_values(by='y')
        return interp_x_at_y(df1, y_target)
    else:
        df1 = df[df['fi'] == fi1].sort_values(by='y')
        df2 = df[df['fi'] == fi2].sort_values(by='y')
        x1 = interp_x_at_y(df1, y_target)
        x2 = interp_x_at_y(df2, y_target)
        if x1 is not None and x2 is not None:
            t = (fi_target - fi1) / (fi2 - fi1)
            return x1 + t * (x2 - x1)
    return None

# Добавяне на червена точка и вертикална червена линия
if point_on_esr_eo is not None:
    fig.add_trace(go.Scatter(
        x=[point_on_esr_eo[0]],
        y=[point_on_esr_eo[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Червена точка (Esr/Eo)'
    ))
    fig.add_trace(go.Scatter(
        x=[ratio, ratio],
        y=[0, point_on_esr_eo[1]],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Вертикална линия H/D → Esr/Eo'
    ))

    # Добавяне на оранжева точка чрез интерполация по fi
    y_red = point_on_esr_eo[1]
    x_orange = interp_x_for_fi_interp(df_fi, Fi_values[layer_idx], y_red)

    if x_orange is not None:
        fig.add_trace(go.Scatter(
            x=[x_orange],
            y=[y_red],
            mode='markers',
            marker=dict(color='orange', size=10),
            name='Оранжева точка'
        ))
        fig.add_trace(go.Scatter(
            x=[point_on_esr_eo[0], x_orange],
            y=[y_red, y_red],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Хоризонтална линия'
        ))
        fig.add_trace(go.Scatter(
            x=[x_orange, x_orange],
            y=[y_red, 1.05],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Вертикална линия до y=1.05'
        ))

# Настройки на графиката
fig.update_layout(
    title="Графика на изолинии и точки",
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Легенда",
    width=900,
    height=600
)

# Определяне на фиксиран мащаб
xaxis_min = 0
xaxis_max = 1.5

# Добавяне на невидим trace за втората ос
fig.add_trace(go.Scatter(
    x=[xaxis_min, xaxis_max],
    y=[None, None],
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip',
    xaxis='x2'
))

# Финални настройки на осите с padding за показване на последните стойности
fig.update_layout(
    title='Графика на изолинии',
    xaxis=dict(
        title='H/D',
        showgrid=True,
        zeroline=False,
        range=[xaxis_min, xaxis_max * 1.005],  # Padding за последна стойност
        tickvals=np.linspace(xaxis_min, xaxis_max, 11),
        gridcolor='lightgray',
        gridwidth=1
    ),
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[xaxis_min, xaxis_max * 1.005],  # Padding за последна стойност
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickvals=np.linspace(xaxis_min, xaxis_max, 11),
        ticktext=[f"{(0.150 * (x - xaxis_min) / (xaxis_max - xaxis_min)):.3f}" 
                 for x in np.linspace(xaxis_min, xaxis_max, 11)],
        ticklabeloverflow="allow",
        title='Ꚍμ/p',
        fixedrange=True
    ),
    yaxis=dict(
        title='y',
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1
    ),
    showlegend=False,
    height=600,
    width=900,
    margin=dict(l=50, r=50, t=50, b=50)  # Допълнителни margin за labels
)



st.plotly_chart(fig, use_container_width=True)

# Изчисление на σr от x на оранжевата точка (ако съществува)
if 'x_orange' in locals() and x_orange is not None:
    sigma_r = round(x_orange / 10, 3)
    x_val = round(x_orange, 3)
    
    # Определяне на p според осовия товар
    p_value = 0.620 if axle_load == 100 else 0.633
    tau_mu = sigma_r * p_value  # Ꚍμ = (Ꚍμ/p) * p
    
    # Показване на стойността на p преди формулата
    st.markdown(f"**p = {p_value} MPa (за осов товар {axle_load} kN)**")
    st.markdown(f"**Ꚍμ/p = {sigma_r}**")
    st.markdown(f"**Ꚍμ = (Ꚍμ/p) × p = {sigma_r} × {p_value} = {tau_mu:.6f} MPa**")
else:
    # Показване на стойността на p преди формулата
    p_value = 0.620 if axle_load == 100 else 0.633
    st.markdown(f"**p = {p_value} MPa (за осов товар {axle_load} kN)**")
    st.markdown("**Ꚍμ/p = -** (Няма изчислена стойност)")
    # Задаваме стойности по подразбиране, за да избегнем грешки по-нататък
    sigma_r = 0.0
    tau_mu = 0.0

# Изчисляване и визуализация на τb за текущия пласт
st.divider()
st.subheader("Изчисление на активно напрежение на срязване τb")

tau_b_fig, tau_b = plot_tau_b(Fi_values[layer_idx], H)
if tau_b_fig is not None and tau_b is not None:
    st.markdown(f"**За пласт {layer_idx+1}:**")
    st.markdown(f"- H = {H:.2f}")
    st.markdown(f"- ϕ = {Fi_values[layer_idx]}")
    st.markdown(f"**τb = {tau_b:.6f}**")
    st.pyplot(tau_b_fig)
else:
    st.error("Неуспешно изчисление на τb")

st.image("9.8 Таблица.png", width=600)

# Инициализиране на session_state за K стойностите и C, ако не съществуват
if 'K_values' not in st.session_state:
    st.session_state.K_values = {}

# Добавяне на полета за въвеждане на K стойностите и C
st.markdown("### Въведете коефициентите за изчисление")
cols = st.columns(4)  # Сега имаме 4 колони

# Вземане или инициализиране на стойностите за текущия пласт
current_layer_key = f"layer_{layer_idx}"
if current_layer_key not in st.session_state.K_values:
    # Инициализираме с всички необходими ключове, включително 'C'
    st.session_state.K_values[current_layer_key] = {'K1': 1.0, 'K2': 1.0, 'K3': 1.0, 'C': 1.0}

# Вземаме стойностите, като гарантираме че 'C' съществува
layer_values = st.session_state.K_values[current_layer_key]
if 'C' not in layer_values:
    layer_values['C'] = 1.0  # Добавяме 'C' ако липсва
    
# K1 е фиксирана на 0.6
K1 = 0.6
layer_values['K1'] = K1

with cols[0]:
    K1 = st.number_input(
        "K₁",
        value=0.6,               # <-- фиксирана стойност
        step=0.1,
        format="%.2f",
        key=f"K1_{layer_idx}",
        disabled=True            # <-- така не може да се променя
    )
    
with cols[1]:
    K2 = st.number_input("K₂", 
                        value=layer_values['K2'], 
                        step=0.1, 
                        format="%.2f",
                        key=f"K2_{layer_idx}",
                        on_change=lambda: layer_values.update({'K2': st.session_state[f"K2_{layer_idx}"]}))

with cols[2]:
    K3 = st.number_input("K₃", 
                        value=layer_values['K3'], 
                        step=0.1, 
                        format="%.2f",
                        key=f"K3_{layer_idx}",
                        on_change=lambda: layer_values.update({'K3': st.session_state[f"K3_{layer_idx}"]}))

with cols[3]:
    C = st.number_input("C", 
                       value=layer_values['C'], 
                       step=0.1, 
                       format="%.3f",
                       key=f"C_{layer_idx}",
                       on_change=lambda: layer_values.update({'C': st.session_state[f"C_{layer_idx}"]}))

# Изчисление на K
d = 1.15
f = 0.65
K = (K1 * K2) / (d * f) * (1 / K3)
tau_dop = K * C

# КОРИГИРАНО: Лявата страна: τμ + τb вместо p*(τμ/p + τb)
left_side = tau_mu + tau_b
right_side = tau_dop

# КОРИГИРАНИ LaTeX формули
formula_k = fr"""
K = \frac{{K_1 \cdot K_2}}{{d \cdot f}} \cdot \frac{{1}}{{K_3}} = 
\frac{{{K1:.2f} \cdot {K2:.2f}}}{{1.15 \cdot 0.65}} \cdot \frac{{1}}{{{K3:.2f}}} = {K:.3f}
"""

main_formula = fr"""
\tau_{{\mu}} + \tau_b \leq K \cdot C \\
{tau_mu:.6f} + ({tau_b:.6f}) = {left_side:.6f} \leq {K:.3f} \cdot {C:.2f} = {right_side:.6f}
"""

st.latex(formula_k)
st.latex(main_formula)

# Проверка на условието
if left_side <= right_side:
    st.success(f"Условието е изпълнено: {left_side:.6f} ≤ {right_side:.6f}")
else:
    st.error(f"Условието НЕ е изпълнено: {left_side:.6f} > {right_side:.6f}")

# Линк към предишната страница
st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")

# -------------------------------------------------
# Функции за PDF отчет със стила от Опън в покритието.py
# -------------------------------------------------

def render_formula_to_image(formula_text, fontsize=26, dpi=150):
    """Рендва формула като изображение чрез matplotlib mathtext"""
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

def generate_pdf_report():
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm
        )
        story = []
        styles = getSampleStyleSheet()

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
        
        story.append(Paragraph("ОПРЕДЕЛЯНЕ НА Ꚍμ/p ЗА НЕСЪРЗАНИ ПОЧВИ", title_style))
        subtitle_style = ParagraphStyle(
            'Subtitle',
            fontSize=14,
            spaceAfter=10,
            alignment=1,
            textColor=colors.HexColor('#2C5530'),
            fontName=font_name
        )
        story.append(Paragraph("Фигура 9.6 - maxH/D=1.5", subtitle_style))
        story.append(Spacer(1, 16.5))

        # ИНФОРМАЦИЯ ЗА ПАРАМЕТРИ
        table_data = [
            ["ПАРАМЕТЪР", "СТОЙНОСТ", "ЕДИНИЦА"],
            ["Диаметър D", f"{st.session_state.get('fig9_6_D', D):.2f}", "cm"],
            ["Брой пластове", f"{n}", ""],
            ["Осова тежест", f"{st.session_state.get('axle_load', axle_load)}", "kN"],
            ["Избран пласт", f"{layer_idx + 1}", ""],
            ["p", f"{p_value:.3f}", "MPa"],
        ]

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
        story.append(Spacer(1, 16.5))

        # ТАБЛИЦА ЗА ПЛАСТОВЕТЕ (Коригирано - използване на ReportLab Table)
        layer_table_data = [
            ["Пласт", "h (cm)", "Ei (MPa)", "Ed (MPa)", "Fi (°)"]
        ]
        
        # Добавяне на данните за пластовете
        for i in range(n):
            layer_table_data.append([
                f"{i+1}",
                f"{h_values[i]}",
                f"{Ei_values[i]}",
                f"{Ed_values[i]}",
                f"{Fi_values[i]}"
            ])
        
        # СЪЗДАВАНЕ НА СПИСЪК СЪС СТИЛОВЕ ДИНАМИЧНО
        layer_table_style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#C8E6C9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1B5E20')),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            
            # Основен фон за всички редове (бял)
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
            
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDBDBD')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
        ]
        
        # ДОБАВЯНЕ НА АЛТЕРНИРАЩИ ЦВЕТОВЕ САМО ЗА СЪЩЕСТВУВАЩИТЕ РЕДОВЕ
        # Добавяме светлосиви редове за всеки четен ред (след заглавния)
        for row in range(2, len(layer_table_data), 2):  # Започваме от ред 2 (първи данни ред е 1)
            if row < len(layer_table_data):  # Проверка за безопасност
                layer_table_style_commands.append(
                    ('BACKGROUND', (0, row), (-1, row), colors.HexColor('#F5F5F5'))
                )
        
        layer_table = Table(layer_table_data, colWidths=[25*mm, 30*mm, 30*mm, 30*mm, 30*mm])
        layer_table.setStyle(TableStyle(layer_table_style_commands))
        
        # Добавяне на заглавие за таблицата на пластовете
        layer_title_style = ParagraphStyle(
            'LayerTitle',
            fontName=font_name,
            fontSize=12,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=8,
            alignment=0
        )
        story.append(Paragraph("Параметри на пластовете:", layer_title_style))
        story.append(layer_table)
        story.append(Spacer(1, 10))
        
        # ... останалият код ...
        
        # ФОРМУЛИ ЗА ИЗЧИСЛЕНИЕ
        formulas_title_style = ParagraphStyle(
            'FormulasTitle',
            fontName=font_name,
            fontSize=14.08,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=11,
            alignment=0
        )
        story.append(Paragraph("2. Формули за изчисление", formulas_title_style))

        formulas = [
            r"H = \sum_{i=1}^n h_i",
            r"Esr = \frac{\sum (E_i h_i)}{\sum h_i}",
            r"E_o = Ed_i", 
            r"\frac{H}{D}",
            r"\frac{Esr}{E_o}",
            r"\tau_\mu = \left(\frac{\tau_\mu}{p}\right) \times p",
            r"\tau_{dop} = K \times C",
            r"K = \frac{K_1 \cdot K_2}{d \cdot f} \cdot \frac{1}{K_3}",
            r"\tau_\mu + \tau_b \leq K \cdot C"
        ]

        formula_table_data = []
        for i in range(0, len(formulas), 3):
            row = []
            for j in range(3):
                idx = i + j
                if idx < len(formulas):
                    try:
                        img_buf = render_formula_to_image(formulas[idx], fontsize=23.76, dpi=150)
                        row.append(RLImage(img_buf, width=66*mm, height=19.8*mm))
                    except:
                        row.append(Paragraph(formulas[idx].replace('_', '').replace('^', ''), formulas_title_style))
                else:
                    row.append('')
            formula_table_data.append(row)

        formula_table = Table(formula_table_data, colWidths=[70*mm, 70*mm, 70*mm])
        formula_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8.8),
            ('TOPPADDING', (0, 0), (-1, -1), 8.8),
        ]))
        
        story.append(formula_table)
        story.append(Spacer(1, 22))

        # ... (продължавате с останалия код както е в оригиналната функция) ...

        # ИЗЧИСЛЕНИЯ
        calculations_title_style = ParagraphStyle(
            'CalculationsTitle',
            fontName=font_name,
            fontSize=14.08,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=11,
            alignment=0
        )
        story.append(Paragraph("3. Изчисления", calculations_title_style))

        calculation_formulas = [
            fr"H = {H:.2f} \ \mathrm{{cm}}",
            fr"Esr = {Esr:.0f} \ \mathrm{{MPa}}",
            fr"E_o = Ed_{{{layer_idx+1}}} = {Eo} \ \mathrm{{MPa}}",
            fr"\frac{{H}}{{D}} = \frac{{{H:.2f}}}{{{D}}} = {ratio:.3f}",
            fr"\frac{{Esr}}{{E_o}} = \frac{{{Esr}}}{{{Eo}}} = {Esr_over_Eo:.3f}",
        ]

        if 'x_orange' in locals() and x_orange is not None:
            calculation_formulas.extend([
                fr"\frac{{\tau_\mu}}{{p}} = {sigma_r:.3f}",
                fr"\tau_\mu = {sigma_r:.3f} \times {p_value:.3f} = {tau_mu:.6f} \ \mathrm{{MPa}}",
            ])
        else:
            calculation_formulas.extend([
                r"\frac{\tau_\mu}{p} = -",
                r"\tau_\mu = -",
            ])

        if tau_b is not None:
            calculation_formulas.extend([
                fr"\tau_b = {tau_b:.6f} \ \mathrm{{MPa}}",
            ])

        calculation_formulas.extend([
            fr"K = \frac{{{K1:.2f} \times {K2:.2f}}}{{1.15 \times 0.65}} \times \frac{{1}}{{{K3:.2f}}} = {K:.3f}",
            fr"\tau_{{dop}} = {K:.3f} \times {C:.3f} = {tau_dop:.6f} \ \mathrm{{MPa}}",
        ])

        if 'tau_mu' in locals() and tau_b is not None:
            calculation_formulas.extend([
                fr"\tau_\mu + \tau_b = {tau_mu:.6f} + {tau_b:.6f} = {left_side:.6f} \ \mathrm{{MPa}}",
            ])

        calc_table_data = []
        for i in range(0, len(calculation_formulas), 2):
            row = []
            if i < len(calculation_formulas):
                try:
                    img_buf1 = render_formula_to_image(calculation_formulas[i], fontsize=21.12, dpi=150)
                    row.append(RLImage(img_buf1, width=99*mm, height=18.48*mm))
                except:
                    simple_text = calculation_formulas[i].replace('{', '').replace('}', '').replace('\\', '')
                    row.append(Paragraph(simple_text, calculations_title_style))
            else:
                row.append('')
            
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
        story.append(Spacer(1, 22))

        # РЕЗУЛТАТ И ПРОВЕРКА
        results_title_style = ParagraphStyle(
            'ResultsTitle',
            fontName=font_name,
            fontSize=17.6,
            textColor=colors.HexColor('#006064'),
            spaceAfter=16.5,
            alignment=1
        )
        story.append(Paragraph("РЕЗУЛТАТ И ПРОВЕРКА", results_title_style))

        check_passed = left_side <= right_side
        results_data = [
            ["ПАРАМЕТЪР", "СТОЙНОСТ"],
            ["τμ + τb", f"{left_side:.6f} MPa"],
            ["K × C", f"{right_side:.6f} MPa"]
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
            story.append(Paragraph("УСЛОВИЕТО Е ИЗПЪЛНЕНО", status_style))
            subtitle_style = ParagraphStyle(
                'SubtitleStyle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=5.5,
                fontName=font_name,
                textColor=colors.HexColor('#5D4037'),
                alignment=1
            )
            story.append(Paragraph("τμ + τb ≤ K × C", subtitle_style))
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
            story.append(Paragraph("УСЛОВИЕТО НЕ Е ИЗПЪЛНЕНО", status_style))
            subtitle_style = ParagraphStyle(
                'SubtitleStyle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=5.5,
                fontName=font_name,
                textColor=colors.HexColor('#5D4037'),
                alignment=1
            )
            story.append(Paragraph("τμ + τb > K × C", subtitle_style))

        # НОВ ЛИСТ ЗА ГРАФИКИ
        story.append(PageBreak())

        # ГРАФИКА НА ИЗОЛИНИИТЕ
        graph_title_style = ParagraphStyle(
            'GraphTitle',
            fontName=font_name,
            fontSize=17.6,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=16.5,
            alignment=1
        )
        story.append(Paragraph("ГРАФИКА НА ИЗОЛИНИИТЕ", graph_title_style))
        
        try:
            # Оптимизирана версия на графиката за PDF
            pdf_fig = go.Figure()
            
            # Добавяне на изолиниите
            for fi_val in unique_fi:
                df_level = df_fi[df_fi['fi'] == fi_val].sort_values(by='H/D')
                pdf_fig.add_trace(go.Scatter(
                    x=df_level['H/D'],
                    y=df_level['y'],
                    mode='lines',
                    name=f'ϕ = {fi_val}',
                    line=dict(width=2)
                ))

            for val in unique_esr_eo:
                df_level = df_esr_eo[df_esr_eo['Esr_Eo'] == val].sort_values(by='H/D')
                pdf_fig.add_trace(go.Scatter(
                    x=df_level['H/D'],
                    y=df_level['y'],
                    mode='lines',
                    name=f'Esr/Eo = {val}',
                    line=dict(width=2)
                ))

            # Добавяне на точките и линиите (ако има изчисление)
            if point_on_esr_eo is not None:
                pdf_fig.add_trace(go.Scatter(
                    x=[point_on_esr_eo[0]],
                    y=[point_on_esr_eo[1]],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Червена точка'
                ))
                
                pdf_fig.add_trace(go.Scatter(
                    x=[ratio, ratio],
                    y=[0, point_on_esr_eo[1]],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Вертикална линия'
                ))

                if 'x_orange' in locals() and x_orange is not None:
                    pdf_fig.add_trace(go.Scatter(
                        x=[x_orange],
                        y=[y_red],
                        mode='markers',
                        marker=dict(color='orange', size=10),
                        name='Оранжева точка'
                    ))
                    
                    pdf_fig.add_trace(go.Scatter(
                        x=[point_on_esr_eo[0], x_orange],
                        y=[y_red, y_red],
                        mode='lines',
                        line=dict(color='orange', dash='dash'),
                        name='Хоризонтална линия'
                    ))

            # Настройки за PDF
            pdf_fig.update_layout(
                title=dict(
                    text="Номограма: Ꚍμ/p за несързани почви (фиг. 9.6)",
                    font=dict(size=19.8, color='black', family="Arial")
                ),
                xaxis=dict(
                    title="H/D",
                    title_font=dict(size=15.4, color='black'),
                    tickfont=dict(size=13.2, color='black'),
                    linecolor='black',
                    gridcolor='lightgray',
                    mirror=True,
                    showgrid=True
                ),
                yaxis=dict(
                    title="y",
                    title_font=dict(size=15.4, color='black'),
                    tickfont=dict(size=13.2, color='black'),
                    linecolor='black',
                    gridcolor='lightgray',
                    mirror=True,
                    showgrid=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend=dict(
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=11, color='black'),
                    x=1.05,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle'
                ),
                width=1200,
                height=800,
                margin=dict(r=150)
            )
            
            img_bytes = pio.to_image(
                pdf_fig, 
                format="png", 
                width=1200, 
                height=800,
                scale=4,
                engine="kaleido"
            )
            
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

        # ГРАФИКА ЗА τb
        if tau_b_fig is not None:
            try:
                # Конвертиране на Matplotlib фигурата
                img_buffer_tau = io.BytesIO()
                tau_b_fig.savefig(img_buffer_tau, format='png', dpi=300, bbox_inches='tight')
                img_buffer_tau.seek(0)
                
                story.append(PageBreak())
                story.append(Paragraph("ГРАФИКА ЗА τb", graph_title_style))
                story.append(RLImage(img_buffer_tau, width=170 * mm, height=130 * mm))
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
                story.append(Paragraph(f"Грешка при добавяне на τb графика: {e}", error_style))

        # ТАБЛИЦА 9.8
        img_path_9_8 = "9.8 Таблица.png"
        if os.path.exists(img_path_9_8):
            story.append(PageBreak())
            table_title_style = ParagraphStyle(
                'TableTitle',
                fontName=font_name,
                fontSize=15.4,
                textColor=colors.HexColor('#2C5530'),
                spaceAfter=11,
                alignment=1
            )
            story.append(Paragraph("ТАБЛИЦА 9.8", table_title_style))
            
            try:
                pil_img = PILImage.open(img_path_9_8)
                img_buffer_table = io.BytesIO()
                pil_img.save(img_buffer_table, format="PNG")
                img_buffer_table.seek(0)
                story.append(RLImage(img_buffer_table, width=170 * mm, height=130 * mm))
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
                story.append(Paragraph(f"Грешка при зареждане на таблицата: {e}", error_style))

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
            try:
                canvas.setFont('DejaVuSans', 8)
            except:
                canvas.setFont('Helvetica', 8)
            page_num = canvas.getPageNumber()
            canvas.drawString(190*mm, 15*mm, f"{page_num}")
            canvas.restoreState()
        
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        buffer.seek(0)
        
        return buffer

    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# -------------------------------------------------
# UI за генериране на PDF отчет
# -------------------------------------------------
st.markdown("---")
st.subheader("Генериране на PDF отчет")

# Избор на начален номер на страница
start_page_number = st.number_input(
    "Начален номер на страница:",
    min_value=1,
    max_value=1000,
    value=1,
    step=1,
    help="Задайте от кой номер да започва номерацията на страниците",
    key="start_page_taumu"
)

if st.button("📄 Генерирай PDF отчет", type="primary"):
    with st.spinner('Генериране на PDF отчет...'):
        try:
            pdf_buffer = generate_pdf_report()
            if pdf_buffer:
                st.success("✅ PDF отчетът с модерно графично оформление е готов!")
                st.download_button(
                    "📥 Изтегли PDF отчет",
                    pdf_buffer,
                    file_name=f"Ꚍμ_p_несързани_почви_Отчет_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("❌ Неуспешно генериране на PDF. Моля, проверете грешките по-горе.")
        except Exception as e:
            st.error(f"Грешка при генериране на PDF: {str(e)}")
