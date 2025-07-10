import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

st.title("Определяне на Ꚍμ/p за сързани почви фиг9.4 maxH/D=2")

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
        
        # Маркиране на пресечната точка
        ax.plot([h_to_x[h_low], h_to_x[h_high]], [y_tau, y_tau], 'g--', alpha=0.6)
        ax.plot(h_to_x[h_value], y_tau, 'ko', markersize=8, 
                label=f'τb = {y_tau:.6f}\nH: {h_low}→{h_value}→{h_high}\nφ: {fi_low}→{fi_value}→{fi_high}')
        
        # Настройки на графиката
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        h_ticks = sorted(set([h_low, h_value, h_high] + h_values_available))
        x_positions = [h_to_x[h] for h in h_ticks if h in h_to_x]
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h:.1f}" for h in h_ticks if h in h_to_x])
        
        ax.set_xlabel('H', fontsize=12)
        ax.set_ylabel('τb', fontsize=12)
        ax.set_title(f'Номограма за активно напрежение на срязване (τb)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')
        
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
n = 3  # Стойност по подразбиране

# Проверка за данни в session_state
session_data_available = all(key in st.session_state for key in ['fig9_4_h']) and \
                         'layers_data' in st.session_state and \
                         len(st.session_state.layers_data) > 0

# Автоматично зареждане на данни ако са налични
if session_data_available:
    try:
        n = len(st.session_state.fig9_4_h)
        h_values = st.session_state.fig9_4_h
        Ed_values = [round(layer["Ed"]) for layer in st.session_state.layers_data]
        Ei_values = [round(layer["Ei"]) for layer in st.session_state.layers_data]
        
        D_options = [32.04, 34.0, 33.0]
        
        if 'fig9_4_D' in st.session_state:
            current_d = st.session_state.fig9_4_D
            if current_d not in D_options:
                D_options.insert(0, current_d)
        else:
            current_d = D_options[0]

        selected_d = st.selectbox("Избери D", options=D_options, index=D_options.index(current_d))
        st.session_state.fig9_4_D = selected_d
        D = selected_d
        
        Fi_input = st.number_input("Fi (ϕ) стойност", value=15, step=1)
        
        st.markdown("### Автоматично заредени данни за пластовете")
        cols = st.columns(3)
        
        h_values_edited = []
        Ei_values_edited = []
        Ed_values_edited = []
        
        for i in range(n):
            with cols[0]:
                # Добавяне на проверка за граници
                default_h = float(h_values[i]) if i < len(h_values) else 4.0
                h_val = st.number_input(f"h{to_subscript(i+1)}", value=default_h, step=0.1, key=f"auto_h_{i}")
                h_values_edited.append(h_val)
            with cols[1]:
                default_ei = int(Ei_values[i]) if i < len(Ei_values) else 1000
                ei_val = st.number_input(f"Ei{to_subscript(i+1)}", value=default_ei, step=1, key=f"auto_Ei_{i}")
                Ei_values_edited.append(ei_val)
            with cols[2]:
                default_ed = int(Ed_values[i]) if i < len(Ed_values) else 1000
                ed_val = st.number_input(f"Ed{to_subscript(i+1)}", value=default_ed, step=1, key=f"auto_Ed_{i}")
                Ed_values_edited.append(ed_val)
        
        h_values = h_values_edited
        Ei_values = Ei_values_edited
        Ed_values = Ed_values_edited

    except Exception as e:
        st.error(f"Грешка при зареждане на данните: {str(e)}")
        session_data_available = False

# Ръчно въвеждане ако няма данни в сесията или има грешка
if not session_data_available:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
    D_options = [32.04, 34.0, 33.0]
    selected_d = st.selectbox("Избери D", options=D_options, index=0)
    st.session_state.fig9_4_D = selected_d
    D = selected_d
    
    Fi_input = st.number_input("Fi (ϕ) стойност", value=15, step=1)
    
    st.markdown("### Въведи стойности за всеки пласт")
    h_values = []
    Ei_values = []
    Ed_values = []
    cols = st.columns(3)
    for i in range(n):
        with cols[0]:
            h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
            h_values.append(h)
        with cols[1]:
            Ei_val = st.number_input(f"Ei{to_subscript(i+1)}", value=1000, step=1, key=f"Ei_{i}")
            Ei_values.append(Ei_val)
        with cols[2]:
            Ed_val = st.number_input(f"Ed{to_subscript(i+1)}", value=1000, step=1, key=f"Ed_{i}")
            Ed_values.append(Ed_val)

# Избор на пласт за проверка
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Задаване на Eo = Ed на избрания пласт (с закръгляне)
Eo = round(Ed_values[layer_idx])
st.markdown(f"**Eo = Ed{to_subscript(layer_idx+1)} = {Eo}**")

# Изчисляване на H и Esr за избрания пласт (с закръгляне)
h_array = np.array(h_values[:layer_idx+1])
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
st.write(f"H = {H:.3f}")

st.latex(r"Esr = \frac{\sum_{i=1}^n (E_i \cdot h_i)}{\sum_{i=1}^n h_i}")
numerator = " + ".join([f"{Ei_rounded[i]} \cdot {h_values[i]}" for i in range(layer_idx+1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(layer_idx+1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum:.3f}}}{{{H:.3f}}} = {Esr}"
st.latex(formula_with_values)

ratio = H / D if D != 0 else 0
st.latex(r"\frac{H}{D} = \frac{" + f"{H:.3f}" + "}{" + f"{D}" + "} = " + f"{ratio:.3f}")

st.latex(r"\frac{Esr}{E_o} = \frac{" + f"{Esr}" + "}{" + f"{Eo}" + "} = " + f"{Esr / Eo:.3f}")
Esr_over_Eo = Esr / Eo if Eo != 0 else 0

# Зареждане на данни
df_fi = pd.read_csv("fi.csv")
df_esr_eo = pd.read_csv("Esr_Eo.csv")

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
    x_orange = interp_x_for_fi_interp(df_fi, Fi_input, y_red)

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
            y=[y_red, 1.35],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name='Вертикална линия до y=1.35'
        ))

# Настройка на графиката
fig.update_layout(
    title=f"Графика на изолинии и точки за пласт {layer_idx+1}",
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Легенда",
    width=900,
    height=600
)

# Определи фиксиран мащаб на основната ос (например 0 до 2)
xaxis_min = 0
xaxis_max = 2

# Добавяне на невидим trace, за да се покаже втората ос x2
fig.add_trace(go.Scatter(
    x=[xaxis_min, xaxis_max],
    y=[None, None],
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip',
    xaxis='x2'
))

fig.update_layout(
    title=f'Графика на изолинии за пласт {layer_idx+1}',
    xaxis=dict(
        title='H/D',
        showgrid=True,
        zeroline=False,
        range=[xaxis_min, xaxis_max],
    ),
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[xaxis_min, xaxis_max],
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickvals=np.linspace(xaxis_min, xaxis_max, 11),
        ticktext=[f"{(0.20 * (x - xaxis_min) / (xaxis_max - xaxis_min)):.3f}" for x in np.linspace(xaxis_min, xaxis_max, 11)],
        title='φ',
        fixedrange=True,
        showticklabels=True,
    ),
    yaxis=dict(
        title='y',
    ),
    showlegend=False,
    height=600,
    width=900
)

st.plotly_chart(fig, use_container_width=True)

# Изчисление на σr от x на оранжевата точка (ако съществува)
if 'x_orange' in locals() and x_orange is not None:
    sigma_r = round(x_orange / 10, 3)
    x_val = round(x_orange, 3)
    st.markdown(f"**Ꚍμ/p = {sigma_r}**")
else:
    st.markdown("**Ꚍμ/p = -** (Няма изчислена стойност)")

# Изчисляване и визуализация на τb за текущия пласт
st.divider()
st.subheader("Изчисление на активно напрежение на срязване τb")

tau_b_fig, tau_b = plot_tau_b(Fi_input, H)
if tau_b_fig is not None and tau_b is not None:
    st.markdown(f"**За пласт {layer_idx+1}:**")
    st.markdown(f"- H = {H:.3f}")
    st.markdown(f"- ϕ = {Fi_input}")
    st.markdown(f"**τb = {tau_b:.6f}**")
    st.pyplot(tau_b_fig)
else:
    st.error("Неуспешно изчисление на τb")
