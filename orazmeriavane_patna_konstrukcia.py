import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")  # активира широк режим

# задаваш конкретна максимална ширина на контейнера
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    # Заменете пътя с вашия CSV файл
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={
        "E1_over_E2": "Ed_over_Ei",
        "Eeq_over_E2": "Ee_over_Ei"
    })
    return df

data = load_data()

# --- Въвеждане на брой пластове ---
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

# Запазване на текущия пласт в session_state
if 'current_layer' not in st.session_state:
    st.session_state['current_layer'] = 1

# Запазване на данните за всеки пласт в речник
if 'layers_data' not in st.session_state:
    st.session_state['layers_data'] = {}

layer = st.session_state['current_layer']

st.title("Оразмеряване на пътна конструкция")

# За всички пластове D и axle_load могат да се ползват глобални настройки или зададени по отделно
d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34], index=0, key="d_value_global")
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115], index=0, key="axle_load_global")

st.subheader(f"Въведете данни за оразмеряване - Пласт {layer}")

# Ако вече имаме данни за този пласт, ги зареждаме, иначе дефолт стойности
default_layer_data = st.session_state['layers_data'].get(layer, {})

# Показваме избраната стойност D (не като входно поле)
st.markdown(f"**Стойност D за пласт {layer}:** {d_value} cm")

Ee = st.number_input(
    f"Въведете стойност за Ee (MPa) - Пласт {layer}:",
    min_value=0.1, step=0.1,
    value=default_layer_data.get('Ee', 2700.0),
    key=f"Ee_{layer}"
)

Ei = st.number_input(
    f"Въведете стойност за Ei (MPa) - Пласт {layer}:",
    min_value=0.1, step=0.1,
    value=default_layer_data.get('Ei', 3000.0),
    key=f"Ei_{layer}"
)

h = st.number_input(
    f"Въведете дебелина h (cm) - Пласт {layer}:",
    min_value=0.1, step=0.1,
    value=default_layer_data.get('h', 4.0),
    key=f"h_{layer}"
)

# Избор на режим за изчисление (запазваме за всеки пласт, ако искаш)
mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    index=default_layer_data.get('mode_index', 0),
    key=f"mode_{layer}"
)

def compute_Ed(h, D, Ee, Ei):
    hD = h / D
    EeEi = Ee / Ei
    tol = 1e-4
    iso_levels = sorted(data['Ee_over_Ei'].unique())

    for low, high in zip(iso_levels, iso_levels[1:]):
        if not (low - tol <= EeEi <= high + tol):
            continue

        grp_low = data[data['Ee_over_Ei'] == low].sort_values('h_over_D')
        grp_high = data[data['Ee_over_Ei'] == high].sort_values('h_over_D')

        h_min = max(grp_low['h_over_D'].min(), grp_high['h_over_D'].min())
        h_max = min(grp_low['h_over_D'].max(), grp_high['h_over_D'].max())
        if not (h_min - tol <= hD <= h_max + tol):
            continue

        y_low = np.interp(hD, grp_low['h_over_D'], grp_low['Ed_over_Ei'])
        y_high = np.interp(hD, grp_high['h_over_D'], grp_high['Ed_over_Ei'])

        frac = 0 if np.isclose(high, low) else (EeEi - low) / (high - low)
        ed_over_ei = y_low + frac * (y_high - y_low)

        return ed_over_ei * Ei, hD, y_low, y_high, low, high

    return None, None, None, None, None, None

def compute_h(Ed, D, Ee, Ei):
    EeEi = Ee / Ei
    EdEi = Ed / Ei
    tol = 1e-4
    iso_levels = sorted(data['Ee_over_Ei'].unique())

    for low, high in zip(iso_levels, iso_levels[1:]):
        if not (low - tol <= EeEi <= high + tol):
            continue

        grp_low = data[data['Ee_over_Ei'] == low].sort_values('h_over_D')
        grp_high = data[data['Ee_over_Ei'] == high].sort_values('h_over_D')

        h_min = max(grp_low['h_over_D'].min(), grp_high['h_over_D'].min())
        h_max = min(grp_low['h_over_D'].max(), grp_high['h_over_D'].max())

        hD_values = np.linspace(h_min, h_max, 1000)

        for hD in hD_values:
            y_low = np.interp(hD, grp_low['h_over_D'], grp_low['Ed_over_Ei'])
            y_high = np.interp(hD, grp_high['h_over_D'], grp_high['Ed_over_Ei'])
            frac = 0 if np.isclose(high, low) else (EeEi - low) / (high - low)
            ed_over_ei = y_low + frac * (y_high - y_low)

            if abs(ed_over_ei - EdEi) < tol:
                return hD * D, hD, y_low, y_high, low, high

    return None, None, None, None, None, None

# Помощна функция за визуализацията на графиката и правоъгълника
def plot_layer_graph(layer, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso, Ei, Ee, h):
    fig = go.Figure()
    for value, group in data.groupby("Ee_over_Ei"):
        group_sorted = group.sort_values("h_over_D")
        fig.add_trace(go.Scatter(
            x=group_sorted["h_over_D"],
            y=group_sorted["Ed_over_Ei"],
            mode='lines',
            name=f"Ee / Ei = {value:.2f}",
            line=dict(width=1)
        ))
    fig.add_trace(go.Scatter(
        x=[hD_point],
        y=[EdEi_point],
        mode='markers',
        name="Твоята точка",
        marker=dict(size=8, color='red', symbol='circle')
    ))
    if y_low is not None and y_high is not None:
        fig.add_trace(go.Scatter(
            x=[hD_point, hD_point],
            y=[y_low, y_high],
            mode='lines',
            line=dict(color='green', width=2, dash='dot'),
            name="Интерполационна линия"
        ))
    fig.update_layout(
        xaxis_title="h / D",
        yaxis_title="Ed / Ei",
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 8px;
            color: #004d40;
        ">
            Пласт {layer}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 400px;
            height: 60px;
            background-color: #add8e6;
            border: 2px solid black;
            border-radius: 6px;
            margin: 0 auto 40px auto;
            padding: 10px;
            font-family: Arial, sans-serif;
            ">
            <!-- Ei в средата -->
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                font-size: 18px;
                color: black;
            ">
                Ei = {Ei} MPa
            </div>
            <!-- Ee в горния десен ъгъл -->
            <div style="
                position: absolute;
                top: -20px;
                right: 10px;
                font-size: 14px;
                color: darkblue;
                font-weight: bold;
            ">
                Ee = {Ee} MPa
            </div>
            <!-- h вдясно центрирано вертикално -->
            <div style="
                position: absolute;
                top: 50%;
                left: 8px;
                transform: translateY(-50%);
                font-size: 14px;
                color: black;
                font-weight: bold;
            ">
                h = {h:.2f} cm
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Изчисленията според режим
Ed = None
h_calc = None
hD_point = None
y_low = None
y_high = None
low_iso = None
high_iso = None

if mode == "Ed / Ei":
    Ed, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)
elif mode == "h / D":
    # Ако има въведена стойност Ed, я ползваме, иначе None
    Ed_input = st.number_input(f"Въведете Ed (MPa) за пласт {layer}:", min_value=0.1, step=0.1, value=default_layer_data.get('Ed', 5.0), key=f"Ed_{layer}")
    h_calc, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed_input, d_value, Ee, Ei)
    Ed = Ed_input

if Ed is not None:
    st.success(f"Изчислен Ed за пласт {layer}: {Ed:.2f} MPa")
elif h_calc is not None:
    st.success(f"Изчислено h за пласт {layer}: {h_calc:.2f} cm")

if st.button("Запази данните", key=f"save_{layer}"):
    st.session_state['layers_data'][layer] = {
        "Ee": Ee,
        "Ei": Ei,
        "h": h,
        "mode_index": 0 if mode == "Ed / Ei" else 1,
        "Ed": Ed,
        "h_calc": h_calc,
        "d_value": d_value,
        "axle_load": axle_load
    }
    st.success(f"Данните за пласт {layer} са запазени!")

# Бутон "Напред" долу вдясно
col1, col2, col3 = st.columns([8, 1, 1])
with col3:
    if st.button("Напред", key=f"next_{layer}"):
        if layer < num_layers:
            # Автоматично записваме, ако не е запазено
            if layer not in st.session_state['layers_data']:
                st.session_state['layers_data'][layer] = {
                    "Ee": Ee,
                    "Ei": Ei,
                    "h": h,
                    "mode_index": 0 if mode == "Ed / Ei" else 1,
                    "Ed": Ed,
                    "h_calc": h_calc,
                    "d_value": d_value,
                    "axle_load": axle_load
                }
            st.session_state['current_layer'] += 1
            st.experimental_rerun()
        else:
            st.success("Оразмеряването на всички пластове приключи!")

# Покажи всички запазени пластове надолу с графики и правоъгълници
if st.session_state['layers_data']:
    st.markdown("---")
    st.header("Вече въведени пластове:")
    for l in range(1, st.session_state['current_layer'] + 1):
        d = st.session_state['layers_data'].get(l)
        if d:
            Ed_val = d.get('Ed')
            h_val = d.get('h')
            Ei_val = d.get('Ei')
            Ee_val = d.get('Ee')
            mode_idx = d.get('mode_index', 0)
            mode_str = "Ed / Ei" if mode_idx == 0 else "h / D"

            st.subheader(f"Пласт {l} - {mode_str}")
            st.markdown(f"- Ee = {Ee_val} MPa")
            st.markdown(f"- Ei = {Ei_val} MPa")
            st.markdown(f"- h = {h_val} cm")
            if Ed_val:
                st.markdown(f"- Ed = {Ed_val:.2f} MPa")

            # Пресмятаме точката за графиката, за да е актуална
            if mode_idx == 0:
                _, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h_val, d_value, Ee_val, Ei_val)
            else:
                _, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed_val, d_value, Ee_val, Ei_val)

            EdEi_point = Ed_val / Ei_val if Ed_val else None

            plot_layer_graph(l, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso, Ei_val, Ee_val, h_val)
