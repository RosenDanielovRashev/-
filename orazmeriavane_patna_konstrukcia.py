import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

# CSS за максимална ширина
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
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={
        "E1_over_E2": "Ed_over_Ei",
        "Eeq_over_E2": "Ee_over_Ei"
    })
    return df

data = load_data()

# --- Инициализация на session_state ---
if 'num_layers' not in st.session_state:
    st.session_state.num_layers = 1
if 'current_layer' not in st.session_state:
    st.session_state.current_layer = 1
if 'layers_data' not in st.session_state:
    st.session_state.layers_data = {}

st.title("Оразмеряване на пътна конструкция")

# Въвеждаме брой пластове (само при първо стартиране)
if st.session_state.current_layer == 1:
    st.session_state.num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

# Задаваме текущия пласт
layer = st.session_state.current_layer
num_layers = st.session_state.num_layers

st.header(f"Пласт {layer} от {num_layers}")

# Ако има запазени данни, ги зареждаме, иначе задаваме стойности по подразбиране
layer_data = st.session_state.layers_data.get(layer, {})

d_value = st.selectbox(
    "Изберете стойност за D (cm):",
    options=[32.04, 34],
    index=[32.04, 34].index(layer_data.get('d_value', 32.04))
)

axle_load = st.selectbox(
    "Изберете стойност за осов товар (kN):",
    options=[100, 115],
    index=[100, 115].index(layer_data.get('axle_load', 100))
)

mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    index=0 if layer_data.get('mode', "Ed / Ei") == "Ed / Ei" else 1
)

if mode == "Ed / Ei":
    Ee = st.number_input(
        "Въведете стойност за Ee (MPa):",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ee', 2700.0)
    )
    Ei = st.number_input(
        "Въведете стойност за Ei (MPa):",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ei', 3000.0)
    )
    h = st.number_input(
        "Въведете дебелина h (cm):",
        min_value=0.1, step=0.1,
        value=layer_data.get('h', 4.0)
    )
else:  # режим "h / D"
    Ed = st.number_input(
        "Ed (MPa):",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ed', 520.0)
    )
    Ee = st.number_input(
        "Въведете стойност за Ee (MPa):",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ee', 2700.0)
    )
    Ei = st.number_input(
        "Въведете стойност за Ei (MPa):",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ei', 3000.0)
    )

# Функции от твоя код
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

# Изчисления и визуализация
if Ei == 0 or d_value == 0:
    st.error("Ei и D не могат да бъдат 0.")
    st.stop()

if mode == "Ed / Ei":
    EeEi = Ee / Ei
    st.subheader("📊 Въведени параметри:")
    st.write(pd.DataFrame({
        "Параметър": ["Ee", "Ei", "h", "D", "Ee / Ei", "h / D"],
        "Стойност": [
            Ee,
            Ei,
            h,
            d_value,
            round(EeEi, 3),
            round(h / d_value, 3)
        ]
    }))

    if st.button("Изчисли Ed"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / Ei
            st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

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

            # Правоъгълник с параметри
            st.markdown(
                f"""
                <div style="
                    position: relative;
                    width: 400px;
                    height: 60px;
                    background-color: #add8e6;
                    border: 2px solid black;
                    border-radius: 6px;
                    margin: 20px auto 40px auto;
                    padding: 10px;
                    font-family: Arial, sans-serif;
                    ">
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
else:
    Ed = st.number_input(
        "Ed (MPa)",
        min_value=0.1, step=0.1,
        value=layer_data.get('Ed', 520.0)
    )
    EeEi = Ee / Ei
    EdEi = Ed / Ei

    st.subheader("📊 Въведени параметри:")
    st.write(pd.DataFrame({
        "Параметър": ["Ed", "Ee", "Ei", "D", "Ee / Ei", "Ed / Ei"],
        "Стойност": [
            Ed,
            Ee,
            Ei,
            d_value,
            round(EeEi, 3),
            round(EdEi, 3),
        ]
    }))

    if st.button("Изчисли h"):
        h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)

        if h_result is None:
            st.warning("❗ Неуспешно намиране на h — точката е извън обхвата.")
        else:
            st.success(f"✅ Изчислено: h = {h_result:.2f} cm (
