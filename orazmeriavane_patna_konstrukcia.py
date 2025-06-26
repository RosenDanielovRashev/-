import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

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

if "num_layers" not in st.session_state:
    st.session_state.num_layers = 1
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 0
if "layers_data" not in st.session_state:
    st.session_state.layers_data = [{}]

st.title("Оразмеряване на пътна конструкция с няколко пластове")

num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=st.session_state.num_layers)
if num_layers != st.session_state.num_layers:
    st.session_state.num_layers = num_layers
    if len(st.session_state.layers_data) < num_layers:
        st.session_state.layers_data += [{} for _ in range(num_layers - len(st.session_state.layers_data))]
    elif len(st.session_state.layers_data) > num_layers:
        st.session_state.layers_data = st.session_state.layers_data[:num_layers]
    if st.session_state.current_layer >= num_layers:
        st.session_state.current_layer = num_layers - 1

d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34])
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("⬅️ Предишен пласт"):
        if st.session_state.current_layer > 0:
            st.session_state.current_layer -= 1
with col3:
    if st.button("Следващ пласт ➡️"):
        if st.session_state.current_layer < st.session_state.num_layers - 1:
            st.session_state.current_layer += 1

layer_idx = st.session_state.current_layer
st.subheader(f"Въвеждане на данни за пласт {layer_idx + 1}")

layer_data = st.session_state.layers_data[layer_idx]

Ee = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ee", 2700.0), key=f"Ee_{layer_idx}")
Ei = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ei", 3000.0), key=f"Ei_{layer_idx}")

mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    key=f"mode_{layer_idx}"
)

if mode == "Ed / Ei":
    h = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=layer_data.get("h", 4.0), key=f"h_{layer_idx}")
else:
    h = layer_data.get("h", None)
    if h is not None:
        st.write(f"Дебелина h (cm): {h:.2f}")
    else:
        st.write("Дебелина h (cm): -")

def compute_Ed(h, D, Ee, Ei):
    hD = h / D
    EeEi = Ee / Ei
    tol = 1e-3
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
    tol = 1e-3
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

if mode == "Ed / Ei":
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / Ei
            st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            st.session_state.layers_data[layer_idx].update({
                "Ee": Ee,
                "Ei": Ei,
                "h": h,
                "Ed": result,
                "EdEi": EdEi_point,
                "mode": mode
            })

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"],
                    y=group_sorted["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee/Ei = {value:.2f}"
                ))
            fig.add_trace(go.Scatter(
                x=[hD_point],
                y=[result / Ei],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Резултат'
            ))
            fig.update_layout(
                title="Ed / Ei в зависимост от h / D",
                xaxis_title="h / D",
                yaxis_title="Ed / Ei",
                legend_title="Изолинии"
            )
            st.plotly_chart(fig, use_container_width=True)

elif mode == "h / D":
    Ed = st.number_input("Ed (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ed", 50.0), key=f"Ed_{layer_idx}")
    if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)
        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            st.success(f"✅ Изчислено: h = {result:.2f} cm  \nh / D = {hD_point:.3f}")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            st.session_state.layers_data[layer_idx].update({
                "Ee": Ee,
                "Ei": Ei,
                "h": result,
                "Ed": Ed,
                "mode": mode
            })

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"],
                    y=group_sorted["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee/Ei = {value:.2f}"
                ))
            fig.add_trace(go.Scatter(
                x=[hD_point],
                y=[Ed / Ei],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Резултат'
            ))
            fig.update_layout(
                title="Ed / Ei в зависимост от h / D",
                xaxis_title="h / D",
                yaxis_title="Ed / Ei",
                legend_title="Изолинии"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Въведени пластове")

for i, layer in enumerate(st.session_state.layers_data):
    Ee = layer.get('Ee', '-')
    Ei = layer.get('Ei', '-')
    Ed = layer.get('Ed', '-')  # Ново
    h_val = layer.get('h', '-')
    h_result = h_val if isinstance(h_val, float) else 0.0

    st.markdown(f"<b>Пласт {i + 1}</b>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 400px;
            height: 60px;
            background-color: #add8e6;
            border: 2px solid black;
            border-radius: 6px;
            margin: 10px auto 30px auto;
            padding: 10px;
            font-family: Arial, sans-serif;
        ">
            <!-- Ei в центъра -->
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
            <!-- Ee горе вдясно -->
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
            <!-- Ed долу вдясно -->
            <div style="
                position: absolute;
                bottom: -20px;
                right: 10px;
                font-size: 14px;
                color: green;
                font-weight: bold;
            ">
                Ed = {Ed} MPa
            </div>
            <!-- h вляво -->
            <div style="
                position: absolute;
                top: 50%;
                left: 8px;
                transform: translateY(-50%);
                font-size: 14px;
                color: black;
                font-weight: bold;
            ">
                h = {h_result:.2f} cm
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
