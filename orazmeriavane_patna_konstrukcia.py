import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .block-container {
        max-width: 1000px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={"E1_over_E2": "Ed_over_Ei", "Eeq_over_E2": "Ee_over_Ei"})
    return df

data = load_data()

# Инициализация на сесийно състояние
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 1
if "layers_data" not in st.session_state:
    st.session_state.layers_data = {}
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 1

# Начални входни стойности (само при първо стартиране)
if st.session_state.current_layer == 1:
    st.session_state.num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34])
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])

layer = st.session_state.current_layer
st.subheader(f"Въведете данни за оразмеряване - Пласт {layer}")

Ee = st.number_input("Ee (MPa)", min_value=0.1, step=0.1, value=2700.0, key=f"Ee_{layer}")
Ei = st.number_input("Ei (MPa)", min_value=0.1, step=0.1, value=3000.0, key=f"Ei_{layer}")
h = st.number_input("h (cm)", min_value=0.1, step=0.1, value=4.0, key=f"h_{layer}")

# Функции за изчисления
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

mode = st.radio("Изберете параметър за отчитане:", ("Ed / Ei", "h / D"), key=f"mode_{layer}")

if mode == "Ed / Ei":
    if st.button("Изчисли Ed", key=f"btn_Ed_{layer}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)
        if result:
            EdEi_point = result / Ei
            st.success(f"✅ Изчислено: Ed = {result:.2f} MPa (Ed / Ei = {EdEi_point:.3f})")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и {high_iso:.3f}")
            # Графика
            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                fig.add_trace(go.Scatter(
                    x=group.sort_values("h_over_D")["h_over_D"],
                    y=group.sort_values("h_over_D")["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee / Ei = {value:.2f}"
                ))
            fig.add_trace(go.Scatter(
                x=[hD_point], y=[EdEi_point],
                mode='markers', marker=dict(size=10, color="red"), name="Твоята точка"
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Визуализация на правоъгълник
            st.markdown(f"""
            <div style="position:relative; width:400px; height:60px; background:#add8e6; border:2px solid black; border-radius:6px; margin:20px auto; padding:10px;">
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-weight:bold;">Ei = {Ei} MPa</div>
                <div style="position:absolute; top:-20px; right:10px; font-weight:bold; color:darkblue;">Ee = {Ee} MPa</div>
                <div style="position:absolute; top:50%; left:8px; transform:translateY(-50%); font-weight:bold;">h = {h:.2f} cm</div>
            </div>
            """, unsafe_allow_html=True)

            # Запазване на данни за пласта
            st.session_state.layers_data[layer] = {
                "Ee": Ee, "Ei": Ei, "h": h,
                "Ed": result, "d_value": d_value,
                "axle_load": axle_load, "mode": mode
            }

elif mode == "h / D":
    Ed = st.number_input("Въведете стойност за Ed (MPa)", min_value=0.1, value=500.0, key=f"Ed_input_{layer}")
    if st.button("Изчисли h", key=f"btn_h_{layer}"):
        h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)
        if h_result:
            st.success(f"✅ Изчислено: h = {h_result:.2f} cm (h / D = {hD_point:.3f})")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и {high_iso:.3f}")

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                fig.add_trace(go.Scatter(
                    x=group.sort_values("h_over_D")["h_over_D"],
                    y=group.sort_values("h_over_D")["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee / Ei = {value:.2f}"
                ))
            fig.add_trace(go.Scatter(
                x=[hD_point], y=[Ed / Ei],
                mode='markers', marker=dict(size=10, color="red"), name="Твоята точка"
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div style="position:relative; width:400px; height:60px; background:#add8e6; border:2px solid black; border-radius:6px; margin:20px auto; padding:10px;">
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-weight:bold;">Ei = {Ei} MPa</div>
                <div style="position:absolute; top:-20px; right:10px; font-weight:bold; color:darkblue;">Ee = {Ee} MPa</div>
                <div style="position:absolute; top:50%; left:8px; transform:translateY(-50%); font-weight:bold;">h = {h_result:.2f} cm</div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.layers_data[layer] = {
                "Ee": Ee, "Ei": Ei, "h": h_result,
                "Ed": Ed, "d_value": d_value,
                "axle_load": axle_load, "mode": mode
            }

# --- Бутон "Напред" ---
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("➡️ Напред", key=f"next_{layer}"):
        if layer < st.session_state.num_layers:
            st.session_state.current_layer += 1
        else:
            st.success("✅ Въведени са данни за всички пластове.")
            st.markdown("### 🧾 Обобщение:")
            summary_df = pd.DataFrame.from_dict(st.session_state.layers_data, orient="index")
            st.dataframe(summary_df)
