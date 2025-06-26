import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

# CSS: Задаване на максимална ширина
st.markdown("""
    <style>
    .block-container { max-width: 1000px; padding-left: 2rem; padding-right: 2rem; }
    .next-button-container { display: flex; justify-content: flex-end; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Зареждане на данни
@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={"E1_over_E2": "Ed_over_Ei", "Eeq_over_E2": "Ee_over_Ei"})
    return df

data = load_data()

# Сесийни променливи
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 1
if "results" not in st.session_state:
    st.session_state.results = []

# Начални входни параметри (първоначални)
if "d_value" not in st.session_state:
    st.session_state.d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34])
if "axle_load" not in st.session_state:
    st.session_state.axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])
if "num_layers" not in st.session_state:
    st.session_state.num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

# Данни за текущия пласт
layer = st.session_state.current_layer
d_value = st.session_state.d_value

st.title("Оразмеряване на пътна конструкция")
st.subheader(f"Въведете данни за оразмеряване - Пласт {layer}")
st.markdown(f"**Стойност D за пласт {layer}:** {d_value} cm")

Ee = st.number_input("Въведете стойност за Ee (MPa):", min_value=0.1, step=0.1, value=2700.0, key=f"Ee_{layer}")
Ei = st.number_input("Въведете стойност за Ei (MPa):", min_value=0.1, step=0.1, value=3000.0, key=f"Ei_{layer}")
h = st.number_input("Въведете дебелина h (cm):", min_value=0.1, step=0.1, value=4.0, key=f"h_{layer}")

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

mode = st.radio("Изберете параметър за отчитане:", ("Ed / Ei",), key=f"mode_{layer}")

if Ei == 0 or d_value == 0:
    st.error("Ei и D не могат да бъдат 0.")
    st.stop()

if mode == "Ed / Ei":
    if st.button("Изчисли Ed", key=f"calc_{layer}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / Ei
            st.success(f"✅ Ed / Ei = {EdEi_point:.3f} ; Ed = {result:.2f} MPa")

            # Запазваме резултата
            st.session_state.results.append({
                "layer": layer, "Ee": Ee, "Ei": Ei, "h": h, "D": d_value,
                "Ed": result, "EdEi": EdEi_point, "EeEi": Ee / Ei
            })

            # Показваме всички предишни правоъгълници
            for r in st.session_state.results:
                st.markdown(f"""
                <div style="
                    font-weight: bold;
                    font-size: 18px;
                    margin-top: 20px;
                    margin-bottom: 8px;
                    color: #004d40;
                ">
                    Пласт {r['layer']}
                </div>
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
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        font-weight: bold;
                        font-size: 18px;
                        color: black;
                    ">
                        Ei = {r['Ei']} MPa
                    </div>
                    <div style="
                        position: absolute;
                        top: -20px;
                        right: 10px;
                        font-size: 14px;
                        color: darkblue;
                        font-weight: bold;
                    ">
                        Ee = {r['Ee']} MPa
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
                        h = {r['h']:.2f} cm
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Показваме графиката
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
            fig.add_trace(go.Scatter(
                x=[hD_point, hD_point],
                y=[y_low, y_high],
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                name="Интерполационна линия"
            ))
            fig.update_layout(xaxis_title="h / D", yaxis_title="Ed / Ei", height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Бутон НАПРЕД – само ако има още пластове
            if st.session_state.current_layer < st.session_state.num_layers:
                with st.container():
                    st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
                    if st.button("➡️ Напред към пласт " + str(layer + 1), key=f"next_{layer}"):
                        st.session_state.current_layer += 1
                        st.experimental_rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("🟢 Всички пластове са въведени.")
