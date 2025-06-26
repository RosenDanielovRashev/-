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

st.title("Оразмеряване на пътна конструкция")

d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34])
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

st.subheader("Въведете данни за оразмеряване - Пласт 1")
st.markdown(f"**Стойност D за пласт 1:** {d_value} cm")

Ee = st.number_input("Въведете стойност за Ee (MPa):", min_value=0.1, step=0.1, value=2700.0)
Ei = st.number_input("Въведете стойност за Ei (MPa):", min_value=0.1, step=0.1, value=3000.0)
h = st.number_input("Въведете дебелина h (cm):", min_value=0.1, step=0.1, value=4.0)

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

mode = st.radio("Изберете параметър за отчитане:", ("Ed / Ei", "h / D"))

if Ei == 0 or d_value == 0:
    st.error("Ei и D не могат да бъдат 0.")
    st.stop()

layer_results = []

for layer in range(1, num_layers + 1):
    st.markdown("---")
    st.subheader(f"Пласт {layer}")

    Ee = st.number_input(f"Ee (MPa) – Пласт {layer}", min_value=0.1, step=0.1, value=2700.0, key=f"ee_{layer}")
    Ei = st.number_input(f"Ei (MPa) – Пласт {layer}", min_value=0.1, step=0.1, value=3000.0, key=f"ei_{layer}")

    if mode == "Ed / Ei":
        h = st.number_input(f"h (cm) – Пласт {layer}", min_value=0.1, step=0.1, value=4.0, key=f"h_{layer}")
        if st.button(f"Изчисли Ed – Пласт {layer}"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)
            if result:
                st.success(f"✅ Пласт {layer}: Ed = {result:.2f} MPa")
                layer_results.append((layer, Ei, Ee, h))
            else:
                st.warning(f"❗ Пласт {layer}: Точката е извън обхвата.")

    else:
        Ed = st.number_input(f"Ed (MPa) – Пласт {layer}", value=500.0, key=f"ed_{layer}")
        if st.button(f"Изчисли h – Пласт {layer}"):
            h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)
            if h_result:
                st.success(f"✅ Пласт {layer}: h = {h_result:.2f} cm")
                layer_results.append((layer, Ei, Ee, h_result))
            else:
                st.warning(f"❗ Пласт {layer}: Точката е извън обхвата.")

if layer_results:
    st.markdown("---")
    st.markdown("### 📋 Обобщение на всички пластове")
    df_summary = pd.DataFrame(layer_results, columns=["Пласт", "Ei (MPa)", "Ee (MPa)", "h (cm)"])
    st.table(df_summary)

st.markdown("<br><hr><center>© 2025 Инженерен калкулатор за пътни конструкции</center>", unsafe_allow_html=True)
