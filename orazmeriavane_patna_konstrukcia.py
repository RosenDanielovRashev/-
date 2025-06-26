import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

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

num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

mode = st.radio("Изберете параметър за отчитане:", ("Ed / Ei", "h / D"))

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

for layer_idx in range(1, num_layers + 1):
    st.header(f"Пласт {layer_idx}")

    Ee = st.number_input(f"Ee (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=2700.0, key=f"Ee_{layer_idx}")
    Ei = st.number_input(f"Ei (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=3000.0, key=f"Ei_{layer_idx}")

    if mode == "Ed / Ei":
        D = st.number_input(f"D (cm) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=32.04, key=f"D_{layer_idx}")
        h = st.number_input(f"Дебелина h (cm) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=4.0, key=f"h_{layer_idx}")

        if st.button(f"Изчисли Ed за пласт {layer_idx}", key=f"btn_Ed_{layer_idx}"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, D, Ee, Ei)

            if result is None:
                st.warning("❗ Точката е извън обхвата на наличните изолинии.")
            else:
                EdEi_point = result / Ei
                st.success(f"Ed / Ei = {EdEi_point:.3f}, Ed = {result:.2f} MPa")
                st.info(f"Интерполация между Ee / Ei = {low_iso:.3f} и {high_iso:.3f}")

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
                    marker=dict(color='red', size=10),
                    name="Твоята точка"
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
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

    else:  # mode == "h / D"
        D = st.number_input(f"D (cm) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=32.04, key=f"D_{layer_idx}")
        Ed = st.number_input(f"Ed (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=50.0, key=f"Ed_{layer_idx}")

        if st.button(f"Изчисли h за пласт {layer_idx}", key=f"btn_h_{layer_idx}"):
            h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, D, Ee, Ei)

            if h_result is None:
                st.warning("❗ Неуспешно намиране на h — точката е извън обхвата.")
            else:
                st.success(f"h = {h_result:.2f} cm (h / D = {hD_point:.3f})")
                st.info(f"Интерполация между Ee / Ei = {low_iso:.3f} и {high_iso:.3f}")

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
                EdEi_point = Ed / Ei
                fig.add_trace(go.Scatter(
                    x=[hD_point],
                    y=[EdEi_point],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name="Твоята точка"
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
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
