import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={
        "E1_over_E2": "Ed_over_Ei",
        "Eeq_over_E2": "Ee_over_Ei"
    })
    return df

data = load_data()

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

# --- Streamlit UI ---

st.title("📐 Калкулатор: Метод на Иванов (multi-layer версия)")

# Инициализация на слоевете
if "layers_data" not in st.session_state:
    st.session_state.layers_data = [{}]  # старт с един слой

# Функция за добавяне и премахване на слой
def add_layer():
    st.session_state.layers_data.append({})
def remove_layer():
    if len(st.session_state.layers_data) > 1:
        st.session_state.layers_data.pop()

# Управление на слоеве
cols = st.columns([1,1,6])
if cols[0].button("➕ Добави слой"):
    add_layer()
if cols[1].button("➖ Премахни слой"):
    remove_layer()

# Входни глобални параметри (за всички слоеве)
Ee = st.number_input("Ee (MPa)", value=2700.0, step=10.0)
Ei = st.number_input("Ei (MPa)", value=3000.0, step=10.0)
D = st.selectbox("D (cm)", options=[34.0, 32.04], index=1)

if Ei == 0 or D == 0:
    st.error("Ei и D не могат да бъдат 0.")
    st.stop()

# Покажи слоевете
for layer_idx, layer_data in enumerate(st.session_state.layers_data):
    st.markdown(f"---\n### Слой {layer_idx + 1}")

    mode = st.radio(
        "Изберете режим на изчисление за този слой:",
        ("Ed / Ei", "h / D"),
        index=0 if layer_data.get("mode") != "h / D" else 1,
        key=f"mode_{layer_idx}"
    )
    st.session_state.layers_data[layer_idx]["mode"] = mode

    if mode == "Ed / Ei":
        h = st.number_input(
            "Дебелина h (cm):",
            min_value=0.1,
            step=0.1,
            value=layer_data.get("h", 4.0),
            key=f"h_{layer_idx}"
        )
        st.session_state.layers_data[layer_idx]["h"] = h

        if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, D, Ee, Ei)
            if result is None:
                st.warning("❗ Точката е извън обхвата на наличните изолинии.")
            else:
                EdEi_point = result / Ei
                st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
                st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

                # Записване
                st.session_state.layers_data[layer_idx].update({
                    "Ee": Ee,
                    "Ei": Ei,
                    "Ed": result,
                    "mode": mode
                })

                # Графика
                fig = go.Figure()
                for value, group in data.groupby("Ee_over_Ei"):
                    group_sorted = group.sort_values("h_over_D")
                    fig.add_trace(go.Scatter(
                        x=group_sorted["h_over_D"],
                        y=group_sorted["Ed_over_Ei"],
                        mode='lines',
                        name=f"Ee / Ei = {value:.2f}"
                    ))
                fig.add_trace(go.Scatter(
                    x=[hD_point],
                    y=[EdEi_point],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Твоята точка'
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
                    title="Интерактивна диаграма на изолинии (Ee / Ei)",
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    legend=dict(orientation="h", y=-0.3),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

    else:  # mode == "h / D"
        Ed = st.number_input(
            "Ed (MPa):",
            min_value=0.1,
            step=0.1,
            value=layer_data.get("Ed", 50.0),
            key=f"Ed_{layer_idx}"
        )

        if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
            h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, D, Ee, Ei)
            if h_result is None:
                st.warning("❗ Неуспешно намиране на h — точката е извън обхвата.")
            else:
                st.success(f"✅ Изчислено: h = {h_result:.2f} cm  (h / D = {hD_point:.3f})")
                st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

                # Записване
                st.session_state.layers_data[layer_idx].update({
                    "Ee": Ee,
                    "Ei": Ei,
                    "h": h_result,
                    "Ed": Ed,
                    "mode": mode
                })

                # Графика
                fig = go.Figure()
                for value, group in data.groupby("Ee_over_Ei"):
                    group_sorted = group.sort_values("h_over_D")
                    fig.add_trace(go.Scatter(
                        x=group_sorted["h_over_D"],
                        y=group_sorted["Ed_over_Ei"],
                        mode='lines',
                        name=f"Ee / Ei = {value:.2f}"
                    ))
                fig.add_trace(go.Scatter(
                    x=[hD_point],
                    y=[Ed / Ei],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name='Твоята точка'
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
                    title="Интерактивна диаграма на изолинии (Ee / Ei)",
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    legend=dict(orientation="h", y=-0.3),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
