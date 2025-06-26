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

# --- Въвеждане на характеристики ---
st.title("Оразмеряване на пътна конструкция")

d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34])

axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])

num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=1)

# Избор на режим за изчисление
mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D")
)

# Функции за изчисление (от твоя код с номограмата)
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
    st.subheader(f"Въведете данни за оразмеряване - Пласт {layer_idx}")

    # За всеки пласт показваме входните полета в зависимост от режима
    if mode == "Ed / Ei":
        Ee = st.number_input(f"Ee (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=2700.0, key=f"Ee_{layer_idx}")
        Ei = st.number_input(f"Ei (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=3000.0, key=f"Ei_{layer_idx}")
        h = st.number_input(f"Дебелина h (cm) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=4.0, key=f"h_{layer_idx}")

        st.markdown(f"**Стойност D за пласт {layer_idx}:** {d_value} cm")

        if st.button(f"Изчисли Ed за пласт {layer_idx}", key=f"calc_Ed_{layer_idx}"):
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

                # Текст преди правоъгълника, само в режим Ed / Ei
                st.markdown(
                    f"""
                    <div style="
                        font-weight: bold;
                        font-size: 18px;
                        margin-top: 20px;
                        margin-bottom: 8px;
                        color: #004d40;
                    ">
                        Пласт {layer_idx}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Добавяне на хоризонтален продълговат правоъгълник под графиката
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
                        <!-- h в ляво центрирано вертикално -->
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

    else:  # режим "h / D"
        Ee = st.number_input(f"Ee (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=2700.0, key=f"Ee_{layer_idx}")
        Ei = st.number_input(f"Ei (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=3000.0, key=f"Ei_{layer_idx}")
        Ed = st.number_input(f"Ed (MPa) за пласт {layer_idx}:", min_value=0.1, step=0.1, value=50.0, key=f"Ed_{layer_idx}")

        st.markdown(f"**Стойност D за пласт {layer_idx}:** {d_value} cm")

        if st.button(f"Изчисли h за пласт {layer_idx}", key=f"calc_h_{layer_idx}"):
            h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)

            if h_result is None:
                st.warning("❗ Неуспешно намиране на h — точката е извън обхвата.")
            else:
                st.success(f"✅ Изчислено: h = {h_result:.2f} cm (h / D = {hD_point:.3f})")
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
                EdEi_point = Ed / Ei
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

                # Текст преди правоъгълника, само в режим Ed / Ei няма тук

                # Добавяне на хоризонтален продълговат правоъгълник под графиката
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
                        <!-- Ed в ляво центрирано вертикално -->
                        <div style="
                            position: absolute;
                            top: 50%;
                            left: 8px;
                            transform: translateY(-50%);
                            font-size: 14px;
                            color: black;
                            font-weight: bold;
                        ">
                            Ed = {Ed:.2f} MPa
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
