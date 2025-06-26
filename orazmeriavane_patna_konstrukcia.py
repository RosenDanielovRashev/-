import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={"E1_over_E2": "Ed_over_Ei", "Eeq_over_E2": "Ee_over_Ei"})
    return df

data = load_data()

# Инициализация на сесията
if "layer_index" not in st.session_state:
    st.session_state.layer_index = 0  # започваме от пласт 0
if "num_layers" not in st.session_state:
    st.session_state.num_layers = st.number_input("Брой пластове:", min_value=1, step=1, value=2)
if "results" not in st.session_state:
    st.session_state.results = [{} for _ in range(st.session_state.num_layers)]

# Общи входни данни
if "d_value" not in st.session_state:
    st.session_state.d_value = st.selectbox("D стойност (cm):", [32.04, 34])
if "axle_load" not in st.session_state:
    st.session_state.axle_load = st.selectbox("Осов товар (kN):", [100, 115])

# Данни за текущия пласт
idx = st.session_state.layer_index
result = st.session_state.results[idx]
d_value = st.session_state.d_value

st.title("Оразмеряване на пътна конструкция")
st.subheader(f"Данни за пласт {idx + 1} от {st.session_state.num_layers}")

Ee = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, value=result.get("Ee", 2700.0), key=f"Ee_{idx}")
Ei = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, value=result.get("Ei", 3000.0), key=f"Ei_{idx}")
h = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=result.get("h", 4.0), key=f"h_{idx}")

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

        return ed_over_ei * Ei, hD, ed_over_ei
    return None, None, None

if st.button("Изчисли Ed"):
    Ed, hD_point, EdEi_point = compute_Ed(h, d_value, Ee, Ei)
    if Ed is None:
        st.warning("❗ Извън обхват на интерполация.")
    else:
        st.success(f"✅ Ed = {Ed:.2f} MPa ; Ed / Ei = {EdEi_point:.3f}")
        st.session_state.results[idx] = {
            "Ee": Ee,
            "Ei": Ei,
            "h": h,
            "Ed": Ed,
            "EdEi": EdEi_point,
            "EeEi": Ee / Ei,
            "hD": hD_point
        }

        # Показване на графиката
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
        fig.update_layout(xaxis_title="h / D", yaxis_title="Ed / Ei", height=600)
        st.plotly_chart(fig, use_container_width=True)

# Навигация
col1, col2 = st.columns(2)
with col1:
    if st.session_state.layer_index > 0:
        if st.button("⬅️ Назад"):
            st.session_state.layer_index -= 1
            st.experimental_rerun()

with col2:
    if st.session_state.layer_index < st.session_state.num_layers - 1:
        if st.button("➡️ Напред"):
            st.session_state.layer_index += 1
            st.experimental_rerun()

# Резултати за всички досега обработени пластове
st.markdown("---")
st.subheader("🧱 Обобщение на пластове")
for i, r in enumerate(st.session_state.results):
    if "Ei" not in r:
        continue
    st.markdown(f"""
    <div style="
        width: 400px;
        background-color: #e3f2fd;
        border: 2px solid #1565c0;
        border-radius: 8px;
        margin-bottom: 20px;
        padding: 12px;
        font-size: 16px;
        position: relative;">
        <b>Пласт {i + 1}</b><br>
        Ei = {r['Ei']} MPa<br>
        Ee = {r['Ee']} MPa<br>
        h = {r['h']} cm<br>
        Ed = {r['Ed']:.2f} MPa<br>
        Ed / Ei = {r['EdEi']:.3f}
    </div>
    """, unsafe_allow_html=True)
