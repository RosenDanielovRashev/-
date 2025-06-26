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

# Сесийни променливи
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 1
if "results" not in st.session_state:
    st.session_state.results = []
if "layer_index" not in st.session_state:
    st.session_state.layer_index = 0
if "d_value" not in st.session_state:
    st.session_state.d_value = 32.04
if "axle_load" not in st.session_state:
    st.session_state.axle_load = 100

st.title("Оразмеряване на пътна конструкция")

# --- Меню за въвеждане само при първия пласт ---
if st.session_state.layer_index == 0 and not st.session_state.results:
    st.session_state.num_layers = st.number_input("Въведи брой пластове:", min_value=1, step=1, value=2)
    st.session_state.d_value = st.selectbox("D стойност (cm):", [32.04, 34])
    st.session_state.axle_load = st.selectbox("Осов товар (kN):", [100, 115])

# Текущ индекс на слоя
idx = st.session_state.layer_index

# Създаване на празен речник, ако още не е добавен
while len(st.session_state.results) <= idx:
    st.session_state.results.append({})

# Въвеждане на данни за слоя
st.header(f"Данни за пласт {idx + 1} от {st.session_state.num_layers}")
res = st.session_state.results[idx]

Ee = st.number_input("Ee (MPa):", value=res.get("Ee", 2700.0), key=f"Ee_{idx}")
Ei = st.number_input("Ei (MPa):", value=res.get("Ei", 3000.0), key=f"Ei_{idx}")
h = st.number_input("Дебелина h (cm):", value=res.get("h", 4.0), key=f"h_{idx}")

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

# Изчисление
if st.button("Изчисли Ed"):
    Ed, hD, EdEi = compute_Ed(h, st.session_state.d_value, Ee, Ei)
    if Ed is None:
        st.error("❌ Данните са извън допустимия обхват за интерполация.")
    else:
        st.success(f"✅ Ed = {Ed:.2f} MPa ; Ed / Ei = {EdEi:.3f}")

        st.session_state.results[idx] = {
            "Ee": Ee, "Ei": Ei, "h": h,
            "Ed": Ed, "hD": hD, "EdEi": EdEi,
            "EeEi": Ee / Ei
        }

# Навигационни бутони
col1, col2 = st.columns([1, 1])
with col1:
    if idx > 0:
        if st.button("⬅️ Назад"):
            st.session_state.layer_index -= 1
with col2:
    if idx < st.session_state.num_layers - 1:
        if st.button("➡️ Напред"):
            st.session_state.layer_index += 1

# --- Визуализация на всички попълнени пластове ---
st.markdown("---")
st.subheader("🧱 Слоеве въведени до момента:")
for i, r in enumerate(st.session_state.results):
    if not r or "Ed" not in r:
        continue
    st.markdown(f"""
    <div style="
        background-color: #d0f0c0;
        padding: 15px;
        border-left: 6px solid green;
        border-radius: 10px;
        margin-bottom: 15px;
        font-family: sans-serif;">
        <b>Пласт {i + 1}</b><br>
        Ee = {r['Ee']} MPa<br>
        Ei = {r['Ei']} MPa<br>
        h = {r['h']} cm<br>
        Ed = {r['Ed']:.2f} MPa<br>
        Ed / Ei = {r['EdEi']:.3f}
    </div>
    """, unsafe_allow_html=True)
