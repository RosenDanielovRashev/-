import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Оразмеряване на пътна конструкция с няколко пластове")

@st.cache_data
def load_data():
    df = pd.read_csv("combined_data.csv")
    df = df.rename(columns={
        "E1_over_E2": "Ed_over_Ei",
        "Eeq_over_E2": "Ee_over_Ei"
    })
    return df

data = load_data()

# Инициализация на session_state
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 1
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 0
if "layers_data" not in st.session_state:
    st.session_state.layers_data = [{} for _ in range(st.session_state.num_layers)]

# Въвеждане на брой пластове
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=st.session_state.num_layers)
if num_layers != st.session_state.num_layers:
    st.session_state.num_layers = num_layers
    if len(st.session_state.layers_data) < num_layers:
        st.session_state.layers_data += [{} for _ in range(num_layers - len(st.session_state.layers_data))]
    elif len(st.session_state.layers_data) > num_layers:
        st.session_state.layers_data = st.session_state.layers_data[:num_layers]
    if st.session_state.current_layer >= num_layers:
        st.session_state.current_layer = num_layers - 1

# Диаметър и осов товар
d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34, 33])
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])

# Навигация между пластовете
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
layer_data = st.session_state.layers_data[layer_idx]

st.subheader(f"Въвеждане на данни за пласт {layer_idx + 1}")

st.markdown("### 🧾 Легенда:")
st.markdown("""
- **Ed** – Модул на еластичност на повърхността под пласта  
- **Ei** – Модул на еластичност на пласта  
- **Ee** – Модул на еластичност на повърхността на пласта  
- **h** – Дебелина на пласта  
- **D** – Диаметър на отпечатък на колелото  
""")

# Въвеждане на стойности с key и стойности от session_state
Ee = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, key=f"Ee_{layer_idx}", value=layer_data.get("Ee", 2700.0))
Ei = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, key=f"Ei_{layer_idx}", value=layer_data.get("Ei", 3000.0))

mode = st.radio("Изберете параметър за отчитане:", ("Ed / Ei", "h / D"), key=f"mode_{layer_idx}")

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

def add_interpolation_line(fig, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso):
    fig.add_trace(go.Scatter(x=[hD_point, hD_point], y=[y_low, y_high],
                             mode='lines', line=dict(color='purple', dash='dash'),
                             name=f"Интерполация Ee/Ei: {low_iso:.2f} - {high_iso:.2f}"))
    fig.add_trace(go.Scatter(x=[hD_point], y=[EdEi_point],
                             mode='markers', marker=dict(color='red', size=12), name='Резултат'))

# MODE: Ed / Ei
if mode == "Ed / Ei":
    h = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, key=f"h_{layer_idx}", value=layer_data.get("h", 4.0))
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)
        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / Ei
            st.success(f"✅ Ed = {result:.2f} MPa (Ed / Ei = {EdEi_point:.3f})")
            st.session_state.layers_data[layer_idx].update({
                "Ee": Ee, "Ei": Ei, "h": h, "Ed": result, "EdEi": EdEi_point, "mode": mode
            })

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"], y=group_sorted["Ed_over_Ei"],
                    mode='lines', name=f"Ee/Ei = {value:.2f}"
                ))
            add_interpolation_line(fig, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso)
            fig.update_layout(title="Ed / Ei в зависимост от h / D", xaxis_title="h / D", yaxis_title="Ed / Ei")
            st.plotly_chart(fig, use_container_width=True)

# MODE: h / D
elif mode == "h / D":
    Ed = st.number_input("Ed (MPa):", min_value=0.1, step=0.1, key=f"Ed_{layer_idx}", value=layer_data.get("Ed", 50.0))
    if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)
        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            st.success(f"✅ h = {result:.2f} cm (h / D = {hD_point:.3f})")
            st.session_state.layers_data[layer_idx].update({
                "Ee": Ee, "Ei": Ei, "h": result, "Ed": Ed, "mode": mode
            })

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"], y=group_sorted["Ed_over_Ei"],
                    mode='lines', name=f"Ee/Ei = {value:.2f}"
                ))
            add_interpolation_line(fig, hD_point, Ed / Ei, y_low, y_high, low_iso, high_iso)
            fig.update_layout(title="Ed / Ei в зависимост от h / D", xaxis_title="h / D", yaxis_title="Ed / Ei")
            st.plotly_chart(fig, use_container_width=True)

# Финален преглед
st.markdown("---")
st.header("Резултати за всички пластове")

for i, layer in enumerate(st.session_state.layers_data):
    Ee = layer.get('Ee', '-')
    Ei = layer.get('Ei', '-')
    Ed = layer.get('Ed', '-')
    h_val = layer.get('h', '-')
    Ed_display = round(Ed) if isinstance(Ed, (float, int)) else Ed
    h_result = h_val if isinstance(h_val, (float, int)) else 0.0

    st.markdown(f"<b>Пласт {i + 1}</b>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="position: relative; width: 400px; height: 60px; background-color: #add8e6;
                border: 2px solid black; border-radius: 6px; margin: 10px auto 30px auto; padding: 10px;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    font-weight: bold; font-size: 18px;">Ei = {Ei} MPa</div>
        <div style="position: absolute; top: -20px; right: 10px; font-size: 14px; font-weight: bold;
                    color: darkblue;">Ee = {Ee} MPa</div>
        <div style="position: absolute; bottom: -20px; right: 10px; font-size: 14px;
                    font-weight: bold; color: green;">Ed = {Ed_display} MPa</div>
        <div style="position: absolute; top: 50%; left: 8px; transform: translateY(-50%);
                    font-size: 14px; font-weight: bold;">h = {h_result:.2f} cm</div>
    </div>
    """, unsafe_allow_html=True)

# Преход към първата страница
st.page_link("pages/second.py", label="Към Опън в покритието", icon="📄")
