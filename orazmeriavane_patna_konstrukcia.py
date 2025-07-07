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
    .layer-card {
        position: relative;
        width: 400px;
        height: 80px;
        background-color: #e0f7fa;
        border: 2px solid #26c6da;
        border-radius: 8px;
        margin: 15px auto 40px auto;
        padding: 10px;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
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

# Инициализация на session state
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 1
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 0
if "layers_data" not in st.session_state:
    st.session_state.layers_data = [{"Ee": 2700.0, "Ei": 3000.0}]
if "axle_load" not in st.session_state:
    st.session_state.axle_load = 100
if "final_D" not in st.session_state:
    st.session_state.final_D = 32.04

st.title("Оразмеряване на пътна конструкция с няколко пластове")

# Избор на брой пластове
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=st.session_state.num_layers)
if num_layers != st.session_state.num_layers:
    st.session_state.num_layers = num_layers
    if len(st.session_state.layers_data) < num_layers:
        for i in range(len(st.session_state.layers_data), num_layers):
            # За пластовете над първия, Ee се взима от Ed на предишния пласт
            prev_ed = st.session_state.layers_data[i-1].get("Ed", 2700.0)
            st.session_state.layers_data.append({"Ee": prev_ed, "Ei": 3000.0})
    elif len(st.session_state.layers_data) > num_layers:
        st.session_state.layers_data = st.session_state.layers_data[:num_layers]
    if st.session_state.current_layer >= num_layers:
        st.session_state.current_layer = num_layers - 1

# Избор на параметри
d_value = st.selectbox("Изберете стойност за D (cm):", options=[32.04, 34, 33])
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=[100, 115])
st.session_state.axle_load = axle_load
st.session_state.final_D = d_value

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

# Показване на текущ пласт
layer_idx = st.session_state.current_layer
st.subheader(f"Въвеждане на данни за пласт {layer_idx + 1}")

# Легенда
st.markdown("### 🧾 Легенда:")
st.markdown("""
- **Ed** – Модул на еластичност на повърхността под пласта  
- **Ei** – Модул на еластичност на пласта  
- **Ee** – Модул на еластичност на повърхността на пласта  
- **h** – Дебелина на пласта  
- **D** – Диаметър на отпечатък на колелото  
""")

# Въвеждане на параметри за пласта
layer_data = st.session_state.layers_data[layer_idx]

# Автоматично определяне на Ee за пластове над първия
if layer_idx > 0:
    prev_layer = st.session_state.layers_data[layer_idx - 1]
    if "Ed" in prev_layer:
        # Автоматично задаване на Ee от Ed на предишния пласт
        layer_data["Ee"] = prev_layer["Ed"]
        st.info(f"ℹ️ Ee е автоматично зададен от Ed на предишния пласт: {prev_layer['Ed']:.2f} MPa")
    else:
        st.warning("⚠️ Предишният пласт все още не е изчислен. Моля, изчислете предишния пласт първо.")

# Показване на Ee (само за четене за пластове над първия)
if layer_idx == 0:
    Ee = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ee", 2700.0), key=f"Ee_{layer_idx}")
    layer_data["Ee"] = Ee
else:
    Ee = layer_data.get("Ee", 2700.0)
    st.write(f"**Ee (автоматично от предишен пласт):** {Ee:.2f} MPa")

Ei = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ei", 3000.0), key=f"Ei_{layer_idx}")
layer_data["Ei"] = Ei

mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    key=f"mode_{layer_idx}"
)

# Функции за изчисления
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
    # Линия между двете изолини на фиксирано hD_point
    fig.add_trace(go.Scatter(
        x=[hD_point, hD_point],
        y=[y_low, y_high],
        mode='lines',
        line=dict(color='purple', dash='dash'),
        name=f"Интерполация Ee/Ei: {low_iso:.2f} - {high_iso:.2f}"
    ))
    # Точка с резултат
    fig.add_trace(go.Scatter(
        x=[hD_point],
        y=[EdEi_point],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Резултат'
    ))

# Обработка на изчисленията
if mode == "Ed / Ei":
    h = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=layer_data.get("h", 4.0), key=f"h_{layer_idx}")
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h, d_value, Ee, Ei)

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / Ei
            st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            layer_data.update({
                "Ee": Ee,
                "Ei": Ei,
                "h": h,
                "Ed": result,
                "EdEi": EdEi_point,
                "mode": mode
            })

            # Ако има следващ пласт, обновяваме неговото Ee
            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = result
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {result:.2f} MPa")

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"],
                    y=group_sorted["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee/Ei = {value:.2f}"
                ))

            add_interpolation_line(fig, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso)

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

            layer_data.update({
                "Ee": Ee,
                "Ei": Ei,
                "h": result,
                "Ed": Ed,
                "mode": mode
            })

            # Ако има следващ пласт, обновяваме неговото Ee
            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = Ed
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {Ed:.2f} MPa")

            fig = go.Figure()
            for value, group in data.groupby("Ee_over_Ei"):
                group_sorted = group.sort_values("h_over_D")
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"],
                    y=group_sorted["Ed_over_Ei"],
                    mode='lines',
                    name=f"Ee/Ei = {value:.2f}"
                ))

            add_interpolation_line(fig, hD_point, Ed / Ei, y_low, y_high, low_iso, high_iso)

            fig.update_layout(
                title="Ed / Ei в зависимост от h / D",
                xaxis_title="h / D",
                yaxis_title="Ed / Ei",
                legend_title="Изолинии"
            )
            st.plotly_chart(fig, use_container_width=True)

# Визуализация на резултатите
st.markdown("---")
st.header("Резултати за всички пластове")

all_data_ready = True
for i, layer in enumerate(st.session_state.layers_data):
    Ee_val = layer.get('Ee', '-')
    Ei_val = layer.get('Ei', '-')
    Ed_val = layer.get('Ed', '-')
    h_val = layer.get('h', '-')
    
    # Проверка за пълнота на данните
    if any(val == '-' for val in [Ee_val, Ei_val, Ed_val, h_val]):
        all_data_ready = False
    
    # HTML за визуализация на пласта
    st.markdown(f"""
    <div class="layer-card">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    font-weight: bold; font-size: 18px; color: #006064;">
            Ei = {Ei_val if Ei_val == '-' else f'{Ei_val:.2f}'} MPa
        </div>
        <div style="position: absolute; top: -20px; right: 10px; font-size: 14px; 
                    color: #00838f; font-weight: bold;">
            Ee = {Ee_val if Ee_val == '-' else f'{Ee_val:.2f}'} MPa
        </div>
        <div style="position: absolute; bottom: -20px; right: 10px; font-size: 14px; 
                    color: #2e7d32; font-weight: bold;">
            Ed = {Ed_val if Ed_val == '-' else f'{round(Ed_val)}'} MPa
        </div>
        <div style="position: absolute; top: 50%; left: 8px; transform: translateY(-50%); 
                    font-size: 14px; color: #d84315; font-weight: bold;">
            h = {h_val if h_val == '-' else f'{h_val:.2f}'} cm
        </div>
        <div style="position: absolute; top: -20px; left: 10px; font-size: 14px; 
                    color: #5d4037; font-weight: bold;">
            Пласт {i+1}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Бутон за преминаване към следваща страница
if all_data_ready:
    if st.button("📤 Изпрати към 'Опън в покритието'", type="primary"):
        last_layer = st.session_state.layers_data[-1]
        st.session_state.final_Ed = last_layer["Ed"]
        st.session_state.Ei_list = [layer["Ei"] for layer in st.session_state.layers_data]
        st.session_state.hi_list = [layer["h"] for layer in st.session_state.layers_data]
        st.success("✅ Данните са подготвени за втората страница.")
        st.page_link("pages/second.py", label="Към Опън в покритието", icon="📄")
else:
    st.warning("ℹ️ Моля, попълнете данните за всички пластове преди да продължите")

# Връзки към другите страници
st.markdown("---")
st.subheader("Навигация към другите модули:")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/second.py", label="Опън в покритието", icon="📄", use_container_width=True)
with col2:
    st.page_link("pages/опън за междиннен плст.py", label="Опън в междинен пласт", icon="📄", use_container_width=True)
