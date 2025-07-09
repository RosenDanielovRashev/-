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
    st.session_state.layers_data = [{"Ee": 2700.0, "Ei": 3000.0, "mode": "Ed / Ei"}]
if "axle_load" not in st.session_state:
    st.session_state.axle_load = 100
if "final_D" not in st.session_state:
    st.session_state.final_D = 32.04

def reset_calculations_from_layer(layer_idx):
    for i in range(layer_idx, st.session_state.num_layers):
        layer = st.session_state.layers_data[i]
        keys_to_remove = ['Ed', 'h', 'hD_point', 'EdEi_point', 'y_low', 'y_high', 'low_iso', 'high_iso']
        for key in keys_to_remove:
            if key in layer:
                del layer[key]
        if i > 0 and i != layer_idx:
            prev_ed = st.session_state.layers_data[i-1].get("Ed", 2700.0)
            layer["Ee"] = prev_ed

st.title("Оразмеряване на пътна конструкция с няколко пластове")

# Избор на брой пластове
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=st.session_state.num_layers)
if num_layers != st.session_state.num_layers:
    st.session_state.num_layers = num_layers
    if len(st.session_state.layers_data) < num_layers:
        for i in range(len(st.session_state.layers_data), num_layers):
            prev_ed = st.session_state.layers_data[i-1].get("Ed", 2700.0)
            st.session_state.layers_data.append({"Ee": prev_ed, "Ei": 3000.0, "mode": "Ed / Ei"})
    elif len(st.session_state.layers_data) > num_layers:
        st.session_state.layers_data = st.session_state.layers_data[:num_layers]
    if st.session_state.current_layer >= num_layers:
        st.session_state.current_layer = num_layers - 1

# Избор на параметри
d_options = [32.04, 34, 33]
current_d_index = d_options.index(st.session_state.final_D) if st.session_state.final_D in d_options else 0

d_value = st.selectbox(
    "Изберете стойност за D (cm):", 
    options=d_options,
    index=current_d_index
)
st.session_state.final_D = d_value

axle_load = st.selectbox(
    "Изберете стойност за осов товар (kN):", 
    options=[100, 115],
    index=0 if st.session_state.axle_load == 100 else 1
)
st.session_state.axle_load = axle_load

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

if layer_idx > 0:
    prev_layer = st.session_state.layers_data[layer_idx - 1]
    if "Ed" in prev_layer:
        if prev_layer["Ed"] != layer_data.get("Ee"):
            layer_data["Ee"] = prev_layer["Ed"]
            reset_calculations_from_layer(layer_idx)
        st.info(f"ℹ️ Ee е автоматично зададен от Ed на предишния пласт: {round(prev_layer['Ed'])} MPa")
    else:
        st.warning("⚠️ Предишният пласт все още не е изчислен. Моля, изчислете предишния пласт първо.")

if layer_idx == 0:
    Ee_input = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ee", 2700.0), key=f"Ee_{layer_idx}")
    if Ee_input != layer_data.get("Ee"):
        layer_data["Ee"] = Ee_input
        reset_calculations_from_layer(0)
else:
    Ee = layer_data.get("Ee", 2700.0)
    st.write(f"**Ee (автоматично от предишен пласт):** {round(Ee)} MPa")

Ei_input = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ei", 3000.0), key=f"Ei_{layer_idx}")
if Ei_input != layer_data.get("Ei"):
    layer_data["Ei"] = Ei_input
    reset_calculations_from_layer(layer_idx)

# Определяне на режим с проверка за промяна
mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    key=f"mode_{layer_idx}",
    index=0 if layer_data.get("mode", "Ed / Ei") == "Ed / Ei" else 1
)

# Проверка за промяна на режима
if "mode" in layer_data and layer_data["mode"] != mode:
    reset_calculations_from_layer(layer_idx)
    layer_data["mode"] = mode

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
    fig.add_trace(go.Scatter(
        x=[hD_point, hD_point],
        y=[y_low, y_high],
        mode='lines',
        line=dict(color='purple', dash='dash'),
        name=f"Интерполация Ee/Ei: {low_iso:.2f} - {high_iso:.2f}"
    ))
    fig.add_trace(go.Scatter(
        x=[hD_point],
        y=[EdEi_point],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Резултат'
    ))

if mode == "Ed / Ei":
    # Въвеждане на дебелина h
    h_input = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=layer_data.get("h", 4.0), key=f"h_{layer_idx}")
    if h_input != layer_data.get("h"):
        layer_data["h"] = h_input
        reset_calculations_from_layer(layer_idx)
    
    # Бутон за изчисляване на Ed
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        # Изчисляване на резултата
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h_input, d_value, layer_data["Ee"], layer_data["Ei"])

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            # Изчисляване на съотношение Ed / Ei
            EdEi_point = result / layer_data["Ei"]
            
            # Актуализиране на данните ПРЕДИ използването им
            layer_data.update({
                "Ee": layer_data["Ee"],
                "Ei": layer_data["Ei"],
                "h": h_input,
                "Ed": result,
                "hD_point": hD_point,
                "EdEi_point": EdEi_point,
                "y_low": y_low,
                "y_high": y_high,
                "low_iso": low_iso,
                "high_iso": high_iso,
                "mode": mode
            })
            
            st.success(
                f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \n"
                f"Изчислено Ed = Ei * c = {layer_data['Ei']} * {EdEi_point:.3f} = {round(result)} MPa  \n"
                f"Ee/Ei = {layer_data['Ee']/layer_data['Ei']:.3f}  \n"
                f"h/D = {hD_point:.3f}"
            )
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            # Обновяване на следващия слой (ако има)
            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = result
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {result:.2f} MPa")


elif mode == "h / D":
    Ed_input = st.number_input("Ed (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ed", 50.0), key=f"Ed_{layer_idx}")
    if Ed_input != layer_data.get("Ed"):
        layer_data["Ed"] = Ed_input
        reset_calculations_from_layer(layer_idx)
    
    if "h" in layer_data and "hD_point" in layer_data:
        st.success(
            f"✅ Вече изчислено: h = {layer_data['h']:.2f} cm\n"
            f"h/D = {layer_data['hD_point']:.3f}\n"
            f"Ed/Ei = {layer_data['Ed']/layer_data['Ei']:.3f}\n"
            f"Ee/Ei = {layer_data['Ee']/layer_data['Ei']:.3f}"
        )
        st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {layer_data['low_iso']:.3f} и Ee / Ei = {layer_data['high_iso']:.3f}")
    
    if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed_input, d_value, layer_data["Ee"], layer_data["Ei"])
        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            # Актуализиране на данните ПРЕДИ използването им
            layer_data.update({
                "Ee": layer_data["Ee"],
                "Ei": layer_data["Ei"],
                "h": result,
                "Ed": Ed_input,
                "hD_point": hD_point,
                "y_low": y_low,
                "y_high": y_high,
                "low_iso": low_iso,
                "high_iso": high_iso,
                "mode": mode
            })
            
            st.success(
                f"✅ Изчислено: h = {result:.2f} cm\n"
                f"h/D = {hD_point:.3f}\n"
                f"Ed/Ei = {Ed_input/layer_data['Ei']:.3f}\n"
                f"Ee/Ei = {layer_data['Ee']/layer_data['Ei']:.3f}"
            )
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = Ed_input
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {Ed_input:.2f} MPa")

# Визуализация на графиката (общо за двата режима)
if "hD_point" in layer_data and "Ed" in layer_data and "Ei" in layer_data:
    fig = go.Figure()
    for value, group in data.groupby("Ee_over_Ei"):
        group_sorted = group.sort_values("h_over_D")
        fig.add_trace(go.Scatter(
            x=group_sorted["h_over_D"],
            y=group_sorted["Ed_over_Ei"],
            mode='lines',
            name=f"Ee/Ei = {value:.2f}"
        ))
    
    hD_point = layer_data['hD_point']
    EdEi_point = layer_data['Ed'] / layer_data['Ei']
    
    # Проверка за наличност на интерполационни данни
    if all(key in layer_data for key in ['y_low', 'y_high', 'low_iso', 'high_iso']):
        add_interpolation_line(fig, 
                              hD_point, 
                              EdEi_point,
                              layer_data['y_low'],
                              layer_data['y_high'],
                              layer_data['low_iso'],
                              layer_data['high_iso'])
    
    fig.update_layout(
        title="Ed / Ei в зависимост от h / D",
        xaxis_title="h / D",
        yaxis_title="Ed / Ei",
        legend_title="Изолинии"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{layer_idx}")

# Визуализация на резултатите
st.markdown("---")
st.header("Резултати за всички пластове")

all_data_ready = True
for i, layer in enumerate(st.session_state.layers_data):
    Ee_val = round(layer['Ee']) if 'Ee' in layer else '-'
    Ei_val = round(layer['Ei']) if 'Ei' in layer else '-'
    Ed_val = round(layer['Ed']) if 'Ed' in layer else '-'
    h_val = layer.get('h', '-')
    
    if any(val == '-' for val in [Ee_val, Ei_val, Ed_val, h_val]):
        all_data_ready = False
    
    status = "✅" if "Ed" in layer else "❌"
    
    st.markdown(f"""
    <div class="layer-card">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    font-weight: bold; font-size: 18px; color: #006064;">
            Ei = {Ei_val} MPa
        </div>
        <div style="position: absolute; top: -20px; right: 10px; font-size: 14px; 
                    color: #00838f; font-weight: bold;">
            Ee = {Ee_val} MPa
        </div>
        <div style="position: absolute; bottom: -20px; right: 10px; font-size: 14px; 
                    color: #2e7d32; font-weight: bold;">
            Ed = {Ed_val} MPa
        </div>
        <div style="position: absolute; top: 50%; left: 8px; transform: translateY(-50%); 
                    font-size: 14px; color: #d84315; font-weight: bold;">
            h = {h_val if h_val == '-' else f'{h_val:.2f}'} cm
        </div>
        <div style="position: absolute; top: -20px; left: 10px; font-size: 14px; 
                    color: #5d4037; font-weight: bold;">
            Пласт {i+1}
        </div>
        <div style="position: absolute; top: 5px; right: 5px; font-size: 20px;">
            {status}
        </div>
    </div>
    """, unsafe_allow_html=True)

if all_data_ready:
    cols = st.columns(2)
    with cols[0]:
        if st.button("📤 Изпрати към 'Опън в покритието'", type="primary", use_container_width=True):
            st.session_state.final_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.Ei_list = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.hi_list = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.final_D_value = st.session_state.final_D
            st.session_state.axle_load_value = st.session_state.axle_load
            st.success("✅ Всички данни са подготвени за втората страница.")
            st.page_link("pages/Опън в покритието.py", label="Към Опън в покритието", icon="📄")
    with cols[1]:
        if st.button("📤 Изпрати към 'Опън в междинен пласт'", type="primary", use_container_width=True, key="to_intermediate"):
            st.session_state.layers_data_all = st.session_state.layers_data
            st.session_state.final_D_all = st.session_state.final_D
            st.success("✅ Данните са запазени за междинния пласт!")
            st.page_link("pages/опън за междиннен плст.py", label="Към Опън в междинен пласт", icon="📄")
else:
    st.warning("ℹ️ Моля, попълнете данните за всички пластове преди да продължите")

st.markdown("---")
st.subheader("Навигация към другите модули:")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/Опън в покритието.py", label="Опън в покритието", icon="📄", use_container_width=True)
with col2:
    st.page_link("pages/опън за междиннен плст.py", label="Опън в междинен пласт", icon="📄", use_container_width=True)
