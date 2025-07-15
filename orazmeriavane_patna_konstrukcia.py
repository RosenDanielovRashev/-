import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go 
import os
import tempfile
from datetime import datetime
import base64
import plotly.io as pio
from fpdf import FPDF
from PIL import Image
import requests
from io import BytesIO

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

# Initialize session state
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
if "calculation_messages" not in st.session_state:
    st.session_state.calculation_messages = {}
if "lambda_values" not in st.session_state:
    st.session_state.lambda_values = [0.5 for _ in range(st.session_state.num_layers)]

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
    for i in range(layer_idx, st.session_state.num_layers):
        if i in st.session_state.calculation_messages:
            del st.session_state.calculation_messages[i]

st.title("Оразмеряване на пътна конструкция с няколко пластове")

# Избор на брой пластове
num_layers = st.number_input("Въведете брой пластове:", min_value=1, step=1, value=st.session_state.num_layers)
if num_layers != st.session_state.num_layers:
    # Първо синхронизирай layers_data
    if len(st.session_state.layers_data) < num_layers:
        for i in range(len(st.session_state.layers_data), num_layers):
            prev_ed = st.session_state.layers_data[i-1].get("Ed", 2700.0)
            st.session_state.layers_data.append({"Ee": prev_ed, "Ei": 3000.0, "mode": "Ed / Ei"})
    elif len(st.session_state.layers_data) > num_layers:
        st.session_state.layers_data = st.session_state.layers_data[:num_layers]
    
    # След това синхронизирай lambda_values
    current_lambda_len = len(st.session_state.lambda_values)
    if current_lambda_len < num_layers:
        st.session_state.lambda_values.extend([0.5 for _ in range(num_layers - current_lambda_len)])
    elif current_lambda_len > num_layers:
        st.session_state.lambda_values = st.session_state.lambda_values[:num_layers]
    
    # Актуализирай текущия пласт ако е необходимо
    if st.session_state.current_layer >= num_layers:
        st.session_state.current_layer = num_layers - 1
    
    st.session_state.num_layers = num_layers
    
# Parameter selection
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

# Layer navigation
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("⬅️ Предишен пласт"):
        if st.session_state.current_layer > 0:
            st.session_state.current_layer -= 1
with col3:
    if st.button("Следващ пласт ➡️"):
        if st.session_state.current_layer < st.session_state.num_layers - 1:
            st.session_state.current_layer += 1

# Current layer display
layer_idx = st.session_state.current_layer
st.subheader(f"Въвеждане на данни за пласт {layer_idx + 1}")

# Legend
st.markdown("### 🧾 Легенда:")
st.markdown("""
- **Ed** – Модул на еластичност на повърхността под пласта  
- **Ei** – Модул на еластичност на пласта  
- **Ee** – Модул на еластичност на повърхността на пласта  
- **h** – Дебелина на пласта  
- **D** – Диаметър на отпечатък на колелото  
""")

# Layer parameters input
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

mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    key=f"mode_{layer_idx}",
    index=0 if layer_data.get("mode", "Ed / Ei") == "Ed / Ei" else 1
)

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

if layer_idx in st.session_state.calculation_messages:
    st.success(st.session_state.calculation_messages[layer_idx])

if mode == "Ed / Ei":
    h_input = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=layer_data.get("h", 4.0), key=f"h_{layer_idx}")
    if h_input != layer_data.get("h"):
        layer_data["h"] = h_input
        reset_calculations_from_layer(layer_idx)
    
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h_input, d_value, layer_data["Ee"], layer_data["Ei"])

        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
            EdEi_point = result / layer_data["Ei"]
            
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
            
            success_message = (
                f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \n"
                f"Ed = Ei * {EdEi_point:.3f} = {layer_data['Ei']} * {EdEi_point:.3f} = {round(result)} MPa  \n"
                f"Ed = {round(result)} MPa  \n"
                f"Ee/Ei = {layer_data['Ee']:.0f}/ {layer_data['Ei']:.0f}= {layer_data['Ee']/layer_data['Ei']:.3f}  \n"
                f"h/D = {layer_data['h']:.1f}/{d_value} = {hD_point:.3f}"
            )
            
            st.session_state.calculation_messages[layer_idx] = success_message
            st.success(success_message)
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = result
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {result:.0f} MPa")

elif mode == "h / D":
    Ed_input = st.number_input("Ed (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ed", 50.0), key=f"Ed_{layer_idx}")
    if Ed_input != layer_data.get("Ed"):
        layer_data["Ed"] = Ed_input
        reset_calculations_from_layer(layer_idx)
    
    if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
        result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed_input, d_value, layer_data["Ee"], layer_data["Ei"])
        if result is None:
            st.warning("❗ Точката е извън обхвата на наличните изолинии.")
        else:
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
            
            success_message = (
                f"✅ Изчислено: h/D = {hD_point:.3f}  \n"
                f"h = D*{hD_point:.3f} = {d_value} * {hD_point:.3f} = {layer_data['h']:.2f}  \n"
                f"h = {result:.2f} cm  \n"
                f"Ed/Ei = {Ed_input:.1f}/{layer_data['Ei']:.0f} = {Ed_input/layer_data['Ei']:.3f}  \n"
                f"Ee/Ei = {layer_data['Ee']:.0f}/ {layer_data['Ei']:.0f}= {layer_data['Ee']/layer_data['Ei']:.3f}  \n"
            )
            
            st.session_state.calculation_messages[layer_idx] = success_message
            st.success(success_message)
            st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

            if layer_idx < st.session_state.num_layers - 1:
                next_layer = st.session_state.layers_data[layer_idx + 1]
                next_layer["Ee"] = Ed_input
                st.info(f"ℹ️ Ee за пласт {layer_idx + 2} е автоматично обновен на {Ed_input:.2f} MPa")

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

# Results display
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
        
        if st.button("📊 Изпрати към 'Ꚍμ_p (фиг9.4)'", type="primary", use_container_width=True, key="to_fig9_4"):
            st.session_state.fig9_4_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_D = st.session_state.final_D
            st.session_state.fig9_4_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value2 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.4 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за сързани почви фиг9.4.py", label="Към Ꚍμ_p (фиг9.4)", icon="📈")

        if st.button("📊 Изпрати към 'Ꚍμ_p (фиг9.6)'", type="primary", use_container_width=True, key="to_fig9_6"):
            st.session_state.fig9_6_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_D = st.session_state.final_D
            st.session_state.fig9_6_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value4 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.6 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за несързани почви фиг9.6.py", label="Към Ꚍμ_p (фиг9.6)", icon="📈")
    
    with cols[1]:
        if st.button("📤 Изпрати към 'Опън в междинен пласт'", type="primary", use_container_width=True, key="to_intermediate"):
            st.session_state.layers_data_all = st.session_state.layers_data
            st.session_state.final_D_all = st.session_state.final_D
            st.success("✅ Данните са запазени за междинния пласт!")
            st.page_link("pages/опън за междиннен плст.py", label="Към Опън в междинен пласт", icon="📄")

        if st.button("📊 Изпрати към 'Ꚍμ_p (фиг9.5)'", type="primary", use_container_width=True, key="to_fig9_5"):
            st.session_state.fig9_5_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_D = st.session_state.final_D
            st.session_state.fig9_5_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value3 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.5 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за сързани почви фиг9.5.py", label="Към Ꚍμ_p (фиг9.5)", icon="📈")
        
        if st.button("📊 Изпрати към 'Ꚍμ_p (фиг9.7)'", type="primary", use_container_width=True, key="to_fig9_7"):
            st.session_state.fig9_7_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_7_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_7_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_7_D = st.session_state.final_D
            st.session_state.fig9_7_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value5 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.7 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за несързани почви фиг9.7.py", label="Към Ꚍμ_p (фиг9.7)", icon="📈")
else:
    st.warning("ℹ️ Моля, попълнете данните за всички пластове преди да продължите")
    
st.markdown("---")
st.subheader("Навигация към другите модули:")
st.image("5.2. Фиг.png", width=800)
st.image("5.3. Фиг.png", width=800)
st.image("5.2. Таблица.png", width=800)
st.image("5.1. Таблица.png", width=800)

st.markdown("---")
st.subheader("Редактиране на пластове")

# Layer editing
for i in range(st.session_state.num_layers):
    col1, col2, col3 = st.columns([2, 3, 3])
    
    with col1:
        st.markdown(f"**Пласт {i+1}**")
    
    with col2:
        if 'h' in st.session_state.layers_data[i]:
            new_h = st.number_input(
                "Дебелина (cm)",
                min_value=0.1,
                step=0.1,
                value=float(st.session_state.layers_data[i]['h']),
                key=f"h_edit_{i}_unique_{st.session_state.layers_data[i].get('h', 0)}",
                label_visibility="collapsed"
            )
            st.session_state.layers_data[i]['h'] = new_h
        else:
            st.markdown("Дебелина: -")
    
    with col3:
        st.session_state.lambda_values[i] = st.number_input(
            "λ коефициент",
            min_value=0.0,
            max_value=4.0,
            step=0.01,
            value=st.session_state.lambda_values[i],
            key=f"lambda_{i}_unique_{st.session_state.lambda_values[i]}",
            label_visibility="collapsed"
        )

# Thermal parameters
st.markdown("---")
st.subheader("Топлинни параметри")

col1, col2 = st.columns(2)

with col1:
    lambda_op = st.number_input(
        "λоп (kcal/mhg)",
        min_value=0.1,
        step=0.1,
        value=2.5,
        key="lambda_op_input"
    )
    st.markdown("""
    <span style="font-size: small; color: #666;">
    Коефициент на топлопроводност в открито поле.<br>
    2.50 kcal/mhg за І климат. зона<br>
    2.20 kcal/mhg за ІІ климат. зона<br>
    (фиг.5.3)
    </span>
    """, unsafe_allow_html=True)

with col2:
    lambda_zp = st.number_input(
        "λзп (kcal/mhg)",
        min_value=0.1,
        step=0.1,
        value=2.5,
        key="lambda_zp_input"
    )
    st.markdown("""
    <span style="font-size: small; color: #666;">
    Коефициент на топлопроводност под настилката.<br>
    Зависи от топлинната съпротивляемост<br>
    (таблица 5.2)
    </span>
    """, unsafe_allow_html=True)

# Calculations
if lambda_op > 0:
    m_value = lambda_zp / lambda_op
    st.latex(rf"m = \frac{{\lambda_{{зп}}}}{{\lambda_{{оп}}}} = \frac{{{lambda_zp:.2f}}}{{{lambda_op:.2f}}} = {m_value:.2f}")
    
    z1 = st.number_input(
        "z₁ (cm)",
        min_value=1,
        step=1,
        value=100,
        key="z1_input"
    )
    st.markdown("""
    <span style="font-size: small; color: #666;">
    Замръзваща дълбочина на почвата в открито поле.<br>
    Определя се от карта с изохети (фиг.5.2)
    </span>
    """, unsafe_allow_html=True)
    
    z_value = z1 * m_value
    st.latex(rf"z = z_1 \cdot m = {z1} \cdot {m_value:.2f} = {z_value:.2f}\ \text{{cm}}")
else:
    st.warning("λоп не може да бъде 0")

# R₀ calculation
st.markdown("---")
st.subheader("Изчисление на R₀")

if all('h' in layer for layer in st.session_state.layers_data):
    sum_h = sum(layer['h'] for layer in st.session_state.layers_data)
    sum_lambda = sum(st.session_state.lambda_values)
    R0 = sum_h / sum_lambda if sum_lambda != 0 else 0
    
    st.latex(rf"""
    R_0 = \frac{{\sum_{{i=0}}^n h_i}}{{\sum_{{i=0}}^n \lambda_i}} = 
    \frac{{{sum_h:.2f}}}{{{sum_lambda:.2f}}} = {R0:.2f}\ \text{{cm}}
    """)
else:
    st.warning("Моля, задайте дебелини за всички пластове преди изчисление")

st.markdown("---")
# Check z vs sum of thicknesses
if all('h' in layer for layer in st.session_state.layers_data):
    sum_h = sum(layer['h'] for layer in st.session_state.layers_data)
    
    st.markdown("---")
    st.subheader("Проверка на изискванията")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Сума на дебелините (H)", f"{sum_h:.2f} cm")
    
    with col2:
        st.metric("Изчислена дълбочина на замръзване (z)", f"{z_value:.2f} cm")
    
    if z_value > sum_h:
        st.success("✅ Условието е изпълнено: z > Σh")
        st.markdown("""
        <div style="background-color:#e8f5e9; padding:10px; border-radius:5px; border-left:4px solid #2e7d32;">
        <span style="color:#2e7d32; font-weight:bold;">Конструкцията удовлетворява изискванията!</span><br>
        Замръзващата дълбочина (z) е по-голяма от общата дебелина на пластовете.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Условието НЕ е изпълнено: z ≤ Σh")
        st.markdown("""
        <div style="background-color:#ffebee; padding:10px; border-radius:5px; border-left:4px solid #c62828;">
        <span style="color:#c62828; font-weight:bold;">Конструкцията НЕ удовлетворява изискванията!</span><br>
        Замръзващата дълбочина (z) трябва да бъде по-голяма от общата дебелина на пластовете.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Препоръки:**
        - Увеличете дебелините на някои от пластовете
        - Използвайте материали с по-ниски λ коефициенти
        - Прегледайте избраните стойности за λоп и λзп
        """)

# Функция за конвертиране на Plotly фигура в изображение
# Updated fig_to_image function
def fig_to_image(fig):
    try:
        # Try using Kaleido if available
        img_bytes = pio.to_image(fig, format="png", width=800, height=600)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Грешка при генериране на изображение: {e}")
        st.info("Моля, добавете 'kaleido==0.2.1' във файла requirements.txt")
        # Return a blank placeholder image
        return Image.new('RGB', (800, 600), color=(255, 255, 255))
        
# Функция за сваляне на изображение от URL
def download_font(url):
    response = requests.get(url)
    return BytesIO(response.content)

# Генериране на PDF отчет
def generate_pdf_report(include_main, include_fig94, include_fig96, include_fig97, include_tension, include_intermediate):
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.temp_font_files = []  # Запазване на пътищата към временните файлове
            
        def header(self):
            self.set_font('DejaVu', 'B', 15)
            self.cell(0, 10, 'ОТЧЕТ ЗА ПЪТНА КОНСТРУКЦИЯ', 0, 1, 'C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('DejaVu', 'I', 8)
            self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
            
        def add_font_from_bytes(self, family, style, font_bytes):
            """Добавя шрифт от байтове чрез временен файл"""
            with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                tmp_file.write(font_bytes)
                tmp_file_path = tmp_file.name
                self.temp_font_files.append(tmp_file_path)
                self.add_font(family, style, tmp_file_path)
                
        def cleanup_fonts(self):
            """Изтрива временните шрифтови файлове"""
            for file_path in self.temp_font_files:
                try:
                    os.unlink(file_path)
                except Exception as e:
                    st.error(f"Грешка при изтриване на временен файл: {e}")

    pdf = PDF()
    
    try:
        # Download fonts at runtime
        dejavu_sans = download_font("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf")
        dejavu_bold = download_font("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf")
        dejavu_italic = download_font("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Oblique.ttf")
        
        # Добавяне на шрифтове чрез временни файлове
        pdf.add_font_from_bytes('DejaVu', '', dejavu_sans.getvalue())
        pdf.add_font_from_bytes('DejaVu', 'B', dejavu_bold.getvalue())
        pdf.add_font_from_bytes('DejaVu', 'I', dejavu_italic.getvalue())
    except Exception as e:
        st.error(f"Грешка при зареждане на шрифтове: {e}")
        return b""  # Връщане на празен byte string при грешка

    pdf.set_font('DejaVu', '', 12)
    
    pdf.add_page()
    
    # Заглавие
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(0, 10, 'ОТЧЕТ ЗА ПЪТНА КОНСТРУКЦИЯ', 0, 1, 'C')
    pdf.ln(10)
    
    # Дата
    pdf.set_font('DejaVu', '', 12)
    today = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    pdf.cell(0, 10, f'Дата: {today}', 0, 1)
    pdf.ln(5)
    
    # Списък с избрани страници
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, 'Включени раздели:', 0, 1)
    pdf.set_font('DejaVu', '', 12)
    
    included_sections = []
    if include_main: included_sections.append("Основна страница")
    if include_fig94: included_sections.append("Ꚍμ/p (фиг9.4)")
    if include_fig96: included_sections.append("Ꚍμ/p (фиг9.6)")
    if include_fig97: included_sections.append("Ꚍμ/p (фиг9.7)")
    if include_tension: included_sections.append("Опън в покритието")
    if include_intermediate: included_sections.append("Опън в междинен пласт")
    
    for section in included_sections:
        pdf.cell(0, 10, f'• {section}', 0, 1)
    pdf.ln(10)
    
    # Основна страница
    if include_main:
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Основна страница - Оразмеряване', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        # Общи параметри
        pdf.cell(0, 10, f'Брой пластове: {st.session_state.num_layers}', 0, 1)
        pdf.cell(0, 10, f'D: {st.session_state.final_D} cm', 0, 1)
        pdf.cell(0, 10, f'Осова тежест: {st.session_state.axle_load} kN', 0, 1)
        pdf.ln(5)
        
        # Данни за пластовете
        col_widths = [20, 30, 30, 30, 30, 30]
        headers = ["Пласт", "Ei (MPa)", "Ee (MPa)", "Ed (MPa)", "h (cm)", "λ"]
        
        # Хедър на таблицата
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # Данни за редовете
        for i in range(st.session_state.num_layers):
            layer = st.session_state.layers_data[i]
            lambda_val = st.session_state.lambda_values[i]
            
            Ei_val = round(layer.get('Ei', 0)) if 'Ei' in layer else '-'
            Ee_val = round(layer.get('Ee', 0)) if 'Ee' in layer else '-'
            Ed_val = round(layer.get('Ed', 0)) if 'Ed' in layer else '-'
            h_val = layer.get('h', '-')
            
            pdf.cell(col_widths[0], 10, str(i+1), 1, 0, 'C')
            pdf.cell(col_widths[1], 10, str(Ei_val), 1, 0, 'C')
            pdf.cell(col_widths[2], 10, str(Ee_val), 1, 0, 'C')
            pdf.cell(col_widths[3], 10, str(Ed_val), 1, 0, 'C')
            pdf.cell(col_widths[4], 10, str(h_val), 1, 0, 'C')
            pdf.cell(col_widths[5], 10, str(lambda_val), 1, 0, 'C')
            pdf.ln()
        
        pdf.ln(10)
        
        # Диаграми за всички пластове
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Диаграми за пластове', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        for i in range(st.session_state.num_layers):
            layer = st.session_state.layers_data[i]
            if "hD_point" in layer and "Ed" in layer and "Ei" in layer:
                # Създаване на фигура
                fig = go.Figure()
                for value, group in data.groupby("Ee_over_Ei"):
                    group_sorted = group.sort_values("h_over_D")
                    fig.add_trace(go.Scatter(
                        x=group_sorted["h_over_D"],
                        y=group_sorted["Ed_over_Ei"],
                        mode='lines',
                        name=f"Ee/Ei = {value:.2f}"
                    ))
                
                hD_point = layer['hD_point']
                EdEi_point = layer['Ed'] / layer['Ei']
                
                if all(key in layer for key in ['y_low', 'y_high', 'low_iso', 'high_iso']):
                    # Добавяне на интерполационна линия
                    fig.add_trace(go.Scatter(
                        x=[hD_point, hD_point],
                        y=[layer['y_low'], layer['y_high']],
                        mode='lines',
                        line=dict(color='purple', dash='dash'),
                        name=f"Интерполация Ee/Ei: {layer['low_iso']:.2f} - {layer['high_iso']:.2f}"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[hD_point],
                        y=[EdEi_point],
                        mode='markers',
                        marker=dict(color='red', size=12),
                        name='Резултат'
                    ))
                
                fig.update_layout(
                    title=f"Ed / Ei в зависимост от h / D за пласт {i+1}",
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    legend_title="Изолинии",
                    width=800,
                    height=600
                )
                
                # Конвертиране на фигурата в изображение и добавяне към PDF
                img = fig_to_image(fig)
                img_path = f"plot_layer_{i}.png"
                img.save(img_path)
                pdf.image(img_path, x=10, w=190)
                pdf.ln(5)
                os.remove(img_path)
        
        # Топлинни параметри
        if 'lambda_op_input' in st.session_state and 'lambda_zp_input' in st.session_state:
            lambda_op = st.session_state.lambda_op_input
            lambda_zp = st.session_state.lambda_zp_input
            m_value = lambda_zp / lambda_op
            z1 = st.session_state.get('z1_input', 100)
            z_value = z1 * m_value
            
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, 'Топлинни параметри', 0, 1)
            pdf.set_font('DejaVu', '', 12)
            pdf.cell(0, 10, f'λоп = {lambda_op} kcal/mhg', 0, 1)
            pdf.cell(0, 10, f'λзп = {lambda_zp} kcal/mhg', 0, 1)
            pdf.cell(0, 10, f'm = λзп / λоп = {lambda_zp} / {lambda_op} = {m_value:.2f}', 0, 1)
            pdf.cell(0, 10, f'z₁ = {z1} cm (дълбочина на замръзване в открито поле)', 0, 1)
            pdf.cell(0, 10, f'z = z₁ * m = {z1} * {m_value:.2f} = {z_value:.2f} cm', 0, 1)
            pdf.ln(10)
            
            # R₀ изчисление
            if all('h' in layer for layer in st.session_state.layers_data):
                sum_h = sum(layer['h'] for layer in st.session_state.layers_data)
                sum_lambda = sum(st.session_state.lambda_values)
                R0 = sum_h / sum_lambda if sum_lambda != 0 else 0
                
                pdf.cell(0, 10, f'R₀ = Σh / Σλ = {sum_h:.2f} / {sum_lambda:.2f} = {R0:.2f} cm', 0, 1)
                pdf.ln(10)
            
            # Проверка
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, 'Проверка на изискванията', 0, 1)
            pdf.set_font('DejaVu', '', 12)
            
            if all('h' in layer for layer in st.session_state.layers_data):
                if z_value > sum_h:
                    pdf.cell(0, 10, '✅ Условието е изпълнено: z > Σh', 0, 1)
                    pdf.cell(0, 10, f'z = {z_value:.2f} cm > Σh = {sum_h:.2f} cm', 0, 1)
                else:
                    pdf.cell(0, 10, '❌ Условието НЕ е изпълнено: z ≤ Σh', 0, 1)
                    pdf.cell(0, 10, f'z = {z_value:.2f} cm ≤ Σh = {sum_h:.2f} cm', 0, 1)
        
        # Добавяне на изображения от основната страница
        image_urls = [
            "https://raw.githubusercontent.com/.../5.2.Фиг.png",
            "https://raw.githubusercontent.com/.../5.3.Фиг.png",
            "https://raw.githubusercontent.com/.../5.2.Таблица.png",
            "https://raw.githubusercontent.com/.../5.1.Таблица.png"
        ]
        
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Допълнителни диаграми и таблици', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        for i, url in enumerate(image_urls):
            try:
                img = download_image(url)
                img_path = f"image_{i}.png"
                img.save(img_path)
                pdf.image(img_path, x=10, w=190)
                pdf.ln(5)
                os.remove(img_path)
            except:
                pdf.cell(0, 10, f'Грешка при зареждане на изображение {i+1}', 0, 1)
    
    # Добавете тук другите раздели (фиг9.4, фиг9.6 и т.н.) по същия начин
    
    pdf.cleanup_fonts()
    return pdf.output(dest='S').encode('utf-8')

# Генериране на отчет
st.markdown("---")
st.subheader("Генериране на отчет")

# Избор на страници за включване в PDF
st.markdown("**Изберете страници за включване в отчета:**")
col1, col2, col3 = st.columns(3)

with col1:
    include_main = st.checkbox("Основна страница", value=True)
    include_fig94 = st.checkbox("Ꚍμ/p (фиг9.4)", value=True)

with col2:
    include_fig96 = st.checkbox("Ꚍμ/p (фиг9.6)", value=True)
    include_fig97 = st.checkbox("Ꚍμ/p (фиг9.7)", value=True)

with col3:
    include_tension = st.checkbox("Опън в покритието", value=True)
    include_intermediate = st.checkbox("Опън в междинен пласт", value=True)

if st.button("📄 Генерирай PDF отчет", key="generate_pdf_button"):
    with st.spinner('Генериране на PDF отчет...'):
        pdf_bytes = generate_pdf_report(
            include_main, 
            include_fig94, 
            include_fig96, 
            include_fig97, 
            include_tension, 
            include_intermediate
        )
        
        # Създаване на временен файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_bytes)
            tmpfile.flush()
            
        # Показване на бутон за сваляне
        with open(tmpfile.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            download_link = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="patna_konstrukcia_report.pdf">Свали PDF отчет</a>'
            st.markdown(download_link, unsafe_allow_html=True)
            st.success("✅ PDF отчетът е успешно генериран!")

# Добавяне на информация за шрифтовете
st.markdown("""
<div class="warning-box">
    <strong>Важно:</strong> За правилно генериране на PDF файлове на кирилица, 
    моля добавете следните файлове в същата директория като приложението:
    <ul>
        <li><a href="https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf">DejaVuSans.ttf</a></li>
        <li><a href="https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf">DejaVuSans-Bold.ttf</a></li>
        <li><a href="https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Oblique.ttf">DejaVuSans-Oblique.ttf</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)
