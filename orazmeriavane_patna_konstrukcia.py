
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
import plotly.express as px

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import io
from PIL import Image as PILImage  # ✅ Преименуваме, за да не се бърка с reportlab Image
from reportlab.platypus import Image as RLImage  # ✅ Ясно разграничение



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
                "EdEi_point": Ed_input / layer_data["Ei"],  # ДОБАВЕНО ТУК
                "y_low": y_low,
                "y_high": y_high,
                "low_iso": low_iso,
                "high_iso": high_iso,
                "mode": mode
            })

            # ДОБАВЕТЕ ТОВА:
            st.rerun()
            
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
        showlegend=False,
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
        if st.button("📤 Към Опън в долния оласт на покритието", type="primary", use_container_width=True):
            st.session_state.final_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.Ei_list = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.hi_list = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.final_D_value = st.session_state.final_D
            st.session_state.axle_load_value = st.session_state.axle_load
            st.success("✅ Всички данни са подготвени за втората страница.")
            st.page_link("pages/Опън в покритието.py", label="Към Опън в покритието", icon="📄")
        
        if st.button("📊 Kъм срязване сързани почви maxH/D=2 (фиг9.4)'", type="primary", use_container_width=True, key="to_fig9_4"):
            st.session_state.fig9_4_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_4_D = st.session_state.final_D
            st.session_state.fig9_4_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value2 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.4 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за сързани почви фиг9.4.py", label="Към Ꚍμ_p (фиг9.4)", icon="📈")

        if st.button("📊 Kъм срязване несързани почви maxH/D=1.5 (фиг9.6)'", type="primary", use_container_width=True, key="to_fig9_6"):
            st.session_state.fig9_6_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_6_D = st.session_state.final_D
            st.session_state.fig9_6_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value4 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.6 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за несързани почви фиг9.6.py", label="Към Ꚍμ_p (фиг9.6)", icon="📈")
    
    with cols[1]:
        if st.button("📤 Към Опън в междинен пласт'", type="primary", use_container_width=True, key="to_intermediate"):
            st.session_state.layers_data_all = st.session_state.layers_data
            st.session_state.final_D_all = st.session_state.final_D
            st.success("✅ Данните са запазени за междинния пласт!")
            st.page_link("pages/опън за междиннен плст.py", label="Към Опън в междинен пласт", icon="📄")

        if st.button("📊 Kъм срязване сързани почви maxH/D=4 (фиг9.5)'", type="primary", use_container_width=True, key="to_fig9_5"):
            st.session_state.fig9_5_Ed_list = [layer["Ed"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_h = [layer["h"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_Ei = [layer["Ei"] for layer in st.session_state.layers_data]
            st.session_state.fig9_5_D = st.session_state.final_D
            st.session_state.fig9_5_last_Ed = st.session_state.layers_data[-1]["Ed"]
            st.session_state.axle_load_value3 = st.session_state.axle_load
            st.success("✅ Данните за фиг.9.5 са готови!")
            st.page_link("pages/Определяне на Ꚍμ_p за сързани почви фиг9.5.py", label="Към Ꚍμ_p (фиг9.5)", icon="📈")
        
        if st.button("📊 Kъм срязване несързани почви maxH/D=2(фиг9.7)'", type="primary", use_container_width=True, key="to_fig9_7"):
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

# Редактиране на пластове
for i in range(st.session_state.num_layers):
    # Разделяне на реда на три колони
    col1, col2, col3 = st.columns([2, 3, 3])

    with col1:
        st.markdown(f"###  Пласт {i + 1}")
        # Ако имаш име на материала, можеш да го покажеш тук:
        if 'name' in st.session_state.layers_data[i]:
            st.markdown(f"**Материал:** {st.session_state.layers_data[i]['name']}")
        st.markdown("---")

    with col2:
        st.markdown("**Дебелина (cm)**")
        if 'h' in st.session_state.layers_data[i]:
            new_h = st.number_input(
                "",
                min_value=0.1,
                step=0.1,
                value=float(st.session_state.layers_data[i]['h']),
                key=f"h_edit_{i}_{st.session_state.layers_data[i].get('h', 0)}",
                label_visibility="collapsed"
            )
            st.session_state.layers_data[i]['h'] = new_h
        else:
            st.markdown("_Дебелина: няма данни_")

    with col3:
        st.markdown("λ коефициент ")
        new_lambda = st.number_input(
            "",
            min_value=0.0,
            max_value=4.0,
            step=0.01,
            value=float(st.session_state.lambda_values[i]),
            key=f"lambda_{i}_{st.session_state.lambda_values[i]}",
            label_visibility="collapsed"
        )
        st.session_state.lambda_values[i] = new_lambda

    # Разделител между пластовете
    st.divider()
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
        value=50,
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

layers = st.session_state.get("layers_data", [])
lambda_values = st.session_state.get("lambda_values", [])

# Проверка дали имаме нужните данни
if layers and lambda_values and len(layers) == len(lambda_values):
    # Проверка дали всеки слой има зададена дебелина 'h'
    if all("h" in layer and layer["h"] is not None for layer in layers):
        terms = []
        for i, (layer, lam) in enumerate(zip(layers, lambda_values)):
            h_cm = layer["h"]
            h_m = h_cm / 100  # преобразуваме cm → m
            if lam != 0:
                terms.append(h_m / lam)
            else:
                st.warning(f"λ_{i+1} не може да бъде 0!")
                st.stop()

        R0 = sum(terms)

        # Формула със символи
        symbolic_terms = [f"\\frac{{{{h_{i+1}}}}}{{{{\\lambda_{i+1}}}}}" for i in range(len(terms))]
        symbolic_formula = " + ".join(symbolic_terms)

        # Формула със заместени стойности (с преобразуване cm → m)
        numeric_terms = [
            f"\\frac{{{layer['h'] / 100:.3f}}}{{{lam:.3f}}}"
            for layer, lam in zip(layers, lambda_values)
        ]
        numeric_formula = " + ".join(numeric_terms)

        # Показваме символна формула
        st.latex(rf"R_0 = {symbolic_formula}")

        # Показваме заместената формула с реални стойности
        st.latex(rf"R_0 = {numeric_formula}")

        # Показваме крайния резултат
        st.latex(rf"R_0 = {R0:.3f}\ \text{{m²K/W}}")

    else:
        st.warning("Моля, задайте дебелини (h) за всички пластове преди изчисление.")
else:
    st.warning("Моля, уверете се, че броят на пластовете и λ-стойностите съвпадат.")


st.markdown("---")
# Check z vs sum of thicknesses
if all('h' in layer for layer in st.session_state.layers_data):
    sum_h = sum(layer['h'] for layer in st.session_state.layers_data)

    st.subheader("Проверка на изискванията")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Сума на дебелините (H)", f"{sum_h:.2f} cm")
    
    with col2:
        st.metric("Изчислена дълбочина на замръзване (z)", f"{z_value:.2f} cm")
    
    if z_value < sum_h:
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
def fig_to_image(fig):
    try:
        img_bytes = pio.to_image(fig, format="png", width=800, height=600)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Грешка при генериране на изображение: {e}")
        st.info("Моля, добавете 'kaleido==0.2.1' във файла requirements.txt")
        return Image.new('RGB', (800, 600), color=(255, 255, 255))


# Генериране на PDF отчет със заглавие, таблици и графики
st.markdown("---")
st.subheader("Генериране на отчет")

if st.button("📄 Генерирай PDF отчет (с графики)", type="primary"):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=10 * mm,  # Много тесни margins
            rightMargin=10 * mm,
            topMargin=10 * mm,
            bottomMargin=10 * mm
        )
        story = []
        styles = getSampleStyleSheet()

        # Зареждане на шрифт
        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
        font_name = 'DejaVuSans-Bold'

        # ЗАГЛАВИЕ (оригиналния стил)
        title_style = ParagraphStyle(
            'CustomTitle',
            fontSize=24,
            spaceAfter=20,  # По-малко разстояние
            alignment=1,
            textColor=colors.HexColor('#006064'),
            fontName=font_name,
            leading=30,
        )
        
        story.append(Paragraph("ОРАЗМЕРЯВАНЕ НА ПЪТНА КОНСТРУКЦИЯ", title_style))
        story.append(Spacer(1, 15))  # По-малко разстояние

        # МОДЕРНА ТАБЛИЦА С ИНФОРМАЦИЯ (20% по-малка, ляво подравняване)
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=6,
            fontName=font_name,
            textColor=colors.HexColor('#333333')
        )

        # Създаване на по-малка таблица с ляво подравняване
        table_data = [
            ["ПАРАМЕТЪР", "СТОЙНОСТ"],
            ["Осов товар", f"{st.session_state.axle_load} kN"],
            ["Диаметър D", f"{st.session_state.final_D} cm"],
            ["Брой пластове", str(st.session_state.num_layers)]
        ]

        # 20% по-малки ширини на колоните с ляво подравняване
        info_table = Table(table_data, colWidths=[64*mm, 48*mm], hAlign='LEFT')
        info_table.setStyle(TableStyle([
            # Header стил
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A7C59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('TOPPADDING', (0, 0), (-1, 0), 5),
            ('LEFTPADDING', (0, 0), (-1, 0), 8),
            ('RIGHTPADDING', (0, 0), (-1, 0), 8),
            
            # Данни стил
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('LEFTPADDING', (0, 1), (-1, -1), 8),
            ('RIGHTPADDING', (0, 1), (-1, -1), 8),
            
            # Grid и border
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 20))  # По-малко разстояние
        
        # ЛЕГЕНДА С ПРОСТ СТИЛ
        legend_title_style = ParagraphStyle(
            'LegendTitleStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=10,  # По-малко разстояние
            fontName=font_name,
            textColor=colors.HexColor('#2C5530')
        )
        
        legend_style = ParagraphStyle(
            'LegendStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5,  # По-малко разстояние
            fontName=font_name,
            textColor=colors.HexColor('#4B5563'),
            leftIndent=0
        )
        
        story.append(Paragraph("ЛЕГЕНДА", legend_title_style))
        story.append(Spacer(1, 5))  # По-малко разстояние
        
        # Елементи на легендата с bullet points
        legend_items = [
            "D – Диаметър на отпечатък на колелото",
            "Ed – Модул на еластичност на повърхността под пласта",
            "Ei – Модул на еластичност на пласта",
            "Ee – Модул на еластичност на повърхността на пласта", 
            "h – Дебелина на пласта"
        ]
        
        for item in legend_items:
            p = Paragraph(f"• {item}", legend_style)
            story.append(p)
        
        story.append(Spacer(1, 20))  # По-малко разстояние

                
        # Заглавие
        summary_title_style = ParagraphStyle(
            'SummaryTitle',
            fontName=font_name,
            fontSize=18,
            textColor=colors.HexColor('#2C5530'),
            spaceAfter=15,
            alignment=1
        )
        story.append(Paragraph("ГРАФИЧНО ОБОБЩЕНИЕ", summary_title_style))
        story.append(Spacer(1, 10))
        
        # Модерна визуализация на пластовете
        layer_title_style = ParagraphStyle('LayerTitle', fontName=font_name, fontSize=12, textColor=colors.HexColor('#5D4037'))
        ee_style = ParagraphStyle('EeValue', fontName=font_name, fontSize=11, textColor=colors.HexColor('#0277BD'), alignment=2)
        ed_style = ParagraphStyle('EdValue', fontName=font_name, fontSize=11, textColor=colors.HexColor('#2E7D32'), alignment=2)
        h_style  = ParagraphStyle('HValue', fontName=font_name, fontSize=11, textColor=colors.HexColor('#D84315'), alignment=0)
        ei_inner_style = ParagraphStyle('EiInner', fontName=font_name, fontSize=11, textColor=colors.HexColor('#004D40'), alignment=0)

        card_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#E0F7FA')),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#26C6DA')),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.white),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('ALIGN', (1, 1), (1, 1), 'LEFT'),
            ('ALIGN', (1, 2), (1, 2), 'RIGHT'),
        ])

        story.append(Paragraph("СХЕМА НА ПЛАСТОВЕТЕ", layer_title_style))
        story.append(Spacer(1, 8))

        for i, layer in enumerate(st.session_state.layers_data):
            if "Ed" not in layer:
                continue
            layer_data = [
                [Paragraph(f"Пласт {i+1}", layer_title_style), Paragraph(f"Ee = {layer['Ee']:.0f} MPa", ee_style)],
                [Paragraph(f"h = {layer['h']:.2f} cm", h_style), Paragraph(f"Ei = {layer['Ei']:.0f} MPa", ei_inner_style)],
                ["", Paragraph(f"Ed = {layer['Ed']:.0f} MPa", ed_style)]
            ]
            layer_card = Table(layer_data, colWidths=[55*mm, 75*mm])
            layer_card.setStyle(card_style)
            story.append(layer_card)
            story.append(Spacer(1, 10))


        # Всеки пласт на нова страница
        for i, layer in enumerate(st.session_state.layers_data):
            if "Ed" not in layer:
                continue

            # Нова страница за всеки пласт
            story.append(PageBreak())

            # Заглавие на пласт
            layer_title_style = ParagraphStyle(
                'LayerTitle',
                fontName=font_name,
                fontSize=16,
                textColor=colors.HexColor('#2C5530'),
                spaceAfter=10,  # По-малко разстояние
                alignment=1
            )
            story.append(Paragraph(f"ПЛАСТ {i + 1}", layer_title_style))
            story.append(Spacer(1, 8))  # По-малко разстояние

            # СТИЛ ЗА ИНФОРМАЦИЯТА ЗА ПЛАСТ
            layer_info_style = ParagraphStyle(
                'LayerInfo',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,  # По-малко разстояние
                fontName=font_name,
                textColor=colors.HexColor('#2C5530'),
                leftIndent=10
            )

            layer_value_style = ParagraphStyle(
                'LayerValue',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=4,  # По-малко разстояние
                fontName=font_name,
                textColor=colors.HexColor('#4B5563'),
                leftIndent=20
            )

            # Изчислителни параметри за конкретния пласт
            hD_point = layer.get('hD_point', 0)
            EdEi_point = layer.get('EdEi_point', 0)
            EeEi_ratio = layer['Ee'] / layer['Ei']
            
            # Информация за пласта - кратко и ясно
            story.append(Paragraph("ИЗЧИСЛЕНИЯ:", layer_info_style))
            story.append(Spacer(1, 3))  # По-малко разстояние
            
            # Основни параметри
            story.append(Paragraph(f"• Ee/Ei = {layer['Ee']:.0f} / {layer['Ei']:.0f} = {EeEi_ratio:.3f}", layer_value_style))
            story.append(Paragraph(f"• h/D = {layer['h']:.1f} / {st.session_state.final_D} = {hD_point:.3f}", layer_value_style))
            story.append(Paragraph(f"• Ed/Ei = {layer['Ed']:.0f} / {layer['Ei']:.0f} = {EdEi_point:.3f}", layer_value_style))
            
            # Основното изчисление - ПОСЛЕДНО
            if layer.get("mode") == "Ed / Ei":
                story.append(Paragraph(f"• Ed = Ei × (Ed/Ei) = {layer['Ei']:.3f} × {EdEi_point:.3f} = {layer['Ed']:.3f} MPa", layer_value_style))
            else:
                story.append(Paragraph(f"• h = D × (h/D) = {st.session_state.final_D} × {hD_point:.3f} = {layer['h']:.2f} cm", layer_value_style))
            
            story.append(Spacer(1, 8))  # По-малко разстояние

            # ГЕНЕРИРАНЕ НА ГРАФИКАТА
            fig = go.Figure()
            
            # Списък с всички уникални стойности на Ee/Ei
            all_e_ei_values = sorted(data["Ee_over_Ei"].unique())
            
            # Намиране на стойността Ee/Ei за текущия пласт
            current_e_ei = layer['Ee'] / layer['Ei']
            
            # Изчисляваме Ed/Ei точката правилно
            if 'Ed' in layer and 'Ei' in layer and layer['Ei'] > 0:
                EdEi_point = layer['Ed'] / layer['Ei']
            else:
                EdEi_point = current_e_ei  # fallback стойност
            
            # Филтрираме само изолиниите, които са кратни на 0.05
            multiples_of_005 = [val for val in all_e_ei_values if abs(val * 100) % 5 == 0]
            
            # Намираме двата най-близки изолинии до текущата стойност, които са кратни на 0.05
            if multiples_of_005:
                distances = [(abs(val - current_e_ei), val) for val in multiples_of_005]
                distances.sort()
                # Вземаме двата най-близки
                isos_to_label = [distances[0][1], distances[1][1]] if len(distances) >= 2 else [distances[0][1]]
            else:
                # Ако няма изолинии, кратни на 0.05, вземаме двата най-близки от всички
                distances = [(abs(val - current_e_ei), val) for val in all_e_ei_values]
                distances.sort()
                isos_to_label = [distances[0][1], distances[1][1]] if len(distances) >= 2 else [distances[0][1]]
                        
            # Добавяме всички изолинии
            for val in all_e_ei_values:
                group_sorted = data[data["Ee_over_Ei"] == val].sort_values("h_over_D")
                
                fig.add_trace(go.Scatter(
                    x=group_sorted["h_over_D"],
                    y=group_sorted["Ed_over_Ei"],
                    mode='lines',
                    line=dict(width=1.5),
                    showlegend=False,
                    hovertemplate=f"Ee/Ei = {val:.2f}<br>h/D = %{{x:.3f}}<br>Ed/Ei = %{{y:.3f}}<extra></extra>"
                ))
                
                # Добавяме надпис само за двата най-близки изолинии, които са кратни на 0.05
                if val in isos_to_label:
                    # Намираме точка в диапазона h/D 0-0.2 за поставяне на надписа
                    points_in_range = group_sorted[group_sorted["h_over_D"] <= 0.2]
                    if len(points_in_range) > 0:
                        # Вземаме средна точка в диапазона 0-0.2
                        mid_idx = len(points_in_range) // 2
                        x_pos = points_in_range.iloc[mid_idx]["h_over_D"]
                        y_pos = points_in_range.iloc[mid_idx]["Ed_over_Ei"]
                    else:
                        # Ако няма точки в диапазона, вземаме първата точка
                        x_pos = group_sorted.iloc[0]["h_over_D"]
                        y_pos = group_sorted.iloc[0]["Ed_over_Ei"]
                    
                    fig.add_annotation(
                        x=x_pos,
                        y=y_pos,
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(size=9, color="black"),
                        bgcolor="rgba(0,0,0,0)",  # Прозрачен фон
                        bordercolor="rgba(0,0,0,0)"  # Прозрачна рамка
                    )
            
            if all(k in layer for k in ["hD_point", "Ed", "Ei"]):
                hD = layer["hD_point"]
                # Използваме изчислената EdEi_point
                EdEi = EdEi_point
                
                # Добавяне на интерполационна линия
                if all(key in layer for key in ['y_low', 'y_high', 'low_iso', 'high_iso']):
                    fig.add_trace(go.Scatter(
                        x=[hD, hD],
                        y=[layer['y_low'], layer['y_high']],
                        mode='lines',
                        line=dict(color='purple', dash='dash', width=2),
                        showlegend=False
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[hD], y=[EdEi],
                    mode='markers',
                    marker=dict(color='red', size=12),
                    showlegend=False
                ))
            fig.update_layout(
                title=f"Пласт {i + 1} - Ed/Ei = f(h/D)",
                xaxis_title="h / D",
                yaxis_title="Ed / Ei",
                showlegend=False,
                template="plotly_white",
                width=1200,
                height=800
            )
            # Конвертиране на фигурата в изображение с PILImage
            try:
                img_bytes = pio.to_image(fig, format="png", width=1200, height=800)
                pil_img = PILImage.open(BytesIO(img_bytes))
            except Exception as e:
                st.error(f"Грешка при генериране на изображение за пласт {i+1}: {e}")
                pil_img = PILImage.new("RGB", (1200, 800), color=(255, 255, 255))

            # Добавяне на изображението към PDF с МАКСИМАЛЕН РАЗМЕР
            img_buffer = io.BytesIO()
            pil_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            # Размери за A4 с narrow margins (190mm ширина, 277mm височина)
            story.append(Paragraph("ГРАФИКА:", layer_info_style))
            story.append(Spacer(1, 3))  # Минимално разстояние
            story.append(RLImage(img_buffer, width=180 * mm, height=140 * mm))  # Максимален размер
            story.append(Spacer(1, 8))  # Минимално разстояние


        # НОВА СТРАНИЦА ЗА ТОПЛИННИ ПАРАМЕТРИ И ПРОВЕРКА
        story.append(PageBreak())

        # СЛЕД ГРАФИЧНО ОБОБЩЕНИЕ - ДОБАВЯНЕ НА ТОПЛИННИ ПАРАМЕТРИ И ПРОВЕРКИ
        story.append(Spacer(1, 15))
        
        # Стилове за топлинни параметри
        thermal_header_style = ParagraphStyle(
            'ThermalHeader',
            fontName=font_name,
            fontSize=12,
            textColor=colors.HexColor('#5D4037'),
            spaceAfter=4,
            leftIndent=10
        )
        
        thermal_value_style = ParagraphStyle(
            'ThermalValue', 
            fontName=font_name,
            fontSize=10,
            textColor=colors.HexColor('#4B5563'),
            spaceAfter=2,
            leftIndent=20
        )
        
        thermal_note_style = ParagraphStyle(
            'ThermalNote',
            fontName=font_name,
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            spaceAfter=4,
            leftIndent=25
        )
        
        # ТАБЛИЦА С ДЕБЕЛИНИ И λ КОЕФИЦИЕНТИ - ПЪРВО
        story.append(Paragraph("ДЕБЕЛИНИ И λ КОЕФИЦИЕНТИ НА ПЛАСТОВЕТЕ:", thermal_header_style))
        story.append(Spacer(1, 5))
        
        # Подготвяне на данните за таблицата
        table_data = [["Пласт", "Дебелина (cm)", "λ коефициент"]]
        
        for i, (layer, lam_val) in enumerate(zip(st.session_state.layers_data, st.session_state.lambda_values)):
            h_val = layer.get('h', '-')
            if h_val != '-':
                h_display = f"{h_val:.2f}"
            else:
                h_display = "-"
            
            table_data.append([
                f"{i+1}",
                h_display,
                f"{lam_val:.2f}"
            ])
        
        # Създаване на таблицата
        layers_table = Table(table_data, colWidths=[30*mm, 45*mm, 45*mm])
        layers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A7C59')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
            ('FONTNAME', (0, 1), (-1, -1), font_name),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
        ]))
        
        story.append(layers_table)
        story.append(Spacer(1, 15))
        
        # ТОПЛИННИ ПАРАМЕТРИ - СЛЕД ТАБЛИЦАТА
        story.append(Paragraph("ТОПЛИННИ ПАРАМЕТРИ:", thermal_header_style))
        story.append(Spacer(1, 5))
        
        # λоп и λзп стойности
        lambda_op = st.session_state.get("lambda_op_input", 2.5)
        lambda_zp = st.session_state.get("lambda_zp_input", 2.5)
        z1 = st.session_state.get("z1_input", 50)
        
        story.append(Paragraph(f"• λоп = {lambda_op:.2f} kcal/mhg", thermal_value_style))
        story.append(Paragraph("Коефициент на топлопроводност в открито поле", thermal_note_style))
        story.append(Paragraph("2.50 kcal/mhg за І климат. зона", thermal_note_style))
        story.append(Paragraph("2.20 kcal/mhg за ІІ климат. зона", thermal_note_style))
        story.append(Paragraph("(фиг.5.3)", thermal_note_style))
        
        story.append(Spacer(1, 3))
        
        story.append(Paragraph(f"• λзп = {lambda_zp:.2f} kcal/mhg", thermal_value_style))
        story.append(Paragraph("Коефициент на топлопроводност под настилката", thermal_note_style))
        story.append(Paragraph("Зависи от топлинната съпротивляемост", thermal_note_style))
        story.append(Paragraph("(таблица 5.2)", thermal_note_style))
        
        story.append(Spacer(1, 8))
        
        # Изчисление на m
        m_value = lambda_zp / lambda_op if lambda_op > 0 else 1.0
        story.append(Paragraph("ИЗЧИСЛЕНИЕ НА m:", thermal_header_style))
        story.append(Spacer(1, 3))
        
        # Формула за m като текст
        formula_text_m = f'm = λзп / λоп = {lambda_zp:.2f} / {lambda_op:.2f} = {m_value:.2f}'
        story.append(Paragraph(formula_text_m, thermal_value_style))
        
        story.append(Spacer(1, 8))
        
        # z₁ и изчисление на z
        story.append(Paragraph(f"• z₁ = {z1} cm", thermal_value_style))
        story.append(Paragraph("Замръзваща дълбочина на почвата в открито поле", thermal_note_style))
        story.append(Paragraph("Определя се от карта с изохети (фиг.5.2)", thermal_note_style))
        
        story.append(Spacer(1, 3))
        
        # Формула за z като текст
        z_value = z1 * m_value
        formula_text_z = f'z = z₁ × m = {z1} × {m_value:.2f} = {z_value:.2f} cm'
        story.append(Paragraph(formula_text_z, thermal_value_style))
        
        story.append(Spacer(1, 15))
        
        # ИЗЧИСЛЕНИЕ НА R₀
        story.append(Paragraph("ИЗЧИСЛЕНИЕ НА R<sub>0</sub>:", thermal_header_style))
        story.append(Spacer(1, 8))
        
        # Проверка дали имаме нужните данни
        layers = st.session_state.get("layers_data", [])
        lambda_values = st.session_state.get("lambda_values", [])
        
        if layers and lambda_values and len(layers) == len(lambda_values):
            if all("h" in layer and layer["h"] is not None for layer in layers):
                terms = []
                symbolic_terms = []
                numeric_terms = []
                
                for i, (layer, lam) in enumerate(zip(layers, lambda_values)):
                    h_cm = layer["h"]
                    h_m = h_cm / 100  # преобразуваме cm → m
                    if lam != 0:
                        terms.append(h_m / lam)
                        # използваме <sub> за индексите
                        symbolic_terms.append(f"h<sub>{i+1}</sub>/λ<sub>{i+1}</sub>")
                        numeric_terms.append(f"{h_m:.3f}/{lam:.3f}")
                
                R0 = sum(terms)
                
                # Символна формула
                symbolic_formula = " + ".join(symbolic_terms)
                story.append(Paragraph("Символна формула:", thermal_value_style))
                formula_text_symbolic = f'R<sub>0</sub> = {symbolic_formula}'
                story.append(Paragraph(formula_text_symbolic, thermal_value_style))
                
                story.append(Spacer(1, 5))
                
                # Числена формула
                numeric_formula = " + ".join(numeric_terms)
                story.append(Paragraph("Формула със заместени стойности:", thermal_value_style))
                formula_text_numeric = f'R<sub>0</sub> = {numeric_formula} = {R0:.3f} m²K/W'
                story.append(Paragraph(formula_text_numeric, thermal_value_style))
                

                                
        story.append(Spacer(1, 15))
        
        # ПРОВЕРКА НА ИЗИСКВАНИЯТА
        story.append(Paragraph("ПРОВЕРКА НА ИЗИСКВАНИЯТА:", thermal_header_style))
        story.append(Spacer(1, 5))
        
        # Изчисляване на сумата на дебелините
        if all('h' in layer for layer in st.session_state.layers_data):
            sum_h = sum(layer['h'] for layer in st.session_state.layers_data)
            
            check_data = [
                ["Параметър", "Стойност"],
                ["Сума на дебелините (H)", f"{sum_h:.2f} cm"],
                ["Изчислена дълбочина на замръзване (z)", f"{z_value:.2f} cm"]
            ]
            
            check_table = Table(check_data, colWidths=[90*mm, 50*mm])
            check_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A7C59')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 0), (-1, 0), 6),
                
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#4A7C59')),
            ]))
            
            story.append(check_table)
            story.append(Spacer(1, 8))
            
            # Проверка на условието
            if z_value > sum_h:
                condition_text ="Условието НЕ е изпълнено: z > Σh"
                condition_style = ParagraphStyle(
                    'ConditionOK',
                    fontName=font_name,
                    fontSize=11,
                    textColor=colors.HexColor('#c62828'),
                    spaceAfter=6,
                    alignment=1,
                    backColor=colors.HexColor('#ffebee')
                )
                story.append(Paragraph(condition_text, condition_style))
                
                conclusion_text = "Конструкцията НЕ удовлетворява изискванията! Дълбочината на замръзване (z) е по-голяма от общата дебелина на пластовете."
                conclusion_style = ParagraphStyle(
                    'ConclusionOK',
                    fontName=font_name,
                    fontSize=9,
                    textColor=colors.HexColor('#c62828'),
                    spaceAfter=12,
                    alignment=1
                )
                
                story.append(Paragraph(conclusion_text, conclusion_style))
            else:
                condition_text =  "Условието е изпълнено: z < Σh"
                condition_style = ParagraphStyle(
                    'ConditionFail',
                    fontName=font_name,
                    fontSize=11,
                    textColor=colors.HexColor('#2e7d32'),
                    spaceAfter=6,
                    alignment=1,
                    backColor=colors.HexColor('#e8f5e9')
                )
                story.append(Paragraph(condition_text, condition_style))
                
                conclusion_text = "Конструкцията удовлетворява изискванията! Дълбочината на замръзване (z) е по-малка от общата дебелина на пластовете."
                conclusion_style = ParagraphStyle(
                    'ConclusionFail',
                    fontName=font_name,
                    fontSize=9,
                    textColor=colors.HexColor('#2e7d32'),
                    spaceAfter=12,
                    alignment=1
                )
                story.append(Paragraph(conclusion_text, conclusion_style))
        
        # Дата и подпис
        story.append(Spacer(1, 20))
        current_date = datetime.now().strftime("%d.%m.%Y %H:%M")
        story.append(Paragraph(f"Генерирано на: {current_date}", ParagraphStyle('Date',
            fontSize=9, alignment=2, textColor=colors.grey, fontName=font_name)))

        # Финализиране
        doc.build(story)
        buffer.seek(0)
        st.success("✅ PDF отчетът с модерно графично оформление и топлинни изчисления е готов!")
        st.download_button("📥 Изтегли PDF отчет", buffer,
            file_name=f"Пътна_Конструкция_Отчет_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf")

    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {e}")
