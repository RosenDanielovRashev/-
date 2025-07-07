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
    st.session_state.layers_data = [{}]
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
        st.session_state.layers_data += [{} for _ in range(num_layers - len(st.session_state.layers_data))]
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

Ee = st.number_input("Ee (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ee", 2700.0), key=f"Ee_{layer_idx}")
Ei = st.number_input("Ei (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ei", 3000.0), key=f"Ei_{layer_idx}")

mode = st.radio(
    "Изберете параметър за отчитане:",
    ("Ed / Ei", "h / D"),
    key=f"mode_{layer_idx}"
)

# Функции за изчисления (остават същите като преди)
def compute_Ed(h, D, Ee, Ei):
    # ... (същия код) ...

def compute_h(Ed, D, Ee, Ei):
    # ... (същия код) ...

def add_interpolation_line(fig, hD_point, EdEi_point, y_low, y_high, low_iso, high_iso):
    # ... (същия код) ...

# Обработка на изчисленията
if mode == "Ed / Ei":
    h = st.number_input("Дебелина h (cm):", min_value=0.1, step=0.1, value=layer_data.get("h", 4.0), key=f"h_{layer_idx}")
    if st.button("Изчисли Ed", key=f"calc_Ed_{layer_idx}"):
        # ... (същия код) ...
        
elif mode == "h / D":
    Ed = st.number_input("Ed (MPa):", min_value=0.1, step=0.1, value=layer_data.get("Ed", 50.0), key=f"Ed_{layer_idx}")
    if st.button("Изчисли h", key=f"calc_h_{layer_idx}"):
        # ... (същия код) ...

# Визуализация на резултатите
st.markdown("---")
st.header("Резултати за всички пластове")

all_data_ready = True
for i, layer in enumerate(st.session_state.layers_data):
    Ee = layer.get('Ee', '-')
    Ei = layer.get('Ei', '-')
    Ed = layer.get('Ed', '-')
    h_val = layer.get('h', '-')
    
    # Проверка за пълнота на данните
    if any(val == '-' for val in [Ee, Ei, Ed, h_val]):
        all_data_ready = False
    
    # HTML за визуализация на пласта
    st.markdown(f"""
    <div class="layer-card">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    font-weight: bold; font-size: 18px; color: #006064;">
            Ei = {Ei} MPa
        </div>
        <div style="position: absolute; top: -20px; right: 10px; font-size: 14px; 
                    color: #00838f; font-weight: bold;">
            Ee = {Ee} MPa
        </div>
        <div style="position: absolute; bottom: -20px; right: 10px; font-size: 14px; 
                    color: #2e7d32; font-weight: bold;">
            Ed = {Ed if Ed == '-' else round(Ed)} MPa
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
