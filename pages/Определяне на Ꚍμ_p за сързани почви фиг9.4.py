import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.markdown("""
    <style>
        .streamlit-expanderHeader {
            font-size: 18px !important;
        }
        .block-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .css-1lcbmi9 {
            max-width: 800px !important;
            margin: 0 auto !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Определяне на Ꚍμ/p за сързани почви фиг9.4 maxH/D=2")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Явна дефиниция на D
D = st.session_state.get('fig9_4_D', 32.04)  # Основна корекция

# Проверка за данни в session_state
session_data_available = all(key in st.session_state for key in ['fig9_4_h']) and \
                         'layers_data' in st.session_state and \
                         len(st.session_state.layers_data) > 0

# Автоматично зареждане на данни ако са налични
if session_data_available:
    n = len(st.session_state.fig9_4_h)
    h_values = st.session_state.fig9_4_h
    E_values = [layer["Ed"] for layer in st.session_state.layers_data]
    
    D_options = [32.04, 34.0, 33.0]
    
    if 'fig9_4_D' in st.session_state:
        current_d = st.session_state.fig9_4_D
        if current_d not in D_options:
            D_options.insert(0, current_d)
    else:
        current_d = D_options[0]

    selected_d = st.selectbox("Избери D", options=D_options, index=D_options.index(current_d))
    st.session_state.fig9_4_D = selected_d
    D = selected_d  # Актуализиране на D
    
    Fi_input = st.number_input("Fi (ϕ) стойност", value=15, step=1)
    
    st.markdown("### Автоматично заредени данни за пластовете")
    cols = st.columns(2)
    for i in range(n):
        with cols[0]:
            st.number_input(f"h{to_subscript(i+1)}", value=h_values[i], disabled=True, key=f"h_{i}")
        with cols[1]:
            st.number_input(f"Ed{to_subscript(i+1)}", value=E_values[i], disabled=True, key=f"E_{i}")

# Ръчно въвеждане ако няма данни в сесията
else:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
    D_options = [32.04, 34.0, 33.0]
    selected_d = st.selectbox("Избери D", options=D_options, index=0)
    st.session_state.fig9_4_D = selected_d
    D = selected_d  # Задаване на D
    
    Fi_input = st.number_input("Fi (ϕ) стойност", value=15, step=1)
    
    st.markdown("### Въведи стойности за всеки пласт")
    h_values = []
    E_values = []
    cols = st.columns(2)
    for i in range(n):
        with cols[0]:
            h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
            h_values.append(h)
        with cols[1]:
            E = st.number_input(f"Ed{to_subscript(i+1)}", value=1000.0, step=0.1, key=f"E_{i}")
            E_values.append(E)

# Избор на пласт за проверка
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Задаване на Eo = Ed на избрания пласт
Eo = E_values[layer_idx]
st.markdown(f"**Eo = Ed{to_subscript(layer_idx+1)} = {Eo}**")

# Изчисляване на H и Esr за избрания пласт
h_array = np.array(h_values[:layer_idx+1])
E_array = np.array(E_values[:layer_idx+1])

H = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / H if H != 0 else 0

# Формули и резултати
st.latex(r"H = \sum_{i=1}^n h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx+1)])
st.latex(r"H = " + h_terms)
st.write(f"H = {H:.3f}")

st.latex(r"Esr = \frac{\sum_{i=1}^n (E_i \cdot h_i)}{\sum_{i=1}^n h_i}")
numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(layer_idx+1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(layer_idx+1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum:.3f}}}{{{H:.3f}}} = {Esr:.3f}"
st.latex(formula_with_values)

ratio = H / D if D != 0 else 0
st.latex(r"\frac{H}{D} = \frac{" + f"{H:.3f}" + "}{" + f"{D}" + "} = " + f"{ratio:.3f}")

st.latex(r"\frac{Esr}{E_o} = \frac{" + f"{Esr:.3f}" + "}{" + f"{Eo}" + "} = " + f"{Esr / Eo:.3f}")
Esr_over_Eo = Esr / Eo if Eo != 0 else 0

# Зареждане на данни
df_fi = pd.read_csv("fi.csv")
df_esr_eo = pd.read_csv("Esr_Eo.csv")

df_fi.rename(columns={df_fi.columns[2]: 'fi'}, inplace=True)
df_esr_eo.rename(columns={df_esr_eo.columns[2]: 'Esr_Eo'}, inplace=True)

fig = go.Figure()

# Настройка на графиката
fig.update_layout(
    title=f"Графика на изолинии и точки за пласт {layer_idx+1}",
    xaxis_title="H/D",
    yaxis_title="y",
    legend_title="Легенда",
    width=900,
    height=600
)

# Определи фиксиран мащаб на основната ос (например 0 до 2)
xaxis_min = 0
xaxis_max = 2

# Добавяне на невидим trace, за да се покаже втората ос x2
fig.add_trace(go.Scatter(
    x=[xaxis_min, xaxis_max],
    y=[None, None],
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip',
    xaxis='x2'
))

fig.update_layout(
    title=f'Графика на изолинии за пласт {layer_idx+1}',
    xaxis=dict(
        title='H/D',
        showgrid=True,
        zeroline=False,
        range=[xaxis_min, xaxis_max],
    ),
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[xaxis_min, xaxis_max],
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickvals=np.linspace(xaxis_min, xaxis_max, 11),
        ticktext=[f"{(0.20 * (x - xaxis_min) / (xaxis_max - xaxis_max)):.3f}" for x in np.linspace(xaxis_min, xaxis_max, 11)],
        title='φ',
        fixedrange=True,
        showticklabels=True,
    ),
    yaxis=dict(
        title='y',
    ),
    showlegend=False,
    height=600,
    width=900
)

st.plotly_chart(fig, use_container_width=True)


