import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Проверка за наличието на данни от първата страница
if "layers_data" not in st.session_state or len(st.session_state.layers_data) == 0:
    st.warning("Моля, първо въведете данните в първата страница (Оразмеряване на пътна конструкция)")
    st.page_link("app.py", label="Към основната страница", icon="🏠")
    st.stop()

# Взимане на данните от първата страница
layers_data = st.session_state.layers_data
D = st.session_state.get("final_D", 32.04)  # Взимаме D от първата страница

# Подготвяме данните за пластовете
h_values = []
E_values = []
Ed_values = []

for i, layer in enumerate(layers_data):
    h_values.append(layer.get('h', 0))
    E_values.append(layer.get('Ei', 0))
    Ed_values.append(layer.get('Ed', 0))

n = len(layers_data)  # Брой пластове автоматично от първата страница

# Показваме информация за пластовете
st.markdown("### Данни за пластовете (взети от първата страница)")

cols = st.columns(3)
for i in range(n):
    with cols[0]:
        st.metric(f"h{to_subscript(i+1)} (cm)", f"{h_values[i]:.2f}" if h_values[i] else "-")
    with cols[1]:
        st.metric(f"E{to_subscript(i+1)} (MPa)", f"{E_values[i]:.0f}" if E_values[i] else "-")
    with cols[2]:
        st.metric(f"Ed{to_subscript(i+1)} (MPa)", f"{Ed_values[i]:.0f}" if Ed_values[i] else "-")

st.markdown(f"**D = {D} cm** (взето от първата страница)")

# Layer selection (само за междинните пластове)
if n < 2:
    st.error("Трябва да имате поне 2 пласта за изчисление на междинен пласт")
    st.stop()

st.markdown("### Избери междинен пласт за проверка")
selected_layer = st.selectbox(
    "Пласт за проверка", 
    options=[f"Пласт {i+1}" for i in range(1, n)],  # Започваме от 2-рия пласт
    index=0
)
layer_idx = int(selected_layer.split()[-1]) - 1

# Останалата част от кода остава същата, но премахваме ръчните inputs
# и използваме вече дефинираните h_values, E_values, Ed_values

# [Останалата част от вашия оригинален код за изчисления и визуализация]
# [Трябва да запазите всички функции и графики, но те вече ще работят с данните от първата страница]
