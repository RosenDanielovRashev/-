import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from fpdf import FPDF
from PIL import Image
import plotly.io as pio
import os
import tempfile
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib import mathtext

# Настройки на страницата
st.set_page_config(layout="wide", page_title="Оразмеряване на пътната конструкция")

# Стилове
st.markdown("""
<style>
    .header {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
    }
    .layer-section {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .calculation {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    table, th, td {
        border: 1px solid #ddd;
    }
    th, td {
        padding: 8px;
        text-align: center;
    }
    th {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# Заглавие
st.markdown('<div class="header">Оразмеряване на пътната конструкция</div>', unsafe_allow_html=True)

# Дата и основни параметри
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Дата:** {pd.Timestamp.now().strftime('%d.%m.%Y')}")
with col2:
    st.write("**Брой пластове:** 4")
with col3:
    st.write("**D:** 32.04 cm")

# Легенда
st.markdown("""
**Легенда:**
- **Ed** – Модул на еластичност на повърхността под пласта
- **Ei** – Модул на еластичност на пласта
- **Ее** – Модул на еластичност на повърхността на пласта
- **h** – Дебелина на пласта
- **D** – Диаметър на отпечатък на колелото
""")

# Таблица с данни за пластовете
st.markdown('<div class="subheader">Параметри на пластовете</div>', unsafe_allow_html=True)

data = {
    "Пласт": [1, 2, 3, 4],
    "Ei (MPa)": [1200, 1000, 800, 500],
    "Ее (MPa)": [260, 232, 208, 82],
    "Ed (MPa)": [232, 208, 82, 30],
    "h (cm)": [4.0, 4.0, 20.0, 16.96],
    "λ": [0.5, 0.5, 0.5, 0.5]
}
df = pd.DataFrame(data)
st.table(df)

# Визуално представяне на пластовете
st.markdown('<div class="subheader">Визуално представяне на пластовете</div>', unsafe_allow_html=True)

for i in range(4):
    with st.container():
        st.markdown(f'<div class="layer-section"><b>Пласт {i+1}</b></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            st.write(f"**Ei:** {data['Ei (MPa)'][i]} MPa")
            st.write(f"**h:** {data['h (cm)'][i]} cm")
        with col2:
            st.write(f"**Ее:** {data['Ее (MPa)'][i]} MPa")
            st.write(f"**Ed:** {data['Ed (MPa)'][i]} MPa")
        
        with col3:
            # Тук може да добавите графика или диаграма за съответния пласт
            st.write("Диаграма на пласта...")

# Изчисления за всеки пласт
st.markdown('<div class="subheader">Изчисления по пластове</div>', unsafe_allow_html=True)

for i in range(4):
    with st.expander(f"Пласт {i+1}"):
        st.markdown(f"""
        <div class="calculation">
            <b>Ei</b> = {data['Ei (MPa)'][i]:.1f} MPa<br>
            <b>Ee</b> = {data['Ее (MPa)'][i]:.1f} MPa<br>
            <b>Ed</b> = {data['Ed (MPa)'][i]:.1f} MPa<br>
            <b>h</b> = {data['h (cm)'][i]:.2f} cm<br>
            <b>λ</b> = {data['λ'][i]:.1f}
        </div>
        """, unsafe_allow_html=True)
        
        # Изчисления
        Ed_Ei = data['Ed (MPa)'][i] / data['Ei (MPa)'][i]
        Ee_Ei = data['Ее (MPa)'][i] / data['Ei (MPa)'][i]
        h_D = data['h (cm)'][i] / 32.04
        
        st.markdown(f"""
        <div class="calculation">
            <b>Изчисления:</b><br>
            Изчислено: Ed / Ei = {Ed_Ei:.3f}<br>
            Ed = Ei * {Ed_Ei:.3f} = {data['Ei (MPa)'][i]:.1f} * {Ed_Ei:.3f} = {data['Ed (MPa)'][i]:.1f} MPa<br>
            Ee/Ei = {data['Ее (MPa)'][i]:.1f}/{data['Ei (MPa)'][i]:.1f} = {Ee_Ei:.3f}<br>
            h/D = {data['h (cm)'][i]:.2f}/32.04 = {h_D:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Графика
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 0.5, 1, 1.5],
            y=[0.9, 0.6, 0.3, 0.1],
            mode='lines',
            name=f'Пласт {i+1}'
        ))
        fig.update_layout(
            title=f'Графика за пласт {i+1}',
            xaxis_title='h / D',
            yaxis_title='Ed / Ei',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# Топлинни параметри
st.markdown('<div class="subheader">Топлинни параметри</div>', unsafe_allow_html=True)

st.latex(r"\lambda_{ОП} = 2.5 \, \text{kcal/mhg}")
st.latex(r"\lambda_{ЗП} = 2.5 \, \text{kcal/mhg}")
st.latex(r"m = \lambda_{ЗП} / \lambda_{ОП} = 2.5 / 2.5 = 1.00")
st.latex(r"z_1 = 100 \, \text{cm} \, (\text{дълбочина на замръзване в открито поле})")
st.latex(r"z = z_1 * m = 100 * 1.00 = 100.00 \, \text{cm}")
st.latex(r"R_o = \Sigma h / \Sigma \lambda = 44.96 / 2.00 = 22.48 \, \text{cm}")

# Проверка на изискванията
st.markdown('<div class="subheader">Проверка на изискванията</div>', unsafe_allow_html=True)

st.markdown("""
<div class="calculation">
    Условието е изпълнено: z > Σh<br>
    z = 100.00 cm > Σh = 44.96 cm
</div>
""", unsafe_allow_html=True)

# Допълнителни таблици
st.markdown('<div class="subheader">Допълнителни таблици</div>', unsafe_allow_html=True)

st.markdown("**Таблица 5.2: ПРЕПОРЪЧИТЕЛНИ СТОЙНОСТИ НА КОЕФИЦИЕНТА λзп**")
table_data = {
    "Rо": ["под 0,18", "от 0,18 до 0,25", "от 0,26 до 0,35", "от 0,36 до 0,45", 
           "от 0,46 до 0,55", "от 0,56 до 0,65", "над 0,65"],
    "λзп": [2.30, 2.15, 2.00, 1.85, 1.70, 1.65, 1.50]
}
st.table(pd.DataFrame(table_data))

st.markdown("**Таблица 5.1: СТОЙНОСТИ НА КОЕФИЦИЕНТА НА ТОПЛОПРОВОДИМОСТ**")
materials = [
    ["Плътна асфалтобетонова смес за износващ пласт", "1,10 – 1,30"],
    ["Пореста асфалтобетонова смес за долен пласт на покритието", "0,90 – 1,00"],
    ["Пореста асфалтобетонова смес за основа", "0,65 – 0,75"],
    ["Високопореста асфалтобетонова смес за основа", "0,55 – 0,65"],
    ["Шлаки металургични", "0,25 – 0,45"],
    ["Минерални материали, стабилизирани с течни органични свързващи вещества", "0,80 – 1,20"],
    ["Трошен камък с подбран зърнометричен състав", "1,80 – 2,20"],
    ["Баластра", "1,90 – 2,40"],
    ["Зърнести минерални материали, стабилизирани с неорганични свързващи вещества", "1,20 – 1,80"],
    ["Почви с различни степени на свързаност", "1,50 – 3,00"]
]
st.table(pd.DataFrame(materials, columns=["Материал", "λ, kcal/mhg"]))

# Генериране на PDF
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Добавяне на съдържание към PDF
    # Тук трябва да добавите кода за генериране на PDF според вашите изисквания
    
    return pdf.output(dest='S').encode('latin1')

if st.button("Генерирай PDF отчет"):
    pdf_bytes = generate_pdf()
    st.download_button(
        label="Свали PDF",
        data=pdf_bytes,
        file_name="patna_konstrukcia_report.pdf",
        mime="application/pdf"
    )
