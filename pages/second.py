import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Опън в покритието")

@st.cache_data
def load_data():
    return pd.read_csv("sigma_data.csv")

data = load_data()

def compute_sigma_R(H, D, Esr, Ed):
    hD = H / D
    Esr_Ed = Esr / Ed
    tol = 1e-3
    iso_levels = sorted(data['Esr_over_Ed'].unique())

    for low, high in zip(iso_levels, iso_levels[1:]):
        if not (low - tol <= Esr_Ed <= high + tol):
            continue

        grp_low = data[data['Esr_over_Ed'] == low].sort_values('H_over_D')
        grp_high = data[data['Esr_over_Ed'] == high].sort_values('H_over_D')

        h_min = max(grp_low['H_over_D'].min(), grp_high['H_over_D'].min())
        h_max = min(grp_low['H_over_D'].max(), grp_high['H_over_D'].max())
        if not (h_min - tol <= hD <= h_max + tol):
            continue

        y_low = np.interp(hD, grp_low['H_over_D'], grp_low['sigma_R'])
        y_high = np.interp(hD, grp_high['H_over_D'], grp_high['sigma_R'])

        frac = 0 if np.isclose(high, low) else (Esr_Ed - low) / (high - low)
        sigma = y_low + frac * (y_high - y_low)

        return sigma, hD, y_low, y_high, low, high

    return None, None, None, None, None, None

st.title("Определяне опънното напрежение в долния плсаст на покритието фиг.9.2")

st.markdown("### Въвеждане на параметри на пластове")

# Избор на D от падащо меню
D = st.selectbox("Диаметър на отпечатъка на колело  D (см)", options=[34.0, 32.04, 33.0])

# Въвеждане на Ed
Ed = st.number_input("Ed (MPa) – Модул на еластичност под пласта", value=500.0)

# Брой пластове
n = st.number_input("Брой пластове", min_value=1, max_value=10, step=1, value=1)

Ei_list = []
hi_list = []

st.markdown("#### Въвеждане на Eᵢ и hᵢ за всеки пласт:")
for i in range(1, n + 1):
    col1, col2 = st.columns(2)
    with col1:
        Ei = st.number_input(f"E{i} (MPa)", key=f"Ei_{i}", value=1000.0)
    with col2:
        hi = st.number_input(f"h{i} (см)", key=f"hi_{i}", value=10.0)
    Ei_list.append(Ei)
    hi_list.append(hi)

# Изчисляване на Esr и H
numerator = sum(Ei * hi for Ei, hi in zip(Ei_list, hi_list))
denominator = sum(hi_list)
if denominator == 0:
    st.error("Сумата на hᵢ не може да бъде 0.")
    st.stop()

Esr = numerator / denominator
H = denominator  # автоматично взимаме сбор от всички hᵢ

# Показване на формулите
st.markdown("### ℹ️ Формули за изчисление")
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

# Показване на заместени стойности
numerator_str = " + ".join([f"{Ei}×{hi}" for Ei, hi in zip(Ei_list, hi_list)])
denominator_str = " + ".join([f"{hi}" for hi in hi_list])
st.latex(fr"Esr = \frac{{{numerator_str}}}{{{denominator_str}}} = {Esr:.2f} \text{{ MPa}}")
st.latex(fr"H = {denominator_str} = {H:.2f} \text{{ см}}")

if st.button("Изчисли σR"):
    sigma, hD, y_low, y_high, low, high = compute_sigma_R(H, D, Esr, Ed)
    
    st.markdown("## 📋 Резултати от изчисленията")

    if sigma is None:
        st.warning("❗ Точката е извън диапазона на наличните данни.")
    else:
        st.markdown(f"""
        **Изчислено:**
        - $Esr / Ed = {Esr:.2f} / {Ed:.2f} = {Esr / Ed:.3f}$
        - $H / D = {H:.2f} / {D:.2f} = {H / D:.3f}$
        """)
        st.success(f"✅ σR = {sigma:.3f}")
        st.info(f"Интерполация между изолинии: Esr/Ed = {low:.2f} и {high:.2f}")

        fig = go.Figure()
        for val, group in data.groupby("Esr_over_Ed"):
            fig.add_trace(go.Scatter(
                x=group["H_over_D"],
                y=group["sigma_R"],
                mode='lines',
                name=f"Esr/Ed = {val:.1f}"
            ))
        fig.add_trace(go.Scatter(
            x=[H / D], y=[sigma],
            mode='markers',
            marker=dict(size=8, color='red'),
            name="Твоята точка"
        ))
        fig.update_layout(
            title="Номограма: σR срещу H/D",
            xaxis_title="H / D",
            yaxis_title="σR",
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")
