import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Опън в покритието")

# Зареждаме стойности от първата страница, ако има
Ed_default = st.session_state.get("final_Ed", 500.0)
D_default = st.session_state.get("final_D", 34.0)

# Вземаме пълните списъци, ако има такива
Ei_list_full = st.session_state.get("Ei_list", [])
hi_list_full = st.session_state.get("hi_list", [])

# Ограничаваме само до първите 2 пласта
Ei_list_default = Ei_list_full[:2]
hi_list_default = hi_list_full[:2]

# Показваме информация ако има повече от 2
if len(Ei_list_full) > 2:
    st.info("ℹ️ Използват се само първите два пласта от въведените на предишната страница.")

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
D = st.selectbox(
    "Диаметър на отпечатъка на колело  D (см)",
    options=[34.0, 32.04, 33.0],
    index=[34.0, 32.04, 33.0].index(D_default) if D_default in [34.0, 32.04, 33.0] else 0
)

# Въвеждане на Ed
Ed = st.number_input("Ed (MPa) – Модул на еластичност под пласта", value=Ed_default)

# Брой пластове
default_n = len(Ei_list_default)
n = st.number_input("Брой пластове", min_value=1, max_value=10, step=1, value=default_n or 1)

Ei_list = []
hi_list = []

st.markdown("#### Въвеждане на Eᵢ и hᵢ за всеки пласт:")
for i in range(1, n + 1):
    col1, col2 = st.columns(2)
    with col1:
        Ei = st.number_input(
            f"E{i} (MPa)",
            key=f"Ei_{i}",
            value=Ei_list_default[i - 1] if i - 1 < len(Ei_list_default) else 1000.0
        )
    with col2:
        hi = st.number_input(
            f"h{i} (см)",
            key=f"hi_{i}",
            value=hi_list_default[i - 1] if i - 1 < len(hi_list_default) else 10.0
        )
    Ei_list.append(Ei)
    hi_list.append(hi)

# Запазваме въведените параметри в session_state
st.session_state["final_D"] = D
st.session_state["final_Ed"] = Ed
st.session_state["Ei_list"] = Ei_list
st.session_state["hi_list"] = hi_list

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

# Вземаме евентуално запазени резултати
sigma_saved = st.session_state.get("final_sigma", None)
hD_saved = st.session_state.get("final_hD", None)
y_low_saved = st.session_state.get("final_y_low", None)
y_high_saved = st.session_state.get("final_y_high", None)
low_saved = st.session_state.get("final_low", None)
high_saved = st.session_state.get("final_high", None)

if st.button("Изчисли σR"):
    sigma, hD, y_low, y_high, low, high = compute_sigma_R(H, D, Esr, Ed)
    
    if sigma is not None:
        # Запазваме резултатите в session_state
        st.session_state["final_sigma"] = sigma
        st.session_state["final_hD"] = hD
        st.session_state["final_y_low"] = y_low
        st.session_state["final_y_high"] = y_high
        st.session_state["final_low"] = low
        st.session_state["final_high"] = high
    else:
        # Премахваме старите резултати, ако няма нови валидни
        for key in ["final_sigma", "final_hD", "final_y_low", "final_y_high", "final_low", "final_high"]:
            if key in st.session_state:
                del st.session_state[key]

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

# Ако има запазени резултати, показваме ги веднага
elif sigma_saved is not None:
    st.markdown("## 📋 Запазени резултати от предишното изчисление")
    st.markdown(f"""
    **Изчислено (запазено):**
    - $Esr / Ed = {Esr:.2f} / {Ed:.2f} = {Esr / Ed:.3f}$
    - $H / D = {H:.2f} / {D:.2f} = {H / D:.3f}$
    """)
    st.success(f"✅ σR = {sigma_saved:.3f}")
    st.info(f"Интерполация между изолинии: Esr/Ed = {low_saved:.2f} и {high_saved:.2f}")

    fig = go.Figure()
    for val, group in data.groupby("Esr_over_Ed"):
        fig.add_trace(go.Scatter(
            x=group["H_over_D"],
            y=group["sigma_R"],
            mode='lines',
            name=f"Esr/Ed = {val:.1f}"
        ))
    fig.add_trace(go.Scatter(
        x=[hD_saved], y=[sigma_saved],
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

st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=800)

# Вземане на осов товар от първата страница
axle_load = st.session_state.get("axle_load", 100)

# Определяне на p според осовия товар
if axle_load == 100:
    p = 0.620
elif axle_load == 115:
    p = 0.633
else:
    p = None
st.markdown(f"### 💡 Стойност на коефициент p според осов товар:")
if p is not None:
    st.success(f"p = {p:.3f} MPa (за осов товар {axle_load} kN)")
else:
    st.warning("❗ Не е зададен валиден осов товар. Не може да се изчисли p.")

# Вземаме sigma от session_state, ако има
sigma = st.session_state.get("final_sigma", None)

if p is not None and sigma is not None:
    sigma_final = 1.15 * p * sigma
    st.markdown("### Формула за изчисление на крайното напрежение σR:")
    st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
    st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
    st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
else:
    st.warning("❗ Липсва p или σR от номограмата за изчисление.")

# Лек акцент за заглавие
st.markdown(
    """
    <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
        <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# CSS за стилизиране на number_input
st.markdown("""
<style>
div[data-baseweb="input"] > input {
    width: 70px !important;
    padding-left: 5px !important;
    padding-right: 5px !important;
    text-align: left !important;  /* Подравняване на текста в input */
}
</style>
""", unsafe_allow_html=True)

# Вземаме изчислената σR от номограмата (ако има)
calculated_sigma = st.session_state.get("final_sigma", None)

# Колони за текст и входно поле на един ред
col1, col2 = st.columns([3, 1])

with col1:
    if calculated_sigma is not None:
        st.markdown(f"**σR = {calculated_sigma:.3f} ≤**")
    else:
        st.markdown("**σR (изчислено) не е налично — въведете ръчно стойност:**")

with col2:
    manual_value = st.number_input(
        label="",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1,
        key="manual_sigma_input",
        label_visibility="collapsed"
    )

# Бутон за проверка на условието
if st.button("Провери дали σR ≤ ръчно въведена стойност"):
    if calculated_sigma is None:
        st.warning("❗ Няма изчислена стойност σR за проверка.")
    else:
        if calculated_sigma <= manual_value:
            st.success(f"✅ Проверката е удовлетворена: {calculated_sigma:.3f} ≤ {manual_value:.3f}")
        else:
            st.error(f"❌ Проверката НЕ е удовлетворена: {calculated_sigma:.3f} > {manual_value:.3f}")


st.page_link("orazmeriavane_patna_konstrukcia.py", label="Към Оразмеряване на пътна конструкция", icon="📄")
