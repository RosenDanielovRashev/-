import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# --- ТВОЯТ ОСНОВЕН КОД ---

# Заглавие
st.title("Оразмеряване на пътна конструкция")

# Секция за въвеждане на данни
st.subheader("Въведете характеристики")

# Падащо меню за D
d_value_str = st.selectbox("Изберете стойност за D:", options=["32.04", "34"])
d_value = float(d_value_str)

# Падащо меню за осов товар
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=["100", "115"])
st.write(f"Избрана стойност за осов товар: {axle_load} kN")

# Брой пластове
st.subheader("Въведете брой на пластовете")
num_layers = st.number_input("Брой пластове:", min_value=1, step=1)
st.write(f"Въведен брой пластове: {int(num_layers)}")

# Данни за пласт 1
st.subheader("Въведете данни за оразмеряване - Пласт 1")
st.write(f"Стойност D за пласт 1: {d_value}")

Ee = st.number_input("Въведете стойност за Ee (MPa):", min_value=0.0, step=0.1)
h = st.number_input("Дебелина h на пласт 1 (cm):", min_value=1.0, step=0.1)
Ei = st.number_input("Модул на еластичност Ei на пласт 1 (MPa):", min_value=1.0, step=0.1)

# Изчисляване на формулите
if Ei > 0 and d_value > 0:
    ratio_h_D = h / d_value
    ratio_Ee_Ei = Ee / Ei
    st.subheader("Резултати от изчисленията")
    st.latex(r"\frac{h}{D} = " + f"{ratio_h_D:.3f}")
    st.latex(r"\frac{Ee}{Ei} = " + f"{ratio_Ee_Ei:.3f}")
else:
    st.write("Моля, въведете валидни стойности за Ei и D за изчисления.")

# --- КОД ЗА НОМОГРАМАТА ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("combined_data.csv")
        df = df.rename(columns={
            "E1_over_E2": "Ed_over_Ei",
            "Eeq_over_E2": "Ee_over_Ei"
        })
        return df
    except FileNotFoundError:
        st.error("Грешка: Файлът 'combined_data.csv' не е намерен. Моля, поставете файла в папката на приложението.")
        return None

data = load_data()

if data is not None:
    def compute_Ed(h, D, Ee, Ei):
        hD = h / D
        EeEi = Ee / Ei
        tol = 1e-4
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
        tol = 1e-4
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

    st.title("📐 Калкулатор: Метод на Иванов (интерактивна версия)")

    mode = st.radio(
        "Изберете параметър за отчитане:",
        ("Ed / Ei", "h / D")
    )

    # Използваме стойностите отгоре за Ee, Ei, D

    if Ei == 0 or d_value == 0:
        st.error("Ei и D не могат да бъдат 0.")
        st.stop()

    if mode == "Ed / Ei":
        h_input = st.number_input("h (cm)", value=h if h > 0 else 4.0)
        EeEi = Ee / Ei
        st.subheader("📊 Въведени параметри:")
        st.write(pd.DataFrame({
            "Параметър": ["Ee", "Ei", "h", "D", "Ee / Ei", "h / D"],
            "Стойност": [
                Ee,
                Ei,
                h_input,
                d_value,
                round(EeEi, 3),
                round(h_input / d_value, 3)
            ]
        }))

        st.markdown("### 🧾 Легенда:")
        st.markdown("""
        - **Ed** – Модул на еластичност на повърхността под пласта  
        - **Ei** – Модул на еластичност на пласта  
        - **Ee** – Модул на еластичност на повърхността на пласта  
        - **h** – Дебелина на пласта  
        - **D** – Диаметър на отпечатък на колелото  
        """)

        if st.button("Изчисли Ed"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h_input, d_value, Ee, Ei)

            if result is None:
                st.warning("❗ Точката е извън обхвата на наличните изолинии.")
            else:
                EdEi_point = result / Ei
                st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
                st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

                fig = go.Figure()
                for value, group in data.groupby("Ee_over_Ei"):
                    group_sorted = group.sort_values("h_over_D")
                    fig.add_trace(go.Scatter(
                        x=group_sorted["h_over_D"],
                        y=group_sorted["Ed_over_Ei"],
                        mode='lines',
                        name=f"Ee / Ei = {value:.2f}",
                        line=dict(width=1)
                    ))
                fig.add_trace(go.Scatter(
                    x=[hD_point],
                    y=[EdEi_point],
                    mode='markers',
                    name="Твоята точка",
                    marker=dict(size=8, color='red', symbol='circle')
                ))
                if y_low is not None and y_high is not None:
                    fig.add_trace(go.Scatter(
                        x=[hD_point, hD_point],
                        y=[y_low, y_high],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dot'),
                        name="Интерполационна линия"
                    ))
                fig.update_layout(
                    title="Интерактивна диаграма на изолинии (Ee / Ei)",
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    xaxis=dict(dtick=0.1),
                    yaxis=dict(dtick=0.05),
                    legend=dict(orientation="h", y=-0.3),
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        Ed = st.number_input("Ed (MPa)", value=520.0)
        EeEi = Ee / Ei
        EdEi = Ed / Ei

        st.subheader("📊 Въведени параметри:")
        st.write(pd.DataFrame({
            "Параметър": ["Ed", "Ee", "Ei", "D", "Ee / Ei", "Ed / Ei"],
            "Стойност": [
                Ed,
                Ee,
                Ei,
                d_value,
                round(EeEi, 3),
                round(EdEi, 3),
            ]
        }))

        st.markdown("### 🧾 Легенда:")
        st.markdown("""
        - **Ed** – Модул на еластичност на повърхността под пласта  
        - **Ei** – Модул на еластичност на пласта  
        - **Ee** – Модул на еластичност на повърхността на пласта  
        - **h** – Дебелина на пласта  
        - **D** – Диаметър на отпечатък на колелото  
        """)

        if st.button("Изчисли h"):
            h_result, hD_point, y_low, y_high, low_iso, high_iso = compute_h(Ed, d_value, Ee, Ei)

            if h_result is None:
                st.warning("❗ Неуспешно намиране на h — точката е извън обхвата.")
            else:
                st.success(f"✅ Изчислено: h = {h_result:.2f} cm (h / D = {hD_point:.3f})")
                st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

                fig = go.Figure()
                for value, group in data.groupby("Ee_over_Ei"):
                    group_sorted = group.sort_values("h_over_D")
                    fig.add_trace(go.Scatter(
                        x=group_sorted["h_over_D"],
                        y=group_sorted["Ed_over_Ei"],
                        mode='lines',
                        name=f"Ee / Ei = {value:.2f}",
                        line=dict(width=1)
                    ))
                fig.add_trace(go.Scatter(
                    x=[hD_point],
                    y=[EdEi],
                    mode='markers',
                    name="Твоята точка",
                    marker=dict(size=8, color='red', symbol='circle')
                ))
                if y_low is not None and y_high is not None:
                    fig.add_trace(go.Scatter(
                        x=[hD_point, hD_point],
                        y=[y_low, y_high],
                        mode='lines',
                        line=dict(color='green', width=2, dash='dot'),
                        name="Интерполационна линия"
                    ))
                fig.update_layout(
                    title="Интерактивна диаграма на изолинии (Ee / Ei)",
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    xaxis=dict(dtick=0.1),
                    yaxis=dict(dtick=0.05),
                    legend=dict(orientation="h", y=-0.3),
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
