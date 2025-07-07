import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("Определяне опънното напрежение в междиен пласт от пътната конструкция (Фиг. 9.3)")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Инициализация на session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}
if 'manual_sigma_values' not in st.session_state:
    st.session_state.manual_sigma_values = {}
if 'check_results' not in st.session_state:
    st.session_state.check_results = {}

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, value=4, step=1)
D = st.selectbox("Избери D", options=[32.04, 34.0])

# Входни стойности за всеки пласт
st.markdown("### Въведи стойности за всички пластове")
h_values, E_values, Ed_values = [], [], []
cols = st.columns(3)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        default_E = [1200, 1000, 800, 400][i] if i < 4 else 400
        E = st.number_input(f"E{to_subscript(i+1)}", value=default_E, step=10.0, key=f"E_{i}")
        E_values.append(E)
    with cols[2]:
        Ed = st.number_input(f"Ed{to_subscript(i+1)}", value=30.0, step=0.1, key=f"Ed_{i}")
        Ed_values.append(Ed)

# Избор на пласт
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(2, n)], index=n-3 if n > 2 else 0)
layer_idx = int(selected_layer.split()[-1]) - 1

# Изчисления
def calculate_layer(layer_index):
    h_array = np.array(h_values[:layer_index+1])
    E_array = np.array(E_values[:layer_index+1])
    current_Ed = Ed_values[layer_index]

    sum_h_n_1 = h_array[:-1].sum()
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 else 0

    H_n = h_array.sum()
    ratio = H_n / D if D != 0 else 0

    return {
        'H_n_1_r': round(sum_h_n_1, 3),
        'H_n_r': round(H_n, 3),
        'Esr_r': round(Esr, 3),
        'ratio_r': round(ratio, 3),
        'En_r': round(E_values[layer_index], 3),
        'Ed_r': round(current_Ed, 3),
        'Esr_over_En_r': round(Esr / E_values[layer_index], 3) if E_values[layer_index] else 0,
        'En_over_Ed_r': round(E_values[layer_index] / current_Ed, 3) if current_Ed else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }

if st.button(f"Изчисли за пласт {layer_idx+1}"):
    results = calculate_layer(layer_idx)
    st.session_state.layer_results[layer_idx] = results
    st.success(f"✅ Изчисленията за пласт {layer_idx+1} са извършени!")

# Показване на резултати
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]

    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    if layer_idx > 0:
        h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx)])
        st.latex(f"H_{{n-1}} = {h_terms} = {results['H_n_1_r']}")
    st.latex(f"H_n = {results['H_n_r']}")
    st.latex(f"\\frac{{H_n}}{{D}} = \\frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
    st.latex(f"E_{{{layer_idx+1}}} = {results['En_r']}")
    st.latex(f"Ed_{{{layer_idx+1}}} = {results['Ed_r']}")
    st.latex(f"\\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
    st.latex(f"\\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = {results['En_over_Ed_r']}")

    # Визуализация
    try:
        df1 = pd.read_csv("danni_1.csv")
        df2 = pd.read_csv("Оразмеряване на опън за междиннен плстH_D_1.csv")
        df2.rename(columns={"Esr/Ei": "sr_Ei"}, inplace=True)

        fig = go.Figure()

        for level in sorted(df1['Ei/Ed'].unique()):
            df_level = df1[df1['Ei/Ed'] == level]
            fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines', name=f"Ei/Ed = {level}"))

        for sr in sorted(df2['sr_Ei'].unique()):
            df_sr = df2[df2['sr_Ei'] == sr]
            fig.add_trace(go.Scatter(x=df_sr['H/D'], y=df_sr['y'], mode='lines', name=f"Esr/Ei = {sr}"))

        sr_target = results['Esr_over_En_r']
        Hn_D = results['ratio_r']
        sigma_r = None

        sr_values = sorted(df2['sr_Ei'].unique())
        if sr_target in sr_values:
            df_target = df2[df2['sr_Ei'] == sr_target]
            y_val = np.interp(Hn_D, df_target['H/D'], df_target['y'])
        else:
            for i in range(len(sr_values)-1):
                if sr_values[i] < sr_target < sr_values[i+1]:
                    df_low = df2[df2['sr_Ei'] == sr_values[i]]
                    df_high = df2[df2['sr_Ei'] == sr_values[i+1]]
                    y_low = np.interp(Hn_D, df_low['H/D'], df_low['y'])
                    y_high = np.interp(Hn_D, df_high['H/D'], df_high['y'])
                    y_val = y_low + (y_high - y_low) * (sr_target - sr_values[i]) / (sr_values[i+1] - sr_values[i])
                    break

        if 'y_val' in locals():
            fig.add_trace(go.Scatter(x=[Hn_D], y=[y_val], mode='markers', marker=dict(color='red', size=10), name='Интерполирана точка'))

            # Пресечна точка с Ei/Ed
            EiEd_target = results['En_over_Ed_r']
            EiEd_values = sorted(df1['Ei/Ed'].unique())

            for i in range(len(EiEd_values)-1):
                if EiEd_values[i] < EiEd_target < EiEd_values[i+1]:
                    df_low = df1[df1['Ei/Ed'] == EiEd_values[i]]
                    df_high = df1[df1['Ei/Ed'] == EiEd_values[i+1]]
                    x_low = np.interp(y_val, df_low['y'], df_low['H/D'])
                    x_high = np.interp(y_val, df_high['y'], df_high['H/D'])
                    x_final = x_low + (x_high - x_low) * (EiEd_target - EiEd_values[i]) / (EiEd_values[i+1] - EiEd_values[i])
                    sigma_r = round(x_final / 2, 3)

                    fig.add_trace(go.Scatter(x=[x_final], y=[y_val], mode='markers', marker=dict(color='orange', size=10), name='Пресечна точка'))

                    break

        fig.update_layout(
            title="Графика на изолинии",
            xaxis_title="H/D",
            yaxis_title="y",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
        st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=800)

        if sigma_r is not None:
            st.session_state.final_sigma = sigma_r
            st.markdown(f"**Изчислено σr = {sigma_r} MPa**")

            # Ръчно въвеждане
            manual = st.number_input("Ръчно въведена стойност σR [MPa]", min_value=0.0, max_value=20.0,
                                     value=sigma_r, step=0.1, key=f"manual_sigma_input_{layer_idx}")
            st.session_state.manual_sigma_values[f"manual_sigma_{layer_idx}"] = manual

            if st.button(f"Провери за пласт {layer_idx+1}"):
                if sigma_r <= manual:
                    st.success(f"✅ σr = {sigma_r} ≤ {manual} → Удовлетворява")
                else:
                    st.error(f"❌ σr = {sigma_r} > {manual} → Не удовлетворява")
        else:
            st.warning("Не е намерена стойност за σr.")
    except Exception as e:
        st.error(f"Грешка при визуализацията: {e}")
