import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Custom CSS to reduce page width
st.markdown(
    """
    <style>
        .main > div {
            max-width: 40%;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Initialize session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}
if 'manual_sigma_values' not in st.session_state:
    st.session_state.manual_sigma_values = {}
if 'check_results' not in st.session_state:
    st.session_state.check_results = {}

# Проверка за данни от główny файл
use_auto_data = False
if 'layers_data_all' in st.session_state and 'final_D_all' in st.session_state:
    layers_data = st.session_state.layers_data_all
    D_auto = st.session_state.final_D_all
    n_auto = len(layers_data)
    h_values_auto = [layer.get('h', 4.0) for layer in layers_data]
    E_values_auto = [layer.get('Ei', 400.0) for layer in layers_data]
    Ed_values_auto = [layer.get('Ed', 30.0) for layer in layers_data]
    use_auto_data = True

# Input parameters
if use_auto_data:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=n_auto)
    D = st.selectbox("Избери D", options=[32.04, 34.0], index=0 if D_auto == 32.04 else 1)
else:
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=4)
    D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Input data for all layers
st.markdown("### Въведи стойности за всички пластове")
h_values = []
E_values = []
Ed_values = []

cols = st.columns(3)
for i in range(n):
    # Автоматично попълване ако има данни
    h_default = h_values_auto[i] if use_auto_data and i < len(h_values_auto) else 4.0
    E_default = E_values_auto[i] if use_auto_data and i < len(E_values_auto) else [1200.0, 1000.0, 800.0, 400.0][i] if i < 4 else 400.0
    Ed_default = Ed_values_auto[i] if use_auto_data and i < len(Ed_values_auto) else 30.0
    
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=h_default, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E{to_subscript(i+1)}", value=E_default, step=0.1, key=f"E_{i}")
        E_values.append(E)
    with cols[2]:
        Ed = st.number_input(f"Ed{to_subscript(i+1)}", value=round(Ed_default), step=1, key=f"Ed_{i}")
        Ed_values.append(Ed)

# Layer selection
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(2, n)], index=n-3 if n > 2 else 0)
layer_idx = int(selected_layer.split()[-1]) - 1

# Calculation function
def calculate_layer(layer_index):
    h_array = np.array(h_values[:layer_index+1])
    E_array = np.array(E_values[:layer_index+1])
    current_Ed = Ed_values[layer_index]
    
    sum_h_n_1 = h_array[:-1].sum() if layer_index > 0 else 0
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1]) if layer_index > 0 else 0
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0
    
    H_n = h_array.sum()
    H_n_1 = sum_h_n_1
    
    results = {
        'H_n_1_r': round(H_n_1, 3),
        'H_n_r': round(H_n, 3),
        'Esr_r': round(Esr, 3),
        'ratio_r': round(H_n / D, 3) if D != 0 else 0,
        'En_r': round(E_values[layer_index], 3),
        'Ed_r': round(current_Ed, 3),
        'Esr_over_En_r': round(Esr / E_values[layer_index], 3) if E_values[layer_index] != 0 else 0,
        'En_over_Ed_r': round(E_values[layer_index] / current_Ed, 3) if current_Ed != 0 else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

# Calculate button
if st.button(f"Изчисли за пласт {layer_idx+1}"):
    results = calculate_layer(layer_idx)
    st.success(f"Изчисленията за пласт {layer_idx+1} са запазени!")

# Display results
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    if layer_idx > 0:
        h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx)])
        st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(layer_idx)} = {results['H_n_1_r']}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(results['n_for_calc'])])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(results['n_for_calc'])} = {results['H_n_r']}")

    if layer_idx > 0:
        st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
        numerator = " + ".join([f"{results['E_values'][i]} \cdot {results['h_values'][i]}" for i in range(layer_idx)])
        denominator = " + ".join([f"{results['h_values'][i]}" for i in range(layer_idx)])
        st.latex(fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']}")
    else:
        st.write("Esr = 0 (няма предишни пластове)")

    st.latex(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
    st.latex(fr"E_{{{layer_idx+1}}} = {results['En_r']}")
    st.latex(fr"\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
    st.latex(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")

    # Visualization
    try:
        df_original = pd.read_csv("danni_1.csv")
        df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D_1.csv")
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

        fig = go.Figure()

        # Add isolines from original df
        if 'Ei/Ed' in df_original.columns:
            for level in sorted(df_original['Ei/Ed'].unique()):
                df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Ei/Ed = {round(level,3)}',
                    line=dict(width=2)
                ))

        # Add isolines from new df
        if 'sr_Ei' in df_new.columns:
            for sr_Ei in sorted(df_new['sr_Ei'].unique()):
                df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Esr/Ei = {round(sr_Ei,3)}',
                    line=dict(width=2)
                ))

        # Interpolation and marking points
        if layer_idx > 0:
            sr_Ei_values = sorted(df_new['sr_Ei'].unique())
            target_sr_Ei = results['Esr_over_En_r']
            target_Hn_D = results['ratio_r']

            y_at_ratio = None
            if min(sr_Ei_values) <= target_sr_Ei <= max(sr_Ei_values):
                if target_sr_Ei in sr_Ei_values:
                    df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
                    y_at_ratio = np.interp(target_Hn_D, df_target['H/D'], df_target['y'])
                else:
                    for i in range(len(sr_Ei_values)-1):
                        if sr_Ei_values[i] < target_sr_Ei < sr_Ei_values[i+1]:
                            df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                            df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[i+1]].sort_values(by='H/D')
                            
                            y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
                            y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])
                            
                            y_at_ratio = y_lower + (y_upper - y_lower) * (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i+1] - sr_Ei_values[i])
                            break

            if y_at_ratio is not None:
                # Вертикална линия (синя)
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                    mode='lines', line=dict(color='blue', dash='dash'),
                    name='Вертикална линия'
                ))

                # Червена точка
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D], y=[y_at_ratio],
                    mode='markers', marker=dict(color='red', size=10),
                    name='Точка на интерполация'
                ))

                # Пресечна точка (оранжева)
                Ei_Ed_target = results['En_over_Ed_r']
                if 'Ei/Ed' in df_original.columns:
                    Ei_Ed_values = sorted(df_original['Ei/Ed'].unique())
                    if min(Ei_Ed_values) <= Ei_Ed_target <= max(Ei_Ed_values):
                        x_intercept = None
                        if Ei_Ed_target in Ei_Ed_values:
                            df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                            x_intercept = np.interp(y_at_ratio, df_level['y'], df_level['H/D'])
                        else:
                            for i in range(len(Ei_Ed_values)-1):
                                if Ei_Ed_values[i] < Ei_Ed_target < Ei_Ed_values[i+1]:
                                    df_lower = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i]].sort_values(by='H/D')
                                    df_upper = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i+1]].sort_values(by='H/D')
                                    
                                    x_lower = np.interp(y_at_ratio, df_lower['y'], df_lower['H/D'])
                                    x_upper = np.interp(y_at_ratio, df_upper['y'], df_upper['H/D'])
                                    
                                    x_intercept = x_lower + (x_upper - x_lower) * (Ei_Ed_target - Ei_Ed_values[i]) / (Ei_Ed_values[i+1] - Ei_Ed_values[i])
                                    break

                        if x_intercept is not None:
                            fig.add_trace(go.Scatter(
                                x=[x_intercept], y=[y_at_ratio],
                                mode='markers', marker=dict(color='orange', size=12),
                                name='Пресечна точка'
                            ))
                            # Хоризонтална линия между червената и оранжевата точка
                            fig.add_trace(go.Scatter(
                                x=[target_Hn_D, x_intercept],
                                y=[y_at_ratio, y_at_ratio],
                                mode='lines',
                                line=dict(color='green', dash='dash'),
                                name='Линия между червена и оранжева точка'
                            ))

                            # Вертикална линия от оранжева точка до y=2.5
                            fig.add_trace(go.Scatter(
                                x=[x_intercept, x_intercept],
                                y=[y_at_ratio, 2.5],
                                mode='lines',
                                line=dict(color='purple', dash='dash'),
                                name='Вертикална линия до y=2.5'
                            ))

        # --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[None, None],  # y не влияе
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
            xaxis='x2'  # Свързваме с втората ос
        ))

        fig.update_layout(
            title='Графика на изолинии',
            xaxis=dict(
                title='H/D',
                showgrid=True,
                zeroline=False,
            ),
            xaxis2=dict(
                overlaying='x',
                side='top',
                range=[fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 0, 1],
                showgrid=False,
                zeroline=False,
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1'],
                title='σr'
            ),
            yaxis=dict(
                title='y',
                range=[0, 3]
            ),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.image("Допустими опънни напрежения.png", caption="Допустими опънни напрежения", width=800)
        
        # Проверка дали x_intercept е дефинирана и не е None
        if ('x_intercept' in locals()) and (x_intercept is not None):
            sigma_r = round(x_intercept / 2, 3)
            st.markdown(f"**Изчислено σr = {sigma_r}**")
            
            # Запазваме стойността в session_state
            st.session_state.final_sigma = sigma_r

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

            # Променлива за крайното σR
            sigma_final = None
            
            if p is not None and sigma is not None:
                sigma_final = 1.15 * p * sigma
                st.markdown("### Формула за изчисление на крайното напрежение σR:")
                st.latex(r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}")
                st.latex(rf"\sigma_R = 1.15 \times {p:.3f} \times {sigma:.3f} = {sigma_final:.3f} \text{{ MPa}}")
                st.success(f"✅ Крайно напрежение σR = {sigma_final:.3f} MPa")
                
                # Запазваме крайната стойност за проверката
                st.session_state["final_sigma_R"] = sigma_final
            else:
                st.warning("❗ Липсва p или σR от номограмата за изчисление.")


    
            # Секция за ръчно въвеждане
            st.markdown(
                """
                <div style="background-color: #f0f9f0; padding: 10px; border-radius: 5px;">
                    <h3 style="color: #3a6f3a; margin: 0;">Ръчно отчитане σR спрямо Таблица 9.7</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Инициализираме ръчната стойност за този пласт, ако не съществува
            if f'manual_sigma_{layer_idx}' not in st.session_state.manual_sigma_values:
                st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = sigma_r

            # Поле за ръчно въвеждане
            manual_value = st.number_input(
                label="Въведете ръчно отчетена стойност σR [MPa]",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.manual_sigma_values.get(f'manual_sigma_{layer_idx}', sigma_r),
                step=0.1,
                key=f"manual_sigma_input_{layer_idx}",
                label_visibility="visible"
            )
            
            # Запазваме ръчно въведената стойност
            st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = manual_value
            
            # Проверка на условието (без бутон, автоматично при промяна)
            sigma_to_compare = st.session_state.get("final_sigma_R", None)
            
            if sigma_to_compare is not None:
                # Проверяваме дали вече имаме резултат за този пласт
                if f'check_result_{layer_idx}' not in st.session_state.check_results:
                    st.session_state.check_results[f'check_result_{layer_idx}'] = None
                
                # Проверка на условието
                check_passed = sigma_to_compare <= manual_value
                st.session_state.check_results[f'check_result_{layer_idx}'] = check_passed
                
                # Показваме резултата
                if check_passed:
                    st.success(
                        f"✅ Проверката е удовлетворена: "
                        f"изчисленото σR = {sigma_to_compare:.3f} MPa ≤ {manual_value:.3f} MPa (допустимото σR)"
                    )
                else:
                    st.error(
                        f"❌ Проверката НЕ е удовлетворена: "
                        f"изчисленото σR = {sigma_to_compare:.3f} MPa > {manual_value:.3f} MPa (допустимото σR)"
                    )
            else:
                st.warning("❗ Няма изчислена стойност σR (след коефициенти) за проверка.")
        else:
            st.markdown("**σr = -** (Няма изчислена стойност)")

    except Exception as e:
        st.error(f"Грешка при визуализацията: {e}")
