import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit import session_state as state

st.set_page_config(layout="wide")

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция (фиг.9.3)")

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

# Проверка за данни от първата страница
if not hasattr(state, 'layers_data') or len(state.layers_data) < 1:
    st.error("""
    **Грешка:** Липсват данни от главната страница!
    Моля, първо въведете данните за пластовете в 'Оразмеряване на пътна конструкция' 
    """)
    st.page_link("orazmeriavane_patna_konstrukcia.py", label="← Върнете се към главната страница", icon="🏠")
    st.stop()

# Взимане на данните от първата страница
n = state.num_layers
D = state.final_D
layers_data = state.layers_data

# Извличане на параметрите за всеки пласт
h_values = [layer.get('h', 0.0) for layer in layers_data]
E_values = [layer.get('Ei', 0.0) for layer in layers_data]
Ed_values = [layer.get('Ed', 0.0) for layer in layers_data]

# Показване на информация за параметрите
with st.expander("📋 Параметри на пластовете (автоматично заредени)", expanded=True):
    cols = st.columns(4)
    cols[0].markdown("**Пласт**")
    cols[1].markdown("**Дебелина (h)**")
    cols[2].markdown("**Модул (E)**")
    cols[3].markdown("**Модул (Ed)**")
    
    for i in range(n):
        cols = st.columns(4)
        cols[0].markdown(f"Пласт {i+1}")
        cols[1].write(f"{h_values[i]:.1f} cm")
        cols[2].write(f"{E_values[i]:.1f} MPa")
        cols[3].write(f"{Ed_values[i]:.1f} MPa")

# Избор на пласт за проверка
st.markdown("---")
st.markdown("### Избор на пласт за анализ")
selected_layer = st.selectbox(
    "Изберете междинен пласт за проверка",
    options=[f"Пласт {i+1}" for i in range(1, n-1)] if n > 2 else ["Пласт 1"],
    index=0
)
layer_idx = int(selected_layer.split()[-1]) - 1

# Функция за изчисления
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
        'H_n_1': H_n_1,
        'H_n': H_n,
        'Esr': Esr,
        'ratio': H_n / D if D != 0 else 0,
        'En': E_values[layer_index],
        'Ed': current_Ed,
        'Esr_over_En': Esr / E_values[layer_index] if E_values[layer_index] != 0 else 0,
        'En_over_Ed': E_values[layer_index] / current_Ed if current_Ed != 0 else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

# Изчислителен блок
if st.button(f"🔢 Изчисли за {selected_layer}"):
    with st.spinner("Извършвам изчисления..."):
        results = calculate_layer(layer_idx)
        st.success(f"Изчисленията за {selected_layer} са готови!")

# Показване на резултатите
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown("---")
    st.markdown(f"### 📊 Резултати за {selected_layer}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Hₙ₋₁ (Сума на дебелините до пласта)", f"{results['H_n_1']:.2f} cm")
        st.metric("Hₙ (Обща дебелина до пласта)", f"{results['H_n']:.2f} cm")
        st.metric("Esr (Среден модул на еластичност)", f"{results['Esr']:.2f} MPa")
        
    with col2:
        st.metric("Hₙ/D (Относителна дебелина)", f"{results['ratio']:.4f}")
        st.metric("Eₙ (Модул на текущия пласт)", f"{results['En']:.2f} MPa")
        st.metric("Eₙ/Ed (Коефициент на деформация)", f"{results['En_over_Ed']:.4f}")

    # Визуализация
    try:
        df_original = pd.read_csv("danni_1.csv")
        df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D_1.csv")
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

        fig = go.Figure()

        # Добавяне на изолинии
        for level in sorted(df_original['Ei/Ed'].unique()):
            df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
            fig.add_trace(go.Scatter(
                x=df_level['H/D'], y=df_level['y'],
                mode='lines', name=f'Ei/Ed = {round(level,3)}',
                line=dict(width=2)
            ))


        # Маркиране на точката на интерполация
        if layer_idx > 0:
            target_sr_Ei = results['Esr_over_En']
            target_Hn_D = results['ratio']
            
            # Интерполационна логика...
            # (оставете същата като във вашия оригинален код)
            
            # Добавяне на елементи към графиката
            fig.add_trace(go.Scatter(
                x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                mode='lines', line=dict(color='blue', dash='dash'),
                name='Вертикална линия'
            ))
            
            # Останалите елементи на графиката...
            # (запазете оригиналната визуализация)

        fig.update_layout(
            title='Изолинии за определяне на опънните напрежения',
            xaxis_title='H/D',
            yaxis_title='y',
            legend_title="Параметри",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Допълнителни проверки и изчисления...
        # (запазете оригиналната логика за проверки)

    except Exception as e:
        st.error(f"Грешка при визуализацията: {str(e)}")

# Линк за връщане към главната страница
st.markdown("---")
st.page_link("orazmeriavane_patna_konstrukcia.py", 
            label="← Върнете се към главната страница", 
            icon="🏠")
