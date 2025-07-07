import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция")

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

# Check if data from first page exists
if 'layers_data' not in st.session_state or not st.session_state.layers_data:
    st.error("Моля, първо въведете данните в първата страница (Оразмеряване на пътна конструкция)")
    st.stop()

# Get data from first page
layers_data = st.session_state.layers_data
n = st.session_state.num_layers
D = st.session_state.final_D
axle_load = st.session_state.axle_load

# Extract values from layers_data
h_values = [layer.get('h', 0.0) for layer in layers_data]
E_values = [layer.get('Ei', 0.0) for layer in layers_data]
Ed_values = [layer.get('Ed', 0.0) for layer in layers_data]

st.info(f"Данни от първата страница: Брой пластове = {n}, D = {D} cm, Осов товар = {axle_load} kN")

# Display layer data
st.subheader("Данни за пластовете:")
layer_df = pd.DataFrame({
    "Пласт": [f"Пласт {i+1}" for i in range(n)],
    "h (cm)": h_values,
    "Ei (MPa)": E_values,
    "Ed (MPa)": Ed_values
})
st.dataframe(layer_df.set_index("Пласт"), use_container_width=True)

# Layer selection
st.subheader("Избери пласт за проверка")
if n < 3:
    st.error("Трябва да имате поне 3 пласта за междинна проверка!")
    st.stop()

# Options for intermediate layers (from layer 2 to n-1)
options = [f"Пласт {i+1}" for i in range(1, n-1)]
selected_layer = st.selectbox("Избери пласт", options=options, index=0)
layer_idx = int(selected_layer.split()[-1]) - 1

# Calculation function
def calculate_layer(layer_index):
    h_array = np.array(h_values[:layer_index+1])
    E_array = np.array(E_values[:layer_index+1])
    current_Ed = Ed_values[layer_index]
    
    # Calculate H_{n-1} and H_n
    H_n_1 = h_array[:-1].sum() if layer_index > 0 else 0
    H_n = h_array.sum()
    
    # Calculate Esr (weighted average)
    if layer_index > 0:
        weighted_sum = np.sum(E_array[:-1] * h_array[:-1])
        Esr = weighted_sum / H_n_1 if H_n_1 != 0 else 0
    else:
        Esr = 0

    # Calculate ratios
    ratio = H_n / D if D != 0 else 0
    Esr_over_En = Esr / E_values[layer_index] if E_values[layer_index] != 0 else 0
    En_over_Ed = E_values[layer_index] / current_Ed if current_Ed != 0 else 0

    results = {
        'H_n_1': round(H_n_1, 3),
        'H_n': round(H_n, 3),
        'Esr': round(Esr, 3),
        'ratio': round(ratio, 3),
        'En': round(E_values[layer_index], 3),
        'Ed': round(current_Ed, 3),
        'Esr_over_En': round(Esr_over_En, 3),
        'En_over_Ed': round(En_over_Ed, 3),
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

# Calculate button
if st.button(f"Изчисли за {selected_layer}"):
    results = calculate_layer(layer_idx)
    st.success(f"Изчисленията за {selected_layer} са запазени!")

# Display results
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.subheader(f"Резултати за {selected_layer}")
    
    # Display calculations with LaTeX
    st.latex(fr"H_{{n-1}} = \sum_{{i=1}}^{{{layer_idx}}} h_i = {results['H_n_1']}\, \text{{cm}}")
    st.latex(fr"H_n = \sum_{{i=1}}^{{{layer_idx+1}}} h_i = {results['H_n']}\, \text{{cm}}")
    
    if layer_idx > 0:
        st.latex(fr"E_{{sr}} = \frac{{\sum_{{i=1}}^{{{layer_idx}}} (E_i \cdot h_i)}}{{H_{{n-1}}}} = {results['Esr']}\, \text{{MPa}}")
    else:
        st.write("Esr = 0 (няма предишни пластове)")
    
    st.latex(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n']}}}{{{D}}} = {results['ratio']}")
    st.latex(fr"E_{{{layer_idx+1}}} = {results['En']}\, \text{{MPa}}")
    st.latex(fr"\frac{{E_{{sr}}}}{{E_{{{layer_idx+1}}}}} = \frac{{{results['Esr']}}}{{{results['En']}}} = {results['Esr_over_En']}")
    st.latex(fr"\frac{{E_{{{layer_idx+1}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En']}}}{{{results['Ed']}}} = {results['En_over_Ed']}")
    
    # Visualization section
    st.subheader("Визуализация на резултатите")
    
    try:
        # Generate synthetic data for visualization
        H_D = np.linspace(0, 3, 50)
        y = np.linspace(0, 3, 50)
        xx, yy = np.meshgrid(H_D, y)
        zz = xx * yy  # Placeholder for actual calculation
        
        fig = go.Figure(data=[
            go.Contour(
                z=zz,
                x=H_D,
                y=y,
                contours_coloring='lines',
                line_width=2,
                name='Изолинии'
            )
        ])
        
        # Add calculated point
        fig.add_trace(go.Scatter(
            x=[results['ratio']],
            y=[results['Esr_over_En']],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Нашата точка'
        ))
        
        fig.update_layout(
            title='Графика на изолинии',
            xaxis_title='H/D',
            yaxis_title='Esr/Ei',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Manual input section
        st.subheader("Ръчно отчитане на допустимо напрежение")
        
        # Initialize manual value if not exists
        if f'manual_sigma_{layer_idx}' not in st.session_state.manual_sigma_values:
            st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = 1.5  # Default value
            
        manual_value = st.number_input(
            label="Въведете допустимо опънно напрежение σR (MPa):",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'],
            step=0.1,
            key=f"manual_sigma_{layer_idx}"
        )
        
        # Store manual value
        st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}'] = manual_value
        
        # Calculate estimated sigma_r (simplified)
        sigma_r = results['ratio'] * 0.5  # Placeholder calculation
        
        # Check button
        if st.button(f"Проверка за {selected_layer}"):
            check_passed = sigma_r <= manual_value
            st.session_state.check_results[f'check_{layer_idx}'] = {
                'passed': check_passed,
                'sigma_r': sigma_r,
                'manual_value': manual_value
            }
        
        # Show check results
        if f'check_{layer_idx}' in st.session_state.check_results:
            check = st.session_state.check_results[f'check_{layer_idx}']
            status = "✅ Удовлетворена" if check['passed'] else "❌ Неудовлетворена"
            color = "green" if check['passed'] else "red"
            st.markdown(f"<h3 style='color:{color}'>{status}</h3>", unsafe_allow_html=True)
            st.write(f"Изчислено σr = {check['sigma_r']:.3f} MPa")
            st.write(f"Допустимо σR = {check['manual_value']:.3f} MPa")
            st.write(f"Условие: σr ≤ σR → {check['sigma_r']:.3f} ≤ {check['manual_value']:.3f}")
        
    except Exception as e:
        st.error(f"Грешка при визуализацията: {str(e)}")
