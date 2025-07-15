# Заменете раздела за PDF отчет със следния код:

st.markdown("---")
st.subheader("Генериране на отчет")

# Избор на страници за включване в PDF
st.markdown("**Изберете страници за включване в отчета:**")
col1, col2, col3 = st.columns(3)

with col1:
    include_main = st.checkbox("Основна страница", value=True)
    include_fig94 = st.checkbox("Ꚍμ/p (фиг9.4)", value=True)

with col2:
    include_fig96 = st.checkbox("Ꚍμ/p (фиг9.6)", value=True)
    include_fig97 = st.checkbox("Ꚍμ/p (фиг9.7)", value=True)

with col3:
    include_tension = st.checkbox("Опън в покритието", value=True)
    include_intermediate = st.checkbox("Опън в междинен пласт", value=True)

if st.button("📄 Генерирай PDF отчет", key="generate_pdf_button"):
    # Функция за създаване на PDF
    def generate_pdf_report():
        class PDF(FPDF):
            def header(self):
                self.set_font('DejaVu', 'B', 15)
                self.cell(0, 10, 'ОТЧЕТ ЗА ПЪТНА КОНСТРУКЦИЯ', 0, 1, 'C')
                self.ln(5)
                
            def footer(self):
                self.set_y(-15)
                self.set_font('DejaVu', 'I', 8)
                self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
        
        # Създаване на PDF обект с поддръжка на кирилица
        pdf = PDF()
        pdf.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
        pdf.add_font('DejaVu', 'I', 'fonts/DejaVuSans-Oblique.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
        pdf.add_page()
        
        # Заглавие
        pdf.set_font('DejaVu', 'B', 16)
        pdf.cell(0, 10, 'ОТЧЕТ ЗА ПЪТНА КОНСТРУКЦИЯ', 0, 1, 'C')
        pdf.ln(10)
        
        # Дата
        pdf.set_font('DejaVu', '', 12)
        today = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        pdf.cell(0, 10, f'Дата: {today}', 0, 1)
        pdf.ln(5)
        
        # Списък с избрани страници
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, 'Включени раздели:', 0, 1)
        pdf.set_font('DejaVu', '', 12)
        
        included_sections = []
        if include_main: included_sections.append("Основна страница")
        if include_fig94: included_sections.append("Ꚍμ/p (фиг9.4)")
        if include_fig96: included_sections.append("Ꚍμ/p (фиг9.6)")
        if include_fig97: included_sections.append("Ꚍμ/p (фиг9.7)")
        if include_tension: included_sections.append("Опън в покритието")
        if include_intermediate: included_sections.append("Опън в междинен пласт")
        
        for section in included_sections:
            pdf.cell(0, 10, f'• {section}', 0, 1)
        pdf.ln(10)
        
        # Основна страница
        if include_main:
            pdf.set_font('DejaVu', 'B', 14)
            pdf.cell(0, 10, 'Основна страница - Оразмеряване', 0, 1)
            pdf.set_font('DejaVu', '', 12)
            
            # Общи параметри
            pdf.cell(0, 10, f'Брой пластове: {st.session_state.num_layers}', 0, 1)
            pdf.cell(0, 10, f'D: {st.session_state.final_D} cm', 0, 1)
            pdf.cell(0, 10, f'Осова тежест: {st.session_state.axle_load} kN', 0, 1)
            pdf.ln(5)
            
            # Данни за пластовете
            col_widths = [20, 30, 30, 30, 30, 30]
            headers = ["Пласт", "Ei (MPa)", "Ee (MPa)", "Ed (MPa)", "h (cm)", "λ"]
            
            # Хедър на таблицата
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
            pdf.ln()
            
            # Данни за редовете
            for i in range(st.session_state.num_layers):
                layer = st.session_state.layers_data[i]
                lambda_val = st.session_state.lambda_values[i]
                
                Ei_val = round(layer.get('Ei', 0)) if 'Ei' in layer else '-'
                Ee_val = round(layer.get('Ee', 0)) if 'Ee' in layer else '-'
                Ed_val = round(layer.get('Ed', 0)) if 'Ed' in layer else '-'
                h_val = layer.get('h', '-')
                
                pdf.cell(col_widths[0], 10, str(i+1), 1, 0, 'C')
                pdf.cell(col_widths[1], 10, str(Ei_val), 1, 0, 'C')
                pdf.cell(col_widths[2], 10, str(Ee_val), 1, 0, 'C')
                pdf.cell(col_widths[3], 10, str(Ed_val), 1, 0, 'C')
                pdf.cell(col_widths[4], 10, str(h_val), 1, 0, 'C')
                pdf.cell(col_widths[5], 10, str(lambda_val), 1, 0, 'C')
                pdf.ln()
            
            pdf.ln(10)
            
            # Текстова репрезентация на графиките
            for i, layer in enumerate(st.session_state.layers_data):
                if 'hD_point' in layer and 'Ed' in layer and 'Ei' in layer:
                    pdf.set_font('DejaVu', 'B', 12)
                    pdf.cell(0, 10, f'Данни за пласт {i+1}:', 0, 1)
                    pdf.set_font('DejaVu', '', 10)
                    
                    hD_point = layer['hD_point']
                    EdEi_point = layer['Ed'] / layer['Ei']
                    
                    pdf.cell(0, 10, f'- h/D = {hD_point:.4f}', 0, 1)
                    pdf.cell(0, 10, f'- Ed/Ei = {EdEi_point:.4f}', 0, 1)
                    
                    if all(key in layer for key in ['y_low', 'y_high', 'low_iso', 'high_iso']):
                        pdf.cell(0, 10, f'- Интерполация между Ee/Ei = {layer["low_iso"]:.4f} и Ee/Ei = {layer["high_iso"]:.4f}', 0, 1)
                        pdf.cell(0, 10, f'- Стойности на изолиниите: y_low = {layer["y_low"]:.4f}, y_high = {layer["y_high"]:.4f}', 0, 1)
                    
                    pdf.ln(5)
            
            # Топлинни параметри
            if 'lambda_op_input' in st.session_state and 'lambda_zp_input' in st.session_state:
                lambda_op = st.session_state.lambda_op_input
                lambda_zp = st.session_state.lambda_zp_input
                m_value = lambda_zp / lambda_op
                z1 = st.session_state.get('z1_input', 100)
                z_value = z1 * m_value
                
                pdf.set_font('DejaVu', 'B', 14)
                pdf.cell(0, 10, 'Топлинни параметри', 0, 1)
                pdf.set_font('DejaVu', '', 12)
                pdf.cell(0, 10, f'λоп = {lambda_op} kcal/mhg', 0, 1)
                pdf.cell(0, 10, f'λзп = {lambda_zp} kcal/mhg', 0, 1)
                pdf.cell(0, 10, f'm = λзп / λоп = {lambda_zp} / {lambda_op} = {m_value:.2f}', 0, 1)
                pdf.cell(0, 10, f'z₁ = {z1} cm (дълбочина на замръзване в открито поле)', 0, 1)
                pdf.cell(0, 10, f'z = z₁ * m = {z1} * {m_value:.2f} = {z_value:.2f} cm', 0, 1)
                pdf.ln(10)
                
                # R₀ изчисление
                if all('h' in layer for layer in st.session_state.layers_data):
                    sum_h = sum(layer['h'] for layer in st.session_state.layers_data)
                    sum_lambda = sum(st.session_state.lambda_values)
                    R0 = sum_h / sum_lambda if sum_lambda != 0 else 0
                    
                    pdf.cell(0, 10, f'R₀ = Σh / Σλ = {sum_h:.2f} / {sum_lambda:.2f} = {R0:.2f} cm', 0, 1)
                    pdf.ln(10)
                
                # Проверка
                pdf.set_font('DejaVu', 'B', 14)
                pdf.cell(0, 10, 'Проверка на изискванията', 0, 1)
                pdf.set_font('DejaVu', '', 12)
                
                if all('h' in layer for layer in st.session_state.layers_data):
                    if z_value > sum_h:
                        pdf.cell(0, 10, '✅ Условието е изпълнено: z > Σh', 0, 1)
                        pdf.cell(0, 10, f'z = {z_value:.2f} cm > Σh = {sum_h:.2f} cm', 0, 1)
                    else:
                        pdf.cell(0, 10, '❌ Условието НЕ е изпълнено: z ≤ Σh', 0, 1)
                        pdf.cell(0, 10, f'z = {z_value:.2f} cm ≤ Σh = {sum_h:.2f} cm', 0, 1)
        
        # Добавяне на другите страници според избора
        # Тук можете да добавите съдържание от другите страници, ако е необходимо
        
        return pdf.output(dest='S').encode('latin1')
    
    # Генериране и сваляне на PDF
    try:
        with st.spinner('Генериране на PDF отчет...'):
            pdf_bytes = generate_pdf_report()
            
            # Показване на бутон за сваляне
            st.success("PDF отчетът е генериран успешно!")
            st.download_button(
                label="📥 Свали PDF отчет",
                data=pdf_bytes,
                file_name="patna_konstrukcia_report.pdf",
                mime="application/pdf"
            )
        
    except Exception as e:
        st.error(f"Грешка при генериране на PDF: {str(e)}")

# Добавяне на информация за шрифтовете
st.markdown("""
<div class="warning-box">
    <strong>Важно:</strong> За правилно генериране на PDF файлове на кирилица, 
    моля добавете следните файлове в поддиректория 'fonts' в същата директория като приложението:
    <ul>
        <li>DejaVuSans.ttf</li>
        <li>DejaVuSans-Bold.ttf</li>
        <li>DejaVuSans-Oblique.ttf</li>
    </ul>
</div>
""", unsafe_allow_html=True)
