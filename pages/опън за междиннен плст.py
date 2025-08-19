# Add these imports at the top of the file (after existing imports)
from matplotlib import mathtext
from io import BytesIO

# Add CairoSVG availability check
try:
    import cairosvg
    _HAS_CAIROSVG = True
except Exception:
    _HAS_CAIROSVG = False

# Formula rendering functions (add before EnhancedPDF class)
def render_formula_to_svg(formula, output_path):
    """
    Рендва формула като SVG чрез matplotlib.mathtext.
    """
    parser = mathtext.MathTextParser("path")
    parser.to_svg(f"${formula}$", output_path)
    return output_path

def svg_to_png(svg_path, png_path=None, dpi=300):
    """
    Конвертира SVG към PNG с висока резолюция. Изисква cairosvg.
    """
    if not _HAS_CAIROSVG:
        raise RuntimeError("cairosvg не е наличен")
    if png_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
            png_path = tmp_png.name
    cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)
    return png_path

def render_formula_to_image_fallback(formula, fontsize=22, dpi=450):
    """
    Fallback: рендва формула директно в PNG чрез matplotlib (растерно, но висок DPI).
    """
    fig = plt.figure(figsize=(8, 2.5))
    fig.text(0.05, 0.5, f'${formula}$', fontsize=fontsize, ha='left', va='center', usetex=False)
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    buf.seek(0)
    return buf

# Replace the EnhancedPDF class with the one from file 1.py
class EnhancedPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_font_files = []
        self.temp_image_files = []

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')

    def add_font_from_bytes(self, family, style, font_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
            tmp_file.write(font_bytes)
            tmp_file_path = tmp_file.name
            self.temp_font_files.append(tmp_file_path)
            self.add_font(family, style, tmp_file_path)

    def add_external_image(self, image_path, width=180):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            img = Image.open(image_path)
            img.save(tmp_file, format='PNG')
            tmp_file_path = tmp_file.name
            self.temp_image_files.append(tmp_file_path)
        self.image(tmp_file_path, x=10, w=width)
        self.ln(10)

    def _formula_png_from_svg_or_fallback(self, formula_text, dpi=300):
        """
        Прави PNG път от формула чрез SVG→PNG, а ако няма cairosvg → fallback PNG буфер.
        Връща път към PNG файл, добавен към temp списъка.
        """
        try:
            # SVG временен файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as tmp_svg:
                render_formula_to_svg(formula_text, tmp_svg.name)
                # PNG от SVG
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
                    svg_to_png(tmp_svg.name, tmp_png.name, dpi=dpi)
                    png_path = tmp_png.name
            self.temp_image_files.append(png_path)
            return png_path
        except Exception:
            # Fallback: директно PNG от matplotlib
            buf = render_formula_to_image_fallback(formula_text, fontsize=22, dpi=450)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(buf.read())
                png_path = tmp_file.name
            self.temp_image_files.append(png_path)
            return png_path

    def add_latex_formula(self, formula_text, width=100, line_gap=12):
        """
        Добавя ЕДНА формула като изображение (векторен рендер до PNG), без фонови плочи.
        """
        try:
            png_path = self._formula_png_from_svg_or_fallback(formula_text)
            # Вмъкване с фиксирана ширина → еднакъв визуален размер
            self.image(png_path, x=self.get_x(), y=self.get_y(), w=width)
            # Приблизителен вертикален интервал
            self.ln(line_gap + width * 0.22)
        except Exception:
            self.set_font('DejaVu', 'I', 12)
            self.multi_cell(0, 8, formula_text)
            self.ln(5)

    def add_plotly_figure(self, fig, width=180):
        try:
            img_bytes = pio.to_image(
                fig,
                format="png",
                width=1200,
                height=900,
                scale=3,
                engine="kaleido"
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file_path = tmp_file.name
                self.temp_image_files.append(tmp_file_path)
            self.image(tmp_file_path, x=10, w=width)
            self.ln(10)
            return True
        except Exception as e:
            print(f"Грешка при добавяне на Plotly фигура: {e}")
            return False

    def cleanup_temp_files(self):
        for file_path in self.temp_font_files + self.temp_image_files:
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Грешка при изтриване на временен файл: {e}")

    def add_formula_section(self, title, formulas, columns=2, col_width=95, img_width=85, row_gap=8):
        """
        Секция с формули, подредени по колони, без фон и с еднакво мащабиране.
        - img_width контролира реалната ширина на всяка формула.
        """
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

        # Групираме по броя колони
        rows = [formulas[i:i+columns] for i in range(0, len(formulas), columns)]

        for row in rows:
            # Начална X позиция
            start_x = 10
            self.set_x(start_x)
            max_row_height = 0  # при нужда може да се развие за още по-точен вертикален интервал

            for idx, formula in enumerate(row):
                try:
                    png_path = self._formula_png_from_svg_or_fallback(formula)
                    # Картинка с фиксиран img_width за еднакъв размер
                    self.image(png_path, x=self.get_x(), y=self.get_y(), w=img_width)
                except Exception:
                    # Текстов fallback
                    self.set_font('DejaVu', '', 11)
                    self.multi_cell(col_width, 6, formula)
                # Преместваме в следващата колона
                self.set_x(start_x + col_width * (idx + 1))
                max_row_height = max(max_row_height, img_width * 0.28)

            # Нов ред с малък промеждутък
            self.ln(max(18, int(max_row_height)) + row_gap)

        self.ln(4)

# Update the generate_pdf_report_2 function
def generate_pdf_report_2():
    pdf = EnhancedPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Шрифтове
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    font_dir = os.path.join(base_dir, "fonts")
    os.makedirs(font_dir, exist_ok=True)

    sans_path = os.path.join(font_dir, "DejaVuSans.ttf")
    bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
    italic_path = os.path.join(font_dir, "DejaVuSans-Oblique.ttf")

    try:
        if all(os.path.exists(p) for p in [sans_path, bold_path, italic_path]):
            with open(sans_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', '', f.read())
            with open(bold_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', 'B', f.read())
            with open(italic_path, "rb") as f:
                pdf.add_font_from_bytes('DejaVu', 'I', f.read())
        else:
            from fpdf.fonts import FontsByFPDF
            fonts = FontsByFPDF()
            for style, data in [('', fonts.helvetica),
                                ('B', fonts.helvetica_bold),
                                ('I', fonts.helvetica_oblique)]:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp_file:
                    tmp_file.write(data)
                    pdf.add_font('DejaVu', style, tmp_file.name)
    except Exception as e:
        st.error(f"Грешка при зареждане на шрифтове: {e}")
        return b""

    # Заглавна страница
    pdf.add_page()
    pdf.set_font('DejaVu', 'B', 18)
    pdf.cell(0, 15, 'ОПЪН В МЕЖДИНЕН ПЛАСТ', ln=True, align='C')
    pdf.set_font('DejaVu', 'I', 12)
    pdf.cell(0, 10, 'Фигура 9.3 - Определяне опънното напрежение в междиен пласт', ln=True, align='C')
    pdf.ln(10)

    # 1. Входни параметри
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '1. Входни параметри', ln=True)

    col_width = 60
    row_height = 8

    pdf.set_font('DejaVu', 'B', 11)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(col_width, row_height, 'Параметър', border=1, align='C', fill=True)
    pdf.cell(col_width, row_height, 'Стойност', border=1, align='C', fill=True)
    pdf.cell(col_width, row_height, 'Мерна единица', border=1, align='C', fill=True)
    pdf.ln(row_height)

    pdf.set_font('DejaVu', '', 10)
    params = [
        ("Диаметър D", f"{D:.2f}", "cm"),
        ("Брой пластове", f"{n}", ""),
    ]
    for i in range(n):
        params.append((f"Пласт {i+1} - Ei", f"{E_values[i]:.2f}", "MPa"))
        params.append((f"Пласт {i+1} - hi", f"{h_values[i]:.2f}", "cm"))
        params.append((f"Пласт {i+1} - Ed", f"{Ed_values[i]:.2f}", "MPa"))
    params.append(("Избран пласт за проверка", f"{layer_idx+1}", ""))

    fill = False
    for p_name, p_val, p_unit in params:
        pdf.set_fill_color(245, 245, 245) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_width, row_height, p_name, border=1, fill=True)
        pdf.cell(col_width, row_height, p_val, border=1, align='C', fill=True)
        pdf.cell(col_width, row_height, p_unit, border=1, align='C', fill=True)
        pdf.ln(row_height)
        fill = not fill

    pdf.ln(10)

    # 2. Формули за изчисление
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '2. Формули за изчисление', ln=True)

    formulas_section2 = [
        r"H_{n-1} = \sum_{i=1}^{n-1} h_i",
        r"H_n = \sum_{i=1}^n h_i", 
        r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}",
        r"\frac{H_n}{D}",
        r"\frac{Esr}{E_n}",
        r"\frac{E_n}{Ed_n}",
        r"\sigma_R = 1.15 \cdot p \cdot \sigma_R^{\mathrm{номограма}}"
    ]
    
    pdf.add_formula_section("Основни формули за изчисление:", formulas_section2, columns=2, col_width=95, img_width=85, row_gap=-3)

    pdf.ln(5)

    # 3. Резултати от изчисленията
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '3. Резултати от изчисленията', ln=True)
    
    if layer_idx in st.session_state.layer_results:
        results = st.session_state.layer_results[layer_idx]
        
        # Prepare formulas with numerical values
        formulas_section3 = []
        
        # H_n-1
        if layer_idx > 0:
            h_terms = " + ".join([f"{h}" for h in results['h_values'][:layer_idx]])
            formulas_section3.append(fr"H_{{{layer_idx}}} = {h_terms} = {results['H_n_1_r']}")
        else:
            formulas_section3.append(fr"H_{{{layer_idx}}} = 0")
        
        # H_n
        h_terms_n = " + ".join([f"{h}" for h in results['h_values'][:results['n_for_calc']]])
        formulas_section3.append(fr"H_{{{results['n_for_calc']}}} = {h_terms_n} = {results['H_n_r']}")
        
        # Esr
        if layer_idx > 0:
            numerator = " + ".join([f"{E} \\times {h}" for E, h in zip(results['E_values'][:layer_idx], results['h_values'][:layer_idx])])
            denominator = " + ".join([f"{h}" for h in results['h_values'][:layer_idx]])
            formulas_section3.append(fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']}")
        else:
            formulas_section3.append("Esr = 0")
        
        # Other formulas
        formulas_section3.append(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
        formulas_section3.append(fr"E_{{{layer_idx+1}}} = {results['En_r']}")
        formulas_section3.append(fr"\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
        formulas_section3.append(fr"\frac{{E_{{{layer_idx+1}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")
        
        if 'final_sigma' in st.session_state:
            formulas_section3.append(fr"\sigma_R^{{\mathrm{{номограма}}}} = {st.session_state.final_sigma:.3f}")
        
        axle_load = st.session_state.get("axle_load", 100)
        p_loc = 0.620 if axle_load == 100 else 0.633 if axle_load == 115 else 0.0
        if p_loc and 'final_sigma' in st.session_state:
            sigma_final_loc = 1.15 * p_loc * st.session_state.final_sigma
            formulas_section3.append(fr"p = {p_loc:.3f} \, \text{{ (за осов товар {axle_load} kN)}}")
            formulas_section3.append(fr"\sigma_R = 1.15 \times {p_loc:.3f} \times {st.session_state.final_sigma:.3f} = {sigma_final_loc:.3f}")
        
        pdf.add_formula_section("Изчислителни формули:", formulas_section3, columns=2, col_width=95, img_width=85, row_gap=-3)

    pdf.ln(10)

    # 4. Графика
    if "fig" in st.session_state:
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '4. Графика на изолинии', ln=True)
        pdf.add_plotly_figure(st.session_state["fig"], width=160)

    # 5. Допустими напрежения
    img_path = "Допустими опънни напрежения.png"
    if os.path.exists(img_path):
        pdf.set_font('DejaVu', 'B', 14)
        pdf.cell(0, 10, '5. Допустими опънни напрежения', ln=True)
        pdf.add_external_image(img_path, width=160)

    # 6. Резултати и проверка
    pdf.set_font('DejaVu', 'B', 14)
    pdf.cell(0, 10, '6. Резултати и проверка', ln=True)

    if 'final_sigma_R' in st.session_state and f'manual_sigma_{layer_idx}' in st.session_state.manual_sigma_values:
        manual_value = st.session_state.manual_sigma_values[f'manual_sigma_{layer_idx}']
        check_passed = st.session_state.final_sigma_R <= manual_value

        pdf.set_font('DejaVu', 'B', 10)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(90, 8, 'Параметър', border=1, align='C', fill=True)
        pdf.cell(90, 8, 'Стойност', border=1, align='C', fill=True)
        pdf.ln(8)

        pdf.set_font('DejaVu', '', 10)
        for label, val in [
            ('Изчислено σR', f"{st.session_state.final_sigma_R:.3f} MPa"),
            ('Допустимо σR (ръчно)', f"{manual_value:.2f} MPa")
        ]:
            pdf.set_fill_color(245, 245, 245) if label.startswith('Изчислено') else pdf.set_fill_color(255, 255, 255)
            pdf.cell(90, 8, label, border=1, fill=True)
            pdf.cell(90, 8, val, border=1, align='C', fill=True)
            pdf.ln(8)

        pdf.ln(5)
        if check_passed:
            pdf.set_text_color(0, 100, 0)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, "✅ Проверка: УДОВЛЕТВОРЕНА", ln=True)
        else:
            pdf.set_text_color(150, 0, 0)
            pdf.set_font('DejaVu', 'B', 12)
            pdf.cell(0, 10, "❌ Проверка: НЕУДОВЛЕТВОРЕНА", ln=True)

        pdf.set_text_color(0, 0, 0)

    # Дата на генериране
    pdf.ln(10)
    pdf.set_font('DejaVu', 'I', 8)
    pdf.cell(0, 8, f"Генерирано на: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.cleanup_temp_files()
    return pdf.output(dest='S')
