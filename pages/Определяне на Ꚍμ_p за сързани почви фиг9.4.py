# ... (останалата част от кода остава същата)

# ============= КОРЕКЦИИ ТУК =============
# 1. Проверка за валидност на стойностите
min_esr_eo = df_esr_eo['Esr_Eo'].min()
max_esr_eo = df_esr_eo['Esr_Eo'].max()
min_fi = df_fi['fi'].min()
max_fi = df_fi['fi'].max()

if Esr_over_Ed < min_esr_eo or Esr_over_Ed > max_esr_eo:
    st.warning(f"Esr/Ed трябва да е между {min_esr_eo} и {max_esr_eo}!")
    point_on_esr_ed = None

if Fi_input < min_fi or Fi_input > max_fi:
    st.warning(f"Fi трябва да е между {min_fi} и {max_fi}!")

# 2. Корекция на изчислението за Ꚍμ/p
if point_on_esr_ed is not None:
    # ... (останалия код за визуализация)
    
    if x_orange is not None:
        # ПРАВИЛНОТО ИЗЧИСЛЕНИЕ (0.10 вместо /10)
        sigma_r = round(x_orange * 0.10, 3)  # Корекция тук!
        st.markdown(f"**Ꚍμ/p = {sigma_r}**")
        st.latex(r"\tau_{\mu}/p = \frac{H/D}{10} \times 0.10 = " + f"{x_orange:.3f} \times 0.10 = {sigma_r:.3f}")
    else:
        st.markdown("**Ꚍμ/p = -** (Неуспешна интерполация за Fi)")
else:
    st.markdown("**Ꚍμ/p = -** (Неуспешна интерполация за Esr/Ed)")

# 3. Корекция на коментарите (променено Eo → Ed)
# В целия код променете коментарите от:
#   # Изолинии Esr/Eo → # Изолинии Esr/Ed
#   # Esr/Eo → # Esr/Ed
