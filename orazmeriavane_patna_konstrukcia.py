import streamlit as st

# Заглавие
st.title("Оразмеряване на пътна конструкция")

# Секция за въвеждане на данни
st.subheader("Въведете характеристики")

# Падащо меню за D (тук под заглавието на характеристиките)
d_value_str = st.selectbox("Изберете стойност за D:", options=["32.04", "34"])
d_value = float(d_value_str)  # конвертираме към число

# Падащо меню за осов товар
axle_load = st.selectbox("Изберете стойност за осов товар (kN):", options=["100", "115"])
st.write(f"Избрана стойност за осов товар: {axle_load} kN")

# Секция за въвеждане на броя на пластовете
st.subheader("Въведете брой на пластовете")
num_layers = st.number_input("Брой пластове:", min_value=1, step=1)
st.write(f"Въведен брой пластове: {int(num_layers)}")

# Секция за въвеждане на данните за оразмеряване - Пласт 1
st.subheader("Въведете данни за оразмеряване - Пласт 1")

# Показване на вече избраната стойност D (само за четене)
st.write(f"Стойност D за пласт 1: {d_value}")

# Въвеждане на Ee в секцията на пласт 1 (под D)
Ee = st.number_input("Въведете стойност за Ee (MPa):", min_value=0.0, step=0.1)

# Дебелина h на пласт 1
h = st.number_input("Дебелина h на пласт 1 (cm):", min_value=1.0, step=0.1)

# Въвеждане на Ei за пласт 1
Ei = st.number_input("Модул на еластичност Ei на пласт 1 (MPa):", min_value=1.0, step=0.1)

# Изчисляваме формулите, ако Ei и d_value не са 0, за да избегнем деление на 0
if Ei > 0 and d_value > 0:
    ratio_h_D = h / d_value
    ratio_Ee_Ei = Ee / Ei if Ei != 0 else None
    
    st.subheader("Резултати от изчисленията")
    st.latex(r" \frac{h}{D} = " + f"{ratio_h_D:.3f}")
    st.latex(r" \frac{Ee}{Ei} = " + f"{ratio_Ee_Ei:.3f}")
else:
    st.write("Моля, въведете валидни стойности за Ei и D за изчисления.")
