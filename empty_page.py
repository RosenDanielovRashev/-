import streamlit as st

st.title("📄 Празна страница")
st.write("Това е празната страница.")

if st.button("⬅️ Върни се на основната страница"):
    st.switch_page("main")
