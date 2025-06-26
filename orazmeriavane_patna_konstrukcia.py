# Премахваме st.title("📐 Калкулатор: Метод на Иванов (интерактивна версия)")

# Вместо това, директно използваме въведените стойности:
# h, Ee, Ei, d_value вече са дефинирани от предишната част на кода

if data is not None:
    mode = st.radio(
        "Изберете параметър за отчитане:",
        ("Ed / Ei", "h / D")
    )

    if Ei == 0 or d_value == 0:
        st.error("Ei и D не могат да бъдат 0.")
        st.stop()

    if mode == "Ed / Ei":
        # Използваме стойността на h, въведена в горната част:
        h_input = h
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

        if st.button("Изчисли Ed"):
            result, hD_point, y_low, y_high, low_iso, high_iso = compute_Ed(h_input, d_value, Ee, Ei)

            if result is None:
                st.warning("❗ Точката е извън обхвата на наличните изолинии.")
            else:
                EdEi_point = result / Ei
                st.success(f"✅ Изчислено: Ed / Ei = {EdEi_point:.3f}  \nEd = Ei * {EdEi_point:.3f} = {result:.2f} MPa")
                st.info(f"ℹ️ Интерполация между изолини: Ee / Ei = {low_iso:.3f} и Ee / Ei = {high_iso:.3f}")

                # Плот с Plotly
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
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Режим "h / D" – използваме въведените Ed, Ee, Ei, D
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
                    xaxis_title="h / D",
                    yaxis_title="Ed / Ei",
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
