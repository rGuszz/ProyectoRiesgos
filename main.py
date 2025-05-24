import streamlit as st
import funciones as fn
import numpy as np

st.set_page_config(
    page_title="Portafolio de rendimiento deseado",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .dataframe td, .dataframe th {
        font-size: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Portafolio de rendimiento deseado")
tab1, tab2, tab3, tab4, tab6, tab7, tab8, tab9 = st.tabs(["Acciones escenario normal", "Portafolio escenario normal", "Simulación de Portafolios escenario normal", "Backtesting escenario normal", "Escenario COVID-19 Acciones", "Escenario COVID-19 Portafolio", "Escenario COVID-19 Backtesting", "Backtesting general"])

with tab1:
    st.subheader("Análisis de Acciones")
    # Nombre acciones
    nombre_acciones = [
        "NTAP",  # NetApp, Inc. (Tecnología)
        "MSFT",  # Microsoft Corp. (Tecnología)
        "JNJ",   # Johnson & Johnson (Salud)
        "XOM",   # Exxon Mobil Corporation (Energía)
        "V",     # Visa Inc. (Servicios Financieros)
        "GOOGL", # Alphabet Inc. (Tecnología)
        "PG",    # Procter & Gamble Co. (Consumo)
        "KO",    # Coca-Cola Co. (Consumo)
        "NVDA",  # NVIDIA Corporation (Tecnología)
        "WMT"    # Walmart Inc. (Retail)
    ]
    precios = fn.precios(nombre_acciones)
    precios_ordenados = precios.sort_index(ascending=False)
    pesos = [0.023531, 0.063926, 0.18491, 0.095493, 0.040738, 0.034043, 0.084526, 0.253479, 0.067492, 0.151863]
    monto = 10000
    VaR_Par = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[1]
    uso_VaR_RD = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[3]
    pos_activo = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[4]

    fecha_de_datos = precios.index[-1]
    fecha_de_datos = fecha_de_datos.strftime("%Y-%m-%d")
    st.markdown(f"**Fecha de datos:** {fecha_de_datos}")

    col1, col2 = st.columns(2)
    with col1:   
        # Información de acciones
        info_acciones = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[0]
        st.dataframe(info_acciones, use_container_width=True, height=497, row_height=51)
    
    with col2:
        @st.fragment
        def fragment():
            col3, col4 = st.columns(2)
            with col3:
                seleccion = st.selectbox("Selecciona la grafica", ["Precios", "Uso VaR Barplot", "Heatmap correlaciones", "Rendimiento", "Histograma rendimiento", "Boxplot rendimiento"])
            with col4:
                if seleccion == "Precios" or seleccion == "Uso VaR Barplot" or seleccion == "Heatmap correlaciones":
                    st.text("")
                else:
                    accion = st.selectbox("Selecciona una acción", nombre_acciones)
            if seleccion == "Precios":
                # Graficar precios de acciones
                fig = fn.graficar_precios(precios, nombre_acciones)
                st.plotly_chart(fig, use_container_width=True)

            elif seleccion == "Rendimiento":
                # Graficar rendimiento de acciones
                fig2 = fn.graficar_rendimientos(precios, accion, pos_activo, nombre_acciones, VaR_Par)
                st.plotly_chart(fig2, use_container_width=True)

            elif seleccion == "Histograma rendimiento":
                # Graficar histograma de rendimiento de acciones
                fig3 = fn.graficar_histograma_log(precios, accion, pos_activo, VaR_Par, nombre_acciones)
                st.plotly_chart(fig3, use_container_width=True)

            elif seleccion == "Uso VaR Barplot":
                # Graficar uso de VaR
                fig4 = fn.graficar_barplot_uso_var(uso_VaR_RD, nombre_acciones)
                st.plotly_chart(fig4, use_container_width=True)

            elif seleccion == "Heatmap correlaciones": 
                # Graficar heatmap de correlaciones
                fig5 = fn.graficar_heatmap_correlaciones(precios, nombre_acciones)
                st.plotly_chart(fig5, use_container_width=True)
            
            elif seleccion == "Boxplot rendimiento":
                # Graficar boxplot de rendimiento
                fig6 = fn.graficar_boxplot_rendimientos(precios, pos_activo, nombre_acciones, accion)
                st.plotly_chart(fig6, use_container_width=True)
        fragment()
    st.dataframe(precios_ordenados, use_container_width=True, height=275)

with tab2:
    st.subheader("Análisis de Portafolio")
    fecha_de_datos = precios.index[-1]
    fecha_de_datos = fecha_de_datos.strftime("%Y-%m-%d")
    st.markdown(f"**Fecha de datos:** {fecha_de_datos}")
    VaR_Par = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[1]
    val_activo = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[2]
    uso_VaR_RD = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[3]
    info_port = fn.informacion_portafolio(precios, pesos, monto, VaR_Par, val_activo, uso_VaR_RD)[0]
    VaR_port = fn.informacion_portafolio(precios, pesos, monto, VaR_Par, val_activo, uso_VaR_RD)[1]
    precios4y = fn.precios_periodo(precios, 4)
    col1, col2 = st.columns(2)
    with col1:
        # Información de portafolio
        st.dataframe(info_port, use_container_width=True, height=493, row_height=65)
    with col2:
        @st.fragment
        def fragment():
            selecc = st.selectbox("Selecciona la grafica", ["Rendimiento portafolio", "Rendimiento acumulado portafolio", "Histograma portafolio"])
            # Graficar portafolio
            if selecc == "Rendimiento portafolio":
                fig = fn.graficar_rendimientos_portafolio(precios, pesos, monto)
                st.plotly_chart(fig, use_container_width=True)
            elif selecc == "Rendimiento acumulado portafolio":
                fig2 = fn.graficar_rendimiento_acumulado(precios, pesos, monto)
                st.plotly_chart(fig2, use_container_width=True)
            elif selecc == "Histograma portafolio":
                fig3 = fn.graficar_histograma_portafolio(precios, pesos, monto, VaR_port)
                st.plotly_chart(fig3, use_container_width=True)
        fragment()

with tab3:
    st.subheader("Simulación de Portafolios")
    precios = fn.precios(nombre_acciones)
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    esperanza_activo = fn.informacion_activos(precios, pesos, monto, nombre_acciones)[5]
    var_port_rend_dado = fn.informacion_portafolio(precios, pesos, monto, VaR_Par, val_activo, uso_VaR_RD)[2]
    esp_port_rend_dado = fn.informacion_portafolio(precios, pesos, monto, VaR_Par, val_activo, uso_VaR_RD)[3]
    @st.fragment
    def fragment():
        num_sim = st.number_input("Número de simulaciones", min_value=10, max_value=30000, step=100, value=10)
        # Simulación de portafolios
        fig = fn.graficar_simulacion_portafolio(nombre_acciones, rendimientos, esperanza_activo, tasa_cetes=0.0837, var_port_rend_dado=var_port_rend_dado, esp_port_rend_dado=esp_port_rend_dado, num_simulaciones=num_sim)
        st.plotly_chart(fig, use_container_width=True)
    fragment()

with tab4:
    st.subheader("Backtesting")
    # Nombre acciones
    nombre_acciones = [
        "NTAP",  # NetApp, Inc. (Tecnología)
        "MSFT",  # Microsoft Corp. (Tecnología)
        "JNJ",   # Johnson & Johnson (Salud)
        "XOM",   # Exxon Mobil Corporation (Energía)
        "V",     # Visa Inc. (Servicios Financieros)
        "GOOGL", # Alphabet Inc. (Tecnología)
        "PG",    # Procter & Gamble Co. (Consumo)
        "KO",    # Coca-Cola Co. (Consumo)
        "NVDA",  # NVIDIA Corporation (Tecnología)
        "WMT"    # Walmart Inc. (Retail)
    ]
    precios = fn.precios(nombre_acciones)
    precios4y = fn.precios_periodo(precios, 4)
    pesos = [0.023531, 0.063926, 0.18491, 0.095493, 0.040738, 0.034043, 0.084526, 0.253479, 0.067492, 0.151863]
    monto = 10000

    fig4 = fn.graficar_var_rolling_portafolio(precios4y, pesos, monto, 504, 0.95)
    st.plotly_chart(fig4, use_container_width=True)


with tab6:
    st.subheader("Análisis de Acciones")
    # Nombre acciones
    nombre_acciones = [
        "NTAP",  # NetApp, Inc. (Tecnología)
        "MSFT",  # Microsoft Corp. (Tecnología)
        "JNJ",   # Johnson & Johnson (Salud)
        "XOM",   # Exxon Mobil Corporation (Energía)
        "V",     # Visa Inc. (Servicios Financieros)
        "GOOGL", # Alphabet Inc. (Tecnología)
        "PG",    # Procter & Gamble Co. (Consumo)
        "KO",    # Coca-Cola Co. (Consumo)
        "NVDA",  # NVIDIA Corporation (Tecnología)
        "WMT"    # Walmart Inc. (Retail)
    ]
    precios_cov = fn.precios_periodo_fechas(nombre_acciones, "2020-1-1", "2023-5-14")
    precios_ordenados_cov = precios_cov.sort_index(ascending=False)
    pesos = [0.023531, 0.063926, 0.18491, 0.095493, 0.040738, 0.034043, 0.084526, 0.253479, 0.067492, 0.151863]
    monto = 10000
    VaR_Par = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[1]
    uso_VaR_RD = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[3]
    pos_activo = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[4]

    fecha_de_datos = precios_cov.index[-1]
    fecha_de_datos = fecha_de_datos.strftime("%Y-%m-%d")
    st.markdown(f"**Fecha de datos:** {fecha_de_datos}")

    col1, col2 = st.columns(2)
    with col1:   
        # Información de acciones
        info_acciones = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[0]
        st.dataframe(info_acciones, use_container_width=True, height=497, row_height=51)
    
    with col2:
        @st.fragment
        def fragment():
            col3, col4 = st.columns(2)
            with col3:
                seleccion = st.selectbox("Selecciona la grafica", ["Precios", "Uso VaR Barplot", "Heatmap correlaciones", "Rendimiento", "Histograma rendimiento", "Boxplot rendimiento"], key="2")
            with col4:
                if seleccion == "Precios" or seleccion == "Uso VaR Barplot" or seleccion == "Heatmap correlaciones":
                    st.text("")
                else:
                    accion = st.selectbox("Selecciona una acción", nombre_acciones)
            if seleccion == "Precios":
                # Graficar precios de acciones
                fig = fn.graficar_precios(precios_cov, nombre_acciones)
                st.plotly_chart(fig, use_container_width=True)

            elif seleccion == "Rendimiento":
                # Graficar rendimiento de acciones
                fig2 = fn.graficar_rendimientos(precios_cov, accion, pos_activo, nombre_acciones, VaR_Par)
                st.plotly_chart(fig2, use_container_width=True)

            elif seleccion == "Histograma rendimiento":
                # Graficar histograma de rendimiento de acciones
                fig3 = fn.graficar_histograma_log(precios_cov, accion, pos_activo, VaR_Par, nombre_acciones)
                st.plotly_chart(fig3, use_container_width=True)

            elif seleccion == "Uso VaR Barplot":
                # Graficar uso de VaR
                fig4 = fn.graficar_barplot_uso_var(uso_VaR_RD, nombre_acciones)
                st.plotly_chart(fig4, use_container_width=True)

            elif seleccion == "Heatmap correlaciones": 
                # Graficar heatmap de correlaciones
                fig5 = fn.graficar_heatmap_correlaciones(precios_cov, nombre_acciones)
                st.plotly_chart(fig5, use_container_width=True)
            
            elif seleccion == "Boxplot rendimiento":
                # Graficar boxplot de rendimiento
                fig6 = fn.graficar_boxplot_rendimientos(precios_cov, pos_activo, nombre_acciones, accion)
                st.plotly_chart(fig6, use_container_width=True)
        fragment()
    st.dataframe(precios_ordenados_cov, use_container_width=True, height=275)

with tab7:
    st.subheader("Análisis de Portafolio")
    VaR_Par_cov = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[1]
    val_activo_cov = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[2]
    uso_VaR_RD_cov = fn.informacion_activos(precios_cov, pesos, monto, nombre_acciones)[3]
    info_port_cov = fn.informacion_portafolio(precios_cov, pesos, monto, VaR_Par_cov, val_activo_cov, uso_VaR_RD_cov)[0]
    VaR_port_cov = fn.informacion_portafolio(precios_cov, pesos, monto, VaR_Par_cov, val_activo_cov, uso_VaR_RD_cov)[1]
    col1, col2 = st.columns(2)
    with col1:
        # Información de portafolio
        st.dataframe(info_port_cov, use_container_width=True, height=493, row_height=65)
    with col2:
        @st.fragment
        def fragment():
            selecc = st.selectbox("Selecciona la grafica", ["Rendimiento portafolio", "Rendimiento acumulado portafolio", "Histograma portafolio"], key="3")
            # Graficar portafolio
            if selecc == "Rendimiento portafolio":
                fig_cov = fn.graficar_rendimientos_portafolio(precios_cov, pesos, monto)
                st.plotly_chart(fig_cov, use_container_width=True)
            elif selecc == "Rendimiento acumulado portafolio":
                fig2_cov = fn.graficar_rendimiento_acumulado(precios_cov, pesos, monto)
                st.plotly_chart(fig2_cov, use_container_width=True)
            elif selecc == "Histograma portafolio":
                fig3_cov = fn.graficar_histograma_portafolio(precios_cov, pesos, monto, VaR_port_cov)
                st.plotly_chart(fig3_cov, use_container_width=True)
        fragment()
with tab8:
    @st.fragment
    def fragmento():
        st.subheader("Backtesting")
        # Nombre acciones
        nombre_acciones = [
            "NTAP",  # NetApp, Inc. (Tecnología)
            "MSFT",  # Microsoft Corp. (Tecnología)
            "JNJ",   # Johnson & Johnson (Salud)
            "XOM",   # Exxon Mobil Corporation (Energía)
            "V",     # Visa Inc. (Servicios Financieros)
            "GOOGL", # Alphabet Inc. (Tecnología)
            "PG",    # Procter & Gamble Co. (Consumo)
            "KO",    # Coca-Cola Co. (Consumo)
            "NVDA",  # NVIDIA Corporation (Tecnología)
            "WMT"    # Walmart Inc. (Retail)
        ]
        fecha_de_inicio = "2018-1-1"
        fecha_final = "2023-5-14"
        window = 504
        conf = 0.95

        precios4y_cov = fn.precios_periodo_fechas(nombre_acciones, fecha_de_inicio, fecha_final)
        pesos = [0.023531, 0.063926, 0.18491, 0.095493, 0.040738, 0.034043, 0.084526, 0.253479, 0.067492, 0.151863]
        monto = 10000

        fig4 = fn.graficar_var_rolling_portafolio(precios4y_cov, pesos, monto, window, conf)
        st.plotly_chart(fig4, use_container_width=True)
    fragmento()
with tab9:
    @st.fragment
    def fragmento():
        st.subheader("Backtesting")
        # Nombre acciones
        nombre_acciones = [
            "NTAP",  # NetApp, Inc. (Tecnología)
            "MSFT",  # Microsoft Corp. (Tecnología)
            "JNJ",   # Johnson & Johnson (Salud)
            "XOM",   # Exxon Mobil Corporation (Energía)
            "V",     # Visa Inc. (Servicios Financieros)
            "GOOGL", # Alphabet Inc. (Tecnología)
            "PG",    # Procter & Gamble Co. (Consumo)
            "KO",    # Coca-Cola Co. (Consumo)
            "NVDA",  # NVIDIA Corporation (Tecnología)
            "WMT"    # Walmart Inc. (Retail)
        ]
        colum1, colum2, colum3, colum4 = st.columns(4)
        with colum1:
            fecha_de_inicio = st.text_input("Fecha de inicio", value="2020-3-11")
        with colum2:    
            fecha_final = st.text_input("Fecha final", value="2023-5-14")
        with colum3:
            window = st.number_input("Ventana de tiempo en dias para calculo de VaR", min_value=1, max_value=1000, value=504)
        with colum4:
            conf = st.number_input("Nivel de confianza", min_value=0.80, max_value=0.99, value=0.95)

        precios4y_cov = fn.precios_periodo_fechas(nombre_acciones, fecha_de_inicio, fecha_final)
        pesos = [0.023531, 0.063926, 0.18491, 0.095493, 0.040738, 0.034043, 0.084526, 0.253479, 0.067492, 0.151863]
        monto = 10000

        fig4 = fn.graficar_var_rolling_portafolio(precios4y_cov, pesos, monto, window, conf)
        st.plotly_chart(fig4, use_container_width=True)
    fragmento()
