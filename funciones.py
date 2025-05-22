# Importar librerias

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px1
import plotly.io as pio
import scipy.stats as stats

# region Nombre acciones
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

# endregion

def precios(nombre_acciones):
    """
    Función para descargar precios de acciones de Yahoo Finance
    :param nombre_acciones: lista de tickers
    :return: DataFrame con precios de cierre
    """
    # Creamos una lista vacía
    recolector = []

    # Inicamos con un for que intere sobre cada activo
    for nemo in nombre_acciones:

        # Creamos "ticker" con ayuda de YF obtenemos acceso a a esa info
        ticker = yf.Ticker(nemo)

        # Utilizamos .history
        px = ticker.history(period="2y")['Close']

        # Asignar el nombre de los precios descargados (funciona como una especie de iD)
        px.name = nemo
        recolector.append(px)

    # Rellenamos los valores NaN
    precios = pd.concat(recolector, axis=1)
    precios = precios.ffill()

    return precios

def graficar_precios(precios, nombre_acciones):
    """
    Función para graficar precios de acciones con paleta en azules.
    :param precios: DataFrame con precios de cierre
    :return: gráfico de líneas
    """
    # Paleta de azules (puedes ajustar los tonos si quieres más variedad)
    blue_palette = [
        "#0d47a1", "#1976d2", "#2196f3", "#42a5f5", "#64b5f6",
        "#90caf9", "#1565c0", "#1e88e5", "#0288d1", "#039be5"
    ]
    fig = go.Figure()
    for i, accion in enumerate(nombre_acciones):
        color = blue_palette[i % len(blue_palette)]
        fig.add_trace(go.Scatter(
            x=precios.index,
            y=precios[accion],
            mode='lines',
            name=accion,
            line=dict(color=color)
        ))

    fig.update_layout(
        title="Precio Histórico de los Activos",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        legend_title="Activos"
    )

    return fig

def informacion_activos(precios, pesos_wi_rd, monto, nombre_acciones):

    # Calculamos los rendimientos logarítmicos
    rendimientos = np.log(precios / precios.shift(1))
    rendimientos = rendimientos.dropna()

    # Calculamos la desviación estándar de los activos
    std_activo = rendimientos.std()
    
    # Calculamos el número de títulos de cada activo (entero)
    num_titulos = np.floor(np.array(pesos_wi_rd) * monto / precios.iloc[-1, :]).astype(int)

    # Calculamos la valuación de los activos (dinero con 2 decimales)
    val_activos = (num_titulos * precios.iloc[-1, :]).round(2)

    # Calculamos la posición de los activos (dinero con 2 decimales)
    pos_activos = (np.array(pesos_wi_rd) * monto).round(2)

    # Calculamos el cuantil
    alpha = 1 - 95/100
    z_alpha_2 = stats.norm.ppf(1-alpha/2)

    # Calculamos el VaR paramétrico (dinero con 2 decimales)
    VaR_Par = (z_alpha_2 * std_activo * pos_activos).round(2)

    # Calculamos la esperanza (rendimiento en porcentaje con 2 decimales)
    rend_activos = rendimientos.mean()
    rend_activos_pct = (rend_activos * 100).round(2)

    # Calculamos el riesgo de los activos (porcentaje con 2 decimales)
    riesgo_activos = (std_activo * pesos_wi_rd * 100).round(2)

    # Uso del VaR (porcentaje con 2 decimales)
    uso_VaR_RD = ((VaR_Par / val_activos) * 100).round(2)

    # Crear un DataFrame con los resultados calculados
    df_port_RD_activo = pd.DataFrame({
        nombre_acciones[i]: [
            precios.iloc[-1, i],             # Precio de los activos hoy (último precio)
            pesos_wi_rd[i],                  # Pesos de los activos
            rend_activos_pct[i] if hasattr(rend_activos_pct, "__len__") else rend_activos_pct,  # Esperanza activo (%)
            riesgo_activos[i],               # Riesgo del activo (%)
            VaR_Par[i],                      # VaR paramétrico ($)
            uso_VaR_RD[i],                   # Uso del VaR (%)
            pos_activos[i],                  # Posición de los activos ($)
            val_activos[i]                   # Valuación real del activo ($)
        ] for i in range(len(nombre_acciones))
    }, index=[
        "Precio de los activos hoy",
        "Pesos de los activos",
        "Esperanza activo (%)",
        "Riesgo del activo (%)",
        "VaR parametrico ($)",
        "Uso del VaR (%)",
        "Posición de los activos ($)",
        "Valuación real del activo ($)"
    ])

    # Agregar fila de número de títulos (sin decimales)
    df_port_RD_activo.loc["Número de títulos"] = num_titulos

    # Reordenar para que 'Número de títulos' esté después de 'Pesos de los activos'
    idx = df_port_RD_activo.index.tolist()
    idx.insert(2, idx.pop(idx.index("Número de títulos")))
    df_port_RD_activo = df_port_RD_activo.reindex(idx)

    # Formatear los valores en dinero y porcentaje
    money_rows = [
        "Precio de los activos hoy",
        "VaR parametrico ($)",
        "Posición de los activos ($)",
        "Valuación real del activo ($)"
    ]
    percent_rows = [
        "Esperanza activo (%)",
        "Riesgo del activo (%)",
        "Uso del VaR (%)"
    ]

    for row in money_rows:
        df_port_RD_activo.loc[row] = df_port_RD_activo.loc[row].apply(
            lambda x: "${:,.2f}".format(x)
        )
    for row in percent_rows:
        df_port_RD_activo.loc[row] = df_port_RD_activo.loc[row].apply(
            lambda x: "{:.2f}%".format(x)
        )
    # Pesos de los activos también en porcentaje
    df_port_RD_activo.loc["Pesos de los activos"] = df_port_RD_activo.loc["Pesos de los activos"].apply(
        lambda x: "{:.2f}%".format(x * 100)
    )
    # Número de títulos como entero sin formato
    df_port_RD_activo.loc["Número de títulos"] = df_port_RD_activo.loc["Número de títulos"].astype(int)

    # Cambiar el nombre del índice
    df_port_RD_activo.index.name = "Valuación diaria"

    return df_port_RD_activo, VaR_Par, val_activos, uso_VaR_RD, pos_activos, rend_activos_pct, riesgo_activos

def informacion_portafolio(precios, pesos_wi_rd, monto, VaR_Par, val_activos, uso_VaR_RD):
    # Calculamos los rendimientos logarítmicos
    rendimientos = np.log(precios / precios.shift(1))
    rendimientos = rendimientos.dropna()

    # Calculamos la matriz de varianza-covarianza
    matriz_var_principal = np.cov(rendimientos, y = None, rowvar=0, bias=True, ddof=None, fweights=None, aweights=None, dtype=None)
    
    # Calculamos la esperanza del portafolio
    rend_port = np.matmul(rendimientos, pesos_wi_rd)

    # Calculamos el riesgo del portafolio
    riesgo_port = np.sqrt(np.matmul(pesos_wi_rd, np.matmul(matriz_var_principal, pesos_wi_rd)))

    # Calculamos el VaR paramétrico del portafolio
    alpha = 1 - 95/100
    z_alpha_2 = stats.norm.ppf(1-alpha/2)
    VaR_port_rd = z_alpha_2 * riesgo_port * monto

    # Calculamos el VaR no diversificado
    VaR_par_nd_RD = sum(VaR_Par)

    # Calculamos el CVaR
    CVaR_RD = np.mean(uso_VaR_RD/100)

    # Calculamos la posición del portafolio
    pos_port_rd = sum(pesos_wi_rd * monto)

    # Calculamos la valuación del portafolio
    val_port_rd = sum(val_activos)

    # Crear un DataFrame con los resultados calculados
    df_port_RD = pd.DataFrame({
        "Esperanza del portafolio (%)": [rend_port.mean() * 100],
        "Riesgo del portafolio (%)": [riesgo_port * 100],
        "VaR paramétrico ($)": [VaR_port_rd],
        "VaR no diversificado ($)": [VaR_par_nd_RD],
        "CVaR (%)": [CVaR_RD * 100],
        "Posición del portafolio ($)": [pos_port_rd],
        "Valuación del portafolio ($)": [val_port_rd]
    })

    # Formatear los valores en dinero y porcentaje
    # Reorganizar el DataFrame para que los títulos sean el índice y los valores estén en una sola columna
    df_port_RD = df_port_RD.T
    df_port_RD.columns = ["Portafolio de rendimiento deseado (16%) Anual"]

    # Formatear los valores en dinero y porcentaje
    money_rows = [
        "VaR paramétrico ($)",
        "VaR no diversificado ($)",
        "Posición del portafolio ($)",
        "Valuación del portafolio ($)"]
    percent_rows = [
        "Esperanza del portafolio (%)",
        "Riesgo del portafolio (%)",
        "CVaR (%)"]

    for row in money_rows:
        if row in df_port_RD.index:
            df_port_RD.loc[row] = df_port_RD.loc[row].apply(lambda x: "${:,.2f}".format(x))
    for row in percent_rows:
        if row in df_port_RD.index:
            df_port_RD.loc[row] = df_port_RD.loc[row].apply(lambda x: "{:.2f}%".format(x))

    # Cambiar el nombre del índice y columna
    df_port_RD.index.name = "Resultados del portafolio"

    return df_port_RD, VaR_port_rd, riesgo_port, rend_port.mean()


def graficar_rendimientos(precios, accion, pos_activo, nombre_acciones, var_dinero):
    """
    Grafica los rendimientos logarítmicos en dinero para una acción específica,
    multiplicando el rendimiento logarítmico por el monto invertido (pos_activo).
    Agrega una línea horizontal roja en -VaR y puntos rojos donde el rendimiento es menor al VaR.
    Muestra en la leyenda el número y porcentaje de veces que el rendimiento fue menor al VaR.
    :param precios: DataFrame con precios de cierre
    :param accion: str, ticker de la acción
    :param pos_activo: np.array con posiciones en dinero para cada acción (mismo orden que nombre_acciones)
    :param nombre_acciones: lista de tickers, mismo orden que pos_activo
    :param var_dinero: np.array con VaR en dinero para cada acción (mismo orden que nombre_acciones)
    :return: figura de plotly
    """
    rendimientos = np.log(precios[accion] / precios[accion].shift(1)).dropna()
    idx = nombre_acciones.index(accion)
    rendimientos_dinero = rendimientos * pos_activo[idx]
    var_val = -1 * var_dinero[idx]

    # Encuentra los puntos donde el rendimiento es menor al VaR
    mask = rendimientos_dinero < var_val
    num_menores = mask.sum()
    total = len(rendimientos_dinero)
    porcentaje = (num_menores / total) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rendimientos_dinero.index,
        y=rendimientos_dinero,
        mode='lines',
        name=f"Rendimientos en dinero {accion}",
        line=dict(color='rgba(0, 123, 255, 0.7)')
    ))
    # Línea horizontal roja en -VaR
    fig.add_shape(
        type="line",
        x0=rendimientos_dinero.index[0],
        x1=rendimientos_dinero.index[-1],
        y0=var_val,
        y1=var_val,
        line=dict(color="red", width=2, dash="dash"),
        xref="x",
        yref="y"
    )
    # Puntos rojos donde el rendimiento es menor al VaR
    fig.add_trace(go.Scatter(
        x=rendimientos_dinero.index[mask],
        y=rendimientos_dinero[mask],
        mode='markers',
        name=f"Menor a -VaR: {num_menores} veces ({porcentaje:.2f}%)",
        marker=dict(color='red', size=8, symbol='circle')
    ))
    fig.update_layout(
        title=f"Rendimientos Logarítmicos en Dinero - {accion}",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Logarítmico en Dinero",
        legend_title="Acción"
    )
    return fig

def graficar_histograma_log(precios, accion, pos_activo, var_dinero, nombre_acciones):
    """
    Grafica un histograma de rendimientos logarítmicos en dinero para una acción específica,
    usando la posición del activo (dinero invertido en esa acción), e indica el VaR en dinero con una línea vertical.
    Los valores menores a var_val se pintan de rojo.
    :param precios: DataFrame con precios de cierre
    :param accion: str, ticker de la acción
    :param pos_activo: np.array con posiciones en dinero para cada acción (mismo orden que nombre_acciones)
    :param var_dinero: np.array con VaR en dinero para cada acción (ordenado igual que nombre_acciones)
    :param nombre_acciones: lista de tickers, mismo orden que var_dinero y pos_activo
    :return: figura de plotly
    """
    # Calcular rendimientos logarítmicos
    rend_log = np.log(precios[accion] / precios[accion].shift(1)).dropna()
    # Obtener el índice de la acción
    idx = nombre_acciones.index(accion)
    # Rendimiento en dinero usando la posición del activo
    rend_log_dinero = rend_log * pos_activo[idx]
    var_val = -var_dinero[idx]

    # Separar los valores menores y mayores/iguales a var_val
    menores = rend_log_dinero[rend_log_dinero < var_val]
    mayores = rend_log_dinero[rend_log_dinero >= var_val]

    # Crear histograma con dos colores
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mayores,
        nbinsx=60,
        marker_color='blue',
        opacity=0.75,
        name='>= VaR'
    ))
    fig.add_trace(go.Histogram(
        x=menores,
        nbinsx=60,
        marker_color='red',
        opacity=0.75,
        name='< VaR'
    ))

    # Agregar línea vertical para el VaR
    fig.add_shape(
        type="line",
        x0=var_val, x1=var_val,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=var_val,
        y=1,
        yref="paper",
        text="VaR",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=0,
        font=dict(color="red")
    )

    fig.update_layout(
        title=f"Histograma de Rendimientos Logarítmicos en Dinero - {accion}",
        xaxis_title="Rendimiento Logarítmico en Dinero",
        yaxis_title="Frecuencia",
        barmode='overlay',
        showlegend=True
    )
    return fig

def graficar_barplot_uso_var(uso_VaR_RD, nombre_acciones):
    """
    Grafica un barplot ordenado de mayor a menor según el uso del VaR.
    :param uso_VaR_RD: np.array o lista con el uso del VaR (%) para cada activo
    :param nombre_acciones: lista de tickers, mismo orden que uso_VaR_RD
    :return: figura de plotly
    """
    # Convertir a DataFrame para ordenar
    df = pd.DataFrame({
        'Acción': nombre_acciones,
        'Uso_VaR': uso_VaR_RD
    })
    df = df.sort_values('Uso_VaR', ascending=False)

    fig = px1.bar(
        df,
        x='Acción',
        y='Uso_VaR',
        text='Uso_VaR',
        color='Uso_VaR',
        color_continuous_scale='Blues',
        title='Uso del VaR por Activo (Ordenado de Mayor a Menor)'
    )
    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig.update_layout(
        xaxis_title='Acción',
        yaxis_title='Uso del VaR (%)',
        coloraxis_showscale=False
    )
    return fig

def graficar_heatmap_correlaciones(precios, nombre_acciones):
    """
    Grafica un heatmap de correlaciones entre los activos usando rendimientos logarítmicos.
    :param precios: DataFrame con precios de cierre
    :param nombre_acciones: lista de tickers
    :return: figura de plotly
    """
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    correlaciones = rendimientos.corr()

    fig = px1.imshow(
        correlaciones,
        x=nombre_acciones,
        y=nombre_acciones,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        text_auto='.2f'
    )
    fig.update_traces(
        textfont_size=15
    )
    fig.update_layout(
        xaxis_title='Activo',
        yaxis_title='Activo',
        width=900,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def graficar_boxplot_rendimientos(precios, pos_activos, nombre_acciones, accion):
    """
    Grafica un boxplot horizontal de los rendimientos logarítmicos en dinero para una acción específica,
    mostrando también los puntos individuales.
    :param precios: DataFrame con precios de cierre
    :param pos_activos: np.array o lista con posiciones en dinero para cada activo (mismo orden que nombre_acciones)
    :param nombre_acciones: lista de tickers
    :param accion: str, ticker de la acción a graficar
    :return: figura de plotly
    """
    rendimientos_log = np.log(precios[accion] / precios[accion].shift(1)).dropna()
    idx = nombre_acciones.index(accion)
    rend_dinero = rendimientos_log * pos_activos[idx]
    df = pd.DataFrame({
        'Acción': [accion] * len(rend_dinero),
        'Rendimiento en Dinero': rend_dinero
    })

    fig = px1.box(
        df,
        x="Rendimiento en Dinero",
        y="Acción",
        points="all",
        color_discrete_sequence=["skyblue"],
        title=f"Boxplot Horizontal de Rendimientos Logarítmicos en Dinero - {accion}"
    )
    fig.update_layout(
        yaxis_title="Acción",
        xaxis_title="Rendimiento Logarítmico en Dinero",
        showlegend=False
    )
    return fig

def graficar_rendimientos_portafolio(precios, pesos_wi_rd, monto):
    """
    Grafica los rendimientos históricos del portafolio en dinero.
    :param precios: DataFrame con precios de cierre
    :param pesos_wi_rd: lista o array de pesos del portafolio (mismo orden que las columnas de precios)
    :return: figura de plotly
    """
    # Calcular rendimientos logarítmicos de cada activo
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    # Rendimiento del portafolio (combinación lineal de rendimientos)
    rend_portafolio = rendimientos.dot(pesos_wi_rd)
    # Evolución del valor del portafolio (asumiendo $1 inicial)
    valor_portafolio = monto * np.exp(rend_portafolio.cumsum())
    # Rendimiento en dinero (cambio diario en el valor del portafolio)
    rend_dinero = valor_portafolio.diff().dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rend_dinero.index,
        y=rend_dinero,
        mode='lines',
        name='Rendimiento diario en dinero',
        line=dict(color='navy')
    ))
    fig.update_layout(
        title="Rendimientos Históricos del Portafolio en Dinero",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Diario en Dinero",
        legend_title="Portafolio"
    )
    return fig

def graficar_rendimiento_acumulado(precios, pesos_wi_rd, monto):
    """
    Grafica la valuación histórica del portafolio asumiendo un monto inicial.
    :param precios: DataFrame con precios de cierre
    :param pesos_wi_rd: lista o array de pesos del portafolio (mismo orden que las columnas de precios)
    :param monto: monto inicial invertido en el portafolio
    :return: figura de plotly
    """
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    rend_portafolio = rendimientos.dot(pesos_wi_rd)
    valor_portafolio = monto * np.exp(rend_portafolio.cumsum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valor_portafolio.index,
        y=valor_portafolio,
        mode='lines',
        name='Valuación del portafolio',
        line=dict(color='green')
    ))
    fig.update_layout(
        title="Valuación Histórica del Portafolio",
        xaxis_title="Fecha",
        yaxis_title="Valuación del Portafolio ($)",
        legend_title="Portafolio"
    )
    return fig

def graficar_histograma_portafolio(precios, pesos_wi_rd, monto, VaR_portafolio):
    """
    Grafica un histograma de los rendimientos logarítmicos en dinero del portafolio,
    usando la posición total (monto invertido), e indica el VaR del portafolio en dinero con una línea vertical.
    Los valores menores a -VaR_portafolio se pintan de rojo.
    :param precios: DataFrame con precios de cierre
    :param pesos_wi_rd: lista o array de pesos del portafolio (mismo orden que las columnas de precios)
    :param monto: monto total invertido en el portafolio
    :param VaR_portafolio: VaR paramétrico del portafolio en dinero (positivo)
    :return: figura de plotly
    """
    # Calcular rendimientos logarítmicos de cada activo
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    # Rendimiento del portafolio (combinación lineal de rendimientos)
    rend_portafolio = rendimientos.dot(pesos_wi_rd)
    # Rendimiento en dinero del portafolio
    rend_dinero_portafolio = rend_portafolio * monto
    var_val = -VaR_portafolio

    # Separar los valores menores y mayores/iguales a var_val
    menores = rend_dinero_portafolio[rend_dinero_portafolio < var_val]
    mayores = rend_dinero_portafolio[rend_dinero_portafolio >= var_val]

    # Crear histograma con dos colores
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mayores,
        nbinsx=60,
        marker_color='blue',
        opacity=0.75,
        name='>= VaR'
    ))
    fig.add_trace(go.Histogram(
        x=menores,
        nbinsx=80,
        marker_color='red',
        opacity=0.75,
        name='< VaR'
    ))

    # Agregar línea vertical para el VaR
    fig.add_shape(
        type="line",
        x0=var_val, x1=var_val,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=var_val,
        y=1,
        yref="paper",
        text="VaR Portafolio",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=0,
        font=dict(color="red")
    )

    fig.update_layout(
        title="Histograma de Rendimientos Logarítmicos en Dinero - Portafolio",
        xaxis_title="Rendimiento Logarítmico en Dinero del Portafolio",
        yaxis_title="Frecuencia",
        barmode='overlay',
        showlegend=True
    )
    return fig

def precios_periodo(nombre_acciones, anios=4):
    """
    Descarga precios de cierre de acciones de Yahoo Finance para un periodo dado en años.
    :param nombre_acciones: lista de tickers
    :param anios: número de años de historial a descargar (int)
    :return: DataFrame con precios de cierre
    """
    recolector = []
    periodo = f"{anios}y"
    for nemo in nombre_acciones:
        ticker = yf.Ticker(nemo)
        px = ticker.history(period=periodo)['Close']
        px.name = nemo
        recolector.append(px)
    precios = pd.concat(recolector, axis=1)
    precios = precios.ffill()
    return precios


def precios_periodo_fechas(nombre_acciones, start, end):
    """
    Descarga precios de cierre de acciones de Yahoo Finance para un periodo dado en años.
    :param nombre_acciones: lista de tickers
    :param anios: número de años de historial a descargar (int)
    :return: DataFrame con precios de cierre
    """
    recolector = []
    for nemo in nombre_acciones:
        ticker = yf.Ticker(nemo)
        px = ticker.history(start=start, end=end)['Close']
        px.name = nemo
        recolector.append(px)
    precios = pd.concat(recolector, axis=1)
    precios = precios.ffill()
    return precios

def graficar_var_rolling_portafolio(precios, pesos_wi_rd, monto, window=504, nivel_confianza=0.95):
    """
    Grafica el VaR rolling paramétrico diario del portafolio con ventana móvil y muestra las violaciones.
    En la leyenda aparece el número y porcentaje de violaciones al VaR.
    :param precios: DataFrame con precios de cierre
    :param pesos_wi_rd: array/lista de pesos del portafolio (mismo orden que las columnas de precios)
    :param monto: monto total invertido en el portafolio
    :param window: tamaño de la ventana rolling (por defecto 504 días ~ 2 años)
    :param nivel_confianza: nivel de confianza para el VaR (por defecto 0.95)
    :return: figura de plotly
    """
    # Calcular rendimientos logarítmicos
    rendimientos = np.log(precios / precios.shift(1)).dropna()
    z_alpha_2 = stats.norm.ppf(1 - (1 - nivel_confianza) / 2)

    var_rolling = []
    for i in range(window, len(rendimientos)):
        datos_ventana = rendimientos.iloc[i - window:i]
        cov_ventana = datos_ventana.cov().values
        sigma_p = np.sqrt(np.dot(pesos_wi_rd, np.dot(cov_ventana, pesos_wi_rd)))
        var_i = -z_alpha_2 * sigma_p * monto
        var_rolling.append(var_i)

    fechas = rendimientos.index[window:]
    var_rolling_series = pd.Series(var_rolling, index=fechas)

    # Rendimientos diarios en dinero del portafolio
    rend_port_diario_dinero = rendimientos.dot(pesos_wi_rd) * monto
    rend_window = rend_port_diario_dinero[window:]

    # Violaciones al VaR
    violaciones_mask = var_rolling_series > rend_window
    violaciones = violaciones_mask.sum()
    violaciones_porcentaje = round((violaciones / len(rend_window)) * 100, 2)
    violaciones_reales = rend_window[violaciones_mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rend_window.index,
        y=rend_window.values,
        mode='lines',
        name='Rendimientos diarios en dinero'
    ))
    fig.add_trace(go.Scatter(
        x=var_rolling_series.index,
        y=var_rolling_series.values,
        mode='lines',
        name=f'VaR Paramétrico ({int(nivel_confianza*100)}%)'
    ))
    fig.add_trace(go.Scatter(
        x=violaciones_reales.index,
        y=violaciones_reales.values,
        mode='markers',
        marker=dict(color='red', size=8),
        name=f'Violaciones al VaR: {violaciones} ({violaciones_porcentaje}%)'
    ))
    fig.update_layout(
        title=f'VaR Paramétrico Rolling del Portafolio (ventana de {window} días)',
        xaxis_title='Fecha',
        yaxis_title='VaR diario [$]',
        template='plotly',
        hovermode='x unified'
    )
    return fig

def graficar_simulacion_portafolio(
    nombre_acciones,
    rendimientos,
    esperanza_activo,
    tasa_cetes,
    var_port_rend_dado,
    esp_port_rend_dado,
    num_simulaciones=1000
):
    """
    Grafica la simulación de portafolios aleatorios en el espacio riesgo-rendimiento.
    Marca el portafolio de rendimiento deseado y el de mínimo riesgo.
    :param nombre_acciones: lista de tickers
    :param rendimientos: DataFrame de rendimientos logarítmicos
    :param esperanza_activo: array/serie de esperanzas de cada activo
    :param tasa_cetes: tasa libre de riesgo anualizada (decimal)
    :param var_port_rend_dado: varianza del portafolio de rendimiento deseado (decimal)
    :param esp_port_rend_dado: esperanza del portafolio de rendimiento deseado (decimal)
    :param var_port_min_riesgo: varianza del portafolio de mínimo riesgo (decimal)
    :param esp_port_min_riesgo: esperanza del portafolio de mínimo riesgo (decimal)
    :param num_simulaciones: número de portafolios aleatorios a simular
    :return: figura de plotly
    """
    m = len(nombre_acciones)
    pesos_ws = pd.DataFrame(columns=nombre_acciones)
    for i in range(num_simulaciones):
        w = np.random.rand(m)
        w = w / np.sum(w)
        pesos_ws.loc[i] = w

    MVCV = rendimientos.cov()
    RI = esperanza_activo

    def varianza_portafolio(W, MVCV):
        return np.dot(W, np.dot(MVCV, W))

    def rendimiento_portafolio(W, RI):
        return np.dot(W, RI)

    VARs = []
    RENs = []

    for i in range(len(pesos_ws)):
        W = pesos_ws.iloc[i].values.astype(float)
        var = np.sqrt(varianza_portafolio(W, MVCV) * 252) * 100
        ren = rendimiento_portafolio(W, RI) * 252
        VARs.append(var)
        RENs.append(ren)

    df_Tasas = pd.DataFrame(list(zip(VARs, RENs)), columns=['% Desviación', '% Rendimiento'])

    fig = px1.scatter(
        df_Tasas,
        x='% Desviación',
        y='% Rendimiento',
        title="Simulación de portafolios",
        opacity=0.5
    )
    # Portafolio de rendimiento deseado (último-1)
    fig.add_trace(go.Scatter(
        x=[var_port_rend_dado * 100 * np.sqrt(252)],
        y=[esp_port_rend_dado * 100 * 252],
        mode='markers',
        marker=dict(color='purple', size=10),
        name="Portafolio de rendimiento deseado"
    ))
    # Tasa CETES (primer punto extra)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[tasa_cetes * 100],
        mode='markers',
        marker=dict(color='green', size=10, symbol='diamond'),
        name="Tasa CETES"
    ))

    fig.update_layout(
        xaxis_title='% Desviación (Riesgo Anualizado)',
        yaxis_title='% Rendimiento Anualizado',
        legend_title="Portafolios"
    )
    return fig