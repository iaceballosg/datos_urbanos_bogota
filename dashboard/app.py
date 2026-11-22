"""
Dashboard interactivo del Proyecto Integrador — Catastro Bogotá.

Aplicación web construida con Streamlit que permite explorar los predios
catastrales de Bogotá: filtros dinámicos, tarjetas KPI ejecutivas y
visualizaciones del análisis.

Ejecutar desde la raíz del proyecto:

    streamlit run dashboard/app.py

Nota: usa la muestra `data/sample/TPREDIO_sample.csv` por defecto (el dataset
completo no se versiona en GitHub por su tamaño). Si dispones del CSV completo
en `data/TPREDIO.csv`, la app lo usará automáticamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Permite importar el paquete `src` desde la raíz del proyecto.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_loader, preprocessing, features  # noqa: E402


# ---------------------------------------------------------------------------
# Carga y preparación (con caché para no recalcular en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos() -> pd.DataFrame:
    """Carga el dataset (completo si existe, si no la muestra) y lo limpia."""
    try:
        df = data_loader.load_data()          # prefiere el CSV completo
    except FileNotFoundError:
        df = data_loader.load_sample()        # fallback a la muestra
    df = preprocessing.clean_data(df)
    df = features.engineer_features(df)
    # Variables auxiliares legibles para el dashboard
    df["ES_RESIDENCIAL"] = (
        df[config.TARGET_CLASIFICACION] == config.CLASE_RESIDENCIAL
    ).astype(int)
    df["DESTINO"] = np.where(df["ES_RESIDENCIAL"] == 1, "Residencial", "No residencial")
    return df


DF = cargar_datos()

st.set_page_config(page_title="Catastro Bogotá — Dashboard", page_icon="🏙️", layout="wide")

st.title("🏙️ Catastro Bogotá — Dashboard Interactivo")
st.caption(
    "Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial · "
    "Universidad de La Salle · Maestría en Inteligencia Artificial"
)

# ---------------------------------------------------------------------------
# Filtros dinámicos (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros dinámicos")

barrios = sorted(DF["PRENBARRIO"].dropna().unique().tolist())
barrio_sel = st.sidebar.multiselect("Barrio", barrios, default=[])

clase_sel = st.sidebar.multiselect(
    "Clase de predio", options=sorted(DF["PRECLASE"].dropna().unique().tolist()), default=[]
)

destino_sel = st.sidebar.multiselect(
    "Destino económico", options=["Residencial", "No residencial"], default=[]
)

area_min, area_max = st.sidebar.slider(
    "Área de terreno (m²)",
    min_value=float(max(0.0, DF["PREATERRE"].min())),
    max_value=float(min(20_000.0, DF["PREATERRE"].quantile(0.99))),
    value=(0.0, float(DF["PREATERRE"].quantile(0.95))),
)

# Aplicación de filtros
filtrado = DF.copy()
if barrio_sel:
    filtrado = filtrado[filtrado["PRENBARRIO"].isin(barrio_sel)]
if clase_sel:
    filtrado = filtrado[filtrado["PRECLASE"].isin(clase_sel)]
if destino_sel:
    filtrado = filtrado[filtrado["DESTINO"].isin(destino_sel)]
filtrado = filtrado[(filtrado["PREATERRE"] >= area_min) & (filtrado["PREATERRE"] <= area_max)]

# ---------------------------------------------------------------------------
# Tarjetas KPI ejecutivas
# ---------------------------------------------------------------------------
st.subheader("Indicadores ejecutivos")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predios", f"{len(filtrado):,}")
k2.metric("Área terreno promedio (m²)", f"{filtrado['PREATERRE'].mean():,.0f}")
k3.metric("% Residencial", f"{filtrado['ES_RESIDENCIAL'].mean() * 100:.1f} %")
k4.metric("Área construida total (m²)", f"{filtrado['PREACONST'].sum():,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución del área de terreno")
    conteo_area = (
        pd.cut(filtrado["PREATERRE"], bins=30)
        .value_counts()
        .sort_index()
        .astype(int)
    )
    # Los intervalos de pd.cut no son serializables por Altair: se convierten a texto
    conteo_area.index = conteo_area.index.astype(str)
    st.bar_chart(conteo_area)

with col2:
    st.subheader("Predios por destino económico")
    st.bar_chart(filtrado["DESTINO"].value_counts())

st.subheader("Área construida vs área de terreno")
muestra = filtrado.sample(min(3000, len(filtrado)), random_state=config.RANDOM_STATE)
st.scatter_chart(muestra, x="PREATERRE", y="PREACONST")

st.subheader("Calidad estructural consolidada por clase de predio")
st.bar_chart(filtrado.groupby("PRECLASE")["CALIDAD_EST"].mean().round(2))

with st.expander("Ver datos filtrados"):
    st.dataframe(filtrado.head(200))
    st.download_button(
        "Descargar CSV filtrado",
        filtrado.to_csv(index=False).encode("utf-8"),
        file_name="predios_filtrados.csv",
        mime="text/csv",
    )
