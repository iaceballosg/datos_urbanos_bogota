# Catastro Bogotá — Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial

Descubrimiento de tipologías prediales ocultas y predicción de variables
catastrales mediante **Machine Learning**, aplicado al dataset abierto de
**predios de Bogotá D.C.** (~3.8 millones de registros).

Proyecto desarrollado para la asignatura **Ciencia de Datos** de la
**Maestría en Inteligencia Artificial — Universidad de La Salle**, bajo la
metodología **CRISP-DM**.

| | |
|---|---|
| **Autor** | Ivan Alejandro Ceballos Galvis |
| **Asignatura** | Ciencia de Datos e Inteligencia Artificial |

---

## 🎯 Objetivo

Desarrollar y evaluar un **pipeline de Machine Learning** (supervisado y no
supervisado) sobre el dataset de Catastro Bogotá para:

1. **Segmentar** el inventario predial en tipologías urbanas ocultas (clustering).
2. **Predecir** el área construida (regresión).
3. **Clasificar** el destino económico del suelo (clasificación).

## 🧠 Enfoque analítico (tres bloques de algoritmos)

| Bloque | Técnica | Variable objetivo |
|--------|---------|-------------------|
| **Regresión** | Lineal, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting | `PREACONST` (área construida) |
| **Clasificación** | Regresión Logística, KNN, Árbol de Decisión, Random Forest, Gradient Boosting | `PRECDESTIN` → binario (residencial / no residencial) |
| **Clustering** | K-Means (codo + silueta), DBSCAN, Agglomerative | tipologías prediales |

## 🗂️ Estructura del repositorio

```
proyecto_datos_urbanos/
├── data/                    # dataset (completo NO versionado) + muestra
│   ├── README.md            # origen, licencia y diccionario de datos
│   └── sample/              # muestra de 6.000 filas para pruebas
├── notebooks/
│   └── proyecto_catastro_bogota.ipynb   # pipeline completo ejecutable en Colab
├── src/                     # código modular reutilizable
│   ├── config.py            # rutas, columnas, objetivos, hiperparámetros
│   ├── data_loader.py       # carga (local / Colab / Drive / muestra)
│   ├── preprocessing.py     # limpieza e imputación documentadas
│   ├── features.py          # ingeniería de variables y partición sin leakage
│   ├── models.py            # regresión, clasificación y clustering
│   └── evaluation.py        # métricas y visualización
├── dashboard/
│   └── app.py               # dashboard interactivo (Streamlit)
├── run_dashboard.bat        # lanzador del dashboard (Windows, doble clic)
├── run_dashboard.sh         # lanzador del dashboard (macOS / Linux)
├── requirements.txt         # dependencias (versiones exactas)
├── .gitignore
└── README.md
```

## 🚀 Ejecución en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iaceballosg/proyecto_datos_urbanos/blob/main/notebooks/proyecto_catastro_bogota.ipynb)

> abra el notebook directamente en Colab.

### Opción A — Abrir el notebook directamente

1. Sube este repositorio a GitHub.
2. Haz clic en el botón "Open in Colab" o importa
   `notebooks/proyecto_catastro_bogota.ipynb`.
3. El notebook monta Google Drive, localiza el dataset y ejecuta todo el
   pipeline de principio a fin.

### Opción B — Clonar el repositorio en Colab

```python
!git clone https://github.com/TU_USUARIO/proyecto_datos_urbanos.git
%cd proyecto_datos_urbanos
```

Luego abre `notebooks/proyecto_catastro_bogota.ipynb`.

### Sobre el dataset

El archivo completo `TPREDIO.csv` (~1.4 GB) **no** se incluye en GitHub por su
tamaño. Opciones:

- Monta Google Drive con el CSV completo.
- Usa la muestra `data/sample/TPREDIO_sample.csv` (incluida) para pruebas.

Consulta [`data/README.md`](data/README.md) para obtener el dataset completo
desde el portal de Datos Abiertos.

## 🖥️ Dashboard interactivo (Fase 4)

**Forma rápida (doble clic):**

- Windows: ejecuta `run_dashboard.bat`
- macOS / Linux: ejecuta `./run_dashboard.sh`

**Forma manual:**

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Los lanzadores instalan automáticamente las dependencias la primera vez y
abren el navegador en `http://localhost:8501`.

El dashboard incluye filtros dinámicos (barrio, clase, destino, rango de
área), tarjetas KPI ejecutivas y visualizaciones del análisis.

## 📦 Dependencias

Instalar con:

```bash
pip install -r requirements.txt
```

En Google Colab la mayoría ya está preinstalada; solo actualiza scikit-learn
si es necesario:

```python
!pip install -q -U scikit-learn
```

## 📝 Licencia

Los datos provienen de fuentes públicas de Datos Abiertos (IDECA / Catastro
Bogotá). El código de este repositorio se distribuye bajo licencia
[MIT](https://opensource.org/licenses/MIT).

## Reconocimientos

Este proyecto fue desarrollado con el apoyo de un asistente de IA generativa
(DeepSeek / DeepSeek Harness) para la generación y depuración de código y de
documentación. El diseño, la verificación de resultados y las decisiones
finales corresponden al autor.
