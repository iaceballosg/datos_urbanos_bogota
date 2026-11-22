"""
Configuración central del proyecto.

Contiene:
  - Rutas y nombres de archivo.
  - Definición de columnas (identificadores, numéricas, categóricas).
  - Variables objetivo (regresión y clasificación).
  - Hiperparámetros generales (semilla, partición, umbrales de filtrado).
  - Límite de filas para modelado (para ejecución ágil en Google Colab).

Toda la configuración relevante se concentra aquí para que los notebooks,
los scripts y el dashboard compartan exactamente los mismos valores.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
# Raíz del proyecto (dos niveles por encima de este archivo).
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "TPREDIO.csv"                 # dataset completo (1.4 GB, ~3.8 M filas)
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "TPREDIO_sample.csv"  # muestra para pruebas (~6.000 filas)

RESULTS_DIR = ROOT_DIR / "resultados"
MODELS_DIR = ROOT_DIR / "modelos"

# ---------------------------------------------------------------------------
# Semilla y partición
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 % entrenamiento / 20 % prueba (según informe)
VAL_SIZE = 0.15           # opcional: fracción de validación dentro del entrenamiento

# ---------------------------------------------------------------------------
# Límite de filas para modelado
# ---------------------------------------------------------------------------
# El dataset completo tiene ~3.8 millones de filas. Para que los modelos
# entrenen en un tiempo razonable en Google Colab (CPU gratuita), se trabaja
# por defecto con una muestra estratificada. Ajusta este valor según tu
# entorno (None = usar todas las filas filtradas).
MAX_ROWS_FOR_MODELING = 200_000

# ---------------------------------------------------------------------------
# Columnas por tipo
# ---------------------------------------------------------------------------
# Códigos catastrales/geográficos: se leen como número pero deben tratarse
# como texto (identificadores) para no perder ceros a la izquierda.
COLS_CODIGO = ["PRECBARRIO", "PRECMANZ", "PRECPREDIO", "PRECCONS", "PRECRESTO"]

# Códigos numéricos de gran longitud que deben forzarse a texto al leer el CSV.
# (PRENUPRE tiene 30 dígitos y pandas lo lee como entero gigante, lo que rompe
#  la conversión a Arrow de Streamlit/pyarrow; PRECZHF es un código de 13 dígitos.)
COLS_CODIGO_LARGO = ["PRENUPRE", "PRECZHF"]

# Columnas que SIEMPRE deben leerse como texto (identificadores, no cantidades).
COLS_A_TEXTO = COLS_CODIGO + COLS_CODIGO_LARGO

# Identificadores únicos y referencias espaciales.
COLS_IDENTIFICADOR = ["PRECHIP", "PRECEDCATA", "PRENUPRE", "PREDIRECC", "PRENBARRIO"]

# Variables físicas (áreas en m²).
COLS_AREA = ["PREATERRE", "PREACONST"]

# Puntajes de calificación constructiva (estructura).
COLS_ESTRUCTURA = ["PREEARMAZ", "PREEMUROS", "PREECUBIER"]

# Puntajes de acabados y demás (mayor proporción de nulos: se imputan con 0).
COLS_ACABADOS = [
    "PREECONS", "PREAFACHAD", "PREACUBIER", "PREAPISOS", "PREACONS",
    "PREBTAMANO", "PREBENCHAP", "PREBMOBILI", "PREBCONS",
    "PRECTAMANO", "PRECENCHAP", "PRECMOBILI", "PRECCONSE",
]

# Variables industriales (nulos masivos ~74 %: imputar con 0 = "no aplica").
COLS_INDUSTRIALES = ["PRECINDUS", "PREACERCHA"]

# Variables categóricas de uso y calificación.
COLS_CATEGORICAS = [
    "PRETPROP", "PRECLASE", "PRECUSO", "PREUSOPH", "PREUSONPH",
    "PREUVIVIEN", "PREUCALIF", "PRECLCONS", "PRETDIRECC",
]

# Fechas (se conservan como texto; pueden convertirse a datetime si se requiere).
COLS_FECHA = ["PREFINCORP", "PREFCALIF"]

# Variable de avalúo (puede usarse como predictor adicional).
COLS_AVALUO = ["PREVACTUAL", "PREVETUSTZ", "PREVFORMA"]

# ---------------------------------------------------------------------------
# Variables objetivo
# ---------------------------------------------------------------------------
# Regresión  -> área construida (m²).
TARGET_REGRESION = "PREACONST"

# Clasificación -> destino económico del suelo.
# Se binariza: 1 = Residencial / Habitacional, resto = No residencial.
# (En el catastro de Bogotá el código de destino 1 corresponde a uso habitacional).
TARGET_CLASIFICACION = "PRECDESTIN"
CLASE_RESIDENCIAL = 1

# Variables base para el modelo de regresión (predictores).
FEATURES_REGRESION = [
    "PREATERRE",
    "PREPUNTAJE",
    "PREEARMAZ",
    "PREEMUROS",
    "PREECUBIER",
    "IND_CONST",
    "CALIDAD_EST",
]

# Variables base para el modelo de clasificación.
FEATURES_CLASIFICACION = FEATURES_REGRESION  # mismas predictoras + dummies categóricas

# Variables para clustering (segmentación no supervisada).
FEATURES_CLUSTERING = [
    "PREATERRE",
    "PREACONST",
    "PREPUNTAJE",
    "PREEARMAZ",
    "PREEMUROS",
    "PREECUBIER",
]

# ---------------------------------------------------------------------------
# Filtrado de valores atípicos para modelado
# ---------------------------------------------------------------------------
# Se excluyen predios con área 0 (propiedad horizontal / baldíos) y se acota
# a rangos razonables para que los valores extremos no dominen los modelos.
AREA_MIN = 0.0
AREA_MAX = 10_000.0
PUNTAJE_MIN = 0.0   # excluye PREPUNTAJE == -1 (predios exentos/sin calificar)

# Columnas que deben quedar libres de nulos en la muestra de modelado.
COLS_SIN_NULOS_MODELADO = ["PREATERRE", "PREACONST", "PREPUNTAJE",
                           "PREEARMAZ", "PREEMUROS", "PREECUBIER"]
