"""
Ingeniería de variables (features) y preparación de los conjuntos de datos.

Variables derivadas (documentadas en el informe técnico):

  - IND_CONST    = PREACONST / PREATERRE
                   Índice de construcción: intensidad de ocupación / altura.
  - CALIDAD_EST  = media de PREEARMAZ, PREEMUROS y PREECUBIER
                   Calidad estructural consolidada (reduce multicolinealidad).

Codificación y escalamiento (sin fuga de datos):
  - One-Hot Encoding para categóricas de baja cardinalidad.
  - Target Encoding para PRECBARRIO (alta cardinalidad) SOLO en clasificación
    y ajustado exclusivamente sobre el conjunto de entrenamiento.
  - RobustScaler para variables numéricas (resiliente a valores atípicos);
    se ajusta solo con entrenamiento y se transforma el resto.

Toda transformación que "aprende" de los datos (fit) se hace únicamente sobre
el conjunto de entrenamiento para prevenir data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from . import config


# ---------------------------------------------------------------------------
# Variables derivadas
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade las variables derivadas IND_CONST y CALIDAD_EST.

    Parámetros
    ----------
    df : dataframe limpio (salida de preprocessing.clean_data).

    Retorna
    -------
    pd.DataFrame con las nuevas columnas.
    """
    out = df.copy()

    # Índice de construcción (evita división por cero).
    out["IND_CONST"] = out["PREACONST"] / out["PREATERRE"].replace(0, np.nan)

    # Calidad estructural consolidada (media de los tres puntajes estructurales).
    estructurales = [c for c in config.COLS_ESTRUCTURA if c in out.columns]
    if estructurales:
        out["CALIDAD_EST"] = out[estructurales].mean(axis=1)

    return out


# ---------------------------------------------------------------------------
# Construcción de objetivos
# ---------------------------------------------------------------------------
def build_regression_target(df: pd.DataFrame) -> pd.Series:
    """Variable objetivo de regresión: área construida (PREACONST)."""
    return df[config.TARGET_REGRESION].astype(float)


def build_classification_target(df: pd.DataFrame) -> pd.Series:
    """
    Variable objetivo de clasificación binaria.

    1 = Residencial (PRECDESTIN == 1), 0 = No residencial.
    """
    return (df[config.TARGET_CLASIFICACION] == config.CLASE_RESIDENCIAL).astype(int)


# ---------------------------------------------------------------------------
# Codificación y escalamiento
# ---------------------------------------------------------------------------
_CATEGORICAS_BAJA_CARD = ["PRECLASE", "PREUSOPH", "PREUCALIF", "PRETPROP"]


def _categorical_columns_available(df: pd.DataFrame) -> list[str]:
    """Categóricas de baja cardinalidad presentes en el dataframe."""
    return [c for c in _CATEGORICAS_BAJA_CARD if c in df.columns]


def build_numeric_pipeline() -> Pipeline:
    """Pipeline de escalado robusto para variables numéricas."""
    return Pipeline([("scaler", RobustScaler())])


def build_preprocessor(df_train: pd.DataFrame, target_encode: bool = False) -> ColumnTransformer:
    """
    Construye el preprocesador (ColumnTransformer) de columnas numéricas y
    categóricas de baja cardinalidad.

    Parámetros
    ----------
    df_train : subconjunto de entrenamiento (solo para detectar columnas).
    target_encode : si es True, incluye Target Encoding de PRECBARRIO.

    Retorna
    -------
    sklearn ColumnTransformer (sin ajustar).
    """
    num_cols = [c for c in config.FEATURES_REGRESION if c in df_train.columns]
    cat_cols = _categorical_columns_available(df_train)

    transformers = [
        ("num", RobustScaler(), num_cols),
    ]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    return ColumnTransformer(transformers, remainder="drop")


def split_regression(df: pd.DataFrame, test_size: float | None = None) -> tuple:
    """
    Partición para regresión (aleatoria, 80/20).

    Retorna X_train, X_test, y_train, y_test (dataframes/Series sin codificar).
    """
    t = test_size if test_size is not None else config.TEST_SIZE
    X = df.drop(columns=[config.TARGET_REGRESION])
    y = build_regression_target(df)
    return train_test_split(X, y, test_size=t, random_state=config.RANDOM_STATE)


def split_classification(df: pd.DataFrame, test_size: float | None = None) -> tuple:
    """
    Partición estratificada para clasificación.

    Usa StratifiedShuffleSplit para conservar las proporciones de clase,
    contrarrestando el desbalance severo de la variable objetivo.
    """
    t = test_size if test_size is not None else config.TEST_SIZE
    X = df.drop(columns=[config.TARGET_CLASIFICACION])
    y = build_classification_target(df)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=t, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def add_target_encoding(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, col: str = "PRECBARRIO") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target Encoding (media suavizada) de una columna de alta cardinalidad.

    Se ajusta SOLO con el entrenamiento para evitar fuga de datos.
    """
    if col not in X_train.columns:
        return X_train, X_test

    prior = y_train.mean()
    n = len(X_train)
    smoothing = 20  # regularización para categorías poco frecuentes

    stats = y_train.groupby(X_train[col].astype(str)).agg(["mean", "count"])
    stats["encoded"] = (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)
    enc_map = stats["encoded"].to_dict()

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        f[f"{col}_ENC"] = f[col].astype(str).map(enc_map).fillna(prior)
        return f.drop(columns=[col])

    return _apply(X_train), _apply(X_test)


def prepare_numeric_features(df: pd.DataFrame, feature_list: list[str] | None = None) -> pd.DataFrame:
    """
    Devuelve solo las columnas numéricas seleccionadas (sin nulos).
    Útil para clustering.
    """
    cols = feature_list or config.FEATURES_CLUSTERING
    cols = [c for c in cols if c in df.columns]
    return df[cols].dropna().copy()


def scale_features(df: pd.DataFrame, scaler=None):
    """Ajusta (o transforma) un escalador sobre las columnas del dataframe."""
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    return scaled, scaler
