"""
Limpieza y preprocesamiento del dataset catastral.

Decisiones documentadas (basadas en el perfilado de calidad del proyecto):

  1. Fila anómala: se elimina la única fila con CHIP nulo (todo predio debe
     tener CHIP, por lo que ese registro está corrupto).
  2. Códigos catastrales: se convierten a texto (identificadores, no números).
  3. Nulos "condicionales" masivos: en catastro, un nulo en variables
     industriales o de acabados significa "no aplica" (el predio no tiene ese
     uso/calificación). Se imputan con 0, no se eliminan.
  4. PREPUNTAJE == -1: comodín para predios exentos/sin calificar; se
     mantiene pero se excluye del modelado supervisado.
"""

from __future__ import annotations

import pandas as pd

from . import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza completa y devuelve una copia del dataframe.

    Parámetros
    ----------
    df : dataframe crudo (salida de data_loader.load_data).

    Retorna
    -------
    pd.DataFrame limpio.
    """
    out = df.copy()

    # 1) Eliminar la fila anómala (PRECHIP nulo). Se usa CHIP como ancla
    #    porque es la llave primaria del predio.
    n_before = len(out)
    out = out.dropna(subset=["PRECHIP"]).copy()

    # 2) Convertir códigos geográficos/catastrales a texto.
    for col in config.COLS_CODIGO:
        if col in out.columns:
            # astype(str) evita que pandas los reintérprete como número.
            out[col] = out[col].astype(str)

    # 3) Imputación lógica de nulos masivos.
    # Variables industriales -> 0 (el lote no tiene uso industrial).
    for col in config.COLS_INDUSTRIALES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    # Acabados y calificación constructiva -> 0 (no aplica / sin acabado).
    for col in config.COLS_ACABADOS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    # 4) Convertir variables numéricas que pandas leyó como texto a float.
    for col in config.COLS_AREA + config.COLS_ESTRUCTURA + config.COLS_AVALUO + ["PREPUNTAJE"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte de calidad por columna:
    tipo, % nulos, % valores únicos, mínimo y máximo (para numéricas).

    Retorna
    -------
    pd.DataFrame con el perfil de calidad.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nulos = s.isnull().sum()
        unicos = s.nunique(dropna=True)
        info = {
            "columna": col,
            "dtype": str(s.dtype),
            "nulos": nulos,
            "pct_nulos": round(nulos / n * 100, 4),
            "pct_unicos": round(unicos / n * 100, 2),
        }
        if pd.api.types.is_numeric_dtype(s):
            info["min"] = s.min()
            info["max"] = s.max()
        rows.append(info)

    report = pd.DataFrame(rows).sort_values("pct_nulos", ascending=False)
    return report.reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Resumen de duplicados exactos y por llave primaria (CHIP)."""
    return {
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_chip": int(df.duplicated(subset=["PRECHIP"]).sum()),
        "filas_totales": int(len(df)),
    }


def filter_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el dataframe para modelado supervisado:

      - Áreas de terreno y construida en (AREA_MIN, AREA_MAX).
      - PREPUNTAJE > PUNTAJE_MIN (excluye -1).
      - Sin nulos en las columnas requeridas.

    Retorna
    -------
    pd.DataFrame filtrado (copia).
    """
    out = df[
        (df["PREATERRE"] > config.AREA_MIN)
        & (df["PREATERRE"] < config.AREA_MAX)
        & (df["PREACONST"] > config.AREA_MIN)
        & (df["PREACONST"] < config.AREA_MAX)
        & (df["PREPUNTAJE"] > config.PUNTAJE_MIN)
    ].copy()

    out = out.dropna(subset=config.COLS_SIN_NULOS_MODELADO)
    return out
