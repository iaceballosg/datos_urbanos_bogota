"""
Carga del dataset catastral.

El archivo completo (TPREDIO.csv, ~1.4 GB, ~3.8 millones de filas) no se
versiona en GitHub por su tamaño. Este módulo centraliza la lógica de carga
para que funcione en tres escenarios:

  1. Local / repositorio clonado: data/TPREDIO.csv o data/sample/TPREDIO_sample.csv.
  2. Google Colab con Google Drive montado.
  3. Google Colab tras clonar el repositorio desde GitHub.

Se especifican los dtypes adecuados para optimizar memoria y evitar la pérdida
de ceros a la izquierda en los códigos catastrales.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def _resolve_path(csv_path: str | Path | None = None, use_sample: bool = False) -> Path:
    """Determina la ruta del CSV a cargar según el entorno de ejecución."""
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"No se encontró el archivo: {p}")

    # 1) Ruta local del repositorio.
    candidates: list[Path] = []
    if use_sample:
        candidates.append(config.SAMPLE_DATA_PATH)
    else:
        candidates.append(config.RAW_DATA_PATH)
        candidates.append(config.SAMPLE_DATA_PATH)  # fallback a la muestra

    # 2) Google Colab: dataset completo montado en Drive.
    candidates.append(Path("/content/drive/MyDrive/proyecto_datos_urbanos/data/TPREDIO.csv"))
    candidates.append(Path("/content/drive/MyDrive/TPREDIO.csv"))

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No se encontró el dataset. Opciones:\n"
        "  - Coloca TPREDIO.csv en data/ (o usa data/sample/TPREDIO_sample.csv).\n"
        "  - En Colab, monta Drive y apunta la ruta en el parámetro csv_path.\n"
        "  - Descarga el dataset desde el portal de Datos Abiertos (ver data/README.md)."
    )


def load_data(
    csv_path: str | Path | None = None,
    nrows: int | None = None,
    use_sample: bool = False,
) -> pd.DataFrame:
    """
    Carga el dataset catastral con tipos de datos optimizados.

    Parámetros
    ----------
    csv_path : ruta explícita al CSV (opcional).
    nrows : número máximo de filas a leer (None = todas).
    use_sample : si es True, prefiere la muestra pequeña para pruebas.

    Retorna
    -------
    pd.DataFrame con las columnas originales.
    """
    path = _resolve_path(csv_path, use_sample)

    # Códigos catastrales se leen como texto para preservar ceros a la izquierda
    # y evitar que códigos largos (PRENUPRE, PRECZHF) se lean como enteros gigantes.
    dtype = {c: str for c in config.COLS_A_TEXTO}

    df = pd.read_csv(
        path,
        sep=",",
        dtype=dtype,
        nrows=nrows,
        low_memory=False,
        keep_default_na=True,
    )

    # Normaliza espacios en nombres de columna (por si hubiera variaciones).
    df.columns = [c.strip() for c in df.columns]

    return df


def load_sample(nrows: int | None = None) -> pd.DataFrame:
    """Atajo para cargar la muestra pequeña (para pruebas rápidas)."""
    return load_data(use_sample=True, nrows=nrows)


def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen rápido de estructura y completitud del dataset."""
    print(f"Filas: {len(df):,} | Columnas: {df.shape[1]}")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos):
        print("Columnas con valores nulos:")
        for col, n in nulos.items():
            print(f"  {col:15s} {n:>10,}  ({n / len(df) * 100:6.2f} %)")
    else:
        print("Sin valores nulos.")
