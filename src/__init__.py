"""
Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial.

Catastro Bogotá — descubrimiento de tipologías y predicción de variables
catastrales mediante Machine Learning.

Módulos:
  - config         : rutas, columnas, objetivos e hiperparámetros.
  - data_loader    : carga del dataset (local, Colab/Drive, muestra).
  - preprocessing  : limpieza e imputación documentadas.
  - features       : ingeniería de variables y partición sin fuga de datos.
  - models         : regresión, clasificación y clustering.
  - evaluation     : métricas y gráficas de evaluación.

Uso típico (notebook o script):

    import sys
    sys.path.append("..")   # para que Python encuentre el paquete `src`
    from src import config, data_loader, preprocessing, features, models, evaluation
"""

__all__ = ["config", "data_loader", "preprocessing", "features", "models", "evaluation"]
__version__ = "1.0.0"
