"""
Evaluación de modelos y visualización de resultados.

Métricas cubiertas (según la guía del proyecto):

  - Regresión     : R², RMSE, MAE.
  - Clasificación : Accuracy, Precision, Recall, F1-Score, AUC-ROC.
  - Clustering    : Silhouette Score (+ perfil de clústeres).

Las funciones de graficado usan matplotlib/seaborn (disponibles en Colab).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, silhouette_score,
)


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------
def evaluate_regression(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcula R², RMSE y MAE para cada modelo de regresión ajustado.

    Retorna un DataFrame ordenado por R² descendente.
    """
    rows = []
    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        rows.append({
            "modelo": name,
            "R2": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds),
        })
    res = pd.DataFrame(rows)
    return res.sort_values("R2", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------
def evaluate_classification(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Calcula las métricas de clasificación (Accuracy, Precision, Recall, F1,
    AUC-ROC) para cada modelo.

    AUC-ROC usa la probabilidad de la clase positiva cuando está disponible.
    """
    rows = []
    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        row = {
            "modelo": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
        }
        try:
            probs = model.predict_proba(X_test)[:, 1]
            row["AUC-ROC"] = roc_auc_score(y_test, probs)
        except Exception:
            row["AUC-ROC"] = np.nan
        rows.append(row)

    res = pd.DataFrame(rows)
    return res.sort_values("F1-Score", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(fitted_models: dict, X_test, y_test, top_n: int = 4) -> None:
    """Dibuja las matrices de confusión de los mejores modelos."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    n = min(top_n, len(fitted_models))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, list(fitted_models.items())[:n]):
        preds = model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds, ax=ax, colorbar=False, cmap="Blues",
        )
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def plot_elbow(inertias: list[float], k_range: range) -> None:
    """Gráfica del método del codo (inercia vs número de clústeres)."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker="o", linestyle="--", color="b")
    plt.title("Método del Codo para determinar K óptimo")
    plt.xlabel("Número de clústeres (K)")
    plt.ylabel("Inercia (WCSS)")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.5)
    plt.show()


def plot_silhouette(scores: list[float], k_range: range) -> None:
    """Gráfica del coeficiente de silueta vs número de clústeres."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, marker="s", linestyle="-", color="g")
    plt.title("Coeficiente de Silueta vs Número de Clústeres")
    plt.xlabel("Número de clústeres (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.5)
    plt.show()


def silhouette_of_labels(X_scaled, labels) -> float:
    """Silueta para etiquetas ya calculadas (DBSCAN, jerárquico, etc.)."""
    if len(set(labels)) < 2:
        return float("nan")
    return silhouette_score(X_scaled, labels, sample_size=min(50_000, len(X_scaled)))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def regression_best(eval_df: pd.DataFrame) -> dict:
    """Devuelve la fila del mejor modelo de regresión."""
    return eval_df.iloc[0].to_dict()


def classification_best(eval_df: pd.DataFrame) -> dict:
    """Devuelve la fila del mejor modelo de clasificación (por F1)."""
    return eval_df.iloc[0].to_dict()
