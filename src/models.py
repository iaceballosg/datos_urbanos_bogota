"""
Entrenamiento de modelos de Machine Learning.

Incluye los tres bloques de algoritmos exigidos por la guía del proyecto:

  1. Regresión (predicción continua) -> PREACONST:
     - Baseline: Regresión Lineal Simple.
     - Regularizados: Ridge, Lasso, ElasticNet.
     - Árboles/ensambles: RandomForest, GradientBoosting.

  2. Clasificación (predicción binaria) -> Residencial vs No residencial:
     - Regresión Logística (baseline), KNN, Árbol de Decisión,
       RandomForest, GradientBoosting y (opcional) SVM.

  3. Clustering (no supervisado):
     - K-Means (con selección de K vía codo + silueta).
     - DBSCAN y Clustering Jerárquico (Agglomerative).

Todos los modelos de scikit-learn se entrenan con una semilla fija para
garantizar reproducibilidad.
"""

from __future__ import annotations

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
)
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config

RS = config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------
def get_regression_models() -> dict:
    """Devuelve el diccionario de modelos de regresión (sin ajustar)."""
    return {
        "Regresión Lineal (baseline)": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0, random_state=RS),
        "Lasso (L1)": Lasso(alpha=0.1, random_state=RS, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RS, max_iter=5000),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=None, n_jobs=-1, random_state=RS,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RS,
        ),
    }


def train_regression(X_train, y_train) -> dict:
    """Ajusta todos los modelos de regresión y devuelve {nombre: modelo_ajustado}."""
    fitted = {}
    for name, model in get_regression_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------
def get_classification_models(include_svm: bool = False) -> dict:
    """Devuelve el diccionario de modelos de clasificación (sin ajustar)."""
    models = {
        "Regresión Logística (baseline)": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RS,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=RS,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=RS,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RS,
        ),
    }
    if include_svm:
        from sklearn.svm import SVC
        models["SVM (RBF)"] = SVC(probability=True, class_weight="balanced", random_state=RS)
    return models


def train_classification(X_train, y_train, include_svm: bool = False) -> dict:
    """Ajusta todos los modelos de clasificación y devuelve {nombre: modelo_ajustado}."""
    fitted = {}
    for name, model in get_classification_models(include_svm).items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def elbow_scores(X_scaled, k_range: range) -> list[float]:
    """Inercia (WCSS) de K-Means para cada k -> método del codo."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RS, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return inertias


def silhouette_scores(X_scaled, k_range: range) -> list[float]:
    """Coeficiente de silueta de K-Means para cada k (k >= 2)."""
    scores = []
    for k in k_range:
        if k < 2:
            scores.append(float("nan"))
            continue
        km = KMeans(n_clusters=k, random_state=RS, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels, sample_size=min(50_000, len(X_scaled))))
    return scores


def fit_kmeans(X_scaled, k: int) -> KMeans:
    """Ajusta K-Means con el número óptimo de clústeres k."""
    return KMeans(n_clusters=k, random_state=RS, n_init=10).fit(X_scaled)


def fit_dbscan(X_scaled, eps: float = 0.5, min_samples: int = 5) -> DBSCAN:
    """Ajusta DBSCAN (segundo algoritmo no supervisado)."""
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X_scaled)


def fit_agglomerative(X_scaled, k: int) -> AgglomerativeClustering:
    """Ajusta Clustering Jerárquico (Agglomerative)."""
    return AgglomerativeClustering(n_clusters=k).fit(X_scaled)


def profile_clusters(df: "pd.DataFrame", labels: "pd.Series", feature_cols: list[str]) -> "pd.DataFrame":
    """Perfil de cada clúster: media de las variables y cantidad de predios."""
    import pandas as pd
    tmp = df.copy()
    tmp["Cluster"] = labels
    profile = tmp.groupby("Cluster")[feature_cols].mean().round(2)
    profile["Cantidad_Predios"] = tmp.groupby("Cluster").size()
    return profile
