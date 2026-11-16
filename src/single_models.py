# -*- coding: utf-8 -*-
"""П. 2.1–2.3: Одиночные методы машинного обучения (KNN, SVR, Decision Tree)."""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from src.config import SEED
from src.data_loader import PreparedData


def run_single_ml_models(data: PreparedData):
    """Обучает KNN, SVR и дерево принятия решений (с масштабированием и без)."""
    models_out = {}

    # 2.1 Метод K ближайших соседей (KNN)
    knn_un = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(data.X_train, data.y_train)
    knn_sc = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(data.X_train_s, data.y_train_s)
    models_out["knn"] = {
        "title": "2.1. Метод K ближайших соседей (KNN)",
        "hyperparams": "n_neighbors=5, weights='distance'",
        "unscaled_test": knn_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(knn_sc.predict(data.X_train_s)),
            data.unscale_y(knn_sc.predict(data.X_val_s)),
            data.unscale_y(knn_sc.predict(data.X_test_s)),
        ),
    }

    # 2.2 Машина опорных векторов (SVR RBF)
    svr_un = SVR(kernel="rbf", C=500.0, gamma=0.01, epsilon=0.1).fit(data.X_train, data.y_train)
    svr_sc = SVR(kernel="rbf", C=100.0, gamma=0.5, epsilon=0.01).fit(data.X_train_s, data.y_train_s)
    models_out["svr"] = {
        "title": "2.2. Машина опорных векторов (SVR RBF)",
        "hyperparams": "kernel='rbf', C=100, gamma=0.5",
        "unscaled_test": svr_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(svr_sc.predict(data.X_train_s)),
            data.unscale_y(svr_sc.predict(data.X_val_s)),
            data.unscale_y(svr_sc.predict(data.X_test_s)),
        ),
    }

    # 2.3 Одиночное дерево принятия решений (Decision Tree)
    dt_un = DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, random_state=SEED).fit(data.X_train, data.y_train)
    dt_sc = DecisionTreeRegressor(max_depth=10, min_samples_leaf=2, random_state=SEED).fit(data.X_train_s, data.y_train_s)
    dt_vis = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=SEED).fit(data.X_train, data.y_train)
    models_out["dt"] = {
        "title": "2.3. Дерево принятия решений (Decision Tree)",
        "hyperparams": "max_depth=10, min_samples_leaf=2",
        "unscaled_test": dt_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(dt_sc.predict(data.X_train_s)),
            data.unscale_y(dt_sc.predict(data.X_val_s)),
            data.unscale_y(dt_sc.predict(data.X_test_s)),
        ),
        "vis_tree": dt_vis,
    }

    return models_out
