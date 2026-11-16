# -*- coding: utf-8 -*-
"""П. 3.1–3.3: Ансамблевые методы машинного обучения (Random Forest, Bagging, Gradient Boosting)."""

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from src.config import SEED
from src.data_loader import PreparedData


def run_ensemble_ml_models(data: PreparedData):
    """Обучает случайный лес, бэггинг и градиентный бустинг (с масштабированием и без)."""
    models_out = {}

    # 3.1 Случайный лес (Random Forest)
    rf_un = RandomForestRegressor(
        n_estimators=60, max_depth=9, max_features=4, min_samples_leaf=2, random_state=SEED
    ).fit(data.X_train, data.y_train)
    rf_sc = RandomForestRegressor(
        n_estimators=60, max_depth=9, max_features=4, min_samples_leaf=2, random_state=SEED
    ).fit(data.X_train_s, data.y_train_s)
    models_out["rf"] = {
        "title": "3.1. Случайный лес (Random Forest)",
        "hyperparams": "n_estimators=60, max_depth=9, max_features=4",
        "unscaled_test": rf_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(rf_sc.predict(data.X_train_s)),
            data.unscale_y(rf_sc.predict(data.X_val_s)),
            data.unscale_y(rf_sc.predict(data.X_test_s)),
        ),
    }

    # 3.2 Бэггинг (Bagging)
    bg_un = BaggingRegressor(n_estimators=60, random_state=SEED).fit(data.X_train, data.y_train)
    bg_sc = BaggingRegressor(n_estimators=60, random_state=SEED).fit(data.X_train_s, data.y_train_s)
    models_out["bagging"] = {
        "title": "3.2. Бэггинг (Bagging)",
        "hyperparams": "n_estimators=60, max_features=5",
        "unscaled_test": bg_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(bg_sc.predict(data.X_train_s)),
            data.unscale_y(bg_sc.predict(data.X_val_s)),
            data.unscale_y(bg_sc.predict(data.X_test_s)),
        ),
    }

    # 3.3 Градиентный бустинг (Gradient Boosting)
    gb_un = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=4, min_samples_leaf=3, random_state=SEED
    ).fit(data.X_train, data.y_train)
    gb_sc = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=4, min_samples_leaf=3, random_state=SEED
    ).fit(data.X_train_s, data.y_train_s)
    models_out["gb"] = {
        "title": "3.3. Градиентный бустинг (Gradient Boosting)",
        "hyperparams": "n_estimators=100, lr=0.08, max_depth=4",
        "unscaled_test": gb_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(gb_sc.predict(data.X_train_s)),
            data.unscale_y(gb_sc.predict(data.X_val_s)),
            data.unscale_y(gb_sc.predict(data.X_test_s)),
        ),
    }

    return models_out
