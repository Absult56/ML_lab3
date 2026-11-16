# -*- coding: utf-8 -*-
"""П. 1.4–1.6: Парная линейная, множественная линейная и нелинейная регрессия (SciPy curve_fit)."""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from src.data_loader import PreparedData


def nonlinear_func(X_T, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11):
    """Аппроксимирующая полиномиальная функция 2-го порядка от всех 5 признаков для curve_fit."""
    a, b, c, d, e = X_T[0], X_T[1], X_T[2], X_T[3], X_T[4]
    return (
        w0 + w1 * a + w2 * b + w3 * c + w4 * d + w5 * e
        + w6 * (a ** 2) + w7 * (b ** 2) + w8 * (c ** 2)
        + w9 * (d ** 2) + w10 * (e ** 2) + w11 * (a * b)
    )


def run_base_regressions(data: PreparedData):
    """Обучает парную, множественную и нелинейную регрессию (с масштабированием и без)."""
    fi = data.best_feat_idx
    fn = data.best_feat_name
    models_out = {}

    # 1.4 Парная линейная регрессия по наиболее коррелирующему признаку
    lr_pair_un = LinearRegression().fit(data.X_train[:, [fi]], data.y_train)
    lr_pair_sc = LinearRegression().fit(data.X_train_s[:, [fi]], data.y_train_s)
    models_out["simple_lr"] = {
        "title": f"1.4. Парная линейная регрессия (признак {fn})",
        "hyperparams": f"OLS (признак {fn})",
        "unscaled_test": lr_pair_un.predict(data.X_test[:, [fi]]),
        "scaled_preds": (
            data.unscale_y(lr_pair_sc.predict(data.X_train_s[:, [fi]])),
            data.unscale_y(lr_pair_sc.predict(data.X_val_s[:, [fi]])),
            data.unscale_y(lr_pair_sc.predict(data.X_test_s[:, [fi]])),
        ),
        "model_unscaled": lr_pair_un,
    }

    # 1.5 Множественная линейная регрессия по всем 5 признакам
    mlr_un = LinearRegression().fit(data.X_train, data.y_train)
    mlr_sc = LinearRegression().fit(data.X_train_s, data.y_train_s)
    models_out["multiple_lr"] = {
        "title": "1.5. Множественная линейная регрессия",
        "hyperparams": "OLS (все 5 признаков)",
        "unscaled_test": mlr_un.predict(data.X_test),
        "scaled_preds": (
            data.unscale_y(mlr_sc.predict(data.X_train_s)),
            data.unscale_y(mlr_sc.predict(data.X_val_s)),
            data.unscale_y(mlr_sc.predict(data.X_test_s)),
        ),
        "model_unscaled": mlr_un,
    }

    # 1.6 Нелинейная регрессия с помощью curve_fit() из SciPy
    p0 = np.zeros(12)
    popt_un, _ = curve_fit(nonlinear_func, data.X_train.T, data.y_train, p0=p0, maxfev=20000)
    popt_sc, _ = curve_fit(nonlinear_func, data.X_train_s.T, data.y_train_s, p0=p0, maxfev=20000)
    models_out["nonlinear"] = {
        "title": "1.6. Нелинейная регрессия (SciPy curve_fit)",
        "hyperparams": "Полином 2-й ст. (12 параметров)",
        "unscaled_test": nonlinear_func(data.X_test.T, *popt_un),
        "scaled_preds": (
            data.unscale_y(nonlinear_func(data.X_train_s.T, *popt_sc)),
            data.unscale_y(nonlinear_func(data.X_val_s.T, *popt_sc)),
            data.unscale_y(nonlinear_func(data.X_test_s.T, *popt_sc)),
        ),
        "popt_unscaled": popt_un,
    }

    return models_out
