# -*- coding: utf-8 -*-
"""П. 5.4–5.6: Расчёт метрик MAE, MSE, коэффициента корреляции R и экспорт в results/results_Comparison.xlsx."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import COMPARISON_XLSX_PATH
from src.data_loader import PreparedData


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Вычисляет MAE, MSE и коэффициент корреляции Пирсона R на тестовой выборке."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 1e-12 else 0.0
    return mae, mse, corr


def evaluate_and_export_results(data: PreparedData, all_models: dict) -> pd.DataFrame:
    """Формирует итоговую таблицу сравнения всех методов и сохраняет её в results/results_Comparison.xlsx."""
    rows = []
    for m_code, m_info in all_models.items():
        is_multi = m_info.get("is_multistep", False)
        y_true_ref = data.Ym_te if is_multi else data.y_test

        mae_un, mse_un, r_un = calc_metrics(y_true_ref, m_info["unscaled_test"])
        mae_sc, mse_sc, r_sc = calc_metrics(y_true_ref, m_info["scaled_preds"][2])

        rows.append({
            "Метод": m_info["title"],
            "Гиперпараметры": m_info["hyperparams"],
            "MAE (без масшт.)": round(mae_un, 4),
            "MSE (без масшт.)": round(mse_un, 4),
            "R (без масшт.)": round(r_un, 4),
            "MAE (StandardScaler)": round(mae_sc, 4),
            "MSE (StandardScaler)": round(mse_sc, 4),
            "R (StandardScaler)": round(r_sc, 4),
        })

    df_res = pd.DataFrame(rows)
    df_res.to_excel(COMPARISON_XLSX_PATH, index=False)
    return df_res
