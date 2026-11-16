"""Загрузка датасета из папки data/, формирование сдвигов на 1 и 7 шагов, разбиение и масштабирование."""

import pathlib
import dataclasses
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    SEED, VARIANT_NAME, FEATURE_NAMES, TARGET_NAME,
    MULTISTEP_HORIZON, TRAIN_RATIO, VAL_RATIO, DATASET_PATH
)


@dataclasses.dataclass
class PreparedData:
    df: pd.DataFrame
    corr_matrix: pd.DataFrame
    best_feat_idx: int
    best_feat_name: str
    # Одношаговая задача: X_t -> y_{t+1}
    X_all: np.ndarray
    y_all: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_s: np.ndarray
    y_train_s: np.ndarray
    X_val_s: np.ndarray
    y_val_s: np.ndarray
    X_test_s: np.ndarray
    y_test_s: np.ndarray
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    # Многошаговая задача (7 шагов вперёд): X_t -> Y_{t+1..t+7}
    Xm_tr: np.ndarray
    Ym_tr: np.ndarray
    Xm_va: np.ndarray
    Ym_va: np.ndarray
    Xm_te: np.ndarray
    Ym_te: np.ndarray
    Xm_tr_s: np.ndarray
    Ym_tr_s: np.ndarray
    Xm_va_s: np.ndarray
    Ym_va_s: np.ndarray
    Xm_te_s: np.ndarray
    Ym_te_s: np.ndarray
    scaler_Xm: StandardScaler
    scaler_Ym: StandardScaler

    def unscale_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()

    def unscale_Ym(self, Ym_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_Ym.inverse_transform(np.asarray(Ym_scaled))


def load_and_prepare_data(xlsx_path: pathlib.Path = DATASET_PATH) -> PreparedData:
    """Читает Вариант 12 из data/Lab3_Dataset.xlsx и выполняет п. 1.2–1.3."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Датасет не найден по пути: {xlsx_path}")

    df_raw = pd.read_excel(xlsx_path)
    col_idx = df_raw.columns.get_loc(VARIANT_NAME)
    cols = df_raw.iloc[0, col_idx:col_idx + 6].tolist()
    df = df_raw.iloc[1:, col_idx:col_idx + 6].astype(float).reset_index(drop=True)
    df.columns = cols

    # Сдвиг на 1 шаг вперёд: X_t -> y_{t+1}
    X_all = df[FEATURE_NAMES].iloc[:-1].values
    y_all = df[TARGET_NAME].iloc[1:].values
    N = len(X_all)

    # Матрица корреляций (п. 1.3)
    df_corr = pd.DataFrame(X_all, columns=FEATURE_NAMES)
    df_corr["y(t+1)"] = y_all
    corr_matrix = df_corr.corr()
    best_feat_idx = int(np.argmax(np.abs(corr_matrix.values[:-1, -1])))
    best_feat_name = FEATURE_NAMES[best_feat_idx]

    # Разбиение на train / val / test (70% / 15% / 15%)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_train = int(TRAIN_RATIO * N)
    n_val = int(VAL_RATIO * N)

    idx_train = np.sort(perm[:n_train])
    idx_val = np.sort(perm[n_train:n_train + n_val])
    idx_test = np.sort(perm[n_train + n_val:])

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_test, y_test = X_all[idx_test], y_all[idx_test]

    # Масштабирование StandardScaler строго по параметрам train-выборки (п. 1.2)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Формирование многошаговой выборки на 7 шагов вперёд (п. 4)
    N_multi = len(df) - MULTISTEP_HORIZON
    X_multi = df[FEATURE_NAMES].iloc[:N_multi].values
    Y_multi = np.column_stack([
        df[TARGET_NAME].iloc[s:s + N_multi].values
        for s in range(1, MULTISTEP_HORIZON + 1)
    ])

    rng_m = np.random.default_rng(SEED)
    perm_m = rng_m.permutation(N_multi)
    n_tr_m = int(TRAIN_RATIO * N_multi)
    n_va_m = int(VAL_RATIO * N_multi)
    idx_tr_m = np.sort(perm_m[:n_tr_m])
    idx_va_m = np.sort(perm_m[n_tr_m:n_tr_m + n_va_m])
    idx_te_m = np.sort(perm_m[n_tr_m + n_va_m:])

    Xm_tr, Ym_tr = X_multi[idx_tr_m], Y_multi[idx_tr_m]
    Xm_va, Ym_va = X_multi[idx_va_m], Y_multi[idx_va_m]
    Xm_te, Ym_te = X_multi[idx_te_m], Y_multi[idx_te_m]

    scaler_Xm = StandardScaler().fit(Xm_tr)
    scaler_Ym = StandardScaler().fit(Ym_tr)

    Xm_tr_s = scaler_Xm.transform(Xm_tr)
    Xm_va_s = scaler_Xm.transform(Xm_va)
    Xm_te_s = scaler_Xm.transform(Xm_te)

    Ym_tr_s = scaler_Ym.transform(Ym_tr)
    Ym_va_s = scaler_Ym.transform(Ym_va)
    Ym_te_s = scaler_Ym.transform(Ym_te)

    return PreparedData(
        df=df, corr_matrix=corr_matrix,
        best_feat_idx=best_feat_idx, best_feat_name=best_feat_name,
        X_all=X_all, y_all=y_all,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_train_s=X_train_s, y_train_s=y_train_s,
        X_val_s=X_val_s, y_val_s=y_val_s,
        X_test_s=X_test_s, y_test_s=y_test_s,
        scaler_X=scaler_X, scaler_y=scaler_y,
        Xm_tr=Xm_tr, Ym_tr=Ym_tr,
        Xm_va=Xm_va, Ym_va=Ym_va,
        Xm_te=Xm_te, Ym_te=Ym_te,
        Xm_tr_s=Xm_tr_s, Ym_tr_s=Ym_tr_s,
        Xm_va_s=Xm_va_s, Ym_va_s=Ym_va_s,
        Xm_te_s=Xm_te_s, Ym_te_s=Ym_te_s,
        scaler_Xm=scaler_Xm, scaler_Ym=scaler_Ym,
    )
