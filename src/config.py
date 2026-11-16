# -*- coding: utf-8 -*-
"""Конфигурация проекта: фиксация seed, параметры варианта и пути к папкам data/ и results/."""

import sys
import pathlib
import numpy as np

SEED = 42
VARIANT_NAME = "Вариант 12"
FEATURE_NAMES = ["a", "b", "c", "d", "e"]
TARGET_NAME = "y"
MULTISTEP_HORIZON = 7

# Пропорции разбиения выборок (п. 1.2: на train не менее 50%)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Определение корневой директории проекта (D:\projects\python\ML_lab\lab3_gm на Windows)
DEFAULT_WIN_DIR = pathlib.Path(r"D:\projects\python\ML_lab\lab3_gm")
PROJECT_ROOT = DEFAULT_WIN_DIR if (sys.platform.startswith("win") and DEFAULT_WIN_DIR.exists()) else pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_5_1_DIR = RESULTS_DIR / "plots_5_1"

DATASET_PATH = DATA_DIR / "Lab3_Dataset.xlsx"
COMPARISON_XLSX_PATH = RESULTS_DIR / "results_Comparison.xlsx"


def init_directories_and_seed():
    """Создаёт папки data/, results/, results/plots_5_1/ и фиксирует генераторы случайных чисел."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_5_1_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except ImportError:
        pass
