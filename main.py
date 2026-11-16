#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №3. Обучение с учителем. Регрессия. Ансамблевое обучение. Нейронные сети.
Вариант: 12
Точка входа (main.py): последовательный запуск всех модулей из пакета src/,
чтение входного датасета из папки data/ и сохранение всех результатов в папку results/.
"""

import warnings
from src.config import init_directories_and_seed, DATASET_PATH, RESULTS_DIR, COMPARISON_XLSX_PATH
from src.data_loader import load_and_prepare_data
from src.base_models import run_base_regressions
from src.single_models import run_single_ml_models
from src.ensemble_models import run_ensemble_ml_models
from src.neural_nets import run_all_neural_networks
from src.visualization import save_eda_plots, save_approx_and_tree_plots, save_nn_and_template_plots
from src.evaluation import evaluate_and_export_results

warnings.filterwarnings("ignore")


def main():
    init_directories_and_seed()
    print("=== Лабораторная работа №3 (Вариант 12) ===")
    print(f"Входной датасет : {DATASET_PATH}")
    print(f"Папка результатов: {RESULTS_DIR}\n")

    # 1. Загрузка данных, разбиение train/val/test, масштабирование и матрица корреляций
    data = load_and_prepare_data(DATASET_PATH)
    print("Матрица взаимных корреляций с целевой переменной y(t+1):")
    print(data.corr_matrix.round(4))
    print(f"\nПризнак с наибольшей по модулю корреляцией: '{data.best_feat_name}' "
          f"(r = {data.corr_matrix.values[data.best_feat_idx, -1]:.4f})\n")

    save_eda_plots(data)

    # 2. Базовые методы регрессии (п. 1.4–1.6)
    print("[1/4] Обучение базовых моделей регрессии (парная, множественная, нелинейная SciPy curve_fit)...")
    base_models = run_base_regressions(data)

    # 3. Одиночные методы машинного обучения (п. 2.1–2.3)
    print("[2/4] Обучение одиночных моделей МО (KNN, SVR, Decision Tree)...")
    single_models = run_single_ml_models(data)

    # 4. Ансамблевые методы машинного обучения (п. 3.1–3.3)
    print("[3/4] Обучение ансамблевых моделей МО (Random Forest, Bagging, Gradient Boosting)...")
    ensemble_models = run_ensemble_ml_models(data)

    # 5. Нейронные сети на 1 и 7 шагов вперёд (п. 4.1–4.3)
    print("[4/4] Обучение нейронных сетей PyTorch (Однослойный персептрон, MLP, ResNet-MLP на 1 и 7 шагов)...")
    nn_models, hist_1step, hist_7step, param_counts, multistep_preds = run_all_neural_networks(data)

    # 6. Построение и сохранение всех графиков в папку results/
    save_approx_and_tree_plots(data, base_models, single_models)
    one_step_models = {
        **base_models,
        **single_models,
        **ensemble_models,
        "slp_1": nn_models["slp_1"],
        "mlp_1": nn_models["mlp_1"],
        "deep_1": nn_models["deep_1"],
    }
    save_nn_and_template_plots(data, one_step_models, hist_1step, hist_7step, param_counts, multistep_preds)

    # 7. Сводное сравнение метрик и экспорт в results/results_Comparison.xlsx (п. 5.4–5.6)
    all_models = {**one_step_models, "slp_7": nn_models["slp_7"], "mlp_7": nn_models["mlp_7"], "deep_7": nn_models["deep_7"]}
    df_results = evaluate_and_export_results(data, all_models)

    print("\n=== ИТОГОВАЯ ТАБЛИЦА МЕТРИК НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    print(df_results.to_string(index=False))
    print(f"\nВсе графики и таблица успешно сохранены в: {RESULTS_DIR}")
    print(f"Файл сравнения метрик: {COMPARISON_XLSX_PATH}")


if __name__ == "__main__":
    main()
