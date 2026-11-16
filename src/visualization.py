"""Построение и сохранение всех графиков лабораторной работы в каталог results/."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src.config import FEATURE_NAMES, TARGET_NAME, MULTISTEP_HORIZON, RESULTS_DIR, PLOTS_5_1_DIR
from src.data_loader import PreparedData
from src.base_models import nonlinear_func

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def save_eda_plots(data: PreparedData):
    """Сохраняет графики п. 1.1 (все признаки и y) и п. 1.3 (матрица корреляций) в results/."""
    df = data.df
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#111111"]
    for col, clr in zip(FEATURE_NAMES, colors[:5]):
        ax.plot(df.index, df[col], label=f"Признак {col}", lw=1.6, color=clr)
    ax.plot(df.index, df[TARGET_NAME], label="Целевая переменная y", lw=2.2, color=colors[5], linestyle="--")
    ax.set_title("П. 1.1. Динамика признаков (a, b, c, d, e) и целевой переменной (y) — Вариант 12", fontweight="bold")
    ax.set_xlabel("Отсчёт времени t")
    ax.set_ylabel("Значение")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_all_features_and_target.png")
    plt.close(fig)

    corr_mat = data.corr_matrix
    fig_c, ax_c = plt.subplots(figsize=(7.5, 6), dpi=160)
    cax = ax_c.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig_c.colorbar(cax, ax=ax_c, fraction=0.046, pad=0.04, label="Коэффициент корреляции Пирсона (r)")
    labels_c = FEATURE_NAMES + ["y(t+1)"]
    ax_c.set_xticks(range(len(labels_c)))
    ax_c.set_yticks(range(len(labels_c)))
    ax_c.set_xticklabels(labels_c, fontweight="bold")
    ax_c.set_yticklabels(labels_c, fontweight="bold")
    for i in range(len(labels_c)):
        for j in range(len(labels_c)):
            v = corr_mat.values[i, j]
            ax_c.text(j, i, f"{v:.4f}", ha="center", va="center", color="white" if abs(v) > 0.65 else "black", fontweight="bold")
    ax_c.set_title("П. 1.3. Матрица взаимных корреляций признаков и y(t+1) (Вариант 12)", fontweight="bold")
    fig_c.tight_layout()
    fig_c.savefig(RESULTS_DIR / "02_correlation_matrix.png")
    plt.close(fig_c)


def save_approx_and_tree_plots(data: PreparedData, base_models: dict, single_models: dict):
    """Сохраняет графики п. 5.2 (аппроксимирующие кривые) и п. 5.3 (дендрограмма дерева) в results/."""
    fi = data.best_feat_idx
    fn = data.best_feat_name
    t_all = np.arange(len(data.X_all))

    fig_ap, axes_ap = plt.subplots(1, 3, figsize=(16, 5.2), dpi=160)
    c_grid = np.linspace(data.df[fn].min(), data.df[fn].max(), 300).reshape(-1, 1)
    lr_pair_un = base_models["simple_lr"]["model_unscaled"]
    mlr_un = base_models["multiple_lr"]["model_unscaled"]
    popt_un = base_models["nonlinear"]["popt_unscaled"]

    axes_ap[0].scatter(data.X_train[:, fi], data.y_train, color="#1f77b4", s=14, alpha=0.5, label="Исходные данные (Train)")
    axes_ap[0].scatter(data.X_test[:, fi], base_models["simple_lr"]["scaled_preds"][2], color="#2ca02c", s=22, marker="^", label="Прогноз (Test)")
    axes_ap[0].plot(c_grid[:, 0], lr_pair_un.predict(c_grid), color="#d62728", linestyle="--", lw=2.2, label="Аппроксимирующая прямая")
    axes_ap[0].set_title(f"1.4. Парная линейная регрессия (по {fn})", fontweight="bold")
    axes_ap[0].grid(True, linestyle=":", alpha=0.6)
    axes_ap[0].legend(fontsize=8.5)

    axes_ap[1].scatter(data.idx_train, data.y_train, color="#1f77b4", s=14, alpha=0.5, label="Исходные данные (Train)")
    axes_ap[1].scatter(data.idx_test, base_models["multiple_lr"]["scaled_preds"][2], color="#ff7f0e", s=22, label="Прогноз (Test)")
    axes_ap[1].plot(t_all, mlr_un.predict(data.X_all), color="#d62728", linestyle="--", lw=2.0, label="Аппроксимирующая функция")
    axes_ap[1].set_title("1.5. Множественная линейная регрессия", fontweight="bold")
    axes_ap[1].grid(True, linestyle=":", alpha=0.6)
    axes_ap[1].legend(fontsize=8.5)

    axes_ap[2].scatter(data.idx_train, data.y_train, color="#1f77b4", s=14, alpha=0.5, label="Исходные данные (Train)")
    axes_ap[2].scatter(data.idx_test, base_models["nonlinear"]["scaled_preds"][2], color="#2ca02c", s=24, marker="D", label="Прогноз (Test)")
    axes_ap[2].plot(t_all, nonlinear_func(data.X_all.T, *popt_un), color="#d62728", linestyle="--", lw=2.2, label="Нелинейная аппроксимация curve_fit")
    axes_ap[2].set_title("1.6. Нелинейная регрессия (SciPy curve_fit)", fontweight="bold")
    axes_ap[2].grid(True, linestyle=":", alpha=0.6)
    axes_ap[2].legend(fontsize=8.5)

    fig_ap.suptitle("П. 5.2. Исходные данные, аппроксимирующая функция и прогноз", fontweight="bold")
    fig_ap.tight_layout()
    fig_ap.savefig(RESULTS_DIR / "03_regression_approx_curves.png")
    plt.close(fig_ap)

    # П. 5.3 Дендрограмма дерева решений
    dt_vis = single_models["dt"]["vis_tree"]
    fig_t, ax_t = plt.subplots(figsize=(15, 8), dpi=160)
    plot_tree(dt_vis, feature_names=FEATURE_NAMES, filled=True, rounded=True, fontsize=9, ax=ax_t)
    ax_t.set_title("П. 5.3. Дендрограмма дерева принятия решений (DecisionTreeRegressor)", fontweight="bold")
    fig_t.tight_layout()
    fig_t.savefig(RESULTS_DIR / "04_decision_tree_dendrogram.png")
    plt.close(fig_t)


def save_nn_and_template_plots(
    data: PreparedData,
    one_step_models: dict,
    hist_1step: dict,
    hist_7step: dict,
    param_counts: dict,
    multistep_preds: dict,
):
    """Сохраняет кривые обучения ИНС (п. 4) и шаблонные графики прогнозов/остатков (п. 5.1) в results/."""
    for horizon_label, hist_dict, fname in [
        ("Прогноз на 1 шаг вперёд", hist_1step, "05_nn_learning_curves_1step.png"),
        ("Прогноз на 7 шагов вперёд", hist_7step, "06_nn_learning_curves_7step.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
        colors_nn = {"slp": "#ff7f0e", "mlp": "#1f77b4", "deep": "#2ca02c"}
        labels_nn = {
            "slp": "4.1. Однослойный персептрон",
            "mlp": "4.2. Многослойный персептрон (MLP)",
            "deep": "4.3. Глубокая сеть (ResNet-MLP)",
        }
        for k_code, h in hist_dict.items():
            short_k = k_code.split("_")[0]
            clr = colors_nn[short_k]
            lbl = f"{labels_nn[short_k]} ({param_counts[k_code]} парам.)"
            ep_arr = np.arange(1, len(h["train_loss"]) + 1)
            axes[0].plot(ep_arr, h["train_loss"], color=clr, lw=1.8, label=f"{lbl} — Train")
            axes[0].plot(ep_arr, h["val_loss"], color=clr, linestyle="--", lw=1.6, label=f"{lbl} — Val")
            axes[1].plot(ep_arr, h["train_mae"], color=clr, lw=1.8, label=f"{lbl} — Train")
            axes[1].plot(ep_arr, h["val_mae"], color=clr, linestyle="--", lw=1.6, label=f"{lbl} — Val")

        for ax_i, metric_name in zip(axes, ["Функция потерь MSE", "Метрика MAE"]):
            ax_i.set_yscale("log")
            ax_i.set_title(f"{metric_name} (лог. шкала) — {horizon_label}", fontweight="bold")
            ax_i.set_xlabel("Эпоха обучения")
            ax_i.grid(True, which="both", linestyle=":", alpha=0.6)
            ax_i.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / fname)
        plt.close(fig)

    # Шаблонные графики п. 5.1
    def _plot_5_1(ax, ptr, pva, pte, title_str, is_res=False, show_leg=True):
        n_tr, n_va, n_te = len(data.y_train), len(data.y_val), len(data.y_test)
        x_tr = np.arange(0, n_tr)
        x_va = np.arange(n_tr, n_tr + n_va)
        x_te = np.arange(n_tr + n_va, n_tr + n_va + n_te)
        if not is_res:
            ax.plot(x_tr, data.y_train, color="#1f77b4", lw=1.1, label="Actual Train")
            ax.plot(x_tr, ptr, color="#ff7f0e", lw=1.0, label="Predicted Train")
            ax.plot(x_va, data.y_val, color="#2ca02c", lw=1.1, label="Actual Validation")
            ax.plot(x_va, pva, color="#d62728", lw=1.0, label="Predicted Validation")
            ax.plot(x_te, data.y_test, color="#9467bd", lw=1.1, label="Actual Test")
            ax.plot(x_te, pte, color="#8c564b", lw=1.0, label="Predicted Test")
        else:
            ax.plot(x_tr, data.y_train - ptr, color="#ff7f0e", lw=1.0, label="Residual Train")
            ax.plot(x_va, data.y_val - pva, color="#d62728", lw=1.0, label="Residual Validation")
            ax.plot(x_te, data.y_test - pte, color="#8c564b", lw=1.0, label="Residual Test")
            ax.axhline(0.0, color="#333333", lw=0.8, linestyle=":")
        ax.axvline(n_tr, color="gray", linestyle="--", lw=1.2, label="Train/Validation Split")
        ax.axvline(n_tr + n_va, color="gray", linestyle="--", lw=1.2, label="Validation/Test Split")
        ymin, ymax = ax.get_ylim()
        y_txt = ymin + 0.88 * (ymax - ymin)
        ax.text(n_tr * 0.45, y_txt, "Train", fontweight="bold", ha="center")
        ax.text(n_tr + n_va * 0.5, y_txt, "Validation", fontweight="bold", ha="center")
        ax.text(n_tr + n_va + n_te * 0.5, y_txt, "Test", fontweight="bold", ha="center")
        ax.set_title(title_str, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        if show_leg:
            ax.legend(loc="upper left", fontsize=7.5)

    fig_p, axes_p = plt.subplots(4, 3, figsize=(18, 15), dpi=150)
    fig_r, axes_r = plt.subplots(4, 3, figsize=(18, 15), dpi=150)
    for idx_m, (m_code, m_info) in enumerate(one_step_models.items()):
        ptr, pva, pte = m_info["scaled_preds"]
        m_title = m_info["title"]
        r_i, c_i = divmod(idx_m, 3)
        _plot_5_1(axes_p[r_i, c_i], ptr, pva, pte, m_title, False, idx_m == 0)
        _plot_5_1(axes_r[r_i, c_i], ptr, pva, pte, f"Ошибка: {m_title}", True, idx_m == 0)

        f_i, ax_i = plt.subplots(2, 1, figsize=(11, 7.5), dpi=150, sharex=True)
        _plot_5_1(ax_i[0], ptr, pva, pte, f"П. 5.1. Прогноз — {m_title}", False, True)
        _plot_5_1(ax_i[1], ptr, pva, pte, f"П. 5.1. Разность (y - ŷ) — {m_title}", True, True)
        f_i.tight_layout()
        f_i.savefig(PLOTS_5_1_DIR / f"5_1_{m_code}.png")
        plt.close(f_i)

    fig_p.tight_layout()
    fig_p.savefig(RESULTS_DIR / "07_template_predictions_all_methods.png")
    plt.close(fig_p)
    fig_r.tight_layout()
    fig_r.savefig(RESULTS_DIR / "08_template_residuals_all_methods.png")
    plt.close(fig_r)

    # Многошаговый прогноз на 7 шагов вперёд
    fig_m, ax_m = plt.subplots(figsize=(11, 5.2), dpi=160)
    for s_i, clr in zip([15, 65, 115], ["#1f77b4", "#2ca02c", "#9467bd"]):
        steps_x = np.arange(1, MULTISTEP_HORIZON + 1)
        ax_m.plot(steps_x, data.Ym_te[s_i], color=clr, marker="o", lw=2.2, label=f"Истинный вектор Y (тест #{s_i})")
        ax_m.plot(steps_x, multistep_preds["deep_7"][s_i], color=clr, marker="s", linestyle="--", lw=1.8, label=f"ResNet-MLP (тест #{s_i})")
        ax_m.plot(steps_x, multistep_preds["mlp_7"][s_i], color=clr, marker="^", linestyle=":", lw=1.5, label=f"MLP (тест #{s_i})")
    ax_m.set_title("П. 4. Многошаговое прогнозирование на 7 шагов вперёд (тестовая выборка)", fontweight="bold")
    ax_m.set_xlabel("Горизонт прогноза (шаг k = 1..7)")
    ax_m.set_ylabel("Значение y(t+k)")
    ax_m.grid(True, linestyle=":", alpha=0.6)
    ax_m.legend(ncol=3, fontsize=8)
    fig_m.tight_layout()
    fig_m.savefig(RESULTS_DIR / "09_multistep_7steps_forecast.png")
    plt.close(fig_m)
