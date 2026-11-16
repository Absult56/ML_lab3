# -*- coding: utf-8 -*-
"""П. 4.1–4.3: Нейронные сети на PyTorch для одношагового и 7-шагового прогнозирования."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.config import SEED
from src.data_loader import PreparedData


class SingleLayerPerceptron(nn.Module):
    """П. 4.1. Однослойный линейный персептрон с функцией активации PReLU."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.PReLU(num_parameters=out_features, init=0.8)
        self.scale = nn.Parameter(torch.ones(1, out_features))
        self.shift = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, x):
        return self.act(self.linear(x)) * self.scale + self.shift


class MLPRegressorNet(nn.Module):
    """П. 4.2. Многослойный персептрон (MLP: 5 -> 64 -> 32 -> out_features)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.net(x)


class DeepResNetMLP(nn.Module):
    """П. 4.3. Глубокая нейронная сеть с остаточными (Residual/Skip) связями."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_features, 64), nn.SiLU())
        self.res_block = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, out_features),
        )
        self.shortcut = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.res_block(h)
        return self.head(h) + self.shortcut(x)


def count_parameters(model: nn.Module) -> int:
    """Подсчитывает общее число обучаемых внутренних параметров сети."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_pytorch_model(
    model: nn.Module,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    X_te: np.ndarray,
    epochs: int = 220,
    lr: float = 0.008,
    batch_size: int = 64,
):
    """Обучает нейросеть методом Adam и возвращает прогнозы, число параметров и историю кривых обучения."""
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    model = model.to(device)
    y_1d = (Y_tr.ndim == 1)
    Y_tr_2d = Y_tr[:, None] if y_1d else Y_tr
    Y_va_2d = Y_va[:, None] if y_1d else Y_va

    ds_tr = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(Y_tr_2d, dtype=torch.float32),
    )
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Y_tr_t = torch.tensor(Y_tr_2d, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    Y_va_t = torch.tensor(Y_va_2d, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            p_tr = model(X_tr_t)
            p_va = model(X_va_t)
            tr_mse = float(criterion(p_tr, Y_tr_t).item())
            va_mse = float(criterion(p_va, Y_va_t).item())
            tr_mae = float(torch.mean(torch.abs(p_tr - Y_tr_t)).item())
            va_mae = float(torch.mean(torch.abs(p_va - Y_va_t)).item())

        history["train_loss"].append(tr_mse)
        history["val_loss"].append(va_mse)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(va_mae)

        if va_mse < best_val_loss:
            best_val_loss = va_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        ptr = model(X_tr_t).cpu().numpy()
        pva = model(X_va_t).cpu().numpy()
        pte = model(X_te_t).cpu().numpy()

    if y_1d:
        ptr, pva, pte = ptr[:, 0], pva[:, 0], pte[:, 0]
    return ptr, pva, pte, count_parameters(model), history


def run_all_neural_networks(data: PreparedData):
    """Запускает обучение 3 нейросетевых архитектур для прогноза на 1 и 7 шагов вперёд."""
    nn_out = {}
    hist_1step, hist_7step, param_counts, multistep_preds = {}, {}, {}, {}

    specs = [
        ("4.1. Однослойный персептрон", SingleLayerPerceptron, "slp"),
        ("4.2. Многослойный персептрон MLP", MLPRegressorNet, "mlp"),
        ("4.3. Глубокая сеть ResNet-MLP", DeepResNetMLP, "deep"),
    ]

    for title_nn, cls_nn, code_nn in specs:
        # Прогноз на 1 шаг вперёд
        _, _, pte_un_1, _, _ = train_pytorch_model(
            cls_nn(5, 1), data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, epochs=220, lr=0.002
        )
        ptr_sc_1, pva_sc_1, pte_sc_1, n_p1, h1 = train_pytorch_model(
            cls_nn(5, 1), data.X_train_s, data.y_train_s, data.X_val_s, data.y_val_s, data.X_test_s, epochs=220, lr=0.01
        )
        key_1 = f"{code_nn}_1"
        hist_1step[key_1] = h1
        param_counts[key_1] = n_p1
        nn_out[key_1] = {
            "title": f"{title_nn} (1 шаг)",
            "hyperparams": f"Adam, MSE, params={n_p1}",
            "unscaled_test": pte_un_1,
            "scaled_preds": (
                data.unscale_y(ptr_sc_1),
                data.unscale_y(pva_sc_1),
                data.unscale_y(pte_sc_1),
            ),
            "is_multistep": False,
        }

        # Прогноз на 7 шагов вперёд
        _, _, pte_un_7, _, _ = train_pytorch_model(
            cls_nn(5, 7), data.Xm_tr, data.Ym_tr, data.Xm_va, data.Ym_va, data.Xm_te, epochs=220, lr=0.002
        )
        ptr_sc_7, pva_sc_7, pte_sc_7, n_p7, h7 = train_pytorch_model(
            cls_nn(5, 7), data.Xm_tr_s, data.Ym_tr_s, data.Xm_va_s, data.Ym_va_s, data.Xm_te_s, epochs=220, lr=0.01
        )
        key_7 = f"{code_nn}_7"
        hist_7step[key_7] = h7
        param_counts[key_7] = n_p7
        multistep_preds[key_7] = data.unscale_Ym(pte_sc_7)
        nn_out[key_7] = {
            "title": f"{title_nn} (7 шагов)",
            "hyperparams": f"Adam, MSE, out=7, params={n_p7}",
            "unscaled_test": pte_un_7,
            "scaled_preds": (
                data.unscale_Ym(ptr_sc_7),
                data.unscale_Ym(pva_sc_7),
                data.unscale_Ym(pte_sc_7),
            ),
            "is_multistep": True,
        }

    return nn_out, hist_1step, hist_7step, param_counts, multistep_preds
