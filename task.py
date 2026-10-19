"""模型、訓練與評估（PyTorch）。資料處理在 data.py，隱私累計在 privacy.py。"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TrafficModel(nn.Module):
    """以過去 lookback 個 5 分鐘車速（已標準化）預測下一個 5 分鐘車速。

    小型 MLP：只用 Linear/ReLU，Opacus 可直接計算 per-sample gradient。
    """

    def __init__(self, lookback: int = 12, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def make_loader(X, y, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def unwrap(model: nn.Module) -> nn.Module:
    """Opacus 會把模型包成 GradSampleModule（state_dict 鍵多 `_module.`），回傳原模型。"""
    return model._module if hasattr(model, "_module") else model


def train(model, loader, optimizer, epochs: int, device="cpu") -> tuple[float, int]:
    """回傳（平均訓練 loss, optimizer.step 次數）。步數用於隱私累計。"""
    model.to(device).train()
    total, n, steps = 0.0, 0, 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if len(X) == 0:
                # Poisson 取樣可能抽到空批次：仍要走一步（Opacus 會只加雜訊），
                # 不能 skip，否則實際步數與隱私計算不一致。
                loss = model(X).sum() * 0.0
            else:
                loss = F.mse_loss(model(X), y)
                total += loss.item() * len(X)
                n += len(X)
            loss.backward()
            optimizer.step()
            steps += 1
    return (total / n if n else float("nan")), steps


@torch.no_grad()
def evaluate(model, X, y, mean: float, std: float, device="cpu") -> dict:
    """回傳標準化尺度的 MSE，以及還原成 km/h 的 MAE、MSE。"""
    model = unwrap(model).to(device).eval()
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    pred = model(Xt)
    loss = F.mse_loss(pred, yt).item()
    err = (pred - yt) * std
    return {"loss": loss, "mae_kmh": err.abs().mean().item(), "mse_kmh": (err**2).mean().item()}
