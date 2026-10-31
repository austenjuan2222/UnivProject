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
    residual=True 時預測「相對最後一筆的變化量」：輸出 = x[:, -1] + MLP(x)，
    初始時就接近 naive（下一筆 = 最後一筆），模型只需學修正量。
    """

    def __init__(self, lookback: int = 12, hidden: int = 32, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out = self.net(x).squeeze(-1)
        return out + x[:, -1] if self.residual else out


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


LOSSES = {
    "mse": F.mse_loss,
    "huber": F.smooth_l1_loss,   # 誤差 < 1（標準化尺度）時同 MSE，較大時線性，對事故等極端值較穩
    "l1": F.l1_loss,
}


def train(model, loader, optimizer, epochs: int, device="cpu", loss: str = "mse") -> tuple[float, int]:
    """回傳（平均訓練 loss, optimizer.step 次數）。步數用於隱私累計。"""
    loss_fn = LOSSES[loss]
    model.to(device).train()
    total, n, steps = 0.0, 0, 0
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if len(X) == 0:
                # Poisson 取樣可能抽到空批次：仍要走一步（Opacus 會只加雜訊），
                # 不能 skip，否則實際步數與隱私計算不一致。
                batch_loss = model(X).sum() * 0.0
            else:
                batch_loss = loss_fn(model(X), y)
                total += batch_loss.item() * len(X)
                n += len(X)
            batch_loss.backward()
            optimizer.step()
            steps += 1
    return (total / n if n else float("nan")), steps


@torch.no_grad()
def evaluate(model, X, y, mean: float, std: float, device="cpu") -> dict:
    """回傳標準化尺度的 MSE，以及還原成 km/h 的 MAE、MSE。

    naive_mae_kmh 是「下一筆 = 最後一筆」（persistence）的 MAE，作為模型是否有學到東西的基準。
    """
    model = unwrap(model).to(device).eval()
    Xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    pred = model(Xt)
    loss = F.mse_loss(pred, yt).item()
    err = (pred - yt) * std
    naive = ((Xt[:, -1] - yt) * std).abs().mean().item()
    return {"loss": loss, "mae_kmh": err.abs().mean().item(), "mse_kmh": (err**2).mean().item(),
            "naive_mae_kmh": naive}
