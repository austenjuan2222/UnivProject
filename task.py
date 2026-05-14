import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 定義簡單的交通預測模型：y = P_0 + P_1 * x
class TrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Linear(1, 1)
    def forward(self, x):
        return self.regressor(x)

def load_data():
    # 模擬簡單的交通數據
    X = torch.randn(100, 1)
    y = 2 * X + 0.5 + torch.randn(100, 1) * 0.1
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=32), DataLoader(ds, batch_size=32)