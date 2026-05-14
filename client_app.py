import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
from flwr.client import NumPyClient, ClientApp
from task import TrafficModel, load_data

class ITSClient(NumPyClient):
    def __init__(self):
        self.model = TrafficModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.train_loader, self.test_loader = load_data() # 載入訓練與測試集
        
        # 差異隱私保護機制 [8, 9, 10]
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model, optimizer=self.optimizer, data_loader=self.train_loader,
            noise_multiplier=1.1, max_grad_norm=1.0
        )

    def fit(self, parameters, config):
        # 接收全局模型並訓練
        self.set_parameters(parameters)
        self.model.train()
        for batch in self.train_loader:
            X, y = batch
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(X), y)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    # 關鍵修正點：新增評估方法以解決 5 failures 問題 [4, 1]
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                X, y = batch
                loss += F.mse_loss(self.model(X), y).item()
        return loss / len(self.test_loader), len(self.test_loader.dataset), {}

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

# 關鍵修正點：使用.to_client() 消除 Deprecation Warning [5]
app = ClientApp(client_fn=lambda context: ITSClient().to_client())