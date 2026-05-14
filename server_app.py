from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context

def server_fn(context: Context):
    # 從 pyproject.toml 讀取設定或預設跑 3 輪 [11, 6, 7]
    num_rounds = context.run_config.get("num-server-rounds", 3)
    config = ServerConfig(num_rounds=num_rounds)

    strategy = FedAvg(
        fraction_fit=1.0,
        min_available_clients=2,
    )
    return ServerAppComponents(strategy=strategy, config=config)

# 使用 server_fn 消除 ServerApp Constructor 警告
app = ServerApp(server_fn=server_fn)