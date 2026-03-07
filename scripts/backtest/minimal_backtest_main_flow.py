import sys
import numpy as np
import pandas as pd
from src.trading.backtester import BacktestEngine

# 1. 构造最小化回测数据
data = pd.DataFrame({
    "open": np.random.uniform(10, 20, 10),
    "high": np.random.uniform(15, 25, 10),
    "low": np.random.uniform(5, 15, 10),
    "close": np.random.uniform(10, 20, 10),
    "volume": np.random.randint(1000, 2000, 10)
})

# 2. 初始化回测引擎
config_path = "tmp_backtest_config.json"
engine = BacktestEngine(config_path=config_path)

# 3. 定义最小化策略


class MinimalStrategy:
    def generate_signals(self, data):
        # 简单全买信号
        return pd.DataFrame({"signal": [1]*len(data)}, index=data.index)


# 4. 执行回测
try:
    report = engine.run(
        strategy="MinimalStrategy",
        data=data,
        portfolio_params={}
    )
    print("回测报告:", report)
    assert "performance" in report
    print("SUCCESS: Minimal backtest main flow test passed.")
except Exception as e:
    print(f"回测主流程异常: {e}")
    sys.exit(1)
