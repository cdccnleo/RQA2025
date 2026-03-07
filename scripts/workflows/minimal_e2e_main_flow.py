import sys
import numpy as np
import pandas as pd
from src.features.feature_manager import FeatureManager
from src.models.model_manager import ModelManager
from src.trading.trading_engine import TradingEngine
from src.trading.backtester import BacktestEngine

# 1. 构造最小化原始数据
dates = pd.date_range("2023-01-01", periods=5)
raw_data = pd.DataFrame({
    "date": dates,
    "close": np.random.uniform(10, 20, 5),
    "high": np.random.uniform(15, 25, 5),
    "low": np.random.uniform(5, 15, 5),
    "volume": np.random.randint(1000, 2000, 5)
}).set_index("date")
news_data = pd.DataFrame(index=dates)

# 2. 特征工程
model_manager = ModelManager()
feature_manager = FeatureManager(model_path="tmp_e2e_feature",
                                 stock_code="000001.SZ", model_manager=model_manager)
try:
    features = feature_manager.execute_pipeline(raw_data, news_data)
    print("特征工程输出:")
    print(features)
except Exception as e:
    print(f"特征工程异常: {e}")
    sys.exit(1)

# 3. 模型训练与推理
try:
    train_df = features.copy()
    train_df["target"] = np.random.randint(0, 2, len(train_df))
    model = model_manager.train_model(train_df, model_name="e2e_test_model")
    model_manager.save_model(model, model_name="e2e_test_model")
    loaded_model = model_manager.load_model(model_name="e2e_test_model")
    preds = model_manager.predict(model=loaded_model, data=features)
    print("模型推理结果:", preds)
except Exception as e:
    print(f"模型训练/推理异常: {e}")
    sys.exit(1)

# 4. 交易信号生成
try:
    trading_engine = TradingEngine(risk_config={
                                   "market_type": "A", "initial_capital": 1000000.0, "per_trade_risk": 0.01, "max_position": {"000001.SZ": 1000}})
    signals = pd.DataFrame({
        "symbol": "000001.SZ",
        "signal": [1 if p > 0.5 else 0 for p in preds],
        "strength": np.ones(len(preds))
    })
    current_prices = {"000001.SZ": float(raw_data["close"].iloc[-1])}
    orders = trading_engine.generate_orders(signals, current_prices)
    print("交易信号与订单:", orders)
except Exception as e:
    print(f"交易信号生成异常: {e}")
    sys.exit(1)

# 5. 回测与绩效分析
try:
    backtest_data = raw_data.copy()
    backtest_data["open"] = backtest_data["close"]  # 简化处理
    engine = BacktestEngine(config_path="tmp_e2e_backtest_config.json")
    report = engine.run(
        strategy="MinimalStrategy",
        data=backtest_data,
        portfolio_params={}
    )
    print("回测报告:", report)
    assert "performance" in report
    print("SUCCESS: Minimal e2e main flow test passed.")
except Exception as e:
    print(f"回测主流程异常: {e}")
    sys.exit(1)
