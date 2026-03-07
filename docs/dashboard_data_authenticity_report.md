# 仪表盘数据来源审计报告

**生成时间**: 2026-01-07T21:09:58.029111

## 摘要

- 模拟数据函数定义: 0
- 模拟数据函数导入: 0
- 模拟数据函数调用: 15
- 硬编码数据: 70
- TODO注释: 10
- 服务层检查: 10
- API路由检查: 15
- 有模拟数据降级的API路由: 0

## 模拟数据函数调用详情

| 文件 | 函数 | 行号 | 有降级注释 |
|------|------|------|------------|
| src\gateway\web\backtest_service.py | _get_mock_backtest_result | 214 | 否 |
| src\gateway\web\data_management_service.py | _get_mock_quality_metrics | 761 | 否 |
| src\gateway\web\data_management_service.py | _get_mock_cache_stats | 762 | 否 |
| src\gateway\web\data_management_service.py | _get_mock_data_lake_stats | 763 | 否 |
| src\gateway\web\data_management_service.py | _get_mock_performance_metrics | 764 | 否 |
| src\gateway\web\model_training_service.py | _get_mock_training_jobs | 187 | 是 |
| src\gateway\web\model_training_service.py | _get_mock_training_metrics | 205 | 否 |
| src\gateway\web\order_routing_service.py | _get_mock_routing_decisions | 206 | 是 |
| src\gateway\web\risk_reporting_service.py | _get_mock_templates | 249 | 是 |
| src\gateway\web\risk_reporting_service.py | _get_mock_generation_tasks | 265 | 否 |
| src\gateway\web\risk_reporting_service.py | _get_mock_report_history | 280 | 否 |
| src\gateway\web\strategy_performance_service.py | _get_mock_strategies | 166 | 是 |
| src\gateway\web\strategy_performance_service.py | _get_mock_performance_metrics | 185 | 否 |
| src\gateway\web\strategy_performance_service.py | _get_mock_strategies | 188 | 否 |
| src\gateway\web\trading_signal_service.py | _get_mock_signals | 164 | 是 |

## API路由状态

| 文件 | 端点数量 | 使用真实服务 | 有模拟数据降级 |
|------|----------|--------------|----------------|
| backtest_routes.py | 0 | 是 | 否 |
| basic_routes.py | 6 | 是 | 否 |
| datasource_routes.py | 10 | 否 | 否 |
| data_management_routes.py | 13 | 是 | 否 |
| feature_engineering_routes.py | 6 | 是 | 否 |
| model_training_routes.py | 5 | 是 | 否 |
| order_routing_routes.py | 3 | 是 | 否 |
| risk_reporting_routes.py | 11 | 是 | 否 |
| strategy_execution_routes.py | 6 | 是 | 否 |
| strategy_lifecycle_routes.py | 3 | 否 | 否 |
| strategy_optimization_routes.py | 9 | 是 | 否 |
| strategy_performance_routes.py | 3 | 是 | 否 |
| strategy_routes.py | 8 | 否 | 否 |
| trading_signal_routes.py | 3 | 是 | 否 |
| websocket_routes.py | 0 | 否 | 否 |

## 服务层状态

| 文件 | 导入真实组件 | 返回空数据 | 有模拟数据降级 |
|------|--------------|------------|----------------|
| backtest_service.py | 是 | 是 | 是 |
| data_management_service.py | 是 | 是 | 是 |
| feature_engineering_service.py | 是 | 是 | 否 |
| model_training_service.py | 是 | 是 | 是 |
| order_routing_service.py | 是 | 是 | 是 |
| risk_reporting_service.py | 是 | 是 | 是 |
| strategy_execution_service.py | 是 | 否 | 否 |
| strategy_optimization_service.py | 是 | 否 | 否 |
| strategy_performance_service.py | 是 | 否 | 是 |
| trading_signal_service.py | 是 | 是 | 是 |

## 硬编码数据

- **src\gateway\web\api_utils.py:229** - 随机生成数据
  ```python
  base_price = random.uniform(10, 500)
  ```
- **src\gateway\web\api_utils.py:235** - 随机生成数据
  ```python
  "date": (current_date - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
  ```
- **src\gateway\web\api_utils.py:236** - 随机生成数据
  ```python
  "open": round(base_price * random.uniform(0.95, 1.05), 2),
  ```
- **src\gateway\web\api_utils.py:237** - 随机生成数据
  ```python
  "high": round(base_price * random.uniform(1.01, 1.08), 2),
  ```
- **src\gateway\web\api_utils.py:238** - 随机生成数据
  ```python
  "low": round(base_price * random.uniform(0.92, 0.99), 2),
  ```
- **src\gateway\web\api_utils.py:239** - 随机生成数据
  ```python
  "close": round(base_price * random.uniform(0.95, 1.05), 2),
  ```
- **src\gateway\web\api_utils.py:240** - 随机生成数据
  ```python
  "volume": random.randint(100000, 10000000),
  ```
- **src\gateway\web\api_utils.py:241** - 随机生成数据
  ```python
  "amount": round(random.uniform(1000000, 100000000), 2),
  ```
- **src\gateway\web\api_utils.py:244** - 随机生成数据
  ```python
  "data_points": random.randint(1000, 5000),
  ```
- **src\gateway\web\api_utils.py:245** - 随机生成数据
  ```python
  "quality_score": round(random.uniform(85, 98), 1)
  ```
- **src\gateway\web\api_utils.py:260** - 随机生成数据
  ```python
  crypto = random.choice(cryptos)
  ```
- **src\gateway\web\api_utils.py:269** - 随机生成数据
  ```python
  "date": (current_date - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
  ```
- **src\gateway\web\api_utils.py:270** - 随机生成数据
  ```python
  "open": round(base_price * random.uniform(0.95, 1.05), 4),
  ```
- **src\gateway\web\api_utils.py:271** - 随机生成数据
  ```python
  "high": round(base_price * random.uniform(1.01, 1.08), 4),
  ```
- **src\gateway\web\api_utils.py:272** - 随机生成数据
  ```python
  "low": round(base_price * random.uniform(0.92, 0.99), 4),
  ```
- **src\gateway\web\api_utils.py:273** - 随机生成数据
  ```python
  "close": round(base_price * random.uniform(0.95, 1.05), 4),
  ```
- **src\gateway\web\api_utils.py:274** - 随机生成数据
  ```python
  "volume": round(random.uniform(1000, 50000), 2),
  ```
- **src\gateway\web\api_utils.py:275** - 随机生成数据
  ```python
  "amount": round(random.uniform(1000000, 50000000), 2),
  ```
- **src\gateway\web\api_utils.py:278** - 随机生成数据
  ```python
  "data_points": random.randint(1000, 5000),
  ```
- **src\gateway\web\api_utils.py:279** - 随机生成数据
  ```python
  "quality_score": round(random.uniform(85, 98), 1)
  ```
- **src\gateway\web\model_training_service.py:193** - 随机生成数据
  ```python
  "model_type": random.choice(["LSTM", "Transformer", "CNN", "RandomForest", "XGBoost"]),
  ```
- **src\gateway\web\model_training_service.py:194** - 随机生成数据
  ```python
  "status": random.choice(["running", "completed", "pending", "failed"]),
  ```
- **src\gateway\web\model_training_service.py:195** - 随机生成数据
  ```python
  "progress": random.randint(0, 100),
  ```
- **src\gateway\web\model_training_service.py:196** - 随机生成数据
  ```python
  "accuracy": random.uniform(0.7, 0.95) if random.random() > 0.3 else None,
  ```
- **src\gateway\web\model_training_service.py:197** - 随机生成数据
  ```python
  "loss": random.uniform(0.1, 0.5) if random.random() > 0.3 else None,
  ```
- **src\gateway\web\model_training_service.py:198** - 随机生成数据
  ```python
  "start_time": int((datetime.now() - timedelta(hours=random.randint(0, 24))).timestamp()),
  ```
- **src\gateway\web\model_training_service.py:199** - 随机生成数据
  ```python
  "training_time": random.randint(30, 300)
  ```
- **src\gateway\web\model_training_service.py:212** - 随机生成数据
  ```python
  {"value": max(0.1, 0.5 - i * 0.008 + random.uniform(-0.01, 0.01)), "epoch": i + 1}
  ```
- **src\gateway\web\model_training_service.py:216** - 随机生成数据
  ```python
  {"value": min(0.95, 0.6 + i * 0.007 + random.uniform(-0.01, 0.01)), "epoch": i + 1}
  ```
- **src\gateway\web\model_training_service.py:221** - 随机生成数据
  ```python
  "gpu_usage": random.uniform(60, 90),
  ```
- **src\gateway\web\model_training_service.py:222** - 随机生成数据
  ```python
  "cpu_usage": random.uniform(40, 70),
  ```
- **src\gateway\web\model_training_service.py:223** - 随机生成数据
  ```python
  "memory_usage": random.uniform(50, 80)
  ```
- **src\gateway\web\order_routing_service.py:212** - 随机生成数据
  ```python
  "routing_strategy": random.choice(["成本优先", "延迟优先", "可靠性优先", "平衡策略"]),
  ```
- **src\gateway\web\order_routing_service.py:213** - 随机生成数据
  ```python
  "target_route": random.choice(["Route A", "Route B", "Route C"]),
  ```
- **src\gateway\web\order_routing_service.py:214** - 随机生成数据
  ```python
  "cost": random.uniform(0.0001, 0.001),
  ```
- **src\gateway\web\order_routing_service.py:215** - 随机生成数据
  ```python
  "latency": random.uniform(1, 50),
  ```
- **src\gateway\web\order_routing_service.py:216** - 随机生成数据
  ```python
  "status": random.choice(["success", "failed", "pending"]),
  ```
- **src\gateway\web\order_routing_service.py:217** - 随机生成数据
  ```python
  "timestamp": int((datetime.now() - timedelta(minutes=random.randint(0, 60))).timestamp()),
  ```
- **src\gateway\web\order_routing_service.py:218** - 随机生成数据
  ```python
  "failure_reason": random.choice(["超时", "连接失败", "余额不足"]) if random.random() < 0.1 else None
  ```
- **src\gateway\web\risk_reporting_service.py:255** - 随机生成数据
  ```python
  "name": random.choice(["每日风险报告", "周度风险报告", "月度风险报告", "实时风险报告"]),
  ```
- **src\gateway\web\risk_reporting_service.py:256** - 随机生成数据
  ```python
  "report_type": random.choice(["daily", "weekly", "monthly", "realtime"]),
  ```
- **src\gateway\web\risk_reporting_service.py:257** - 随机生成数据
  ```python
  "frequency": random.choice(["每日", "每周", "每月", "实时"]),
  ```
- **src\gateway\web\risk_reporting_service.py:258** - 随机生成数据
  ```python
  "status": random.choice(["active", "inactive", "paused"]),
  ```
- **src\gateway\web\risk_reporting_service.py:259** - 随机生成数据
  ```python
  "last_generated": int((datetime.now() - timedelta(days=random.randint(0, 7))).timestamp())
  ```
- **src\gateway\web\risk_reporting_service.py:271** - 随机生成数据
  ```python
  "template_name": random.choice(["每日风险报告", "周度风险报告", "月度风险报告"]),
  ```
- **src\gateway\web\risk_reporting_service.py:272** - 随机生成数据
  ```python
  "status": random.choice(["generating", "completed", "failed", "pending"]),
  ```
- **src\gateway\web\risk_reporting_service.py:273** - 随机生成数据
  ```python
  "progress": random.randint(0, 100),
  ```
- **src\gateway\web\risk_reporting_service.py:274** - 随机生成数据
  ```python
  "start_time": int((datetime.now() - timedelta(minutes=random.randint(0, 60))).timestamp())
  ```
- **src\gateway\web\risk_reporting_service.py:287** - 随机生成数据
  ```python
  "report_type": random.choice(["daily", "weekly", "monthly"]),
  ```
- **src\gateway\web\risk_reporting_service.py:288** - 随机生成数据
  ```python
  "generated_at": int((datetime.now() - timedelta(days=random.randint(0, 30))).timestamp()),
  ```
- **src\gateway\web\risk_reporting_service.py:289** - 随机生成数据
  ```python
  "size": random.randint(100 * 1024, 5 * 1024 * 1024),  # 100KB to 5MB
  ```
- **src\gateway\web\risk_reporting_service.py:290** - 随机生成数据
  ```python
  "generation_time": random.randint(1, 10)
  ```
- **src\gateway\web\strategy_optimization_service.py:213** - 随机生成数据
  ```python
  'volume': np.random.randint(1000000, 10000000, len(dates))
  ```
- **src\gateway\web\strategy_performance_service.py:173** - 随机生成数据
  ```python
  "type": random.choice(["趋势跟踪", "均值回归", "套利", "高频"]),
  ```
- **src\gateway\web\strategy_performance_service.py:174** - 随机生成数据
  ```python
  "status": random.choice(["active", "inactive"]),
  ```
- **src\gateway\web\strategy_performance_service.py:175** - 随机生成数据
  ```python
  "total_return": random.uniform(-0.1, 0.3),
  ```
- **src\gateway\web\strategy_performance_service.py:176** - 随机生成数据
  ```python
  "sharpe_ratio": random.uniform(0.5, 2.5),
  ```
- **src\gateway\web\strategy_performance_service.py:177** - 随机生成数据
  ```python
  "max_drawdown": random.uniform(0.05, 0.2),
  ```
- **src\gateway\web\strategy_performance_service.py:178** - 随机生成数据
  ```python
  "annual_return": random.uniform(0.05, 0.25),
  ```
- **src\gateway\web\strategy_performance_service.py:179** - 随机生成数据
  ```python
  "win_rate": random.uniform(0.4, 0.7)
  ```
- **src\gateway\web\strategy_performance_service.py:203** - 随机生成数据
  ```python
  daily_return = random.uniform(-0.02, 0.02)
  ```
- **src\gateway\web\trading_signal_service.py:170** - 随机生成数据
  ```python
  "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]),
  ```
- **src\gateway\web\trading_signal_service.py:171** - 随机生成数据
  ```python
  "type": random.choice(["buy", "sell", "hold"]),
  ```
- **src\gateway\web\trading_signal_service.py:172** - 随机生成数据
  ```python
  "strength": random.uniform(0.5, 1.0),
  ```
- **src\gateway\web\trading_signal_service.py:173** - 随机生成数据
  ```python
  "price": random.uniform(100, 200),
  ```
- **src\gateway\web\trading_signal_service.py:174** - 随机生成数据
  ```python
  "status": random.choice(["executed", "pending", "expired", "cancelled"]),
  ```
- **src\gateway\web\trading_signal_service.py:175** - 随机生成数据
  ```python
  "timestamp": int((datetime.now() - timedelta(minutes=random.randint(0, 60))).timestamp()),
  ```
- **src\gateway\web\trading_signal_service.py:176** - 随机生成数据
  ```python
  "accuracy": random.uniform(0.6, 0.9),
  ```
- **src\gateway\web\trading_signal_service.py:177** - 随机生成数据
  ```python
  "latency": random.uniform(1, 10),
  ```
- **src\gateway\web\trading_signal_service.py:178** - 随机生成数据
  ```python
  "quality": random.uniform(0.7, 0.95)
  ```

## TODO注释（需要对接实际系统）

- **src\gateway\web\api_utils.py:17**
  ```python
  # TODO: 实现实际的数据源配置管理器
  ```
- **src\gateway\web\config_manager.py:332**
  ```python
  # TODO: 实现实际的数据源配置管理器
  ```
- **src\gateway\web\model_training_routes.py:46**
  ```python
  # TODO: 创建实际训练任务
  ```
- **src\gateway\web\model_training_routes.py:60**
  ```python
  # TODO: 停止实际任务
  ```
- **src\gateway\web\risk_reporting_routes.py:45**
  ```python
  # TODO: 创建实际模板
  ```
- **src\gateway\web\risk_reporting_routes.py:59**
  ```python
  # TODO: 删除实际模板
  ```
- **src\gateway\web\risk_reporting_routes.py:92**
  ```python
  # TODO: 创建实际任务
  ```
- **src\gateway\web\risk_reporting_routes.py:108**
  ```python
  # TODO: 取消实际任务
  ```
- **src\gateway\web\risk_reporting_routes.py:155**
  ```python
  # TODO: 返回实际报告文件
  ```
- **src\gateway\web\risk_reporting_routes.py:167**
  ```python
  # TODO: 删除实际报告
  ```
