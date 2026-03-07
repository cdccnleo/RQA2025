# 💡 RQA2025核心功能教程

## 🎯 概述

本教程将详细介绍RQA2025量化交易系统的核心功能，包括策略配置、回测执行、实时交易、风险管理和性能监控等关键模块。

## 📊 1. 策略配置

### 1.1 内置策略

RQA2025提供了多种内置交易策略：

#### 移动平均策略
```python
from rqa2025.strategies import MovingAverageStrategy

strategy = MovingAverageStrategy(
    symbol="AAPL",
    short_window=5,      # 短期均线周期
    long_window=20,      # 长期均线周期
    position_size=100,   # 每次交易股数
    stop_loss=0.02,      # 止损比例2%
    take_profit=0.05     # 止盈比例5%
)

# 注册策略
system.register_strategy(strategy, name="ma_strategy")
```

#### 动量策略
```python
from rqa2025.strategies import MomentumStrategy

strategy = MomentumStrategy(
    symbol="TSLA",
    lookback_period=20,   # 观察周期
    momentum_threshold=0.05,  # 动量阈值
    position_size=50,
    max_holding_days=5    # 最大持仓天数
)

system.register_strategy(strategy, name="momentum_strategy")
```

#### 均值回归策略
```python
from rqa2025.strategies import MeanReversionStrategy

strategy = MeanReversionStrategy(
    symbol="GOOGL",
    window=20,            # 计算窗口
    z_score_threshold=2.0, # Z分数阈值
    position_size=75,
    max_position_size=0.1  # 最大仓位比例
)

system.register_strategy(strategy, name="mr_strategy")
```

### 1.2 自定义策略

创建自定义策略：

```python
from rqa2025.strategies import BaseStrategy
from rqa2025.core import MarketData, Signal

class CustomStrategy(BaseStrategy):
    def __init__(self, symbol, param1, param2):
        super().__init__(symbol)
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, market_data: MarketData) -> list[Signal]:
        signals = []

        # 自定义信号生成逻辑
        if self.custom_condition(market_data):
            signal = Signal(
                symbol=self.symbol,
                signal_type="BUY",
                quantity=self.calculate_position_size(market_data),
                price=market_data.close,
                timestamp=market_data.timestamp
            )
            signals.append(signal)

        return signals

    def custom_condition(self, data: MarketData) -> bool:
        # 实现自定义交易条件
        return data.close > data.close.rolling(20).mean() * 1.05

    def calculate_position_size(self, data: MarketData) -> int:
        # 实现自定义仓位计算
        risk_amount = self.portfolio_value * 0.01  # 1%风险
        stop_loss_price = data.close * 0.98  # 2%止损
        position_size = risk_amount / (data.close - stop_loss_price)
        return int(position_size)

# 注册自定义策略
custom_strategy = CustomStrategy("NVDA", param1=10, param2=0.8)
system.register_strategy(custom_strategy, name="custom_strategy")
```

### 1.3 策略组合

创建多策略组合：

```python
from rqa2025.portfolio import PortfolioManager

# 创建投资组合管理器
portfolio = PortfolioManager(
    total_capital=100000.0,
    max_strategies=5,
    risk_parity=True
)

# 添加策略到组合
portfolio.add_strategy("ma_strategy", weight=0.3)
portfolio.add_strategy("momentum_strategy", weight=0.3)
portfolio.add_strategy("mr_strategy", weight=0.2)
portfolio.add_strategy("custom_strategy", weight=0.2)

# 设置组合风险管理
portfolio.set_risk_limits(
    max_drawdown=0.1,      # 最大回撤10%
    max_correlation=0.7,   # 最大相关性0.7
    rebalance_frequency="daily"
)

# 注册组合
system.register_portfolio(portfolio, name="diversified_portfolio")
```

## 📈 2. 策略回测

### 2.1 基础回测

```python
from rqa2025.backtest import BacktestEngine

# 创建回测引擎
backtest_engine = BacktestEngine()

# 配置回测参数
backtest_config = {
    "strategy_name": "ma_strategy",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_balance": 100000.0,
    "commission": 0.001,      # 0.1%手续费
    "slippage": 0.0005,       # 0.05%滑点
    "benchmark": "SPY"        # 基准指数
}

# 执行回测
result = backtest_engine.run_backtest(backtest_config)

# 输出回测结果
print("=== 回测结果 ===")
print(f"总收益率: {result['total_return']:.2%}")
print(f"年化收益率: {result['annual_return']:.2%}")
print(f"最大回撤: {result['max_drawdown']:.2%}")
print(f"夏普比率: {result['sharpe_ratio']:.2f}")
print(f"胜率: {result['win_rate']:.1%}")
print(f"总交易次数: {result['total_trades']}")
```

### 2.2 组合回测

```python
# 组合回测配置
portfolio_backtest_config = {
    "portfolio_name": "diversified_portfolio",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_balance": 100000.0,
    "rebalance_frequency": "monthly",
    "transaction_costs": {
        "commission": 0.001,
        "slippage": 0.0005,
        "market_impact": 0.001
    }
}

# 执行组合回测
portfolio_result = backtest_engine.run_portfolio_backtest(portfolio_backtest_config)

print("=== 组合回测结果 ===")
print(f"组合收益率: {portfolio_result['portfolio_return']:.2%}")
print(f"基准收益率: {portfolio_result['benchmark_return']:.2%}")
print(f"超额收益: {portfolio_result['excess_return']:.2%}")
print(f"组合波动率: {portfolio_result['portfolio_volatility']:.2%}")
print(f"信息比率: {portfolio_result['information_ratio']:.2f}")
```

### 2.3 回测报告

```python
from rqa2025.reporting import BacktestReport

# 生成详细报告
reporter = BacktestReport()

# 绩效分析报告
performance_report = reporter.generate_performance_report(result)
print("绩效分析报告已生成: performance_report.html")

# 风险分析报告
risk_report = reporter.generate_risk_report(result)
print("风险分析报告已生成: risk_report.html")

# 交易明细报告
trade_report = reporter.generate_trade_report(result)
print("交易明细报告已生成: trade_report.html")

# 月度绩效报告
monthly_report = reporter.generate_monthly_report(result)
print("月度绩效报告已生成: monthly_report.html")
```

## ⚡ 3. 实时交易

### 3.1 交易执行

```python
from rqa2025.execution import ExecutionEngine

# 创建执行引擎
execution_engine = ExecutionEngine(
    account_name="live_account",
    broker="interactive_brokers",  # 或其他券商
    max_slippage=0.002,
    enable_partial_fills=True
)

# 配置交易参数
execution_config = {
    "strategy_name": "ma_strategy",
    "symbol": "AAPL",
    "position_size": 100,
    "order_type": "MARKET",  # 市价单
    "time_in_force": "DAY"   # 当日有效
}

# 执行交易
order_result = execution_engine.execute_order(execution_config)

if order_result['success']:
    print(f"订单执行成功: {order_result['order_id']}")
    print(f"成交价格: ${order_result['execution_price']:.2f}")
    print(f"成交数量: {order_result['executed_quantity']}")
else:
    print(f"订单执行失败: {order_result['error_message']}")
```

### 3.2 订单管理

```python
from rqa2025.order_manager import OrderManager

# 创建订单管理器
order_manager = OrderManager()

# 查询订单状态
orders = order_manager.get_orders(
    strategy_name="ma_strategy",
    status="OPEN"
)

for order in orders:
    print(f"订单ID: {order['order_id']}")
    print(f"状态: {order['status']}")
    print(f"剩余数量: {order['remaining_quantity']}")

# 取消订单
cancel_result = order_manager.cancel_order("ORDER_12345")
if cancel_result['success']:
    print("订单取消成功")
else:
    print(f"订单取消失败: {cancel_result['error']}")

# 修改订单
modify_result = order_manager.modify_order(
    order_id="ORDER_12345",
    new_quantity=50,
    new_price=150.0
)
```

### 3.3 持仓管理

```python
from rqa2025.portfolio import PortfolioManager

# 获取当前持仓
portfolio_manager = PortfolioManager()

positions = portfolio_manager.get_positions()

for position in positions:
    print(f"股票: {position['symbol']}")
    print(f"持仓数量: {position['quantity']}")
    print(f"平均成本: ${position['avg_cost']:.2f}")
    print(f"当前价格: ${position['current_price']:.2f}")
    print(f"盈亏: ${position['unrealized_pnl']:.2f}")
    print(f"盈亏比例: {position['pnl_percentage']:.2%}")
    print("---")

# 计算投资组合指标
portfolio_metrics = portfolio_manager.calculate_metrics()

print("=== 投资组合指标 ===")
print(f"总价值: ${portfolio_metrics['total_value']:.2f}")
print(f"现金余额: ${portfolio_metrics['cash_balance']:.2f}")
print(f"总收益率: {portfolio_metrics['total_return']:.2%}")
print(f"日收益率: {portfolio_metrics['daily_return']:.2%}")
print(f"波动率: {portfolio_metrics['volatility']:.2%}")
```

## 🛡️ 4. 风险管理

### 4.1 风险控制配置

```python
from rqa2025.risk import RiskManager

# 创建风险管理器
risk_manager = RiskManager()

# 配置风险限制
risk_limits = {
    # 仓位风险
    "max_position_size": 0.05,      # 单股票最大5%仓位
    "max_sector_exposure": 0.25,    # 单行业最大25%仓位
    "max_single_stock_loss": 0.03,  # 单股票最大3%亏损

    # 总体风险
    "max_portfolio_drawdown": 0.1,  # 组合最大10%回撤
    "max_daily_loss": 0.02,         # 日最大2%亏损
    "max_monthly_loss": 0.05,       # 月最大5%亏损

    # 交易风险
    "max_trades_per_day": 20,       # 日最大20笔交易
    "max_order_size": 10000,        # 单笔最大1万美元
    "min_order_interval": 60,       # 最小60秒间隔

    # 市场风险
    "max_gap_risk": 0.05,          # 跳空风险5%
    "max_volatility_risk": 0.03,   # 波动率风险3%
    "correlation_limit": 0.8        # 相关性限制80%
}

risk_manager.set_risk_limits(risk_limits)
```

### 4.2 止损止盈设置

```python
from rqa2025.risk import StopLossManager

# 创建止损管理器
stop_loss_manager = StopLossManager()

# 配置止损策略
stop_loss_config = {
    "strategy_name": "ma_strategy",
    "symbol": "AAPL",

    # 固定百分比止损
    "fixed_stop_loss": 0.02,        # 2%止损
    "fixed_take_profit": 0.05,      # 5%止盈

    # 跟踪止损
    "trailing_stop_enabled": True,
    "trailing_stop_percentage": 0.03,  # 3%跟踪止损

    # 波动率止损
    "volatility_stop_enabled": True,
    "volatility_multiplier": 2.0,   # 2倍波动率

    # 时间止损
    "time_stop_enabled": True,
    "max_holding_days": 30,         # 最大持仓30天

    # 自动执行
    "auto_execute_stops": True
}

stop_loss_manager.configure_stops(stop_loss_config)
```

### 4.3 风险监控

```python
from rqa2025.monitoring import RiskMonitor

# 创建风险监控器
risk_monitor = RiskMonitor()

# 设置监控指标
monitor_config = {
    "portfolio_metrics": {
        "var_95": 0.03,        # 95% VaR不超过3%
        "expected_shortfall": 0.05,  # 期望亏损不超过5%
        "beta_limit": 1.2     # Beta不超过1.2
    },

    "position_metrics": {
        "concentration_limit": 0.1,  # 前十大持仓不超过10%
        "liquidity_ratio": 0.8       # 流动性比率80%以上
    },

    "alerts": {
        "email": "risk@company.com",
        "sms": "+1234567890",
        "slack": "#risk-alerts"
    }
}

risk_monitor.configure_monitoring(monitor_config)

# 实时风险评估
risk_assessment = risk_monitor.assess_portfolio_risk()

print("=== 风险评估结果 ===")
print(f"VaR (95%): {risk_assessment['var_95']:.2%}")
print(f"压力测试结果: {'通过' if risk_assessment['stress_test_passed'] else '失败'}")
print(f"流动性风险: {'正常' if risk_assessment['liquidity_ok'] else '警告'}")

# 生成风险报告
risk_report = risk_monitor.generate_risk_report()
print("风险报告已生成: risk_report.pdf")
```

## 📊 5. 性能监控

### 5.1 系统监控

```python
from rqa2025.monitoring import SystemMonitor

# 创建系统监控器
system_monitor = SystemMonitor()

# 配置监控指标
system_config = {
    "performance_metrics": {
        "cpu_threshold": 80,      # CPU使用率阈值
        "memory_threshold": 85,   # 内存使用率阈值
        "disk_threshold": 90,     # 磁盘使用率阈值
        "network_threshold": 100  # 网络使用率阈值(Mbps)
    },

    "application_metrics": {
        "response_time_threshold": 100,  # 响应时间阈值(ms)
        "error_rate_threshold": 0.01,     # 错误率阈值1%
        "throughput_threshold": 1000      # 吞吐量阈值
    },

    "alert_channels": {
        "email": "admin@company.com",
        "webhook": "https://hooks.slack.com/...",
        "sms": "+1234567890"
    }
}

system_monitor.configure_monitoring(system_config)

# 获取系统状态
system_status = system_monitor.get_system_status()

print("=== 系统状态 ===")
print(f"CPU使用率: {system_status['cpu_percent']}%")
print(f"内存使用率: {system_status['memory_percent']}%")
print(f"磁盘使用率: {system_status['disk_percent']}%")
print(f"网络连接: {system_status['network_connections']}")
```

### 5.2 策略监控

```python
from rqa2025.monitoring import StrategyMonitor

# 创建策略监控器
strategy_monitor = StrategyMonitor()

# 配置策略监控
strategy_config = {
    "strategy_name": "ma_strategy",

    "performance_metrics": {
        "min_win_rate": 0.55,     # 最低胜率55%
        "max_drawdown": 0.08,     # 最大回撤8%
        "min_sharpe_ratio": 1.5   # 最低夏普比率1.5
    },

    "alert_rules": {
        "performance_decline": True,   # 性能下降告警
        "risk_increase": True,         # 风险增加告警
        "anomaly_detection": True      # 异常检测告警
    }
}

strategy_monitor.configure_strategy_monitoring(strategy_config)

# 监控策略性能
strategy_metrics = strategy_monitor.get_strategy_metrics("ma_strategy")

print("=== 策略性能指标 ===")
print(f"胜率: {strategy_metrics['win_rate']:.1%}")
print(f"夏普比率: {strategy_metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {strategy_metrics['max_drawdown']:.2%}")
print(f"年化收益率: {strategy_metrics['annual_return']:.2%}")
```

### 5.3 实时告警

```python
from rqa2025.alerts import AlertManager

# 创建告警管理器
alert_manager = AlertManager()

# 配置告警规则
alert_config = {
    "rules": {
        "high_cpu_usage": {
            "condition": "cpu_percent > 90",
            "severity": "critical",
            "channels": ["email", "sms"],
            "cooldown": 300  # 5分钟冷却
        },

        "portfolio_loss": {
            "condition": "daily_loss > 0.03",
            "severity": "high",
            "channels": ["email", "slack"],
            "cooldown": 3600  # 1小时冷却
        },

        "strategy_failure": {
            "condition": "error_rate > 0.05",
            "severity": "medium",
            "channels": ["email"],
            "cooldown": 1800  # 30分钟冷却
        }
    },

    "escalation": {
        "enabled": True,
        "levels": [
            {"delay": 0, "channels": ["email"]},
            {"delay": 600, "channels": ["sms"]},
            {"delay": 1800, "channels": ["call"]}
        ]
    }
}

alert_manager.configure_alerts(alert_config)

# 查看活跃告警
active_alerts = alert_manager.get_active_alerts()

print("=== 活跃告警 ===")
for alert in active_alerts:
    print(f"[{alert['severity']}] {alert['message']} ({alert['timestamp']})")
```

## 🎯 总结

通过本教程，您已经掌握了RQA2025的核心功能：

- ✅ **策略配置**: 内置策略 + 自定义策略 + 策略组合
- ✅ **回测执行**: 基础回测 + 组合回测 + 详细报告
- ✅ **实时交易**: 订单执行 + 订单管理 + 持仓管理
- ✅ **风险管理**: 风险控制 + 止损止盈 + 风险监控
- ✅ **性能监控**: 系统监控 + 策略监控 + 实时告警

现在您可以开始构建自己的量化交易策略了！

---

**💡 提示**: 建议从简单策略开始，逐步增加复杂度。始终在模拟环境中测试策略，然后再应用于实盘交易。

