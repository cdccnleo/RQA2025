# 交易层（trading）架构设计说明

## 1. 模块定位
trading模块为RQA2025系统所有主流程提供高性能、合规、安全的交易执行、风控、策略、订单、撮合、账户、网关、监控等能力，是主流程的交易决策与执行核心。

## 2. 主要子系统
- **交易执行与撮合**：ExecutionEngine、OrderManager、OrderRouter 负责订单执行、撮合、路由、算法管理（TWAP/VWAP等）。
- **风控与合规**：RiskController、ChinaRiskController、风控规则（T+1、涨跌停、熔断、科创板等）统一入口，保障合规。
- **策略管理与信号生成**：BaseStrategy、EnhancedTradingStrategy、SignalGenerator、策略注册与信号生成，支持A股特有规则。
- **订单与账户管理**：OrderManager、TradingGateway、BaseGateway、账户、持仓、订单簿管理。
- **实时与批量交易**：LiveTrader、LiveTradingManager、TradingEngine 支持实盘、模拟、回测等多模式。
- **监控与回溯**：TradingMonitor、监控交易状态、风控违规、自动触发熔断等。

## 3. 典型用法
### 交易执行
```python
from src.trading.execution.execution_engine import ExecutionEngine
engine = ExecutionEngine(...)
result = engine.execute_order(order)
```

### 风控与合规
```python
from src.trading.risk.china.risk_controller import ChinaRiskController
risk = ChinaRiskController(config)
check = risk.check(order)
```

### 策略与信号
```python
from src.trading.strategies.core import BaseStrategy
strategy = BaseStrategy()
signals = strategy.generate_signals(market_data)
```

### 订单与账户
```python
from src.trading.order_manager import OrderManager
om = OrderManager()
order_id = om.generate_order_id()
```

### 实时交易
```python
from src.trading.live_trader import LiveTrader
trader = LiveTrader(gateway)
await trader.run()
```

## 4. 在主流程中的地位
- 所有主流程的交易决策与执行核心，负责信号转订单、风控校验、订单撮合、账户管理、实时监控等。
- 支持多市场、多策略、多账户、A股特有规则、算法撮合、风控合规，保障主流程的灵活性和安全性。
- 接口抽象与注册机制，便于扩展新市场、新策略、新风控规则、Mock测试等。

## 5. 测试与质量保障
- 已实现高质量pytest单元测试，覆盖交易执行、风控、策略、订单、撮合、账户、网关、监控等主要功能和边界。
- 测试用例见：tests/unit/trading/ 目录下相关文件。 