# 策略服务层API参考文档

## 📋 文档概述

本文档提供策略服务层所有API的完整参考，包括接口定义、方法签名、参数说明和使用示例。

**版本**: v1.0.0
**更新时间**: 2025年01月27日
**适用范围**: 策略服务层所有API接口

## 📚 目录

- [核心接口](#核心接口)
  - [IStrategyService](#istrategyservice)
  - [IStrategy](#istrategy)
  - [IStrategyFactory](#istrategyfactory)
  - [IStrategyPersistence](#istrategypersistence)
- [数据结构](#数据结构)
  - [StrategyConfig](#strategyconfig)
  - [StrategySignal](#strategysignal)
  - [StrategyResult](#strategyresult)
- [枚举类型](#枚举类型)
  - [StrategyType](#strategytype)
  - [StrategyStatus](#strategystatus)
- [实现类](#实现类)
  - [BaseStrategy](#basestrategy)
  - [StrategyFactory](#strategyfactory)
  - [StrategyPersistence](#strategypersistence)
- [异常处理](#异常处理)

---

## 🔌 核心接口

### IStrategyService

策略服务的主要接口，提供策略的创建、执行和管理功能。

#### 方法定义

##### `create_strategy(config: StrategyConfig) -> str`

创建新的策略实例。

**参数**:
- `config` (StrategyConfig): 策略配置对象

**返回值**:
- `str`: 创建成功的策略ID

**异常**:
- `ValueError`: 配置无效
- `RuntimeError`: 创建失败

**示例**:
```python
from src.strategy.interfaces.strategy_interfaces import IStrategyService, StrategyConfig, StrategyType

service: IStrategyService = get_strategy_service()
config = StrategyConfig(
    strategy_id="momentum_001",
    strategy_name="Momentum Strategy",
    strategy_type=StrategyType.MOMENTUM,
    parameters={"lookback_period": 20}
)

strategy_id = service.create_strategy(config)
```

##### `execute_strategy(strategy_id: str, market_data: Dict[str, Any]) -> StrategyResult`

执行策略并生成交易信号。

**参数**:
- `strategy_id` (str): 策略ID
- `market_data` (Dict[str, Any]): 市场数据字典

**返回值**:
- `StrategyResult`: 策略执行结果

**异常**:
- `ValueError`: 策略ID不存在或数据无效
- `RuntimeError`: 执行失败

**示例**:
```python
market_data = {
    "AAPL": [
        {"close": 150.0, "volume": 1000000, "timestamp": "2023-01-01"},
        {"close": 152.0, "volume": 1100000, "timestamp": "2023-01-02"}
    ]
}

result = service.execute_strategy("momentum_001", market_data)
for signal in result.signals:
    print(f"Signal: {signal.signal_type} {signal.symbol}")
```

##### `get_strategy_performance(strategy_id: str) -> Dict[str, Any]`

获取策略的性能指标。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `Dict[str, Any]`: 性能指标字典

**异常**:
- `ValueError`: 策略ID不存在

**示例**:
```python
performance = service.get_strategy_performance("momentum_001")
print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}")
print(f"Win Rate: {performance.get('win_rate', 'N/A')}")
```

##### `update_strategy_config(strategy_id: str, config: Dict[str, Any]) -> bool`

更新策略配置。

**参数**:
- `strategy_id` (str): 策略ID
- `config` (Dict[str, Any]): 新的配置参数

**返回值**:
- `bool`: 更新是否成功

**异常**:
- `ValueError`: 策略ID不存在或配置无效

##### `start_strategy(strategy_id: str) -> bool`

启动策略。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `bool`: 启动是否成功

##### `stop_strategy(strategy_id: str) -> bool`

停止策略。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `bool`: 停止是否成功

##### `get_strategy_status(strategy_id: str) -> StrategyStatus`

获取策略状态。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `StrategyStatus`: 策略状态枚举

---

### IStrategy

策略实例的接口定义，所有具体策略实现都需要继承此接口。

#### 方法定义

##### `initialize(config: StrategyConfig) -> bool`

初始化策略。

**参数**:
- `config` (StrategyConfig): 策略配置

**返回值**:
- `bool`: 初始化是否成功

##### `generate_signals(market_data: Dict[str, Any]) -> List[StrategySignal]`

生成交易信号。

**参数**:
- `market_data` (Dict[str, Any]): 市场数据

**返回值**:
- `List[StrategySignal]`: 交易信号列表

##### `update_parameters(parameters: Dict[str, Any]) -> bool`

更新策略参数。

**参数**:
- `parameters` (Dict[str, Any]): 新参数

**返回值**:
- `bool`: 更新是否成功

##### `get_performance_metrics() -> Dict[str, Any]`

获取性能指标。

**返回值**:
- `Dict[str, Any]`: 性能指标字典

---

### IStrategyFactory

策略工厂接口，负责创建和管理策略实例。

#### 方法定义

##### `create_strategy(config: StrategyConfig) -> IStrategy`

创建策略实例。

**参数**:
- `config` (StrategyConfig): 策略配置

**返回值**:
- `IStrategy`: 策略实例

##### `get_supported_types() -> List[StrategyType]`

获取支持的策略类型。

**返回值**:
- `List[StrategyType]`: 支持的策略类型列表

---

### IStrategyPersistence

策略持久化接口，负责策略数据的存储和管理。

#### 方法定义

##### `save_strategy(strategy_id: str, data: Dict[str, Any]) -> bool`

保存策略数据。

**参数**:
- `strategy_id` (str): 策略ID
- `data` (Dict[str, Any]): 要保存的数据

**返回值**:
- `bool`: 保存是否成功

##### `load_strategy(strategy_id: str) -> Optional[Dict[str, Any]]`

加载策略数据。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `Optional[Dict[str, Any]]`: 加载的数据，如果不存在返回None

##### `delete_strategy(strategy_id: str) -> bool`

删除策略数据。

**参数**:
- `strategy_id` (str): 策略ID

**返回值**:
- `bool`: 删除是否成功

---

## 📊 数据结构

### StrategyConfig

策略配置数据类。

#### 属性

- `strategy_id: str` - 策略唯一标识（必需）
- `strategy_name: str` - 策略名称（必需）
- `strategy_type: StrategyType` - 策略类型（必需）
- `parameters: Dict[str, Any]` - 策略参数（默认空字典）
- `risk_limits: Dict[str, Any]` - 风险限制（默认空字典）
- `market_data_sources: List[str]` - 数据源列表（默认空列表）
- `created_at: datetime` - 创建时间（自动生成）
- `updated_at: datetime` - 更新时间（自动生成）

#### 示例

```python
from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType

config = StrategyConfig(
    strategy_id="momentum_001",
    strategy_name="Momentum Strategy",
    strategy_type=StrategyType.MOMENTUM,
    parameters={
        "lookback_period": 20,
        "momentum_threshold": 0.05
    },
    risk_limits={
        "max_drawdown": 0.1,
        "max_position": 1000
    },
    market_data_sources=["bloomberg", "yahoo"]
)
```

### StrategySignal

策略信号数据类。

#### 属性

- `signal_id: str` - 信号唯一标识
- `strategy_id: str` - 策略ID
- `signal_type: str` - 信号类型 ('BUY', 'SELL', 'HOLD')
- `symbol: str` - 交易标的
- `price: float` - 交易价格
- `quantity: int` - 交易数量
- `timestamp: datetime` - 时间戳
- `confidence: float` - 置信度 (0-1)
- `metadata: Dict[str, Any]` - 元数据

#### 示例

```python
from src.strategy.interfaces.strategy_interfaces import StrategySignal
from datetime import datetime

signal = StrategySignal(
    signal_id="sig_001",
    strategy_id="momentum_001",
    signal_type="BUY",
    symbol="AAPL",
    price=150.0,
    quantity=100,
    timestamp=datetime.now(),
    confidence=0.85,
    metadata={"reason": "momentum_uptrend"}
)
```

### StrategyResult

策略执行结果数据类。

#### 属性

- `result_id: str` - 结果唯一标识
- `strategy_id: str` - 策略ID
- `signals: List[StrategySignal]` - 交易信号列表
- `performance_metrics: Dict[str, Any]` - 性能指标
- `execution_time: float` - 执行时间（秒）
- `timestamp: datetime` - 时间戳
- `status: str` - 执行状态

#### 示例

```python
from src.strategy.interfaces.strategy_interfaces import StrategyResult, StrategySignal

signals = [
    StrategySignal(signal_id="sig_001", strategy_id="strat_001",
                  signal_type="BUY", symbol="AAPL", price=150.0, quantity=100,
                  timestamp=datetime.now(), confidence=0.8)
]

result = StrategyResult(
    result_id="result_001",
    strategy_id="strat_001",
    signals=signals,
    performance_metrics={"sharpe_ratio": 1.5, "win_rate": 0.65},
    execution_time=0.05,
    timestamp=datetime.now(),
    status="success"
)
```

---

## 🔢 枚举类型

### StrategyType

策略类型枚举。

#### 值

- `MOMENTUM` - 动量策略
- `MEAN_REVERSION` - 均值回归策略
- `ARBITRAGE` - 套利策略
- `MACHINE_LEARNING` - 机器学习策略
- `REINFORCEMENT_LEARNING` - 强化学习策略
- `TREND_FOLLOWING` - 趋势跟随策略
- `CUSTOM` - 自定义策略

### StrategyStatus

策略状态枚举。

#### 值

- `CREATED` - 已创建
- `INITIALIZING` - 初始化中
- `RUNNING` - 运行中
- `PAUSED` - 已暂停
- `STOPPED` - 已停止
- `ERROR` - 错误状态

---

## 🏗️ 实现类

### BaseStrategy

策略基础实现类，所有具体策略都应该继承此类。

#### 继承关系
```python
class BaseStrategy(IStrategy):
    # 实现IStrategy的所有抽象方法
    # 提供通用的策略功能
    pass
```

#### 主要方法

##### `__init__(self, config: StrategyConfig)`
初始化基础策略。

##### `initialize(self, config: StrategyConfig) -> bool`
实现策略初始化。

##### `generate_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]`
实现信号生成（调用`_generate_signals_impl`）。

##### `update_parameters(self, parameters: Dict[str, Any]) -> bool`
更新策略参数。

##### `get_performance_metrics(self) -> Dict[str, Any]`
获取性能指标。

##### `_generate_signals_impl(self, market_data: Dict[str, Any]) -> List[StrategySignal]` (抽象方法)
具体策略实现信号生成的逻辑。

##### `_validate_market_data(self, market_data: Dict[str, Any]) -> None` (抽象方法)
验证市场数据的格式。

### StrategyFactory

策略工厂实现类。

#### 主要方法

##### `__init__(self)`
初始化工厂并注册内置策略。

##### `create_strategy(self, config: StrategyConfig) -> IStrategy`
创建策略实例。

##### `get_supported_types(self) -> List[StrategyType]`
获取支持的策略类型。

##### `register_strategy(self, strategy_type: StrategyType, strategy_class: Type[IStrategy])`
注册自定义策略。

### StrategyPersistence

策略持久化实现类。

#### 主要方法

##### `save_strategy(self, strategy_id: str, data: Dict[str, Any]) -> bool`
保存策略数据。

##### `load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]`
加载策略数据。

##### `save_strategy_config(self, config: StrategyConfig) -> bool`
保存策略配置。

##### `load_strategy_config(self, strategy_id: str) -> Optional[StrategyConfig]`
加载策略配置。

##### `save_strategy_result(self, result: StrategyResult) -> bool`
保存策略执行结果。

##### `get_strategy_history(self, strategy_id: str) -> List[StrategyResult]`
获取策略执行历史。

---

## 🚨 异常处理

### 标准异常

#### `StrategyError`
策略相关的基础异常。

```python
class StrategyError(Exception):
    """策略基础异常"""
    pass
```

#### `StrategyConfigError`
策略配置相关的异常。

```python
class StrategyConfigError(StrategyError):
    """策略配置异常"""
    pass
```

#### `StrategyExecutionError`
策略执行相关的异常。

```python
class StrategyExecutionError(StrategyError):
    """策略执行异常"""
    pass
```

#### `StrategyPersistenceError`
策略持久化相关的异常。

```python
class StrategyPersistenceError(StrategyError):
    """策略持久化异常"""
    pass
```

### 异常处理示例

```python
from src.strategy.interfaces.strategy_interfaces import StrategyError, StrategyConfigError

try:
    strategy = factory.create_strategy(config)
    result = strategy.generate_signals(market_data)
except StrategyConfigError as e:
    print(f"配置错误: {e}")
    # 处理配置问题
except StrategyExecutionError as e:
    print(f"执行错误: {e}")
    # 处理执行问题
except StrategyError as e:
    print(f"策略错误: {e}")
    # 处理其他策略问题
except Exception as e:
    print(f"未知错误: {e}")
    # 处理未知错误
```

---

## 🔧 工厂函数

### `get_strategy_factory() -> StrategyFactory`

获取全局策略工厂实例。

**返回值**:
- `StrategyFactory`: 策略工厂实例

**示例**:
```python
from src.strategy.strategies.factory import get_strategy_factory

factory = get_strategy_factory()
strategy = factory.create_strategy(config)
```

### `get_strategy_service() -> IStrategyService`

获取全局策略服务实例。

**返回值**:
- `IStrategyService`: 策略服务实例

**示例**:
```python
from src.strategy.core.strategy_service import get_strategy_service

service = get_strategy_service()
result = service.execute_strategy(strategy_id, market_data)
```

---

## 📈 性能指标

### 响应时间
- **信号生成**: <50ms (P95)
- **策略初始化**: <10ms
- **参数更新**: <5ms

### 并发能力
- **同时运行策略**: >100个
- **并发信号生成**: >2000 TPS

### 资源使用
- **内存使用**: <50MB (单策略)
- **CPU使用**: <5% (单策略)

---

## 🔗 相关链接

- [策略服务层架构设计](../architecture/strategy_layer_architecture_design.md)
- [使用指南](STRATEGY_USAGE_GUIDE.md)
- [测试文档](../../tests/unit/strategy/)

---

**策略服务层API参考文档** - 完整、准确、实用的API指南！
