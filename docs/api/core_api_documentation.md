# RQA2025 核心服务API文档

## 概述

RQA2025量化交易系统的核心服务API接口文档。

## 版本信息

- 版本: 1.0.0
- 更新时间: 2025-08-24
- 状态: 活跃

## 核心组件

### 1. 事件总线 (EventBus)

#### 功能描述
提供模块间解耦的事件驱动架构，支持异步事件处理和优先级管理。

#### 主要方法
- `publish(event_type, data, priority)` - 发布事件
- `subscribe(event_type, handler, priority)` - 订阅事件
- `unsubscribe(event_type, handler)` - 取消订阅

#### 使用示例
```python
from src.core import EventBus

event_bus = EventBus()
event_bus.subscribe('data_ready', handle_data)
event_bus.publish('data_ready', {'symbol': 'AAPL', 'price': 150.0})
```

### 2. 依赖注入容器 (DependencyContainer)

#### 功能描述
管理组件依赖关系，实现服务定位和服务生命周期管理。

#### 主要方法
- `register(name, service, lifecycle)` - 注册服务
- `get(name)` - 获取服务实例
- `check_health(name)` - 检查服务健康状态

### 3. 业务流程编排器 (BusinessProcessOrchestrator)

#### 功能描述
编排和管理业务流程的执行，实现业务逻辑的自动化流转。

#### 主要方法
- `start_trading_cycle(symbols, strategy_config)` - 启动交易周期
- `pause_process(process_id)` - 暂停流程
- `resume_process(process_id)` - 恢复流程

## 基础设施服务

### 配置管理 (UnifiedConfigManager)
- 功能: 统一配置管理、环境变量处理、配置验证
- 接口: `get()`, `set()`, `load()`, `save()`

### 健康检查 (EnhancedHealthChecker)
- 功能: 系统健康监控、诊断报告、告警机制
- 接口: `check_health()`, `get_status()`, `get_metrics()`

### 缓存系统 (CacheManager)
- 功能: 多级缓存管理、缓存策略、数据持久化
- 接口: `get()`, `set()`, `delete()`, `clear()`

## 数据服务

### 数据管理器 (DataManagerSingleton)
- 功能: 数据源适配、实时数据采集、数据验证
- 接口: `get_instance()`, `store_data()`, `get_data()`

### 数据验证器 (DataValidator)
- 功能: 数据质量检查、异常检测、数据修复
- 接口: `validate()`, `check_quality()`, `repair_data()`

## 业务服务

### 交易引擎 (TradingEngine)
- 功能: 完整的量化交易业务流程
- 接口: `process_trading_cycle()`, `execute_trade()`

### 风险管理器 (RiskManager)
- 功能: 风险检查、合规验证、风险监控
- 接口: `check_risk()`, `validate_compliance()`

## 错误处理

### 异常类型
- `CoreException` - 核心服务异常
- `EventBusException` - 事件总线异常
- `ContainerException` - 依赖注入异常
- `ValidationException` - 数据验证异常

## 数据格式

### 标准响应格式
```json
{
  "status": "success|error",
  "data": {...},
  "message": "操作结果描述",
  "timestamp": "ISO时间戳"
}
```

## 安全说明

- 所有API调用都需要身份验证
- 敏感数据使用加密传输
- 定期进行安全审计

---

**文档生成时间**: 2025-08-24T13:13:58.445860
**文档版本**: v1.0
**适用系统**: RQA2025 量化交易平台
