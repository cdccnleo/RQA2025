# RQA2025系统事件格式规范

## 文档信息

| 属性 | 值 |
|------|-----|
| 规范编号 | STD-EVENT-001 |
| 版本 | 1.0.0 |
| 制定日期 | 2026-03-22 |
| 状态 | 已批准 |

---

## 1. 概述

### 1.1 目的

本规范定义RQA2025量化交易系统的事件格式标准，确保各业务层事件的一致性和互操作性。

### 1.2 适用范围

本规范适用于所有通过事件总线发布的事件，包括：
- 数据层事件
- 特征层事件
- 模型层事件
- 策略层事件
- 交易层事件
- 风险控制层事件

### 1.3 术语定义

| 术语 | 定义 |
|------|------|
| 事件 | 系统中发生的离散动作或状态变更 |
| 事件类型 | 事件的分类标识 |
| 事件负载 | 事件携带的数据内容 |
| 时间戳 | 事件发生的时间 |
| 关联ID | 用于追踪事件链路的唯一标识 |

---

## 2. 事件格式标准

### 2.1 标准事件结构

所有事件必须遵循以下JSON结构：

```json
{
  "event_id": "uuid-v4-format",
  "event_type": "EVENT_TYPE_NAME",
  "timestamp": "2026-03-22T10:30:00.000000+08:00",
  "correlation_id": "uuid-v4-format",
  "source": "module_name",
  "version": "1.0",
  "payload": {
    // 事件特定数据
  },
  "metadata": {
    "user_id": "optional",
    "session_id": "optional",
    "trace_id": "optional"
  }
}
```

### 2.2 必填字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| event_id | string | 是 | 事件唯一标识，UUID v4格式 |
| event_type | string | 是 | 事件类型，大写字母+下划线命名 |
| timestamp | string | 是 | ISO 8601格式时间戳，带时区 |
| correlation_id | string | 是 | 关联ID，用于事件链路追踪 |
| source | string | 是 | 事件来源模块名称 |
| version | string | 是 | 事件格式版本号 |
| payload | object | 是 | 事件负载数据 |
| metadata | object | 否 | 元数据信息 |

### 2.3 字段命名规范

1. **事件类型命名**: 使用大写字母+下划线，如 `ORDER_EXECUTED`
2. **字段命名**: 使用小写字母+下划线，如 `order_id`
3. **枚举值**: 使用小写字母+下划线，如 `market_order`
4. **时间戳**: ISO 8601格式，带时区信息

---

## 3. 各层事件规范

### 3.1 交易层事件

#### ORDER_CREATED - 订单创建

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "ORDER_CREATED",
  "timestamp": "2026-03-22T10:30:00.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
  "source": "trading.execution",
  "version": "1.0",
  "payload": {
    "order_id": "ORD-20260322-001",
    "symbol": "000001.SZ",
    "side": "buy",
    "order_type": "limit",
    "quantity": 100,
    "price": 10.50,
    "strategy_id": "STRAT-001",
    "account_id": "ACC-001"
  },
  "metadata": {
    "user_id": "USER-001",
    "session_id": "SESSION-001"
  }
}
```

#### ORDER_EXECUTED - 订单执行

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440002",
  "event_type": "ORDER_EXECUTED",
  "timestamp": "2026-03-22T10:30:01.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
  "source": "trading.execution",
  "version": "1.0",
  "payload": {
    "order_id": "ORD-20260322-001",
    "execution_id": "EXEC-001",
    "filled_quantity": 100,
    "remaining_quantity": 0,
    "avg_price": 10.50,
    "status": "filled",
    "execution_time_ms": 150
  }
}
```

#### ORDER_CANCELLED - 订单取消

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440003",
  "event_type": "ORDER_CANCELLED",
  "timestamp": "2026-03-22T10:30:02.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
  "source": "trading.execution",
  "version": "1.0",
  "payload": {
    "order_id": "ORD-20260322-001",
    "cancelled_quantity": 50,
    "reason": "user_request",
    "cancelled_at": "2026-03-22T10:30:02.000000+08:00"
  }
}
```

### 3.2 风险控制层事件

#### RISK_INTERCEPTED - 风险拦截

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440004",
  "event_type": "RISK_INTERCEPTED",
  "timestamp": "2026-03-22T10:30:00.500000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
  "source": "risk.control",
  "version": "1.0",
  "payload": {
    "risk_type": "position_limit",
    "risk_level": 3,
    "symbol": "000001.SZ",
    "order_id": "ORD-20260322-001",
    "action": "cancel_order",
    "reason": "Position limit exceeded",
    "threshold": 10000,
    "current_value": 15000
  }
}
```

#### RISK_ALERT - 风险告警

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440005",
  "event_type": "RISK_ALERT",
  "timestamp": "2026-03-22T10:30:00.600000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440001",
  "source": "risk.monitoring",
  "version": "1.0",
  "payload": {
    "alert_type": "volatility_spike",
    "severity": "high",
    "symbol": "000001.SZ",
    "message": "Volatility increased by 50%",
    "metrics": {
      "current_volatility": 0.25,
      "baseline_volatility": 0.17
    }
  }
}
```

### 3.3 数据层事件

#### DATA_COLLECTED - 数据采集完成

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440006",
  "event_type": "DATA_COLLECTED",
  "timestamp": "2026-03-22T10:00:00.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440007",
  "source": "data.collection",
  "version": "1.0",
  "payload": {
    "data_source": "tushare",
    "data_type": "daily_price",
    "symbol_count": 3000,
    "record_count": 300000,
    "start_date": "2026-03-21",
    "end_date": "2026-03-22",
    "collection_time_ms": 5000
  }
}
```

#### DATA_QUALITY_CHECKED - 数据质量检查

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440008",
  "event_type": "DATA_QUALITY_CHECKED",
  "timestamp": "2026-03-22T10:05:00.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440007",
  "source": "data.quality",
  "version": "1.0",
  "payload": {
    "check_type": "completeness",
    "passed": true,
    "score": 98.5,
    "issues": [
      {
        "type": "missing_value",
        "symbol": "000001.SZ",
        "field": "volume",
        "severity": "low"
      }
    ]
  }
}
```

### 3.4 策略层事件

#### SIGNAL_GENERATED - 信号生成

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440009",
  "event_type": "SIGNAL_GENERATED",
  "timestamp": "2026-03-22T10:30:00.000000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440010",
  "source": "strategy.signals",
  "version": "1.0",
  "payload": {
    "signal_id": "SIG-001",
    "strategy_id": "STRAT-001",
    "symbol": "000001.SZ",
    "signal_type": "buy",
    "strength": 0.85,
    "confidence": 0.92,
    "features": {
      "momentum": 0.75,
      "mean_reversion": 0.60
    }
  }
}
```

#### STRATEGY_DECISION_READY - 策略决策就绪

```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440011",
  "event_type": "STRATEGY_DECISION_READY",
  "timestamp": "2026-03-22T10:30:00.100000+08:00",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440010",
  "source": "strategy.engine",
  "version": "1.0",
  "payload": {
    "decision_id": "DEC-001",
    "strategy_id": "STRAT-001",
    "decision_type": "enter_position",
    "symbol": "000001.SZ",
    "side": "buy",
    "suggested_quantity": 100,
    "suggested_price": 10.50,
    "rationale": "Strong momentum signal with positive earnings surprise"
  }
}
```

---

## 4. 事件验证

### 4.1 验证规则

1. **event_id**: 必须是有效的UUID v4格式
2. **event_type**: 必须在预定义的事件类型列表中
3. **timestamp**: 必须是有效的ISO 8601格式，带时区
4. **correlation_id**: 必须是有效的UUID v4格式
5. **source**: 必须是有效的模块名称
6. **version**: 必须符合语义化版本规范
7. **payload**: 不能为空对象

### 4.2 验证工具

```python
from src.core.event_bus.validation import EventValidator

validator = EventValidator()

# 验证事件
is_valid, errors = validator.validate(event_data)

if not is_valid:
    print(f"验证失败: {errors}")
```

---

## 5. 向后兼容性

### 5.1 版本策略

1. **主版本号**: 不兼容的API修改
2. **次版本号**: 向下兼容的功能添加
3. **修订号**: 向下兼容的问题修复

### 5.2 兼容性保证

- 次版本和修订版本更新保持向后兼容
- 新增字段为可选字段
- 废弃字段保留至少3个主版本

---

## 6. 实施要求

### 6.1 实施检查清单

- [ ] 所有新事件符合本规范
- [ ] 事件格式验证通过率100%
- [ ] 向后兼容性测试通过
- [ ] 文档更新完成

### 6.2 迁移计划

1. **第一阶段**: 新事件采用新规范
2. **第二阶段**: 逐步迁移现有事件
3. **第三阶段**: 废弃旧格式事件

---

## 7. 附录

### 7.1 事件类型列表

| 事件类型 | 所属层 | 说明 |
|----------|--------|------|
| ORDER_CREATED | 交易层 | 订单创建 |
| ORDER_EXECUTED | 交易层 | 订单执行 |
| ORDER_CANCELLED | 交易层 | 订单取消 |
| ORDER_REJECTED | 交易层 | 订单拒绝 |
| RISK_INTERCEPTED | 风控层 | 风险拦截 |
| RISK_ALERT | 风控层 | 风险告警 |
| DATA_COLLECTED | 数据层 | 数据采集完成 |
| DATA_QUALITY_CHECKED | 数据层 | 数据质量检查 |
| SIGNAL_GENERATED | 策略层 | 信号生成 |
| STRATEGY_DECISION_READY | 策略层 | 策略决策就绪 |

### 7.2 变更记录

| 版本 | 日期 | 变更内容 | 变更人 |
|------|------|----------|--------|
| 1.0.0 | 2026-03-22 | 初始版本 | AI系统集成助手 |

---

*规范结束*
