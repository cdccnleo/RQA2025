# API使用指南

## 概述

RQA2025数据层提供了REST API和WebSocket API两种接口，支持数据的查询、监控和管理。

## REST API

### 基础信息

- **基础URL**: `http://localhost:8000/api/v1`
- **认证**: 暂不需要
- **格式**: JSON

### 主要端点

#### 1. 健康检查

```bash
GET /api/v1/data/health
```

响应示例:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

#### 2. 数据源列表

```bash
GET /api/v1/data/sources
```

响应示例:
```json
{
  "sources": [
    {
      "name": "crypto",
      "description": "加密货币数据",
      "status": "active"
    },
    {
      "name": "macro",
      "description": "宏观经济数据", 
      "status": "active"
    }
  ]
}
```

#### 3. 加载数据

```bash
POST /api/v1/data/load
Content-Type: application/json

{
  "source": "crypto",
  "symbol": "BTC",
  "timeframe": "1d"
}
```

#### 4. 性能指标

```bash
GET /api/v1/data/performance
```

#### 5. 数据质量检查

```bash
GET /api/v1/data/quality
```

## WebSocket API

### 连接

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market_data');
```

### 订阅频道

- `market_data`: 实时市场数据
- `quality_monitor`: 数据质量监控
- `performance_monitor`: 性能监控
- `alerts`: 告警信息

### 消息格式

```json
{
  "channel": "market_data",
  "data": {
    "symbol": "BTC",
    "price": 50000,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## 客户端SDK

### Python SDK

```python
from src.engine.web.client_sdk import RQA2025DataClient

# 创建客户端
client = RQA2025DataClient("http://localhost:8000")

# 获取数据源列表
sources = await client.list_data_sources()

# 加载数据
data = await client.load_data("crypto", "BTC")

# 获取性能指标
metrics = await client.get_performance_metrics()
```

## 错误处理

### HTTP状态码

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

### 错误响应格式

```json
{
  "error": "错误描述",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 最佳实践

1. **缓存策略**: 合理使用缓存减少请求
2. **错误处理**: 实现重试机制和降级策略
3. **监控**: 监控API调用性能和错误率
4. **限流**: 遵守API调用频率限制
