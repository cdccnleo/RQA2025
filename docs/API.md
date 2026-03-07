# RQA2025 API 文档

## 概述

RQA2025 提供完整的 RESTful API 接口，支持策略管理、回测分析、参数优化、系统监控等功能。

## 认证

API 使用 JWT (JSON Web Token) 进行认证。获取令牌后，在请求头中包含：

```
Authorization: Bearer <your_jwt_token>
```

## 基础 URL

```
http://localhost:8000/api
```

## 策略管理 API

### 创建策略

```http
POST /strategies
```

**请求体:**
```json
{
  "strategy_name": "动量策略",
  "strategy_type": "momentum",
  "parameters": {
    "lookback_period": 20,
    "momentum_threshold": 0.05
  },
  "risk_limits": {
    "max_position": 1000
  }
}
```

**响应:**
```json
{
  "success": true,
  "strategy_id": "strategy_20231201_120000_abc123",
  "message": "策略创建成功"
}
```

### 获取策略列表

```http
GET /strategies
```

**响应:**
```json
{
  "strategies": [
    {
      "strategy_id": "strategy_001",
      "strategy_name": "动量策略",
      "strategy_type": "momentum",
      "status": "active",
      "created_at": "2023-12-01T12:00:00Z"
    }
  ],
  "count": 1
}
```

## 回测分析 API

### 创建回测

```http
POST /backtests
```

**请求体:**
```json
{
  "strategy_id": "strategy_001",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 100000.0,
  "commission": 0.0003
}
```

### 获取回测结果

```http
GET /backtests/{backtest_id}
```

## 参数优化 API

### 创建优化任务

```http
POST /optimizations
```

**请求体:**
```json
{
  "strategy_id": "strategy_001",
  "algorithm": "bayesian_optimization",
  "parameter_ranges": {
    "lookback_period": [10, 20, 30, 50],
    "momentum_threshold": [0.01, 0.05, 0.1]
  },
  "max_iterations": 50
}
```

## 监控 API

### 获取监控指标

```http
GET /monitoring/metrics
```

### 获取告警信息

```http
GET /monitoring/alerts
```

## 认证 API

### 用户登录

```http
POST /auth/login
```

**请求体:**
```json
{
  "username": "testuser",
  "password": "password123"
}
```

### 用户注册

```http
POST /auth/register
```

**请求体:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "password123",
  "full_name": "新用户"
}
```

## 调试 API

### 调试会话管理

```http
POST /debug/sessions
GET /debug/sessions/{session_id}
DELETE /debug/sessions/{session_id}
```

### 性能分析

```http
POST /debug/performance/profile
GET /debug/performance/results/{session_id}
```

## 错误处理

API 使用标准的 HTTP 状态码：

- `200`: 成功
- `400`: 请求错误
- `401`: 未认证
- `403`: 权限不足
- `404`: 资源不存在
- `500`: 服务器错误

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

## 分页

支持分页的 API 端点接受以下查询参数：

- `page`: 页码 (默认: 1)
- `per_page`: 每页数量 (默认: 20, 最大: 100)

分页响应格式：

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "total_pages": 8
  }
}
```

## 速率限制

API 实施速率限制以确保公平使用：

- 认证用户: 1000 次/小时
- 未认证用户: 100 次/小时

## SDK 和示例

### Python SDK

```python
from rqa2025 import RQA2025Client

client = RQA2025Client(base_url="http://localhost:8000")
client.login("username", "password")

# 创建策略
strategy = client.create_strategy({
    "strategy_name": "示例策略",
    "strategy_type": "momentum"
})

# 执行回测
backtest = client.run_backtest({
    "strategy_id": strategy["strategy_id"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
})
```

## 更新日志

### v1.0.0 (2024-01-01)
- 🎉 初始版本发布
- ✨ 完整的策略开发工作流
- 🚀 高性能回测引擎
- 🎯 智能参数优化
- 📊 实时监控和告警
- 🖥️ 现代化的Web界面

---

*API 版本: v1.0.0*
*最后更新: {datetime.now().strftime('%Y-%m-%d')}*
