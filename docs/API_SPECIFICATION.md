# RQA2025 API 规范文档

## 概述

本文档定义了 RQA2025 量化交易系统的 REST API 规范，包括端点命名、请求/响应格式、错误处理等标准。

**版本**: 1.0.0  
**最后更新**: 2026-02-13  
**协议**: REST API over HTTPS  
**基础路径**: `/api/v1`

---

## 命名规范

### 端点命名规则

1. **使用名词复数形式**: `/strategies` 而非 `/strategy`
2. **使用小写字母**: `/market-data` 而非 `/MarketData`
3. **使用连字符分隔**: `/trading-orders` 而非 `/trading_orders`
4. **避免动词**: 使用 HTTP 方法表示操作
   - `GET /strategies` - 获取列表
   - `POST /strategies` - 创建
   - `PUT /strategies/{id}` - 更新
   - `DELETE /strategies/{id}` - 删除

### HTTP 方法使用

| 方法 | 用途 | 幂等性 |
|------|------|--------|
| GET | 获取资源 | 是 |
| POST | 创建资源 | 否 |
| PUT | 完整更新 | 是 |
| PATCH | 部分更新 | 否 |
| DELETE | 删除资源 | 是 |

---

## 端点规范

### 1. 策略管理 (Strategies)

#### 1.1 获取策略列表
```
GET /api/v1/strategies
```

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | int | 否 | 页码，默认 1 |
| size | int | 否 | 每页大小，默认 20 |
| status | string | 否 | 状态过滤 |
| type | string | 否 | 类型过滤 |
| search | string | 否 | 搜索关键词 |

**响应**:
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [...],
    "total": 100,
    "page": 1,
    "size": 20
  }
}
```

#### 1.2 创建策略
```
POST /api/v1/strategies
```

**请求体**:
```json
{
  "name": "策略名称",
  "type": "trend_following",
  "description": "策略描述",
  "config": {...}
}
```

#### 1.3 获取策略详情
```
GET /api/v1/strategies/{id}
```

#### 1.4 更新策略
```
PUT /api/v1/strategies/{id}
```

#### 1.5 删除策略
```
DELETE /api/v1/strategies/{id}
```

#### 1.6 执行策略
```
POST /api/v1/strategies/{id}/execute
```

#### 1.7 停止策略
```
POST /api/v1/strategies/{id}/stop
```

#### 1.8 获取策略回测历史
```
GET /api/v1/strategies/{id}/backtests
```

---

### 2. 特征工程 (Features)

#### 2.1 获取特征任务列表
```
GET /api/v1/features/tasks
```

#### 2.2 创建特征任务
```
POST /api/v1/features/tasks
```

#### 2.3 获取特征任务详情
```
GET /api/v1/features/tasks/{id}
```

#### 2.4 取消特征任务
```
DELETE /api/v1/features/tasks/{id}
```

#### 2.5 获取特征列表
```
GET /api/v1/features
```

#### 2.6 获取特征详情
```
GET /api/v1/features/{id}
```

#### 2.7 获取技术指标列表
```
GET /api/v1/features/indicators
```

---

### 3. 模型训练 (Models)

#### 3.1 获取模型列表
```
GET /api/v1/models
```

#### 3.2 创建训练任务
```
POST /api/v1/models/training-jobs
```

#### 3.3 获取训练任务详情
```
GET /api/v1/models/training-jobs/{id}
```

#### 3.4 取消训练任务
```
DELETE /api/v1/models/training-jobs/{id}
```

#### 3.5 获取训练任务状态
```
GET /api/v1/models/training-jobs/{id}/status
```

#### 3.6 获取训练日志
```
GET /api/v1/models/training-jobs/{id}/logs
```

#### 3.7 部署模型
```
POST /api/v1/models/{id}/deploy
```

#### 3.8 获取可用模型列表
```
GET /api/v1/models/available
```

---

### 4. 策略回测 (Backtests)

#### 4.1 执行回测
```
POST /api/v1/backtests
```

**请求体**:
```json
{
  "strategy_id": "策略ID",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 1000000,
  "commission_rate": 0.0003,
  "slippage": 0.0001
}
```

#### 4.2 获取回测列表
```
GET /api/v1/backtests
```

#### 4.3 获取回测详情
```
GET /api/v1/backtests/{id}
```

#### 4.4 获取回测结果
```
GET /api/v1/backtests/{id}/results
```

#### 4.5 删除回测
```
DELETE /api/v1/backtests/{id}
```

---

### 5. 交易执行 (Trading)

#### 5.1 获取账户信息
```
GET /api/v1/trading/account
```

#### 5.2 获取持仓列表
```
GET /api/v1/trading/positions
```

#### 5.3 获取订单列表
```
GET /api/v1/trading/orders
```

#### 5.4 创建订单
```
POST /api/v1/trading/orders
```

**请求体**:
```json
{
  "symbol": "000001.SZ",
  "side": "buy",
  "type": "limit",
  "quantity": 100,
  "price": 10.5
}
```

#### 5.5 取消订单
```
DELETE /api/v1/trading/orders/{id}
```

#### 5.6 获取成交记录
```
GET /api/v1/trading/trades
```

#### 5.7 获取市场数据
```
GET /api/v1/trading/market-data/{symbol}
```

---

### 6. 数据源 (Data Sources)

#### 6.1 获取数据源列表
```
GET /api/v1/data-sources
```

#### 6.2 创建数据源
```
POST /api/v1/data-sources
```

#### 6.3 获取数据源详情
```
GET /api/v1/data-sources/{id}
```

#### 6.4 更新数据源
```
PUT /api/v1/data-sources/{id}
```

#### 6.5 删除数据源
```
DELETE /api/v1/data-sources/{id}
```

#### 6.6 测试数据源连接
```
POST /api/v1/data-sources/{id}/test
```

#### 6.7 获取历史数据
```
GET /api/v1/data-sources/{id}/historical-data
```

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| symbol | string | 是 | 股票代码 |
| start_date | string | 是 | 开始日期 |
| end_date | string | 是 | 结束日期 |
| frequency | string | 否 | 数据频率，默认 1d |

---

### 7. 风控管理 (Risk)

#### 7.1 获取风控规则列表
```
GET /api/v1/risk/rules
```

#### 7.2 创建风控规则
```
POST /api/v1/risk/rules
```

#### 7.3 更新风控规则
```
PUT /api/v1/risk/rules/{id}
```

#### 7.4 删除风控规则
```
DELETE /api/v1/risk/rules/{id}
```

#### 7.5 获取风控事件
```
GET /api/v1/risk/events
```

#### 7.6 获取风险报告
```
GET /api/v1/risk/reports
```

---

### 8. 系统管理 (System)

#### 8.1 获取系统状态
```
GET /api/v1/system/status
```

#### 8.2 获取系统配置
```
GET /api/v1/system/config
```

#### 8.3 更新系统配置
```
PUT /api/v1/system/config
```

#### 8.4 获取日志
```
GET /api/v1/system/logs
```

#### 8.5 获取性能指标
```
GET /api/v1/system/metrics
```

---

## 响应格式

### 成功响应

```json
{
  "code": 200,
  "message": "success",
  "data": {...},
  "timestamp": "2026-02-13T10:30:00Z",
  "request_id": "req-123456"
}
```

### 错误响应

```json
{
  "code": 400,
  "message": "请求参数错误",
  "error": {
    "type": "validation_error",
    "details": [
      {
        "field": "name",
        "message": "名称不能为空"
      }
    ]
  },
  "timestamp": "2026-02-13T10:30:00Z",
  "request_id": "req-123456"
}
```

### 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 204 | 无内容（删除成功） |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 禁止访问 |
| 404 | 资源不存在 |
| 409 | 资源冲突 |
| 422 | 验证错误 |
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用 |

---

## 分页规范

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| page | int | 1 | 页码，从1开始 |
| size | int | 20 | 每页大小，最大100 |
| sort | string | - | 排序字段，如 `-created_at` 表示降序 |

### 响应格式

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "size": 20,
      "total": 100,
      "total_pages": 5,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

---

## 错误处理

### 错误类型

| 类型 | 说明 |
|------|------|
| validation_error | 参数验证错误 |
| authentication_error | 认证错误 |
| authorization_error | 授权错误 |
| not_found_error | 资源不存在 |
| conflict_error | 资源冲突 |
| rate_limit_error | 请求频率限制 |
| internal_error | 内部服务器错误 |
| service_unavailable_error | 服务不可用 |

### 错误示例

```json
{
  "code": 422,
  "message": "参数验证失败",
  "error": {
    "type": "validation_error",
    "details": [
      {
        "field": "strategy_id",
        "message": "策略ID不能为空",
        "code": "required"
      },
      {
        "field": "start_date",
        "message": "日期格式错误，应为 YYYY-MM-DD",
        "code": "invalid_format"
      }
    ]
  }
}
```

---

## WebSocket 规范

### 连接

```
WS /ws/{channel}
```

### 频道

| 频道 | 说明 |
|------|------|
| /ws/market-data | 实时行情数据 |
| /ws/execution-status | 策略执行状态 |
| /ws/notifications | 系统通知 |
| /ws/alerts | 告警信息 |

### 消息格式

```json
{
  "type": "market_data_update",
  "timestamp": "2026-02-13T10:30:00Z",
  "data": {...}
}
```

---

## 认证

### API Key 认证

```
Authorization: Bearer {api_key}
```

### JWT 认证

```
Authorization: Bearer {jwt_token}
```

---

## 限流

### 限制规则

| 级别 | 限制 | 说明 |
|------|------|------|
| IP | 1000/小时 | 基于IP地址 |
| 用户 | 10000/小时 | 基于用户ID |
| 端点 | 100/分钟 | 特定端点 |

### 响应头

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1644729600
```

---

## 版本控制

API 版本通过 URL 路径指定：

```
/api/v1/strategies
/api/v2/strategies
```

---

## 变更日志

### v1.0.0 (2026-02-13)
- 初始版本发布
- 统一 API 端点命名规范
- 定义标准响应格式
- 完善错误处理规范

---

## 附录

### A. 股票代码格式

- A股: `000001.SZ`, `600000.SH`
- 港股: `00700.HK`
- 美股: `AAPL.US`

### B. 日期时间格式

- 日期: `YYYY-MM-DD`
- 时间: `HH:MM:SS`
- 日期时间: `YYYY-MM-DDTHH:MM:SSZ` (ISO 8601)

### C. 数据频率

| 代码 | 说明 |
|------|------|
| 1m | 1分钟 |
| 5m | 5分钟 |
| 15m | 15分钟 |
| 30m | 30分钟 |
| 1h | 1小时 |
| 1d | 日线 |
| 1w | 周线 |
| 1M | 月线 |
