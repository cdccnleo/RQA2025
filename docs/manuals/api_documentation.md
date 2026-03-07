# RQA2025 API文档

## 概述
RQA2025 量化交易分析系统API基于RESTful架构设计...

## 认证
### JWT Token认证
```bash
curl -X POST https://api.rqa2025.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

## 核心API

### 策略管理API

#### 创建策略
```http
POST /api/v1/strategies
```

**请求参数**:
```json
{
  "name": "MACD策略",
  "description": "基于MACD指标的交易策略",
  "parameters": {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
  }
}
```

**响应**:
```json
{
  "id": "strategy_001",
  "name": "MACD策略",
  "status": "created",
  "created_at": "2025-01-15T10:00:00Z"
}
```

#### 获取策略列表
```http
GET /api/v1/strategies
```

**查询参数**:
- `page`: 页码 (默认: 1)
- `limit`: 每页数量 (默认: 20)
- `status`: 策略状态

**响应**:
```json
{
  "strategies": [
    {
      "id": "strategy_001",
      "name": "MACD策略",
      "status": "active",
      "performance": {
        "total_return": 15.3,
        "sharpe_ratio": 1.8,
        "max_drawdown": 8.5
      }
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20
}
```

### 投资组合API

#### 创建组合
```http
POST /api/v1/portfolios
```

**请求参数**:
```json
{
  "name": "稳健成长组合",
  "description": "注重长期稳健增长的投资组合",
  "target_return": 12.0,
  "risk_tolerance": "medium",
  "assets": [
    {"symbol": "AAPL", "weight": 0.3},
    {"symbol": "GOOGL", "weight": 0.25},
    {"symbol": "MSFT", "weight": 0.25},
    {"symbol": "AMZN", "weight": 0.2}
  ]
}
```

### 数据API

#### 获取历史数据
```http
GET /api/v1/data/historical
```

**查询参数**:
- `symbol`: 股票代码
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `interval`: 数据间隔 (1m, 5m, 1h, 1d)

**响应**:
```json
{
  "symbol": "AAPL",
  "data": [
    {
      "timestamp": "2025-01-15T09:30:00Z",
      "open": 180.50,
      "high": 182.30,
      "low": 179.80,
      "close": 181.20,
      "volume": 15420000
    }
  ]
}
```

### 风险监控API

#### 获取风险指标
```http
GET /api/v1/risk/metrics
```

**响应**:
```json
{
  "portfolio_id": "portfolio_001",
  "metrics": {
    "var_95": 2.5,
    "sharpe_ratio": 1.8,
    "max_drawdown": 8.5,
    "beta": 0.9,
    "alpha": 0.5
  },
  "alerts": [
    {
      "type": "high_volatility",
      "severity": "warning",
      "message": "组合波动率超过阈值"
    }
  ]
}
```

## 错误处理
### 常见错误码

#### 400 Bad Request
```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "参数格式错误",
    "details": {
      "field": "start_date",
      "expected": "YYYY-MM-DD",
      "received": "2025/01/15"
    }
  }
}
```

#### 401 Unauthorized
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "认证失败",
    "details": "请检查API密钥是否正确"
  }
}
```

#### 404 Not Found
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "资源不存在",
    "details": "策略ID 'strategy_999' 不存在"
  }
}
```

#### 500 Internal Server Error
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "服务器内部错误",
    "details": "请联系技术支持",
    "trace_id": "abc-123-def"
  }
}
```

## 速率限制
- 普通用户: 1000 requests/hour
- 高级用户: 10000 requests/hour
- 企业用户: 100000 requests/hour

## SDK和工具
### Python SDK
```python
from rqa2025 import RQA2025Client

client = RQA2025Client(api_key="your_api_key")
strategies = client.get_strategies()
```

### JavaScript SDK
```javascript
import { RQA2025Client } from 'rqa2025-sdk';

const client = new RQA2025Client({ apiKey: 'your_api_key' });
const strategies = await client.getStrategies();
```

### 命令行工具
```bash
# 安装CLI工具
pip install rqa2025-cli

# 使用CLI
rqa2025 login --api-key your_api_key
rqa2025 strategies list
rqa2025 data download --symbol AAPL --start 2025-01-01
```

## 最佳实践
### 错误处理
```python
try:
    strategy = client.create_strategy(strategy_data)
except RQA2025Error as e:
    if e.code == 'INVALID_PARAMETER':
        # 处理参数错误
        print(f"参数错误: {e.details}")
    elif e.code == 'UNAUTHORIZED':
        # 处理认证错误
        print("请检查API密钥")
    else:
        # 处理其他错误
        print(f"未知错误: {e.message}")
```

### 分页处理
```python
all_strategies = []
page = 1
while True:
    response = client.get_strategies(page=page, limit=100)
    all_strategies.extend(response['strategies'])
    if len(response['strategies']) < 100:
        break
    page += 1
```

### 重试机制
```python
import time
import random

def retry_request(func, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            return func()
        except RQA2025Error as e:
            if e.code in ['INTERNAL_ERROR', 'SERVICE_UNAVAILABLE']:
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
            raise
```

## 更新日志
### v1.0.0 (2025-01-15)
- 初始版本发布
- 支持策略管理API
- 支持投资组合API
- 支持数据API
- 支持风险监控API

### v1.1.0 (计划)
- 添加实时数据流API
- 支持批量操作
- 增强错误处理
- 添加更多示例

## 联系我们
- 技术支持: api-support@rqa2025.com
- 开发者论坛: https://dev.rqa2025.com
- GitHub: https://github.com/rqa2025/api-sdk

---
*版本：1.0.0 | 更新日期：2025-01-15 | 格式：OpenAPI 3.0*
