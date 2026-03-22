# RQA2025 API 参考文档

## 文档信息
- **版本**: v1.0.0
- **更新日期**: 2026-03-29
- **文档状态**: 已发布
- **作者**: 王芳

---

## 目录

1. [概述](#概述)
2. [认证](#认证)
3. [基础URL](#基础url)
4. [错误处理](#错误处理)
5. [API端点](#api端点)
   - [策略管理](#策略管理)
   - [特征管理](#特征管理)
   - [模型管理](#模型管理)
   - [交易执行](#交易执行)
   - [监控指标](#监控指标)
6. [示例代码](#示例代码)
7. [变更日志](#变更日志)

---

## 概述

RQA2025 API提供了一套完整的RESTful接口，用于量化交易策略的开发、测试和部署。API涵盖了策略管理、特征工程、模型管理、交易执行等核心功能。

### 特性

- **RESTful设计**: 遵循RESTful API设计原则
- **JSON格式**: 请求和响应均使用JSON格式
- **认证安全**: 基于JWT的认证机制
- **版本控制**: API版本管理
- **分页支持**: 列表接口支持分页
- **错误处理**: 统一的错误响应格式

---

## 认证

### JWT认证

所有API请求（除登录外）都需要在请求头中携带JWT令牌。

**请求头格式**:
```
Authorization: Bearer <your_jwt_token>
```

### 获取Token

**请求**:
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

---

## 基础URL

### 开发环境
```
https://api-dev.rqa2025.com/v1
```

### 生产环境
```
https://api.rqa2025.com/v1
```

---

## 错误处理

### 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述信息",
    "details": {
      "field": "具体错误字段",
      "reason": "错误原因"
    }
  }
}
```

### 错误码列表

| 错误码 | HTTP状态码 | 描述 |
|--------|------------|------|
| `UNAUTHORIZED` | 401 | 未授权，Token无效或过期 |
| `FORBIDDEN` | 403 | 禁止访问，权限不足 |
| `NOT_FOUND` | 404 | 资源不存在 |
| `VALIDATION_ERROR` | 422 | 请求参数验证失败 |
| `INTERNAL_ERROR` | 500 | 服务器内部错误 |
| `RATE_LIMITED` | 429 | 请求过于频繁 |

---

## API端点

### 策略管理

#### 1. 创建策略

**请求**:
```http
POST /api/v1/strategies
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "双均线策略",
  "description": "基于5日和20日均线的交叉策略",
  "type": "technical",
  "parameters": {
    "short_period": 5,
    "long_period": 20,
    "symbol": "000001.SZ"
  },
  "code": "strategy_code_here"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": "strategy_001",
    "name": "双均线策略",
    "description": "基于5日和20日均线的交叉策略",
    "type": "technical",
    "status": "active",
    "created_at": "2026-03-29T10:00:00Z",
    "updated_at": "2026-03-29T10:00:00Z"
  }
}
```

#### 2. 获取策略列表

**请求**:
```http
GET /api/v1/strategies?page=1&page_size=20&status=active
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "strategy_001",
        "name": "双均线策略",
        "type": "technical",
        "status": "active",
        "created_at": "2026-03-29T10:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 20,
      "total": 100,
      "total_pages": 5
    }
  }
}
```

#### 3. 获取策略详情

**请求**:
```http
GET /api/v1/strategies/{strategy_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": "strategy_001",
    "name": "双均线策略",
    "description": "基于5日和20日均线的交叉策略",
    "type": "technical",
    "parameters": {
      "short_period": 5,
      "long_period": 20,
      "symbol": "000001.SZ"
    },
    "code": "strategy_code_here",
    "status": "active",
    "performance": {
      "total_return": 15.5,
      "sharpe_ratio": 1.2,
      "max_drawdown": -8.3
    },
    "created_at": "2026-03-29T10:00:00Z",
    "updated_at": "2026-03-29T10:00:00Z"
  }
}
```

#### 4. 更新策略

**请求**:
```http
PUT /api/v1/strategies/{strategy_id}
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "双均线策略V2",
  "description": "优化版双均线策略",
  "parameters": {
    "short_period": 10,
    "long_period": 30,
    "symbol": "000001.SZ"
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": "strategy_001",
    "name": "双均线策略V2",
    "description": "优化版双均线策略",
    "updated_at": "2026-03-29T11:00:00Z"
  }
}
```

#### 5. 删除策略

**请求**:
```http
DELETE /api/v1/strategies/{strategy_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "message": "策略删除成功"
  }
}
```

#### 6. 执行策略回测

**请求**:
```http
POST /api/v1/strategies/{strategy_id}/backtest
Content-Type: application/json
Authorization: Bearer <token>

{
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "initial_capital": 100000,
  "commission": 0.001
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "backtest_id": "bt_001",
    "status": "running",
    "parameters": {
      "start_date": "2025-01-01",
      "end_date": "2025-12-31",
      "initial_capital": 100000
    },
    "created_at": "2026-03-29T10:00:00Z"
  }
}
```

---

### 特征管理

#### 1. 创建特征

**请求**:
```http
POST /api/v1/features
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "MA5",
  "description": "5日移动平均线",
  "type": "technical",
  "category": "trend",
  "parameters": {
    "period": 5,
    "field": "close"
  },
  "computation": "computation_logic_here"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": "feature_001",
    "name": "MA5",
    "description": "5日移动平均线",
    "type": "technical",
    "category": "trend",
    "status": "active",
    "created_at": "2026-03-29T10:00:00Z"
  }
}
```

#### 2. 计算特征

**请求**:
```http
POST /api/v1/features/{feature_id}/compute
Content-Type: application/json
Authorization: Bearer <token>

{
  "symbols": ["000001.SZ", "000002.SZ"],
  "start_date": "2025-01-01",
  "end_date": "2025-12-31"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "computation_id": "comp_001",
    "status": "completed",
    "results": {
      "000001.SZ": [10.5, 10.6, 10.7],
      "000002.SZ": [20.1, 20.2, 20.3]
    },
    "execution_time": 1.5
  }
}
```

---

### 模型管理

#### 1. 创建模型

**请求**:
```http
POST /api/v1/models
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "XGBoost预测模型",
  "description": "基于XGBoost的价格预测模型",
  "type": "xgboost",
  "version": "1.0.0",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "id": "model_001",
    "name": "XGBoost预测模型",
    "type": "xgboost",
    "version": "1.0.0",
    "status": "created",
    "created_at": "2026-03-29T10:00:00Z"
  }
}
```

#### 2. 训练模型

**请求**:
```http
POST /api/v1/models/{model_id}/train
Content-Type: application/json
Authorization: Bearer <token>

{
  "dataset_id": "dataset_001",
  "features": ["MA5", "MA20", "RSI"],
  "target": "future_return",
  "train_test_split": 0.8,
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 8
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "training_id": "train_001",
    "status": "running",
    "progress": 0,
    "estimated_time": 300
  }
}
```

#### 3. 模型推理

**请求**:
```http
POST /api/v1/models/{model_id}/predict
Content-Type: application/json
Authorization: Bearer <token>

{
  "features": {
    "MA5": 10.5,
    "MA20": 10.3,
    "RSI": 65.5
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "prediction": 0.85,
    "probability": 0.92,
    "confidence": "high",
    "execution_time": 0.05
  }
}
```

---

### 交易执行

#### 1. 下单

**请求**:
```http
POST /api/v1/orders
Content-Type: application/json
Authorization: Bearer <token>

{
  "symbol": "000001.SZ",
  "side": "buy",
  "order_type": "limit",
  "quantity": 1000,
  "price": 10.5,
  "strategy_id": "strategy_001"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "order_id": "order_001",
    "symbol": "000001.SZ",
    "side": "buy",
    "order_type": "limit",
    "quantity": 1000,
    "price": 10.5,
    "status": "submitted",
    "created_at": "2026-03-29T10:00:00Z"
  }
}
```

#### 2. 查询订单

**请求**:
```http
GET /api/v1/orders/{order_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "order_id": "order_001",
    "symbol": "000001.SZ",
    "side": "buy",
    "quantity": 1000,
    "price": 10.5,
    "filled_quantity": 500,
    "avg_price": 10.48,
    "status": "partially_filled",
    "created_at": "2026-03-29T10:00:00Z",
    "updated_at": "2026-03-29T10:05:00Z"
  }
}
```

#### 3. 撤单

**请求**:
```http
DELETE /api/v1/orders/{order_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "order_id": "order_001",
    "status": "cancelled",
    "cancelled_at": "2026-03-29T10:10:00Z"
  }
}
```

---

### 监控指标

#### 1. 获取系统指标

**请求**:
```http
GET /api/v1/metrics/system
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": 67.5,
    "disk_usage": 78.3,
    "network_io": {
      "in": 1024,
      "out": 2048
    },
    "timestamp": "2026-03-29T10:00:00Z"
  }
}
```

#### 2. 获取应用指标

**请求**:
```http
GET /api/v1/metrics/application
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "active_users": 150,
    "requests_per_second": 45,
    "average_response_time": 0.12,
    "error_rate": 0.01,
    "timestamp": "2026-03-29T10:00:00Z"
  }
}
```

#### 3. 获取业务指标

**请求**:
```http
GET /api/v1/metrics/business
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "strategies_executed": 1200,
    "trades_executed": 500,
    "features_computed": 50000,
    "model_inferences": 10000,
    "timestamp": "2026-03-29T10:00:00Z"
  }
}
```

---

## 示例代码

### Python示例

```python
import requests
import json

# 配置
BASE_URL = "https://api.rqa2025.com/v1"
API_TOKEN = "your_jwt_token"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

# 1. 创建策略
def create_strategy():
    url = f"{BASE_URL}/strategies"
    data = {
        "name": "双均线策略",
        "description": "基于5日和20日均线的交叉策略",
        "type": "technical",
        "parameters": {
            "short_period": 5,
            "long_period": 20,
            "symbol": "000001.SZ"
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 2. 执行回测
def run_backtest(strategy_id):
    url = f"{BASE_URL}/strategies/{strategy_id}/backtest"
    data = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "initial_capital": 100000
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 3. 获取策略列表
def list_strategies():
    url = f"{BASE_URL}/strategies?page=1&page_size=20"
    
    response = requests.get(url, headers=headers)
    return response.json()

# 4. 模型推理
def predict(model_id, features):
    url = f"{BASE_URL}/models/{model_id}/predict"
    data = {"features": features}
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 创建策略
    result = create_strategy()
    print(f"策略创建结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    if result["success"]:
        strategy_id = result["data"]["id"]
        
        # 执行回测
        backtest_result = run_backtest(strategy_id)
        print(f"回测结果: {json.dumps(backtest_result, indent=2, ensure_ascii=False)}")
```

### JavaScript示例

```javascript
const axios = require('axios');

// 配置
const BASE_URL = 'https://api.rqa2025.com/v1';
const API_TOKEN = 'your_jwt_token';

// 请求头
const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_TOKEN}`
};

// 1. 创建策略
async function createStrategy() {
  const url = `${BASE_URL}/strategies`;
  const data = {
    name: '双均线策略',
    description: '基于5日和20日均线的交叉策略',
    type: 'technical',
    parameters: {
      short_period: 5,
      long_period: 20,
      symbol: '000001.SZ'
    }
  };
  
  try {
    const response = await axios.post(url, data, { headers });
    return response.data;
  } catch (error) {
    console.error('创建策略失败:', error.response?.data || error.message);
    throw error;
  }
}

// 2. 获取策略列表
async function listStrategies() {
  const url = `${BASE_URL}/strategies?page=1&page_size=20`;
  
  try {
    const response = await axios.get(url, { headers });
    return response.data;
  } catch (error) {
    console.error('获取策略列表失败:', error.response?.data || error.message);
    throw error;
  }
}

// 使用示例
async function main() {
  try {
    // 创建策略
    const result = await createStrategy();
    console.log('策略创建结果:', JSON.stringify(result, null, 2));
    
    // 获取策略列表
    const strategies = await listStrategies();
    console.log('策略列表:', JSON.stringify(strategies, null, 2));
  } catch (error) {
    console.error('操作失败:', error);
  }
}

main();
```

### cURL示例

```bash
# 1. 登录获取Token
curl -X POST https://api.rqa2025.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'

# 2. 创建策略
curl -X POST https://api.rqa2025.com/v1/strategies \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "name": "双均线策略",
    "description": "基于5日和20日均线的交叉策略",
    "type": "technical",
    "parameters": {
      "short_period": 5,
      "long_period": 20,
      "symbol": "000001.SZ"
    }
  }'

# 3. 获取策略列表
curl -X GET "https://api.rqa2025.com/v1/strategies?page=1&page_size=20" \
  -H "Authorization: Bearer your_jwt_token"

# 4. 执行回测
curl -X POST https://api.rqa2025.com/v1/strategies/strategy_001/backtest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 100000
  }'
```

---

## 变更日志

### v1.0.0 (2026-03-29)

#### 新增
- ✅ 策略管理API（创建、查询、更新、删除、回测）
- ✅ 特征管理API（创建、计算）
- ✅ 模型管理API（创建、训练、推理）
- ✅ 交易执行API（下单、查询、撤单）
- ✅ 监控指标API（系统、应用、业务指标）
- ✅ 完整的认证机制
- ✅ 详细的错误处理
- ✅ Python、JavaScript、cURL示例代码

#### 优化
- 统一的API响应格式
- 完善的错误码体系
- 详细的参数说明

---

## 附录

### 数据类型定义

#### Strategy（策略）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 否 | 策略ID（系统自动生成） |
| name | string | 是 | 策略名称 |
| description | string | 否 | 策略描述 |
| type | string | 是 | 策略类型（technical/fundamental/ml） |
| parameters | object | 是 | 策略参数 |
| code | string | 否 | 策略代码 |
| status | string | 否 | 策略状态（active/inactive） |
| created_at | string | 否 | 创建时间（ISO 8601格式） |
| updated_at | string | 否 | 更新时间（ISO 8601格式） |

#### Feature（特征）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 否 | 特征ID |
| name | string | 是 | 特征名称 |
| description | string | 否 | 特征描述 |
| type | string | 是 | 特征类型 |
| category | string | 是 | 特征类别 |
| parameters | object | 是 | 特征参数 |
| status | string | 否 | 特征状态 |

#### Model（模型）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | string | 否 | 模型ID |
| name | string | 是 | 模型名称 |
| description | string | 否 | 模型描述 |
| type | string | 是 | 模型类型 |
| version | string | 是 | 模型版本 |
| parameters | object | 是 | 模型参数 |
| status | string | 否 | 模型状态 |

---

**文档维护**

| 角色 | 姓名 | 更新频率 |
|------|------|----------|
| 文档维护 | 王芳 | 随API更新 |
| 技术审核 | 李明 | 每版本 |
| 发布确认 | 张伟 | 每版本 |

---

*文档版本: v1.0.0*  
*最后更新: 2026-03-29*  
*下次审核: 2026-04-29*
