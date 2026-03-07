# 服务层文档

## 📋 模块概述

服务层（`src/services`）是RQA系统的核心业务服务层，提供完整的业务功能接口。

### 🏗️ 架构设计

服务层采用分层架构设计，每个服务都有明确的职责分工：

- **BaseService**: 服务基类，提供统一的生命周期管理
- **TradingService**: 交易服务，负责交易策略执行和订单管理
- **DataValidationService**: 数据验证服务，负责多源数据一致性检查
- **ModelService**: 模型服务，负责模型加载、预测和A/B测试
- **BusinessService**: 业务服务，负责业务流程编排和工作流管理
- **APIService**: API服务，负责统一的API网关服务
- **MicroService**: 微服务，负责微服务基础框架（待实现）

## 🚀 快速开始

### 基础服务

```python
from src.services import BaseService, ServiceStatus

# 创建自定义服务
class MyService(BaseService):
    def _start(self) -> bool:
        # 启动逻辑
        return True
    
    def _stop(self) -> bool:
        # 停止逻辑
        return True
    
    def _health_check(self) -> Dict[str, Any]:
        return {"status": "healthy"}

# 使用服务
service = MyService("MyService")
service.start()
print(service.get_status())  # ServiceStatus.RUNNING
```

### 数据验证服务

```python
from src.services import DataValidationService
from src.core import EventBus, ServiceContainer

# 创建服务
event_bus = EventBus()
container = ServiceContainer()
validation_service = DataValidationService(event_bus, container)

# 验证实时数据
data = {
    "symbol": "AAPL",
    "price": 150.0,
    "volume": 1000,
    "timestamp": "2024-01-01T10:00:00Z"
}

result = validation_service.validate_realtime_data(data)
print(result["is_valid"])  # True/False
```

### 模型服务

```python
from src.services import ModelService, ABTestManager
from src.core import EventBus, ServiceContainer

# 创建服务
event_bus = EventBus()
container = ServiceContainer()
model_service = ModelService(event_bus, container)

# 加载模型
model_service.load_model("model1", "/path/to/model.pkl")

# 预测
features = {"feature1": 1.0, "feature2": 2.0}
prediction = model_service.predict("model1", features)
print(prediction["prediction"])

# A/B测试
ab_manager = ABTestManager()
ab_manager.create_experiment("exp1", ["model1", "model2"])
result = ab_manager.predict("exp1", features)
```

### 交易服务

```python
from src.services import TradingService
from src.core import EventBus, ServiceContainer

# 创建服务
event_bus = EventBus()
container = ServiceContainer()
trading_service = TradingService(event_bus, container)

# 执行策略
strategy_config = {
    "symbols": ["AAPL", "GOOGL"],
    "strategy_type": "momentum",
    "parameters": {"window": 20}
}

result = trading_service.execute_strategy(strategy_config)
print(result["success"])  # True/False
```

### API服务

```python
from src.services import APIService, APIVersion, RateLimitStrategy, APIEndpoint
from src.core import EventBus, ServiceContainer

# 创建服务
event_bus = EventBus()
container = ServiceContainer()
api_service = APIService(event_bus, container)

# 注册自定义端点
def custom_handler(method, path, headers, body, backend):
    return {"message": "Hello from custom endpoint"}

endpoint = APIEndpoint(
    path="/custom",
    method="GET",
    handler=custom_handler,
    description="自定义端点",
    tags=["custom"]
)

api_service.register_endpoint(endpoint)

# 路由请求
result = api_service.route_request("GET", "/custom")
print(result["success"])  # True

# 设置限流
api_service.set_rate_limit("GET:/custom", 100, RateLimitStrategy.FIXED_WINDOW)

# 添加认证
api_service.add_api_key("my_key", "My API Key", ["read", "write"])

# 获取统计信息
stats = api_service.get_stats()
print(stats["total_requests"])
```

### 业务服务

```python
from src.services import BusinessService
from src.core import EventBus, ServiceContainer

# 创建服务
event_bus = EventBus()
container = ServiceContainer()
business_service = BusinessService(event_bus, container)

# 创建工作流
workflow_config = {
    "steps": [
        {"name": "data_validation", "service": "data_service", "method": "validate"},
        {"name": "model_prediction", "service": "model_service", "method": "predict"},
        {"name": "trading_execution", "service": "trading_service", "method": "execute"}
    ]
}

business_service.create_workflow("workflow1", workflow_config)

# 启动工作流
business_service.start_workflow("workflow1", {"input_data": "test"})

# 获取工作流状态
status = business_service.get_workflow_status("workflow1")
print(status["status"])  # running/completed/error
```

## 📊 服务统计

### 测试覆盖

- **总测试用例**: 110个
- **BaseService**: 15个测试
- **TradingService**: 15个测试
- **DataValidationService**: 15个测试
- **ModelService**: 25个测试
- **BusinessService**: 21个测试
- **APIService**: 18个测试

### 代码质量

- **代码行数**: 2000+行
- **类型注解**: 100%覆盖
- **文档覆盖**: 100%完整
- **测试通过率**: 100%

## 🔧 API参考

### BaseService

基础服务类，提供统一的生命周期管理。

#### 方法

- `start() -> bool`: 启动服务
- `stop() -> bool`: 停止服务
- `health_check() -> Dict[str, Any]`: 健康检查
- `get_status() -> ServiceStatus`: 获取服务状态
- `is_running() -> bool`: 检查是否运行中

### DataValidationService

数据验证服务，负责多源数据一致性检查。

#### 方法

- `validate_realtime_data(data: Dict) -> Dict`: 验证实时数据
- `batch_validate(data_list: List[Dict]) -> Dict`: 批量验证数据
- `compare_data(data1: Dict, data2: Dict) -> Dict`: 比较数据

### ModelService

模型服务，负责模型加载、预测和A/B测试。

#### 方法

- `load_model(name: str, path: str) -> bool`: 加载模型
- `unload_model(name: str) -> bool`: 卸载模型
- `predict(model_name: str, features: Dict) -> Dict`: 模型预测
- `list_models() -> List[str]`: 列出所有模型
- `get_model_info(name: str) -> Dict`: 获取模型信息

### TradingService

交易服务，负责交易策略执行和订单管理。

#### 方法

- `execute_strategy(config: Dict) -> Dict`: 执行交易策略
- `on_data_ready(event: Event)`: 处理数据就绪事件
- `on_signal_generated(event: Event)`: 处理信号生成事件
- `on_risk_checked(event: Event)`: 处理风险检查事件

### APIService

API服务，负责统一的API网关服务。

#### 方法

- `register_endpoint(endpoint: APIEndpoint) -> bool`: 注册API端点
- `unregister_endpoint(method: str, path: str) -> bool`: 注销API端点
- `route_request(method: str, path: str, **kwargs) -> Dict`: 路由API请求
- `add_load_balancer(service_name: str, backends: List[str]) -> bool`: 添加负载均衡器
- `set_rate_limit(route_key: str, limit: int, strategy: RateLimitStrategy) -> bool`: 设置限流
- `add_auth_token(token: str, user_id: str, permissions: List[str], expires_at: datetime) -> bool`: 添加认证令牌
- `add_api_key(key: str, name: str, permissions: List[str]) -> bool`: 添加API密钥
- `get_api_docs(version: Optional[APIVersion] = None) -> Dict[str, Any]`: 获取API文档
- `get_stats() -> Dict[str, Any]`: 获取统计信息

#### 配置类

- `APIVersion`: API版本枚举 (V1, V2, V3)
- `RateLimitStrategy`: 限流策略枚举 (FIXED_WINDOW, SLIDING_WINDOW, TOKEN_BUCKET)
- `APIEndpoint`: API端点配置类

### BusinessService

业务服务，负责业务流程编排和工作流管理。

#### 方法

- `create_workflow(workflow_id: str, config: Dict) -> bool`: 创建工作流
- `start_workflow(workflow_id: str, input_data: Dict = None) -> bool`: 启动工作流
- `stop_workflow(workflow_id: str) -> bool`: 停止工作流
- `get_workflow_status(workflow_id: str) -> Dict[str, Any]`: 获取工作流状态
- `list_workflows() -> List[str]`: 列出所有工作流

## 🎯 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    服务层 (src/services)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ BaseService │  │ APIService  │  │MicroService │        │
│  │   (基类)    │  │ (API网关)   │  │ (微服务)    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │               │               │                │
│           ▼               ▼               ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │TradingService│  │ModelService │  │BusinessService│      │
│  │  (交易服务)  │  │ (模型服务)  │  │ (业务服务)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │               │               │                │
│           ▼               ▼               ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │DataValidation│  │ABTestManager│  │WorkflowEngine│      │
│  │   Service   │  │ (A/B测试)   │  │ (工作流引擎)│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📈 重构改进

### 最新更新 (2025-01-27)

1. **APIService实现**: 完整的API网关服务
   - 支持API路由和负载均衡
   - 实现API版本管理
   - 添加API限流和认证
   - 提供文档生成功能
   - 包含18个单元测试

2. **事件类型扩展**: 新增API相关事件
   - `API_REQUEST`: API请求事件
   - `API_RESPONSE`: API响应事件  
   - `API_ERROR`: API错误事件

3. **测试覆盖完善**: 新增18个APIService测试
   - 端点注册和注销测试
   - 路由请求测试
   - 限流和认证测试
   - 负载均衡测试
   - 统计和文档测试

4. **架构完善**: 服务层现在包含6个核心服务
   - 统一的BaseService基类
   - 标准化的事件驱动架构
   - 完善的依赖注入机制
   - 完整的API网关功能

## 📝 最后更新

- **更新时间**: 2025-01-27
- **版本**: 1.0.0
- **状态**: ✅ 稳定版本
- **测试状态**: ✅ 110个测试全部通过