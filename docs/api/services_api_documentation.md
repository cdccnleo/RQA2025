# RQA 服务层 API 文档

## 概述

本文档详细描述了RQA系统中所有服务的API接口，包括基础服务、模型服务、微服务架构等。

## 目录

1. [基础服务 (BaseService)](#基础服务-baseservice)
2. [模型服务 (ModelService)](#模型服务-modelservice)
3. [微服务框架 (MicroService)](#微服务框架-microservice)
4. [缓存服务 (CacheService)](#缓存服务-cacheservice)
5. [智能监控 (IntelligentMonitoring)](#智能监控-intelligentmonitoring)

---

## 基础服务 (BaseService)

### 类定义

```python
class BaseService(ABC):
    """所有服务的基础抽象类"""
```

### 核心方法

#### start()
启动服务

**参数**: 无

**返回值**: `bool` - 启动是否成功

**示例**:
```python
service = ModelService()
success = service.start()
if success:
    print("服务启动成功")
```

#### stop()
停止服务

**参数**: 无

**返回值**: `bool` - 停止是否成功

**示例**:
```python
service.stop()
```

#### health_check()
健康检查

**参数**: 无

**返回值**: `Dict[str, Any]` - 健康状态信息

**示例**:
```python
health = service.health_check()
print(f"服务状态: {health['status']}")
print(f"运行时间: {health['uptime']}")
```

#### get_status()
获取服务状态

**参数**: 无

**返回值**: `ServiceStatus` - 服务状态枚举

**示例**:
```python
status = service.get_status()
if status == ServiceStatus.RUNNING:
    print("服务正在运行")
```

---

## 模型服务 (ModelService)

### 类定义

```python
class ModelService(BaseService):
    """模型服务，负责模型加载、预测和A/B测试"""
```

### 初始化

```python
ModelService(
    model_path: str,
    model_type: str = "auto",
    enable_ab_test: bool = True,
    cache_predictions: bool = True
)
```

**参数**:
- `model_path` (str): 模型文件路径
- `model_type` (str): 模型类型 ("auto", "joblib", "pickle", "pkl")
- `enable_ab_test` (bool): 是否启用A/B测试
- `cache_predictions` (bool): 是否缓存预测结果

### 核心方法

#### predict(data: Any) -> Dict[str, Any]
执行模型预测

**参数**:
- `data` (Any): 输入数据

**返回值**: `Dict[str, Any]` - 预测结果

**示例**:
```python
model_service = ModelService("models/stock_predictor.joblib")
model_service.start()

# 执行预测
result = model_service.predict({
    "features": [1.2, 3.4, 5.6],
    "timestamp": "2025-08-04T10:00:00Z"
})

print(f"预测结果: {result['prediction']}")
print(f"置信度: {result['confidence']}")
```

#### load_model() -> bool
加载模型

**参数**: 无

**返回值**: `bool` - 加载是否成功

**示例**:
```python
success = model_service.load_model()
if success:
    print("模型加载成功")
```

#### get_model_info() -> Dict[str, Any]
获取模型信息

**参数**: 无

**返回值**: `Dict[str, Any]` - 模型信息

**示例**:
```python
info = model_service.get_model_info()
print(f"模型类型: {info['model_type']}")
print(f"特征数量: {info['feature_count']}")
print(f"创建时间: {info['created_at']}")
```

#### ab_test_predict(data: Any, variant: str = "A") -> Dict[str, Any]
A/B测试预测

**参数**:
- `data` (Any): 输入数据
- `variant` (str): 测试变体 ("A" 或 "B")

**返回值**: `Dict[str, Any]` - A/B测试结果

**示例**:
```python
# A组测试
result_a = model_service.ab_test_predict(data, "A")

# B组测试
result_b = model_service.ab_test_predict(data, "B")

print(f"A组结果: {result_a['prediction']}")
print(f"B组结果: {result_b['prediction']}")
```

---

## 微服务框架 (MicroService)

### 类定义

```python
class MicroService(BaseService):
    """微服务框架，提供服务注册、发现、负载均衡等功能"""
```

### 初始化

```python
MicroService(
    service_name: str,
    service_type: ServiceType,
    port: int = 8000,
    host: str = "localhost"
)
```

**参数**:
- `service_name` (str): 服务名称
- `service_type` (ServiceType): 服务类型
- `port` (int): 服务端口
- `host` (str): 服务主机

### 核心方法

#### register_service() -> bool
注册服务

**参数**: 无

**返回值**: `bool` - 注册是否成功

**示例**:
```python
micro_service = MicroService("api-service", ServiceType.API)
success = micro_service.register_service()
if success:
    print("服务注册成功")
```

#### discover_service(service_name: str) -> Optional[Dict[str, Any]]
发现服务

**参数**:
- `service_name` (str): 要发现的服务名称

**返回值**: `Optional[Dict[str, Any]]` - 服务信息

**示例**:
```python
service_info = micro_service.discover_service("model-service")
if service_info:
    print(f"发现服务: {service_info['name']}")
    print(f"地址: {service_info['address']}")
    print(f"端口: {service_info['port']}")
```

#### health_check() -> Dict[str, Any]
健康检查

**参数**: 无

**返回值**: `Dict[str, Any]` - 健康状态信息

**示例**:
```python
health = micro_service.health_check()
print(f"服务状态: {health['status']}")
print(f"响应时间: {health['response_time']}ms")
print(f"内存使用: {health['memory_usage']}MB")
```

#### get_service_registry() -> Dict[str, Any]
获取服务注册表

**参数**: 无

**返回值**: `Dict[str, Any]` - 服务注册表信息

**示例**:
```python
registry = micro_service.get_service_registry()
for service_name, info in registry.items():
    print(f"服务: {service_name}")
    print(f"  类型: {info['type']}")
    print(f"  状态: {info['status']}")
```

---

## 缓存服务 (CacheService)

### 类定义

```python
class CacheService(BaseService):
    """缓存服务，提供数据缓存功能"""
```

### 初始化

```python
CacheService(
    cache_type: str = "memory",
    max_size: int = 1000,
    ttl: int = 3600
)
```

**参数**:
- `cache_type` (str): 缓存类型 ("memory", "redis")
- `max_size` (int): 最大缓存条目数
- `ttl` (int): 缓存生存时间（秒）

### 核心方法

#### set(key: str, value: Any, ttl: Optional[int] = None) -> bool
设置缓存

**参数**:
- `key` (str): 缓存键
- `value` (Any): 缓存值
- `ttl` (Optional[int]): 生存时间（秒）

**返回值**: `bool` - 设置是否成功

**示例**:
```python
cache_service = CacheService()
cache_service.start()

# 设置缓存
success = cache_service.set("user:123", {"name": "张三", "age": 30}, ttl=3600)
if success:
    print("缓存设置成功")
```

#### get(key: str) -> Optional[Any]
获取缓存

**参数**:
- `key` (str): 缓存键

**返回值**: `Optional[Any]` - 缓存值

**示例**:
```python
value = cache_service.get("user:123")
if value:
    print(f"用户信息: {value}")
else:
    print("缓存未找到")
```

#### delete(key: str) -> bool
删除缓存

**参数**:
- `key` (str): 缓存键

**返回值**: `bool` - 删除是否成功

**示例**:
```python
success = cache_service.delete("user:123")
if success:
    print("缓存删除成功")
```

#### clear() -> bool
清空缓存

**参数**: 无

**返回值**: `bool` - 清空是否成功

**示例**:
```python
success = cache_service.clear()
if success:
    print("缓存清空成功")
```

#### get_cache_stats() -> Dict[str, Any]
获取缓存统计

**参数**: 无

**返回值**: `Dict[str, Any]` - 缓存统计信息

**示例**:
```python
stats = cache_service.get_cache_stats()
print(f"缓存条目数: {stats['size']}")
print(f"命中率: {stats['hit_rate']:.2%}")
print(f"内存使用: {stats['memory_usage']}MB")
```

---

## 智能监控 (IntelligentMonitoring)

### 类定义

```python
class IntelligentMonitoring:
    """智能监控系统，提供自动调优、预测维护和智能告警"""
```

### 初始化

```python
IntelligentMonitoring(
    config_path: Optional[str] = None
)
```

**参数**:
- `config_path` (Optional[str]): 配置文件路径

### 核心方法

#### start_monitoring() -> bool
启动监控

**参数**: 无

**返回值**: `bool` - 启动是否成功

**示例**:
```python
monitoring = IntelligentMonitoring()
success = monitoring.start_monitoring()
if success:
    print("智能监控启动成功")
```

#### collect_metrics() -> Dict[str, Any]
收集指标

**参数**: 无

**返回值**: `Dict[str, Any]` - 监控指标

**示例**:
```python
metrics = monitoring.collect_metrics()
print(f"CPU使用率: {metrics['cpu_usage']:.2f}%")
print(f"内存使用率: {metrics['memory_usage']:.2f}%")
print(f"响应时间: {metrics['response_time']:.2f}ms")
```

#### perform_auto_tuning() -> Dict[str, Any]
执行自动调优

**参数**: 无

**返回值**: `Dict[str, Any]` - 调优结果

**示例**:
```python
tuning_result = monitoring.perform_auto_tuning()
print(f"调优项目数: {tuning_result['tuning_count']}")
print(f"性能提升: {tuning_result['performance_improvement']:.2f}%")
```

#### check_alerts() -> List[Dict[str, Any]]
检查告警

**参数**: 无

**返回值**: `List[Dict[str, Any]]` - 告警列表

**示例**:
```python
alerts = monitoring.check_alerts()
for alert in alerts:
    print(f"告警级别: {alert['severity']}")
    print(f"告警消息: {alert['message']}")
    print(f"告警时间: {alert['timestamp']}")
```

#### generate_monitoring_report() -> str
生成监控报告

**参数**: 无

**返回值**: `str` - 报告文件路径

**示例**:
```python
report_path = monitoring.generate_monitoring_report()
print(f"监控报告已生成: {report_path}")
```

---

## 错误处理

### 常见错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| SERVICE_NOT_FOUND | 服务未找到 | 检查服务名称是否正确 |
| SERVICE_UNAVAILABLE | 服务不可用 | 检查服务是否启动 |
| MODEL_LOAD_ERROR | 模型加载失败 | 检查模型文件路径和格式 |
| CACHE_FULL | 缓存已满 | 清理缓存或增加缓存大小 |
| TIMEOUT | 操作超时 | 检查网络连接和服务响应时间 |

### 异常处理示例

```python
try:
    model_service = ModelService("models/model.joblib")
    result = model_service.predict(data)
except ModelLoadError as e:
    print(f"模型加载失败: {e}")
except PredictionError as e:
    print(f"预测失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

---

## 最佳实践

### 1. 服务初始化

```python
# 推荐做法
def initialize_services():
    services = {}
    
    # 初始化模型服务
    model_service = ModelService("models/predictor.joblib")
    if model_service.start():
        services['model'] = model_service
    
    # 初始化缓存服务
    cache_service = CacheService()
    if cache_service.start():
        services['cache'] = cache_service
    
    return services
```

### 2. 错误处理和重试

```python
import time
from typing import Callable, Any

def retry_operation(operation: Callable, max_retries: int = 3, delay: float = 1.0) -> Any:
    """重试操作"""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            continue
```

### 3. 健康检查监控

```python
def monitor_services(services: Dict[str, BaseService]):
    """监控服务健康状态"""
    for name, service in services.items():
        health = service.health_check()
        if health['status'] != 'healthy':
            print(f"警告: 服务 {name} 状态异常")
            # 可以在这里添加告警逻辑
```

### 4. 性能优化

```python
# 使用缓存提高性能
def get_cached_prediction(cache_service: CacheService, model_service: ModelService, data: Any):
    cache_key = f"prediction:{hash(str(data))}"
    
    # 尝试从缓存获取
    cached_result = cache_service.get(cache_key)
    if cached_result:
        return cached_result
    
    # 执行预测
    result = model_service.predict(data)
    
    # 缓存结果
    cache_service.set(cache_key, result, ttl=3600)
    
    return result
```

---

## 版本信息

- **API版本**: 1.0.0
- **最后更新**: 2025-08-04
- **兼容性**: Python 3.8+

---

## 联系信息

如有问题或建议，请联系开发团队。 