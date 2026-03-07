# RQA2025 基础设施层 API 参考文档

## 概述

本文档提供了RQA2025项目基础设施层的完整API参考，包括所有核心接口、类、方法和使用示例。

**版本**: 4.0  
**更新时间**: 2025-01-27  
**适用版本**: RQA2025 4.0+

## 目录

1. [配置管理接口](#配置管理接口)
2. [监控系统接口](#监控系统接口)
3. [缓存系统接口](#缓存系统接口)
4. [依赖注入接口](#依赖注入接口)
5. [日志系统接口](#日志系统接口)
6. [健康检查接口](#健康检查接口)
7. [错误处理接口](#错误处理接口)
8. [存储系统接口](#存储系统接口)
9. [安全系统接口](#安全系统接口)
10. [数据库管理接口](#数据库管理接口)
11. [服务启动器接口](#服务启动器接口)
12. [部署验证接口](#部署验证接口)
13. [性能优化组件](#性能优化组件)

## 配置管理接口

### IConfigManager

配置管理器的核心接口，提供配置的获取、设置、更新和监听功能。

#### 方法

##### `get(key: str, default: Any = None) -> Any`

获取配置值。

**参数**:
- `key`: 配置键名
- `default`: 默认值，当键不存在时返回

**返回值**: 配置值或默认值

**示例**:
```python
from src.infrastructure.interfaces.unified_interfaces import IConfigManager

class MyConfigManager(IConfigManager):
    def get(self, key: str, default: Any = None) -> Any:
        # 实现配置获取逻辑
        return self._config.get(key, default)

# 使用示例
config_manager = MyConfigManager()
db_host = config_manager.get("database.host", "localhost")
```

##### `set(key: str, value: Any) -> None`

设置配置值。

**参数**:
- `key`: 配置键名
- `value`: 配置值

**示例**:
```python
config_manager.set("database.port", 5432)
config_manager.set("cache.ttl", 3600)
```

##### `update(config: Dict[str, Any]) -> None`

批量更新配置。

**参数**:
- `config`: 配置字典

**示例**:
```python
config_manager.update({
    "database.host": "new-host",
    "database.port": 5433,
    "cache.enabled": True
})
```

##### `watch(key: str, callback: Callable[[str, Any], None]) -> None`

监听配置变化。

**参数**:
- `key`: 要监听的配置键
- `callback`: 配置变化时的回调函数

**示例**:
```python
def on_config_change(key: str, value: Any):
    print(f"配置 {key} 已更改为 {value}")

config_manager.watch("database.host", on_config_change)
```

##### `reload() -> None`

重新加载配置。

**示例**:
```python
# 重新加载配置文件
config_manager.reload()
```

##### `validate(config: Dict[str, Any]) -> bool`

验证配置。

**参数**:
- `config`: 要验证的配置字典

**返回值**: 验证是否通过

**示例**:
```python
config = {"database.host": "localhost", "database.port": 5432}
if config_manager.validate(config):
    print("配置验证通过")
else:
    print("配置验证失败")
```

### IConfigManagerFactory

配置管理器工厂接口，用于创建不同类型的配置管理器。

#### 方法

##### `create_manager(manager_type: str, **kwargs) -> IConfigManager`

创建配置管理器。

**参数**:
- `manager_type`: 管理器类型
- `**kwargs`: 创建参数

**返回值**: 配置管理器实例

**示例**:
```python
from src.infrastructure.interfaces.unified_interfaces import IConfigManagerFactory

class MyConfigManagerFactory(IConfigManagerFactory):
    def create_manager(self, manager_type: str, **kwargs) -> IConfigManager:
        if manager_type == "file":
            return FileConfigManager(**kwargs)
        elif manager_type == "database":
            return DatabaseConfigManager(**kwargs)
        else:
            raise ValueError(f"不支持的管理器类型: {manager_type}")

# 使用示例
factory = MyConfigManagerFactory()
file_config = factory.create_manager("file", config_path="/etc/app/config.yaml")
db_config = factory.create_manager("database", connection_string="postgresql://...")
```

## 监控系统接口

### IMonitor

监控系统的核心接口，提供指标记录、告警管理和状态监控功能。

#### 方法

##### `start() -> None`

启动监控系统。

**示例**:
```python
monitor.start()
```

##### `stop() -> None`

停止监控系统。

**示例**:
```python
monitor.stop()
```

##### `record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None`

记录监控指标。

**参数**:
- `name`: 指标名称
- `value`: 指标值
- `tags`: 指标标签

**示例**:
```python
# 记录响应时间指标
monitor.record_metric("api.response_time", 150.5, {"endpoint": "/api/users"})

# 记录内存使用指标
monitor.record_metric("system.memory_usage", 1024.0, {"unit": "MB"})

# 记录业务指标
monitor.record_metric("business.order_count", 100, {"status": "completed"})
```

##### `record_alert(level: str, message: str, tags: Optional[Dict[str, str]] = None) -> None`

记录告警信息。

**参数**:
- `level`: 告警级别 (info, warning, error, critical)
- `message`: 告警消息
- `tags`: 告警标签

**示例**:
```python
# 记录错误告警
monitor.record_alert("error", "数据库连接失败", {"service": "user-service"})

# 记录警告告警
monitor.record_alert("warning", "内存使用率过高", {"threshold": "80%"})

# 记录信息告警
monitor.record_alert("info", "服务启动完成", {"version": "1.0.0"})
```

##### `get_metrics(name: str, time_range: Optional[tuple] = None) -> List[Dict]`

获取监控指标。

**参数**:
- `name`: 指标名称
- `time_range`: 时间范围 (start_time, end_time)

**返回值**: 指标数据列表

**示例**:
```python
# 获取最近1小时的响应时间指标
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)
metrics = monitor.get_metrics("api.response_time", (start_time, end_time))

for metric in metrics:
    print(f"时间: {metric['timestamp']}, 值: {metric['value']}")
```

##### `get_alerts(level: Optional[str] = None) -> List[Dict]`

获取告警信息。

**参数**:
- `level`: 告警级别过滤

**返回值**: 告警信息列表

**示例**:
```python
# 获取所有告警
all_alerts = monitor.get_alerts()

# 获取错误级别告警
error_alerts = monitor.get_alerts("error")

# 获取特定服务的告警
service_alerts = [alert for alert in all_alerts if alert.get("tags", {}).get("service") == "user-service"]
```

##### `get_status() -> ServiceStatus`

获取监控系统状态。

**返回值**: 服务状态

**示例**:
```python
status = monitor.get_status()
if status == ServiceStatus.UP:
    print("监控系统运行正常")
elif status == ServiceStatus.DEGRADED:
    print("监控系统性能下降")
elif status == ServiceStatus.DOWN:
    print("监控系统已停止")
```

### IMonitorFactory

监控系统工厂接口，用于创建不同类型的监控器。

#### 方法

##### `create_monitor(monitor_type: str, **kwargs) -> IMonitor`

创建监控器。

**参数**:
- `monitor_type`: 监控器类型
- `**kwargs`: 创建参数

**返回值**: 监控器实例

**示例**:
```python
from src.infrastructure.interfaces.unified_interfaces import IMonitorFactory

class MyMonitorFactory(IMonitorFactory):
    def create_monitor(self, monitor_type: str, **kwargs) -> IMonitor:
        if monitor_type == "prometheus":
            return PrometheusMonitor(**kwargs)
        elif monitor_type == "datadog":
            return DataDogMonitor(**kwargs)
        elif monitor_type == "custom":
            return CustomMonitor(**kwargs)
        else:
            raise ValueError(f"不支持的监控器类型: {monitor_type}")

# 使用示例
factory = MyMonitorFactory()
prometheus_monitor = factory.create_monitor("prometheus", endpoint="http://localhost:9090")
datadog_monitor = factory.create_monitor("datadog", api_key="your-api-key")
```

## 缓存系统接口

### ICacheManager

缓存管理器的核心接口，提供缓存数据的存储、检索和管理功能。

#### 方法

##### `get(key: str) -> Optional[Any]`

获取缓存值。

**参数**:
- `key`: 缓存键

**返回值**: 缓存值或None

**示例**:
```python
# 获取用户信息缓存
user_cache = cache_manager.get("user:12345")
if user_cache:
    print(f"从缓存获取用户: {user_cache['name']}")
else:
    print("缓存未命中，从数据库获取")
```

##### `set(key: str, value: Any, ttl: Optional[int] = None) -> None`

设置缓存值。

**参数**:
- `key`: 缓存键
- `value`: 缓存值
- `ttl`: 生存时间（秒）

**示例**:
```python
# 缓存用户信息，TTL为1小时
cache_manager.set("user:12345", {
    "id": 12345,
    "name": "张三",
    "email": "zhangsan@example.com"
}, ttl=3600)

# 缓存配置信息，无TTL
cache_manager.set("app.config", {"debug": True, "log_level": "INFO"})
```

##### `delete(key: str) -> None`

删除缓存项。

**参数**:
- `key`: 缓存键

**示例**:
```python
# 删除用户缓存
cache_manager.delete("user:12345")

# 删除配置缓存
cache_manager.delete("app.config")
```

##### `clear() -> None`

清空所有缓存。

**示例**:
```python
# 清空所有缓存
cache_manager.clear()
```

##### `exists(key: str) -> bool`

检查缓存键是否存在。

**参数**:
- `key`: 缓存键

**返回值**: 是否存在

**示例**:
```python
if cache_manager.exists("user:12345"):
    print("用户缓存存在")
else:
    print("用户缓存不存在")
```

### ICacheManagerFactory

缓存管理器工厂接口，用于创建不同类型的缓存管理器。

#### 方法

##### `create_manager(manager_type: str, **kwargs) -> ICacheManager`

创建缓存管理器。

**参数**:
- `manager_type`: 管理器类型
- `**kwargs`: 创建参数

**返回值**: 缓存管理器实例

**示例**:
```python
from src.infrastructure.interfaces.unified_interfaces import ICacheManagerFactory

class MyCacheManagerFactory(ICacheManagerFactory):
    def create_manager(self, manager_type: str, **kwargs) -> ICacheManager:
        if manager_type == "redis":
            return RedisCacheManager(**kwargs)
        elif manager_type == "memory":
            return MemoryCacheManager(**kwargs)
        elif manager_type == "file":
            return FileCacheManager(**kwargs)
        else:
            raise ValueError(f"不支持的缓存管理器类型: {manager_type}")

# 使用示例
factory = MyCacheManagerFactory()

# 创建Redis缓存管理器
redis_cache = factory.create_manager("redis", 
    host="localhost", 
    port=6379, 
    db=0
)

# 创建内存缓存管理器
memory_cache = factory.create_manager("memory", 
    max_size=1000, 
    eviction_policy="lru"
)

# 创建文件缓存管理器
file_cache = factory.create_manager("file", 
    cache_dir="/tmp/cache", 
    max_file_size=1024*1024
)
```

## 依赖注入接口

### IDependencyContainer

依赖注入容器的核心接口，提供服务的注册、获取和生命周期管理功能。

#### 方法

##### `register(name: str, service_type: Type, factory: Optional[Callable] = None, lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON) -> None`

注册服务。

**参数**:
- `name`: 服务名称
- `service_type`: 服务类型
- `factory`: 工厂函数
- `lifecycle`: 生命周期类型

**示例**:
```python
from src.infrastructure.interfaces.unified_interfaces import IDependencyContainer, ServiceLifecycle

class MyDependencyContainer(IDependencyContainer):
    def register(self, name: str, service_type: Type, 
                factory: Optional[Callable] = None,
                lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON) -> None:
        # 实现服务注册逻辑
        pass

# 使用示例
container = MyDependencyContainer()

# 注册单例服务
container.register("config_manager", ConfigManager)

# 注册瞬态服务
container.register("logger", Logger, lifecycle=ServiceLifecycle.TRANSIENT)

# 注册作用域服务
container.register("database", Database, lifecycle=ServiceLifecycle.SCOPED)

# 注册带工厂函数的服务
def create_cache_manager():
    return CacheManager(max_size=1000)

container.register("cache_manager", CacheManager, factory=create_cache_manager)
```

##### `get(name: str) -> Any`

获取服务实例。

**参数**:
- `name`: 服务名称

**返回值**: 服务实例

**示例**:
```python
# 获取配置管理器
config_manager = container.get("config_manager")

# 获取日志记录器
logger = container.get("logger")

# 获取缓存管理器
cache_manager = container.get("cache_manager")
```

##### `has(name: str) -> bool`

检查服务是否已注册。

**参数**:
- `name`: 服务名称

**返回值**: 是否已注册

**示例**:
```python
if container.has("config_manager"):
    config_manager = container.get("config_manager")
else:
    print("配置管理器未注册")
```

##### `unregister(name: str) -> bool`

注销服务。

**参数**:
- `name`: 服务名称

**返回值**: 是否成功注销

**示例**:
```python
if container.unregister("old_service"):
    print("服务注销成功")
else:
    print("服务注销失败")
```

##### `list_services() -> Dict[str, ServiceRegistration]`

列出所有已注册的服务。

**返回值**: 服务注册信息字典

**示例**:
```python
services = container.list_services()
for name, registration in services.items():
    print(f"服务: {name}")
    print(f"  类型: {registration.service_type}")
    print(f"  生命周期: {registration.lifecycle}")
    print(f"  创建时间: {registration.created_at}")
```

## 日志系统接口

### ILogger

日志记录器的核心接口，提供不同级别的日志记录功能。

#### 方法

##### `info(message: str, **kwargs) -> None`

记录信息级别日志。

**参数**:
- `message`: 日志消息
- `**kwargs`: 额外参数

**示例**:
```python
logger.info("用户登录成功", user_id=12345, ip="192.168.1.100")
logger.info("服务启动完成", version="1.0.0", port=8080)
```

##### `error(message: str, **kwargs) -> None`

记录错误级别日志。

**参数**:
- `message`: 日志消息
- `**kwargs`: 额外参数

**示例**:
```python
logger.error("数据库连接失败", error_code=500, retry_count=3)
logger.error("API调用异常", endpoint="/api/users", status_code=500)
```

##### `debug(message: str, **kwargs) -> None`

记录调试级别日志。

**参数**:
- `message`: 日志消息
- `**kwargs`: 额外参数

**示例**:
```python
logger.debug("开始处理请求", request_id="req-123", method="POST")
logger.debug("缓存命中", key="user:12345", ttl=3600)
```

##### `warning(message: str, **kwargs) -> None`

记录警告级别日志。

**参数**:
- `message`: 日志消息
- `**kwargs`: 额外参数

**示例**:
```python
logger.warning("内存使用率过高", usage_percent=85, threshold=80)
logger.warning("配置项缺失", missing_key="database.password")
```

##### `critical(message: str, **kwargs) -> None`

记录严重级别日志。

**参数**:
- `message`: 日志消息
- `**kwargs`: 额外参数

**示例**:
```python
logger.critical("系统崩溃", error_code="SYS_001", stack_trace="...")
logger.critical("数据库连接丢失", connection_id="db-001", retry_failed=True)
```

## 健康检查接口

### IHealthChecker

健康检查器的核心接口，提供服务和系统健康状态检查功能。

#### 方法

##### `check_health() -> Dict[str, Any]`

检查整体健康状态。

**返回值**: 健康状态信息

**示例**:
```python
health_status = health_checker.check_health()
print(f"整体状态: {health_status['overall_status']}")
print(f"健康服务: {health_status['healthy_services']}")
print(f"异常服务: {health_status['unhealthy_services']}")
```

##### `register_service(name: str, check_func: Callable) -> None`

注册服务健康检查函数。

**参数**:
- `name`: 服务名称
- `check_func`: 健康检查函数

**示例**:
```python
def check_database_health():
    try:
        # 执行数据库连接测试
        db.execute("SELECT 1")
        return {"status": "healthy", "response_time": 5.2}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def check_cache_health():
    try:
        # 执行缓存测试
        cache.set("health_check", "ok")
        cache.delete("health_check")
        return {"status": "healthy", "operation_time": 1.1}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# 注册健康检查
health_checker.register_service("database", check_database_health)
health_checker.register_service("cache", check_cache_health)
```

##### `check_service(name: str, timeout: int = 5) -> Dict[str, Any]`

检查特定服务健康状态。

**参数**:
- `name`: 服务名称
- `timeout`: 超时时间（秒）

**返回值**: 服务健康状态

**示例**:
```python
# 检查数据库健康状态
db_health = health_checker.check_service("database", timeout=10)
if db_health["status"] == "healthy":
    print(f"数据库健康，响应时间: {db_health['response_time']}ms")
else:
    print(f"数据库异常: {db_health['error']}")

# 检查缓存健康状态
cache_health = health_checker.check_service("cache")
if cache_health["status"] == "healthy":
    print(f"缓存健康，操作时间: {cache_health['operation_time']}ms")
else:
    print(f"缓存异常: {cache_health['error']}")
```

##### `get_status() -> Dict[str, Any]`

获取所有服务状态。

**返回值**: 所有服务状态信息

**示例**:
```python
all_status = health_checker.get_status()
for service_name, status in all_status.items():
    print(f"服务: {service_name}")
    print(f"  状态: {status['status']}")
    if 'response_time' in status:
        print(f"  响应时间: {status['response_time']}ms")
    if 'error' in status:
        print(f"  错误: {status['error']}")
```

## 错误处理接口

### IErrorHandler

错误处理器的核心接口，提供错误处理、重试和熔断器功能。

#### 方法

##### `handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None`

处理错误。

**参数**:
- `error`: 异常对象
- `context`: 错误上下文

**示例**:
```python
try:
    result = risky_operation()
except Exception as e:
    error_handler.handle_error(e, {
        "operation": "risky_operation",
        "user_id": 12345,
        "timestamp": datetime.now()
    })
```

##### `retry(func: Callable, max_retries: int = 3, delay: float = 1.0) -> Any`

重试函数执行。

**参数**:
- `func`: 要重试的函数
- `max_retries`: 最大重试次数
- `delay`: 重试延迟（秒）

**返回值**: 函数执行结果

**示例**:
```python
def fetch_user_data(user_id: int):
    # 模拟可能失败的操作
    response = requests.get(f"/api/users/{user_id}")
    response.raise_for_status()
    return response.json()

# 使用重试机制
try:
    user_data = error_handler.retry(
        lambda: fetch_user_data(12345),
        max_retries=5,
        delay=2.0
    )
    print(f"获取用户数据成功: {user_data}")
except Exception as e:
    print(f"重试后仍然失败: {e}")
```

##### `circuit_breaker(func: Callable, failure_threshold: int = 5) -> Any`

熔断器模式执行函数。

**参数**:
- `func`: 要执行的函数
- `failure_threshold`: 失败阈值

**返回值**: 函数执行结果

**示例**:
```python
def call_external_api():
    # 模拟外部API调用
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# 使用熔断器
try:
    result = error_handler.circuit_breaker(
        call_external_api,
        failure_threshold=3
    )
    print(f"API调用成功: {result}")
except Exception as e:
    print(f"熔断器触发: {e}")
```

##### `get_error_stats() -> Dict[str, Any]`

获取错误统计信息。

**返回值**: 错误统计信息

**示例**:
```python
stats = error_handler.get_error_stats()
print(f"总错误数: {stats['total_errors']}")
print(f"重试次数: {stats['retry_count']}")
print(f"熔断器触发次数: {stats['circuit_breaker_triggers']}")
print(f"平均响应时间: {stats['avg_response_time']}ms")
```

## 存储系统接口

### IStorage

存储系统的核心接口，提供数据的存储、检索和管理功能。

#### 方法

##### `get(key: str) -> Optional[Any]`

获取存储的数据。

**参数**:
- `key`: 数据键

**返回值**: 存储的数据或None

**示例**:
```python
# 获取用户数据
user_data = storage.get("user:12345")
if user_data:
    print(f"用户: {user_data['name']}")
else:
    print("用户数据不存在")
```

##### `set(key: str, value: Any, ttl: Optional[int] = None) -> None`

存储数据。

**参数**:
- `key`: 数据键
- `value`: 数据值
- `ttl`: 生存时间（秒）

**示例**:
```python
# 存储用户数据
storage.set("user:12345", {
    "id": 12345,
    "name": "张三",
    "email": "zhangsan@example.com",
    "created_at": datetime.now()
}, ttl=86400)  # 24小时

# 存储配置数据
storage.set("app.config", {
    "debug": True,
    "log_level": "INFO",
    "max_connections": 100
})
```

##### `delete(key: str) -> None`

删除数据。

**参数**:
- `key`: 数据键

**示例**:
```python
# 删除用户数据
storage.delete("user:12345")

# 删除配置数据
storage.delete("app.config")
```

##### `exists(key: str) -> bool`

检查数据是否存在。

**参数**:
- `key`: 数据键

**返回值**: 是否存在

**示例**:
```python
if storage.exists("user:12345"):
    print("用户数据存在")
else:
    print("用户数据不存在")
```

## 安全系统接口

### ISecurity

安全系统的核心接口，提供配置验证、访问控制和审计功能。

#### 方法

##### `validate_config(config: Dict[str, Any]) -> tuple[bool, Optional[Dict]]`

验证配置安全性。

**参数**:
- `config`: 要验证的配置

**返回值**: (是否通过, 验证结果)

**示例**:
```python
config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "password": "secret123"
    },
    "api": {
        "secret_key": "api-secret-key"
    }
}

is_valid, result = security.validate_config(config)
if is_valid:
    print("配置验证通过")
else:
    print(f"配置验证失败: {result['errors']}")
```

##### `check_access(resource: str, user: str) -> bool`

检查用户访问权限。

**参数**:
- `resource`: 资源名称
- `user`: 用户名

**返回值**: 是否有访问权限

**示例**:
```python
# 检查用户是否有权限访问用户数据
if security.check_access("user:read", "admin"):
    print("管理员可以读取用户数据")
else:
    print("权限不足")

# 检查用户是否有权限修改配置
if security.check_access("config:write", "developer"):
    print("开发者可以修改配置")
else:
    print("权限不足")
```

##### `audit(action: str, details: Dict[str, Any]) -> None`

记录审计日志。

**参数**:
- `action`: 操作类型
- `details`: 操作详情

**示例**:
```python
# 记录用户登录
security.audit("user.login", {
    "user_id": 12345,
    "ip_address": "192.168.1.100",
    "timestamp": datetime.now(),
    "success": True
})

# 记录配置修改
security.audit("config.modify", {
    "user_id": 12345,
    "config_key": "database.host",
    "old_value": "localhost",
    "new_value": "new-host",
    "timestamp": datetime.now()
})

# 记录权限检查
security.audit("access.check", {
    "user_id": 12345,
    "resource": "user:read",
    "result": "granted",
    "timestamp": datetime.now()
})
```

## 数据库管理接口

### IDatabaseManager

数据库管理器的核心接口，提供数据库连接、执行和状态管理功能。

#### 方法

##### `get_adapter(name: str) -> Any`

获取数据库适配器。

**参数**:
- `name`: 适配器名称

**返回值**: 数据库适配器

**示例**:
```python
# 获取PostgreSQL适配器
postgres_adapter = db_manager.get_adapter("postgresql")

# 获取MySQL适配器
mysql_adapter = db_manager.get_adapter("mysql")

# 获取SQLite适配器
sqlite_adapter = db_manager.get_adapter("sqlite")
```

##### `connect(adapter_name: str) -> bool`

连接数据库。

**参数**:
- `adapter_name`: 适配器名称

**返回值**: 连接是否成功

**示例**:
```python
# 连接PostgreSQL数据库
if db_manager.connect("postgresql"):
    print("PostgreSQL连接成功")
else:
    print("PostgreSQL连接失败")

# 连接MySQL数据库
if db_manager.connect("mysql"):
    print("MySQL连接成功")
else:
    print("MySQL连接失败")
```

##### `disconnect(adapter_name: str) -> None`

断开数据库连接。

**参数**:
- `adapter_name`: 适配器名称

**示例**:
```python
# 断开PostgreSQL连接
db_manager.disconnect("postgresql")

# 断开MySQL连接
db_manager.disconnect("mysql")
```

##### `execute(adapter_name: str, query: str, params: Optional[Dict] = None) -> Any`

执行数据库查询。

**参数**:
- `adapter_name`: 适配器名称
- `query`: SQL查询语句
- `params`: 查询参数

**返回值**: 查询结果

**示例**:
```python
# 执行查询
users = db_manager.execute("postgresql", 
    "SELECT * FROM users WHERE age > %s", 
    {"age": 18}
)

# 执行插入
result = db_manager.execute("postgresql",
    "INSERT INTO users (name, email) VALUES (%s, %s)",
    {"name": "张三", "email": "zhangsan@example.com"}
)

# 执行更新
result = db_manager.execute("postgresql",
    "UPDATE users SET email = %s WHERE id = %s",
    {"email": "new-email@example.com", "id": 12345}
)
```

##### `get_status() -> ServiceStatus`

获取数据库状态。

**返回值**: 服务状态

**示例**:
```python
status = db_manager.get_status()
if status == ServiceStatus.UP:
    print("数据库服务正常")
elif status == ServiceStatus.DEGRADED:
    print("数据库服务性能下降")
elif status == ServiceStatus.DOWN:
    print("数据库服务已停止")
```

## 服务启动器接口

### IServiceLauncher

服务启动器的核心接口，提供服务的启动、停止和重启功能。

#### 方法

##### `start_service(service_name: str, config: Dict[str, Any]) -> bool`

启动服务。

**参数**:
- `service_name`: 服务名称
- `config`: 服务配置

**返回值**: 启动是否成功

**示例**:
```python
# 启动用户服务
user_service_config = {
    "port": 8080,
    "host": "0.0.0.0",
    "workers": 4,
    "timeout": 30
}

if service_launcher.start_service("user-service", user_service_config):
    print("用户服务启动成功")
else:
    print("用户服务启动失败")

# 启动缓存服务
cache_service_config = {
    "redis_host": "localhost",
    "redis_port": 6379,
    "max_connections": 100
}

if service_launcher.start_service("cache-service", cache_service_config):
    print("缓存服务启动成功")
else:
    print("缓存服务启动失败")
```

##### `stop_service(service_name: str) -> bool`

停止服务。

**参数**:
- `service_name`: 服务名称

**返回值**: 停止是否成功

**示例**:
```python
# 停止用户服务
if service_launcher.stop_service("user-service"):
    print("用户服务停止成功")
else:
    print("用户服务停止失败")

# 停止缓存服务
if service_launcher.stop_service("cache-service"):
    print("缓存服务停止成功")
else:
    print("缓存服务停止失败")
```

##### `restart_service(service_name: str) -> bool`

重启服务。

**参数**:
- `service_name`: 服务名称

**返回值**: 重启是否成功

**示例**:
```python
# 重启用户服务
if service_launcher.restart_service("user-service"):
    print("用户服务重启成功")
else:
    print("用户服务重启失败")

# 重启缓存服务
if service_launcher.restart_service("cache-service"):
    print("缓存服务重启成功")
else:
    print("缓存服务重启失败")
```

##### `get_service_status(service_name: str) -> ServiceStatus`

获取服务状态。

**参数**:
- `service_name`: 服务名称

**返回值**: 服务状态

**示例**:
```python
# 检查用户服务状态
user_status = service_launcher.get_service_status("user-service")
if user_status == ServiceStatus.UP:
    print("用户服务运行正常")
elif user_status == ServiceStatus.DOWN:
    print("用户服务已停止")
elif user_status == ServiceStatus.DEGRADED:
    print("用户服务性能下降")

# 检查缓存服务状态
cache_status = service_launcher.get_service_status("cache-service")
if cache_status == ServiceStatus.UP:
    print("缓存服务运行正常")
else:
    print("缓存服务异常")
```

## 部署验证接口

### IDeploymentValidator

部署验证器的核心接口，提供部署配置和环境验证功能。

#### 方法

##### `validate_deployment(deployment_config: Dict[str, Any]) -> bool`

验证部署配置。

**参数**:
- `deployment_config`: 部署配置

**返回值**: 验证是否通过

**示例**:
```python
deployment_config = {
    "environment": "production",
    "services": [
        {
            "name": "user-service",
            "replicas": 3,
            "resources": {
                "cpu": "500m",
                "memory": "1Gi"
            }
        },
        {
            "name": "cache-service",
            "replicas": 2,
            "resources": {
                "cpu": "200m",
                "memory": "512Mi"
            }
        }
    ],
    "config": {
        "database_url": "postgresql://user:pass@host:5432/db",
        "redis_url": "redis://localhost:6379"
    }
}

if deployment_validator.validate_deployment(deployment_config):
    print("部署配置验证通过")
else:
    errors = deployment_validator.get_validation_errors()
    print(f"部署配置验证失败: {errors}")
```

##### `validate_environment(env_name: str) -> bool`

验证环境配置。

**参数**:
- `env_name`: 环境名称

**返回值**: 验证是否通过

**示例**:
```python
# 验证生产环境
if deployment_validator.validate_environment("production"):
    print("生产环境配置验证通过")
else:
    errors = deployment_validator.get_validation_errors()
    print(f"生产环境配置验证失败: {errors}")

# 验证测试环境
if deployment_validator.validate_environment("testing"):
    print("测试环境配置验证通过")
else:
    errors = deployment_validator.get_validation_errors()
    print(f"测试环境配置验证失败: {errors}")
```

##### `get_validation_errors() -> List[str]`

获取验证错误信息。

**返回值**: 验证错误列表

**示例**:
```python
# 验证部署配置
is_valid = deployment_validator.validate_deployment(deployment_config)

if not is_valid:
    errors = deployment_validator.get_validation_errors()
    print("验证错误:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
else:
    print("部署配置验证通过")
```

## 性能优化组件

### CachePerformanceOptimizer

缓存性能优化器，提供智能缓存策略选择、内存使用优化和性能监控功能。

#### 主要功能

- **智能缓存策略选择**: 根据访问模式、数据特性、系统负载等因素，智能选择最优的缓存级别和策略
- **内存使用优化**: 实现内存策略（保守、平衡、激进），支持内存监控和自动优化
- **缓存预热和预加载**: 支持数据预热和预加载，提高缓存命中率
- **性能监控和分析**: 实时收集性能指标，支持性能分析和优化建议
- **自适应缓存调整**: 根据性能指标自动调整优化策略

#### 使用示例

```python
from src.infrastructure.core.cache.performance_optimizer import CachePerformanceOptimizer

# 创建性能优化器
optimizer = CachePerformanceOptimizer({
    "optimization_level": "advanced",
    "memory_policy": "balanced",
    "max_memory_mb": 2048,
    "target_hit_rate": 0.85
})

# 开始性能监控
optimizer.start_monitoring()

# 获取性能报告
report = optimizer.get_performance_report()
print(f"缓存命中率: {report['cache_hit_rate']:.2%}")
print(f"平均响应时间: {report['response_time_ms']:.2f}ms")
print(f"内存使用: {report['memory_usage_mb']:.2f}MB")

# 停止监控
optimizer.stop_monitoring()
```

### AdvancedCacheManager

高级缓存管理器，继承自基础缓存管理器，提供性能优化和预加载功能。

#### 主要功能

- **性能优化**: 集成性能优化器，自动优化缓存策略
- **数据预加载**: 支持数据预加载，提高缓存命中率
- **性能报告**: 提供详细的性能报告和优化建议

#### 使用示例

```python
from src.infrastructure.core.cache.performance_optimizer import AdvancedCacheManager

# 创建高级缓存管理器
cache_manager = AdvancedCacheManager({
    "max_size": 10000,
    "ttl": 3600,
    "enable_compression": True,
    "enable_preloading": True
})

# 预加载数据
cache_manager.preload_data({
    "user:12345": {"name": "张三", "email": "zhangsan@example.com"},
    "user:67890": {"name": "李四", "email": "lisi@example.com"}
})

# 获取性能报告
performance_report = cache_manager.get_performance_report()
print(f"性能报告: {performance_report}")

# 优化性能
optimization_result = cache_manager.optimize_performance()
print(f"优化结果: {optimization_result}")

# 停止管理器
cache_manager.stop()
```

## 总结

本文档提供了RQA2025基础设施层的核心API参考，包括：

1. **核心接口**: 主要组件的接口定义
2. **详细方法**: 每个接口的完整方法说明
3. **使用示例**: 实际使用场景的代码示例
4. **最佳实践**: 推荐的接口使用方式

通过这些接口，开发者可以：

- 统一管理配置、监控、缓存等基础设施服务
- 实现高可用、高性能的系统架构
- 快速集成和扩展基础设施功能
- 遵循最佳实践和设计模式

---

**文档版本**: 4.0  
**最后更新**: 2025-01-27  
**维护者**: RQA2025开发团队
