# RQA2025 完整API文档和使用指南

## 📖 概述

本文档提供了RQA2025量化交易系统所有组件的完整API文档和使用指南。系统采用分层架构设计，包含基础设施层、核心服务层、业务逻辑层等多个层次。

## 🏗️ 系统架构

```
RQA2025 系统架构
├── 基础设施层 (Infrastructure Layer)
│   ├── 配置管理 (Configuration Management)
│   ├── 缓存系统 (Caching System)
│   ├── 安全模块 (Security Module)
│   ├── 监控告警 (Monitoring & Alerting)
│   └── 日志系统 (Logging System)
├── 核心服务层 (Core Services Layer)
│   ├── 事件总线 (Event Bus)
│   ├── 依赖注入 (Dependency Injection)
│   └── 业务流程编排 (Business Process Orchestration)
├── 业务逻辑层 (Business Logic Layer)
│   ├── 数据管理 (Data Management)
│   ├── 特征处理 (Feature Processing)
│   ├── 模型推理 (Model Inference)
│   ├── 策略决策 (Strategy Decision)
│   └── 风控合规 (Risk & Compliance)
└── 交易执行层 (Trading Execution Layer)
    ├── 订单管理 (Order Management)
    ├── 交易网关 (Trading Gateway)
    └── 执行引擎 (Execution Engine)
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 依赖包: 详见 `requirements.txt`
- 系统资源: 建议4GB+ RAM, 2+ CPU核心

### 安装和配置

```bash
# 1. 克隆项目
git clone <repository-url>
cd RQA2025

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境
cp config/default.yaml config/local.yaml
# 编辑 local.yaml 配置数据库、缓存等参数

# 4. 初始化系统
python scripts/init_system.py

# 5. 启动服务
python run.py
```

### 基本使用示例

```python
from src.infrastructure.config.unified_container import UnifiedContainer
from src.infrastructure.cache.unified_cache import UnifiedCache
from src.infrastructure.security.authentication_service import MultiFactorAuthenticationService

# 1. 初始化容器
container = UnifiedContainer()

# 2. 配置服务
cache = UnifiedCache(capacity=1000)
container.register("cache", cache)

auth_service = MultiFactorAuthenticationService()
container.register("auth", auth_service)

# 3. 使用服务
user_id = auth_service.create_user("demo_user", "secure_password", "demo@example.com")
cache.set("user_data", {"user_id": user_id, "status": "active"})
```

## 📚 基础设施层API文档

### 1. 配置管理模块

#### 统一配置管理器 (UnifiedConfigManager)

**功能描述**: 提供统一的配置管理功能，支持多环境配置、热更新和配置验证。

**主要接口**:

```python
class UnifiedConfigManager:
    def __init__(self, config_path: str = None)
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
    def reload(self) -> bool
    def validate_config(self) -> List[str]
    def get_all(self) -> Dict[str, Any]
```

**使用示例**:

```python
from src.infrastructure.config.unified_config_manager import UnifiedConfigManager

# 初始化配置管理器
config_manager = UnifiedConfigManager("config/local.yaml")

# 获取配置
database_url = config_manager.get("database.url")
cache_capacity = config_manager.get("cache.capacity", 1000)

# 设置配置
config_manager.set("logging.level", "DEBUG")

# 重新加载配置
config_manager.reload()

# 验证配置
errors = config_manager.validate_config()
if errors:
    print("配置错误:", errors)
```

#### 配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `database.url` | string | - | 数据库连接URL |
| `database.pool_size` | int | 10 | 数据库连接池大小 |
| `cache.capacity` | int | 1000 | 缓存容量 |
| `cache.ttl` | int | 3600 | 缓存默认TTL(秒) |
| `security.jwt_secret` | string | - | JWT密钥 |
| `security.jwt_expiration` | int | 3600 | JWT过期时间 |
| `monitoring.enabled` | bool | true | 是否启用监控 |
| `logging.level` | string | INFO | 日志级别 |

### 2. 缓存系统模块

#### 统一缓存 (UnifiedCache)

**功能描述**: 提供多种缓存策略的统一缓存接口，支持LRU、LFU、LRU-K等多种淘汰算法。

**主要接口**:

```python
class UnifiedCache:
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU,
                 capacity: int = 1000, default_ttl: int = 3600, **kwargs)
    def get(self, key: str) -> Optional[Any]
    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None, **kwargs) -> None
    def delete(self, key: str) -> bool
    def clear(self) -> None
    def size(self) -> int
    def get_stats(self) -> Dict[str, Any]
```

**支持的缓存策略**:

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| LRU | 最近最少使用 | 通用缓存 |
| LFU | 最少频率使用 | 热点数据 |
| LRU-K | LRU-K算法 | 短期热点 |
| ADAPTIVE | 自适应缓存 | 动态调整 |
| PRIORITY | 优先级缓存 | 分级存储 |
| COST_AWARE | 成本感知 | 计算优化 |

**使用示例**:

```python
from src.infrastructure.cache.unified_cache import UnifiedCache, CacheStrategy

# 创建LRU缓存
cache = UnifiedCache(strategy=CacheStrategy.LRU, capacity=1000)

# 基本操作
cache.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=3600)
user_data = cache.get("user:123")
cache.delete("user:123")

# 获取统计信息
stats = cache.get_stats()
print(f"命中率: {stats['hit_rate']}%, 大小: {stats['size']}")

# 创建自适应缓存
adaptive_cache = UnifiedCache(
    strategy=CacheStrategy.ADAPTIVE,
    capacity=5000,
    max_memory_mb=100
)
```

### 3. 安全模块

#### 多因素认证服务 (MultiFactorAuthenticationService)

**功能描述**: 提供完整的用户认证、授权和会话管理功能。

**主要接口**:

```python
class MultiFactorAuthenticationService:
    def create_user(self, username: str, email: str, password: str,
                   role: UserRole = UserRole.VIEWER) -> Optional[str]
    def authenticate_user(self, username: str, credentials: Dict[str, Any],
                         required_factors: List[AuthMethod] = None) -> AuthResult
    def authorize_access(self, user_id: str, resource: str,
                        action: str) -> AuthorizationResult
    def logout(self, token: str) -> bool
    def verify_token(self, token: str) -> Optional[User]
    def change_password(self, user_id: str, old_password: str,
                       new_password: str) -> bool
```

**使用示例**:

```python
from src.infrastructure.security.authentication_service import (
    MultiFactorAuthenticationService, UserRole
)

# 初始化认证服务
auth_service = MultiFactorAuthenticationService()

# 创建用户
user_id = auth_service.create_user(
    username="trader_john",
    email="john@trading.com",
    password="SecurePass123!",
    role=UserRole.TRADER
)

# 用户认证
auth_result = auth_service.authenticate_user(
    "trader_john",
    {"password": "SecurePass123!"}
)

if auth_result.status.name == "SUCCESS":
    token = auth_result.token
    print(f"登录成功，Token: {token}")

    # 验证权限
    authz_result = auth_service.authorize_access(
        user_id, "portfolio", "read"
    )

    # 注销
    auth_service.logout(token)
```

#### 数据保护服务 (DataProtectionService)

**功能描述**: 提供数据脱敏、加密、标记化和哈希等数据保护功能。

**主要接口**:

```python
class DataProtectionService:
    def protect_data(self, data: Dict[str, Any], rule_id: str,
                    user_id: str = "system") -> Dict[str, Any]
    def unprotect_data(self, data: Dict[str, Any], rule_id: str,
                      user_id: str = "system") -> Dict[str, Any]
    def add_protection_rule(self, rule: ProtectionRule) -> None
    def create_protection_rule(self, rule_name: str, data_type: str,
                              fields: List[DataField]) -> ProtectionRule
```

**使用示例**:

```python
from src.infrastructure.security.data_protection_service import (
    DataProtectionService, ProtectionMethod, DataSensitivity
)

# 初始化数据保护服务
protection_service = DataProtectionService()

# 创建保护规则
rule = protection_service.create_protection_rule(
    rule_name="user_data_protection",
    data_type="user",
    fields=[
        DataField(
            name="phone",
            sensitivity=DataSensitivity.SENSITIVE,
            protection_method=ProtectionMethod.MASKING,
            pattern=r"(\d{3})\d{4}(\d{4})",
            description="手机号脱敏"
        ),
        DataField(
            name="password",
            sensitivity=DataSensitivity.SENSITIVE,
            protection_method=ProtectionMethod.HASHING,
            description="密码哈希"
        )
    ]
)

protection_service.add_protection_rule(rule)

# 保护数据
user_data = {
    "phone": "13800138000",
    "password": "mypassword",
    "email": "user@example.com"
}

protected_data = protection_service.protect_data(user_data, "user_data_protection")
print(protected_data)
# 输出: {"phone": "138****8000", "password": "HASH:...", "email": "user@example.com"}
```

### 4. 监控告警模块

#### 应用监控器 (ApplicationMonitor)

**功能描述**: 提供全面的系统监控功能，包括系统指标、业务指标、安全指标和性能指标。

**主要接口**:

```python
class ApplicationMonitor:
    def collect_system_metrics(self) -> Dict[str, float]
    def collect_business_metrics(self) -> Dict[str, float]
    def collect_security_metrics(self) -> Dict[str, Any]
    def collect_performance_metrics(self) -> Dict[str, float]
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None)
    def generate_dashboard_data(self) -> Dict[str, Any]
    def generate_report(self) -> Dict[str, Any]
```

**监控指标说明**:

| 指标分类 | 指标名称 | 说明 |
|----------|----------|------|
| 系统指标 | cpu_percent | CPU使用率 |
| 系统指标 | memory_percent | 内存使用率 |
| 系统指标 | disk_percent | 磁盘使用率 |
| 系统指标 | network_bytes_sent | 网络发送字节数 |
| 业务指标 | cache_hit_rate | 缓存命中率 |
| 业务指标 | auth_success_rate | 认证成功率 |
| 业务指标 | transaction_success_rate | 事务成功率 |
| 业务指标 | error_rate | 错误率 |
| 性能指标 | avg_response_time | 平均响应时间 |
| 性能指标 | throughput_requests_per_second | 吞吐量 |
| 安全指标 | failed_login_attempts | 失败登录尝试次数 |
| 安全指标 | suspicious_activities_count | 可疑活动数量 |

**使用示例**:

```python
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor

# 初始化监控器
monitor = ApplicationMonitor("RQA2025_Application")

# 收集系统指标
system_metrics = monitor.collect_system_metrics()
print(f"CPU使用率: {system_metrics['cpu_percent']}%")
print(f"内存使用率: {system_metrics['memory_percent']}%")

# 记录业务指标
monitor.record_metric("api_requests", 150.5, {"endpoint": "/api/trade"})
monitor.record_metric("cache_hits", 95, {"cache_type": "lru"})

# 收集业务指标
business_metrics = monitor.collect_business_metrics()
print(f"缓存命中率: {business_metrics.get('cache_hit_rate', 0)}%")

# 生成仪表板数据
dashboard_data = monitor.generate_dashboard_data()
print("仪表板数据:", dashboard_data)
```

#### 智能告警系统 (IntelligentAlertSystem)

**功能描述**: 提供智能告警规则配置、触发和通知功能。

**主要接口**:

```python
class IntelligentAlertSystem:
    def add_rule(self, rule: AlertRule) -> None
    def process_alerts(self) -> None
    def get_active_alerts(self) -> List[Alert]
    def get_rules(self) -> Dict[str, AlertRule]
    def suppress_alert(self, alert_id: str, duration_minutes: int) -> None
```

**使用示例**:

```python
from src.infrastructure.monitoring.alert_system import (
    IntelligentAlertSystem, AlertRule, AlertLevel, AlertChannel
)

# 初始化告警系统
alert_system = IntelligentAlertSystem()

# 添加告警规则
cpu_alert_rule = AlertRule(
    rule_id="high_cpu_usage",
    name="High CPU Usage Alert",
    condition="cpu_percent > 80",
    level=AlertLevel.WARNING,
    channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
    cooldown_minutes=5
)

alert_system.add_rule(cpu_alert_rule)

# 处理告警（通常由监控系统自动调用）
alert_system.process_alerts()

# 获取活跃告警
active_alerts = alert_system.get_active_alerts()
for alert in active_alerts:
    print(f"告警: {alert.title} - {alert.message}")
```

### 5. 日志系统模块

#### 统一日志器 (UnifiedLogger)

**功能描述**: 提供统一的日志记录和管理功能，支持多级别日志、结构化日志和日志轮转。

**主要接口**:

```python
class UnifiedLogger:
    def __init__(self, name: str, level: LogLevel = LogLevel.INFO)
    def debug(self, message: str, **kwargs) -> None
    def info(self, message: str, **kwargs) -> None
    def warning(self, message: str, **kwargs) -> None
    def error(self, message: str, **kwargs) -> None
    def critical(self, message: str, **kwargs) -> None
    def log_performance(self, operation: str, duration: float, **kwargs) -> None
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None
    def log_business_event(self, event_type: str, details: Dict[str, Any]) -> None
```

**使用示例**:

```python
from src.infrastructure.logging.unified_logger import UnifiedLogger, LogLevel

# 初始化日志器
logger = UnifiedLogger("trading_service", LogLevel.INFO)

# 记录不同级别日志
logger.info("交易服务启动成功")
logger.warning("缓存使用率过高", cache_usage=85.5)
logger.error("数据库连接失败", error="Connection timeout")

# 记录性能日志
start_time = time.time()
# 执行业务逻辑...
duration = time.time() - start_time
logger.log_performance("process_trade", duration, symbol="AAPL", volume=100)

# 记录安全事件
logger.log_security_event("login_attempt", {
    "user": "john_doe",
    "ip": "192.168.1.100",
    "success": True
})

# 记录业务事件
logger.log_business_event("trade_executed", {
    "symbol": "AAPL",
    "quantity": 100,
    "price": 150.25,
    "profit": 250.00
})
```

## 🔧 高级配置

### 1. 依赖注入配置

```python
from src.infrastructure.config.unified_container import UnifiedContainer

# 创建容器
container = UnifiedContainer()

# 注册服务
container.register("cache", UnifiedCache(capacity=5000), lifecycle="singleton")
container.register("auth", MultiFactorAuthenticationService(), lifecycle="singleton")
container.register("monitor", ApplicationMonitor(), lifecycle="singleton")

# 获取服务
cache = container.get("cache")
auth_service = container.get("auth")

# 检查服务健康状态
health_status = container.check_health("cache")
print(f"Cache健康状态: {health_status}")
```

### 2. 事件驱动架构

```python
from src.core.event_bus import EventBus
from src.core.business_process_orchestrator import BusinessProcessOrchestrator

# 初始化事件总线
event_bus = EventBus()

# 订阅事件
@event_bus.subscribe("market_data_received")
def handle_market_data(data):
    print(f"收到市场数据: {data}")

@event_bus.subscribe("trade_signal_generated")
def handle_trade_signal(signal):
    print(f"收到交易信号: {signal}")

# 发布事件
event_bus.publish("market_data_received", {
    "symbol": "AAPL",
    "price": 150.25,
    "volume": 1000
})

# 业务流程编排
orchestrator = BusinessProcessOrchestrator(event_bus)
orchestrator.start_trading_cycle(["AAPL", "GOOGL"], strategy_config)
```

## 📊 性能优化指南

### 1. 缓存优化

```python
# 选择合适的缓存策略
cache_configs = {
    "user_sessions": {
        "strategy": CacheStrategy.LRU,
        "capacity": 10000,
        "ttl": 3600
    },
    "market_data": {
        "strategy": CacheStrategy.LFU,
        "capacity": 50000,
        "ttl": 300
    },
    "analytics": {
        "strategy": CacheStrategy.ADAPTIVE,
        "capacity": 100000,
        "max_memory_mb": 500
    }
}

# 配置多级缓存
l1_cache = UnifiedCache(**cache_configs["user_sessions"])  # L1缓存
l2_cache = UnifiedCache(**cache_configs["market_data"])   # L2缓存
```

### 2. 数据库优化

```python
# 配置连接池
db_config = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

# 使用连接池
from src.infrastructure.data.database_connection_pool import DatabaseConnectionPool
db_pool = DatabaseConnectionPool(**db_config)
connection = db_pool.get_connection()
```

### 3. 异步处理

```python
import asyncio
from src.core.async_task_processor import AsyncTaskProcessor

# 创建异步任务处理器
processor = AsyncTaskProcessor(max_workers=10)

# 提交异步任务
async def process_trade_async(trade_data):
    # 异步处理交易逻辑
    await asyncio.sleep(0.1)  # 模拟处理时间
    return {"status": "completed", "trade_id": trade_data["id"]}

# 批量处理交易
trades = [{"id": i, "symbol": f"SYMBOL{i}", "quantity": 100} for i in range(100)]
results = await processor.process_batch(process_trade_async, trades)
```

## 🔍 故障排除

### 常见问题

#### 1. 配置加载失败
```python
# 问题: 配置文件不存在或格式错误
try:
    config_manager = UnifiedConfigManager("config/local.yaml")
except FileNotFoundError:
    print("配置文件不存在，请检查路径")
except yaml.YAMLError as e:
    print(f"配置文件格式错误: {e}")

# 解决方案: 使用默认配置
config_manager = UnifiedConfigManager()
```

#### 2. 缓存性能问题
```python
# 问题: 缓存命中率低
cache_stats = cache.get_stats()
if cache_stats['hit_rate'] < 70:
    # 增加缓存容量
    cache.capacity *= 2
    # 或调整缓存策略
    cache = UnifiedCache(strategy=CacheStrategy.LFU, capacity=cache.capacity)
```

#### 3. 认证失败
```python
# 问题: 用户认证失败
auth_result = auth_service.authenticate_user(username, credentials)
if auth_result.status.name == "FAILED":
    if "密码错误" in auth_result.message:
        # 重置密码或检查密码复杂度
        pass
    elif "账户锁定" in auth_result.message:
        # 等待锁定解除或联系管理员
        pass
```

### 调试模式

```python
# 启用调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用监控调试
monitor = ApplicationMonitor("debug_mode")
monitor.set_debug_mode(True)

# 查看详细日志
logger = UnifiedLogger("debug", LogLevel.DEBUG)
logger.debug("启用调试模式", component="main", action="startup")
```

## 📈 监控和维护

### 健康检查

```python
from src.infrastructure.monitoring.health_checker import HealthChecker

# 创建健康检查器
health_checker = HealthChecker()

# 执行全面健康检查
health_status = health_checker.check_all_services()

for service, status in health_status.items():
    if status['healthy']:
        print(f"✅ {service}: 正常")
    else:
        print(f"❌ {service}: 异常 - {status['message']}")

# 生成健康报告
report = health_checker.generate_report()
print("健康检查报告:", report)
```

### 性能监控

```python
# 监控关键性能指标
performance_monitor = PerformanceMonitor()

@performance_monitor.monitor_function
def critical_business_function(data):
    # 业务逻辑...
    return result

# 定期检查性能
def check_performance():
    metrics = monitor.collect_performance_metrics()

    # 检查响应时间
    if metrics['avg_response_time'] > 1000:  # 1秒
        alert_system.alert("High Response Time",
                          f"平均响应时间过高: {metrics['avg_response_time']}ms")

    # 检查吞吐量
    if metrics['throughput_requests_per_second'] < 100:
        alert_system.alert("Low Throughput",
                          f"吞吐量不足: {metrics['throughput_requests_per_second']} req/s")

# 每分钟检查一次
import threading
def performance_check_loop():
    while True:
        check_performance()
        time.sleep(60)

threading.Thread(target=performance_check_loop, daemon=True).start()
```

## 🔗 相关文档

- [基础设施层架构设计](docs/architecture/infrastructure_layer_design.md)
- [安全模块详细文档](docs/api/security_api.md)
- [配置管理系统使用指南](docs/api/config_management_api.md)
- [监控告警最佳实践](docs/best_practices/monitoring_best_practices.md)
- [性能优化指南](docs/performance/performance_optimization_guide.md)

## 📞 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请通过以下方式联系：

- 📧 邮箱: support@rqa2025.com
- 💬 Slack: #rqa2025-support
- 📖 文档: docs/support/README.md
- 🐛 问题跟踪: github.com/rqa2025/issues

---

**版本**: 1.0.0 | **更新时间**: 2025-08-28 | **状态**: 活跃
