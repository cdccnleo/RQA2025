# 基础设施层文档

## 📋 概述

基础设施层为整个RQA2025系统提供基础服务支持，包括配置管理、日志系统、监控系统、数据库访问、安全控制等核心功能。

## 🏗️ 架构组件

### 配置管理 (config/)
- **配置管理器**: 系统配置的统一管理
- **配置验证**: 配置参数的有效性验证
- **配置热更新**: 运行时配置的动态更新
- **配置版本控制**: 配置变更的版本管理

### 日志系统 (m_logging/)
- **现有日志系统**: 基础日志功能
  - LogManager: 日志管理器
  - LogSampler: 基础采样器
  - LogAggregator: 日志聚合器
  - JsonFormatter: JSON格式化器
  - QuantFilter: 量化过滤器
  - LogMetrics: 日志指标
  - ResourceManager: 资源管理器

- **增强日志系统**: 业务日志功能
  - EnhancedLogSampler: 增强采样器
  - BusinessLogManager: 业务日志管理器
  - LogCorrelationQuery: 关联查询器
  - UnifiedLoggingInterface: 统一日志接口

### 监控系统 (monitoring/)
- **健康检查**: 系统组件健康状态监控
- **性能监控**: 系统性能指标收集
- **资源监控**: CPU、内存、磁盘等资源监控
- **告警系统**: 异常情况告警机制

### 数据库 (database/)
- **数据连接管理**: 数据库连接池管理
- **查询优化**: SQL查询性能优化
- **数据迁移**: 数据库结构变更管理
- **备份恢复**: 数据备份和恢复策略

### 安全系统 (security/)
- **认证机制**: 用户身份认证
- **授权控制**: 访问权限管理
- **数据加密**: 敏感数据加密
- **审计日志**: 安全事件审计

## 🔧 核心功能

### 1. 配置管理
```python
from src.infrastructure.config import ConfigManager

# 获取配置管理器
config_manager = ConfigManager()

# 加载配置
config = config_manager.load_config('app.json')

# 获取配置值
db_url = config.get('database.url')
log_level = config.get('logging.level', 'INFO')
```

### 2. 日志系统
```python
from src.infrastructure.m_logging import (
    log_basic, log_business, log_debug, query_correlation
)

# 基础日志 (现有系统)
log_basic("system.startup", "INFO", "系统启动完成")

# 业务日志 (增强系统)
correlation_id = log_business(
    operation="order_processing",
    business_type=BusinessLogType.ORDER,
    message="订单处理开始"
)

# 调试日志 (增强系统)
trace_id = log_debug(
    operation="market_analysis",
    message="市场分析完成"
)

# 关联查询
result = query_correlation(
    CorrelationQuery(trace_id=trace_id)
)
```

### 3. 监控系统
```python
from src.infrastructure.monitoring import HealthChecker, PerformanceMonitor

# 健康检查
health_checker = HealthChecker()
status = health_checker.check_all_components()

# 性能监控
perf_monitor = PerformanceMonitor()
metrics = perf_monitor.collect_metrics()
```

### 4. 数据库访问
```python
from src.infrastructure.database import DatabaseManager

# 数据库管理器
db_manager = DatabaseManager()

# 执行查询
result = db_manager.execute_query("SELECT * FROM users")

# 事务管理
with db_manager.transaction():
    db_manager.execute("INSERT INTO logs (message) VALUES (?)", ["test"])
    db_manager.execute("UPDATE counters SET value = value + 1")
```

### 5. 安全控制
```python
from src.infrastructure.security import AuthManager, EncryptionManager

# 认证管理
auth_manager = AuthManager()
user = auth_manager.authenticate(username, password)

# 加密管理
encryption_manager = EncryptionManager()
encrypted_data = encryption_manager.encrypt(sensitive_data)
```

## 📊 性能指标

### 配置管理
- 配置加载时间: < 100ms
- 配置热更新延迟: < 50ms
- 配置验证成功率: > 99.9%

### 日志系统
- 日志写入延迟: < 10ms
- 采样率调整响应时间: < 100ms
- 关联查询响应时间: < 500ms

### 监控系统
- 健康检查频率: 30秒
- 性能指标收集间隔: 60秒
- 告警响应时间: < 10秒

### 数据库
- 连接池大小: 10-50
- 查询响应时间: < 100ms
- 事务成功率: > 99.9%

### 安全系统
- 认证响应时间: < 200ms
- 加密/解密延迟: < 50ms
- 审计日志完整性: 100%

## 🔧 配置示例

### 基础配置
```json
{
  "infrastructure": {
    "config": {
      "auto_reload": true,
      "validation": true,
      "backup_count": 5
    },
    "logging": {
      "basic": {
        "level": "INFO",
        "handlers": ["file", "console"]
      },
      "enhanced": {
        "sampling_rate": 0.3,
        "critical_business_types": ["order", "trade", "risk"]
      }
    },
    "monitoring": {
      "health_check_interval": 30,
      "performance_collection_interval": 60,
      "alert_channels": ["email", "slack"]
    },
    "database": {
      "connection_pool_size": 20,
      "max_retries": 3,
      "timeout": 30
    },
    "security": {
      "encryption_algorithm": "AES-256",
      "session_timeout": 3600,
      "audit_log_enabled": true
    }
  }
}
```

### 环境特定配置
```json
{
  "development": {
    "logging": {
      "level": "DEBUG",
      "sampling_rate": 1.0
    },
    "monitoring": {
      "health_check_interval": 10
    }
  },
  "production": {
    "logging": {
      "level": "WARNING",
      "sampling_rate": 0.1
    },
    "monitoring": {
      "health_check_interval": 60
    }
  }
}
```

## 🚀 部署指南

### 1. 环境准备
```bash
# 安装依赖
conda activate rqa  # 推荐，项目默认
# 如需base环境
# conda activate base
pip install -r requirements/infrastructure.txt

# 配置环境变量
export RQA_CONFIG_PATH=/path/to/config
export RQA_LOG_PATH=/path/to/logs
export RQA_DB_URL=postgresql://user:pass@host:port/db
```

### 2. 初始化配置
```python
from src.infrastructure.config import ConfigManager

config_manager = ConfigManager()
config_manager.initialize_default_config()
```

### 3. 启动服务
```python
from src.infrastructure import InfrastructureManager

# 启动基础设施服务
infra_manager = InfrastructureManager()
infra_manager.start_all_services()
```

### 4. 健康检查
```python
# 检查所有组件状态
status = infra_manager.check_health()
if status.is_healthy():
    print("所有基础设施组件运行正常")
else:
    print(f"发现问题: {status.issues}")
```

## 🧪 测试指南

### 单元测试
```bash
# 运行基础设施层测试
python -m pytest tests/unit/infrastructure/ -v

# 运行特定模块测试
python -m pytest tests/unit/infrastructure/test_config_manager.py -v
```

### 集成测试
```bash
# 运行基础设施集成测试
python -m pytest tests/integration/infrastructure/ -v
```

### 性能测试
```bash
# 运行性能测试
python -m pytest tests/performance/infrastructure/ -v
```

## 📈 监控和告警

### 关键指标
- **配置管理**: 配置加载时间、热更新成功率
- **日志系统**: 日志写入延迟、采样率、关联查询性能
- **监控系统**: 健康检查成功率、指标收集延迟
- **数据库**: 连接池使用率、查询响应时间
- **安全系统**: 认证成功率、加密性能

### 告警规则
- 配置加载失败
- 日志系统异常
- 健康检查失败
- 数据库连接异常
- 安全认证失败

## 🔗 相关文档

- [配置管理详细文档](./config/README.md)
- [日志系统详细文档](./m_logging/README.md)
- [监控系统详细文档](./monitoring/README.md)
- [数据库详细文档](./database/README.md)
- [安全系统详细文档](./security/README.md)

## 🏗️ 架构流程图

```mermaid
digraph TD
    A[Infrastructure入口] -> B[配置管理]
    A -> C[日志系统]
    A -> D[错误处理]
    A -> E[资源管理]
    A -> F[监控系统]
    B ->|ConfigManager| G[配置热更新/版本控制]
    C ->|LogManager| H[日志采样/聚合/指标]
    D ->|ErrorHandler| I[重试/告警]
    E ->|ResourceManager| J[CPU/GPU/内存/磁盘]
    F ->|SystemMonitor| K[健康检查/性能监控]
```

## 🔄 组件生命周期管理

- 基础设施层采用单例模式，所有核心组件在系统启动时统一初始化，并注册到全局上下文。
- 初始化顺序：配置管理 → 日志系统 → 错误处理 → 资源管理 → 监控系统，确保依赖关系正确。
- 支持优雅关闭（shutdown），各子系统资源可平滑释放，便于系统维护和升级。
- 部分高级功能（如配置热更新、GPU监控）视环境支持情况启用，具体以实际部署环境为准。

## ⚠️ 注意事项与已知限制

- 当前部分功能（如配置热更新监听、GPU监控、配置监听关闭）因兼容性或实现进度暂未完全实现，后续版本将逐步完善。
- 建议业务方在使用基础设施层时，关注各子系统的健康状态和异常告警，及时处理潜在风险。
- 对于非核心组件，建议支持降级启动，提升系统整体可用性。

---

**最后更新**: 2025-07-19  
**文档版本**: v1.0  
**维护状态**: ✅ 活跃维护中 