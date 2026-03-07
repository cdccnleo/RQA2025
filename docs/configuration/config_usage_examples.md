# RQA2025 配置管理使用示例

## 📋 文档概述

本文档为RQA2025项目的其他模块提供配置管理的具体使用示例，包括常见场景、最佳实践和代码示例。

**版本**: v3.5  
**最后更新**: 2025-01-27  
**维护者**: 配置管理团队  
**状态**: ✅ 活跃维护

---

## 🚀 快速开始

### 1. 基础使用示例

#### 1.1 创建配置管理器

```python
# 方式1: 直接创建
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

config_manager = UnifiedConfigManager(
    config_dir="config",
    env="production",
    enable_hot_reload=True,
    enable_distributed_sync=True
)

# 方式2: 使用工厂创建
from src.infrastructure.config import ConfigFactory

config_manager = ConfigFactory.create_complete_config_service(
    env="production",
    enable_hot_reload=True
)
```

#### 1.2 基本配置操作

```python
# 设置配置
config_manager.set("app.name", "RQA2025", ConfigScope.GLOBAL)
config_manager.set("database.host", "localhost", ConfigScope.INFRASTRUCTURE)
config_manager.set("features.ml_enabled", True, ConfigScope.FEATURES)

# 获取配置
app_name = config_manager.get("app.name", ConfigScope.GLOBAL)
db_host = config_manager.get("database.host", ConfigScope.INFRASTRUCTURE)
ml_enabled = config_manager.get("features.ml_enabled", ConfigScope.FEATURES, default=False)

print(f"应用名称: {app_name}")
print(f"数据库主机: {db_host}")
print(f"ML功能启用: {ml_enabled}")
```

---

## 📊 模块配置示例

### 1. 基础设施模块配置

#### 1.1 数据库配置

```python
# src/infrastructure/database/db_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class DatabaseConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_database_config(self):
        """获取数据库配置"""
        return {
            "host": self.config_manager.get("database.host", ConfigScope.INFRASTRUCTURE, "localhost"),
            "port": self.config_manager.get("database.port", ConfigScope.INFRASTRUCTURE, 5432),
            "name": self.config_manager.get("database.name", ConfigScope.INFRASTRUCTURE, "rqa2025"),
            "user": self.config_manager.get("database.user", ConfigScope.INFRASTRUCTURE, "postgres"),
            "password": self.config_manager.get("database.password", ConfigScope.INFRASTRUCTURE, ""),
            "pool_size": self.config_manager.get("database.pool_size", ConfigScope.INFRASTRUCTURE, 10),
            "max_overflow": self.config_manager.get("database.max_overflow", ConfigScope.INFRASTRUCTURE, 20)
        }
    
    def get_redis_config(self):
        """获取Redis配置"""
        return {
            "host": self.config_manager.get("redis.host", ConfigScope.INFRASTRUCTURE, "localhost"),
            "port": self.config_manager.get("redis.port", ConfigScope.INFRASTRUCTURE, 6379),
            "db": self.config_manager.get("redis.db", ConfigScope.INFRASTRUCTURE, 0),
            "password": self.config_manager.get("redis.password", ConfigScope.INFRASTRUCTURE, None)
        }

# 使用示例
db_config = DatabaseConfig()
db_settings = db_config.get_database_config()
redis_settings = db_config.get_redis_config()
```

#### 1.2 缓存配置

```python
# src/infrastructure/cache/cache_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class CacheConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_cache_config(self):
        """获取缓存配置"""
        return {
            "type": self.config_manager.get("cache.type", ConfigScope.INFRASTRUCTURE, "redis"),
            "ttl": self.config_manager.get("cache.ttl", ConfigScope.INFRASTRUCTURE, 3600),
            "max_size": self.config_manager.get("cache.max_size", ConfigScope.INFRASTRUCTURE, 10000),
            "enable_compression": self.config_manager.get("cache.enable_compression", ConfigScope.INFRASTRUCTURE, True)
        }
    
    def get_memory_cache_config(self):
        """获取内存缓存配置"""
        return {
            "max_entries": self.config_manager.get("cache.memory.max_entries", ConfigScope.INFRASTRUCTURE, 1000),
            "eviction_policy": self.config_manager.get("cache.memory.eviction_policy", ConfigScope.INFRASTRUCTURE, "lru")
        }
```

### 2. 数据模块配置

#### 2.1 数据源配置

```python
# src/data/data_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class DataConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_data_source_config(self):
        """获取数据源配置"""
        return {
            "primary_source": self.config_manager.get("data.primary_source", ConfigScope.DATA, "mysql"),
            "backup_source": self.config_manager.get("data.backup_source", ConfigScope.DATA, "postgresql"),
            "batch_size": self.config_manager.get("data.batch_size", ConfigScope.DATA, 1000),
            "timeout": self.config_manager.get("data.timeout", ConfigScope.DATA, 30)
        }
    
    def get_data_processing_config(self):
        """获取数据处理配置"""
        return {
            "parallel_workers": self.config_manager.get("data.processing.parallel_workers", ConfigScope.DATA, 4),
            "chunk_size": self.config_manager.get("data.processing.chunk_size", ConfigScope.DATA, 10000),
            "enable_validation": self.config_manager.get("data.processing.enable_validation", ConfigScope.DATA, True)
        }
```

#### 2.2 特征工程配置

```python
# src/features/feature_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class FeatureConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_feature_config(self):
        """获取特征配置"""
        return {
            "feature_store_enabled": self.config_manager.get("features.store_enabled", ConfigScope.FEATURES, True),
            "feature_cache_size": self.config_manager.get("features.cache_size", ConfigScope.FEATURES, 5000),
            "feature_ttl": self.config_manager.get("features.ttl", ConfigScope.FEATURES, 86400),
            "enable_feature_monitoring": self.config_manager.get("features.monitoring_enabled", ConfigScope.FEATURES, True)
        }
    
    def get_feature_engineering_config(self):
        """获取特征工程配置"""
        return {
            "window_size": self.config_manager.get("features.engineering.window_size", ConfigScope.FEATURES, 20),
            "lookback_period": self.config_manager.get("features.engineering.lookback_period", ConfigScope.FEATURES, 60),
            "feature_selection_method": self.config_manager.get("features.engineering.selection_method", ConfigScope.FEATURES, "correlation")
        }
```

### 3. 模型模块配置

#### 3.1 模型训练配置

```python
# src/models/model_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class ModelConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_model_config(self):
        """获取模型配置"""
        return {
            "model_type": self.config_manager.get("models.type", ConfigScope.MODELS, "xgboost"),
            "model_version": self.config_manager.get("models.version", ConfigScope.MODELS, "1.0.0"),
            "model_path": self.config_manager.get("models.path", ConfigScope.MODELS, "models/"),
            "enable_auto_retrain": self.config_manager.get("models.auto_retrain", ConfigScope.MODELS, True)
        }
    
    def get_training_config(self):
        """获取训练配置"""
        return {
            "epochs": self.config_manager.get("models.training.epochs", ConfigScope.MODELS, 100),
            "batch_size": self.config_manager.get("models.training.batch_size", ConfigScope.MODELS, 32),
            "learning_rate": self.config_manager.get("models.training.learning_rate", ConfigScope.MODELS, 0.01),
            "validation_split": self.config_manager.get("models.training.validation_split", ConfigScope.MODELS, 0.2)
        }
    
    def get_inference_config(self):
        """获取推理配置"""
        return {
            "batch_size": self.config_manager.get("models.inference.batch_size", ConfigScope.MODELS, 64),
            "timeout": self.config_manager.get("models.inference.timeout", ConfigScope.MODELS, 30),
            "enable_caching": self.config_manager.get("models.inference.enable_caching", ConfigScope.MODELS, True)
        }
```

### 4. 交易模块配置

#### 4.1 交易策略配置

```python
# src/trading/trading_config.py
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

class TradingConfig:
    def __init__(self):
        self.config_manager = UnifiedConfigManager(env="production")
    
    def get_trading_config(self):
        """获取交易配置"""
        return {
            "strategy": self.config_manager.get("trading.strategy", ConfigScope.TRADING, "momentum"),
            "risk_limit": self.config_manager.get("trading.risk_limit", ConfigScope.TRADING, 0.02),
            "position_size": self.config_manager.get("trading.position_size", ConfigScope.TRADING, 0.1),
            "enable_stop_loss": self.config_manager.get("trading.enable_stop_loss", ConfigScope.TRADING, True)
        }
    
    def get_risk_config(self):
        """获取风控配置"""
        return {
            "max_drawdown": self.config_manager.get("trading.risk.max_drawdown", ConfigScope.TRADING, 0.1),
            "max_position_size": self.config_manager.get("trading.risk.max_position_size", ConfigScope.TRADING, 0.5),
            "enable_circuit_breaker": self.config_manager.get("trading.risk.enable_circuit_breaker", ConfigScope.TRADING, True)
        }
    
    def get_execution_config(self):
        """获取执行配置"""
        return {
            "slippage_tolerance": self.config_manager.get("trading.execution.slippage_tolerance", ConfigScope.TRADING, 0.001),
            "execution_timeout": self.config_manager.get("trading.execution.timeout", ConfigScope.TRADING, 5),
            "retry_attempts": self.config_manager.get("trading.execution.retry_attempts", ConfigScope.TRADING, 3)
        }
```

---

## 🔧 高级使用示例

### 1. 配置热重载

```python
# 启用热重载的配置管理器
config_manager = UnifiedConfigManager(
    config_dir="config",
    env="production",
    enable_hot_reload=True
)

# 启动热重载
config_manager.start_hot_reload()

# 监控配置文件变更
def on_config_changed(key, old_value, new_value):
    print(f"配置变更: {key} = {old_value} -> {new_value}")
    # 重新加载相关模块
    reload_affected_modules(key)

# 添加观察者
config_manager.add_watcher("database.host", on_config_changed)
config_manager.add_watcher("trading.strategy", on_config_changed)

# 检查热重载状态
if config_manager.is_hot_reload_running():
    print("热重载已启动")
```

### 2. 配置验证

```python
# 定义配置验证规则
def validate_database_config(config):
    """验证数据库配置"""
    errors = []
    
    if not config.get("host"):
        errors.append("数据库主机不能为空")
    
    port = config.get("port", 0)
    if not (1024 <= port <= 65535):
        errors.append("数据库端口必须在1024-65535之间")
    
    if not config.get("password"):
        errors.append("数据库密码不能为空")
    
    return len(errors) == 0, errors

# 注册验证规则
config_manager.add_validation_rule("database", validate_database_config)

# 验证配置
is_valid, errors = config_manager.validate()
if not is_valid:
    print("配置验证失败:")
    for error in errors:
        print(f"  - {error}")
```

### 3. 配置版本管理

```python
# 创建配置版本
version_id = config_manager.create_version({
    "database": {"host": "localhost", "port": 5432},
    "trading": {"strategy": "momentum", "risk_limit": 0.02}
}, env="production")

print(f"创建配置版本: {version_id}")

# 获取版本历史
versions = config_manager.list_versions()
print(f"配置版本: {versions}")

# 回滚到指定版本
if config_manager.rollback(version_id):
    print(f"成功回滚到版本: {version_id}")
else:
    print("回滚失败")
```

### 4. 分布式配置同步

```python
# 启用分布式同步
config_manager = UnifiedConfigManager(
    config_dir="config",
    env="production",
    enable_distributed_sync=True
)

# 注册同步节点
config_manager.register_sync_node("node1", "192.168.1.100", 8080)
config_manager.register_sync_node("node2", "192.168.1.101", 8080)

# 设置配置并同步
config_manager.set("shared.config", "shared_value")
sync_result = config_manager.sync_config_to_nodes(["node1", "node2"])

if sync_result["success"]:
    print("配置同步成功")
else:
    print(f"配置同步失败: {sync_result['error']}")

# 检查同步状态
sync_status = config_manager.get_sync_status()
print(f"同步状态: {sync_status}")
```

### 5. 配置加密

```python
# 启用配置加密
config_manager = UnifiedConfigManager(
    config_dir="config",
    env="production",
    enable_encryption=True
)

# 设置敏感配置（会自动加密）
config_manager.set("database.password", "secret_password")
config_manager.set("api.key", "api_secret_key")

# 获取配置（会自动解密）
password = config_manager.get("database.password")
api_key = config_manager.get("api.key")

print(f"数据库密码: {password}")
print(f"API密钥: {api_key}")
```

---

## 📊 性能优化示例

### 1. 缓存优化

```python
# 配置缓存策略
config_manager = UnifiedConfigManager(
    config_dir="config",
    env="production",
    cache_policy=CachePolicy.LRU,
    cache_size=1000
)

# 获取缓存统计
cache_stats = config_manager.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
print(f"缓存大小: {cache_stats['total_entries']}")

# 清除缓存
config_manager.clear_cache()
```

### 2. 性能监控

```python
# 获取性能指标
metrics = config_manager.get_performance_metrics()
print(f"总操作数: {metrics['total_operations']}")
print(f"平均响应时间: {metrics['average_response_time']:.3f}ms")
print(f"最大响应时间: {metrics['max_response_time']:.3f}ms")

# 重置性能指标
config_manager.reset_performance_metrics()
```

---

## 🔒 安全配置示例

### 1. 访问控制

```python
# 设置访问权限
config_manager.set_access_control("admin", ["read", "write", "delete"])
config_manager.set_access_control("user", ["read"])
config_manager.set_access_control("guest", ["read"])

# 验证访问权限
def check_access(user, operation, key):
    permissions = config_manager.get_user_permissions(user)
    return operation in permissions

# 使用示例
if check_access("admin", "write", "database.password"):
    config_manager.set("database.password", "new_password")
else:
    print("权限不足")
```

### 2. 审计日志

```python
# 启用审计日志
config_manager.enable_audit_logging()

# 设置配置（会记录审计日志）
config_manager.set("database.host", "new_host", user="admin", comment="更新数据库主机")

# 获取审计日志
audit_logs = config_manager.get_audit_logs(limit=10)
for log in audit_logs:
    print(f"{log['timestamp']} - {log['user']} - {log['operation']} - {log['key']}")
```

---

## 🧪 测试配置示例

### 1. 测试环境配置

```python
# tests/conftest.py
import pytest
from src.infrastructure.config import UnifiedConfigManager, ConfigScope

@pytest.fixture(scope="session")
def test_config_manager():
    """创建测试配置管理器"""
    return UnifiedConfigManager(
        config_dir="tests/test_config",
        env="test",
        enable_hot_reload=False,
        enable_distributed_sync=False
    )

@pytest.fixture
def sample_config(test_config_manager):
    """设置测试配置"""
    test_config_manager.set("app.name", "TestApp", ConfigScope.GLOBAL)
    test_config_manager.set("database.host", "test-db", ConfigScope.INFRASTRUCTURE)
    test_config_manager.set("features.enabled", True, ConfigScope.FEATURES)
    return test_config_manager
```

### 2. 配置测试用例

```python
# tests/unit/test_config_integration.py
import pytest
from src.infrastructure.config import ConfigScope

class TestConfigIntegration:
    
    def test_database_config_integration(self, sample_config):
        """测试数据库配置集成"""
        from src.infrastructure.database.db_config import DatabaseConfig
        
        db_config = DatabaseConfig()
        db_settings = db_config.get_database_config()
        
        assert db_settings["host"] == "test-db"
        assert db_settings["port"] == 5432
    
    def test_trading_config_integration(self, sample_config):
        """测试交易配置集成"""
        from src.trading.trading_config import TradingConfig
        
        trading_config = TradingConfig()
        trading_settings = trading_config.get_trading_config()
        
        assert trading_settings["strategy"] == "momentum"
        assert trading_settings["risk_limit"] == 0.02
```

---

## 📋 最佳实践

### 1. 配置命名规范

```python
# 推荐的配置键命名
config_manager.set("app.name", "RQA2025")                    # 应用名称
config_manager.set("database.host", "localhost")             # 数据库主机
config_manager.set("trading.strategy", "momentum")           # 交易策略
config_manager.set("features.ml_enabled", True)              # 功能开关
config_manager.set("models.xgboost.parameters", {...})       # 模型参数
```

### 2. 配置作用域使用

```python
# 根据功能模块使用不同的作用域
config_manager.set("database.host", "localhost", ConfigScope.INFRASTRUCTURE)
config_manager.set("data.batch_size", 1000, ConfigScope.DATA)
config_manager.set("features.enabled", True, ConfigScope.FEATURES)
config_manager.set("models.type", "xgboost", ConfigScope.MODELS)
config_manager.set("trading.strategy", "momentum", ConfigScope.TRADING)
config_manager.set("app.debug", True, ConfigScope.GLOBAL)
```

### 3. 配置验证

```python
# 为重要配置添加验证
def validate_critical_config(config):
    """验证关键配置"""
    required_keys = ["database.host", "database.password", "api.key"]
    for key in required_keys:
        if not config.get(key):
            return False, f"缺少必需配置: {key}"
    return True, None

# 注册验证规则
config_manager.add_validation_rule("critical", validate_critical_config)
```

### 4. 错误处理

```python
# 配置获取错误处理
try:
    db_host = config_manager.get("database.host", ConfigScope.INFRASTRUCTURE)
    if not db_host:
        raise ValueError("数据库主机配置缺失")
except Exception as e:
    logger.error(f"配置获取失败: {e}")
    # 使用默认值或抛出异常
    db_host = "localhost"
```

---

## 📞 支持

如有使用问题，请联系配置管理团队或提交Issue。

**联系方式**:
- 邮箱: config-team@rqa2025.com
- 文档: docs/configuration/
- 代码: src/infrastructure/config/ 