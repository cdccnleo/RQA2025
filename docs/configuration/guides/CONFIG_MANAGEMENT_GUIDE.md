# 配置管理模块使用指南

## 📋 概述

配置管理模块经过全面重构，现已提供统一、高效、可扩展的配置管理解决方案。本指南将详细介绍如何使用重构后的配置管理功能。

## 🏗️ 架构概览

### **核心组件**
- **UnifiedConfigManager**: 统一配置管理器，整合所有功能
- **UnifiedConfigCore**: 核心配置管理逻辑
- **ConfigSyncService**: 分布式配置同步服务
- **WebManagementService**: Web界面配置管理服务
- **ConfigValidatorFactory**: 统一验证器工厂

### **设计模式**
- **代理模式**: 统一配置管理器、同步服务、Web管理服务
- **工厂模式**: 验证器工厂、配置管理器工厂
- **策略模式**: 验证策略、缓存策略
- **观察者模式**: 配置变更通知

## 🚀 快速开始

### **基础使用**

```python
from src.infrastructure.config import UnifiedConfigManager, CachePolicy

# 创建配置管理器
config_manager = UnifiedConfigManager(
    cache_policy=CachePolicy.LRU,
    cache_size=1000,
    enable_encryption=True
)

# 设置配置
config_manager.set("database.host", "localhost")
config_manager.set("database.port", 5432)
config_manager.set("database.password", "secret_password")  # 自动加密

# 获取配置
host = config_manager.get("database.host")
password = config_manager.get("database.password")  # 自动解密
```

### **配置验证**

```python
from src.infrastructure.config import ConfigValidatorFactory, ValidatorType

# 创建验证器
validator_factory = ConfigValidatorFactory()
validator = validator_factory.create_validator(ValidatorType.JSON_SCHEMA)

# 定义配置模式
schema = {
    "type": "object",
    "properties": {
        "database": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "password": {"type": "string", "minLength": 8}
            },
            "required": ["host", "port", "password"]
        }
    }
}

# 验证配置
config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "password": "secret_password"
    }
}

result = validator.validate(config)
if result.is_valid:
    print("配置验证通过")
else:
    print(f"配置验证失败: {result.errors}")
```

### **配置同步**

```python
from src.infrastructure.config import ConfigSyncService

# 创建同步服务
sync_service = ConfigSyncService()

# 注册同步节点
sync_service.register_node("node1", "192.168.1.100", 8080)
sync_service.register_node("node2", "192.168.1.101", 8080)

# 同步配置到所有节点
config = {"database": {"host": "localhost", "port": 5432}}
result = sync_service.sync_config(config)

if result["success"]:
    print(f"配置同步成功，同步节点: {result['synced_nodes']}")
else:
    print(f"配置同步失败: {result['failed_nodes']}")

# 启动自动同步
sync_service.start_auto_sync()
```

### **Web管理**

```python
from src.infrastructure.config import WebManagementService

# 创建Web管理服务
web_service = WebManagementService()

# 用户认证
user_info = web_service.authenticate_user("admin", "admin123")
if user_info:
    print(f"用户认证成功: {user_info['username']}")

# 创建会话
session_id = web_service.create_session("admin")

# 获取配置树
config = {"database": {"host": "localhost", "port": 5432}}
tree = web_service.get_config_tree(config)

# 更新配置
success = web_service.update_config_value(config, "database.host", "new_host")
```

## 🔧 高级功能

### **缓存管理**

```python
# 获取缓存统计
cache_stats = config_manager.get_cache_stats()
print(f"缓存大小: {cache_stats['size']}")
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")

# 清除缓存
config_manager.clear_cache()
```

### **性能监控**

```python
# 获取性能指标
metrics = config_manager.get_performance_metrics()
print(f"平均响应时间: {metrics['avg_response_time']:.2f}ms")
print(f"总请求数: {metrics['total_requests']}")

# 重置性能指标
config_manager.reset_performance_metrics()
```

### **配置热重载**

```python
from src.infrastructure.config import UnifiedHotReload

# 创建热重载服务
hot_reload = UnifiedHotReload()

# 启动热重载
hot_reload.start_hot_reload()

# 监听配置文件
hot_reload.watch_file("config/app.json")

# 监听目录
hot_reload.watch_directory("config/")

# 获取热重载状态
status = hot_reload.get_hot_reload_status()
print(f"热重载状态: {status}")
```

### **分布式同步**

```python
# 获取同步状态
sync_status = sync_service.get_sync_status()
print(f"活跃节点数: {sync_status['active_nodes']}")
print(f"冲突数量: {sync_status['conflict_count']}")

# 获取同步历史
history = sync_service.get_sync_history(limit=10)
for entry in history:
    print(f"同步时间: {entry['timestamp']}, 成功节点: {entry['success_count']}")

# 检测冲突
conflicts = sync_service.get_conflicts()
if conflicts:
    print(f"发现 {len(conflicts)} 个冲突")
    # 解决冲突
    resolved = sync_service.resolve_conflicts(conflicts, strategy="merge")
```

## 🔒 安全功能

### **加密配置**

```python
# 启用加密
config_manager = UnifiedConfigManager(enable_encryption=True)

# 敏感配置自动加密
config_manager.set("api.key", "secret_api_key")
config_manager.set("database.password", "secret_password")

# 获取时自动解密
api_key = config_manager.get("api.key")  # 自动解密
```

### **权限控制**

```python
# 检查用户权限
has_permission = web_service.check_permission("admin", "write")
if has_permission:
    print("用户有写权限")
else:
    print("用户无写权限")

# 获取用户权限
permissions = web_service.get_permissions()
print(f"可用权限: {list(permissions.keys())}")
```

## 📊 监控和报告

### **配置监控**

```python
from src.infrastructure.config import ConfigMonitor

# 创建配置监控器
monitor = ConfigMonitor()

# 获取监控报告
report = monitor.get_monitoring_report()
print(f"配置变更次数: {report['change_count']}")
print(f"最后变更时间: {report['last_change_time']}")
```

### **审计日志**

```python
from src.infrastructure.config import ConfigAuditLogger

# 创建审计日志记录器
audit_logger = ConfigAuditLogger()

# 记录配置变更
audit_logger.log_config_change("admin", "database.host", "localhost", "new_host")

# 获取审计日志
logs = audit_logger.get_audit_logs(limit=10)
for log in logs:
    print(f"{log['timestamp']}: {log['user']} 修改了 {log['key']}")
```

## 🧪 测试支持

### **单元测试**

```python
import pytest
from unittest.mock import Mock
from src.infrastructure.config import UnifiedConfigManager

def test_config_manager():
    # 创建Mock对象
    mock_cache = Mock()
    mock_cache.get.return_value = None
    
    # 创建配置管理器
    config_manager = UnifiedConfigManager()
    
    # 测试设置和获取
    config_manager.set("test.key", "test_value")
    value = config_manager.get("test.key")
    
    assert value == "test_value"
```

### **集成测试**

```python
def test_config_sync():
    # 创建同步服务
    sync_service = ConfigSyncService()
    
    # 注册测试节点
    sync_service.register_node("test_node", "localhost", 8080)
    
    # 测试配置同步
    config = {"test": "value"}
    result = sync_service.sync_config(config)
    
    assert result["success"] == True
    assert "test_node" in result["synced_nodes"]
```

## 🚀 最佳实践

### **配置组织**

1. **分层配置**: 按环境、模块、功能分层组织配置
2. **命名规范**: 使用点分隔的层次结构命名
3. **敏感信息**: 敏感配置自动加密存储
4. **版本控制**: 重要配置变更记录版本

### **性能优化**

1. **缓存策略**: 根据访问模式选择合适的缓存策略
2. **批量操作**: 批量设置和获取配置
3. **异步处理**: 非关键配置使用异步处理
4. **监控指标**: 定期监控性能指标

### **安全考虑**

1. **权限控制**: 严格控制配置访问权限
2. **审计日志**: 记录所有配置变更
3. **加密存储**: 敏感配置必须加密
4. **传输安全**: 配置同步使用加密传输

## 🔧 故障排除

### **常见问题**

1. **导入错误**: 确保所有依赖模块正确安装
2. **配置验证失败**: 检查配置格式和验证规则
3. **同步失败**: 检查网络连接和节点状态
4. **性能问题**: 检查缓存配置和监控指标

### **调试技巧**

1. **启用详细日志**: 设置日志级别为DEBUG
2. **监控性能指标**: 定期检查性能报告
3. **验证配置**: 使用配置验证器检查配置格式
4. **测试连接**: 测试网络连接和节点可达性

## 📚 相关文档

- [配置管理模块架构重构完成报告](../architecture/config_final_architecture.md)
- [配置管理模块架构总结](../architecture/config_architecture_summary.md)
- [配置热重载功能实现报告](../../reports/architecture/hot_reload_implementation_report.md)

---

**版本**: 1.0.0  
**更新时间**: 2025-01-27  
**状态**: ✅ 重构完成，生产就绪