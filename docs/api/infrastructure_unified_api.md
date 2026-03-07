# 统一基础设施模块 API 文档

## 概述

统一基础设施模块 (`src/infrastructure/unified_infrastructure/`) 提供了基础设施层的统一访问接口，通过工厂模式和依赖注入容器，实现了配置管理、监控系统和缓存系统的统一管理。

## 核心组件

### 1. InfrastructureManager - 统一入口管理器

`InfrastructureManager` 是整个基础设施层的统一入口，提供对所有核心组件的访问。

#### 初始化
```python
from src.infrastructure.unified_infrastructure import InfrastructureManager

# 创建基础设施管理器实例
infra_manager = InfrastructureManager()
```

#### 主要方法

##### 配置管理
```python
# 获取统一配置管理器（默认）
config_manager = infra_manager.get_config_manager()

# 获取特定类型的配置管理器
env_config = infra_manager.get_config_manager('environment')
cached_config = infra_manager.get_config_manager('cached')
distributed_config = infra_manager.get_config_manager('distributed')
encrypted_config = infra_manager.get_config_manager('encrypted')
hot_reload_config = infra_manager.get_config_manager('hot_reload')
```

##### 监控系统
```python
# 获取统一监控器（默认）
monitor = infra_manager.get_monitor()

# 获取特定类型的监控器
perf_monitor = infra_manager.get_monitor('performance')
business_monitor = infra_manager.get_monitor('business')
system_monitor = infra_manager.get_monitor('system')
automation_monitor = infra_manager.get_monitor('automation')
```

##### 缓存系统
```python
# 获取统一缓存管理器（默认）
cache = infra_manager.get_cache()

# 获取特定类型的缓存管理器
smart_cache = infra_manager.get_cache('smart')
memory_cache = infra_manager.get_cache('memory')
redis_cache = infra_manager.get_cache('redis')
```

##### 依赖注入服务
```python
# 注册服务
infra_manager.register_service('my_service', MyServiceClass)

# 获取服务
my_service = infra_manager.get_service('my_service')
```

### 2. 工厂模式组件

#### ConfigManagerFactory - 配置管理器工厂

```python
from src.infrastructure.core.config.unified_config_factory import ConfigManagerFactory

# 创建配置管理器
config_manager = ConfigManagerFactory.create_manager('unified')

# 注册新的配置管理器类型
ConfigManagerFactory.register_manager('custom', CustomConfigManager)

# 获取可用的配置管理器类型
available_types = ConfigManagerFactory.get_available_managers()
```

#### MonitorFactory - 监控系统工厂

```python
from src.infrastructure.core.monitoring.unified_monitor_factory import MonitorFactory

# 创建监控器
monitor = MonitorFactory.create_monitor('unified')

# 注册新的监控器类型
MonitorFactory.register_monitor('custom', CustomMonitor)
```

#### CacheFactory - 缓存系统工厂

```python
from src.infrastructure.core.cache.unified_cache_factory import CacheFactory

# 创建缓存管理器
cache_manager = CacheFactory.create_cache('unified')

# 注册新的缓存管理器类型
CacheFactory.register_cache('custom', CustomCacheManager)
```

### 3. 依赖注入容器

#### UnifiedDependencyContainer - 统一依赖注入容器

```python
from src.infrastructure.di.unified_dependency_container import (
    UnifiedDependencyContainer, 
    ServiceLifecycle
)

# 创建容器实例
container = UnifiedDependencyContainer()

# 注册服务（单例模式）
container.register('database', DatabaseService, ServiceLifecycle.SINGLETON)

# 注册服务（瞬态模式）
container.register('logger', LoggerService, ServiceLifecycle.TRANSIENT)

# 注册服务（作用域模式）
container.register('session', SessionService, ServiceLifecycle.SCOPED)

# 获取服务
db_service = container.get('database')
logger = container.get('logger')
```

#### 全局容器管理

```python
from src.infrastructure.di.unified_dependency_container import (
    get_container, 
    register_service, 
    get_service
)

# 获取全局容器实例
container = get_container()

# 使用便捷函数注册服务
register_service('my_service', MyService)

# 使用便捷函数获取服务
my_service = get_service('my_service')
```

## 使用示例

### 基础使用示例

```python
from src.infrastructure.unified_infrastructure import InfrastructureManager

def setup_infrastructure():
    """设置基础设施组件"""
    infra = InfrastructureManager()
    
    # 配置管理
    config = infra.get_config_manager()
    config.set('app.name', 'RQA2025')
    config.set('app.version', '1.0.0')
    
    # 监控系统
    monitor = infra.get_monitor()
    monitor.start_monitoring()
    
    # 缓存系统
    cache = infra.get_cache()
    cache.set('key', 'value', ttl=3600)
    
    # 注册业务服务
    infra.register_service('user_service', UserService)
    infra.register_service('order_service', OrderService)
    
    return infra

def use_infrastructure():
    """使用基础设施组件"""
    infra = InfrastructureManager()
    
    # 获取配置
    app_name = infra.get_config_manager().get('app.name')
    
    # 获取监控器
    monitor = infra.get_monitor()
    monitor.record_metric('api_calls', 1)
    
    # 获取缓存
    cache = infra.get_cache()
    cached_value = cache.get('key')
    
    # 获取业务服务
    user_service = infra.get_service('user_service')
    users = user_service.get_all_users()
```

### 高级配置示例

```python
def advanced_configuration():
    """高级配置示例"""
    infra = InfrastructureManager()
    
    # 使用加密配置管理器
    encrypted_config = infra.get_config_manager('encrypted')
    encrypted_config.set('db.password', 'secret_password')
    
    # 使用分布式配置管理器
    distributed_config = infra.get_config_manager('distributed')
    distributed_config.set('cluster.nodes', ['node1', 'node2', 'node3'])
    
    # 使用热重载配置管理器
    hot_reload_config = infra.get_config_manager('hot_reload')
    hot_reload_config.watch_file('config/app.yml')
    
    # 使用性能优化监控器
    perf_monitor = infra.get_monitor('performance')
    perf_monitor.enable_profiling()
    
    # 使用业务指标监控器
    business_monitor = infra.get_monitor('business')
    business_monitor.track_business_metric('revenue', 10000)
    
    # 使用智能缓存管理器
    smart_cache = infra.get_cache('smart')
    smart_cache.set_with_strategy('user_profile', user_data, strategy='lru')
```

### 依赖注入示例

```python
def dependency_injection_example():
    """依赖注入示例"""
    from src.infrastructure.di.unified_dependency_container import (
        get_container, 
        ServiceLifecycle
    )
    
    container = get_container()
    
    # 注册服务层次结构
    container.register('logger', LoggerService, ServiceLifecycle.SINGLETON)
    container.register('database', DatabaseService, ServiceLifecycle.SINGLETON)
    container.register('user_repository', UserRepository, ServiceLifecycle.TRANSIENT)
    container.register('user_service', UserService, ServiceLifecycle.TRANSIENT)
    
    # 获取服务（自动解析依赖）
    user_service = container.get('user_service')
    
    # 使用服务
    users = user_service.get_all_users()
```

## 配置选项

### 配置管理器类型

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `unified` | 统一配置管理器 | 默认选择，支持多种配置源 |
| `environment` | 环境变量配置管理器 | 容器化部署，环境配置 |
| `cached` | 缓存配置管理器 | 高频配置访问，性能优化 |
| `distributed` | 分布式配置管理器 | 集群部署，配置同步 |
| `encrypted` | 加密配置管理器 | 敏感配置，安全要求 |
| `hot_reload` | 热重载配置管理器 | 开发环境，配置实时更新 |

### 监控器类型

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `unified` | 统一监控器 | 默认选择，综合监控 |
| `performance` | 性能优化监控器 | 性能调优，瓶颈分析 |
| `business` | 业务指标监控器 | 业务监控，KPI跟踪 |
| `system` | 系统监控器 | 系统资源，健康检查 |
| `automation` | 自动化监控器 | CI/CD，自动化流程 |

### 缓存管理器类型

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `unified` | 统一缓存管理器 | 默认选择，智能缓存策略 |
| `smart` | 智能缓存管理器 | 自适应缓存，策略优化 |
| `memory` | 内存缓存管理器 | 高速缓存，临时数据 |
| `redis` | Redis缓存管理器 | 分布式缓存，持久化 |

## 错误处理

### 常见错误及解决方案

#### 1. 配置管理器类型不存在
```python
try:
    config = infra_manager.get_config_manager('invalid_type')
except ValueError as e:
    print(f"配置管理器类型不存在: {e}")
    # 使用默认类型
    config = infra_manager.get_config_manager()
```

#### 2. 服务未注册
```python
try:
    service = infra_manager.get_service('unregistered_service')
except KeyError as e:
    print(f"服务未注册: {e}")
    # 先注册服务
    infra_manager.register_service('unregistered_service', MyService)
    service = infra_manager.get_service('unregistered_service')
```

#### 3. 依赖注入失败
```python
try:
    service = container.get('service_with_dependencies')
except Exception as e:
    print(f"依赖注入失败: {e}")
    # 检查依赖是否已注册
    print(f"已注册的服务: {list(container._services.keys())}")
```

## 最佳实践

### 1. 服务注册顺序
```python
# 先注册基础服务
container.register('logger', LoggerService, ServiceLifecycle.SINGLETON)
container.register('config', ConfigService, ServiceLifecycle.SINGLETON)

# 再注册依赖服务
container.register('database', DatabaseService, ServiceLifecycle.SINGLETON)
container.register('cache', CacheService, ServiceLifecycle.SINGLETON)

# 最后注册业务服务
container.register('user_service', UserService, ServiceLifecycle.TRANSIENT)
```

### 2. 生命周期管理
```python
# 单例服务：全局共享，内存效率高
container.register('config', ConfigService, ServiceLifecycle.SINGLETON)

# 瞬态服务：每次获取新实例，适合无状态服务
container.register('validator', ValidatorService, ServiceLifecycle.TRANSIENT)

# 作用域服务：在特定作用域内共享，适合请求级别服务
container.register('request_context', RequestContext, ServiceLifecycle.SCOPED)
```

### 3. 错误处理和回退
```python
def get_config_with_fallback():
    """获取配置，支持回退策略"""
    try:
        # 尝试获取分布式配置
        config = infra_manager.get_config_manager('distributed')
        return config
    except Exception:
        try:
            # 回退到环境配置
            config = infra_manager.get_config_manager('environment')
            return config
        except Exception:
            # 最终回退到统一配置
            return infra_manager.get_config_manager('unified')
```

### 4. 性能优化
```python
def optimize_cache_usage():
    """优化缓存使用"""
    cache = infra_manager.get_cache('smart')
    
    # 设置缓存策略
    cache.set_cache_strategy('user_data', 'lru')
    cache.set_cache_strategy('config_data', 'ttl')
    
    # 预热缓存
    cache.warm_up(['user_data', 'config_data'])
    
    # 监控缓存命中率
    hit_rate = cache.get_hit_rate()
    if hit_rate < 0.8:
        cache.optimize_strategy()
```

## 测试指南

### 单元测试
```python
import pytest
from src.infrastructure.unified_infrastructure import InfrastructureManager

class TestInfrastructureManager:
    def setup_method(self):
        self.manager = InfrastructureManager()
    
    def test_config_manager_creation(self):
        config = self.manager.get_config_manager()
        assert config is not None
    
    def test_monitor_creation(self):
        monitor = self.manager.get_monitor()
        assert monitor is not None
    
    def test_cache_creation(self):
        cache = self.manager.get_cache()
        assert cache is not None
```

### 集成测试
```python
def test_service_registration_flow():
    """测试服务注册流程"""
    manager = InfrastructureManager()
    
    # 注册服务
    manager.register_service('test_service', TestService)
    
    # 获取服务
    service = manager.get_service('test_service')
    assert isinstance(service, TestService)
    
    # 验证服务功能
    result = service.test_method()
    assert result == 'test_result'
```

## 总结

统一基础设施模块通过以下方式简化了基础设施层的使用：

1. **统一入口**: 通过 `InfrastructureManager` 提供单一访问接口
2. **工厂模式**: 统一管理各种组件的创建和配置
3. **依赖注入**: 简化服务管理和依赖解析
4. **类型安全**: 支持类型检查和错误处理
5. **扩展性**: 易于添加新的组件类型和服务

使用此模块可以显著减少代码重复，提高代码质量，并简化基础设施组件的管理。
