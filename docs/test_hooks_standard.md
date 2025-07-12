# 测试钩子标准 (Test Hooks Standard)

## 概述

本文档定义了RQA2025项目中统一的依赖注入测试钩子标准，用于解决复杂依赖模块的测试问题。

## 背景

在轻量级量化交易模型中，许多模块依赖于复杂的服务（如ConfigManager、SecurityService等），这些依赖在测试时难以mock。通过引入测试钩子，我们可以：

1. **提高可测试性**：允许在测试时注入mock对象
2. **保持向后兼容**：不影响生产代码的正常使用
3. **统一测试模式**：建立一致的依赖注入标准

## 测试钩子设计原则

### 1. 可选参数模式
```python
def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
    """
    初始化模块
    
    Args:
        config: 系统配置
        config_manager: 可选的配置管理器实例，用于测试时注入mock对象
    """
    # 测试钩子：允许注入mock的ConfigManager
    if config_manager is not None:
        self.config_manager = config_manager
    else:
        self.config_manager = ConfigManager(config)
```

### 2. 依赖优先级
1. **测试注入的依赖**：如果提供了测试钩子参数，优先使用
2. **默认依赖**：否则使用正常的依赖初始化

### 3. 类型安全
- 使用`Optional[Type]`类型注解
- 在运行时检查依赖的有效性
- 提供清晰的错误信息

## 适用场景

### 需要测试钩子的模块

#### 1. 监控模块
- **PerformanceMonitor**: 依赖ConfigManager获取监控配置
- **AlertManager**: 依赖ConfigManager获取告警规则和通知配置
- **HealthChecker**: 依赖ConfigManager获取健康检查配置
- **SystemMonitor**: 依赖外部系统调用，需要mock

#### 2. 配置管理模块
- **DeploymentManager**: 依赖ConfigManager管理部署配置
- **ConfigManager**: 本身依赖SecurityService，需要mock

#### 3. 服务管理模块
- **DegradationManager**: 依赖ConfigManager、HealthChecker、CircuitBreaker
- **ServiceLauncher**: 依赖DeploymentManager

#### 4. 数据库模块
- **DatabaseManager**: 依赖文件系统读取配置
- **ConnectionPool**: 依赖外部数据库连接

## 实现方式

### 1. 构造函数注入
```python
class ExampleManager:
    def __init__(
        self, 
        config: Dict[str, Any], 
        config_manager: Optional[ConfigManager] = None,
        health_checker: Optional[HealthChecker] = None
    ):
        self.config = config
        
        # 测试钩子：允许注入mock的依赖
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config)
            
        if health_checker is not None:
            self.health_checker = health_checker
        else:
            self.health_checker = HealthChecker(config)
```

### 2. 方法注入
```python
def process_data(self, data: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
    """处理数据，支持注入mock的ConfigManager"""
    cm = config_manager or self.config_manager
    # 使用cm进行配置操作
```

### 3. 属性注入
```python
@property
def config_manager(self) -> ConfigManager:
    """获取配置管理器，支持测试时注入"""
    if hasattr(self, '_test_config_manager'):
        return self._test_config_manager
    return self._default_config_manager

@config_manager.setter
def config_manager(self, value: ConfigManager):
    """设置配置管理器，用于测试"""
    self._test_config_manager = value
```

## 测试用例标准

### 1. 基础测试结构
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.module import ExampleManager

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {
        'test_config': 'test_value'
    }
    return mock_cm

@pytest.fixture
def example_manager(mock_config_manager):
    """创建ExampleManager实例"""
    config = {'test': 'config'}
    return ExampleManager(config, config_manager=mock_config_manager)

class TestExampleManager:
    """ExampleManager测试类"""
    
    def test_init_with_test_hook(self, mock_config_manager):
        """测试使用测试钩子初始化"""
        config = {'test': 'config'}
        manager = ExampleManager(config, config_manager=mock_config_manager)
        
        assert manager.config_manager == mock_config_manager
    
    def test_init_without_test_hook(self):
        """测试不使用测试钩子初始化"""
        config = {'test': 'config'}
        with patch('src.infrastructure.module.ConfigManager') as mock_cm_class:
            mock_cm_instance = MagicMock()
            mock_cm_class.return_value = mock_cm_instance
            
            manager = ExampleManager(config)
            
            assert manager.config_manager == mock_cm_instance
            mock_cm_class.assert_called_once_with(config)
```

### 2. 配置加载测试
```python
def test_load_config_from_manager(self, example_manager, mock_config_manager):
    """测试从配置管理器加载配置"""
    # 清空现有配置
    example_manager.config.clear()
    
    # 重新加载配置
    example_manager._load_config()
    
    # 验证配置被正确加载
    assert 'test_config' in example_manager.config
    assert example_manager.config['test_config'] == 'test_value'
```

### 3. 依赖方法测试
```python
def test_method_with_mock_dependency(self, example_manager):
    """测试使用mock依赖的方法"""
    # 模拟依赖方法的行为
    example_manager.config_manager.get_config.return_value = {
        'method_config': 'method_value'
    }
    
    result = example_manager.some_method()
    
    assert result == 'expected_result'
    example_manager.config_manager.get_config.assert_called_once()
```

## 注释规范

### 1. 构造函数注释
```python
def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
    """
    初始化模块
    
    Args:
        config: 系统配置字典
        config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        
    Note:
        测试钩子：如果提供了config_manager参数，将使用该实例而不是创建新的ConfigManager。
        这允许在测试时注入mock对象，避免复杂的依赖mock操作。
    """
```

### 2. 测试钩子属性注释
```python
@property
def config_manager(self) -> ConfigManager:
    """
    获取配置管理器实例
    
    Returns:
        ConfigManager实例
        
    Note:
        测试钩子：如果设置了_test_config_manager，将返回该实例。
        这允许在测试时注入mock的ConfigManager。
    """
```

### 3. 方法注释
```python
def process_with_config(self, data: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
    """
    使用配置处理数据
    
    Args:
        data: 要处理的数据
        config_manager: 可选的配置管理器，用于测试时注入mock对象
        
    Returns:
        处理结果
        
    Note:
        测试钩子：如果提供了config_manager参数，将使用该实例而不是默认的配置管理器。
    """
```

## 最佳实践

### 1. 依赖注入顺序
1. **测试注入的依赖**：优先使用通过参数注入的mock对象
2. **默认依赖**：如果没有注入，使用正常的依赖初始化
3. **错误处理**：在依赖初始化失败时提供清晰的错误信息

### 2. 类型安全
- 使用`Optional[Type]`类型注解
- 在运行时检查依赖的有效性
- 提供默认值和错误处理

### 3. 测试隔离
- 每个测试用例使用独立的mock对象
- 避免全局mock和副作用
- 使用fixture确保测试隔离

### 4. 文档化
- 在构造函数中明确说明测试钩子的用途
- 在测试文件中添加详细的测试用例说明
- 更新相关文档和开发规范

## 已实现的模块

### 1. PerformanceMonitor ✅
- **文件**: `src/infrastructure/monitoring/performance_monitor.py`
- **测试**: `tests/unit/infrastructure/monitoring/test_performance_monitor.py`
- **钩子**: `config_manager: Optional[ConfigManager] = None`

### 2. DegradationManager ✅
- **文件**: `src/infrastructure/degradation_manager.py`
- **测试**: `tests/unit/infrastructure/degradation/test_degradation_manager.py`
- **钩子**: `config_manager`, `health_checker`, `circuit_breaker`

### 3. AlertManager ✅
- **文件**: `src/infrastructure/monitoring/alert_manager.py`
- **测试**: `tests/unit/infrastructure/monitoring/test_alert_manager.py`
- **钩子**: `config_manager: Optional[ConfigManager] = None`

### 4. HealthChecker ✅
- **文件**: `src/infrastructure/health/health_checker.py`
- **测试**: `tests/unit/infrastructure/health/test_health_checker.py`
- **钩子**: `config_manager: Optional[ConfigManager] = None`

### 5. DeploymentManager ✅
- **文件**: `src/infrastructure/config/deployment_manager.py`
- **测试**: `tests/unit/infrastructure/config/test_deployment_manager.py`
- **钩子**: `config_manager: Optional[ConfigManager] = None`

## 待实现的模块

### 1. SystemMonitor
- **文件**: `src/infrastructure/monitoring/system_monitor.py`
- **依赖**: 外部系统调用 (psutil, os.getloadavg)
- **钩子**: 需要mock系统调用

### 2. ApplicationMonitor
- **文件**: `src/infrastructure/monitoring/application_monitor.py`
- **依赖**: InfluxDB客户端
- **钩子**: 需要mock InfluxDB连接

### 3. DatabaseManager
- **文件**: `src/infrastructure/database/database_manager.py`
- **依赖**: 文件系统读取
- **钩子**: 需要mock文件读取操作

### 4. ServiceLauncher
- **文件**: `src/infrastructure/service_launcher.py`
- **依赖**: DeploymentManager
- **钩子**: 需要mock DeploymentManager

## 持续集成

### 1. 自动化检测
- 开发脚本自动检测未实现钩子的基础设施模块
- 在CI/CD流程中验证测试钩子的实现
- 监控测试覆盖率和测试通过率

### 2. 代码审查
- 在PR审查中检查测试钩子的实现
- 确保新增模块遵循测试钩子标准
- 验证测试用例的完整性和质量

### 3. 文档更新
- 持续更新测试钩子标准文档
- 记录新实现的模块和测试用例
- 维护最佳实践和常见问题

## 总结

通过实施统一的测试钩子标准，我们：

1. **提高了代码可测试性**：所有复杂依赖模块都支持依赖注入
2. **建立了统一的测试模式**：所有测试用例遵循相同的结构和规范
3. **提升了开发效率**：减少了mock的复杂性和测试维护成本
4. **保证了代码质量**：通过完善的测试覆盖确保系统稳定性

这个标准为RQA2025项目的长期维护和扩展提供了坚实的基础。 