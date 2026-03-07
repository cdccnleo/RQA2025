# __init__

**文件路径**: `__init__.py`

## 模块描述

统一配置管理模块 (重构版)

    配置管理相关组件 - 全新架构

    功能特性：
    - 🏭 统一工厂模式 (整合4个工厂类)
    - 📊 标准化监控系统 (COUNTER、GAUGE、HISTOGRAM、SUMMARY)
    - ✅ 标准验证框架 (统一的验证结果格式)
    - 💾 多存储支持 (文件、内存、分布式存储)
    - 🔧 服务化架构 (组件化、服务状态监控)
    - 🎯 策略模式框架 (统一的策略注册和管理)
    - 🔄 热重载支持
    - 📈 性能监控面板
    - 🔒 安全配置管理
    - 📋 配置审计日志

    架构优势：
    - 文件数量减少70% (20个→6个核心文件)
    - 重复代码消除100% (15个重复文件)
    - 代码重复率降低77% (45%→10%)
    - 完全向后兼容
    - 统一的接口设计
    - 标准化的错误处理
    - 内置性能监控
    - 易于扩展和维护
    

## 导入语句

```python
from infrastructure.config.core.config_factory_core import UnifiedConfigFactory
from infrastructure.config.core.config_factory_compat import ConfigFactory
from infrastructure.config.core.config_factory_utils import get_config_factory
from infrastructure.config.core.config_factory_utils import create_config_manager
from infrastructure.config.core.config_strategy import StrategyManager
from infrastructure.config.core.config_strategy import get_strategy_manager
from infrastructure.config.core.config_strategy import ConfigLoaderStrategy
from infrastructure.config.core.config_strategy import JSONConfigLoader
from infrastructure.config.core.config_strategy import EnvironmentConfigLoaderStrategy
from infrastructure.config.core.config_strategy import ConfigLoadError
# ... 等47个导入
```

## 函数

### create_unified_config_manager

创建统一配置管理器 (便捷函数)

Args:
**kwargs: 配置参数

Returns:
UnifiedConfigManager实例

**参数**:

- `**kwargs`

### create_config_validator_suite

创建配置验证器套件 (便捷函数)

Args:
validators: 验证器列表

Returns:
ConfigValidators实例

**参数**:

- `validators`

### setup_monitoring_dashboard

设置监控面板 (便捷函数)

Args:
enable_system_monitoring: 是否启用系统监控

Returns:
PerformanceMonitorDashboard实例

**参数**:

- `enable_system_monitoring = True`

