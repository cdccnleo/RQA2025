# 特征层配置开发指南

## 概述

本指南介绍了特征层配置系统的开发规范、最佳实践和注意事项，帮助开发者正确使用和扩展配置功能。

## 架构原则

### 1. 单一来源原则
- 所有配置类都在`src/features/core/config.py`统一定义
- 避免重复定义和循环导入
- 确保配置的一致性和可维护性

### 2. 分层架构
- **core层**: 配置类的定义和核心逻辑
- **特征层**: 配置的使用和业务逻辑
- **基础设施层**: 配置管理的底层支持

### 3. 向后兼容性
- 保持原有导入路径的兼容性
- 新增功能不影响现有代码
- 提供平滑的迁移路径

## 配置类开发规范

### 1. 基本结构
```python
@dataclass
class ConfigClass:
    """配置类 - 单一来源定义
    
    遵循单一来源原则，在core层统一定义。
    """
    
    # 配置字段定义
    param1: str = "default_value"
    param2: int = 10
    param3: bool = True
    
    # 自定义配置
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'param1': self.param1,
            'param2': self.param2,
            'param3': self.param3,
            'custom_config': self.custom_config
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigClass':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证逻辑
            if self.param2 <= 0:
                raise ValueError("param2必须大于0")
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
```

### 2. 枚举类型处理
```python
class ConfigType(Enum):
    """配置类型枚举"""
    TYPE1 = "type1"
    TYPE2 = "type2"

@dataclass
class ConfigClass:
    config_type: ConfigType = ConfigType.TYPE1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_type': self.config_type.value,
            # 其他字段...
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigClass':
        # 处理枚举类型
        if 'config_type' in config_dict and isinstance(config_dict['config_type'], str):
            config_dict['config_type'] = ConfigType(config_dict['config_type'])
        
        return cls(**config_dict)
```

### 3. 嵌套配置处理
```python
@dataclass
class SubConfig:
    """子配置类"""
    sub_param: int = 5

@dataclass
class MainConfig:
    """主配置类"""
    main_param: str = "default"
    sub_config: SubConfig = field(default_factory=SubConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'main_param': self.main_param,
            'sub_config': self.sub_config.to_dict()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MainConfig':
        # 处理嵌套配置
        if 'sub_config' in config_dict:
            config_dict['sub_config'] = SubConfig.from_dict(config_dict['sub_config'])
        
        return cls(**config_dict)
```

## 导入路径规范

### 1. 推荐导入方式
```python
# 从core层直接导入（推荐）
from src.features.core.config import ConfigClass

# 从特征层主模块导入（兼容）
from src.features import ConfigClass
```

### 2. 避免的导入方式
```python
# 避免：从具体文件导入
from src.features.config.feature_configs import ConfigClass

# 避免：使用相对导入（除非在同一包内）
from ..config import ConfigClass
```

### 3. 导入路径更新
当添加新的配置类时，需要更新以下文件：

#### 更新core层导出
```python
# src/features/core/__init__.py
from .config import (
    FeatureConfig,
    OrderBookConfig,
    NewConfigClass,  # 新增配置类
    # 其他配置类...
)

__all__ = [
    'FeatureConfig',
    'OrderBookConfig',
    'NewConfigClass',  # 新增配置类
    # 其他配置类...
]
```

#### 更新特征层导出
```python
# src/features/__init__.py
from .core.config import (
    FeatureConfig,
    OrderBookConfig,
    NewConfigClass,  # 新增配置类
    # 其他配置类...
)
```

## 配置验证规范

### 1. 基本验证规则
```python
def validate(self) -> bool:
    """验证配置有效性"""
    try:
        # 数值验证
        if self.depth <= 0:
            raise ValueError("深度必须大于0")
        
        # 范围验证
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("阈值必须在0-1之间")
        
        # 枚举验证
        if self.type not in ConfigType:
            raise ValueError(f"无效的配置类型: {self.type}")
        
        # 依赖验证
        if self.enable_feature and not self.feature_params:
            raise ValueError("启用特征时必须提供特征参数")
        
        return True
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False
```

### 2. 复杂验证规则
```python
def validate(self) -> bool:
    """复杂配置验证"""
    try:
        # 基本验证
        if not self._validate_basic():
            return False
        
        # 业务逻辑验证
        if not self._validate_business_logic():
            return False
        
        # 性能验证
        if not self._validate_performance():
            return False
        
        return True
    except Exception as e:
        print(f"配置验证失败: {e}")
        return False

def _validate_basic(self) -> bool:
    """基本参数验证"""
    return self.depth > 0 and self.batch_size > 0

def _validate_business_logic(self) -> bool:
    """业务逻辑验证"""
    return self.max_features >= self.min_features

def _validate_performance(self) -> bool:
    """性能参数验证"""
    return self.max_workers > 0 and self.timeout > 0
```

## 配置使用最佳实践

### 1. 配置创建
```python
# 使用默认配置
config = OrderBookConfig()

# 自定义配置
config = OrderBookConfig(
    depth=15,
    orderbook_type=OrderBookType.LEVEL2,
    enable_imbalance_analysis=True
)

# 从字典创建
config_dict = {
    'depth': 20,
    'orderbook_type': 'level3'
}
config = OrderBookConfig.from_dict(config_dict)
```

### 2. 配置验证
```python
# 验证配置
if config.validate():
    # 使用配置
    engine = FeatureEngine()
    features = engine.process_features(data, config)
else:
    # 处理验证失败
    print("配置无效，使用默认配置")
    config = OrderBookConfig()  # 使用默认配置
```

### 3. 配置组合
```python
# 组合多个配置
feature_config = DefaultConfigs.comprehensive_technical()
orderbook_config = OrderBookConfig(depth=10)

# 在引擎中使用
engine = FeatureEngine()
engine.set_feature_config(feature_config)
engine.set_orderbook_config(orderbook_config)
```

### 4. 配置序列化
```python
import json

# 保存配置
config = OrderBookConfig(depth=15)
config_dict = config.to_dict()

with open('config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

# 加载配置
with open('config.json', 'r') as f:
    config_dict = json.load(f)

config = OrderBookConfig.from_dict(config_dict)
```

## 错误处理

### 1. 配置验证错误
```python
try:
    config = OrderBookConfig(depth=-1)  # 无效配置
    if not config.validate():
        print("配置验证失败")
        # 使用默认配置
        config = OrderBookConfig()
except ValueError as e:
    print(f"配置错误: {e}")
    # 使用默认配置
    config = OrderBookConfig()
```

### 2. 导入错误
```python
try:
    from src.features.core.config import OrderBookConfig
except ImportError as e:
    print(f"导入错误: {e}")
    # 使用兼容导入
    from src.features import OrderBookConfig
```

### 3. 类型转换错误
```python
def safe_from_dict(config_dict: Dict[str, Any]) -> OrderBookConfig:
    """安全的字典转换"""
    try:
        return OrderBookConfig.from_dict(config_dict)
    except (TypeError, ValueError) as e:
        print(f"配置转换失败: {e}")
        # 返回默认配置
        return OrderBookConfig()
```

## 测试规范

### 1. 单元测试
```python
import pytest
from src.features.core.config import OrderBookConfig, OrderBookType

def test_orderbook_config_creation():
    """测试配置创建"""
    config = OrderBookConfig(depth=10)
    assert config.depth == 10
    assert config.orderbook_type == OrderBookType.LEVEL2

def test_orderbook_config_validation():
    """测试配置验证"""
    # 有效配置
    config = OrderBookConfig(depth=10)
    assert config.validate() == True
    
    # 无效配置
    config = OrderBookConfig(depth=-1)
    assert config.validate() == False

def test_orderbook_config_serialization():
    """测试配置序列化"""
    config = OrderBookConfig(depth=15, orderbook_type=OrderBookType.LEVEL3)
    
    # 转换为字典
    config_dict = config.to_dict()
    assert config_dict['depth'] == 15
    assert config_dict['orderbook_type'] == 'level3'
    
    # 从字典创建
    new_config = OrderBookConfig.from_dict(config_dict)
    assert new_config.depth == 15
    assert new_config.orderbook_type == OrderBookType.LEVEL3
```

### 2. 集成测试
```python
def test_config_integration():
    """测试配置集成"""
    from src.features import FeatureEngine, OrderBookConfig
    
    # 创建配置
    orderbook_config = OrderBookConfig(depth=10)
    
    # 在引擎中使用
    engine = FeatureEngine()
    engine.set_orderbook_config(orderbook_config)
    
    # 验证配置设置
    assert engine.orderbook_config == orderbook_config
    assert engine.orderbook_config.depth == 10
```

### 3. 性能测试
```python
import time

def test_config_performance():
    """测试配置性能"""
    start_time = time.time()
    
    # 创建大量配置
    configs = []
    for i in range(1000):
        config = OrderBookConfig(depth=i % 20 + 1)
        configs.append(config)
    
    creation_time = time.time() - start_time
    print(f"创建1000个配置耗时: {creation_time:.4f}秒")
    
    # 验证性能
    start_time = time.time()
    for config in configs:
        config.validate()
    
    validation_time = time.time() - start_time
    print(f"验证1000个配置耗时: {validation_time:.4f}秒")
```

## 性能优化

### 1. 配置缓存
```python
class ConfigCache:
    """配置缓存管理器"""
    
    def __init__(self):
        self._cache = {}
    
    def get_config(self, config_name: str, **kwargs) -> OrderBookConfig:
        """获取缓存的配置"""
        cache_key = f"{config_name}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._cache:
            self._cache[cache_key] = OrderBookConfig(**kwargs)
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

# 使用缓存
cache = ConfigCache()
config = cache.get_config("orderbook", depth=10)
```

### 2. 内存优化
```python
# 对于大量配置，使用字典而不是对象
config_dict = {
    'depth': 10,
    'orderbook_type': 'level2'
}

# 只在需要时创建对象
def create_config_when_needed(config_dict: Dict[str, Any]) -> OrderBookConfig:
    """按需创建配置对象"""
    return OrderBookConfig.from_dict(config_dict)
```

## 文档规范

### 1. 类文档
```python
@dataclass
class OrderBookConfig:
    """订单簿配置类 - 单一来源定义
    
    此配置类在core层定义，作为整个特征层的单一来源。
    所有其他模块都应该从core层导入此配置类。
    
    属性:
        depth: 订单簿深度，必须大于0
        orderbook_type: 订单簿类型，支持level1/level2/level3
        enable_imbalance_analysis: 是否启用不平衡分析
        imbalance_threshold: 不平衡阈值，范围0-1
    
    示例:
        >>> config = OrderBookConfig(depth=10)
        >>> if config.validate():
        ...     print("配置有效")
    """
```

### 2. 方法文档
```python
def validate(self) -> bool:
    """验证配置有效性
    
    验证规则:
    - depth必须大于0
    - update_frequency必须大于0
    - max_workers必须大于0
    - batch_size必须大于0
    
    返回:
        bool: True表示配置有效，False表示配置无效
    
    示例:
        >>> config = OrderBookConfig(depth=10)
        >>> config.validate()
        True
    """
```

## 迁移指南

### 1. 从旧版本迁移
```python
# 旧版本导入（仍然支持）
from src.features.config import OrderBookConfig

# 新版本导入（推荐）
from src.features.core.config import OrderBookConfig
# 或者
from src.features import OrderBookConfig
```

### 2. 配置更新
```python
# 旧版本配置
old_config = {
    'depth': 10,
    'type': 'level2'
}

# 新版本配置
new_config = OrderBookConfig.from_dict(old_config)
```

### 3. 验证更新
```python
# 旧版本：无验证
config = OrderBookConfig(depth=-1)  # 可能创建无效配置

# 新版本：有验证
config = OrderBookConfig(depth=-1)
if not config.validate():
    config = OrderBookConfig()  # 使用默认配置
```

## 总结

遵循本指南可以确保：

1. **代码质量**: 统一的配置管理，避免重复定义
2. **可维护性**: 清晰的架构设计，易于扩展和维护
3. **向后兼容**: 平滑的迁移路径，不影响现有代码
4. **性能优化**: 合理的缓存策略，提高运行效率
5. **测试覆盖**: 完善的测试规范，确保代码质量

通过遵循这些规范，可以构建高质量、可维护的特征层配置系统。 