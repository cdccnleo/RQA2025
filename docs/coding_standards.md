# RQA2025基础设施层代码规范

## 📋 文档信息

- **版本**: v1.0
- **创建日期**: 2025年9月23日
- **适用范围**: 基础设施层所有Python代码
- **执行标准**: 强制性规范

## 🎯 规范目标

建立统一的代码编写标准，提升代码质量、可维护性和一致性。

### 质量目标
- **代码重复率**: <5%
- **圈复杂度**: ≤10
- **函数长度**: ≤40行
- **类方法数**: ≤20个
- **可维护性指数**: ≥60

## 📏 代码风格规范

### 1. 命名规范

#### 1.1 变量命名
```python
# ✅ 正确
user_name = "john"
user_age = 25
is_active = True
config_data = {}

# ❌ 错误
userName = "john"  # 混合大小写
u_name = "john"    # 缩写
isActive = True    # 混合大小写
data = {}          # 过于宽泛
```

#### 1.2 函数命名
```python
# ✅ 正确
def get_user_data():
def validate_config():
def process_cache_item():
def calculate_metrics():

# ❌ 错误
def GetUserData():      # 大驼峰
def get_user_data():    # 正确但上下文不清
def process():          # 过于宽泛
def calc():             # 缩写
```

#### 1.3 类命名
```python
# ✅ 正确
class CacheManager:
class ConfigValidator:
class PerformanceMonitor:

# ❌ 错误
class cacheManager:     # 小写开头
class Cache_Manager:    # 下划线分隔
class cache_manager:    # 全小写
```

#### 1.4 常量命名
```python
# ✅ 正确
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
CACHE_LEVELS = ['L1', 'L2', 'L3']

# ❌ 错误
maxRetryCount = 3      # 非常量风格
default_timeout = 30    # 非常量风格
```

### 2. 代码结构规范

#### 2.1 导入规范
```python
# ✅ 标准库导入
import os
import sys
from typing import Dict, List, Optional

# ✅ 第三方库导入
import redis
from prometheus_client import Counter

# ✅ 本地模块导入 (使用绝对导入)
from infrastructure.cache.core import UnifiedCacheManager
from infrastructure.config.validators import ConfigValidator

# ❌ 错误做法
import *                           # 通配符导入
from ..core import *               # 相对导入
import infrastructure.cache.core   # 过长导入
```

#### 2.2 类定义规范
```python
class CacheComponent(ICacheComponent):
    """
    缓存组件实现类。

    详细描述类的功能、用法和注意事项。
    """

    def __init__(self, component_id: int, config: Dict[str, Any]):
        """
        初始化缓存组件。

        Args:
            component_id: 组件唯一标识
            config: 配置字典

        Raises:
            ValueError: 配置无效时抛出
        """
        self.component_id = component_id
        self.config = config
        self._validate_config()

    def get_cache_item(self, key: str) -> Optional[Any]:
        """获取缓存项。"""
        pass

    def _validate_config(self) -> None:
        """验证配置参数。"""
        pass
```

#### 2.3 函数定义规范
```python
def get_cache_item(self, key: str) -> Optional[Any]:
    """
    获取缓存项。

    从缓存中检索指定键的值。如果键不存在或已过期，
    返回None。

    Args:
        key: 缓存键，必须为字符串类型

    Returns:
        缓存的值，None表示未找到

    Raises:
        TypeError: key不是字符串类型时抛出
    """
    if not isinstance(key, str):
        raise TypeError("缓存键必须是字符串类型")

    # 函数实现
    return self._cache_store.get(key)
```

### 3. 错误处理规范

#### 3.1 异常处理
```python
# ✅ 正确
try:
    result = self.redis_client.get(key)
    return pickle.loads(result)
except redis.ConnectionError as e:
    self.logger.error(f"Redis连接失败: {e}")
    raise CacheConnectionError(f"无法连接到Redis: {e}")
except pickle.PickleError as e:
    self.logger.error(f"数据反序列化失败: {e}")
    raise CacheSerializationError(f"数据格式错误: {e}")
except Exception as e:
    self.logger.error(f"未知错误: {e}")
    raise CacheError(f"缓存操作失败: {e}")

# ❌ 错误
try:
    # 危险操作
    pass
except:  # 捕获所有异常
    pass  # 静默处理
```

#### 3.2 自定义异常
```python
class CacheError(Exception):
    """缓存基础异常"""

    def __init__(self, message: str, error_code: str = "CACHE_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

class CacheConnectionError(CacheError):
    """缓存连接异常"""

    def __init__(self, message: str, host: str = "", port: int = 0):
        super().__init__(message, "CONNECTION_ERROR")
        self.host = host
        self.port = port
```

### 4. 文档规范

#### 4.1 模块文档
```python
"""
缓存管理器模块。

提供统一的缓存管理接口，支持多级缓存架构：
- 内存缓存 (L1)
- Redis缓存 (L2)
- 磁盘缓存 (L3)

主要组件：
- UnifiedCacheManager: 统一缓存管理器
- MultiLevelCache: 多级缓存实现
- CacheComponent: 缓存组件基类

作者: 专项修复小组
版本: 2.0.0
创建日期: 2025-09-23
"""

__version__ = "2.0.0"
__author__ = "专项修复小组"
```

#### 4.2 类型注解
```python
# ✅ 正确
from typing import Dict, List, Optional, Union, Any

def process_data(data: Dict[str, Any]) -> Optional[List[str]]:
    """
    处理数据字典。

    Args:
        data: 输入数据字典

    Returns:
        处理后的字符串列表，None表示处理失败
    """
    if not data:
        return None

    return [str(value) for value in data.values()]

# ❌ 错误
def process_data(data):  # 无类型注解
    return []
```

## 🏗️ 架构设计规范

### 1. 接口设计规范

#### 1.1 接口继承体系
```python
from abc import ABC, abstractmethod
from typing import Protocol

class IComponent(Protocol):
    """组件基础接口"""

    @property
    def component_name(self) -> str:
        """组件名称"""
        ...

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        ...

class ICacheComponent(IComponent):
    """缓存组件接口"""

    def get_cache_item(self, key: str) -> Any:
        """获取缓存项"""
        ...

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        ...
```

#### 1.2 实现类规范
```python
class MemoryCacheComponent(ICacheComponent):
    """内存缓存组件实现"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cache: Dict[str, Any] = {}

    @property
    def component_name(self) -> str:
        return f"MemoryCache_{id(self)}"

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """实现IComponent接口"""
        self.config.update(config)
        return True

    def get_cache_item(self, key: str) -> Any:
        """实现ICacheComponent接口"""
        return self._cache.get(key)

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """实现ICacheComponent接口"""
        self._cache[key] = value
        return True
```

### 2. 设计模式应用

#### 2.1 工厂模式
```python
class CacheFactory:
    """缓存工厂"""

    @staticmethod
    def create_cache(cache_type: str, config: Dict[str, Any]) -> ICacheComponent:
        """
        创建缓存实例。

        Args:
            cache_type: 缓存类型 ('memory', 'redis', 'disk')
            config: 配置字典

        Returns:
            缓存组件实例

        Raises:
            ValueError: 不支持的缓存类型
        """
        if cache_type == 'memory':
            return MemoryCacheComponent(config)
        elif cache_type == 'redis':
            return RedisCacheComponent(config)
        elif cache_type == 'disk':
            return DiskCacheComponent(config)
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}")
```

#### 2.2 单例模式 (谨慎使用)
```python
class ConfigManager:
    """配置管理器 (单例)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._config = {}
            self._initialized = True
```

## 🔍 质量检查规范

### 1. 代码重复检查
- **相似度阈值**: >80% 的代码块需要重构
- **重复次数阈值**: 出现3次以上的模式需要提取
- **忽略规则**: import语句、注释、文档字符串

### 2. 复杂度检查
- **圈复杂度**: ≤10 (函数)、≤15 (类)
- **函数长度**: ≤40行
- **嵌套深度**: ≤4层
- **参数数量**: ≤5个

### 3. 接口一致性检查
- **抽象方法**: 所有抽象方法必须实现
- **方法签名**: 参数数量和类型必须匹配
- **返回值类型**: 必须与接口定义一致

## 🧪 测试规范

### 1. 单元测试
```python
import unittest
from unittest.mock import Mock, patch

class TestCacheComponent(unittest.TestCase):
    """缓存组件测试"""

    def setUp(self):
        """测试前准备"""
        self.component = CacheComponent(component_id=1)

    def tearDown(self):
        """测试后清理"""
        pass

    def test_get_cache_item_existing_key(self):
        """测试获取存在的缓存项"""
        self.component.set_cache_item("test_key", "test_value")
        result = self.component.get_cache_item("test_key")
        self.assertEqual(result, "test_value")

    def test_get_cache_item_non_existing_key(self):
        """测试获取不存在的缓存项"""
        result = self.component.get_cache_item("non_existing")
        self.assertIsNone(result)
```

### 2. 集成测试
```python
class TestCacheIntegration(unittest.TestCase):
    """缓存集成测试"""

    def test_multi_level_cache_integration(self):
        """测试多级缓存集成"""
        config = {
            'levels': {
                'L1': {'type': 'memory', 'max_size': 100},
                'L2': {'type': 'redis', 'max_size': 1000}
            }
        }

        cache = MultiLevelCache(config)

        # 测试写入
        cache.set('key1', 'value1')

        # 测试读取
        value = cache.get('key1')
        self.assertEqual(value, 'value1')
```

## 📋 检查清单

### 代码提交前检查
- [ ] 所有函数都有类型注解
- [ ] 所有公共方法都有文档字符串
- [ ] 代码通过了复杂度检查
- [ ] 单元测试覆盖率 > 80%
- [ ] 无代码重复问题
- [ ] 接口实现正确

### 代码审查检查
- [ ] 命名符合规范
- [ ] 错误处理完善
- [ ] 日志记录合适
- [ ] 性能考虑充分
- [ ] 安全性检查通过

## 🔧 工具使用

### 质量检查工具
```bash
# 检查当前目录
python -m tools.quality_check .

# 检查特定模块
python -m tools.quality_check src/infrastructure/cache

# 生成报告
python -m tools.quality_check --reports html --output-dir reports .
```

### 代码格式化
```bash
# 使用black格式化
black src/infrastructure/

# 使用isort整理导入
isort src/infrastructure/
```

## 📞 联系与支持

- **规范维护者**: 专项修复小组
- **问题反馈**: 通过GitHub Issues提出
- **更新记录**: 查看CHANGELOG.md

---

**遵循这些规范，我们将共同维护高质量、一致的代码库！** 🎯
