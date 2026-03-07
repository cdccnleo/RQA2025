# RQA2025基础设施层接口使用指南

## 📋 文档信息

- **版本**: v1.0
- **创建日期**: 2025年9月23日
- **适用范围**: 基础设施层接口使用
- **受众**: 开发人员、架构师

## 🎯 指南目标

帮助开发人员正确理解和使用基础设施层的接口体系，确保接口实现的一致性和正确性。

## 🏗️ 接口体系架构

### 1. 接口继承层次

```
IComponent (Protocol)
├── component_name: str
├── initialize_component(config: Dict) -> bool
├── get_component_status() -> Dict
├── shutdown_component() -> None
└── health_check() -> bool

ICacheComponent (继承自 IComponent)
├── get_cache_item(key: str) -> Any
├── set_cache_item(key: str, value: Any, ttl?: int) -> bool
├── delete_cache_item(key: str) -> bool
├── has_cache_item(key: str) -> bool
├── clear_all_cache() -> bool
├── get_cache_size() -> int
└── get_cache_stats() -> Dict
```

### 2. 核心接口说明

#### 2.1 IComponent - 组件基础接口

所有基础设施组件的根接口，定义了组件的生命周期管理。

```python
from typing import Dict, Any
from infrastructure.cache.interfaces import IComponent

class MyComponent(IComponent):
    """自定义组件实现"""

    @property
    def component_name(self) -> str:
        """组件名称，必须唯一"""
        return "MyComponent_001"

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """
        初始化组件

        Args:
            config: 配置字典

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 初始化逻辑
            self._config = config
            self._initialized = True
            return True
        except Exception as e:
            return False

    def get_component_status(self) -> Dict[str, Any]:
        """
        获取组件状态

        Returns:
            Dict包含:
            - status: 'healthy'|'warning'|'error'
            - initialized: bool
            - last_check: datetime
            - error_count: int
        """
        return {
            'status': 'healthy' if self._initialized else 'error',
            'initialized': self._initialized,
            'last_check': datetime.now(),
            'error_count': getattr(self, '_error_count', 0)
        }

    def shutdown_component(self) -> None:
        """关闭组件，清理资源"""
        self._initialized = False
        # 清理逻辑

    def health_check(self) -> bool:
        """健康检查"""
        return self._initialized
```

#### 2.2 ICacheComponent - 缓存组件接口

缓存组件的标准接口，继承自IComponent。

```python
from typing import Any, Optional
from infrastructure.cache.interfaces import ICacheComponent

class MemoryCache(ICacheComponent):
    """内存缓存实现"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._config = {}
        self.initialize_component({})

    # IComponent接口实现
    @property
    def component_name(self) -> str:
        return "MemoryCache"

    # ICacheComponent接口实现
    def get_cache_item(self, key: str) -> Any:
        """获取缓存项"""
        return self._cache.get(key)

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        try:
            self._cache[key] = value
            return True
        except Exception:
            return False

    def delete_cache_item(self, key: str) -> bool:
        """删除缓存项"""
        return self._cache.pop(key, None) is not None

    def has_cache_item(self, key: str) -> bool:
        """检查是否存在"""
        return key in self._cache

    def clear_all_cache(self) -> bool:
        """清空缓存"""
        try:
            self._cache.clear()
            return True
        except Exception:
            return False

    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self._cache),
            'type': 'memory',
            'status': 'healthy'
        }
```

## 🔧 接口实现最佳实践

### 1. 初始化模式

#### 1.1 延迟初始化
```python
class LazyInitCache(ICacheComponent):
    """延迟初始化缓存"""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._cache = None
        self._initialized = False

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """延迟初始化"""
        self._config.update(config)
        # 不在这里初始化缓存
        self._initialized = True
        return True

    def _ensure_cache(self):
        """确保缓存已初始化"""
        if self._cache is None:
            self._cache = {}
            # 实际的初始化逻辑

    def get_cache_item(self, key: str) -> Any:
        self._ensure_cache()
        return self._cache.get(key)
```

#### 1.2 预热初始化
```python
class PreloadCache(ICacheComponent):
    """预热初始化缓存"""

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """预热初始化"""
        try:
            self._cache = {}

            # 预热数据
            preload_data = config.get('preload', {})
            for key, value in preload_data.items():
                self.set_cache_item(key, value)

            return True
        except Exception:
            return False
```

### 2. 错误处理模式

#### 2.1 优雅降级
```python
class ResilientCache(ICacheComponent):
    """具有容错能力的缓存"""

    def get_cache_item(self, key: str) -> Any:
        try:
            return self._unsafe_get(key)
        except Exception as e:
            self._log_error(f"获取缓存失败: {e}")
            return None

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            return self._unsafe_set(key, value, ttl)
        except Exception as e:
            self._log_error(f"设置缓存失败: {e}")
            return False

    def _unsafe_get(self, key: str) -> Any:
        """不安全的获取操作"""
        # 实际的缓存操作，可能抛出异常
        pass

    def _unsafe_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """不安全设置操作"""
        # 实际的缓存操作，可能抛出异常
        pass

    def _log_error(self, message: str):
        """记录错误"""
        logger = logging.getLogger(__name__)
        logger.error(message)
```

#### 2.2 熔断模式
```python
class CircuitBreakerCache(ICacheComponent):
    """带熔断器的缓存"""

    def __init__(self):
        self._failure_count = 0
        self._circuit_open = False
        self._last_failure_time = None

    def get_cache_item(self, key: str) -> Any:
        if self._circuit_open:
            if self._should_attempt_reset():
                return self._try_operation(lambda: self._unsafe_get(key))
            else:
                return None  # 熔断器开启，返回默认值

        return self._try_operation(lambda: self._unsafe_get(key))

    def _try_operation(self, operation):
        """尝试执行操作"""
        try:
            result = operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功回调"""
        self._failure_count = 0
        self._circuit_open = False

    def _on_failure(self):
        """失败回调"""
        self._failure_count += 1
        if self._failure_count >= 5:  # 连续失败阈值
            self._circuit_open = True
            self._last_failure_time = time.time()

    def _should_attempt_reset(self) -> bool:
        """是否应该尝试重置熔断器"""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time > 60  # 60秒后尝试
```

### 3. 性能优化模式

#### 3.1 批量操作
```python
class BatchCache(ICacheComponent):
    """支持批量操作的缓存"""

    def set_multiple(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        批量设置缓存项

        Args:
            items: 键值对字典
            ttl: 过期时间

        Returns:
            bool: 是否全部成功
        """
        success_count = 0
        for key, value in items.items():
            if self.set_cache_item(key, value, ttl):
                success_count += 1

        return success_count == len(items)

    def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存项

        Args:
            keys: 键列表

        Returns:
            Dict[str, Any]: 键值对字典，未找到的键值为None
        """
        results = {}
        for key in keys:
            results[key] = self.get_cache_item(key)
        return results
```

#### 3.2 异步操作
```python
import asyncio
from typing import Awaitable

class AsyncCache(ICacheComponent):
    """支持异步操作的缓存"""

    async def get_cache_item_async(self, key: str) -> Any:
        """异步获取缓存项"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_cache_item, key)

    async def set_cache_item_async(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """异步设置缓存项"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.set_cache_item, key, value, ttl)

    async def preload_data_async(self, data: Dict[str, Any]):
        """异步预热数据"""
        tasks = []
        for key, value in data.items():
            task = self.set_cache_item_async(key, value)
            tasks.append(task)

        await asyncio.gather(*tasks)
```

## 📊 接口一致性检查

### 1. 自动化检查

使用质量检查工具验证接口实现：

```bash
# 检查接口一致性
python -m tools.quality_check --checkers interface src/infrastructure/

# 生成详细报告
python -m tools.quality_check --reports html --checkers interface .
```

### 2. 常见问题

#### 2.1 方法签名不匹配
```python
# ❌ 错误：参数顺序不匹配
class WrongImpl(ICacheComponent):
    def set_cache_item(self, value: Any, key: str, ttl: int = None) -> bool:
        # 参数顺序错误
        pass

# ✅ 正确：与接口签名一致
class CorrectImpl(ICacheComponent):
    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass
```

#### 2.2 返回类型不匹配
```python
# ❌ 错误：返回类型不匹配
class WrongReturn(ICacheComponent):
    def get_cache_size(self) -> str:  # 应该是int
        return "100"

# ✅ 正确：返回类型匹配
class CorrectReturn(ICacheComponent):
    def get_cache_size(self) -> int:
        return 100
```

#### 2.3 缺少必需方法
```python
# ❌ 错误：缺少接口方法
class IncompleteImpl(ICacheComponent):
    def get_cache_item(self, key: str) -> Any:
        pass
    # 缺少 set_cache_item, delete_cache_item 等方法

# ✅ 正确：实现所有必需方法
class CompleteImpl(ICacheComponent):
    def get_cache_item(self, key: str) -> Any:
        pass

    def set_cache_item(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass

    def delete_cache_item(self, key: str) -> bool:
        pass

    # 实现所有其他必需方法...
```

## 🔄 接口版本管理

### 1. 版本兼容性

```python
class CacheInterfaceV1:
    """缓存接口 V1"""

    def get(self, key: str) -> Any:
        pass

    def set(self, key: str, value: Any) -> bool:
        pass

class CacheInterfaceV2(CacheInterfaceV1):
    """缓存接口 V2 - 向后兼容"""

    def get_with_ttl(self, key: str) -> Tuple[Any, Optional[int]]:
        """获取缓存项和TTL"""
        pass

    def set_with_options(self, key: str, value: Any, options: Dict[str, Any]) -> bool:
        """设置缓存项与选项"""
        pass

class BackwardCompatibleCache(CacheInterfaceV2):
    """向后兼容的缓存实现"""

    def get(self, key: str) -> Any:
        """V1接口实现"""
        return self.get_with_ttl(key)[0]

    def set(self, key: str, value: Any) -> bool:
        """V1接口实现"""
        return self.set_with_options(key, value, {})
```

### 2. 迁移策略

```python
class InterfaceMigrator:
    """接口迁移工具"""

    @staticmethod
    def migrate_v1_to_v2(v1_cache: 'CacheInterfaceV1') -> 'CacheInterfaceV2':
        """将V1接口迁移到V2"""

        class MigratedCache(CacheInterfaceV2):
            def __init__(self, v1_cache):
                self._v1_cache = v1_cache

            def get(self, key: str) -> Any:
                return self._v1_cache.get(key)

            def set(self, key: str, value: Any) -> bool:
                return self._v1_cache.set(key, value)

            def get_with_ttl(self, key: str) -> Tuple[Any, Optional[int]]:
                # V1没有TTL信息，返回None
                value = self._v1_cache.get(key)
                return (value, None)

            def set_with_options(self, key: str, value: Any, options: Dict[str, Any]) -> bool:
                # 忽略V2特有的选项
                return self._v1_cache.set(key, value)

        return MigratedCache(v1_cache)
```

## 📚 使用示例

### 1. 基本使用

```python
from infrastructure.cache.core.cache_components import CacheComponent

# 创建缓存组件
cache = CacheComponent(component_id=1, component_type='memory')

# 初始化
config = {'max_size': 1000, 'ttl': 300}
cache.initialize_component(config)

# 使用缓存
cache.set_cache_item('user:123', {'name': 'John', 'age': 30})
user = cache.get_cache_item('user:123')
print(user)  # {'name': 'John', 'age': 30}

# 检查状态
status = cache.get_component_status()
print(f"状态: {status['status']}")

# 清理资源
cache.shutdown_component()
```

### 2. 高级使用

```python
from infrastructure.cache.core import UnifiedCacheManager

# 创建统一缓存管理器
manager = UnifiedCacheManager()

# 配置多级缓存
config = {
    'multi_level': {
        'level': 'HYBRID',
        'memory_config': {'max_size': 1000},
        'redis_config': {'host': 'localhost', 'port': 6379},
        'file_config': {'cache_dir': '/tmp/cache'}
    }
}

manager.initialize(config)

# 使用多级缓存
manager.set('data:key1', 'value1')
value = manager.get('data:key1')

# 获取统计信息
stats = manager.get_cache_stats()
print(f"缓存命中率: {stats.get('hit_rate', 0):.2%}")
```

## 🔧 故障排除

### 1. 接口实现错误

**问题**: `TypeError: Can't instantiate abstract class`

**原因**: 没有实现所有必需的抽象方法

**解决**:
1. 检查是否实现了所有接口方法
2. 验证方法签名是否正确
3. 运行接口一致性检查

### 2. 类型注解错误

**问题**: 类型检查器报告类型不匹配

**解决**:
1. 检查方法的返回类型注解
2. 验证参数类型注解
3. 确保与接口定义一致

### 3. 初始化失败

**问题**: `initialize_component` 返回 `False`

**解决**:
1. 检查配置文件格式
2. 验证必要的依赖项
3. 查看组件日志输出

## 📞 支持与反馈

- **接口设计咨询**: 架构师团队
- **实现问题反馈**: 开发团队
- **文档更新建议**: 技术写作团队

---

**正确使用接口体系，是构建高质量基础设施的基础！** 🎯
