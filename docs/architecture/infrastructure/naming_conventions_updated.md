# 基础设施层命名规范 2025

## 概述

本文档定义了RQA2025项目基础设施层的命名规范，确保代码的一致性和可维护性。

**版本**: 2025.1  
**适用范围**: `src/infrastructure` 所有代码文件  
**更新日期**: 2025年1月

## 1. 文件命名规范

### 1.1 Python文件命名

**规则**:
- 使用小写字母和下划线
- 文件名应清晰表达其功能
- 避免使用缩写，除非是广泛接受的缩写

**正确示例**:
```
unified_config_manager.py      # 统一配置管理器
database_connection_pool.py    # 数据库连接池
memory_cache_manager.py        # 内存缓存管理器
automation_monitor.py          # 自动化监控器
```

**错误示例**:
```
UnifiedConfigManager.py        # 使用了大写字母
db_conn_pool.py              # 使用了不明确的缩写
cache_mgr.py                 # 使用了不明确的缩写
```

### 1.2 目录命名

**规则**:
- 使用小写字母和下划线
- 目录名应反映其包含的模块类型
- 避免使用复数形式

**正确示例**:
```
config/           # 配置管理模块
database/         # 数据库管理模块
cache/            # 缓存管理模块
monitoring/       # 监控管理模块
interfaces/       # 接口定义模块
```

**错误示例**:
```
Config/           # 使用了大写字母
databases/        # 使用了复数形式
cache_manager/    # 过于具体的名称
```

### 1.3 特殊文件命名

**测试文件**:
```
test_unified_config_manager.py    # 测试文件前缀为test_
test_database_connection_pool.py  # 测试文件前缀为test_
```

**接口文件**:
```
icache_manager.py                 # 接口文件前缀为i
idatabase_manager.py              # 接口文件前缀为i
```

**适配器文件**:
```
postgresql_adapter.py             # 适配器文件后缀为_adapter
redis_adapter.py                  # 适配器文件后缀为_adapter
```

## 2. 类命名规范

### 2.1 类名

**规则**:
- 使用PascalCase（大驼峰命名法）
- 类名应清晰表达其功能
- 避免使用缩写

**正确示例**:
```python
class UnifiedConfigManager:        # 统一配置管理器
class DatabaseConnectionPool:      # 数据库连接池
class MemoryCacheManager:          # 内存缓存管理器
class AutomationMonitor:           # 自动化监控器
```

**错误示例**:
```python
class unifiedConfigManager:        # 使用了camelCase
class DBConnPool:                 # 使用了缩写
class cache_mgr:                  # 使用了snake_case
```

### 2.2 接口类命名

**规则**:
- 接口类名以I开头
- 使用PascalCase
- 清晰表达接口功能

**正确示例**:
```python
class IConfigManager:              # 配置管理接口
class IDatabaseManager:            # 数据库管理接口
class ICacheManager:               # 缓存管理接口
class IMonitor:                    # 监控接口
```

**错误示例**:
```python
class ConfigManagerInterface:      # 没有使用I前缀
class iConfigManager:              # 使用了小写i
class ConfigMgrInterface:          # 使用了缩写
```

### 2.3 枚举类命名

**规则**:
- 使用PascalCase
- 以Enum结尾（如果继承自Enum）
- 清晰表达枚举含义

**正确示例**:
```python
class ServiceStatus(Enum):         # 服务状态枚举
class ConfigScope(Enum):           # 配置作用域枚举
class CachePolicy(Enum):           # 缓存策略枚举
```

**错误示例**:
```python
class service_status(Enum):        # 使用了snake_case
class ConfigScopes(Enum):          # 使用了复数形式
class CachePolicies(Enum):         # 使用了复数形式
```

## 3. 方法命名规范

### 3.1 方法名

**规则**:
- 使用snake_case（小写字母和下划线）
- 方法名应清晰表达其功能
- 动词开头，表达动作

**正确示例**:
```python
def get_config(self, key: str) -> Any:           # 获取配置
def set_config(self, key: str, value: Any):      # 设置配置
def validate_config(self, config: Dict) -> bool:  # 验证配置
def clear_cache(self):                            # 清空缓存
```

**错误示例**:
```python
def getConfig(self, key: str) -> Any:            # 使用了camelCase
def setConfig(self, key: str, value: Any):       # 使用了camelCase
def validateConfig(self, config: Dict) -> bool:   # 使用了camelCase
def clearCache(self):                             # 使用了camelCase
```

### 3.2 私有方法命名

**规则**:
- 以单下划线开头
- 使用snake_case
- 清晰表达私有方法功能

**正确示例**:
```python
def _load_config(self):                           # 加载配置
def _validate_input(self, data: Any) -> bool:     # 验证输入
def _initialize_components(self):                  # 初始化组件
```

**错误示例**:
```python
def load_config(self):                            # 没有下划线前缀
def __load_config(self):                          # 使用了双下划线
def _LoadConfig(self):                            # 使用了PascalCase
```

### 3.3 特殊方法命名

**规则**:
- 遵循Python特殊方法命名规范
- 使用双下划线包围

**正确示例**:
```python
def __init__(self):                               # 初始化方法
def __str__(self) -> str:                         # 字符串表示
def __repr__(self) -> str:                        # 对象表示
def __enter__(self):                              # 上下文管理器入口
def __exit__(self, exc_type, exc_val, exc_tb):   # 上下文管理器出口
```

## 4. 变量命名规范

### 4.1 变量名

**规则**:
- 使用snake_case
- 变量名应清晰表达其含义
- 避免使用单字母变量名（除了循环变量）

**正确示例**:
```python
config_manager = UnifiedConfigManager()           # 配置管理器
database_connection = get_database_connection()   # 数据库连接
cache_size = 1000                                # 缓存大小
max_retry_count = 3                              # 最大重试次数
```

**错误示例**:
```python
configManager = UnifiedConfigManager()            # 使用了camelCase
db_conn = get_database_connection()              # 使用了缩写
cacheSize = 1000                                 # 使用了camelCase
maxRetryCount = 3                                # 使用了camelCase
```

### 4.2 常量命名

**规则**:
- 使用大写字母和下划线
- 清晰表达常量含义

**正确示例**:
```python
DEFAULT_CACHE_SIZE = 1000                        # 默认缓存大小
MAX_CONNECTION_POOL_SIZE = 50                    # 最大连接池大小
DEFAULT_TIMEOUT = 30                             # 默认超时时间
```

**错误示例**:
```python
defaultCacheSize = 1000                          # 使用了camelCase
max_connection_pool_size = 50                    # 使用了snake_case
DEFAULT_TIMEOUT_SECONDS = 30                     # 过于冗长
```

### 4.3 私有变量命名

**规则**:
- 以单下划线开头
- 使用snake_case

**正确示例**:
```python
self._config_cache = {}                          # 配置缓存
self._connection_pool = None                     # 连接池
self._monitoring_enabled = True                  # 监控启用状态
```

**错误示例**:
```python
self.config_cache = {}                           # 没有下划线前缀
self.__connection_pool = None                    # 使用了双下划线
self._ConnectionPool = None                      # 使用了PascalCase
```

## 5. 函数命名规范

### 5.1 函数名

**规则**:
- 使用snake_case
- 函数名应清晰表达其功能
- 动词开头，表达动作

**正确示例**:
```python
def get_unified_config_manager() -> UnifiedConfigManager:    # 获取统一配置管理器
def create_database_connection() -> Connection:              # 创建数据库连接
def validate_config_data(data: Dict) -> bool:               # 验证配置数据
def initialize_monitoring_system():                          # 初始化监控系统
```

**错误示例**:
```python
def getUnifiedConfigManager() -> UnifiedConfigManager:      # 使用了camelCase
def createDatabaseConnection() -> Connection:                # 使用了camelCase
def validateConfigData(data: Dict) -> bool:                 # 使用了camelCase
def initMonitoringSystem():                                  # 使用了缩写
```

### 5.2 工厂函数命名

**规则**:
- 以create_、get_、build_等动词开头
- 清晰表达创建的对象类型

**正确示例**:
```python
def create_config_manager() -> ConfigManager:               # 创建配置管理器
def get_database_adapter(db_type: str) -> DatabaseAdapter: # 获取数据库适配器
def build_cache_manager(cache_type: str) -> CacheManager:  # 构建缓存管理器
```

## 6. 模块命名规范

### 6.1 模块名

**规则**:
- 使用小写字母和下划线
- 模块名应清晰表达其功能
- 避免使用缩写

**正确示例**:
```python
# 在__init__.py中导入
from .unified_manager import UnifiedConfigManager
from .database_connection_pool import DatabaseConnectionPool
from .memory_cache_manager import MemoryCacheManager
```

**错误示例**:
```python
# 在__init__.py中导入
from .unifiedManager import UnifiedConfigManager
from .db_conn_pool import DatabaseConnectionPool
from .cache_mgr import MemoryCacheManager
```

## 7. 包命名规范

### 7.1 包名

**规则**:
- 使用小写字母和下划线
- 包名应清晰表达其功能
- 避免使用复数形式

**正确示例**:
```python
from src.infrastructure.config import UnifiedConfigManager
from src.infrastructure.database import UnifiedDatabaseManager
from src.infrastructure.cache import MemoryCacheManager
```

**错误示例**:
```python
from src.infrastructure.Config import UnifiedConfigManager
from src.infrastructure.databases import UnifiedDatabaseManager
from src.infrastructure.cache_manager import MemoryCacheManager
```

## 8. 测试文件命名规范

### 8.1 测试文件名

**规则**:
- 以test_开头
- 使用snake_case
- 对应被测试的模块名

**正确示例**:
```
test_unified_config_manager.py      # 测试统一配置管理器
test_database_connection_pool.py    # 测试数据库连接池
test_memory_cache_manager.py        # 测试内存缓存管理器
```

**错误示例**:
```
TestUnifiedConfigManager.py         # 使用了PascalCase
testUnifiedConfigManager.py         # 使用了camelCase
unified_config_manager_test.py      # 没有test_前缀
```

### 8.2 测试函数命名

**规则**:
- 以test_开头
- 使用snake_case
- 清晰表达测试内容

**正确示例**:
```python
def test_get_config_success():                              # 测试获取配置成功
def test_set_config_with_invalid_key():                     # 测试设置无效配置
def test_cache_manager_initialization():                    # 测试缓存管理器初始化
```

**错误示例**:
```python
def testGetConfigSuccess():                                 # 使用了camelCase
def test_set_config_with_invalid_key():                     # 测试设置无效配置
def TestCacheManagerInitialization():                       # 使用了PascalCase
```

## 9. 文档字符串规范

### 9.1 类文档字符串

**规则**:
- 使用简洁的描述
- 说明类的主要功能
- 使用中文描述

**正确示例**:
```python
class UnifiedConfigManager:
    """统一配置管理器
    
    提供统一的配置管理功能，支持多种配置源和验证机制。
    """
```

### 9.2 方法文档字符串

**规则**:
- 使用简洁的描述
- 说明参数和返回值
- 使用中文描述

**正确示例**:
```python
def get_config(self, key: str, default: Any = None) -> Any:
    """获取配置值
    
    Args:
        key: 配置键
        default: 默认值
        
    Returns:
        配置值
    """
```

## 10. 注释规范

### 10.1 行内注释

**规则**:
- 使用中文注释
- 简洁明了
- 在代码行上方或行尾

**正确示例**:
```python
# 初始化配置管理器
config_manager = UnifiedConfigManager()

# 获取数据库连接
connection = get_database_connection()  # 从连接池获取连接
```

### 10.2 块注释

**规则**:
- 使用中文注释
- 说明代码块的功能
- 使用三引号或#号

**正确示例**:
```python
"""
初始化基础设施组件
包括配置管理、数据库连接、缓存等
"""
config_manager = UnifiedConfigManager()
database_manager = UnifiedDatabaseManager()
cache_manager = MemoryCacheManager()
```

## 11. 实施建议

### 11.1 代码审查

- 在代码审查时检查命名规范
- 使用自动化工具检查命名
- 建立命名规范检查清单

### 11.2 重构计划

- 逐步重构不符合规范的代码
- 优先重构核心模块
- 保持向后兼容性

### 11.3 培训计划

- 团队培训命名规范
- 建立命名规范文档
- 定期更新规范

## 12. 总结

本命名规范旨在提高代码的可读性和可维护性。通过统一的命名规范，可以：

1. **提高代码可读性**: 清晰的命名使代码更容易理解
2. **减少维护成本**: 统一的规范减少学习成本
3. **提高开发效率**: 规范的命名提高开发速度
4. **增强团队协作**: 统一的规范便于团队协作

建议团队严格执行本规范，并在代码审查中重点关注命名规范。 