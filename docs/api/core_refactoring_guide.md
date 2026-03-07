# 核心服务层重构API文档和使用指南

**版本**: 2.0  
**更新时间**: 2025-11-03  
**适用范围**: src/core 核心服务层  

---

## 📚 目录

1. [快速开始](#快速开始)
2. [BaseComponent API](#basecomponent-api)
3. [BaseAdapter API](#baseadapter-api)
4. [UnifiedBusinessAdapter API](#unifiedbusinessadapter-api)
5. [迁移指南](#迁移指南)
6. [最佳实践](#最佳实践)
7. [常见问题](#常见问题)

---

## 快速开始

### 安装和导入

```python
# 导入BaseComponent
from src.core.foundation.base_component import BaseComponent, ComponentFactory, component

# 导入BaseAdapter
from src.core.foundation.base_adapter import BaseAdapter, adapter

# 导入UnifiedBusinessAdapter
from src.core.integration.unified_business_adapters import (
    get_business_adapter,
    BusinessLayerType
)
```

### 创建一个简单的组件

```python
from src.core.foundation.base_component import BaseComponent, component

@component("my_component")
class MyComponent(BaseComponent):
    """我的组件"""
    
    def _do_initialize(self, config):
        # 组件初始化逻辑
        self.data = config.get('data', [])
        return True  # 返回True表示初始化成功
    
    def _do_execute(self, *args, **kwargs):
        # 组件执行逻辑
        operation = kwargs.get('operation')
        if operation == 'process':
            return self._process_data()
        return None
    
    def _process_data(self):
        # 具体业务逻辑
        return {'processed': len(self.data)}

# 使用组件
component = MyComponent("my_comp")
component.initialize({'data': [1, 2, 3]})
result = component.execute(operation='process')
print(result)  # {'processed': 3}
```

### 创建一个简单的适配器

```python
from src.core.foundation.base_adapter import BaseAdapter, adapter
from typing import Dict

@adapter("my_adapter", enable_cache=True)
class MyAdapter(BaseAdapter[Dict, Dict]):
    """我的适配器"""
    
    def _do_adapt(self, data: Dict) -> Dict:
        # 适配逻辑
        return {
            'original': data,
            'adapted': True,
            'timestamp': datetime.now()
        }
    
    def validate_input(self, data: Dict) -> bool:
        # 验证输入
        return 'required_field' in data

# 使用适配器
adapter = MyAdapter(name="my_adapter")
result = adapter.adapt({'required_field': 'value'})
print(result)  # {'original': {...}, 'adapted': True, 'timestamp': ...}
```

---

## BaseComponent API

### 类：BaseComponent

统一的组件基类，提供所有组件的公共功能。

#### 构造函数

```python
def __init__(self, name: str, config: Optional[Dict[str, Any]] = None)
```

**参数**:
- `name` (str): 组件名称
- `config` (Dict[str, Any], 可选): 组件配置

**示例**:
```python
component = MyComponent(name="test_component", config={'key': 'value'})
```

#### 核心方法

##### initialize

```python
def initialize(self, config: Dict[str, Any]) -> bool
```

初始化组件。

**参数**:
- `config` (Dict[str, Any]): 初始化配置

**返回**:
- `bool`: 初始化是否成功

**示例**:
```python
success = component.initialize({'database_url': 'postgresql://...'})
if success:
    print("组件初始化成功")
```

##### execute

```python
def execute(self, *args, **kwargs) -> Any
```

执行组件功能。

**参数**:
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回**:
- `Any`: 执行结果

**异常**:
- `RuntimeError`: 如果组件未初始化

**示例**:
```python
result = component.execute(operation='process', data=[1, 2, 3])
```

##### get_info

```python
def get_info(self) -> Dict[str, Any]
```

获取组件信息。

**返回**:
- `Dict[str, Any]`: 包含组件状态、配置等信息的字典

**示例**:
```python
info = component.get_info()
print(f"组件状态: {info['status']}")
print(f"创建时间: {info['created_at']}")
```

#### 状态管理方法

##### get_status

```python
def get_status() -> ComponentStatus
```

获取组件当前状态。

**返回**:
- `ComponentStatus`: 组件状态枚举值

##### is_initialized

```python
def is_initialized() -> bool
```

检查组件是否已初始化。

**返回**:
- `bool`: 是否已初始化

##### reset

```python
def reset()
```

重置组件状态。

#### 需要子类实现的方法

##### _do_initialize

```python
def _do_initialize(self, config: Dict[str, Any]) -> bool
```

子类实现具体的初始化逻辑。

**参数**:
- `config` (Dict[str, Any]): 初始化配置

**返回**:
- `bool`: 初始化是否成功

**示例**:
```python
def _do_initialize(self, config):
    self.db_connection = connect_to_db(config['db_url'])
    return self.db_connection is not None
```

##### _do_execute

```python
@abstractmethod
def _do_execute(self, *args, **kwargs) -> Any
```

子类实现具体的执行逻辑（必须实现）。

**参数**:
- `*args`: 位置参数
- `**kwargs`: 关键字参数

**返回**:
- `Any`: 执行结果

**示例**:
```python
def _do_execute(self, *args, **kwargs):
    operation = kwargs.get('operation')
    if operation == 'query':
        return self.db_connection.query(kwargs['sql'])
    return None
```

---

## BaseAdapter API

### 类：BaseAdapter[InputType, OutputType]

统一的适配器基类，提供所有适配器的公共功能。

#### 构造函数

```python
def __init__(
    self,
    name: str,
    config: Optional[Dict[str, Any]] = None,
    enable_cache: bool = False
)
```

**参数**:
- `name` (str): 适配器名称
- `config` (Dict[str, Any], 可选): 配置参数
- `enable_cache` (bool): 是否启用缓存

**示例**:
```python
adapter = MyAdapter(
    name="data_adapter",
    config={'timeout': 30},
    enable_cache=True
)
```

#### 核心方法

##### adapt

```python
def adapt(self, data: InputType) -> OutputType
```

适配数据（主要入口）。

**参数**:
- `data` (InputType): 输入数据

**返回**:
- `OutputType`: 适配后的输出数据

**异常**:
- `ValueError`: 输入数据无效
- `RuntimeError`: 适配过程失败

**示例**:
```python
input_data = {'raw_value': 100}
output_data = adapter.adapt(input_data)
```

##### validate_input

```python
def validate_input(self, data: InputType) -> bool
```

验证输入数据。

**参数**:
- `data` (InputType): 输入数据

**返回**:
- `bool`: 数据是否有效

**示例**:
```python
def validate_input(self, data: Dict) -> bool:
    return 'required_field' in data and data['required_field'] is not None
```

##### get_stats

```python
def get_stats() -> Dict[str, Any]
```

获取适配器统计信息。

**返回**:
- `Dict[str, Any]`: 包含成功次数、错误次数等统计信息

**示例**:
```python
stats = adapter.get_stats()
print(f"成功率: {stats['success_rate']}")
print(f"缓存大小: {stats['cache_size']}")
```

#### 缓存管理

##### clear_cache

```python
def clear_cache()
```

清空缓存。

**示例**:
```python
adapter.clear_cache()
```

#### 健康检查

##### is_healthy

```python
def is_healthy() -> bool
```

检查适配器健康状态。

**返回**:
- `bool`: 是否健康

**示例**:
```python
if adapter.is_healthy():
    print("适配器运行正常")
```

#### 需要子类实现的方法

##### _do_adapt

```python
@abstractmethod
def _do_adapt(self, data: InputType) -> OutputType
```

子类实现具体的适配逻辑（必须实现）。

**参数**:
- `data` (InputType): 预处理后的输入数据

**返回**:
- `OutputType`: 适配后的输出数据

**示例**:
```python
def _do_adapt(self, data: Dict) -> Dict:
    return {
        'value': data['raw_value'] * 100,
        'currency': 'USD',
        'formatted': f"${data['raw_value']}"
    }
```

##### _preprocess (可选)

```python
def _preprocess(self, data: InputType) -> InputType
```

数据预处理。

**示例**:
```python
def _preprocess(self, data: Dict) -> Dict:
    # 清理和标准化数据
    cleaned_data = data.copy()
    cleaned_data['value'] = str(data['value']).strip()
    return cleaned_data
```

##### _postprocess (可选)

```python
def _postprocess(self, data: OutputType) -> OutputType
```

数据后处理。

**示例**:
```python
def _postprocess(self, data: Dict) -> Dict:
    # 添加元数据
    data['processed_at'] = datetime.now()
    return data
```

---

## UnifiedBusinessAdapter API

### 函数：get_business_adapter

```python
def get_business_adapter(
    layer_type: BusinessLayerType,
    config: Optional[Dict[str, Any]] = None
) -> UnifiedBusinessAdapter
```

获取业务层适配器的便捷函数。

**参数**:
- `layer_type` (BusinessLayerType): 业务层类型
- `config` (Dict[str, Any], 可选): 配置参数

**返回**:
- `UnifiedBusinessAdapter`: 适配器实例

**示例**:
```python
from src.core.integration.unified_business_adapters import (
    get_business_adapter,
    BusinessLayerType
)

# 获取交易层适配器
trading_adapter = get_business_adapter(BusinessLayerType.TRADING)

# 获取基础设施服务
services = trading_adapter.get_infrastructure_services()
logger = services['logger']
cache = services['cache_manager']

# 执行健康检查
health = trading_adapter.health_check()
print(f"适配器状态: {health['status']}")
```

### BusinessLayerType 枚举

```python
class BusinessLayerType(Enum):
    DATA = "data"
    FEATURES = "features"
    TRADING = "trading"
    RISK = "risk"
    MODELS = "models"
    ML = "ml"
    STRATEGY = "strategy"
    ENGINE = "engine"
    HEALTH = "health"
```

---

## 迁移指南

### 从旧组件迁移到BaseComponent

#### 步骤1：识别旧代码模式

**旧代码**:
```python
# container_components.py
import logging
logger = logging.getLogger(__name__)

class ComponentFactory:  # 重复定义
    def __init__(self):
        self._components = {}
    # ... 30行重复代码

class IContainerComponent(ABC):
    @abstractmethod
    def get_info(self): pass
    
class ContainerComponent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = {}
    
    def initialize(self, config):
        # 重复的初始化逻辑
        pass
```

#### 步骤2：重构为BaseComponent

**新代码**:
```python
from src.core.foundation.base_component import BaseComponent, component

@component("container")
class ContainerComponent(BaseComponent):
    def _do_initialize(self, config):
        # 只需实现特定逻辑
        self.container_data = config.get('data', {})
        return True
    
    def _do_execute(self, *args, **kwargs):
        # 只需实现业务逻辑
        operation = kwargs.get('operation')
        return self._handle_operation(operation)
```

#### 步骤3：更新调用代码

**旧调用**:
```python
component = ContainerComponent()
component.initialize({'data': {}})
# 直接调用方法
component.some_method()
```

**新调用**:
```python
component = ContainerComponent("container")
component.initialize({'data': {}})
# 通过execute调用
result = component.execute(operation='some_operation')
```

### 从旧适配器迁移到BaseAdapter

#### 步骤1：识别重复代码

**旧代码**:
```python
class TradingAdapter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)  # 重复
        self.error_handler = ErrorHandler()  # 重复
        self._stats = {'success': 0, 'error': 0}  # 重复
    
    def adapt(self, data):
        try:
            if not self.validate(data):  # 重复的验证逻辑
                raise ValueError("Invalid")
            
            result = self._do_adapt(data)
            self._stats['success'] += 1  # 重复的统计
            return result
        except Exception as e:
            self.logger.error(f"Error: {e}")  # 重复的错误处理
            self._stats['error'] += 1
            raise
```

#### 步骤2：重构为BaseAdapter

**新代码**:
```python
from src.core.foundation.base_adapter import BaseAdapter, adapter

@adapter("trading", enable_cache=True)
class TradingAdapter(BaseAdapter[Dict, Dict]):
    def _do_adapt(self, data: Dict) -> Dict:
        # 只需实现核心适配逻辑
        return {
            'symbol': data['symbol'].upper(),
            'price': Decimal(str(data['price'])),
            'quantity': int(data['quantity'])
        }
    
    def validate_input(self, data: Dict) -> bool:
        # 只需实现特定验证
        return all(k in data for k in ['symbol', 'price', 'quantity'])
```

---

## 最佳实践

### 1. 使用装饰器

```python
@component("my_component")
class MyComponent(BaseComponent):
    pass

@adapter("my_adapter", enable_cache=True)
class MyAdapter(BaseAdapter):
    pass
```

### 2. 错误处理

```python
class MyAdapter(BaseAdapter):
    def _do_adapt(self, data):
        # BaseAdapter会自动处理异常
        return self.process(data)
    
    def _handle_error(self, data, error):
        # 自定义错误恢复
        self._logger.warning(f"适配失败，使用默认值: {error}")
        return {'default': True}
```

### 3. 使用缓存

```python
# 启用缓存
adapter = MyAdapter(enable_cache=True)

# 相同输入会从缓存获取
result1 = adapter.adapt(data)  # 执行适配
result2 = adapter.adapt(data)  # 从缓存获取
```

### 4. 监控和统计

```python
# 获取组件信息
info = component.get_info()
print(f"状态: {info['status']}")

# 获取适配器统计
stats = adapter.get_stats()
print(f"成功率: {stats['success_rate']}")
print(f"总调用: {stats['total_count']}")
```

### 5. 使用ComponentFactory

```python
factory = ComponentFactory()

# 批量创建组件
for name, ComponentClass in components.items():
    factory.create_component(name, ComponentClass, config)

# 统一管理
all_components = factory.get_all_components()
```

---

## 常见问题

### Q1: 如何处理异步操作？

**A**: 在`_do_execute`或`_do_adapt`中使用async/await：

```python
class AsyncComponent(BaseComponent):
    async def _do_execute_async(self, *args, **kwargs):
        result = await self.async_operation()
        return result
    
    def _do_execute(self, *args, **kwargs):
        import asyncio
        return asyncio.run(self._do_execute_async(*args, **kwargs))
```

### Q2: 如何禁用缓存？

**A**: 在创建适配器时设置`enable_cache=False`：

```python
adapter = MyAdapter(enable_cache=False)
```

或者动态清空缓存：

```python
adapter.clear_cache()
```

### Q3: 如何处理依赖注入？

**A**: 通过config传递依赖：

```python
component.initialize({
    'database': db_instance,
    'cache': cache_instance
})
```

在`_do_initialize`中获取：

```python
def _do_initialize(self, config):
    self.db = config['database']
    self.cache = config['cache']
    return True
```

### Q4: 如何实现组件链？

**A**: 使用AdapterChain：

```python
from src.core.foundation.base_adapter import AdapterChain

chain = AdapterChain("processing_chain")
chain.add_adapter(Validator())
chain.add_adapter(Transformer())
chain.add_adapter(Enricher())

result = chain.execute(input_data)
```

### Q5: 如何进行单元测试？

**A**: 参考测试文件：

```python
def test_component():
    component = MyComponent("test")
    assert component.initialize({})
    result = component.execute(operation='test')
    assert result is not None
```

---

## 相关资源

- [核心服务层重构优化总结报告](../../test_logs/核心服务层重构优化总结报告.md)
- [核心服务层代码审查报告](../../test_logs/核心服务层代码审查报告.md)
- [BaseComponent测试示例](../../tests/unit/core/foundation/test_base_component.py)
- [BaseAdapter测试示例](../../tests/unit/core/foundation/test_base_adapter.py)

---

*文档版本: 2.0*  
*最后更新: 2025-11-03*  
*维护者: RQA2025 Team*

