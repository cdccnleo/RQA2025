# 基础设施层接口设计标准

## 📊 文档总览

**版本**: v1.0
**创建时间**: 2025年9月30日
**适用范围**: 基础设施层所有组件
**设计原则**: 统一性、一致性、可扩展性

---

## 🎯 设计目标

### 核心原则
1. **统一性**: 所有接口遵循相同的命名和结构规范
2. **一致性**: 接口行为和错误处理方式统一
3. **可扩展性**: 支持未来功能扩展而不破坏现有接口
4. **类型安全**: 充分利用Python类型提示系统

### 设计理念
- **Protocol优先**: 使用Protocol模式实现结构化子类型
- **组合优于继承**: 通过组合多个Protocol实现复杂接口
- **文档驱动**: 每个接口方法都有详细的文档说明

---

## 🏗️ 接口层次结构

### 1. 基础层 (Base Layer)

#### IBaseComponent (所有组件的基础)
```python
class IBaseComponent(Protocol):
    """基础组件协议 - 所有基础设施组件的标准接口"""

    @property
    def component_name(self) -> str:
        """组件名称标识符"""
        ...

    @property
    def component_type(self) -> str:
        """组件类型标识符"""
        ...

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        ...

    def shutdown_component(self) -> bool:
        """关闭组件"""
        ...

    def get_component_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        ...
```

### 2. 服务层 (Service Layer)

#### IServiceComponent (服务型组件)
```python
class IServiceComponent(IBaseComponent, Protocol):
    """服务组件协议 - 提供服务的组件标准接口"""

    def start_service(self) -> bool:
        """启动服务"""
        ...

    def stop_service(self) -> bool:
        """停止服务"""
        ...

    def restart_service(self) -> bool:
        """重启服务"""
        ...

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        ...
```

### 3. 数据层 (Data Layer)

#### IDataComponent (数据处理组件)
```python
class IDataComponent(IBaseComponent, Protocol):
    """数据组件协议 - 处理数据的组件标准接口"""

    def validate_data(self, data: Any) -> bool:
        """验证数据"""
        ...

    def process_data(self, data: Any) -> Any:
        """处理数据"""
        ...

    def get_data_schema(self) -> Dict[str, Any]:
        """获取数据模式"""
        ...
```

### 4. 缓存层 (Cache Layer)

#### ICacheComponent (缓存组件)
```python
class ICacheComponent(IBaseComponent, Protocol):
    """缓存组件协议 - 缓存功能的标准接口"""

    def get(self, key: str) -> Any:
        """获取缓存值"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        ...

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        ...

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        ...

    def clear(self) -> bool:
        """清空缓存"""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        ...
```

---

## 📋 接口实现规范

### 命名规范

#### 类命名
- 接口类: `I{ComponentName}Component`
- 实现类: `{ComponentName}{ImplType}`
- 抽象类: `Abstract{ComponentName}`

#### 方法命名
- 属性: `snake_case`
- 方法: `snake_case`
- 私有方法: `_snake_case`

#### 常量命名
- 全部大写: `UPPER_CASE`
- 单词间用下划线分隔

### 文档规范

#### 类文档
```python
class IExampleComponent(IBaseComponent, Protocol):
    """
    示例组件标准协议

    继承自IBaseComponent协议，定义了示例组件特有的方法。
    使用Protocol模式，支持结构化子类型，无需显式继承。

    实现要求：
    - 必须实现IBaseComponent的所有方法
    - 必须实现示例组件特有的业务方法
    - 必须提供准确的状态信息
    - 必须处理并发访问

    示例:
        class ExampleComponent:
            @property
            def component_name(self) -> str:
                return "example_component"

            def initialize_component(self, config: Dict[str, Any]) -> bool:
                # 实现初始化逻辑
                return True
    """
```

#### 方法文档
```python
def example_method(self, param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    示例方法说明

    详细描述方法的用途、参数含义和返回值。

    Args:
        param1: 必需参数说明
        param2: 可选参数说明，默认为None

    Returns:
        Dict[str, Any]: 返回结果说明

    Raises:
        ValueError: 参数无效时抛出
        RuntimeError: 执行失败时抛出

    Example:
        result = component.example_method("value", 42)
    """
```

### 错误处理规范

#### 异常类型
- `ValueError`: 参数验证失败
- `RuntimeError`: 运行时错误
- `ConnectionError`: 连接相关错误
- `TimeoutError`: 超时错误
- `PermissionError`: 权限相关错误

#### 异常处理模式
```python
try:
    # 业务逻辑
    result = self._perform_operation()
    return result
except ValueError as e:
    InfrastructureLogger.log_operation_failure("操作名", e)
    raise
except Exception as e:
    InfrastructureLogger.log_operation_failure("操作名", e)
    raise RuntimeError(f"操作失败: {e}") from e
```

---

## 🔧 接口实现模板

### 标准实现模板
```python
"""
基础设施层 - 示例组件

本组件实现了IExampleComponent协议。
"""

from typing import Any, Dict, List, Optional
from src.infrastructure.utils.common_patterns import InfrastructureLogger
import logging

logger = logging.getLogger(__name__)


class ExampleComponent:
    """
    示例组件实现

    实现IExampleComponent协议的标准组件。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化组件"""
        self._config = config or {}
        self._initialized = False

    @property
    def component_name(self) -> str:
        """组件名称"""
        return "example_component"

    @property
    def component_type(self) -> str:
        """组件类型"""
        return "example"

    def initialize_component(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        try:
            InfrastructureLogger.log_initialization_success(self.component_name, self.component_type)

            # 初始化逻辑
            self._config.update(config)
            self._initialized = True

            InfrastructureLogger.log_initialization_success(self.component_name, self.component_type)
            return True
        except Exception as e:
            InfrastructureLogger.log_initialization_failure(self.component_name, e, self.component_type)
            return False

    def shutdown_component(self) -> bool:
        """关闭组件"""
        try:
            # 关闭逻辑
            self._initialized = False
            InfrastructureLogger.log_operation_success("组件关闭", f"{self.component_name}已关闭")
            return True
        except Exception as e:
            InfrastructureLogger.log_operation_failure("组件关闭", e)
            return False

    def get_component_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type,
            "initialized": self._initialized,
            "config": self._config
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 健康检查逻辑
            is_healthy = self._initialized

            return {
                "healthy": is_healthy,
                "timestamp": "2025-09-30T12:00:00Z",
                "component": self.component_name,
                "checks": {
                    "initialization": {"healthy": self._initialized}
                }
            }
        except Exception as e:
            return {
                "healthy": False,
                "timestamp": "2025-09-30T12:00:00Z",
                "component": self.component_name,
                "error": str(e)
            }

    # 业务方法实现
    def example_business_method(self) -> str:
        """示例业务方法"""
        return "example result"
```

### 测试模板
```python
"""
基础设施层 - 示例组件测试
"""

import pytest
from unittest.mock import Mock
from src.infrastructure.components.example_component import ExampleComponent


class TestExampleComponent:
    """示例组件测试"""

    def test_component_name(self):
        """测试组件名称"""
        component = ExampleComponent()
        assert component.component_name == "example_component"

    def test_component_type(self):
        """测试组件类型"""
        component = ExampleComponent()
        assert component.component_type == "example"

    def test_initialization(self):
        """测试初始化"""
        component = ExampleComponent()
        config = {"test": "value"}

        result = component.initialize_component(config)
        assert result is True
        assert component._initialized is True

    def test_health_check(self):
        """测试健康检查"""
        component = ExampleComponent()
        component.initialize_component({})

        result = component.health_check()
        assert result["healthy"] is True
        assert result["component"] == "example_component"

    def test_business_method(self):
        """测试业务方法"""
        component = ExampleComponent()
        result = component.example_business_method()
        assert result == "example result"
```

---

## 📊 合规性检查

### 自动化检查工具
```python
from src.infrastructure.utils.common_patterns import InfrastructureInterfaceTemplate

def check_interface_compliance(cls: type, interface: type) -> List[str]:
    """检查类是否符合接口规范"""
    return InfrastructureInterfaceTemplate.validate_interface_compliance(cls, interface)
```

### 持续集成检查
```yaml
# .github/workflows/interface-check.yml
name: Interface Compliance Check

on: [push, pull_request]

jobs:
  interface-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Check Interface Compliance
      run: python scripts/check_interface_compliance.py
```

---

## 🚀 迁移指南

### 从旧接口迁移到新接口

#### 步骤1: 分析现有代码
```python
# 识别所有组件类
component_classes = []
for module in infrastructure_modules:
    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if hasattr(cls, 'component_name'):  # 简单识别组件
            component_classes.append(cls)
```

#### 步骤2: 生成接口代码
```python
from src.infrastructure.utils.common_patterns import InfrastructureInterfaceTemplate

# 为每个组件生成标准接口
for component_cls in component_classes:
    interface_code = InfrastructureInterfaceTemplate.generate_standard_interface(
        interface_type="component",
        component_name=component_cls.__name__,
        custom_methods=get_component_methods(component_cls)
    )
    # 保存接口代码
    save_interface_code(component_cls.__name__, interface_code)
```

#### 步骤3: 更新实现类
```python
# 让实现类实现新的Protocol接口
class ExampleComponent:
    """
    示例组件实现

    实现IExampleComponent协议。
    """

    # 实现协议方法
    def component_name(self) -> str:
        return "example_component"

    def component_type(self) -> str:
        return "example"

    # ... 其他协议方法
```

#### 步骤4: 验证合规性
```python
# 运行合规性检查
issues = check_interface_compliance(ExampleComponent, IExampleComponent)
if issues:
    print(f"合规性问题: {issues}")
else:
    print("接口合规性检查通过")
```

---

## 📈 质量指标

### 合规性指标
- **接口一致性**: >95% 的组件遵循统一接口规范
- **文档完整性**: >90% 的接口方法有完整文档
- **类型覆盖率**: >85% 的方法有类型提示
- **测试覆盖率**: >80% 的接口有对应的测试

### 维护性指标
- **代码重复率**: <10% 的接口代码重复
- **变更影响度**: 修改接口时影响的组件 <20%
- **扩展性指数**: 新功能扩展的平均代码量 <50行

---

## 🔄 演进计划

### Phase 1: 基础标准化 (当前)
- [x] 定义统一接口规范
- [x] 建立接口模板系统
- [x] 实现合规性检查工具
- [ ] 迁移核心组件接口

### Phase 2: 全面应用 (下月)
- [ ] 完成所有组件接口标准化
- [ ] 建立接口版本管理机制
- [ ] 完善接口文档系统
- [ ] 建立接口变更审查流程

### Phase 3: 智能化治理 (季度)
- [ ] 实现接口自动化生成
- [ ] 建立接口质量监控
- [ ] 实现接口演进预测
- [ ] 建立接口生态系统

---

## 📞 总结

基础设施层接口设计标准为整个基础设施层的组件开发提供了统一的设计规范和实现指南。通过Protocol模式、标准化的命名规范、完整的文档要求和严格的质量检查，确保了基础设施层代码的一致性、可维护性和可扩展性。

这个标准不仅规范了当前组件的开发，也为未来组件的快速开发和集成奠定了坚实的基础。通过持续的质量监控和演进机制，基础设施层将保持高标准的代码质量和架构一致性。
