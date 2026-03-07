# 基础设施层测试质量标准

## 概述

本文档定义了基础设施层测试用例的质量标准和最佳实践，确保测试代码的可维护性、可靠性和有效性。

## 1. 测试命名规范

### 1.1 测试方法命名
```python
def test_[模块名]_[功能名]_[场景名]():
    """测试[模块名][功能名]在[场景名]下的行为"""
```

**良好示例:**
```python
def test_config_manager_get_existing_key():
def test_cache_manager_set_with_ttl():
def test_logger_error_message_formatting():
```

**避免示例:**
```python
def test1():          # 无意义名称
def test_config():    # 过于宽泛
def test_everything(): # 范围过大
```

### 1.2 测试类命名
```python
class Test[模块名][组件名]:
    """[模块名][组件名]的测试用例集合"""
```

**示例:**
```python
class TestConfigManagerOperations:
class TestCacheSystemIntegration:
class TestLoggingErrorHandling:
```

## 2. 测试结构标准

### 2.1 测试方法结构
每个测试方法应遵循以下结构：

```python
def test_feature_scenario(self):
    """测试描述

    测试目标：验证XXX功能在XXX场景下的行为
    前置条件：XXX
    测试步骤：1.XXX 2.XXX 3.XXX
    预期结果：XXX
    """
    # 1. 准备测试数据 (Arrange)
    test_data = create_test_data()

    # 2. 执行测试操作 (Act)
    result = perform_operation(test_data)

    # 3. 验证测试结果 (Assert)
    assert_expected_result(result)
```

### 2.2 Setup/Teardown 使用
```python
class TestExample:
    def setup_method(self):
        """测试前准备"""
        self.test_resource = create_resource()

    def teardown_method(self):
        """测试后清理"""
        cleanup_resource(self.test_resource)

    def test_operation(self):
        # 使用self.test_resource进行测试
        pass
```

## 3. 断言标准

### 3.1 断言类型选择
- `assert result == expected`: 精确匹配
- `assert result is not None`: 非空检查
- `assert isinstance(obj, ExpectedClass)`: 类型检查
- `assert hasattr(obj, 'attribute')`: 属性存在检查
- `assert len(collection) > 0`: 集合非空检查

### 3.2 断言消息
```python
# 良好示例
assert result == expected, f"Expected {expected}, got {result}"

# 避免示例
assert result == expected  # 无错误信息
```

### 3.3 异常断言
```python
# 正确方式
with pytest.raises(ValueError):
    function_that_raises()

# 错误方式
try:
    function_that_raises()
    assert False, "Should have raised ValueError"
except ValueError:
    pass
```

## 4. Mock和Patch 使用标准

### 4.1 Mock 对象创建
```python
from unittest.mock import Mock, patch, MagicMock

# 创建简单mock
mock_service = Mock()
mock_service.method.return_value = expected_value

# 创建具有规格的mock
mock_service = Mock(spec=RealService)
```

### 4.2 Patch 装饰器使用
```python
@patch('module.Class.method')
@patch('module.function')
def test_with_patches(self, mock_method, mock_function):
    # 参数顺序与装饰器相反
    pass
```

### 4.3 Context Manager 使用
```python
def test_with_context_manager(self):
    with patch('module.function') as mock_func:
        mock_func.return_value = 'mocked'
        result = call_function()
        assert result == 'mocked'
```

## 5. 错误处理和跳过测试

### 5.1 条件跳过
```python
@pytest.mark.skipif(not has_dependency, reason="Dependency not available")
def test_requiring_dependency(self):
    pass

# 或在测试中动态跳过
def test_optional_feature(self):
    if not feature_available:
        pytest.skip("Feature not available in this environment")
```

### 5.2 异常处理
```python
def test_robust_operation(self):
    try:
        from some.module import Class
        instance = Class()
        # 测试逻辑
    except ImportError:
        pytest.skip("Required module not available")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
```

## 6. 测试数据管理

### 6.1 测试数据常量
```python
class TestConstants:
    TEST_CONFIG_DATA = {
        'app': {'name': 'test_app', 'version': '1.0'},
        'database': {'host': 'localhost', 'port': 5432}
    }
    TEST_CACHE_KEY = 'test_cache_key'
    TEST_CACHE_VALUE = {'data': 'test_value'}
```

### 6.2 工厂方法
```python
def create_test_config_manager():
    """创建测试用的配置管理器"""
    manager = ConfigManager()
    manager.load_test_config()
    return manager

def create_test_service_with_dependencies():
    """创建具有依赖关系的测试服务"""
    config = create_test_config_manager()
    cache = create_test_cache_manager()
    return TestService(config, cache)
```

## 7. 文档和注释标准

### 7.1 模块文档
```python
"""
测试模块描述

测试目标：验证XXX模块的功能正确性
测试范围：XXX功能点
测试策略：XXX测试方法
"""

import pytest
# 其他导入
```

### 7.2 测试方法文档
```python
def test_specific_functionality(self):
    """测试特定功能的行为

    测试场景：正常情况下XXX功能的工作
    验证内容：
    - 输入验证
    - 处理逻辑
    - 输出结果
    - 异常处理
    """
```

### 7.3 复杂逻辑注释
```python
def test_complex_business_logic(self):
    # 准备复杂的测试数据
    complex_data = {
        'nested': {
            'structures': ['with', 'multiple', 'levels']
        }
    }

    # 执行多步骤业务逻辑
    result = business_logic.process(complex_data)

    # 验证每个层级的结果
    assert result['level1'] == expected_level1
    assert result['level2']['nested'] == expected_nested
```

## 8. 性能和效率考虑

### 8.1 测试执行时间
- 单个测试方法应在1秒内完成
- 集成测试可在10秒内完成
- 性能测试可适当延长

### 8.2 资源清理
```python
def test_with_resource_cleanup(self):
    resource = create_expensive_resource()
    try:
        # 测试逻辑
        assert operation_succeeds(resource)
    finally:
        # 确保资源被清理
        resource.cleanup()
```

### 8.3 并行执行考虑
- 测试应避免相互依赖
- 使用唯一标识符避免冲突
- 考虑共享资源的同步

## 9. 覆盖率目标

### 9.1 语句覆盖率
- 核心业务逻辑：≥90%
- 工具类方法：≥80%
- 异常处理路径：≥70%

### 9.2 分支覆盖率
- 条件分支：≥85%
- 异常处理：≥75%

### 9.3 测试维护
- 定期审查和更新测试用例
- 移除过时和冗余的测试
- 补充新功能的测试覆盖

## 10. 持续改进

### 10.1 代码审查
- 新测试用例需要代码审查
- 审查重点：命名、结构、断言、文档
- 审查通过标准：符合本规范

### 10.2 质量指标监控
- 测试通过率：≥95%
- 覆盖率稳定性：波动 ≤5%
- 测试执行时间：趋势下降

### 10.3 反馈循环
- 收集测试失败和缺陷报告
- 分析测试效果和改进空间
- 定期更新测试策略和标准

---

## 附录：常用测试模式

### 模式1：基本CRUD操作测试
```python
def test_crud_operations(self):
    # Create
    item = create_item()
    assert item.id is not None

    # Read
    retrieved = get_item(item.id)
    assert retrieved == item

    # Update
    updated = update_item(item.id, new_data)
    assert updated.field == new_data

    # Delete
    delete_item(item.id)
    assert get_item(item.id) is None
```

### 模式2：异常场景测试
```python
@pytest.mark.parametrize("invalid_input,expected_error", [
    (None, ValueError),
    ("", ValueError),
    (999999, OverflowError),
])
def test_error_handling(self, invalid_input, expected_error):
    with pytest.raises(expected_error):
        function_under_test(invalid_input)
```

### 模式3：边界条件测试
```python
@pytest.mark.parametrize("boundary_value", [
    0, 1, -1, 999, 1000, 1001,  # 数值边界
    "", "a", "a" * 1000,          # 字符串边界
    [], [1], [1] * 1000,          # 列表边界
    {}, {"key": "value"},         # 字典边界
])
def test_boundary_conditions(self, boundary_value):
    # 测试边界值的处理
    result = process_boundary_value(boundary_value)
    assert is_valid_result(result, boundary_value)
```
