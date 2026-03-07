# 测试 Mock 规范与最佳实践

## 概述

本文档定义了项目中测试时使用 Mock 的规范和最佳实践，旨在避免全局 Mock 污染导致的测试问题。

## 问题背景

在之前的测试中，我们遇到了 `sklearn` 被全局 Mock 污染的问题：

```python
# ❌ 错误示例：全局 Mock 导致污染
sys.modules['sklearn'] = MagicMock()
```

这会导致：
- 其他测试文件无法正常导入 `sklearn`
- `ModuleNotFoundError: No module named 'sklearn.linear_model'`
- 测试环境被污染，影响所有后续测试

## Mock 规范

### 1. 禁止全局 Mock

**❌ 禁止的做法：**

```python
# 在文件顶层进行全局 Mock
import sys
from unittest.mock import MagicMock

sys.modules['some_module'] = MagicMock()
sys.modules['another_module'] = MagicMock()
```

**✅ 推荐的做法：**

```python
# 使用 @patch 装饰器进行局部 Mock
from unittest.mock import patch

@patch('module.to.mock')
def test_something(mock_module):
    # 测试代码
    pass
```

### 2. 局部 Mock 最佳实践

#### 2.1 使用 @patch 装饰器

```python
import pytest
from unittest.mock import patch, MagicMock

class TestExample:
    @patch('src.models.some_module')
    def test_with_mock(self, mock_module):
        # 配置 mock
        mock_module.some_function.return_value = "mocked_result"
        
        # 执行测试
        result = some_function_under_test()
        
        # 验证结果
        assert result == "expected_result"
        mock_module.some_function.assert_called_once()
```

#### 2.2 使用 patch 上下文管理器

```python
def test_with_context_manager():
    with patch('src.models.some_module') as mock_module:
        mock_module.some_function.return_value = "mocked_result"
        
        # 执行测试
        result = some_function_under_test()
        
        # 验证结果
        assert result == "expected_result"
```

#### 2.3 使用 patch.object 进行部分 Mock

```python
@patch.object(SomeClass, 'some_method')
def test_partial_mock(self, mock_method):
    mock_method.return_value = "mocked_result"
    
    # 测试代码
    instance = SomeClass()
    result = instance.some_method()
    
    assert result == "mocked_result"
```

### 3. Mock 配置规范

#### 3.1 设置返回值

```python
# 简单返回值
mock_function.return_value = "result"

# 复杂返回值
mock_function.return_value = {
    "status": "success",
    "data": [1, 2, 3]
}

# 异常返回值
mock_function.side_effect = ValueError("Error message")
```

#### 3.2 设置副作用

```python
# 抛出异常
mock_function.side_effect = ValueError("Error")

# 多次调用返回不同值
mock_function.side_effect = ["first", "second", "third"]

# 自定义函数
def custom_side_effect(*args, **kwargs):
    return f"called with {args}, {kwargs}"

mock_function.side_effect = custom_side_effect
```

#### 3.3 验证调用

```python
# 验证被调用
mock_function.assert_called()

# 验证调用次数
mock_function.assert_called_once()
mock_function.assert_called_times(3)

# 验证调用参数
mock_function.assert_called_with("expected_arg")
mock_function.assert_called_with(arg1="value1", arg2="value2")

# 验证调用历史
expected_calls = [
    call("first_call"),
    call("second_call")
]
mock_function.assert_has_calls(expected_calls)
```

### 4. 常见 Mock 场景

#### 4.1 Mock 外部 API

```python
@patch('requests.get')
def test_api_call(self, mock_get):
    # 配置 mock 响应
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": "test"}
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    # 执行测试
    result = api_client.get_data()
    
    # 验证
    assert result == {"data": "test"}
    mock_get.assert_called_once_with("https://api.example.com/data")
```

#### 4.2 Mock 数据库操作

```python
@patch('src.database.connection.execute')
def test_database_operation(self, mock_execute):
    # 配置 mock
    mock_execute.return_value.fetchall.return_value = [
        {"id": 1, "name": "test"}
    ]
    
    # 执行测试
    result = database_service.get_users()
    
    # 验证
    assert len(result) == 1
    assert result[0]["name"] == "test"
```

#### 4.3 Mock 文件操作

```python
@patch('builtins.open', mock_open(read_data='file content'))
def test_file_operation(self):
    # 执行测试
    content = file_service.read_file("test.txt")
    
    # 验证
    assert content == "file content"
```

### 5. 测试隔离原则

#### 5.1 每个测试独立

```python
class TestIsolated:
    def setup_method(self):
        """每个测试方法前的设置"""
        self.mock_data = {"test": "data"}
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理 mock 状态
        pass
    
    @patch('external.module')
    def test_isolated_method(self, mock_module):
        # 测试代码
        pass
```

#### 5.2 避免测试间依赖

```python
# ❌ 错误：测试间共享状态
class TestSharedState:
    shared_mock = None
    
    def test_first(self):
        self.shared_mock = MagicMock()
        # 测试代码

# ✅ 正确：每个测试独立
class TestIndependent:
    def test_first(self):
        mock = MagicMock()
        # 测试代码
```

### 6. 性能考虑

#### 6.1 避免过度 Mock

```python
# ❌ 过度 Mock
@patch('module1')
@patch('module2')
@patch('module3')
@patch('module4')
def test_over_mocked(self, mock4, mock3, mock2, mock1):
    # 测试代码

# ✅ 适度 Mock
@patch('critical.external.dependency')
def test_focused_mock(self, mock_dependency):
    # 只 Mock 关键依赖
    pass
```

#### 6.2 使用 Mock 缓存

```python
class TestWithCachedMock:
    @classmethod
    def setup_class(cls):
        """类级别的 Mock 设置"""
        cls.mock_patcher = patch('external.service')
        cls.mock_service = cls.mock_patcher.start()
    
    @classmethod
    def teardown_class(cls):
        """类级别的 Mock 清理"""
        cls.mock_patcher.stop()
```

### 7. 调试 Mock 问题

#### 7.1 检查 Mock 状态

```python
def test_debug_mock():
    with patch('target.module') as mock:
        # 检查 Mock 是否被正确创建
        print(f"Mock type: {type(mock)}")
        print(f"Mock called: {mock.called}")
        print(f"Mock call count: {mock.call_count}")
        print(f"Mock call args: {mock.call_args_list}")
```

#### 7.2 验证 Mock 配置

```python
def test_verify_mock_config():
    with patch('target.module') as mock:
        # 配置 Mock
        mock.some_method.return_value = "expected"
        
        # 验证配置
        assert mock.some_method.return_value == "expected"
        
        # 执行测试
        result = target_function()
        
        # 验证调用
        mock.some_method.assert_called_once()
```

## 迁移指南

### 从全局 Mock 迁移到局部 Mock

#### 步骤 1：识别全局 Mock

```python
# 查找所有全局 Mock
grep -r "sys.modules\[" tests/
```

#### 步骤 2：替换为局部 Mock

```python
# 原来的全局 Mock
sys.modules['external_module'] = MagicMock()

# 替换为局部 Mock
@patch('external_module')
def test_function(self, mock_module):
    # 测试代码
    pass
```

#### 步骤 3：更新测试逻辑

```python
# 原来的测试
def test_old_way():
    # 全局 Mock 已经生效
    result = function_under_test()
    assert result == "expected"

# 新的测试
@patch('external_module')
def test_new_way(self, mock_module):
    # 配置 Mock
    mock_module.some_method.return_value = "mocked"
    
    # 执行测试
    result = function_under_test()
    
    # 验证
    assert result == "expected"
    mock_module.some_method.assert_called_once()
```

## 检查清单

在编写测试时，请确保：

- [ ] 没有使用 `sys.modules[...] = MagicMock()`
- [ ] 使用 `@patch` 装饰器进行局部 Mock
- [ ] 每个测试都是独立的
- [ ] Mock 配置清晰明确
- [ ] 验证 Mock 的调用
- [ ] 测试后正确清理 Mock 状态

## 常见问题

### Q: 为什么不能使用全局 Mock？

A: 全局 Mock 会污染整个测试环境，影响其他测试文件的正常运行。当 pytest 收集测试时，会导入所有测试文件，全局 Mock 会在导入阶段就生效，导致其他测试无法正常导入被 Mock 的模块。

### Q: 如何处理复杂的依赖关系？

A: 使用 `@patch` 装饰器可以精确控制 Mock 的范围和生命周期，确保只在需要的测试中生效，不会影响其他测试。

### Q: 如何 Mock 第三方库？

A: 使用 `@patch('third_party_library')` 或 `@patch('third_party_library.specific_function')` 来 Mock 第三方库的特定功能。

## 总结

遵循这些 Mock 规范可以：

1. **避免测试污染**：确保测试之间相互独立
2. **提高测试可靠性**：减少因 Mock 问题导致的测试失败
3. **便于维护**：清晰的 Mock 配置便于理解和维护
4. **提升性能**：避免不必要的全局 Mock 开销

记住：**局部 Mock 优于全局 Mock，精确 Mock 优于过度 Mock**。

## 相关文件

- [Mock 使用示例](mock_examples.py) - 完整的 Mock 使用示例代码
- [测试框架文档](README.md) - 测试框架完整文档 