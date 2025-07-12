# Pytest Fixture 正确使用指南

## 错误模式（需修复）
```python
# 错误：直接调用fixture
def test_example():
    data = sample_fixture()  # 直接调用错误
    assert process(data) == expected
```

## 正确模式
```python
# 正确：通过测试参数使用
def test_example(sample_fixture):  # fixture作为参数
    assert process(sample_fixture) == expected
```

## 常见修复场景

1. **基础fixture**:
```python
# 修复前
data = load_test_data()

# 修复后
def test_data_processing(load_test_data):
    process(load_test_data)
```

2. **参数化fixture**:
```python
# 修复前
for scenario in test_scenarios():
    test_case(scenario)

# 修复后
def test_all_scenarios(test_scenarios):
    process(test_scenarios)
```

## 需要修改的测试文件
根据错误日志，以下文件需要检查：
- tests/features/test_feature_standardizer.py
- tests/features/test_feature_manager.py
- tests/infrastructure/test_logger.py
