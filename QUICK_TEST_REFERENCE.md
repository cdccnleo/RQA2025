# 测试开发快速参考卡 🚀

> 快速查阅测试开发关键信息，完整规范见 [docs/testing_guidelines.md](docs/testing_guidelines.md)

---

## ⚡ 快速开始

### 运行测试

```bash
# 运行所有单元测试
pytest -m unit

# 运行单个测试文件
pytest tests/unit/path/to/test_file.py -v

# 运行指定测试
pytest tests/unit/path/to/test_file.py::TestClass::test_method -v

# 查看最慢的测试
pytest --durations=20

# 并行执行（推荐）
pytest -n auto
```

---

## 📏 测试规模标准（必读！）

### ⚠️ 迭代次数上限

| 测试类型 | 推荐 | 最大 | 执行时间 |
|---------|------|------|---------|
| **适配器测试** | 500次 | 1000次 | <3秒 |
| **缓存测试** | 1000次 | 2000次 | <3秒 |
| **监控器测试** | 500次 | 1000次 | <5秒 |
| **解析器测试** | 100次 | 500次 | <3秒 |

### 🚨 警告阈值

```
🟡 黄色警告: >2000次 或 >5秒
🔴 红色警告: >10000次 或 >30秒
❌ 禁止提交: >50000次
```

---

## ✅ 测试模板

### 适配器测试模板

```python
import unittest
from unittest.mock import Mock, patch

class TestYourAdapter(unittest.TestCase):
    @patch('module.path.Dependency')
    def test_adapter_operations(self, mock_dep):
        """测试适配器操作"""
        # 1. Mock设置
        mock_dep.return_value = Mock()
        
        # 2. 创建适配器
        adapter = YourAdapter(config)
        
        # 3. 执行测试（合理规模）
        success_count = 0
        for i in range(500):  # ✅ 合理规模
            try:
                result = adapter.operation(f"data_{i}")
                self.assertIsNotNone(result)
                success_count += 1
            except Exception as e:
                if i < 3:  # 记录前3个错误
                    print(f"Error at {i}: {e}")
        
        # 4. 验证成功率
        self.assertGreater(success_count, 450, 
                          f"Success rate: {success_count}/500")
```

### 组件测试模板

```python
class TestYourComponent(unittest.TestCase):
    def setUp(self):
        self.component = YourComponent(config)
    
    def test_component_functionality(self):
        """测试组件功能"""
        operation_count = 0
        
        for i in range(1000):  # ✅ 合理规模
            result = self.component.process(f"data_{i}")
            self.assertIn('status', result)
            operation_count += 1
        
        self.assertEqual(operation_count, 1000)
```

---

## ❌ 避免的反模式

### 反模式1: 过度迭代

```python
# ❌ 错误
for i in range(100000):  # 太多！
    operation()

# ✅ 正确
for i in range(500):  # 合理
    operation()
    verify_result()
```

### 反模式2: 空异常处理

```python
# ❌ 错误
try:
    operation()
except:
    pass  # 忽略错误

# ✅ 正确
try:
    operation()
    success_count += 1
except Exception as e:
    failed_count += 1
    if failed_count <= 3:
        print(f"Error: {e}")
```

### 反模式3: 无结果验证

```python
# ❌ 错误
for i in range(N):
    operation()
# 没有断言

# ✅ 正确
result_count = 0
for i in range(N):
    result = operation()
    self.assertIsNotNone(result)
    result_count += 1
self.assertEqual(result_count, N)
```

---

## 🏷️ pytest标记使用

### 常用标记

```python
import pytest

# 单元测试（快速，每次运行）
@pytest.mark.unit
def test_quick():
    assert True

# 集成测试（中速，PR前运行）
@pytest.mark.integration
def test_integration():
    assert system_works()

# 性能测试（慢速，按需运行）
@pytest.mark.performance
@pytest.mark.skip(reason="Run manually")
def test_performance():
    benchmark()

# 慢速测试（需要优化）
@pytest.mark.slow
def test_slow():
    time_consuming_operation()
```

### 运行特定标记

```bash
# 只运行单元测试
pytest -m unit

# 运行单元和集成测试
pytest -m "unit or integration"

# 排除慢速测试
pytest -m "not slow"

# 手动运行性能测试
pytest -m performance --no-skip
```

---

## 📊 测试质量检查清单

### 提交前检查

- [ ] 迭代次数 < 2000
- [ ] 执行时间 < 5秒（单元测试）
- [ ] 添加了操作计数器
- [ ] 添加了成功率断言（>90%）
- [ ] 记录了错误信息（前3-5个）
- [ ] 添加了pytest标记
- [ ] 无空的except块
- [ ] 有清晰的测试名称和文档

### Code Review检查

- [ ] 测试目的明确
- [ ] 规模合理
- [ ] 有效验证
- [ ] 代码清晰
- [ ] 无重复代码

---

## 🛠️ 常用命令

### 基础命令

```bash
# 运行所有测试
pytest

# 详细输出
pytest -v

# 显示print输出
pytest -s

# 失败时停止
pytest -x

# 并行执行（推荐）
pytest -n auto

# 重新运行失败的测试
pytest --lf
```

### 性能相关

```bash
# 显示最慢的10个测试
pytest --durations=10

# 显示所有>1秒的测试
pytest --durations=0 --durations-min=1.0

# 超时保护
pytest --timeout=300
```

### 覆盖率相关

```bash
# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 只看覆盖率
pytest --cov=src --cov-report=term-missing
```

---

## 📚 资源链接

| 资源 | 位置 | 说明 |
|-----|------|------|
| **完整测试规范** | `docs/testing_guidelines.md` | ⭐ 必读 |
| **pytest配置** | `tests/pytest.ini` | 自动应用 |
| **优化报告** | `test_logs/TEST_EFFICIENCY_OPTIMIZATION_PROJECT_COMPLETE.md` | 项目成果 |
| **AI审查报告** | `reports/README_infrastructure_ai_review.md` | 代码质量 |
| **项目索引** | `PROJECT_ACHIEVEMENTS_INDEX.md` | 成果汇总 |

---

## 💡 常见问题

### Q: 测试应该迭代多少次？
**A**: 单元测试500-1000次足够，DateTimeParser只需100次。参考规模标准表。

### Q: 测试运行太慢怎么办？
**A**: 
1. 检查迭代次数是否过多
2. 使用 `pytest --durations=10` 找出慢速测试
3. 参考优化后的测试示例

### Q: 如何验证测试结果？
**A**: 
1. 添加操作计数器
2. 使用 `assertGreater()` 验证成功率
3. 记录前几个失败的错误

### Q: 如何区分单元测试和性能测试？
**A**:
- 单元测试: <5秒，标记 `@pytest.mark.unit`
- 性能测试: 不限时，标记 `@pytest.mark.performance` + `@pytest.mark.skip`

---

## 🎓 最佳实践速记

### ✅ DO（要做）

- ✅ 使用500-1000次迭代
- ✅ 添加结果验证
- ✅ 记录错误信息
- ✅ 使用pytest标记
- ✅ 保持测试<5秒

### ❌ DON'T（不要做）

- ❌ 超过10000次迭代
- ❌ 空的except块
- ❌ 无结果验证
- ❌ 忽略测试时间
- ❌ 重复创建对象

---

## 🚀 快速示例

### 最简单的测试

```python
def test_basic():
    """最简单的测试示例"""
    result = my_function()
    assert result == expected
```

### 标准循环测试

```python
def test_standard_loop():
    """标准循环测试示例"""
    success_count = 0
    
    for i in range(500):  # ✅ 合理规模
        result = operation(i)
        assert result is not None
        success_count += 1
    
    assert success_count > 450  # ✅ 验证成功率
```

### 完整功能测试

```python
class TestComplete(unittest.TestCase):
    def setUp(self):
        self.component = Component()
    
    def test_complete_functionality(self):
        """完整功能测试示例"""
        success = fail = 0
        
        for i in range(1000):  # ✅ 合理规模
            try:
                result = self.component.process(i)
                self.assertIsNotNone(result)
                success += 1
            except Exception as e:
                fail += 1
                if fail <= 3:  # ✅ 记录错误
                    print(f"Failed: {e}")
        
        # ✅ 验证成功率
        rate = success / 1000
        self.assertGreater(rate, 0.90, f"Rate: {rate:.2%}")
```

---

**最后更新**: 2025-10-24  
**版本**: v1.0.0  
**状态**: ✅ 可用

📚 **完整规范**: [docs/testing_guidelines.md](docs/testing_guidelines.md)

