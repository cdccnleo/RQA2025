# 测试开发规范和最佳实践

## 📋 文档信息

**版本**: 1.0.0  
**最后更新**: 2025-10-24  
**适用范围**: RQA2025项目所有测试开发  
**维护团队**: 测试团队

---

## 🎯 测试原则

### 核心原则
1. **快速反馈**: 单元测试应在秒级完成
2. **有效验证**: 测试应验证功能正确性，而非盲目重复
3. **易于维护**: 测试代码应清晰、简洁、易于理解
4. **适当规模**: 测试规模应与测试类型匹配

### 测试金字塔

```
        /\
       /  \      E2E测试 (少量，慢速)
      /----\
     /      \    集成测试 (中等，中速)
    /--------\
   /          \  单元测试 (大量，快速)
  /-----------  \
```

---

## ⚡ 测试规模指导原则

### 单元测试迭代次数标准

| 测试类型 | 推荐迭代次数 | 最大迭代次数 | 预期执行时间 | 理由 |
|---------|------------|------------|-------------|------|
| **适配器测试（Mock）** | 500次 | 1000次 | 1-3秒 | 覆盖各种操作类型，验证稳定性 |
| **缓存管理测试** | 1000次 | 2000次 | 1-3秒 | 验证命中/未命中/淘汰机制 |
| **监控器测试** | 500次 | 1000次 | 2-5秒 | 多监控器并行，总操作量已足够 |
| **策略评估测试** | 500次 | 1000次 | 1-2秒 | 验证策略逻辑和决策 |
| **数据解析测试** | 100次 | 500次 | 1-3秒 | 每次处理多行，避免性能问题 |
| **配置操作测试** | 500次 | 1000次 | 1-2秒 | 验证配置读写和事件处理 |

### ⚠️ 警告阈值

- **黄色警告**: 迭代次数 > 2000次或执行时间 > 5秒
- **红色警告**: 迭代次数 > 10000次或执行时间 > 30秒
- **禁止提交**: 迭代次数 > 50000次

---

## ✅ 测试质量保证

### 必须添加的验证

#### 1. 操作计数器
```python
# ✅ 好的实践
def test_adapter_operations(self):
    success_count = 0
    failed_count = 0
    
    for i in range(500):
        try:
            result = adapter.operation()
            if result:
                success_count += 1
        except Exception as e:
            failed_count += 1
    
    # 验证成功率
    self.assertGreater(success_count, 450)  # >90%
```

#### 2. 成功率断言
```python
# ✅ 验证整体质量
success_rate = successful_operations / total_operations
self.assertGreater(success_rate, 0.90, 
                  f"Success rate too low: {success_rate:.2%}")
```

#### 3. 错误日志记录
```python
# ✅ 记录前几个失败用于调试
if failed_count <= 3:
    print(f"Operation failed at iteration {i}: {str(e)[:100]}")
```

### ❌ 避免的反模式

#### 反模式1：空异常处理
```python
# ❌ 错误做法
try:
    operation()
except:
    pass  # 忽略所有错误，无法发现问题

# ✅ 正确做法
try:
    operation()
    success_count += 1
except Exception as e:
    failed_count += 1
    if failed_count <= 3:
        print(f"Error: {e}")
```

#### 反模式2：过度迭代
```python
# ❌ 错误做法
for i in range(100000):  # 不合理的大规模迭代
    simple_operation()

# ✅ 正确做法
for i in range(500):  # 合理规模
    simple_operation()
    verify_result()
```

#### 反模式3：无结果验证
```python
# ❌ 错误做法
for i in range(N):
    operation()
# 没有任何断言

# ✅ 正确做法
result_count = 0
for i in range(N):
    result = operation()
    self.assertIsNotNone(result)
    result_count += 1

self.assertEqual(result_count, N)
```

---

## 📊 测试分层策略

### 层级1：快速单元测试

```python
import pytest

@pytest.mark.unit
def test_basic_functionality(self):
    """单元测试：快速验证核心功能"""
    for i in range(500):
        result = component.basic_operation(i)
        assert result is not None
```

**特征**:
- 迭代次数：100-1000次
- 执行时间：<5秒
- 覆盖：单个组件/函数
- 运行频率：每次提交

### 层级2：集成测试

```python
@pytest.mark.integration
def test_components_integration(self):
    """集成测试：验证组件协作"""
    for i in range(200):
        result = system.integrated_operation(i)
        assert result.status == "success"
```

**特征**:
- 迭代次数：100-500次
- 执行时间：<30秒
- 覆盖：多个组件协作
- 运行频率：PR合并前

### 层级3：性能测试

```python
@pytest.mark.performance
@pytest.mark.skip(reason="Performance test, run manually")
def test_performance_benchmark(self):
    """性能测试：基准测试（按需执行）"""
    import time
    
    start = time.time()
    for i in range(50000):
        operation()
    duration = time.time() - start
    
    assert duration < 10.0, f"Performance degraded: {duration}s"
```

**特征**:
- 迭代次数：10000+次
- 执行时间：不限制
- 覆盖：性能指标
- 运行频率：手动/定期

### 层级4：端到端测试

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_end_to_end_workflow(self):
    """E2E测试：完整业务流程"""
    # 模拟真实用户操作
    result = system.complete_workflow(test_data)
    assert result.success
```

**特征**:
- 迭代次数：1-10次
- 执行时间：分钟级
- 覆盖：完整业务流程
- 运行频率：发布前

---

## 🏷️ 测试标记使用

### Pytest标记配置

在 `pytest.ini` 中配置：

```ini
[pytest]
markers =
    unit: 快速单元测试（<5秒）
    integration: 集成测试（<30秒）
    performance: 性能测试（按需执行）
    slow: 慢速测试（需要特别关注）
    e2e: 端到端测试
    smoke: 冒烟测试

# 默认跳过性能测试
addopts = -m "not performance"
```

### 使用示例

```python
import pytest

@pytest.mark.unit
def test_quick_operation():
    """快速单元测试"""
    assert operation() == expected

@pytest.mark.integration
def test_system_integration():
    """集成测试"""
    assert system.works()

@pytest.mark.performance
@pytest.mark.skip(reason="Run manually")
def test_performance():
    """性能测试"""
    benchmark_operation()
```

### 运行不同层级的测试

```bash
# 只运行单元测试
pytest -m unit

# 运行单元测试和集成测试
pytest -m "unit or integration"

# 运行所有测试（包括性能测试）
pytest -m ""

# 排除慢速测试
pytest -m "not slow"
```

---

## 📝 测试编写模板

### 适配器测试模板

```python
import unittest
from unittest.mock import Mock, patch

class TestAdapterName(unittest.TestCase):
    """适配器测试 - 描述"""
    
    @patch('module.path.ExternalDependency')
    def test_adapter_operations(self, mock_dep):
        """测试适配器基本操作"""
        # 1. 设置Mock
        mock_dep.return_value = Mock()
        mock_dep.return_value.method.return_value = "expected"
        
        # 2. 创建适配器
        adapter = Adapter(config)
        
        # 3. 执行测试（合理规模）
        success_count = 0
        for i in range(500):
            try:
                result = adapter.operation(f"data_{i}")
                self.assertIsNotNone(result)
                success_count += 1
            except Exception as e:
                if i < 5:  # 记录前5个错误
                    print(f"Failed at {i}: {e}")
        
        # 4. 验证结果
        self.assertGreater(success_count, 450, 
                          f"Success rate too low: {success_count}/500")
```

### 组件测试模板

```python
class TestComponentName(unittest.TestCase):
    """组件测试 - 描述"""
    
    def setUp(self):
        """测试前准备"""
        self.component = Component(config)
    
    def tearDown(self):
        """测试后清理"""
        self.component.cleanup()
    
    def test_component_functionality(self):
        """测试组件功能"""
        operation_count = 0
        
        for i in range(1000):
            result = self.component.process(f"data_{i}")
            
            # 验证每次结果
            self.assertIn('status', result)
            self.assertEqual(result['status'], 'success')
            operation_count += 1
        
        # 验证整体
        self.assertEqual(operation_count, 1000)
```

### DateTimeParser测试模板

```python
import pandas as pd

class TestDateTimeParser(unittest.TestCase):
    """DateTimeParser测试 - 特别注意性能"""
    
    def test_datetime_parser_operations(self):
        """测试日期时间解析（优化规模）"""
        from src.infrastructure.utils.tools.datetime_parser import DateTimeParser
        
        successful_parses = 0
        failed_parses = 0
        
        # 注意：DateTimeParser性能较差，使用较小规模
        for i in range(100):  # 仅100次
            size = (i % 100) + 1
            df = pd.DataFrame({
                'date': [f'2023-{((j % 12) + 1):02d}-{((j % 28) + 1):02d}' 
                        for j in range(size)],
                'time': [f'{(j % 24):02d}:{(j % 60):02d}:00' 
                        for j in range(size)]
            })
            
            try:
                result = DateTimeParser.parse_datetime(df, 'date', 'time')
                
                # 验证结果
                self.assertIn('publish_time', result.columns)
                self.assertEqual(len(result), size)
                successful_parses += 1
            except Exception as e:
                failed_parses += 1
                if failed_parses <= 3:
                    print(f"Parse failed: {str(e)[:80]}")
        
        # 验证成功率
        success_rate = successful_parses / 100
        self.assertGreater(success_rate, 0.95)
```

---

## 🚀 性能优化建议

### 1. 使用合理的测试数据规模

```python
# ❌ 不好的做法
test_data = [generate_data() for i in range(100000)]  # 太大

# ✅ 好的做法
test_data = [generate_data() for i in range(100)]  # 合理

# ✅ 更好的做法：按需生成
def test_with_generator():
    for i in range(100):
        data = generate_data(i)  # 即用即生成
        process(data)
```

### 2. 避免不必要的对象创建

```python
# ❌ 不好的做法
for i in range(1000):
    adapter = Adapter(config)  # 每次都创建
    adapter.operation()

# ✅ 好的做法
adapter = Adapter(config)  # 创建一次
for i in range(1000):
    adapter.operation()
```

### 3. 使用setUp/tearDown复用资源

```python
class TestWithSetup(unittest.TestCase):
    def setUp(self):
        """准备可复用的资源"""
        self.adapter = Adapter(config)
        self.test_data = load_test_data()
    
    def tearDown(self):
        """清理资源"""
        self.adapter.cleanup()
    
    def test_operation_1(self):
        """测试1 - 复用setUp的资源"""
        result = self.adapter.operation(self.test_data)
        self.assertTrue(result)
```

### 4. 定期清理内存

```python
import gc

def test_large_operations(self):
    for i in range(5000):
        operation(i)
        
        # 每1000次清理一次
        if i % 1000 == 0:
            gc.collect()
```

---

## 🔍 测试代码审查检查清单

### 提交前自检

- [ ] 测试迭代次数是否合理（<2000次）
- [ ] 是否添加了操作计数器和成功率验证
- [ ] 是否记录了错误信息
- [ ] 是否使用了适当的pytest标记
- [ ] 测试执行时间是否<5秒（单元测试）
- [ ] 是否有空的except块
- [ ] 是否有无意义的大规模重复
- [ ] 变量命名是否清晰
- [ ] 是否添加了必要的注释

### Code Review检查点

1. **测试目的明确**
   - 测试名称清晰描述测试内容
   - 有适当的文档字符串

2. **合理的测试规模**
   - 迭代次数符合标准
   - 执行时间在预期范围

3. **有效的结果验证**
   - 有明确的断言
   - 验证成功率
   - 记录错误信息

4. **代码质量**
   - 无重复代码
   - 合理的抽象和复用
   - 清晰的结构

---

## 🛠️ 持续集成配置

### CI Pipeline配置示例

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist pytest-timeout
    
    - name: Run unit tests
      run: |
        pytest -m unit -v --cov --timeout=300
      timeout-minutes: 10
    
    - name: Run integration tests
      run: |
        pytest -m integration -v --timeout=600
      timeout-minutes: 20
    
    - name: Check for slow tests
      run: |
        pytest --durations=10 | tee durations.txt
        # 检查是否有超过30秒的测试
        if grep -E "([3-9][0-9]|[0-9]{3,})\\..*s" durations.txt; then
          echo "Warning: Found slow tests"
          exit 1
        fi
```

### 本地测试脚本

```bash
#!/bin/bash
# scripts/run_tests.sh

echo "🧪 运行快速单元测试..."
pytest -m unit -v --durations=10

if [ $? -eq 0 ]; then
    echo "✅ 单元测试通过"
else
    echo "❌ 单元测试失败"
    exit 1
fi

echo "🔍 检查测试时长..."
pytest --durations=20 | grep -E "([5-9]|[0-9]{2,})\\..*s"

echo "✨ 所有测试完成"
```

---

## 📚 参考资源

### 内部文档
- [测试覆盖率改进计划](../TEST_COVERAGE_IMPROVEMENT_PLAN.md)
- [测试效率优化报告](../test_logs/test_efficiency_optimization_report.md)
- [测试效率优化报告（第二阶段）](../test_logs/test_efficiency_optimization_report_phase2.md)

### 外部资源
- [Pytest官方文档](https://docs.pytest.org/)
- [unittest文档](https://docs.python.org/3/library/unittest.html)
- [测试金字塔](https://martinfowler.com/articles/practical-test-pyramid.html)

### 工具推荐
- `pytest-xdist`: 并行测试执行
- `pytest-cov`: 代码覆盖率
- `pytest-timeout`: 测试超时保护
- `pytest-benchmark`: 性能基准测试
- `pytest-watch`: 自动运行测试

---

## 🔄 文档维护

**更新频率**: 每季度或重大变更时  
**责任人**: 测试团队负责人  
**审核流程**: PR审核 + 团队会议讨论

**变更历史**:
- 2025-10-24: v1.0.0 - 初始版本（基于测试效率优化项目）

---

**最后更新**: 2025-10-24  
**版本**: 1.0.0  
**维护**: RQA2025测试团队

