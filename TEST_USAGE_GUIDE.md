# 基础设施层测试使用指南

## 📋 概述

本指南介绍如何使用基础设施层的测试系统，包括测试执行、质量监控和持续改进。

## 🏗️ 测试架构

### 目录结构
```
tests/unit/infrastructure/
├── config/           # 配置系统测试 (12个文件)
├── cache/            # 缓存系统测试 (28个文件)
├── logging/          # 日志系统测试 (22个文件)
├── health/           # 健康检查测试 (9个文件)
├── error/            # 错误处理测试 (9个文件)
├── monitoring/       # 监控系统测试 (16个文件)
├── service/          # 服务组件测试 (4个文件)
└── utils/            # 工具库测试 (8个文件)
```

## 🚀 快速开始

### 1. 运行单个测试文件
```bash
# 运行缓存系统测试
python -m pytest tests/unit/infrastructure/test_cache_system.py -v

# 运行日志系统测试
python -m pytest tests/unit/infrastructure/test_logging_system.py -v
```

### 2. 运行优化后的测试
```bash
# 使用优化测试执行器
python run_optimized_tests.py --workers=2 --timeout=30

# 运行分批测试
python run_infrastructure_tests.py --batch --timeout=60
```

### 3. 质量监控
```bash
# 运行质量监控
python simple_quality_monitor.py

# 运行完整质量检查（包含覆盖率）
python simple_quality_monitor.py --full
```

## 🛠️ 测试工具

### pytest 配置
```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts =
    --strict-markers
    --timeout=120
    --durations=10
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing

markers =
    timeout_60: 60秒超时
    timeout_120: 120秒超时
    timeout_300: 300秒超时
    concurrent: 并发测试
    slow: 运行时间较长的测试
    infrastructure: 基础设施层测试
```

### 超时控制
```python
# 为耗时测试添加标记
@pytest.mark.timeout(120)
def test_concurrent_access(self):
    """测试并发访问"""
    # 测试代码...

# 全局超时设置
pytest --timeout=300
```

## 📊 质量监控

### 质量指标
- **测试成功率**: 目标 ≥ 70%
- **覆盖率**: 目标 ≥ 50%
- **执行时间**: 目标 < 10秒/测试

### 监控命令
```bash
# 生成质量报告
python simple_quality_monitor.py

# 查看详细报告
cat reports/quality_report.json

# 查看覆盖率报告
python -m pytest --cov-report=html
open htmlcov/index.html
```

## 🔧 故障排除

### 常见问题

#### 1. Unicode解码错误
```python
# 解决方案：设置编码
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    encoding='utf-8'
)
```

#### 2. 测试超时
```python
# 增加超时时间
@pytest.mark.timeout(300)
def test_slow_operation(self):
    pass
```

#### 3. 内存不足
```python
# 减少并发数
python run_optimized_tests.py --workers=1
```

## 📈 持续改进

### 优化策略

#### 1. 分批执行
```python
# 优先运行快速测试
fast_tests = ["test_basic_operations", "test_initialization"]
slow_tests = ["test_performance", "test_concurrent"]
```

#### 2. 并行执行
```python
# 使用多进程
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_test, test) for test in tests]
```

#### 3. 智能重试
```python
# 对不稳定测试重试
@pytest.mark.flaky(reruns=3)
def test_unstable_operation(self):
    pass
```

## 📋 最佳实践

### 1. 测试命名
```python
def test_cache_set_get_basic():
def test_cache_concurrent_access():
def test_logging_format_validation():
```

### 2. 测试组织
```python
class TestCacheSystem:
    def setup_method(self):
        self.cache = UnifiedCache()

    def test_basic_operations(self):
        # 测试基本操作

    def test_error_handling(self):
        # 测试错误处理
```

### 3. 断言清晰
```python
# 好的断言
assert result == expected, f"期望 {expected}, 实际 {result}"
assert len(items) > 0, "列表不应为空"

# 不好的断言
assert result
assert not None
```

## 🔍 调试技巧

### 1. 详细输出
```bash
# 增加详细程度
pytest -v -s --tb=long

# 显示执行时间
pytest --durations=10
```

### 2. 调试特定测试
```bash
# 只运行特定测试
pytest tests/unit/infrastructure/test_cache_system.py::TestUnifiedCache::test_initialization -v

# 调试模式
pytest --pdb
```

### 3. 性能分析
```bash
# 性能分析
pytest --durations=20 --durations-min=1.0

# 覆盖率分析
pytest --cov=src/infrastructure --cov-report=html
```

## 📚 参考资料

### 相关文档
- [pytest 官方文档](https://docs.pytest.org/)
- [覆盖率工具文档](https://coverage.readthedocs.io/)
- [测试驱动开发](https://en.wikipedia.org/wiki/Test-driven_development)

### 示例代码
```python
# 完整的测试示例
import pytest
from src.infrastructure.cache.unified_cache import UnifiedCache

class TestUnifiedCache:
    def setup_method(self):
        self.cache = UnifiedCache()

    def test_cache_operations(self):
        """测试缓存基本操作"""
        # Given
        key = "test_key"
        value = "test_value"

        # When
        result = self.cache.set(key, value)

        # Then
        assert result == True
        assert self.cache.get(key) == value

    @pytest.mark.timeout(60)
    def test_cache_performance(self):
        """测试缓存性能"""
        import time

        start_time = time.time()
        for i in range(1000):
            self.cache.set(f"key_{i}", f"value_{i}")
        end_time = time.time()

        duration = end_time - start_time
        assert duration < 1.0  # 应在1秒内完成
```

## 🎯 总结

通过本指南，您可以：

1. **高效执行测试**: 使用优化工具快速运行测试套件
2. **监控质量**: 实时了解测试覆盖率和成功率
3. **持续改进**: 根据监控结果优化测试策略
4. **故障排除**: 快速定位和解决问题

基础设施层测试系统现在具备了生产级别的可靠性和可维护性。
