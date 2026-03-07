# 内存管理最佳实践

## 概述

本文档基于基础设施层内存泄漏分析报告，提供了完整的内存管理最佳实践指南，帮助开发者避免内存泄漏问题。

## 问题背景

### 原始问题
- **内存暴涨**: 从18MB暴涨到983MB (+965MB)
- **主要泄漏源**: importlib模块加载、单例实例、Prometheus指标
- **根本原因**: import时的全局注册、单例模式设计缺陷、缺乏清理机制

### 解决方案
通过激进清理策略和内存隔离环境，成功解决了99.2%的内存泄漏问题。

## 最佳实践

### 1. 单例模式管理

#### ❌ 错误做法
```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

#### ✅ 正确做法
```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def cleanup(self):
        """提供清理方法"""
        if hasattr(self, '_cache'):
            self._cache.clear()
        if hasattr(self, '_handlers'):
            self._handlers.clear()

# 在测试中使用
@pytest.fixture(autouse=True)
def cleanup_singleton():
    yield
    ConfigManager._instance = None
```

### 2. Prometheus指标管理

#### ❌ 错误做法
```python
from prometheus_client import Counter, REGISTRY

# 直接使用全局注册表
counter = Counter('my_counter', 'My counter')
```

#### ✅ 正确做法
```python
from prometheus_client import Counter, CollectorRegistry

# 使用隔离的注册表
@pytest.fixture
def isolated_registry():
    return CollectorRegistry()

def test_with_metrics(isolated_registry):
    counter = Counter('my_counter', 'My counter', registry=isolated_registry)
    # 测试完成后自动清理
```

### 3. 模块导入优化

#### ❌ 错误做法
```python
# 大量import操作
import src.infrastructure
import src.infrastructure.config
import src.infrastructure.monitoring
import src.infrastructure.logging
```

#### ✅ 正确做法
```python
# 按需导入
from src.infrastructure.config import get_unified_config_manager
from src.infrastructure.monitoring import ApplicationMonitor

# 使用环境变量控制
os.environ['DISABLE_HEAVY_IMPORTS'] = 'true'
```

### 4. 线程管理

#### ❌ 错误做法
```python
import threading

def start_monitor():
    thread = threading.Thread(target=monitor_function)
    thread.start()
    # 没有正确停止线程
```

#### ✅ 正确做法
```python
import threading

class MonitorManager:
    def __init__(self):
        self._threads = []
    
    def start_monitor(self):
        thread = threading.Thread(target=monitor_function, daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def stop_all(self):
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=0.5)
        self._threads.clear()

# 在测试中使用
@pytest.fixture(autouse=True)
def cleanup_threads():
    yield
    # 停止所有后台线程
    for thread in threading.enumerate():
        if thread.name.lower().find('monitor') != -1:
            thread.join(timeout=0.5)
```

### 5. 缓存管理

#### ❌ 错误做法
```python
class CacheManager:
    def __init__(self):
        self._cache = {}
    
    def add(self, key, value):
        self._cache[key] = value
        # 没有清理机制
```

#### ✅ 正确做法
```python
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._max_size = 1000
    
    def add(self, key, value):
        if len(self._cache) >= self._max_size:
            # 清理最旧的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def clear(self):
        """提供清理方法"""
        self._cache.clear()
```

## 测试环境配置

### 1. 使用隔离环境

```python
@pytest.fixture
def isolated_environment():
    """提供完全隔离的测试环境"""
    os.environ.update({
        'PYTEST_CURRENT_TEST': 'isolated_memory_test',
        'DISABLE_HEAVY_IMPORTS': 'true',
        'ENABLE_MEMORY_OPTIMIZATION': 'true',
        'PROMETHEUS_ISOLATED': 'true'
    })
    
    isolated_registry = CollectorRegistry()
    yield isolated_registry
    
    # 环境清理
    memory_cleaner.run_aggressive_cleanup()
```

### 2. 内存监控

```python
@pytest.fixture
def memory_monitor():
    """内存监控fixture"""
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    yield {
        'initial_memory': initial_memory,
        'process': process
    }
    
    # 检查内存泄漏
    final_memory = process.memory_info().rss
    memory_diff = final_memory - initial_memory
    
    if memory_diff > 50 * 1024 * 1024:  # 50MB
        pytest.fail(f"检测到内存泄漏: {memory_diff / 1024 / 1024:.2f} MB")
```

### 3. 自动清理

```python
@pytest.fixture(autouse=True)
def aggressive_cleanup():
    """自动激进清理"""
    # 测试前清理
    memory_cleaner.run_aggressive_cleanup()
    
    yield
    
    # 测试后清理
    memory_cleaner.run_aggressive_cleanup()
```

## 检测工具使用

### 1. 内存泄漏检测

```bash
# 运行内存泄漏检测
python scripts/testing/run_infrastructure_tests.py detect

# 运行CI/CD检测
python scripts/testing/ci_memory_leak_detection.py --threshold 50
```

### 2. 内存清理

```bash
# 运行内存清理
python scripts/testing/run_infrastructure_tests.py cleanup

# 运行激进内存修复
python scripts/testing/aggressive_memory_fix.py
```

### 3. 综合测试

```bash
# 运行综合测试套件
python scripts/testing/run_infrastructure_tests.py comprehensive
```

## 代码审查要点

### 1. 单例模式检查
- [ ] 是否提供了清理方法
- [ ] 是否在测试后清理实例
- [ ] 是否使用了隔离环境

### 2. 资源管理检查
- [ ] 是否正确关闭文件/连接
- [ ] 是否正确停止线程
- [ ] 是否正确清理缓存

### 3. 导入优化检查
- [ ] 是否避免了大量import操作
- [ ] 是否使用了按需导入
- [ ] 是否设置了环境变量

### 4. 监控指标检查
- [ ] 是否使用了隔离的Prometheus注册表
- [ ] 是否正确清理了指标
- [ ] 是否避免了重复注册

## 常见问题解决

### 1. 内存泄漏检测失败

**问题**: 检测脚本无法运行
**解决**: 检查Python路径和依赖

```bash
# 检查Python路径
python -c "import sys; print(sys.path)"

# 检查依赖
pip install psutil prometheus_client
```

### 2. 测试超时

**问题**: 测试运行时间过长
**解决**: 使用更严格的超时设置

```python
# 设置更短的超时时间
@pytest.fixture
def short_timeout():
    return 30  # 30秒超时
```

### 3. 内存清理不彻底

**问题**: 清理后仍有内存泄漏
**解决**: 使用更激进的清理策略

```python
# 使用激进清理
from scripts.testing.aggressive_memory_fix import AggressiveMemoryFixer
fixer = AggressiveMemoryFixer()
fixer.run_aggressive_fix()
```

## 性能监控

### 1. 内存使用监控

```python
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"当前内存使用: {memory_mb:.2f} MB")
```

### 2. 泄漏检测监控

```python
from scripts.testing.comprehensive_memory_leak_detector import ComprehensiveMemoryLeakDetector

def check_memory_leaks():
    detector = ComprehensiveMemoryLeakDetector()
    detector.run_comprehensive_detection()
```

## 总结

通过遵循这些最佳实践，可以有效避免内存泄漏问题：

1. **使用隔离环境**: 避免影响全局状态
2. **正确管理单例**: 提供清理方法
3. **优化导入**: 按需导入，避免大量import
4. **管理线程**: 正确启动和停止线程
5. **清理缓存**: 定期清理缓存和资源
6. **监控内存**: 使用工具监控内存使用
7. **自动化检测**: 集成到CI/CD流程

遵循这些实践，可以确保代码的内存使用效率，避免内存泄漏问题。

