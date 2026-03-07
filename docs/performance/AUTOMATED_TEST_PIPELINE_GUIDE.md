# 自动化测试流水线使用指南

## 概述

自动化测试流水线是RQA2025基础设施层性能模块的核心组件，它集成了测试优化器和框架集成器，提供完整的自动化测试解决方案。该流水线支持多种测试模式，包括单元测试、性能测试、集成测试和压力测试，并具备并行执行、性能监控、失败恢复等高级功能。

## 核心组件

### 1. AutomatedTestRunner (自动化测试运行器)

主要的测试执行引擎，负责：
- 测试队列管理
- 测试环境配置
- 并行/顺序执行控制
- 性能监控集成
- 失败处理和恢复

### 2. TestSuiteConfig (测试套件配置)

配置测试套件的各种参数：
- 测试模式选择
- 工作线程数量
- 超时设置
- 执行策略配置

### 3. TestResult (测试结果)

记录每个测试的详细信息：
- 执行状态
- 时间统计
- 错误信息
- 性能指标

## 快速开始

### 基本用法

```python
from src.infrastructure.performance.automated_test_runner import (
    create_test_suite, TestMode
)

# 创建测试套件
runner = create_test_suite("我的测试套件", TestMode.PERFORMANCE)

# 添加测试
def test_function1():
    # 测试逻辑
    return "result1"

def test_function2():
    # 测试逻辑
    return "result2"

runner.add_test("test1", test_function1)
runner.add_test("test2", test_function2)

# 运行测试
results = runner.run_tests()

# 查看结果
for result in results:
    print(f"{result.test_name}: {result.status.value}")
```

### 便捷函数

```python
from src.infrastructure.performance.automated_test_runner import (
    run_performance_tests, run_integration_tests, run_stress_tests
)

# 定义测试
tests = [
    ("performance_test", lambda: "performance_result"),
    ("integration_test", lambda: "integration_result"),
    ("stress_test", lambda: "stress_result")
]

# 快速运行不同类型的测试
performance_results = run_performance_tests(tests)
integration_results = run_integration_tests(tests)
stress_results = run_stress_tests(tests)
```

## 高级功能

### 1. 并行执行

```python
# 配置并行执行
config = TestSuiteConfig(
    name="并行测试套件",
    test_mode=TestMode.PERFORMANCE,
    max_workers=4,  # 4个工作线程
    parallel_execution=True
)

runner = AutomatedTestRunner(config)
```

### 2. 性能监控

```python
# 启用性能监控
config = TestSuiteConfig(
    name="性能监控测试",
    performance_monitoring=True  # 默认启用
)

runner = AutomatedTestRunner(config)

# 测试结果将包含性能指标
results = runner.run_tests()
for result in results:
    if result.performance_metrics:
        print(f"CPU使用率: {result.performance_metrics['cpu_usage']}%")
        print(f"内存使用率: {result.performance_metrics['memory_usage']}%")
```

### 3. 失败恢复

```python
# 配置失败恢复
config = TestSuiteConfig(
    name="失败恢复测试",
    cleanup_on_failure=True,  # 失败时自动清理
    retry_count=2  # 重试次数
)

runner = AutomatedTestRunner(config)
```

### 4. 测试模式配置

```python
# 不同测试模式的配置
unit_config = TestSuiteConfig(
    name="单元测试",
    test_mode=TestMode.UNIT,
    max_workers=2,
    timeout=60
)

performance_config = TestSuiteConfig(
    name="性能测试",
    test_mode=TestMode.PERFORMANCE,
    max_workers=4,
    timeout=300
)

integration_config = TestSuiteConfig(
    name="集成测试",
    test_mode=TestMode.INTEGRATION,
    max_workers=3,
    timeout=180
)

stress_config = TestSuiteConfig(
    name="压力测试",
    test_mode=TestMode.STRESS,
    max_workers=6,
    timeout=600
)
```

## 测试执行流程

### 1. 环境设置阶段

```python
def _setup_test_environment(self):
    # 应用测试优化配置
    self.test_optimizer.apply_optimizations()
    
    # 设置框架集成器环境
    self.framework_integrator.setup_test_environment(self.config.test_mode)
```

### 2. 测试执行阶段

```python
def _execute_single_test(self, test_result):
    # 获取测试函数
    test_func = test_result.metadata.get('test_func')
    
    # 执行测试（带性能监控）
    if self.config.performance_monitoring:
        result = self.framework_integrator.run_optimized_performance_test(
            test_result.test_name, test_func, test_result.test_mode
        )
        test_result.performance_metrics = result.get('performance_metrics')
    else:
        # 直接执行
        test_func()
```

### 3. 环境清理阶段

```python
def _cleanup_test_environment(self):
    # 恢复测试优化配置
    self.test_optimizer.restore_optimizations()
    
    # 清理框架集成器环境
    self.framework_integrator.cleanup_test_environment()
```

## 监控和报告

### 1. 执行状态监控

```python
# 获取执行状态
status = runner.get_execution_status()
print(f"总测试数: {status['total_tests']}")
print(f"运行中: {status['running_tests']}")
print(f"已完成: {status['completed_tests']}")
print(f"待执行: {status['pending_tests']}")
```

### 2. 测试报告生成

测试完成后会自动生成详细的执行报告：

```
============================================================
测试执行报告
============================================================
测试套件: 性能测试套件
总测试数: 5
通过: 4
失败: 1
跳过: 0
成功率: 80.0%
总执行时间: 12.345s
平均执行时间: 2.469s
============================================================

失败的测试:
  - test_failure: 测试函数异常
```

### 3. 性能指标分析

```python
# 分析性能指标
def analyze_performance(results):
    cpu_usage = []
    memory_usage = []
    execution_times = []
    
    for result in results:
        if result.performance_metrics:
            cpu_usage.append(result.performance_metrics['cpu_usage'])
            memory_usage.append(result.performance_metrics['memory_usage'])
        execution_times.append(result.execution_time)
    
    print(f"平均CPU使用率: {sum(cpu_usage)/len(cpu_usage):.1f}%")
    print(f"平均内存使用率: {sum(memory_usage)/len(memory_usage):.1f}%")
    print(f"平均执行时间: {sum(execution_times)/len(execution_times):.3f}s")
```

## 最佳实践

### 1. 测试组织

```python
# 按功能模块组织测试
def create_module_test_suite(module_name: str):
    return create_test_suite(
        name=f"{module_name}模块测试",
        test_mode=TestMode.UNIT,
        max_workers=2
    )

# 创建不同模块的测试套件
data_tests = create_module_test_suite("数据层")
business_tests = create_module_test_suite("业务层")
infrastructure_tests = create_module_test_suite("基础设施层")
```

### 2. 错误处理

```python
# 自定义错误处理
def robust_test_function():
    try:
        # 测试逻辑
        result = perform_operation()
        assert result is not None
        return result
    except Exception as e:
        logger.error(f"测试执行异常: {e}")
        raise
```

### 3. 资源管理

```python
# 使用上下文管理器确保资源清理
with runner.test_context(TestMode.PERFORMANCE):
    # 测试执行
    results = runner.run_tests()
    # 自动清理环境
```

### 4. 性能基准

```python
# 设置性能基准
def performance_baseline_test():
    start_time = time.time()
    result = expensive_operation()
    execution_time = time.time() - start_time
    
    # 性能基准检查
    assert execution_time < 1.0, f"执行时间过长: {execution_time:.3f}s"
    return result
```

## 故障排除

### 1. 常见问题

**问题**: 测试执行超时
```python
# 解决方案：增加超时时间
config = TestSuiteConfig(
    name="长耗时测试",
    timeout=600  # 10分钟超时
)
```

**问题**: 内存使用过高
```python
# 解决方案：减少并行度
config = TestSuiteConfig(
    name="内存敏感测试",
    max_workers=2,  # 减少工作线程
    parallel_execution=False  # 禁用并行执行
)
```

**问题**: 测试环境配置冲突
```python
# 解决方案：使用独立的测试环境
runner1 = create_test_suite("测试套件1", TestMode.UNIT)
runner2 = create_test_suite("测试套件2", TestMode.PERFORMANCE)

# 分别运行，避免环境冲突
results1 = runner1.run_tests()
results2 = runner2.run_tests()
```

### 2. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查测试状态
print(f"测试队列: {len(runner.test_results)}")
print(f"集成状态: {runner.framework_integrator.get_integration_status()}")

# 单步执行测试
for test_result in runner.test_results:
    print(f"执行测试: {test_result.test_name}")
    runner._execute_single_test(test_result)
    print(f"测试结果: {test_result.status.value}")
```

## 扩展和定制

### 1. 自定义测试结果处理器

```python
class CustomTestResult(TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics = {}
    
    def add_custom_metric(self, key: str, value: Any):
        self.custom_metrics[key] = value

# 在测试中添加自定义指标
def test_with_custom_metrics():
    result = CustomTestResult(...)
    result.add_custom_metric("api_calls", 100)
    result.add_custom_metric("cache_hits", 85)
    return result
```

### 2. 自定义执行策略

```python
class CustomExecutionStrategy:
    def execute_tests(self, runner, test_results):
        # 自定义执行逻辑
        pass

# 集成自定义策略
runner.execution_strategy = CustomExecutionStrategy()
```

## 总结

自动化测试流水线为RQA2025提供了强大而灵活的测试执行能力。通过合理配置和使用，可以显著提高测试效率、稳定性和可维护性。该流水线特别适合：

- 大规模测试套件的执行
- 性能测试和基准测试
- 持续集成和部署流程
- 复杂系统的集成测试
- 压力测试和稳定性验证

通过遵循本指南的最佳实践，您可以充分利用自动化测试流水线的功能，构建高质量的测试体系。
