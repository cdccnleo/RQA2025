# CI/CD集成指南

## 概述

本指南介绍如何将RQA2025自动化测试流水线与持续集成/持续部署(CI/CD)流程集成，实现自动化测试的持续执行和质量保证。

## 核心特性

### 1. CI环境自动检测
- 自动识别CI/CD环境变量
- 根据环境动态调整配置
- 支持多种CI/CD平台

### 2. 多格式报告生成
- JSON格式：机器可读，便于CI工具解析
- XML格式：兼容JUnit等标准格式
- HTML格式：人类可读，支持浏览器查看

### 3. 工作流程集成
- 支持CI/CD流水线的各个阶段
- 自动环境配置和清理
- 失败处理和恢复机制

## 支持的CI/CD平台

### GitHub Actions
```yaml
name: 自动化测试
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: 设置Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: 安装依赖
        run: |
          pip install -r requirements.txt
      - name: 运行自动化测试
        run: |
          python -m pytest tests/unit/infrastructure/performance/test_cicd_integration.py
        env:
          CI: true
          GITHUB_RUN_NUMBER: ${{ github.run_number }}
```

### Jenkins
```groovy
pipeline {
    agent any
    environment {
        CI = 'true'
        BUILD_NUMBER = "${env.BUILD_NUMBER}"
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'python -m pytest tests/unit/infrastructure/performance/test_cicd_integration.py'
            }
        }
    }
}
```

### GitLab CI
```yaml
stages:
  - test
  - report

test:
  stage: test
  script:
    - pip install -r requirements.txt
    - python -m pytest tests/unit/infrastructure/performance/test_cicd_integration.py
  variables:
    CI: "true"
  artifacts:
    reports:
      junit: test-results.xml
```

## 快速开始

### 1. 基本CI/CD集成

```python
from src.infrastructure.performance.automated_test_runner import (
    create_test_suite, TestMode
)

# 检测CI环境
def is_ci_environment():
    return os.environ.get('CI') == 'true'

# 根据环境创建测试套件
if is_ci_environment():
    config = TestSuiteConfig(
        name="CI测试套件",
        test_mode=TestMode.UNIT,
        max_workers=1,  # CI环境下减少工作线程
        timeout=60,     # CI环境下减少超时时间
        parallel_execution=False,  # CI环境下禁用并行执行
        performance_monitoring=False  # CI环境下禁用性能监控
    )
else:
    config = TestSuiteConfig(
        name="本地测试套件",
        test_mode=TestMode.PERFORMANCE,
        max_workers=4,
        timeout=300,
        parallel_execution=True,
        performance_monitoring=True
    )

runner = create_test_suite("测试套件", config.test_mode, **config.__dict__)
```

### 2. CI环境变量处理

```python
import os

def get_ci_environment_info():
    """获取CI环境信息"""
    ci_info = {
        'is_ci': os.environ.get('CI') == 'true',
        'platform': None,
        'build_number': None,
        'branch': None,
        'commit': None
    }
    
    # 检测GitHub Actions
    if os.environ.get('GITHUB_ACTIONS'):
        ci_info['platform'] = 'GitHub Actions'
        ci_info['build_number'] = os.environ.get('GITHUB_RUN_NUMBER')
        ci_info['branch'] = os.environ.get('GITHUB_REF_NAME')
        ci_info['commit'] = os.environ.get('GITHUB_SHA')
    
    # 检测Jenkins
    elif os.environ.get('JENKINS_URL'):
        ci_info['platform'] = 'Jenkins'
        ci_info['build_number'] = os.environ.get('BUILD_NUMBER')
        ci_info['branch'] = os.environ.get('BRANCH_NAME')
        ci_info['commit'] = os.environ.get('COMMIT_HASH')
    
    # 检测Travis CI
    elif os.environ.get('TRAVIS'):
        ci_info['platform'] = 'Travis CI'
        ci_info['build_number'] = os.environ.get('TRAVIS_BUILD_NUMBER')
        ci_info['branch'] = os.environ.get('TRAVIS_BRANCH')
        ci_info['commit'] = os.environ.get('TRAVIS_COMMIT')
    
    return ci_info
```

### 3. 多格式报告生成

```python
def generate_ci_reports(results, output_dir="test-reports"):
    """生成CI友好的测试报告"""
    import os
    import json
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成JSON报告
    json_report = {
        'test_suite': 'CI测试套件',
        'timestamp': time.time(),
        'ci_environment': get_ci_environment_info(),
        'summary': {
            'total_tests': len(results),
            'passed_tests': len([r for r in results if r.status.value == 'passed']),
            'failed_tests': len([r for r in results if r.status.value == 'failed']),
            'success_rate': len([r for r in results if r.status.value == 'passed']) / len(results) * 100 if results else 0
        },
        'tests': [
            {
                'name': r.test_name,
                'status': r.status.value,
                'execution_time': r.execution_time,
                'error_message': r.error_message
            }
            for r in results
        ]
    }
    
    with open(os.path.join(output_dir, 'test-results.json'), 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    # 生成XML报告（JUnit格式）
    xml_report = generate_junit_xml(results)
    with open(os.path.join(output_dir, 'test-results.xml'), 'w', encoding='utf-8') as f:
        f.write(xml_report)
    
    # 生成HTML报告
    html_report = generate_html_report(results)
    with open(os.path.join(output_dir, 'test-results.html'), 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    return output_dir

def generate_junit_xml(results):
    """生成JUnit格式的XML报告"""
    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_lines.append('<testsuite>')
    
    for result in results:
        xml_lines.append(f'  <testcase name="{result.test_name}">')
        if result.status.value == 'failed':
            xml_lines.append(f'    <failure message="{result.error_message}"/>')
        xml_lines.append(f'    <system-out>执行时间: {result.execution_time:.3f}s</system-out>')
        xml_lines.append('  </testcase>')
    
    xml_lines.append('</testsuite>')
    return '\n'.join(xml_lines)
```

## 高级配置

### 1. CI环境特定配置

```python
def get_ci_optimized_config():
    """获取CI环境优化的配置"""
    base_config = {
        'name': 'CI优化测试套件',
        'test_mode': TestMode.UNIT,
        'max_workers': 1,
        'timeout': 60,
        'parallel_execution': False,
        'performance_monitoring': False,
        'cleanup_on_failure': True
    }
    
    # 根据CI平台进一步优化
    if os.environ.get('GITHUB_ACTIONS'):
        # GitHub Actions优化
        base_config.update({
            'max_workers': 2,  # GitHub Actions支持更多并发
            'timeout': 120     # 增加超时时间
        })
    elif os.environ.get('JENKINS_URL'):
        # Jenkins优化
        base_config.update({
            'max_workers': 1,  # Jenkins环境保守配置
            'timeout': 300     # 较长超时时间
        })
    
    return base_config
```

### 2. 测试阶段管理

```python
def run_ci_test_pipeline():
    """运行CI测试流水线"""
    stages = [
        ('单元测试', run_unit_tests),
        ('集成测试', run_integration_tests),
        ('性能测试', run_performance_tests)
    ]
    
    all_results = []
    stage_results = {}
    
    for stage_name, stage_func in stages:
        print(f"开始执行 {stage_name}...")
        
        try:
            results = stage_func()
            stage_results[stage_name] = results
            all_results.extend(results)
            
            # 检查阶段结果
            failed_tests = [r for r in results if r.status.value == 'failed']
            if failed_tests:
                print(f"⚠️  {stage_name} 有 {len(failed_tests)} 个测试失败")
                if stage_name == '单元测试':
                    print("❌ 单元测试失败，停止流水线")
                    return False
            else:
                print(f"✅ {stage_name} 全部通过")
        
        except Exception as e:
            print(f"❌ {stage_name} 执行异常: {e}")
            return False
    
    # 生成综合报告
    generate_ci_reports(all_results)
    
    # 返回总体结果
    total_failed = len([r for r in all_results if r.status.value == 'failed'])
    return total_failed == 0

def run_unit_tests():
    """运行单元测试"""
    runner = create_test_suite("单元测试", TestMode.UNIT, max_workers=1)
    
    # 添加单元测试
    runner.add_test("test_basic_functionality", lambda: "pass")
    runner.add_test("test_error_handling", lambda: "pass")
    
    return runner.run_tests()

def run_integration_tests():
    """运行集成测试"""
    runner = create_test_suite("集成测试", TestMode.INTEGRATION, max_workers=2)
    
    # 添加集成测试
    runner.add_test("test_component_integration", lambda: "pass")
    runner.add_test("test_data_flow", lambda: "pass")
    
    return runner.run_tests()

def run_performance_tests():
    """运行性能测试"""
    runner = create_test_suite("性能测试", TestMode.PERFORMANCE, max_workers=1)
    
    # 添加性能测试
    def performance_test():
        import time
        start_time = time.time()
        time.sleep(0.1)  # 模拟工作负载
        execution_time = time.time() - start_time
        assert execution_time < 0.2, f"性能测试执行时间过长: {execution_time:.3f}s"
        return "performance_ok"
    
    runner.add_test("test_performance_baseline", performance_test)
    
    return runner.run_tests()
```

### 3. 失败处理和重试

```python
def run_tests_with_retry(test_func, max_retries=3):
    """带重试的测试执行"""
    for attempt in range(max_retries):
        try:
            results = test_func()
            
            # 检查是否有失败的测试
            failed_tests = [r for r in results if r.status.value == 'failed']
            if not failed_tests:
                return results
            
            print(f"第 {attempt + 1} 次尝试有 {len(failed_tests)} 个测试失败")
            
            if attempt < max_retries - 1:
                print("等待重试...")
                time.sleep(5)  # 等待5秒后重试
            else:
                print("达到最大重试次数，测试失败")
                return results
        
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试异常: {e}")
            if attempt < max_retries - 1:
                print("等待重试...")
                time.sleep(5)
            else:
                raise
    
    return []
```

## 监控和告警

### 1. 测试执行监控

```python
def monitor_test_execution(runner):
    """监控测试执行状态"""
    import threading
    import time
    
    def monitor_loop():
        while True:
            status = runner.get_execution_status()
            
            print(f"测试状态: {status['completed_tests']}/{status['total_tests']} 完成")
            
            if status['stop_requested'] or status['completed_tests'] == status['total_tests']:
                break
            
            time.sleep(5)  # 每5秒检查一次
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    return monitor_thread
```

### 2. 性能指标收集

```python
def collect_performance_metrics(results):
    """收集性能指标"""
    metrics = {
        'total_tests': len(results),
        'total_execution_time': sum(r.execution_time for r in results),
        'average_execution_time': sum(r.execution_time for r in results) / len(results) if results else 0,
        'performance_tests': len([r for r in results if r.performance_metrics]),
        'cpu_usage': [],
        'memory_usage': []
    }
    
    for result in results:
        if result.performance_metrics:
            if 'cpu_usage' in result.performance_metrics:
                metrics['cpu_usage'].append(result.performance_metrics['cpu_usage'])
            if 'memory_usage' in result.performance_metrics:
                metrics['memory_usage'].append(result.performance_metrics['memory_usage'])
    
    if metrics['cpu_usage']:
        metrics['avg_cpu_usage'] = sum(metrics['cpu_usage']) / len(metrics['cpu_usage'])
    if metrics['memory_usage']:
        metrics['avg_memory_usage'] = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
    
    return metrics
```

## 最佳实践

### 1. CI环境优化

- **减少并行度**: CI环境下使用较少的工作线程
- **缩短超时时间**: 设置合理的超时时间，避免长时间等待
- **禁用性能监控**: CI环境下通常不需要详细的性能监控
- **快速失败**: 单元测试失败时立即停止流水线

### 2. 报告管理

- **多格式输出**: 同时生成JSON、XML、HTML格式报告
- **CI工具集成**: 确保报告能被CI工具正确解析
- **历史记录**: 保存测试结果历史，便于趋势分析
- **失败详情**: 提供详细的失败信息和调试建议

### 3. 错误处理

- **优雅降级**: 非关键测试失败时继续执行
- **重试机制**: 对偶发性失败实施重试策略
- **环境清理**: 确保测试环境在失败后能正确清理
- **日志记录**: 详细记录执行过程和错误信息

### 4. 性能优化

- **资源控制**: 根据CI环境限制资源使用
- **缓存策略**: 利用CI工具的缓存机制
- **并行优化**: 在资源允许的情况下使用并行执行
- **增量测试**: 只运行受影响的测试

## 故障排除

### 1. 常见问题

**问题**: 测试在CI环境中超时
```python
# 解决方案：调整CI环境配置
ci_config = TestSuiteConfig(
    name="CI测试套件",
    timeout=300,  # 增加超时时间
    max_workers=1,  # 减少工作线程
    parallel_execution=False  # 禁用并行执行
)
```

**问题**: 内存使用过高
```python
# 解决方案：优化内存使用
ci_config = TestSuiteConfig(
    name="内存优化测试套件",
    max_workers=1,  # 单线程执行
    performance_monitoring=False,  # 禁用性能监控
    cleanup_on_failure=True  # 启用失败清理
)
```

**问题**: 测试环境冲突
```python
# 解决方案：使用独立的测试环境
def create_isolated_test_environment():
    """创建隔离的测试环境"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 设置环境变量
        os.environ['TEST_TEMP_DIR'] = temp_dir
        
        # 运行测试
        results = run_tests()
        
        return results
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
```

### 2. 调试技巧

```python
def debug_ci_environment():
    """调试CI环境"""
    print("=== CI环境信息 ===")
    print(f"CI: {os.environ.get('CI')}")
    print(f"Platform: {os.environ.get('PLATFORM')}")
    print(f"Build Number: {os.environ.get('BUILD_NUMBER')}")
    print(f"Branch: {os.environ.get('BRANCH_NAME')}")
    print(f"Commit: {os.environ.get('COMMIT_HASH')}")
    
    print("\n=== 系统信息 ===")
    import platform
    print(f"OS: {platform.system()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    
    print("\n=== 资源信息 ===")
    import psutil
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
```

## 总结

通过本指南，您可以成功将RQA2025自动化测试流水线与各种CI/CD平台集成，实现：

- **自动化测试执行**: 代码提交时自动运行测试
- **质量门禁**: 测试失败时阻止代码合并
- **持续反馈**: 及时发现问题并修复
- **质量保证**: 确保代码质量和系统稳定性

遵循最佳实践，合理配置CI环境，您将能够构建一个高效、可靠的持续测试体系。
