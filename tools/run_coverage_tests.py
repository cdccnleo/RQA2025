#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage测试运行脚本
专门用于解决coverage统计问题的测试运行器
"""

import os
import sys
import subprocess
import coverage
import importlib.util
import importlib.machinery


def setup_coverage():
    """设置coverage环境"""
    # 确保src目录在Python路径中
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # 创建coverage实例
    cov = coverage.Coverage(
        source=['src'],
        omit=[
            '*/tests/*',
            '*/test_*',
            '*/scripts/*',
            '*/temp_*',
            '*/__pycache__/*',
            '*.pyc',
            'setup.py'
        ],
        branch=True,
        concurrency='thread'
    )

    return cov


def run_module_with_coverage(module_path, cov):
    """使用coverage运行模块"""
    try:
        # 解析模块路径
        if module_path.startswith('src/'):
            module_name = module_path[4:].replace('/', '.').replace('.py', '')
        else:
            module_name = module_path.replace('/', '.').replace('.py', '')

        print(f"Loading module: {module_name}")

        # 直接加载模块文件
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            print(f"Cannot load module: {module_path}")
            return False

        module = importlib.util.module_from_spec(spec)

        # 在coverage跟踪下执行模块
        cov.start()
        try:
            spec.loader.exec_module(module)
            print(f"Successfully loaded: {module_name}")
            return True
        finally:
            cov.stop()
            cov.save()

    except Exception as e:
        print(f"Error loading {module_path}: {e}")
        return False


def run_test_with_coverage(test_file, cov):
    """使用coverage运行测试"""
    try:
        # 启动coverage
        cov.start()

        # 运行pytest
        cmd = [
            sys.executable, '-m', 'pytest',
            test_file,
            '-v', '--tb=short',
            '--maxfail=5'
        ]

        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())

        return result.returncode == 0

    finally:
        # 停止coverage
        cov.stop()
        cov.save()


def generate_coverage_report(cov):
    """生成coverage报告"""
    try:
        # 生成报告
        cov.report(show_missing=True)
        cov.html_report(directory='htmlcov')
        cov.json_report(outfile='coverage.json')

        print("\nCoverage report generated:")
        print("- HTML report: htmlcov/index.html")
        print("- JSON report: coverage.json")

    except Exception as e:
        print(f"Error generating report: {e}")


def run_coverage_tests():
    """运行coverage测试的主函数"""
    print("Setting up coverage environment...")

    # 初始化coverage
    cov = setup_coverage()

    # 需要测试的模块和对应的测试文件
    test_modules = [
        ('src/aliases.py', 'tests/unit/test_aliases.py'),
        ('src/async/components/health_checker.py', 'tests/unit/infrastructure/test_health_checker.py'),
        ('src/async/components/infra_processor.py', 'tests/unit/infrastructure/test_infra_processor.py'),
        ('src/async/components/monitoring_processor.py',
         'tests/unit/infrastructure/test_monitoring_processor.py'),
        ('src/async/components/system_processor.py',
         'tests/unit/infrastructure/test_system_processor.py'),
    ]

    success_count = 0
    total_count = len(test_modules)

    for module_path, test_file in test_modules:
        print(f"\n{'='*60}")
        print(f"Testing: {module_path}")
        print(f"Test file: {test_file}")
        print(f"{'='*60}")

        # 检查文件是否存在
        if not os.path.exists(module_path):
            print(f"Module file not found: {module_path}")
            continue

        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            continue

        # 先运行模块以确保coverage能跟踪到
        print(f"Loading module for coverage tracking...")
        module_loaded = run_module_with_coverage(module_path, cov)

        # 运行测试
        print(f"Running tests...")
        test_success = run_test_with_coverage(test_file, cov)

        if test_success:
            print(f"✓ Tests passed for {module_path}")
            success_count += 1
        else:
            print(f"✗ Tests failed for {module_path}")

    # 生成最终报告
    print(f"\n{'='*60}")
    print("Generating coverage report...")
    print(f"{'='*60}")

    generate_coverage_report(cov)

    # 统计结果
    print(f"\n{'='*60}")
    print("Coverage Test Summary:")
    print(f"{'='*60}")
    print(f"Total modules tested: {total_count}")
    print(f"Successful tests: {success_count}")
    print(f"Failed tests: {total_count - success_count}")
    print(".1f")

    return success_count == total_count


def run_standard_coverage():
    """运行标准的coverage测试"""
    print("Running standard pytest with coverage...")

    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/unit/infrastructure/',
        'tests/unit/test_aliases.py',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=json:coverage.json',
        '-v', '--tb=short',
        '--maxfail=5'
    ]

    result = subprocess.run(cmd, cwd=os.getcwd())

    return result.returncode == 0


if __name__ == '__main__':
    print("RQA2025 Coverage Test Runner")
    print("=" * 60)

    # 尝试运行专门的coverage测试
    print("Attempting specialized coverage testing...")
    success = run_coverage_tests()

    if not success:
        print("\nFalling back to standard coverage testing...")
        success = run_standard_coverage()

    if success:
        print("\n✓ Coverage testing completed successfully!")
    else:
        print("\n✗ Coverage testing failed!")
        sys.exit(1)
