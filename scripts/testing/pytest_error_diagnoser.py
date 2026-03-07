#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest错误诊断脚本
精确捕获pytest收集和运行时的具体错误信息
"""
import sys
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DIR = PROJECT_ROOT / 'tests' / 'unit' / 'infrastructure'
TIMEOUT = 60


def run_pytest_with_detailed_output(options=None, test_path=None):
    """运行pytest并捕获详细输出"""
    if options is None:
        options = []
    if test_path is None:
        test_path = str(TEST_DIR)

    cmd = [sys.executable, '-m', 'pytest'] + options + [test_path]

    print(f"\n🔍 执行命令: {' '.join(cmd)}")
    print("-" * 80)

    try:
        # 使用subprocess.Popen实时输出
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        # 设置超时
        timer = threading.Timer(TIMEOUT, proc.kill)
        timer.start()

        # 实时读取输出
        stdout_lines = []
        stderr_lines = []

        while True:
            stdout_line = proc.stdout.readline()
            stderr_line = proc.stderr.readline()

            if stdout_line:
                print(f"STDOUT: {stdout_line.rstrip()}")
                stdout_lines.append(stdout_line)

            if stderr_line:
                print(f"STDERR: {stderr_line.rstrip()}")
                stderr_lines.append(stderr_line)

            # 检查进程是否结束
            if proc.poll() is not None:
                # 读取剩余输出
                remaining_stdout, remaining_stderr = proc.communicate()
                if remaining_stdout:
                    print(f"STDOUT: {remaining_stdout}")
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    print(f"STDERR: {remaining_stderr}")
                    stderr_lines.append(remaining_stderr)
                break

        timer.cancel()

        return {
            'returncode': proc.returncode,
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines),
            'success': proc.returncode == 0
        }

    except Exception as e:
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def diagnose_collection_errors():
    """诊断收集阶段错误"""
    print("🔍 诊断pytest收集阶段错误...")

    # 测试1: 基本收集
    print("\n1️⃣ 测试基本收集:")
    result1 = run_pytest_with_detailed_output(['--collect-only', '-v'])

    # 测试2: 详细错误信息
    print("\n2️⃣ 测试详细错误信息:")
    result2 = run_pytest_with_detailed_output(['--collect-only', '-v', '--tb=long'])

    # 测试3: 禁用插件
    print("\n3️⃣ 测试禁用插件:")
    result3 = run_pytest_with_detailed_output(['--collect-only', '-v', '--disable-warnings'])

    # 测试4: 单个文件测试
    print("\n4️⃣ 测试单个文件:")
    test_files = list(TEST_DIR.rglob('test_*.py'))
    if test_files:
        first_file = str(test_files[0])
        print(f"   测试文件: {first_file}")
        result4 = run_pytest_with_detailed_output(['--collect-only', '-v'], first_file)
    else:
        result4 = {'success': False, 'stderr': '未找到测试文件'}

    return {
        'basic_collection': result1,
        'detailed_errors': result2,
        'no_plugins': result3,
        'single_file': result4
    }


def analyze_error_patterns(results):
    """分析错误模式"""
    print("\n" + "="*80)
    print("📊 错误分析报告")
    print("="*80)

    error_patterns = []

    for test_name, result in results.items():
        print(f"\n🔍 {test_name}:")
        print(f"  成功: {'✅' if result['success'] else '❌'}")
        print(f"  返回码: {result.get('returncode', 'N/A')}")

        if not result['success']:
            stderr = result.get('stderr', '')
            if stderr:
                print(f"  错误信息: {stderr[:500]}...")

                # 分析错误模式
                if 'ImportError' in stderr:
                    error_patterns.append('ImportError')
                if 'ModuleNotFoundError' in stderr:
                    error_patterns.append('ModuleNotFoundError')
                if 'SyntaxError' in stderr:
                    error_patterns.append('SyntaxError')
                if 'IndentationError' in stderr:
                    error_patterns.append('IndentationError')
                if 'AttributeError' in stderr:
                    error_patterns.append('AttributeError')
                if 'TypeError' in stderr:
                    error_patterns.append('TypeError')
                if 'UnicodeDecodeError' in stderr:
                    error_patterns.append('UnicodeDecodeError')

    # 总结错误模式
    if error_patterns:
        print(f"\n🎯 发现的主要错误类型:")
        for pattern in set(error_patterns):
            count = error_patterns.count(pattern)
            print(f"  - {pattern}: {count} 次")

    return error_patterns


def suggest_fixes(error_patterns):
    """根据错误模式提供修复建议"""
    print("\n🔧 修复建议:")

    if 'ImportError' in error_patterns:
        print("1. ImportError: 检查模块路径和依赖安装")
        print("   - 确保所有依赖已正确安装")
        print("   - 检查PYTHONPATH环境变量")
        print("   - 验证相对导入路径")

    if 'ModuleNotFoundError' in error_patterns:
        print("2. ModuleNotFoundError: 模块未找到")
        print("   - 检查模块是否在正确的Python环境中")
        print("   - 验证conda环境激活状态")

    if 'SyntaxError' in error_patterns or 'IndentationError' in error_patterns:
        print("3. 语法错误: 检查测试文件语法")
        print("   - 运行 python -m py_compile <file> 检查语法")
        print("   - 检查缩进和括号匹配")

    if 'UnicodeDecodeError' in error_patterns:
        print("4. 编码错误: 文件编码问题")
        print("   - 确保所有文件使用UTF-8编码")
        print("   - 检查文件中的特殊字符")

    print("\n5. 通用修复步骤:")
    print("   - 运行: python -c 'import pytest; print(pytest.__version__)'")
    print("   - 检查: conda list | grep pytest")
    print("   - 尝试: python -m pytest --version")


def main():
    print("🚀 开始pytest错误诊断...")

    # 诊断收集错误
    results = diagnose_collection_errors()

    # 分析错误模式
    error_patterns = analyze_error_patterns(results)

    # 提供修复建议
    suggest_fixes(error_patterns)

    print("\n✅ 诊断完成！")


if __name__ == "__main__":
    main()
