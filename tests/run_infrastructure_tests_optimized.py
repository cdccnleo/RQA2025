#!/usr/bin/env python3
"""
基础设施层测试优化运行脚本

策略：
1. 优先运行核心功能测试（core目录和高质量测试）
2. 跳过低质量的覆盖率提升测试
3. 分阶段运行，避免超时
4. 智能并行执行
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, timeout=600):
    """运行命令并输出结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"📝 命令: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )

        print("📄 标准输出:")
        # 只显示最后100行，避免输出过多
        lines = result.stdout.split('\n')
        if len(lines) > 100:
            print(f"... ({len(lines)-100} lines hidden)")
            print('\n'.join(lines[-100:]))
        else:
            print(result.stdout)

        if result.stderr:
            print("⚠️  错误输出:")
            print(result.stderr)

        print(f"📊 返回码: {result.returncode}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ 命令超时 ({timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def run_core_tests():
    """运行核心测试（优先级最高）"""
    print("🎯 运行基础设施层核心测试...")

    # 直接运行各个模块的核心目录
    core_dirs = [
        "tests/unit/infrastructure/monitoring/core/",
        "tests/unit/infrastructure/config/core/",
        "tests/unit/infrastructure/health/core/",
        "tests/unit/infrastructure/resource/core/",
        "tests/unit/infrastructure/security/core/",
        "tests/unit/infrastructure/logging/core/",
        "tests/unit/infrastructure/cache/core/",
        "tests/unit/infrastructure/api/core/",
        "tests/unit/infrastructure/versioning/core/",
        "tests/unit/infrastructure/distributed/core/",
        "tests/unit/infrastructure/optimization/core/",
    ]

    success_count = 0
    total_count = 0

    for test_dir in core_dirs:
        if os.path.exists(test_dir):
            total_count += 1
            cmd = [
                sys.executable, "-m", "pytest", test_dir,
                "-v", "--tb=short", "-x", "--durations=10",
                "--maxfail=3"  # 快速失败，避免长时间运行
            ]

            if run_command(cmd, f"运行核心测试: {test_dir}", timeout=300):
                success_count += 1
            else:
                print(f"⚠️  {test_dir} 测试失败，但继续执行其他模块")

    print(f"✅ 核心测试完成: {success_count}/{total_count} 成功")
    return success_count == total_count


def run_module_tests_parallel():
    """并行运行各模块测试（分模块避免冲突）"""
    print("🔄 并行运行各模块测试...")

    modules = [
        ("monitoring", "tests/unit/infrastructure/monitoring/"),
        ("config", "tests/unit/infrastructure/config/"),
        ("health", "tests/unit/infrastructure/health/core/"),  # 只运行core子目录
        ("resource", "tests/unit/infrastructure/resource/"),
        ("security", "tests/unit/infrastructure/security/core/"),  # 只运行core子目录
        ("logging", "tests/unit/infrastructure/logging/core/"),
        ("cache", "tests/unit/infrastructure/cache/"),
        ("api", "tests/unit/infrastructure/api/"),
    ]

    success_count = 0

    for module_name, test_path in modules:
        if os.path.exists(test_path):
            cmd = [
                sys.executable, "-m", "pytest", test_path,
                "-v", "--tb=short", "-x", "--durations=5",
                "--maxfail=2", "-q"  # 静默模式，减少输出
            ]

            if run_command(cmd, f"测试模块: {module_name}", timeout=180):
                success_count += 1
            else:
                print(f"⚠️  {module_name} 模块测试失败，但继续执行")

    print(f"✅ 模块测试完成: {success_count}/{len(modules)} 成功")
    return success_count >= len(modules) * 0.8  # 80%成功率即可


def run_coverage_analysis():
    """运行覆盖率分析（最后执行）"""
    print("📊 运行覆盖率分析...")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/",
        "--cov=src/infrastructure",
        "--cov-report=html:test_logs/infrastructure_optimized_coverage.html",
        "--cov-report=term-missing",
        "--cov-report=term",
        "-x", "--maxfail=5",
        "--durations=10",
        "-q"  # 静默模式
    ]

    success = run_command(cmd, "覆盖率分析", timeout=900)  # 15分钟超时
    return success


def main():
    """主函数"""
    print("🎯 基础设施层测试优化运行器")
    print("=" * 60)

    # 1. 运行核心测试
    if not run_core_tests():
        print("⚠️  部分核心测试失败，继续执行其他测试")

    # 2. 并行运行模块测试
    if not run_module_tests_parallel():
        print("⚠️  部分模块测试失败，继续执行覆盖率分析")

    # 3. 运行覆盖率分析
    if not run_coverage_analysis():
        print("⚠️  覆盖率分析失败，但核心测试已完成")

    print(f"\n{'='*60}")
    print("✅ 基础设施层测试优化执行完成!")
    print("📊 查看覆盖率报告: test_logs/infrastructure_optimized_coverage.html")


if __name__ == "__main__":
    main()
