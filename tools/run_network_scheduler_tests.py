#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network和Scheduler模块测试运行脚本
运行所有Network和Scheduler模块的测试用例
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_file(test_file):
    """运行单个测试文件"""
    print(f"\n{'='*60}")
    print(f"运行测试文件: {test_file}")
    print(f"{'='*60}")

    try:
        # 使用pytest运行测试
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--timeout=300"  # 5分钟超时
        ], capture_output=True, text=True, timeout=600)  # 10分钟总超时

        if result.returncode == 0:
            print("✅ 测试通过")
            return True
        else:
            print("❌ 测试失败")
            print("错误输出:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"💥 运行测试时出错: {e}")
        return False


def run_network_tests():
    """运行Network模块测试"""
    print("\n" + "🚀"*20 + " 开始运行Network模块测试 " + "🚀"*20)

    network_tests = [
        "tests/unit/infrastructure/network/test_network_manager.py",
        "tests/unit/infrastructure/network/test_connection_pool.py",
        "tests/unit/infrastructure/network/test_load_balancer.py",
        "tests/unit/infrastructure/network/test_retry_policy.py",
        "tests/unit/infrastructure/network/test_network_monitor.py"
    ]

    passed = 0
    failed = 0

    for test_file in network_tests:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
            failed += 1

    return passed, failed


def run_scheduler_tests():
    """运行Scheduler模块测试"""
    print("\n" + "⚡"*20 + " 开始运行Scheduler模块测试 " + "⚡"*20)

    scheduler_tests = [
        "tests/unit/infrastructure/scheduler/test_task_scheduler.py",
        "tests/unit/infrastructure/scheduler/test_priority_queue.py",
        "tests/unit/infrastructure/scheduler/test_job_scheduler.py",
        "tests/unit/infrastructure/scheduler/test_scheduler_manager.py"
    ]

    passed = 0
    failed = 0

    for test_file in scheduler_tests:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  测试文件不存在: {test_file}")
            failed += 1

    return passed, failed


def run_integration_tests():
    """运行集成测试"""
    print("\n" + "🔗"*20 + " 开始运行集成测试 " + "🔗"*20)

    integration_tests = [
        "tests/integration/test_network_scheduler_integration.py"
    ]

    passed = 0
    failed = 0

    for test_file in integration_tests:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"⚠️  集成测试文件不存在: {test_file}")

    return passed, failed


def generate_test_report(network_results, scheduler_results, integration_results):
    """生成测试报告"""
    print("\n" + "📊"*20 + " 测试报告 " + "📊"*20)

    network_passed, network_failed = network_results
    scheduler_passed, scheduler_failed = scheduler_results
    integration_passed, integration_failed = integration_results

    total_passed = network_passed + scheduler_passed + integration_passed
    total_failed = network_failed + scheduler_failed + integration_failed
    total_tests = total_passed + total_failed

    print(f"\n📈 测试统计:")
    print(f"   Network模块: {network_passed} 通过, {network_failed} 失败")
    print(f"   Scheduler模块: {scheduler_passed} 通过, {scheduler_failed} 失败")
    print(f"   集成测试: {integration_passed} 通过, {integration_failed} 失败")
    print(f"   总计: {total_passed} 通过, {total_failed} 失败")

    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"   成功率: {success_rate:.1f}%")

        if success_rate >= 90:
            print("   🎉 测试结果优秀!")
        elif success_rate >= 80:
            print("   ✅ 测试结果良好")
        elif success_rate >= 70:
            print("   ⚠️  测试结果一般")
        else:
            print("   ❌ 测试结果需要改进")

    return total_passed, total_failed, total_tests


def main():
    """主函数"""
    print("🚀 Network和Scheduler模块测试运行器")
    print("="*60)

    start_time = time.time()

    # 运行Network模块测试
    network_results = run_network_tests()

    # 运行Scheduler模块测试
    scheduler_results = run_scheduler_tests()

    # 运行集成测试
    integration_results = run_integration_tests()

    # 生成测试报告
    total_passed, total_failed, total_tests = generate_test_report(
        network_results, scheduler_results, integration_results
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n⏱️  总执行时间: {execution_time:.2f} 秒")

    # 返回退出码
    if total_failed == 0:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n❌ 有 {total_failed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
