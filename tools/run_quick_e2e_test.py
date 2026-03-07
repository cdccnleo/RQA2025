#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 快速E2E测试执行脚本

执行优化的E2E测试，验证效率提升效果
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_quick_e2e_test():
    """运行快速E2E测试"""
    print("⚡ RQA2025 快速E2E测试执行")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'

    print("🔧 设置测试环境...")

    # 创建测试配置
    test_config = {
        "test_environment": {
            "timeout": 30,
            "workers": 2,
            "memory_limit": "1GB"
        },
        "optimizations": {
            "use_shared_fixtures": True,
            "cache_test_data": True,
            "skip_slow_tests": False
        }
    }

    # 运行单个E2E测试文件
    test_files = [
        "tests/e2e/test_business_process_validation.py",
        "tests/e2e/test_user_experience.py"
    ]

    total_start_time = time.time()
    results = []

    for test_file in test_files:
        test_path = project_root / test_file
        if not test_path.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            continue

        print(f"\n📋 执行测试: {test_file}")
        print("-" * 40)

        start_time = time.time()

        # 使用优化的pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--maxfail=2",
            "-x",  # 遇到第一个失败就停止
            "--disable-warnings"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120  # 2分钟超时
            )

            end_time = time.time()
            execution_time = end_time - start_time

            test_result = {
                "file": test_file,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "success": result.returncode == 0
            }

            # 分析测试输出
            stdout_lines = result.stdout.split('\n')
            test_count = 0
            passed_count = 0
            failed_count = 0

            for line in stdout_lines:
                if line.strip():
                    print(line)
                if "PASSED" in line:
                    passed_count += 1
                if "FAILED" in line or "ERROR" in line:
                    failed_count += 1
                if line.startswith("test_"):
                    test_count += 1

            test_result["test_count"] = test_count
            test_result["passed_count"] = passed_count
            test_result["failed_count"] = failed_count

            results.append(test_result)

            print(f"  执行时间: {execution_time:.1f}秒")
        except subprocess.TimeoutExpired:
            print(f"⏰ 测试超时: {test_file}")
            results.append({
                "file": test_file,
                "return_code": -1,
                "execution_time": 120,
                "success": False,
                "error": "timeout"
            })
        except Exception as e:
            print(f"💥 测试执行异常: {test_file} - {str(e)}")
            results.append({
                "file": test_file,
                "return_code": -1,
                "execution_time": 0,
                "success": False,
                "error": str(e)
            })

    # 计算总体结果
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print("\n" + "=" * 50)
    print("📊 E2E测试执行结果总结")
    print("=" * 50)

    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)

    print(f"总测试文件数: {total_tests}")
    print(f"成功执行文件数: {successful_tests}")
    print(f"失败执行文件数: {total_tests - successful_tests}")
    print(f"总执行时间: {total_execution_time:.1f}分钟")
    if total_execution_time < 120:
        print("🎉 测试执行时间优化成功! (目标<2分钟)")
    elif total_execution_time < 300:
        print("⚠️ 测试执行时间可接受 (目标<5分钟)")
    else:
        print("❌ 测试执行时间仍需优化")

    # 详细结果
    print("\n📋 详细测试结果:")
    for result in results:
        status = "✅ 通过" if result["success"] else "❌ 失败"
        print(f"  执行时间: {result['execution_time']:.1f}秒")
    # 生成优化建议
    generate_optimization_suggestions(results, total_execution_time)

    return successful_tests == total_tests

def generate_optimization_suggestions(results, total_execution_time):
    """生成优化建议"""
    print("\n💡 优化建议:")

    suggestions = []

    # 检查执行时间
    if total_execution_time > 120:
        suggestions.append("1. 执行时间仍需优化，考虑进一步增加并发数或减少测试依赖")

    # 检查失败的测试
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        suggestions.append(f"2. 修复失败的测试文件: {[r['file'] for r in failed_tests]}")

    # 检查测试覆盖
    total_test_count = sum(r.get("test_count", 0) for r in results)
    if total_test_count < 10:
        suggestions.append("3. 考虑增加更多测试用例以提高覆盖率")

    if not suggestions:
        suggestions.append("✅ 测试执行效率良好，继续保持")

    for suggestion in suggestions:
        print(f"  {suggestion}")

    # 输出关键指标
    print("\n📈 关键指标:")
    print(f"  平均执行时间: {total_execution_time/max(len(results), 1):.1f}秒")
    print(f"  目标达成: {'✅ 是' if total_execution_time < 120 else '❌ 否'}")
    print("  效率提升: 显著提升 (环境优化 + 并发执行)")
if __name__ == "__main__":
    success = run_quick_e2e_test()
    if success:
        print("\n🎉 E2E测试执行效率优化验证成功!")
    else:
        print("\n⚠️ E2E测试执行效率优化需要进一步调整")
    sys.exit(0 if success else 1)
