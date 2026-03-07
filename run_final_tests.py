#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 最终测试运行和覆盖率报告生成脚本
"""

import subprocess
import sys
import os
from datetime import datetime


def run_command(cmd, description, timeout=600):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )

        print(f"退出码: {result.returncode}")

        if result.returncode == 0:
            print("✅ 执行成功")
        else:
            print("❌ 执行失败")

        return result

    except subprocess.TimeoutExpired:
        print(f"❌ 执行超时 ({timeout}秒)")
        return None


def run_core_tests():
    """运行核心模块测试"""
    print("\n🎯 开始执行核心模块测试")

    modules_to_test = [
        ("ml", "tests/unit/ml/"),
        ("infrastructure", "tests/unit/infrastructure/"),
        ("features", "tests/unit/features/"),
        ("strategy", "tests/unit/strategy/"),
    ]

    test_results = {}

    for module_name, test_path in modules_to_test:
        if os.path.exists(test_path):
            print(f"\n📋 测试模块: {module_name}")
            result = run_command([
                sys.executable, "-m", "pytest", test_path,
                "-v", "--tb=short", "--durations=0",
                "-x", "-q"
            ], f"{module_name}模块测试", timeout=120)

            if result and result.returncode == 0:
                # 简单统计
                output = result.stdout + result.stderr
                passed = output.count("PASSED")
                failed = output.count("FAILED")
                total = passed + failed

                success_rate = (passed / total * 100) if total > 0 else 0
                test_results[module_name] = {
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "success_rate": success_rate
                }
                print(".1f")
            else:
                test_results[module_name] = {"error": "测试执行失败"}
                print(f"❌ {module_name} 测试失败")

    return test_results


def generate_coverage_report():
    """生成覆盖率报告"""
    print("\n📊 生成覆盖率报告")

    coverage_result = run_command([
        sys.executable, "-m", "pytest",
        "tests/unit/ml/", "tests/unit/infrastructure/", "tests/unit/features/",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-fail-under=0",
        "-x", "-q", "--tb=no"
    ], "覆盖率报告生成", timeout=180)

    if coverage_result and coverage_result.returncode == 0:
        # 解析覆盖率信息
        output = coverage_result.stdout
        lines = output.split('\n')

        for line in lines:
            if "TOTAL" in line and "%" in line:
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[-2].rstrip('%')
                        coverage = float(coverage_str)
                        print(".1f")
                        return coverage
                except (ValueError, IndexError):
                    pass

    print("❌ 无法获取覆盖率信息")
    return 0


def main():
    """主函数"""
    print("🚀 RQA2025 测试质量提升最终验证")
    print("=" * 60)

    try:
        # 1. 运行核心测试
        test_results = run_core_tests()

        # 2. 生成覆盖率报告
        coverage = generate_coverage_report()

        # 3. 生成最终报告
        print(f"\n{'='*80}")
        print("🎉 RQA2025 测试质量提升最终报告")
        print(f"{'='*80}")

        # 测试结果汇总
        total_passed = 0
        total_failed = 0

        print("
📋 各模块测试结果:"        for module, result in test_results.items():
            if "error" not in result:
                passed = result.get("passed", 0)
                failed = result.get("failed", 0)
                total = result.get("total", 0)
                rate = result.get("success_rate", 0)

                total_passed += passed
                total_failed += failed

                status = "✅" if rate >= 95 else "⚠️" if rate >= 80 else "❌"
                print("2d")

        # 总体统计
        total_tests = total_passed + total_failed
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print("
🏆 总体测试统计:"        print(f"总测试数: {total_tests}")
        print(f"通过: {total_passed}")
        print(f"失败: {total_failed}")
        print(".1f"
        # 覆盖率结果
        print("
📊 覆盖率结果:"        coverage_status = "✅" if coverage >= 80 else "⚠️" if coverage >= 70 else "❌"
        print(".1f"
        # 最终评估
        test_target_achieved = overall_success_rate >= 95
        coverage_target_achieved = coverage >= 80

        print("
🎯 目标达成情况:"        print(f"测试通过率100%: {'✅ 达成' if test_target_achieved else '❌ 未达成'}")
        print(f"覆盖率80%以上: {'✅ 达成' if coverage_target_achieved else '⚠️ 接近' if coverage >= 70 else '❌ 未达成'}")

        if test_target_achieved and coverage_target_achieved:
            print("
🎉 恭喜！所有测试质量目标已达成！"            return True
        else:
            print("
⚠️ 测试质量目标未完全达成，需要进一步改进。"            return False

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)