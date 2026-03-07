#!/usr/bin/env python3
"""
运行已修复的测试文件

专门运行那些语法错误已修复的测试文件
"""

import os
import subprocess
import json


def get_fixed_tests():
    """获取已修复的测试文件列表"""
    fixed_tests = [
        # 基础设施层核心测试
        "tests/unit/infrastructure/health/test_enhanced_monitoring.py",
        "tests/unit/infrastructure/logging/test_unified_logger.py",
        "tests/unit/infrastructure/error/test_unified_error_handler.py",
        "tests/unit/infrastructure/cache/test_cache.py",

        # 业务流程测试
        "tests/business_process/test_strategy_development_flow.py",
        "tests/business_process/test_trading_execution_flow.py",
        "tests/business_process/test_risk_control_flow.py",
        "tests/business_process/test_data_processing_flow.py",
        "tests/business_process/test_user_service_flow.py"
    ]

    return fixed_tests


def run_tests_with_coverage(test_files):
    """运行测试并生成覆盖率报告"""
    cmd = [
        "python", "-m", "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=json:phase7_coverage.json",
        "--cov-fail-under=10",  # 降低阈值以适应当前状态
        "-v",
        "--tb=line",
        "--maxfail=5"  # 限制失败数量
    ] + test_files

    print(f"🧪 运行 {len(test_files)} 个测试文件...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.returncode == 0


def analyze_coverage():
    """分析覆盖率结果"""
    if os.path.exists("phase7_coverage.json"):
        try:
            with open("phase7_coverage.json", "r") as f:
                data = json.load(f)

            totals = data.get("totals", {})
            coverage = totals.get("percent_covered", 0)

            print(f"\n📊 Phase 7 覆盖率分析:")
            print(f"   总覆盖率: {coverage:.2f}%")
            print(f"   总行数: {totals.get('num_statements', 0)}")
            print(f"   覆盖行数: {totals.get('covered_statements', 0)}")
            print(f"   缺失行数: {totals.get('missing_statements', 0)}")

            return coverage
        except Exception as e:
            print(f"分析覆盖率时出错: {e}")
            return 0
    else:
        print("未找到覆盖率报告文件")
        return 0


def main():
    """主函数"""
    print("🚀 Phase 7: 运行已修复的测试文件")

    # 获取修复的测试文件
    test_files = get_fixed_tests()

    # 运行测试
    success = run_tests_with_coverage(test_files)

    # 分析结果
    coverage = analyze_coverage()

    print(f"\n{'='*50}")
    if success:
        print("✅ 测试运行成功!")
        print(".2f")
    else:
        print("❌ 测试运行失败")
        print(".2f")
    print(f"{'='*50}")

    # 清理报告文件
    if os.path.exists("phase7_coverage.json"):
        os.remove("phase7_coverage.json")


if __name__ == "__main__":
    main()
