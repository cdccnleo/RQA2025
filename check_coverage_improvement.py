#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
立即检查测试覆盖率提升效果
"""

import subprocess
import sys
import json
import os
from pathlib import Path


def run_tests_and_check_coverage():
    """运行测试并检查覆盖率"""
    print("🚀 开始运行新增测试并检查覆盖率...")

    # 确保目录存在
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # 运行测试命令
    test_commands = [
        # 运行新增的核心测试
        [
            sys.executable, "-m", "pytest",
            "tests/test_core_simplified.py",
            "tests/test_data_management.py",
            "tests/test_strategy_trading.py",
            "-v",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage_new_tests",
            "--cov-report=json:reports/coverage_new_tests.json",
            "--maxfail=10",
            "--tb=short"
        ],

        # 运行所有测试获取完整覆盖率
        [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:reports/coverage_complete",
            "--cov-report=json:reports/coverage_complete.json",
            "--maxfail=20",
            "--tb=short"
        ]
    ]

    results = []

    for i, cmd in enumerate(test_commands):
        print(f"\n📊 执行测试组 {i+1}/{len(test_commands)}...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=os.getcwd()
            )

            results.append({
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })

            print(f"✅ 测试组 {i+1} 完成，返回码: {result.returncode}")
            if result.stdout:
                # 只显示关键信息
                lines = result.stdout.split('\n')
                coverage_lines = [
                    line for line in lines if 'TOTAL' in line or 'coverage' in line.lower()]
                for line in coverage_lines[-3:]:  # 显示最后几行覆盖率信息
                    print(f"   {line}")

        except subprocess.TimeoutExpired:
            print(f"⏰ 测试组 {i+1} 超时")
            results.append({
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": "Timeout"
            })
        except Exception as e:
            print(f"❌ 测试组 {i+1} 执行失败: {e}")
            results.append({
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            })

    return results


def analyze_coverage_improvement():
    """分析覆盖率提升效果"""
    print("\n📈 分析覆盖率提升效果...")

    # 检查覆盖率报告文件
    coverage_files = [
        "reports/coverage_new_tests.json",
        "reports/coverage_complete.json",
        "reports/coverage.json"  # 原有的覆盖率文件
    ]

    coverage_data = {}

    for file_path in coverage_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if 'totals' in data:
                        coverage_data[file_path] = {
                            "num_statements": data['totals'].get('num_statements', 0),
                            "missing_lines": data['totals'].get('missing_lines', 0),
                            "percent_covered": data['totals'].get('percent_covered', 0)
                        }
                        print(f"📄 {file_path}: {data['totals'].get('percent_covered', 0):.2f}% 覆盖率")
            except Exception as e:
                print(f"⚠️  读取 {file_path} 失败: {e}")

    # 生成覆盖率提升报告
    if coverage_data:
        print("\n📊 覆盖率对比:")
        for file_path, data in coverage_data.items():
            print(f"  {file_path}:")
            print(f"    - 总语句数: {data['num_statements']}")
            print(f"    - 覆盖率: {data['percent_covered']:.2f}%")
            print(f"    - 未覆盖行数: {data['missing_lines']}")

    return coverage_data


def generate_summary_report(test_results, coverage_data):
    """生成汇总报告"""
    print("\n📝 生成汇总报告...")

    report = {
        "测试执行结果": {
            "总测试组数": len(test_results),
            "成功测试组": len([r for r in test_results if r["returncode"] == 0]),
            "失败测试组": len([r for r in test_results if r["returncode"] != 0])
        },
        "覆盖率数据": coverage_data,
        "关键发现": []
    }

    # 分析关键发现
    if coverage_data:
        max_coverage = max(data["percent_covered"] for data in coverage_data.values())
        report["关键发现"].append(f"最高覆盖率: {max_coverage:.2f}%")

        if max_coverage >= 30:
            report["关键发现"].append("✅ 覆盖率已达到30%阶段目标")
        if max_coverage >= 60:
            report["关键发现"].append("✅ 覆盖率已达到60%阶段目标")
        if max_coverage >= 90:
            report["关键发现"].append("🎉 覆盖率已达到90%最终目标")

    # 保存报告
    with open("reports/coverage_improvement_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("📄 汇总报告已保存到: reports/coverage_improvement_report.json")
    return report


def main():
    """主函数"""
    print("🎯 立即检查测试覆盖率提升效果")
    print("=" * 50)

    # 1. 运行测试
    test_results = run_tests_and_check_coverage()

    # 2. 分析覆盖率
    coverage_data = analyze_coverage_improvement()

    # 3. 生成报告
    summary_report = generate_summary_report(test_results, coverage_data)

    # 4. 显示最终结果
    print("\n🎯 最终结果:")
    print("=" * 30)

    if coverage_data:
        max_coverage = max(data["percent_covered"] for data in coverage_data.values())
        print(f"🎊 当前最高覆盖率: {max_coverage:.2f}%")

        if max_coverage >= 9.45:  # 原始覆盖率
            improvement = max_coverage - 9.45
            print(f"📈 覆盖率提升: +{improvement:.2f}%")

        # 判断目标达成情况
        if max_coverage >= 90:
            print("🎉 恭喜！已达成90%覆盖率目标！")
        elif max_coverage >= 60:
            print("✅ 已达成60%覆盖率阶段目标")
        elif max_coverage >= 30:
            print("✅ 已达成30%覆盖率阶段目标")
        else:
            print(f"📊 当前覆盖率: {max_coverage:.2f}%，继续努力！")
    else:
        print("⚠️  未能获取到覆盖率数据")

    print("\n📂 详细报告请查看 reports/ 目录下的文件")


if __name__ == "__main__":
    main()
