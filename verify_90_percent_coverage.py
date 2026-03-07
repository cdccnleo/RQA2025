#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 测试覆盖率90%目标验证脚本
验证当前测试覆盖率是否达到90%目标
"""

import json
import os
import sys
import subprocess
from datetime import datetime


def verify_90_percent_coverage():
    """验证是否达到90%覆盖率目标"""

    print("=" * 80)
    print("🎯 RQA2025 测试覆盖率90%目标验证")
    print("=" * 80)
    print(f"📅 验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 检查现有覆盖率报告
    coverage_files = [
        "reports/coverage.json",
        "reports/coverage_dashboard.html",
        "reports/coverage_validation_report.json",
        "coverage.json"
    ]

    current_coverage = None
    report_source = None

    # 尝试从各个文件获取覆盖率数据
    for file_path in coverage_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 从coverage.json获取覆盖率
                    if 'totals' in data:
                        current_coverage = data['totals'].get('percent_covered', 0)
                        report_source = file_path
                        break

                    # 从其他JSON文件获取覆盖率
                    elif 'overall_coverage' in data:
                        current_coverage = data['overall_coverage']
                        report_source = file_path
                        break

                elif file_path.endswith('.html'):
                    # 从HTML文件提取覆盖率数据
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找覆盖率数据
                    import re
                    pattern = r'当前覆盖率:\s*(\d+\.?\d*)%'
                    match = re.search(pattern, content)
                    if match:
                        current_coverage = float(match.group(1))
                        report_source = file_path
                        break

            except Exception as e:
                print(f"❌ 读取文件 {file_path} 失败: {e}")
                continue

    if current_coverage is None:
        print("⚠️  未找到现有覆盖率报告，运行新的覆盖率测试...")
        current_coverage = run_fresh_coverage_test()
        report_source = "新生成的覆盖率测试"

    # 显示当前覆盖率状况
    print(f"📊 数据来源: {report_source}")
    print(f"📈 当前覆盖率: {current_coverage:.2f}%")
    print(f"🎯 目标覆盖率: 90.00%")

    # 计算差距
    coverage_gap = 90.0 - current_coverage

    if current_coverage >= 90.0:
        # 达到目标
        print()
        print("🎉" * 20)
        print("✅ 恭喜！已达到90%覆盖率目标！")
        print(f"🏆 超出目标: {current_coverage - 90.0:.2f}%")
        print("🚀 代码质量达到生产部署标准！")
        print("🎉" * 20)

        # 生成达标报告
        generate_achievement_report(current_coverage, report_source)
        return True

    else:
        # 未达到目标
        print()
        print("❌" * 20)
        print("❌ 未达到90%覆盖率目标")
        print(f"📉 覆盖率不足: {coverage_gap:.2f}%")
        print(f"📊 完成度: {(current_coverage/90.0)*100:.1f}%")
        print("🔧 需要继续改进测试覆盖率")
        print("❌" * 20)

        # 生成改进建议
        generate_improvement_suggestions(current_coverage, coverage_gap)
        return False


def run_fresh_coverage_test():
    """运行新的覆盖率测试"""
    try:
        print("🔄 运行pytest覆盖率测试...")

        # 运行pytest测试
        result = subprocess.run([
            "python", "-m", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=json:fresh_coverage.json",
            "--cov-report=term"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ 覆盖率测试执行成功")

            # 读取生成的覆盖率数据
            if os.path.exists("fresh_coverage.json"):
                with open("fresh_coverage.json", 'r') as f:
                    data = json.load(f)
                return data.get('totals', {}).get('percent_covered', 0)
            else:
                print("❌ 未找到生成的覆盖率文件")
                return 0
        else:
            print(f"❌ 覆盖率测试执行失败: {result.stderr}")
            return 0

    except subprocess.TimeoutExpired:
        print("❌ 覆盖率测试执行超时")
        return 0
    except Exception as e:
        print(f"❌ 覆盖率测试执行异常: {e}")
        return 0


def generate_achievement_report(coverage, source):
    """生成达标报告"""
    report = {
        "achievement_status": "ACHIEVED",
        "verification_time": datetime.now().isoformat(),
        "target_coverage": 90.0,
        "actual_coverage": coverage,
        "excess_coverage": coverage - 90.0,
        "data_source": source,
        "completion_rate": (coverage / 90.0) * 100,
        "quality_level": "PRODUCTION_READY",
        "recommendations": [
            "✅ 代码质量已达到生产部署标准",
            "🚀 可以继续后续的部署流程",
            "📊 建议保持当前覆盖率水平",
            "🔄 定期监控覆盖率变化趋势"
        ]
    }

    # 保存报告
    with open("coverage_90_percent_achievement_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 达标报告已保存: coverage_90_percent_achievement_report.json")


def generate_improvement_suggestions(coverage, gap):
    """生成改进建议"""

    # 根据覆盖率水平提供不同建议
    if coverage >= 80:
        priority = "HIGH"
        suggestions = [
            "🎯 已接近目标，重点关注剩余的高价值测试用例",
            "🔍 识别未覆盖的关键业务逻辑",
            "🧪 增加边界条件和异常处理测试",
            "📈 预计还需1-2周达到90%目标"
        ]
    elif coverage >= 70:
        priority = "MEDIUM"
        suggestions = [
            "📊 覆盖率处于良好水平，继续稳步提升",
            "🔧 重点补充核心模块的测试用例",
            "🧩 完善集成测试和端到端测试",
            "📈 预计还需2-3周达到90%目标"
        ]
    else:
        priority = "URGENT"
        suggestions = [
            "🚨 覆盖率偏低，需要大幅提升测试投入",
            "🏗️ 建立系统性的测试开发计划",
            "👥 考虑增加测试开发人力投入",
            "📈 预计还需4-6周达到90%目标"
        ]

    improvement_report = {
        "improvement_status": "IN_PROGRESS",
        "verification_time": datetime.now().isoformat(),
        "target_coverage": 90.0,
        "actual_coverage": coverage,
        "coverage_gap": gap,
        "completion_rate": (coverage / 90.0) * 100,
        "priority_level": priority,
        "improvement_suggestions": suggestions,
        "next_steps": [
            "🔍 运行详细的覆盖率分析: python verify_coverage.py",
            "📊 查看覆盖率仪表板: 打开 reports/coverage_dashboard.html",
            "🧪 重点改进低覆盖率模块",
            "📈 制定周期性覆盖率提升计划"
        ]
    }

    # 保存改进报告
    with open("coverage_90_percent_improvement_plan.json", 'w', encoding='utf-8') as f:
        json.dump(improvement_report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 改进计划已保存: coverage_90_percent_improvement_plan.json")
    print("\n🔧 立即行动建议:")
    for suggestion in suggestions:
        print(f"   {suggestion}")


def check_quality_gates():
    """检查质量门禁状态"""
    print("\n🚪 质量门禁检查")
    print("-" * 60)

    # 检查各个质量门禁配置
    quality_gates_files = [
        "scripts/ci/coverage_quality_gate.py",
        "test_coverage_monitoring_setup.py",
        "scripts/check_quality_gates.py"
    ]

    gates_found = 0
    for gate_file in quality_gates_files:
        if os.path.exists(gate_file):
            gates_found += 1
            print(f"✅ 质量门禁配置: {gate_file}")

    if gates_found > 0:
        print(f"📊 发现 {gates_found} 个质量门禁配置文件")
        print("🔄 建议运行质量门禁检查以验证详细状态")
    else:
        print("⚠️  未找到质量门禁配置文件")


def main():
    """主函数"""
    success = verify_90_percent_coverage()

    # 检查质量门禁
    check_quality_gates()

    print("\n" + "=" * 80)
    print("📊 验证完成")
    print("=" * 80)

    if success:
        print("🎯 结论: 已达到90%覆盖率目标！")
        sys.exit(0)
    else:
        print("🎯 结论: 尚未达到90%覆盖率目标")
        sys.exit(1)


if __name__ == "__main__":
    main()
