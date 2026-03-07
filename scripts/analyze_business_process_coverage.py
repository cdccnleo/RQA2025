#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 业务流程测试覆盖率分析脚本

分析业务流程测试覆盖情况，生成详细的覆盖率报告
"""

from tests.business_process.framework.business_process_test_framework import (
    business_process_registry,
    get_business_process
)
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 导入业务流程测试框架
sys.path.append(str(Path(__file__).parent.parent))


def analyze_business_process_coverage():
    """分析业务流程测试覆盖率"""
    print("📊 RQA2025 业务流程测试覆盖率分析")
    print("=" * 60)

    # 获取所有业务流程
    all_processes = business_process_registry.list_processes()
    print(f"📋 发现 {len(all_processes)} 个业务流程")

    coverage_analysis = {
        "analysis_time": datetime.now().isoformat(),
        "total_processes": len(all_processes),
        "covered_processes": 0,
        "total_steps": 0,
        "covered_steps": 0,
        "total_rules": 0,
        "covered_rules": 0,
        "process_coverage": [],
        "overall_coverage": {}
    }

    # 分析每个业务流程
    for process_name in all_processes:
        process = get_business_process(process_name)
        if not process:
            continue

        print(f"\n🔍 分析业务流程: {process_name}")
        print("-" * 40)

        process_coverage = analyze_single_process(process_name, process)
        coverage_analysis["process_coverage"].append(process_coverage)

        # 累计统计
        coverage_analysis["total_steps"] += process_coverage["total_steps"]
        coverage_analysis["covered_steps"] += process_coverage["covered_steps"]
        coverage_analysis["total_rules"] += process_coverage["total_rules"]
        coverage_analysis["covered_rules"] += process_coverage["covered_rules"]

        if process_coverage["has_test_file"]:
            coverage_analysis["covered_processes"] += 1

    # 计算总体覆盖率
    coverage_analysis["overall_coverage"] = calculate_overall_coverage(coverage_analysis)

    # 生成报告
    generate_coverage_report(coverage_analysis)

    return coverage_analysis


def analyze_single_process(process_name: str, process) -> Dict[str, Any]:
    """分析单个业务流程的覆盖情况"""
    process_coverage = {
        "process_name": process_name,
        "process_description": process.description,
        "total_steps": len(process.steps),
        "covered_steps": 0,
        "total_rules": len(process.business_rules),
        "covered_rules": 0,
        "total_scenarios": len(process.test_scenarios),
        "covered_scenarios": 0,
        "has_test_file": False,
        "test_file_path": "",
        "step_coverage": [],
        "rule_coverage": [],
        "scenario_coverage": []
    }

    # 检查测试文件是否存在
    test_file_path = f"tests/business_process/test_{process_name}_flow.py"
    if os.path.exists(test_file_path):
        process_coverage["has_test_file"] = True
        process_coverage["test_file_path"] = test_file_path

        # 分析测试文件内容
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_content = f.read()

        # 分析步骤覆盖
        for step in process.steps:
            step_name = step["name"]
            step_covered = step_name in test_content
            process_coverage["step_coverage"].append({
                "step_name": step_name,
                "covered": step_covered
            })
            if step_covered:
                process_coverage["covered_steps"] += 1

        # 分析业务规则覆盖
        for rule in process.business_rules:
            rule_covered = rule in test_content
            process_coverage["rule_coverage"].append({
                "rule": rule,
                "covered": rule_covered
            })
            if rule_covered:
                process_coverage["covered_rules"] += 1

        # 分析测试场景覆盖
        for scenario in process.test_scenarios:
            scenario_covered = scenario in test_content
            process_coverage["scenario_coverage"].append({
                "scenario": scenario,
                "covered": scenario_covered
            })
            if scenario_covered:
                process_coverage["covered_scenarios"] += 1

    else:
        print(f"⚠️  缺少测试文件: {test_file_path}")

    # 计算过程覆盖率
    if process_coverage["total_steps"] > 0:
        process_coverage["step_coverage_rate"] = (
            process_coverage["covered_steps"] / process_coverage["total_steps"]
        ) * 100
    else:
        process_coverage["step_coverage_rate"] = 0

    if process_coverage["total_rules"] > 0:
        process_coverage["rule_coverage_rate"] = (
            process_coverage["covered_rules"] / process_coverage["total_rules"]
        ) * 100
    else:
        process_coverage["rule_coverage_rate"] = 0

    if process_coverage["total_scenarios"] > 0:
        process_coverage["scenario_coverage_rate"] = (
            process_coverage["covered_scenarios"] / process_coverage["total_scenarios"]
        ) * 100
    else:
        process_coverage["scenario_coverage_rate"] = 0

    # 输出分析结果
    print(
        f"📊 流程步骤: {process_coverage['covered_steps']}/{process_coverage['total_steps']} ({process_coverage['step_coverage_rate']:.1f}%)")
    print(
        f"📊 业务规则: {process_coverage['covered_rules']}/{process_coverage['total_rules']} ({process_coverage['rule_coverage_rate']:.1f}%)")
    print(
        f"📊 测试场景: {process_coverage['covered_scenarios']}/{process_coverage['total_scenarios']} ({process_coverage['scenario_coverage_rate']:.1f}%)")

    if process_coverage["has_test_file"]:
        print("✅ 测试文件存在")
    else:
        print("❌ 测试文件缺失")

    return process_coverage


def calculate_overall_coverage(coverage_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """计算总体覆盖率"""
    overall_coverage = {
        "process_coverage_rate": 0,
        "step_coverage_rate": 0,
        "rule_coverage_rate": 0,
        "scenario_coverage_rate": 0,
        "weighted_score": 0
    }

    # 计算流程覆盖率
    if coverage_analysis["total_processes"] > 0:
        overall_coverage["process_coverage_rate"] = (
            coverage_analysis["covered_processes"] / coverage_analysis["total_processes"]
        ) * 100

    # 计算步骤覆盖率
    if coverage_analysis["total_steps"] > 0:
        overall_coverage["step_coverage_rate"] = (
            coverage_analysis["covered_steps"] / coverage_analysis["total_steps"]
        ) * 100

    # 计算规则覆盖率
    if coverage_analysis["total_rules"] > 0:
        overall_coverage["rule_coverage_rate"] = (
            coverage_analysis["covered_rules"] / coverage_analysis["total_rules"]
        ) * 100

    # 计算加权得分 (步骤40% + 规则30% + 场景20% + 流程10%)
    overall_coverage["weighted_score"] = (
        overall_coverage["step_coverage_rate"] * 0.4 +
        overall_coverage["rule_coverage_rate"] * 0.3 +
        (sum(p.get("scenario_coverage_rate", 0) for p in coverage_analysis["process_coverage"]) / len(coverage_analysis["process_coverage"]) if coverage_analysis["process_coverage"] else 0) * 0.2 +
        overall_coverage["process_coverage_rate"] * 0.1
    )

    return overall_coverage


def generate_coverage_report(coverage_analysis: Dict[str, Any]):
    """生成覆盖率报告"""
    print("\n" + "=" * 60)
    print("📊 业务流程测试覆盖率分析报告")
    print("=" * 60)

    # 总体覆盖率
    overall = coverage_analysis["overall_coverage"]
    print("\n🎯 总体覆盖率:")
    print(
        f"  📈 流程覆盖率: {overall['process_coverage_rate']:.1f}% ({coverage_analysis['covered_processes']}/{coverage_analysis['total_processes']})")
    print(
        f"  📊 步骤覆盖率: {overall['step_coverage_rate']:.1f}% ({coverage_analysis['covered_steps']}/{coverage_analysis['total_steps']})")
    print(
        f"  📋 规则覆盖率: {overall['rule_coverage_rate']:.1f}% ({coverage_analysis['covered_rules']}/{coverage_analysis['total_rules']})")
    print(f"  🏆 加权得分: {overall['weighted_score']:.1f}%")

    # 覆盖率等级评定
    weighted_score = overall['weighted_score']
    if weighted_score >= 90:
        print("  🎉 覆盖等级: 优秀 (目标达成)")
    elif weighted_score >= 80:
        print("  ✅ 覆盖等级: 良好 (接近目标)")
    elif weighted_score >= 70:
        print("  ⚠️ 覆盖等级: 一般 (需要改进)")
    else:
        print("  ❌ 覆盖等级: 不及格 (需要重点改进)")

    # 详细分析每个流程
    print("\n📋 各流程详细分析:")
    for process_cov in coverage_analysis["process_coverage"]:
        print(f"\n🔍 {process_cov['process_name']}")
        print(f"  描述: {process_cov['process_description']}")
        print(
            f"  步骤覆盖: {process_cov['covered_steps']}/{process_cov['total_steps']} ({process_cov['step_coverage_rate']:.1f}%)")
        print(
            f"  规则覆盖: {process_cov['covered_rules']}/{process_cov['total_rules']} ({process_cov['rule_coverage_rate']:.1f}%)")
        print(
            f"  场景覆盖: {process_cov['covered_scenarios']}/{process_cov['total_scenarios']} ({process_cov['scenario_coverage_rate']:.1f}%)")

        if process_cov['has_test_file']:
            print(f"  测试文件: ✅ {process_cov['test_file_path']}")
        else:
            print(f"  测试文件: ❌ 缺失 {process_cov['test_file_path']}")

    # 生成改进建议
    generate_improvement_suggestions(coverage_analysis)

    # 保存报告
    save_coverage_report(coverage_analysis)


def generate_improvement_suggestions(coverage_analysis: Dict[str, Any]):
    """生成改进建议"""
    print("\n💡 改进建议:")
    suggestions = []

    # 检查缺失的测试文件
    missing_tests = [p for p in coverage_analysis["process_coverage"] if not p["has_test_file"]]
    if missing_tests:
        suggestions.append(f"🔴 优先级: 创建 {len(missing_tests)} 个缺失的测试文件")

    # 检查低覆盖率的流程
    low_coverage = [p for p in coverage_analysis["process_coverage"]
                    if p["has_test_file"] and p.get("step_coverage_rate", 0) < 80]
    if low_coverage:
        suggestions.append(f"🟡 优先级: 提升 {len(low_coverage)} 个流程的步骤覆盖率")

    # 检查规则覆盖不足
    low_rule_coverage = [p for p in coverage_analysis["process_coverage"]
                         if p["has_test_file"] and p.get("rule_coverage_rate", 0) < 70]
    if low_rule_coverage:
        suggestions.append(f"🟡 优先级: 完善 {len(low_rule_coverage)} 个流程的业务规则测试")

    # 检查场景覆盖不足
    low_scenario_coverage = [p for p in coverage_analysis["process_coverage"]
                             if p["has_test_file"] and p.get("scenario_coverage_rate", 0) < 80]
    if low_scenario_coverage:
        suggestions.append(f"🟢 优先级: 增加 {len(low_scenario_coverage)} 个流程的测试场景覆盖")

    if not suggestions:
        suggestions.append("✅ 所有流程覆盖率良好，继续保持")

    for suggestion in suggestions:
        print(f"  {suggestion}")


def save_coverage_report(coverage_analysis: Dict[str, Any]):
    """保存覆盖率报告"""
    print("\n💾 保存覆盖率报告...")
    project_root = Path(__file__).parent.parent

    # 创建报告目录
    report_dir = project_root / "reports" / "business_process_coverage"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成JSON报告
    json_path = report_dir / \
        f"business_process_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coverage_analysis, f, ensure_ascii=False, indent=2)

    # 生成Markdown报告
    md_path = report_dir / \
        f"business_process_coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(generate_markdown_report(coverage_analysis))

    print(f"✅ JSON报告已保存: {json_path}")
    print(f"✅ Markdown报告已保存: {md_path}")


def generate_markdown_report(coverage_analysis: Dict[str, Any]) -> str:
    """生成Markdown格式的覆盖率报告"""
    overall = coverage_analysis["overall_coverage"]

    report = f"""# RQA2025 业务流程测试覆盖率分析报告

## 📊 总体覆盖率

| 指标 | 覆盖率 | 统计 |
|------|--------|------|
| 流程覆盖率 | {overall['process_coverage_rate']:.1f}% | {coverage_analysis['covered_processes']}/{coverage_analysis['total_processes']} |
| 步骤覆盖率 | {overall['step_coverage_rate']:.1f}% | {coverage_analysis['covered_steps']}/{coverage_analysis['total_steps']} |
| 规则覆盖率 | {overall['rule_coverage_rate']:.1f}% | {coverage_analysis['covered_rules']}/{coverage_analysis['total_rules']} |
| 加权得分 | {overall['weighted_score']:.1f}% | - |

## 📋 各流程详细分析

| 流程名称 | 描述 | 步骤覆盖 | 规则覆盖 | 场景覆盖 | 测试文件状态 |
|----------|------|----------|----------|----------|--------------|
"""

    for process_cov in coverage_analysis["process_coverage"]:
        test_status = "✅ 存在" if process_cov['has_test_file'] else "❌ 缺失"
        report += f"""| {process_cov['process_name']} | {process_cov['process_description']} | {process_cov['covered_steps']}/{process_cov['total_steps']} ({process_cov['step_coverage_rate']:.1f}%) | {process_cov['covered_rules']}/{process_cov['total_rules']} ({process_cov['rule_coverage_rate']:.1f}%) | {process_cov['covered_scenarios']}/{process_cov['total_scenarios']} ({process_cov['scenario_coverage_rate']:.1f}%) | {test_status} |
"""

    report += f"""
## 📈 分析结果

### 覆盖等级评定
- **当前加权得分**: {overall['weighted_score']:.1f}%
- **目标得分**: ≥90%
"""

    weighted_score = overall['weighted_score']
    if weighted_score >= 90:
        report += "- **覆盖等级**: 优秀 🎉 (已达成目标)"
    elif weighted_score >= 80:
        report += "- **覆盖等级**: 良好 ✅ (接近目标)"
    elif weighted_score >= 70:
        report += "- **覆盖等级**: 一般 ⚠️ (需要改进)"
    else:
        report += "- **覆盖等级**: 不及格 ❌ (需要重点改进)"

    report += f"""

### 改进建议
1. **目标达成情况**: {'✅ 已达成' if weighted_score >= 90 else '🔄 进行中'}
2. **距离目标差距**: {90 - weighted_score:.1f}% (需要提升)
3. **建议重点关注**: 步骤覆盖率和规则覆盖率的提升

## 📅 报告信息

- **分析时间**: {coverage_analysis['analysis_time']}
- **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **分析工具**: RQA2025 业务流程测试覆盖率分析脚本

---

*RQA2025 业务流程测试覆盖率分析完成*
"""

    return report


def main():
    """主函数"""
    try:
        coverage_analysis = analyze_business_process_coverage()

        overall_score = coverage_analysis["overall_coverage"]["weighted_score"]

        if overall_score >= 90:
            print("\n🎉 业务流程测试覆盖率目标达成!")
            return 0
        elif overall_score >= 70:
            print("\n⚠️ 业务流程测试覆盖率接近目标，继续努力!")
            return 0
        else:
            print("\n❌ 业务流程测试覆盖率需要重点改进!")
            return 1

    except Exception as e:
        print(f"💥 覆盖率分析失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
