#!/usr/bin/env python3
"""
AI代码审查结果分析脚本

分析AI智能化代码分析器的输出结果并生成详细报告。
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_review_results(json_file_path):
    """分析审查结果并生成报告"""

    # 读取分析结果
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("📊 AI代码分析详细报告")
    print("=" * 60)

    # 基本信息
    print(f"🎯 分析目标: {data['target_path']}")
    print(f"📅 分析时间: {data['timestamp']}")
    print()

    # 质量评分
    print("⭐ 质量评估:")
    print(".3f")
    print(".3f")
    if 'organization_quality_score' in data:
        print(".3f")
    print(f"  风险等级: {data['risk_assessment']['overall_risk']}")
    print()

    # 风险评估详情
    risk = data['risk_assessment']
    print("⚠️  风险评估:")
    total_opportunities = risk['risk_breakdown']['low'] + risk['risk_breakdown']['high']
    print(f"  总重构机会: {total_opportunities}")
    print(f"  低风险机会: {risk['risk_breakdown']['low']}")
    print(f"  高风险机会: {risk['risk_breakdown']['high']}")
    print(f"  可自动化修复: {risk['automated_opportunities']}")
    print(f"  需要手动修复: {risk['manual_opportunities']}")
    print()

    # 重构机会统计
    opportunities = data.get('opportunities', [])
    if opportunities:
        titles = [opp['title'] for opp in opportunities]
        severities = [opp['severity'] for opp in opportunities]
        efforts = [opp.get('effort', 'unknown') for opp in opportunities]

        title_counts = Counter(titles)
        severity_counts = Counter(severities)
        effort_counts = Counter(efforts)

        print("🔧 重构机会统计:")

        print("  按类型:")
        for title, count in title_counts.most_common(10):
            print(f"    {title}: {count}个")

        print("  按严重程度:")
        for severity, count in severity_counts.most_common():
            print(f"    {severity}: {count}个")

        print("  按工作量:")
        for effort, count in effort_counts.most_common():
            print(f"    {effort}: {count}个")
    print()

    # 文件分析
    print("📁 文件分析:")
    print(f"  总文件数: {data['metrics']['total_files']}")
    print(f"  总代码行: {data['metrics']['total_lines']}")
    print(f"  识别模式: {data['metrics']['total_patterns']}")
    print()

    # 生成详细的重构建议报告
    generate_detailed_report(data, opportunities)


def generate_detailed_report(data, opportunities):
    """生成详细的重构建议报告"""

    print("📋 详细重构建议")
    print("=" * 60)

    # 按优先级分组
    priority_groups = defaultdict(list)

    for opp in opportunities:
        severity = opp['severity']
        if severity == 'high':
            priority = 'P0 - 紧急处理'
        elif severity == 'medium':
            priority = 'P1 - 重要处理'
        else:
            priority = 'P2 - 一般处理'

        priority_groups[priority].append(opp)

    # 输出按优先级排序的建议
    for priority in ['P0 - 紧急处理', 'P1 - 重要处理', 'P2 - 一般处理']:
        if priority in priority_groups:
            print(f"\n🎯 {priority} ({len(priority_groups[priority])}个问题):")

            # 按类型分组显示
            type_groups = defaultdict(list)
            for opp in priority_groups[priority]:
                opp_type = opp['title'].split(':')[0] if ':' in opp['title'] else opp['title']
                type_groups[opp_type].append(opp)

            for opp_type, opps in type_groups.items():
                print(f"  📌 {opp_type} ({len(opps)}个):")
                for opp in opps[:3]:  # 只显示前3个示例
                    file_name = Path(opp['file_path']).name
                    print(f"    • {file_name}:{opp['line_number']} - {opp['description']}")

                if len(opps) > 3:
                    print(f"    ... 还有{len(opps) - 3}个类似问题")


def generate_action_plan(data, opportunities):
    """生成行动计划"""

    print("\n🎯 优化行动计划")
    print("=" * 60)

    # 统计可自动化修复的问题
    automated_opportunities = [opp for opp in opportunities if opp.get('automated', False)]

    print("🚀 Phase 1: 自动化修复 (预计1-2天)")
    print(f"  可自动化修复问题: {len(automated_opportunities)}个")

    if automated_opportunities:
        auto_types = Counter([opp['title'] for opp in automated_opportunities])
        for opp_type, count in auto_types.most_common(3):
            print(f"  • {opp_type}: {count}个")

    print("\n🔧 Phase 2: 核心重构 (预计3-5天)")
    print("  重点处理长函数和长参数列表问题")

    # 统计长函数问题
    long_function_opps = [opp for opp in opportunities if '长函数' in opp['title']]
    print(f"  长函数重构: {len(long_function_opps)}个")

    # 统计长参数问题
    long_param_opps = [opp for opp in opportunities if '长参数列表' in opp['title']]
    print(f"  长参数优化: {len(long_param_opps)}个")

    print("\n✅ Phase 3: 验证和完善 (预计2-3天)")
    print("  回归测试和性能验证")

    print("\n📊 预期收益:")
    print("  • 代码可维护性提升 40-60%")
    print("  • 代码可读性显著改善")
    print("  • 潜在缺陷风险降低")
    print("  • 开发效率提升 20-30%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python analyze_review_results.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    analyze_review_results(json_file_path)
