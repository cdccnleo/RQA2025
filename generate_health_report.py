#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成健康管理系统代码审查报告
"""

import json


def main():
    # 读取分析结果
    with open('health_analysis_result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print('🏥 基础设施层健康管理系统 - AI智能化代码审查报告')
    print('=' * 80)
    print()

    # 基本指标
    print('📊 基本指标:')
    print(f'  • 总文件数: {data["metrics"]["total_files"]}')
    print(f'  • 总代码行: {data["metrics"]["total_lines"]:,}')
    print(f'  • 识别模式: {data["metrics"]["total_patterns"]}')
    print(f'  • 重构机会: {data["metrics"]["refactor_opportunities"]}')
    print()

    # 质量评分
    print('🎯 质量评估:')
    print(f'  • 代码质量评分: {data["quality_score"]:.3f}')
    print(f'  • 综合评分: {data["overall_score"]:.3f}')
    print(f'  • 风险等级: {data["risk_assessment"]["overall_risk"]}')
    print()

    # 风险分析
    risk = data['risk_assessment']
    print('⚠️  风险分析:')
    print(f'  • 总体风险: {risk["overall_risk"]}')
    print(f'  • 高风险问题: {risk["risk_breakdown"]["high"]}')
    print(f'  • 自动化修复机会: {risk["automated_opportunities"]}')
    print(f'  • 手动修复机会: {risk["manual_opportunities"]}')
    print()

    # 组织分析
    org = data['organization_analysis']
    print('🏗️  组织结构分析:')
    print(f'  • 组织质量评分: {org["metrics"]["quality_score"]:.3f}')
    print(f'  • 问题数量: {org["issues_count"]}')
    print(f'  • 改进建议: {org["recommendations_count"]}')
    print(f'  • 平均文件大小: {org["metrics"]["avg_file_size"]:.1f} 行')
    print(f'  • 最大文件: {org["metrics"]["largest_file"]} ({org["metrics"]["max_file_size"]} 行)')
    print()

    # 文件分类统计
    categories = org['categories']
    print('📁 文件分类统计:')
    for category, files in categories.items():
        print(f'  • {category}: {len(files)} 个文件')

    print()
    print('🎯 主要发现:')
    print('1. 🔴 高风险: 多处语法错误和缩进问题')
    print('2. 🟡 中等风险: 函数过长，复杂度较高')
    print('3. 🟢 机遇: 组织结构合理，模块分类清晰')
    print('4. 📈 改进空间: 493个重构机会，主要集中在代码简化')
    print()

    print('💡 建议行动:')
    print('1. 🚨 紧急修复: 解决所有语法错误和AST解析失败')
    print('2. 🔧 重构优化: 拆分长函数，提高可维护性')
    print('3. 📝 代码规范: 统一编码风格，移除BOM字符')
    print('4. 🧪 测试完善: 为核心健康检查功能添加单元测试')
    print('5. 📚 文档补充: 为各组件添加详细的文档说明')
    print()

    # 显示前几个重构机会
    opportunities = data.get('opportunities', [])
    if opportunities:
        print('🔧 主要重构机会 (前10个):')
        for i, opp in enumerate(opportunities[:10]):
            print(f'{i+1}. {opp["title"]}')
            print(f'   📁 {opp["file_path"]}:{opp["line_number"]}')
            print(f'   💡 {opp["suggested_fix"]}')
            print()


if __name__ == "__main__":
    main()
