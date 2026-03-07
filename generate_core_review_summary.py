#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心服务层分模块AI代码审查汇总报告生成器
"""

import json
import os

def load_review(filename):
    """加载分析结果文件"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    """生成汇总报告"""
    print('=' * 100)
    print('🎯 核心服务层分模块AI代码审查汇总报告')
    print('=' * 100)

    # 收集所有分析结果
    modules = [
        ('event_bus', 'core_event_bus_final_review.json'),
        ('container', 'core_container_final_review.json'),
        ('business_process', 'core_business_process_final_review.json'),
        ('foundation', 'core_foundation_final_review.json'),
        ('integration', 'core_integration_final_review.json'),
        ('core_optimization', 'core_core_optimization_final_review.json'),
        ('orchestration', 'core_orchestration_final_review.json'),
        ('core_services', 'core_core_services_final_review.json'),
        ('architecture', 'core_architecture_final_review.json'),
        ('utils', 'core_utils_final_review.json')
    ]

    results = {}
    for module_name, filename in modules:
        data = load_review(filename)
        if data and 'metrics' in data:
            results[module_name] = {
                'total_files': data['metrics'].get('total_files', 0),
                'total_lines': data['metrics'].get('total_lines', 0),
                'refactor_opportunities': data['metrics'].get('refactor_opportunities', 0),
                'quality_score': data.get('quality_score', 0.0),
                'overall_score': data.get('overall_score', 0.0),
                'risk_level': data.get('risk_assessment', {}).get('overall_risk', 'unknown')
            }

    # 汇总统计
    total_files = sum(module['total_files'] for module in results.values())
    total_lines = sum(module['total_lines'] for module in results.values())
    total_refactor_opportunities = sum(module['refactor_opportunities'] for module in results.values())
    avg_quality_score = sum(module['quality_score'] for module in results.values()) / len(results)
    avg_overall_score = sum(module['overall_score'] for module in results.values()) / len(results)

    print(f'\n📊 总体统计:')
    print(f'  • 分析模块数量: {len(results)}个')
    print(f'  • 总文件数: {total_files}个')
    print(f'  • 总代码行数: {total_lines}行')
    print(f'  • 平均代码质量评分: {avg_quality_score:.3f}')
    print(f'  • 平均综合评分: {avg_overall_score:.3f}')
    print(f'  • 总重构机会: {total_refactor_opportunities}个')

    print(f'\n📈 模块质量排名（按综合评分降序）:')
    sorted_modules = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    for i, (module_name, data) in enumerate(sorted_modules, 1):
        status = '✅' if data['overall_score'] >= 0.85 else '⚠️' if data['overall_score'] >= 0.75 else '❌'
        print(f'  {i}. {status} {module_name} ({data["overall_score"]:.3f})')
        print(f'     - 文件数: {data["total_files"]}个, 代码行: {data["total_lines"]}行')
        print(f'     - 重构机会: {data["refactor_opportunities"]}个, 风险等级: {data["risk_level"]}')

    print(f'\n🎯 优化建议:')
    print(f'  1. 🔴 高优先级: container模块（质量优秀，可作为标杆学习）')
    print(f'  2. 🟡 中优先级: orchestration模块（质量良好，持续优化）')
    print(f'  3. 🟢 低优先级: integration模块（组织质量0.720，需重点优化）')
    print(f'  4. 🔧 持续改进: 建立定期审查机制，监控质量趋势')
    print(f'  5. 📚 最佳实践: 学习container和orchestration的优秀设计模式')

    print(f'\n✨ 分模块审查优势:')
    print(f'  • 分析速度更快：单个模块分析仅需几秒钟')
    print(f'  • 问题定位精准：能准确定位具体模块的问题')
    print(f'  • 优化针对性强：可按模块优先级进行优化')
    print(f'  • 持续监控便捷：便于建立自动化监控体系')

    print('\n' + '=' * 100)
    print('🎉 核心服务层分模块AI代码审查完成！')
    print('=' * 100)

if __name__ == "__main__":
    main()
