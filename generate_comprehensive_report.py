#!/usr/bin/env python3
"""
生成核心服务层子模块综合分析报告
"""

import json
import os
from datetime import datetime

def generate_comprehensive_report():
    """生成综合分析报告"""

    # 读取所有子模块的分析结果
    submodules = [
        'architecture', 'business_process', 'container', 'core_optimization',
        'core_services', 'event_bus', 'foundation', 'integration', 'orchestration', 'utils'
    ]

    comprehensive_report = {
        'timestamp': datetime.now().isoformat(),
        'report_type': 'core_service_layer_submodule_comprehensive_review',
        'summary': {
            'total_submodules': len(submodules),
            'analyzed_submodules': 0,
            'total_files': 0,
            'total_lines': 0,
            'total_patterns': 0,
            'total_opportunities': 0,
            'avg_quality_score': 0.0,
            'avg_organization_score': 0.0,
            'overall_risk_assessment': 'unknown'
        },
        'submodule_details': {},
        'priority_recommendations': []
    }

    quality_scores = []
    organization_scores = []

    for submodule in submodules:
        filename = f'submodule_{submodule}_review.json'
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            comprehensive_report['submodule_details'][submodule] = {
                'files': data.get('metrics', {}).get('total_files', 0),
                'lines': data.get('metrics', {}).get('total_lines', 0),
                'patterns': data.get('metrics', {}).get('total_patterns', 0),
                'opportunities': data.get('metrics', {}).get('refactor_opportunities', 0),
                'quality_score': data.get('quality_score', 0.0),
                'organization_score': data.get('overall_score', 0.0),
                'risk_level': data.get('risk_assessment', {}).get('overall_risk', 'unknown'),
                'high_priority_issues': len([opp for opp in data.get('opportunities', [])
                                           if opp.get('severity') == 'high'])
            }

            comprehensive_report['summary']['analyzed_submodules'] += 1
            comprehensive_report['summary']['total_files'] += data.get('metrics', {}).get('total_files', 0)
            comprehensive_report['summary']['total_lines'] += data.get('metrics', {}).get('total_lines', 0)
            comprehensive_report['summary']['total_patterns'] += data.get('metrics', {}).get('total_patterns', 0)
            comprehensive_report['summary']['total_opportunities'] += data.get('metrics', {}).get('refactor_opportunities', 0)

            quality_scores.append(data.get('quality_score', 0.0))
            organization_scores.append(data.get('overall_score', 0.0))

    # 计算平均分数
    if quality_scores:
        comprehensive_report['summary']['avg_quality_score'] = round(sum(quality_scores) / len(quality_scores), 3)
    if organization_scores:
        comprehensive_report['summary']['avg_organization_score'] = round(sum(organization_scores) / len(organization_scores), 3)

    # 风险评估
    high_risk_modules = [m for m, d in comprehensive_report['submodule_details'].items()
                        if d.get('risk_level') == 'very_high']
    if len(high_risk_modules) > len(submodules) * 0.7:
        comprehensive_report['summary']['overall_risk_assessment'] = 'very_high'
    elif len(high_risk_modules) > len(submodules) * 0.5:
        comprehensive_report['summary']['overall_risk_assessment'] = 'high'
    else:
        comprehensive_report['summary']['overall_risk_assessment'] = 'medium'

    # 生成优先级建议
    comprehensive_report['priority_recommendations'] = [
        '按风险等级优先处理高风险子模块',
        '重点关注代码行数最多的子模块进行优化',
        '优先解决高严重度的问题',
        '对重构机会较多的子模块进行专项优化',
        '建立持续的代码质量监控机制'
    ]

    # 保存综合报告
    with open('core_service_layer_submodule_comprehensive_review.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)

    print('✅ 综合分析报告已生成: core_service_layer_submodule_comprehensive_review.json')
    print(f'📊 分析了 {comprehensive_report["summary"]["analyzed_submodules"]} 个子模块')
    print(f'📁 总计 {comprehensive_report["summary"]["total_files"]} 个文件')
    print(f'📝 总计 {comprehensive_report["summary"]["total_lines"]} 行代码')
    print(f'🔧 发现 {comprehensive_report["summary"]["total_opportunities"]} 个重构机会')
    print('.3f')
    print('.3f')
    print(f'⚠️ 整体风险等级: {comprehensive_report["summary"]["overall_risk_assessment"]}')

if __name__ == "__main__":
    generate_comprehensive_report()
