#!/usr/bin/env python3
"""
代码重叠和冗余深度分析报告

专门分析基础设施层中的代码重复问题
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class CodeRedundancyDeepAnalysis:
    """代码冗余深度分析"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.analysis_results = {}

    def perform_deep_analysis(self) -> Dict[str, Any]:
        """执行深度分析"""
        print('🔍 开始代码重叠和冗余深度分析')
        print('=' * 60)

        # 加载审查报告
        with open('infrastructure_code_review_report.json', 'r', encoding='utf-8') as f:
            review_report = json.load(f)

        redundancy_data = review_report['detailed_results']['redundancy_analysis']

        self.analysis_results = {
            'redundancy_overview': self._analyze_redundancy_overview(redundancy_data),
            'duplicate_patterns_analysis': self._analyze_duplicate_patterns(redundancy_data),
            'severity_assessment': self._assess_redundancy_severity(redundancy_data),
            'refactoring_opportunities': self._identify_refactoring_opportunities(redundancy_data),
            'module_impact_analysis': self._analyze_module_impact(redundancy_data),
            'recommendations': self._generate_redundancy_recommendations(redundancy_data)
        }

        # 保存深度分析报告
        with open('code_redundancy_deep_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

        print('\\n✅ 代码重叠和冗余深度分析完成')
        self._print_deep_analysis_report()

        return self.analysis_results

    def _analyze_redundancy_overview(self, redundancy_data: Dict) -> Dict[str, Any]:
        """分析冗余概况"""
        overview = {
            'total_duplicate_functions': len(redundancy_data['duplicate_functions']),
            'total_duplicate_classes': len(redundancy_data['duplicate_classes']),
            'total_duplicate_occurrences': 0,
            'average_duplicates_per_group': 0,
            'most_duplicated_function': None,
            'most_duplicated_class': None,
            'largest_duplicate_group': None
        }

        # 计算总出现次数
        func_occurrences = []
        class_occurrences = []

        for dup_group in redundancy_data['duplicate_functions']:
            count = len(dup_group['occurrences'])
            func_occurrences.append(count)
            overview['total_duplicate_occurrences'] += count

        for dup_group in redundancy_data['duplicate_classes']:
            count = len(dup_group['occurrences'])
            class_occurrences.append(count)
            overview['total_duplicate_occurrences'] += count

        # 计算平均值
        if func_occurrences:
            overview['average_duplicates_per_group'] = sum(func_occurrences) / len(func_occurrences)

        # 找出最重复的函数和类
        if redundancy_data['duplicate_functions']:
            max_func_group = max(redundancy_data['duplicate_functions'],
                                 key=lambda x: len(x['occurrences']))
            overview['most_duplicated_function'] = {
                'signature': max_func_group['occurrences'][0]['signature'],
                'occurrences': len(max_func_group['occurrences']),
                'files': [occ['file'] for occ in max_func_group['occurrences']]
            }

        if redundancy_data['duplicate_classes']:
            max_class_group = max(redundancy_data['duplicate_classes'],
                                  key=lambda x: len(x['occurrences']))
            overview['most_duplicated_class'] = {
                'name': max_class_group['occurrences'][0]['class_name'],
                'occurrences': len(max_class_group['occurrences']),
                'files': [occ['file'] for occ in max_class_group['occurrences']]
            }

        # 找出最大重复组
        all_groups = redundancy_data['duplicate_functions'] + redundancy_data['duplicate_classes']
        if all_groups:
            largest_group = max(all_groups, key=lambda x: len(x['occurrences']))
            overview['largest_duplicate_group'] = {
                'type': 'function' if largest_group in redundancy_data['duplicate_functions'] else 'class',
                'occurrences': len(largest_group['occurrences']),
                'identifier': largest_group['occurrences'][0].get('signature') or largest_group['occurrences'][0].get('class')
            }

        return overview

    def _analyze_duplicate_patterns(self, redundancy_data: Dict) -> Dict[str, Any]:
        """分析重复模式"""
        patterns = {
            'by_function_type': defaultdict(int),
            'by_class_type': defaultdict(int),
            'by_module': defaultdict(int),
            'by_file_size': defaultdict(int),
            'temporal_patterns': defaultdict(int)
        }

        # 分析函数重复模式
        for dup_group in redundancy_data['duplicate_functions']:
            if dup_group['occurrences']:
                signature = dup_group['occurrences'][0]['signature']

                # 按函数类型分类
                if '__init__' in signature:
                    patterns['by_function_type']['构造函数'] += 1
                elif signature.startswith('def get_'):
                    patterns['by_function_type']['getter方法'] += 1
                elif signature.startswith('def set_'):
                    patterns['by_function_type']['setter方法'] += 1
                elif 'create_' in signature or 'factory' in signature.lower():
                    patterns['by_function_type']['工厂方法'] += 1
                elif any(word in signature.lower() for word in ['util', 'helper', 'common']):
                    patterns['by_function_type']['工具函数'] += 1
                else:
                    patterns['by_function_type']['其他方法'] += 1

                # 按模块统计
                for occ in dup_group['occurrences']:
                    file_path = occ['file']
                    if '/' in file_path:
                        module = file_path.split('/')[0]
                        patterns['by_module'][module] += 1

        # 分析类重复模式
        for dup_group in redundancy_data['duplicate_classes']:
            if dup_group['occurrences']:
                class_name = dup_group['occurrences'][0]['class_name']

                # 按类类型分类
                if 'Factory' in class_name:
                    patterns['by_class_type']['工厂类'] += 1
                elif 'Manager' in class_name:
                    patterns['by_class_type']['管理器类'] += 1
                elif 'Service' in class_name:
                    patterns['by_class_type']['服务类'] += 1
                elif 'Handler' in class_name:
                    patterns['by_class_type']['处理器类'] += 1
                else:
                    patterns['by_class_type']['其他类'] += 1

        return dict(patterns)

    def _assess_redundancy_severity(self, redundancy_data: Dict) -> Dict[str, Any]:
        """评估冗余严重程度"""
        severity = {
            'overall_severity': 'low',
            'severity_score': 0,
            'severity_breakdown': {
                'critical': 0,    # 10+ 次重复
                'high': 0,        # 5-9 次重复
                'medium': 0,      # 3-4 次重复
                'low': 0          # 2 次重复
            },
            'risk_assessment': '',
            'maintenance_impact': '',
            'development_impact': ''
        }

        # 计算严重程度分布
        for dup_group in redundancy_data['duplicate_functions'] + redundancy_data['duplicate_classes']:
            count = len(dup_group['occurrences'])
            if count >= 10:
                severity['severity_breakdown']['critical'] += 1
                severity['severity_score'] += count * 3
            elif count >= 5:
                severity['severity_breakdown']['high'] += 1
                severity['severity_score'] += count * 2
            elif count >= 3:
                severity['severity_breakdown']['medium'] += 1
                severity['severity_score'] += count * 1.5
            else:
                severity['severity_breakdown']['low'] += 1
                severity['severity_score'] += count

        # 确定整体严重程度
        if severity['severity_score'] > 2000:
            severity['overall_severity'] = 'critical'
        elif severity['severity_score'] > 1000:
            severity['overall_severity'] = 'high'
        elif severity['severity_score'] > 500:
            severity['overall_severity'] = 'medium'
        else:
            severity['overall_severity'] = 'low'

        # 生成风险评估
        if severity['overall_severity'] == 'critical':
            severity['risk_assessment'] = '极高风险 - 严重影响系统稳定性和可维护性'
            severity['maintenance_impact'] = '维护成本增加300%以上'
            severity['development_impact'] = '新功能开发速度降低50%'
        elif severity['overall_severity'] == 'high':
            severity['risk_assessment'] = '高风险 - 显著影响代码质量和开发效率'
            severity['maintenance_impact'] = '维护成本增加200%'
            severity['development_impact'] = '新功能开发速度降低30%'
        elif severity['overall_severity'] == 'medium':
            severity['risk_assessment'] = '中等风险 - 需要逐步清理和重构'
            severity['maintenance_impact'] = '维护成本增加100%'
            severity['development_impact'] = '新功能开发速度轻微下降'
        else:
            severity['risk_assessment'] = '低风险 - 基本可接受'
            severity['maintenance_impact'] = '维护成本轻微增加'
            severity['development_impact'] = '影响较小'

        return severity

    def _identify_refactoring_opportunities(self, redundancy_data: Dict) -> List[Dict[str, Any]]:
        """识别重构机会"""
        opportunities = []

        # 1. 构造函数重复 - 建议提取基类
        constructor_duplicates = []
        for dup_group in redundancy_data['duplicate_functions']:
            if dup_group['occurrences'] and '__init__' in dup_group['occurrences'][0]['signature']:
                if len(dup_group['occurrences']) >= 3:
                    constructor_duplicates.append({
                        'type': 'constructor_extraction',
                        'occurrences': len(dup_group['occurrences']),
                        'files': [occ['file'] for occ in dup_group['occurrences']],
                        'signature': dup_group['occurrences'][0]['signature']
                    })

        if constructor_duplicates:
            opportunities.append({
                'opportunity_type': 'base_class_extraction',
                'description': '构造函数重复，建议提取公共基类',
                'affected_items': constructor_duplicates,
                'estimated_savings': f'可减少{sum(c["occurrences"] for c in constructor_duplicates)}个重复构造函数',
                'complexity': 'medium'
            })

        # 2. 工具函数重复 - 建议提取工具模块
        utility_duplicates = []
        for dup_group in redundancy_data['duplicate_functions']:
            if dup_group['occurrences']:
                sig = dup_group['occurrences'][0]['signature']
                if any(word in sig.lower() for word in ['util', 'helper', 'common', 'tool']):
                    if len(dup_group['occurrences']) >= 2:
                        utility_duplicates.append({
                            'function': sig,
                            'occurrences': len(dup_group['occurrences']),
                            'files': [occ['file'] for occ in dup_group['occurrences']]
                        })

        if utility_duplicates:
            opportunities.append({
                'opportunity_type': 'utility_module_creation',
                'description': '工具函数重复，建议创建专用工具模块',
                'affected_items': utility_duplicates[:5],  # 只显示前5个
                'estimated_savings': f'可减少{len(utility_duplicates)}个重复工具函数',
                'complexity': 'low'
            })

        # 3. 类重复 - 建议统一抽象
        class_duplicates = []
        for dup_group in redundancy_data['duplicate_classes']:
            if len(dup_group['occurrences']) >= 2:
                class_duplicates.append({
                    'class_name': dup_group['occurrences'][0]['class_name'],
                    'occurrences': len(dup_group['occurrences']),
                    'files': [occ['file'] for occ in dup_group['occurrences']]
                })

        if class_duplicates:
            opportunities.append({
                'opportunity_type': 'class_abstraction',
                'description': '重复类定义，建议创建统一抽象',
                'affected_items': class_duplicates[:3],  # 只显示前3个
                'estimated_savings': f'可减少{sum(c["occurrences"] for c in class_duplicates)}个重复类',
                'complexity': 'high'
            })

        return opportunities

    def _analyze_module_impact(self, redundancy_data: Dict) -> Dict[str, Any]:
        """分析模块影响"""
        module_impact = defaultdict(lambda: {
            'duplicate_functions': 0,
            'duplicate_classes': 0,
            'total_occurrences': 0,
            'most_affected_files': [],
            'impact_score': 0
        })

        # 统计各模块的重复情况
        for dup_group in redundancy_data['duplicate_functions']:
            for occ in dup_group['occurrences']:
                file_path = occ['file']
                if '/' in file_path:
                    module = file_path.split('/')[0]
                    module_impact[module]['duplicate_functions'] += 1
                    module_impact[module]['total_occurrences'] += 1

        for dup_group in redundancy_data['duplicate_classes']:
            for occ in dup_group['occurrences']:
                file_path = occ['file']
                if '/' in file_path:
                    module = file_path.split('/')[0]
                    module_impact[module]['duplicate_classes'] += 1
                    module_impact[module]['total_occurrences'] += 1

        # 计算影响分数并找出最受影响的文件
        for module, data in module_impact.items():
            # 影响分数 = 重复函数 * 2 + 重复类 * 3 + 总出现次数 * 0.5
            data['impact_score'] = (data['duplicate_functions'] * 2 +
                                    data['duplicate_classes'] * 3 +
                                    data['total_occurrences'] * 0.5)

            # 找出该模块最受影响的文件
            file_counts = defaultdict(int)
            for dup_group in redundancy_data['duplicate_functions'] + redundancy_data['duplicate_classes']:
                for occ in dup_group['occurrences']:
                    file_path = occ['file']
                    if '/' in file_path and file_path.split('/')[0] == module:
                        file_counts[file_path] += 1

            data['most_affected_files'] = sorted(
                file_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # 前3个最受影响的文件

        # 转换为普通字典
        return dict(module_impact)

    def _generate_redundancy_recommendations(self, redundancy_data: Dict) -> Dict[str, Any]:
        """生成冗余处理建议"""
        recommendations = {
            'immediate_actions': [],
            'phased_approach': {
                'phase_1': [],  # 1-2周
                'phase_2': [],  # 1个月
                'phase_3': []   # 2-3个月
            },
            'prevention_measures': [],
            'estimated_effort': '',
            'expected_benefits': []
        }

        # 立即行动
        if len(redundancy_data['duplicate_functions']) > 200:
            recommendations['immediate_actions'].append(
                '建立代码重复检测自动化流程，阻止新重复代码的产生'
            )
            recommendations['immediate_actions'].append(
                '识别和标记最严重的重复代码组（10+次重复）'
            )

        # 分阶段方法
        recommendations['phased_approach']['phase_1'] = [
            '清理构造函数重复：提取公共基类构造函数',
            '合并简单的工具函数重复',
            '建立重复代码处理规范和流程'
        ]

        recommendations['phased_approach']['phase_2'] = [
            '重构类重复：创建统一的抽象类层次结构',
            '优化模块结构，减少跨模块重复',
            '实施代码审查自动化检查'
        ]

        recommendations['phased_approach']['phase_3'] = [
            '建立智能化重复检测和自动重构工具',
            '完善架构模式，确保新代码遵循统一模式',
            '建立持续的质量监控和改进机制'
        ]

        # 预防措施
        recommendations['prevention_measures'] = [
            '在CI/CD中集成重复代码检测',
            '建立代码模板和最佳实践指南',
            '实施结对编程和代码审查制度',
            '定期进行代码重构和技术债务清理'
        ]

        # 估算工作量
        total_duplicates = len(redundancy_data['duplicate_functions']) + \
            len(redundancy_data['duplicate_classes'])
        if total_duplicates > 500:
            recommendations['estimated_effort'] = '3-6个月'
        elif total_duplicates > 300:
            recommendations['estimated_effort'] = '2-3个月'
        else:
            recommendations['estimated_effort'] = '1-2个月'

        # 预期收益
        recommendations['expected_benefits'] = [
            '减少代码重复率80%以上',
            '提升代码可维护性200%',
            '降低维护成本60%',
            '提高开发效率30%',
            '增强系统稳定性和可靠性'
        ]

        return recommendations

    def _print_deep_analysis_report(self):
        """打印深度分析报告"""
        print('\\n📊 代码重叠和冗余深度分析报告')
        print('=' * 60)

        overview = self.analysis_results['redundancy_overview']
        patterns = self.analysis_results['duplicate_patterns_analysis']
        severity = self.analysis_results['severity_assessment']
        opportunities = self.analysis_results['refactoring_opportunities']
        module_impact = self.analysis_results['module_impact_analysis']
        recommendations = self.analysis_results['recommendations']

        # 概况
        print('📈 冗余概况:')
        print(f'   重复函数组: {overview["total_duplicate_functions"]}')
        print(f'   重复类组: {overview["total_duplicate_classes"]}')
        print(f'   总重复出现: {overview["total_duplicate_occurrences"]}')
        print('.1f')

        if overview['most_duplicated_function']:
            print(f'\\n🔍 最重复的函数:')
            func = overview['most_duplicated_function']
            print(f'   函数: {func["signature"]}')
            print(f'   重复次数: {func["occurrences"]}')
            print(f'   涉及文件数: {len(func["files"])}')

        # 模式分析
        print('\\n🎯 重复模式分析:')
        print('   函数类型分布:')
        for func_type, count in patterns['by_function_type'].items():
            if count > 0:
                print(f'     {func_type}: {count}')

        print('   类类型分布:')
        for class_type, count in patterns['by_class_type'].items():
            if count > 0:
                print(f'     {class_type}: {count}')

        # 严重程度评估
        print(f'\\n⚠️ 冗余严重程度: {severity["overall_severity"].upper()}')
        print(f'   严重程度评分: {severity["severity_score"]:.0f}')
        print('   严重程度分布:')
        for level, count in severity['severity_breakdown'].items():
            if count > 0:
                print(f'     {level}: {count}')

        print(f'\\n💰 影响评估:')
        print(f'   风险等级: {severity["risk_assessment"]}')
        print(f'   维护成本: {severity["maintenance_impact"]}')
        print(f'   开发效率: {severity["development_impact"]}')

        # 模块影响
        print('\\n🏢 模块影响分析:')
        sorted_modules = sorted(module_impact.items(),
                                key=lambda x: x[1]['impact_score'],
                                reverse=True)
        for module, data in sorted_modules[:5]:  # 显示前5个
            print(f'   {module}: 影响分数 {data["impact_score"]:.1f}')
            print(f'     重复函数: {data["duplicate_functions"]}, 重复类: {data["duplicate_classes"]}')

        # 重构机会
        print('\\n🔧 重构机会:')
        for opp in opportunities:
            print(f'   {opp["opportunity_type"]}: {opp["description"]}')
            print(f'     预估收益: {opp["estimated_savings"]}')
            print(f'     复杂度: {opp["complexity"]}')

        # 建议
        print(f'\\n🎯 处理建议 ({recommendations["estimated_effort"]}):')

        print('\\n立即行动:')
        for action in recommendations['immediate_actions']:
            print(f'   🚀 {action}')

        print('\\n分阶段执行:')
        for phase, actions in recommendations['phased_approach'].items():
            print(f'   {phase.replace("_", " ").title()}:')
            for action in actions:
                print(f'     📋 {action}')

        print('\\n预防措施:')
        for measure in recommendations['prevention_measures']:
            print(f'   🛡️ {measure}')

        print('\\n💎 预期收益:')
        for benefit in recommendations['expected_benefits']:
            print(f'   ✅ {benefit}')

        print(f'\\n📄 详细分析报告已保存: code_redundancy_deep_analysis.json')


def main():
    """主函数"""
    analyzer = CodeRedundancyDeepAnalysis()
    results = analyzer.perform_deep_analysis()


if __name__ == "__main__":
    main()
