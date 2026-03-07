#!/usr/bin/env python3
"""
基础设施层代码审查分析报告

根据架构设计对代码实现进行全面审查
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any


class InfrastructureReviewAnalysis:
    """基础设施审查分析"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.report = {}

    def generate_review_report(self) -> Dict[str, Any]:
        """生成审查报告"""
        print('🔍 开始基础设施层代码审查分析')
        print('=' * 60)

        # 加载审查数据
        with open('infrastructure_code_review_report.json', 'r', encoding='utf-8') as f:
            self.report = json.load(f)

        analysis = {
            'executive_summary': self._generate_executive_summary(),
            'code_organization_review': self._review_code_organization(),
            'redundancy_analysis': self._analyze_code_redundancy(),
            'architecture_compliance_review': self._review_architecture_compliance(),
            'quality_assessment': self._assess_code_quality(),
            'improvement_recommendations': self._generate_improvement_plan()
        }

        # 保存分析报告
        with open('infrastructure_review_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print('\\n✅ 基础设施层代码审查分析完成')
        self._print_final_report(analysis)

        return analysis

    def _generate_executive_summary(self) -> Dict[str, Any]:
        """生成执行摘要"""
        summary = {
            'overall_assessment': 'good',
            'key_findings': [],
            'critical_issues': 0,
            'major_issues': 0,
            'minor_issues': 0,
            'strengths': [],
            'weaknesses': []
        }

        # 从审查报告中提取关键指标
        redundancy = self.report['detailed_results']['redundancy_analysis']
        architecture = self.report['summary']['architecture_compliance']
        quality = self.report['summary']['code_quality_score']

        # 分析冗余问题
        duplicate_functions = len(redundancy['duplicate_functions'])
        duplicate_classes = len(redundancy['duplicate_classes'])

        if duplicate_functions > 200:
            summary['critical_issues'] += 1
            summary['key_findings'].append(f'严重代码重复: {duplicate_functions}组重复函数')
            summary['weaknesses'].append('代码冗余度过高')

        if architecture < 0.8:
            summary['major_issues'] += 1
            summary['key_findings'].append(f'架构一致性不足: {architecture*100:.1f}%')
            summary['weaknesses'].append('架构设计与实现不一致')

        if quality < 80:
            summary['minor_issues'] += 1
            summary['key_findings'].append(f'代码质量待提升: {quality}/100')
            summary['weaknesses'].append('代码质量有改善空间')

        # 分析优势
        if architecture >= 0.7:
            summary['strengths'].append('架构基础良好')

        doc_coverage = self.report['detailed_results']['quality_metrics']['documentation_coverage']
        if doc_coverage > 95:
            summary['strengths'].append('文档覆盖率优秀')

        # 确定整体评估
        total_issues = summary['critical_issues'] + \
            summary['major_issues'] + summary['minor_issues']
        if total_issues == 0:
            summary['overall_assessment'] = 'excellent'
        elif total_issues <= 2:
            summary['overall_assessment'] = 'good'
        elif total_issues <= 4:
            summary['overall_assessment'] = 'fair'
        else:
            summary['overall_assessment'] = 'needs_improvement'

        return summary

    def _review_code_organization(self) -> Dict[str, Any]:
        """审查代码组织"""
        org_data = self.report['detailed_results']['code_organization']

        review = {
            'file_count': org_data['file_sizes']['total_files'],
            'average_file_size': org_data['file_sizes']['avg_size_kb'],
            'organization_rating': 'good',
            'issues': [],
            'recommendations': []
        }

        # 评估文件数量
        if review['file_count'] > 400:
            review['issues'].append('文件数量过多，建议合并')
            review['organization_rating'] = 'poor'
        elif review['file_count'] > 300:
            review['issues'].append('文件数量偏多，考虑优化')
            review['organization_rating'] = 'fair'
        else:
            review['recommendations'].append('文件数量合理')

        # 评估文件大小分布
        size_dist = org_data['file_sizes']['size_distribution']
        small_files = size_dist.get('small', 0) + size_dist.get('tiny', 0)
        large_files = size_dist.get('large', 0) + size_dist.get('very_large', 0)

        total_files = sum(size_dist.values())
        if total_files > 0:
            small_ratio = small_files / total_files
            large_ratio = large_files / total_files

            if small_ratio > 0.4:
                review['issues'].append('过多的细小文件，建议合并')
            if large_ratio > 0.1:
                review['issues'].append('存在过大的文件，建议拆分')

        return review

    def _analyze_code_redundancy(self) -> Dict[str, Any]:
        """分析代码冗余"""
        redundancy_data = self.report['detailed_results']['redundancy_analysis']

        analysis = {
            'duplicate_functions': len(redundancy_data['duplicate_functions']),
            'duplicate_classes': len(redundancy_data['duplicate_classes']),
            'total_duplicate_occurrences': 0,
            'redundancy_severity': 'low',
            'most_common_duplicates': [],
            'impact_assessment': ''
        }

        # 计算总的重复出现次数
        for dup_group in redundancy_data['duplicate_functions']:
            analysis['total_duplicate_occurrences'] += len(dup_group['occurrences'])

        for dup_group in redundancy_data['duplicate_classes']:
            analysis['total_duplicate_occurrences'] += len(dup_group['occurrences'])

        # 评估冗余严重程度
        if analysis['duplicate_functions'] > 300:
            analysis['redundancy_severity'] = 'critical'
            analysis['impact_assessment'] = '严重影响维护效率和发展速度'
        elif analysis['duplicate_functions'] > 200:
            analysis['redundancy_severity'] = 'high'
            analysis['impact_assessment'] = '显著影响代码质量和开发效率'
        elif analysis['duplicate_functions'] > 100:
            analysis['redundancy_severity'] = 'medium'
            analysis['impact_assessment'] = '需要逐步清理和重构'
        else:
            analysis['redundancy_severity'] = 'low'
            analysis['impact_assessment'] = '基本可接受，建议持续监控'

        # 找出最常见的重复模式
        function_patterns = defaultdict(int)
        for dup_group in redundancy_data['duplicate_functions']:
            if dup_group['occurrences']:
                pattern = dup_group['occurrences'][0]['signature']
                if '__init__' in pattern:
                    function_patterns['构造函数'] += 1
                elif any(kw in pattern.lower() for kw in ['get_', 'set_', 'create_']):
                    function_patterns['Getter/Setter/Factory方法'] += 1
                else:
                    function_patterns['工具函数'] += 1

        analysis['most_common_duplicates'] = sorted(
            function_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return analysis

    def _review_architecture_compliance(self) -> Dict[str, Any]:
        """审查架构一致性"""
        compliance_data = self.report['detailed_results']['architecture_compliance']
        compliance_score = self.report['summary']['architecture_compliance']

        review = {
            'compliance_score': compliance_score * 100,
            'inheritance_issues': 0,
            'compliance_rating': 'good',
            'problematic_modules': [],
            'architecture_strengths': [],
            'architecture_weaknesses': []
        }

        # 分析继承问题
        if 'interface_inheritance' in compliance_data:
            issues = compliance_data['interface_inheritance'].get('issues', [])
            review['inheritance_issues'] = len(issues)

            # 按模块分组问题
            module_issues = defaultdict(int)
            for issue in issues:
                file_path = issue.get('file', '')
                if '/' in file_path:
                    module = file_path.split('/')[0]
                    module_issues[module] += 1

            # 找出问题最多的模块
            review['problematic_modules'] = sorted(
                module_issues.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

        # 评估一致性等级
        if review['compliance_score'] >= 90:
            review['compliance_rating'] = 'excellent'
            review['architecture_strengths'].append('架构设计与实现高度一致')
        elif review['compliance_score'] >= 80:
            review['compliance_rating'] = 'good'
            review['architecture_strengths'].append('架构基础良好')
        elif review['compliance_score'] >= 70:
            review['compliance_rating'] = 'fair'
            review['architecture_weaknesses'].append('架构一致性有待提升')
        else:
            review['compliance_rating'] = 'poor'
            review['architecture_weaknesses'].append('架构设计与实现严重脱节')

        return review

    def _assess_code_quality(self) -> Dict[str, Any]:
        """评估代码质量"""
        quality_data = self.report['detailed_results']['quality_metrics']

        assessment = {
            'overall_quality_score': self.report['summary']['code_quality_score'],
            'maintainability_index': self.report['summary']['maintainability_index'],
            'documentation_coverage': quality_data.get('documentation_coverage', 0),
            'quality_rating': 'good',
            'quality_strengths': [],
            'quality_weaknesses': []
        }

        # 评估质量等级
        if assessment['overall_quality_score'] >= 90:
            assessment['quality_rating'] = 'excellent'
        elif assessment['overall_quality_score'] >= 80:
            assessment['quality_rating'] = 'good'
        elif assessment['overall_quality_score'] >= 70:
            assessment['quality_rating'] = 'fair'
        elif assessment['overall_quality_score'] >= 60:
            assessment['quality_rating'] = 'poor'
        else:
            assessment['quality_rating'] = 'critical'

        # 分析各项指标
        if assessment['documentation_coverage'] > 95:
            assessment['quality_strengths'].append('文档覆盖率优秀')
        elif assessment['documentation_coverage'] < 80:
            assessment['quality_weaknesses'].append('文档覆盖率不足')

        if assessment['maintainability_index'] < 50:
            assessment['quality_weaknesses'].append('可维护性严重不足')
        elif assessment['maintainability_index'] > 80:
            assessment['quality_strengths'].append('可维护性良好')

        return assessment

    def _generate_improvement_plan(self) -> Dict[str, Any]:
        """生成改进计划"""
        plan = {
            'immediate_actions': [],    # 立即执行 (< 1周)
            'short_term_goals': [],     # 短期目标 (1-4周)
            'medium_term_goals': [],    # 中期目标 (1-3个月)
            'long_term_vision': [],     # 长期愿景 (3-6个月)
            'estimated_effort': '',
            'expected_benefits': []
        }

        # 基于问题严重程度生成计划
        redundancy = self._analyze_code_redundancy()
        architecture = self._review_architecture_compliance()
        quality = self._assess_code_quality()

        # 紧急行动
        if redundancy['redundancy_severity'] == 'critical':
            plan['immediate_actions'].append('启动大规模代码重复消除项目')
            plan['immediate_actions'].append('建立代码重复检测自动化流程')

        if architecture['compliance_score'] < 75:
            plan['immediate_actions'].append('修复关键架构一致性问题')

        # 短期目标
        plan['short_term_goals'].append('完善自动化代码审查体系')
        plan['short_term_goals'].append('优化代码组织结构')
        plan['short_term_goals'].append('提升文档质量和覆盖率')

        # 中期目标
        plan['medium_term_goals'].append('实现全面的架构一致性')
        plan['medium_term_goals'].append('建立持续集成和质量门禁')
        plan['medium_term_goals'].append('优化系统性能和可维护性')

        # 长期愿景
        plan['long_term_vision'].append('建立智能化代码质量保障体系')
        plan['long_term_vision'].append('实现DevOps全流程自动化')
        plan['long_term_vision'].append('打造行业领先的代码质量标准')

        # 估算工作量
        total_issues = len(plan['immediate_actions']) + len(plan['short_term_goals'])
        if total_issues > 5:
            plan['estimated_effort'] = '2-3个月'
        elif total_issues > 3:
            plan['estimated_effort'] = '1-2个月'
        else:
            plan['estimated_effort'] = '2-4周'

        # 预期收益
        plan['expected_benefits'] = [
            '显著提升代码质量和可维护性',
            '减少开发和维护成本',
            '提高系统稳定性和可靠性',
            '增强团队开发效率',
            '建立可持续的质量保障体系'
        ]

        return plan

    def _print_final_report(self, analysis: Dict[str, Any]):
        """打印最终报告"""
        print('\\n📋 基础设施层代码审查分析报告')
        print('=' * 60)

        # 执行摘要
        summary = analysis['executive_summary']
        print(f'🎯 总体评估: {summary["overall_assessment"].upper()}')
        print(f'📊 关键发现: {len(summary["key_findings"])} 项')
        print(f'🚨 严重问题: {summary["critical_issues"]} 个')
        print(f'⚠️ 主要问题: {summary["major_issues"]} 个')
        print(f'💡 改进建议: {summary["minor_issues"]} 个')

        print('\\n🏆 优势:')
        for strength in summary['strengths']:
            print(f'   ✅ {strength}')

        print('\\n⚠️ 需要改进:')
        for weakness in summary['weaknesses']:
            print(f'   ❌ {weakness}')

        # 代码组织审查
        org = analysis['code_organization_review']
        print(f'\\n📁 代码组织: {org["organization_rating"].upper()}')
        print(f'   文件数量: {org["file_count"]} 个')
        print(f'   平均文件大小: {org["average_file_size"]:.1f} KB')

        # 代码冗余分析
        red = analysis['redundancy_analysis']
        print(f'\\n🔄 代码冗余: {red["redundancy_severity"].upper()}')
        print(f'   重复函数: {red["duplicate_functions"]} 组')
        print(f'   重复类: {red["duplicate_classes"]} 组')
        print(f'   总重复出现: {red["total_duplicate_occurrences"]} 次')
        print(f'   影响评估: {red["impact_assessment"]}')

        # 架构一致性审查
        arch = analysis['architecture_compliance_review']
        print(f'\\n🏗️ 架构一致性: {arch["compliance_rating"].upper()}')
        print(f'   一致性评分: {arch["compliance_score"]:.1f}%')
        print(f'   继承问题: {arch["inheritance_issues"]} 个')

        # 代码质量评估
        quality = analysis['quality_assessment']
        print(f'\\n📊 代码质量: {quality["quality_rating"].upper()}')
        print(f'   整体评分: {quality["overall_quality_score"]}/100')
        print(f'   可维护性: {quality["maintainability_index"]}/100')
        print(f'   文档覆盖: {quality["documentation_coverage"]:.1f}%')

        # 改进计划
        plan = analysis['improvement_recommendations']
        print(f'\\n🎯 改进计划 ({plan["estimated_effort"]}):')

        print('\\n立即行动 (< 1周):')
        for action in plan['immediate_actions']:
            print(f'   🚀 {action}')

        print('\\n短期目标 (1-4周):')
        for goal in plan['short_term_goals']:
            print(f'   📅 {goal}')

        print('\\n中期目标 (1-3个月):')
        for goal in plan['medium_term_goals']:
            print(f'   🎯 {goal}')

        print('\\n长期愿景 (3-6个月):')
        for vision in plan['long_term_vision']:
            print(f'   🚀 {vision}')

        print('\\n💰 预期收益:')
        for benefit in plan['expected_benefits']:
            print(f'   ✅ {benefit}')

        print(f'\\n📄 详细分析报告已保存: infrastructure_review_analysis_report.json')


def main():
    """主函数"""
    analyzer = InfrastructureReviewAnalysis()
    results = analyzer.generate_review_report()


if __name__ == "__main__":
    main()
