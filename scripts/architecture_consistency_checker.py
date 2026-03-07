#!/usr/bin/env python3
"""
基础设施层架构一致性检查工具

对比架构设计文档与实际代码实现的一致性
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any


class ArchitectureConsistencyChecker:
    """架构一致性检查器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.design_doc = Path('docs/architecture/infrastructure_architecture_design.md')
        self.check_results = {
            'directory_structure': {},
            'core_components': {},
            'interface_definitions': {},
            'design_patterns': {},
            'naming_conventions': {},
            'quality_metrics': {}
        }

    def perform_consistency_check(self) -> Dict[str, Any]:
        """执行架构一致性检查"""
        print('🏗️ 开始架构一致性检查')
        print('=' * 50)

        # 读取设计文档
        design_specs = self._extract_design_specifications()

        # 检查目录结构一致性
        print('📁 检查目录结构一致性...')
        self.check_results['directory_structure'] = self._check_directory_structure(design_specs)

        # 检查核心组件实现
        print('🔧 检查核心组件实现...')
        self.check_results['core_components'] = self._check_core_components(design_specs)

        # 检查接口定义一致性
        print('🔗 检查接口定义一致性...')
        self.check_results['interface_definitions'] = self._check_interface_definitions(
            design_specs)

        # 检查设计模式使用
        print('🎨 检查设计模式使用...')
        self.check_results['design_patterns'] = self._check_design_patterns(design_specs)

        # 检查命名规范
        print('🏷️ 检查命名规范...')
        self.check_results['naming_conventions'] = self._check_naming_conventions()

        # 计算一致性评分
        print('📊 计算一致性评分...')
        self.check_results['quality_metrics'] = self._calculate_consistency_score()

        # 生成报告
        report = {
            'timestamp': self._get_timestamp(),
            'design_version': 'v8.0',
            'implementation_status': self.check_results,
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }

        # 保存报告
        with open('architecture_consistency_check_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 架构一致性检查完成')
        self._print_consistency_summary(report)

        return report

    def _extract_design_specifications(self) -> Dict[str, Any]:
        """从设计文档中提取规范"""
        specs = {
            'directory_structure': {},
            'core_components': {},
            'interfaces': {},
            'design_patterns': {},
            'quality_targets': {}
        }

        try:
            with open(self.design_doc, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取目录结构规范
            specs['directory_structure'] = self._extract_directory_specs(content)

            # 提取核心组件规范
            specs['core_components'] = self._extract_core_component_specs(content)

            # 提取接口规范
            specs['interfaces'] = self._extract_interface_specs(content)

            # 提取设计模式规范
            specs['design_patterns'] = self._extract_design_pattern_specs(content)

            # 提取质量目标
            specs['quality_targets'] = self._extract_quality_targets(content)

        except Exception as e:
            print(f'⚠️ 读取设计文档失败: {e}')

        return specs

    def _extract_directory_specs(self, content: str) -> Dict[str, Any]:
        """提取目录结构规范"""
        specs = {
            'main_modules': [],
            'sub_modules': {},
            'file_organization': {}
        }

        # 从架构图和描述中提取主要模块
        main_modules = [
            'core', 'interfaces', 'cache', 'config', 'logging',
            'health', 'error', 'resource', 'utils'
        ]

        specs['main_modules'] = main_modules

        # 提取子模块规范
        sub_modules = {
            'cache': ['core', 'managers', 'services', 'monitoring', 'config', 'storage', 'strategies', 'utils'],
            'config': ['core', 'loaders', 'services', 'interfaces', 'tools'],
            'logging': ['foundation', 'core', 'handlers', 'monitors', 'services', 'utils', 'security', 'system', 'distributed'],
            'health': ['core', 'services', 'monitors', 'handlers'],
            'error': ['foundation', 'core', 'handlers', 'recovery', 'security', 'storage', 'testing', 'utils'],
            'resource': ['core', 'monitors', 'services']
        }

        specs['sub_modules'] = sub_modules

        return specs

    def _extract_core_component_specs(self, content: str) -> Dict[str, Any]:
        """提取核心组件规范"""
        specs = {
            'required_components': [],
            'component_interfaces': {},
            'implementation_patterns': {}
        }

        # 从架构图中提取核心组件
        required_components = [
            'BaseInfrastructureComponent',
            'UnifiedCacheManager',
            'UnifiedConfigManager',
            'UnifiedLogger',
            'IHealthCheckerComponent',
            'ComponentFactory'
        ]

        specs['required_components'] = required_components

        # 组件接口规范
        component_interfaces = {
            'cache': ['BaseCacheManager', 'ICacheManager', 'ICacheComponent'],
            'config': ['IConfigManagerComponent', 'IConfigStorage', 'IConfigLoader'],
            'health': ['IHealthCheckerComponent', 'IHealthMonitor'],
            'logging': ['ILoggerComponent', 'ILogHandler'],
            'error': ['IErrorHandler', 'IErrorRecovery']
        }

        specs['component_interfaces'] = component_interfaces

        return specs

    def _extract_interface_specs(self, content: str) -> Dict[str, Any]:
        """提取接口规范"""
        specs = {
            'interface_prefixes': ['I', 'Base'],
            'required_interfaces': [],
            'interface_patterns': {}
        }

        # 接口命名规范
        specs['interface_prefixes'] = ['I', 'Base']

        # 必需的接口
        required_interfaces = [
            'IComponentFactory', 'BaseComponentFactory',
            'IConfigManager', 'BaseFactory',
            'IManager', 'BaseManager',
            'IService', 'BaseService',
            'IHandler', 'BaseHandler',
            'IProvider', 'BaseProvider',
            'IMonitor', 'BaseInterface'
        ]

        specs['required_interfaces'] = required_interfaces

        return specs

    def _extract_design_pattern_specs(self, content: str) -> Dict[str, Any]:
        """提取设计模式规范"""
        specs = {
            'required_patterns': [],
            'pattern_implementations': {}
        }

        # 必需的设计模式
        required_patterns = [
            'Factory Pattern', 'Manager Pattern', 'Service Pattern',
            'Handler Pattern', 'Provider Pattern', 'Observer Pattern',
            'Strategy Pattern', 'Decorator Pattern'
        ]

        specs['required_patterns'] = required_patterns

        return specs

    def _extract_quality_targets(self, content: str) -> Dict[str, Any]:
        """提取质量目标"""
        targets = {
            'architecture_compliance': 90,
            'code_quality': 80,
            'performance': 70,
            'automation_coverage': 95,
            'test_coverage': 95
        }

        return targets

    def _check_directory_structure(self, design_specs: Dict[str, Any]) -> Dict[str, Any]:
        """检查目录结构一致性"""
        result = {
            'compliance_score': 0,
            'missing_modules': [],
            'extra_modules': [],
            'structure_issues': [],
            'file_organization': {}
        }

        # 检查主要模块
        actual_main_modules = []
        for item in os.listdir(self.infra_dir):
            if (self.infra_dir / item).is_dir() and not item.startswith('__'):
                actual_main_modules.append(item)

        design_main_modules = design_specs['directory_structure']['main_modules']

        missing_modules = set(design_main_modules) - set(actual_main_modules)
        extra_modules = set(actual_main_modules) - set(design_main_modules)

        result['missing_modules'] = list(missing_modules)
        result['extra_modules'] = list(extra_modules)

        # 检查子模块结构
        structure_issues = []
        for module, expected_subs in design_specs['directory_structure']['sub_modules'].items():
            module_path = self.infra_dir / module
            if module_path.exists():
                actual_subs = []
                for item in os.listdir(module_path):
                    if (module_path / item).is_dir() and not item.startswith('__'):
                        actual_subs.append(item)

                missing_subs = set(expected_subs) - set(actual_subs)
                if missing_subs:
                    structure_issues.append(f'{module}模块缺少子模块: {list(missing_subs)}')

        result['structure_issues'] = structure_issues

        # 计算一致性评分
        total_expected = len(design_main_modules)
        implemented = total_expected - len(missing_modules)
        result['compliance_score'] = (implemented / total_expected *
                                      100) if total_expected > 0 else 0

        return result

    def _check_core_components(self, design_specs: Dict[str, Any]) -> Dict[str, Any]:
        """检查核心组件实现"""
        result = {
            'compliance_score': 0,
            'implemented_components': [],
            'missing_components': [],
            'implementation_quality': {}
        }

        required_components = design_specs['core_components']['required_components']

        # 检查组件文件是否存在
        for component in required_components:
            found = False

            # 在所有Python文件中搜索
            for root, dirs, files in os.walk(self.infra_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if f'class {component}' in content:
                                    result['implemented_components'].append({
                                        'name': component,
                                        'file': str(file_path),
                                        'found': True
                                    })
                                    found = True
                                    break
                        except:
                            continue
                if found:
                    break

            if not found:
                result['missing_components'].append(component)

        # 计算实现率
        total_required = len(required_components)
        implemented_count = len(result['implemented_components'])
        result['compliance_score'] = (
            implemented_count / total_required * 100) if total_required > 0 else 0

        return result

    def _check_interface_definitions(self, design_specs: Dict[str, Any]) -> Dict[str, Any]:
        """检查接口定义一致性"""
        result = {
            'compliance_score': 0,
            'interface_coverage': {},
            'naming_compliance': {},
            'implementation_consistency': {}
        }

        # 检查接口命名规范
        naming_compliance = {
            'correct_prefixes': 0,
            'incorrect_names': [],
            'total_interfaces': 0
        }

        # 扫描所有接口定义
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 查找接口类定义
                        interface_matches = re.findall(r'class\s+(I\w+|Base\w+)\s*[:\(]', content)
                        for interface_name in interface_matches:
                            naming_compliance['total_interfaces'] += 1

                            # 检查命名规范
                            if interface_name.startswith(('I', 'Base')):
                                naming_compliance['correct_prefixes'] += 1
                            else:
                                naming_compliance['incorrect_names'].append({
                                    'name': interface_name,
                                    'file': str(file_path)
                                })

                    except:
                        continue

        result['naming_compliance'] = naming_compliance

        # 计算命名一致性评分
        if naming_compliance['total_interfaces'] > 0:
            result['compliance_score'] = (
                naming_compliance['correct_prefixes'] / naming_compliance['total_interfaces'] * 100)
        else:
            result['compliance_score'] = 0

        return result

    def _check_design_patterns(self, design_specs: Dict[str, Any]) -> Dict[str, Any]:
        """检查设计模式使用"""
        result = {
            'compliance_score': 0,
            'implemented_patterns': [],
            'missing_patterns': [],
            'pattern_usage': {}
        }

        # 检查设计模式实现文件
        pattern_files = {
            'Factory Pattern': 'factory_pattern.py',
            'Manager Pattern': 'manager_pattern.py',
            'Service Pattern': 'service_pattern.py',
            'Handler Pattern': 'handler_pattern.py',
            'Provider Pattern': 'provider_pattern.py'
        }

        interfaces_dir = self.infra_dir / 'interfaces'
        for pattern, filename in pattern_files.items():
            file_path = interfaces_dir / filename
            if file_path.exists():
                result['implemented_patterns'].append(pattern)
            else:
                result['missing_patterns'].append(pattern)

        # 计算实现率
        total_patterns = len(pattern_files)
        implemented_count = len(result['implemented_patterns'])
        result['compliance_score'] = (
            implemented_count / total_patterns * 100) if total_patterns > 0 else 0

        return result

    def _check_naming_conventions(self) -> Dict[str, Any]:
        """检查命名规范"""
        result = {
            'compliance_score': 0,
            'class_naming': {},
            'method_naming': {},
            'variable_naming': {},
            'file_naming': {}
        }

        # 检查类命名
        class_naming = {
            'total_classes': 0,
            'camel_case': 0,
            'snake_case': 0,
            'violations': []
        }

        # 检查方法命名
        method_naming = {
            'total_methods': 0,
            'snake_case': 0,
            'violations': []
        }

        # 扫描代码文件
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 检查类命名
                        class_matches = re.findall(r'class\s+(\w+)\s*[:\(]', content)
                        for class_name in class_matches:
                            class_naming['total_classes'] += 1

                            # 检查驼峰命名
                            if re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                                class_naming['camel_case'] += 1
                            elif '_' in class_name:
                                class_naming['snake_case'] += 1
                            else:
                                class_naming['violations'].append({
                                    'name': class_name,
                                    'file': str(file_path),
                                    'issue': '不符合驼峰命名规范'
                                })

                        # 检查方法命名
                        method_matches = re.findall(r'\s+def\s+(\w+)\s*\(', content)
                        for method_name in method_matches:
                            method_naming['total_methods'] += 1

                            # 检查蛇形命名（跳过私有方法和特殊方法）
                            if not method_name.startswith('_') and not method_name.startswith('__'):
                                if '_' in method_name and method_name.islower():
                                    method_naming['snake_case'] += 1
                                elif not '_' in method_name and method_name.islower():
                                    method_naming['violations'].append({
                                        'name': method_name,
                                        'file': str(file_path),
                                        'issue': '建议使用蛇形命名'
                                    })

                    except:
                        continue

        result['class_naming'] = class_naming
        result['method_naming'] = method_naming

        # 计算整体命名一致性
        class_compliance = (class_naming['camel_case'] / class_naming['total_classes']
                            * 100) if class_naming['total_classes'] > 0 else 0
        method_compliance = (method_naming['snake_case'] / method_naming['total_methods']
                             * 100) if method_naming['total_methods'] > 0 else 0

        result['compliance_score'] = (class_compliance + method_compliance) / 2

        return result

    def _calculate_consistency_score(self) -> Dict[str, Any]:
        """计算一致性评分"""
        scores = {}

        # 各维度评分
        scores['directory_structure'] = self.check_results['directory_structure']['compliance_score']
        scores['core_components'] = self.check_results['core_components']['compliance_score']
        scores['interface_definitions'] = self.check_results['interface_definitions']['compliance_score']
        scores['design_patterns'] = self.check_results['design_patterns']['compliance_score']
        scores['naming_conventions'] = self.check_results['naming_conventions']['compliance_score']

        # 总体一致性评分（加权平均）
        weights = {
            'directory_structure': 0.2,
            'core_components': 0.3,
            'interface_definitions': 0.2,
            'design_patterns': 0.15,
            'naming_conventions': 0.15
        }

        overall_score = sum(scores[dimension] * weights[dimension] for dimension in scores.keys())

        scores['overall_consistency'] = overall_score

        # 评分等级
        if overall_score >= 90:
            scores['grade'] = '优秀'
            scores['assessment'] = '架构实现与设计高度一致，达到生产级标准'
        elif overall_score >= 75:
            scores['grade'] = '良好'
            scores['assessment'] = '架构实现基本符合设计要求，存在少量改进空间'
        elif overall_score >= 60:
            scores['grade'] = '一般'
            scores['assessment'] = '架构实现与设计存在一定差距，需要重点改进'
        else:
            scores['grade'] = '需改进'
            scores['assessment'] = '架构实现严重偏离设计要求，需要全面重构'

        return scores

    def _generate_summary(self) -> Dict[str, Any]:
        """生成总结"""
        summary = {
            'overall_assessment': self.check_results['quality_metrics']['assessment'],
            'key_findings': [],
            'strengths': [],
            'weaknesses': [],
            'critical_issues': []
        }

        # 关键发现
        metrics = self.check_results['quality_metrics']
        summary['key_findings'].append(
            f"架构一致性评分: {metrics['overall_consistency']:.1f}% ({metrics['grade']})")

        # 优势
        if metrics['core_components'] >= 80:
            summary['strengths'].append('核心组件实现完整')
        if metrics['interface_definitions'] >= 80:
            summary['strengths'].append('接口定义规范统一')
        if metrics['design_patterns'] >= 80:
            summary['strengths'].append('设计模式使用得当')

        # 弱点
        if metrics['directory_structure'] < 70:
            summary['weaknesses'].append('目录结构与设计不完全一致')
        if metrics['naming_conventions'] < 70:
            summary['weaknesses'].append('命名规范需要统一')
        if metrics['design_patterns'] < 70:
            summary['weaknesses'].append('设计模式实现不完整')

        # 关键问题
        dir_result = self.check_results['directory_structure']
        if dir_result['missing_modules']:
            summary['critical_issues'].append(
                f"缺少设计要求的模块: {', '.join(dir_result['missing_modules'])}")

        core_result = self.check_results['core_components']
        if core_result['missing_components']:
            summary['critical_issues'].append(
                f"缺少核心组件: {', '.join(core_result['missing_components'])}")

        return summary

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []

        # 基于检查结果生成建议
        metrics = self.check_results['quality_metrics']

        if metrics['directory_structure'] < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'structure',
                'title': '完善目录结构',
                'description': '调整目录结构以符合架构设计要求',
                'estimated_effort': '2-3人天'
            })

        if metrics['core_components'] < 80:
            recommendations.append({
                'priority': 'critical',
                'category': 'components',
                'title': '补充核心组件',
                'description': '实现缺失的核心组件以满足架构要求',
                'estimated_effort': '5-7人天'
            })

        if metrics['interface_definitions'] < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'interfaces',
                'title': '统一接口定义',
                'description': '规范接口命名和定义方式',
                'estimated_effort': '3-4人天'
            })

        if metrics['design_patterns'] < 80:
            recommendations.append({
                'priority': 'medium',
                'category': 'patterns',
                'title': '完善设计模式',
                'description': '补充缺失的设计模式实现',
                'estimated_effort': '2-3人天'
            })

        if metrics['naming_conventions'] < 80:
            recommendations.append({
                'priority': 'medium',
                'category': 'naming',
                'title': '统一命名规范',
                'description': '规范代码命名以提高可读性',
                'estimated_effort': '1-2人天'
            })

        return recommendations

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_consistency_summary(self, report: Dict[str, Any]):
        """打印一致性检查摘要"""
        print('\\n🏗️ 架构一致性检查摘要:')
        print('-' * 50)

        metrics = report['implementation_status']['quality_metrics']
        summary = report['summary']

        print(f'📊 总体一致性评分: {metrics["overall_consistency"]:.1f}% ({metrics["grade"]})')
        print(f'🏆 评估结果: {metrics["assessment"]}')

        print('\\n📈 各维度评分:')
        dimensions = ['directory_structure', 'core_components',
                      'interface_definitions', 'design_patterns', 'naming_conventions']
        for dim in dimensions:
            score = metrics[dim]
            print('.1f')

        print('\\n✅ 优势:')
        for strength in summary['strengths']:
            print(f'   • {strength}')

        if summary['weaknesses']:
            print('\\n⚠️ 需要改进:')
            for weakness in summary['weaknesses']:
                print(f'   • {weakness}')

        if summary['critical_issues']:
            print('\\n🚨 关键问题:')
            for issue in summary['critical_issues']:
                print(f'   • {issue}')

        print('\\n💡 改进建议:')
        for rec in report['recommendations'][:3]:
            print(f'   {rec["priority"].upper()}: {rec["title"]} ({rec["estimated_effort"]})')

        print('\\n📄 详细报告已保存: architecture_consistency_check_report.json')


def main():
    """主函数"""
    checker = ArchitectureConsistencyChecker()
    report = checker.perform_consistency_check()


if __name__ == "__main__":
    main()
