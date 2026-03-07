#!/usr/bin/env python3
"""
增强的代码审查系统

提供全面的代码质量检查和改进建议
"""

import json
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Any


class EnhancedCodeReviewer:
    """增强的代码审查器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.check_rules = {
            'code_quality': self._check_code_quality,
            'architecture_consistency': self._check_architecture_consistency,
            'security_issues': self._check_security_issues,
            'performance_concerns': self._check_performance_concerns,
            'maintainability_metrics': self._check_maintainability_metrics,
            'best_practices': self._check_best_practices
        }

    def perform_enhanced_review(self) -> Dict[str, Any]:
        """执行增强审查"""
        print('🔍 开始增强代码审查')
        print('=' * 50)

        # 获取所有Python文件
        python_files = self._get_python_files()
        print(f'📁 发现 {len(python_files)} 个Python文件')

        # 执行各项检查
        review_results = {}
        for check_name, check_func in self.check_rules.items():
            print(f'🔍 执行 {check_name} 检查...')
            review_results[check_name] = check_func(python_files)

        # 生成综合报告
        comprehensive_report = {
            'timestamp': self._get_timestamp(),
            'files_analyzed': len(python_files),
            'check_results': review_results,
            'summary': self._generate_summary(review_results),
            'recommendations': self._generate_recommendations(review_results),
            'quality_score': self._calculate_quality_score(review_results)
        }

        # 保存报告
        with open('enhanced_code_review_report.json', 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 增强代码审查完成')
        self._print_enhanced_summary(comprehensive_report)

        return comprehensive_report

    def _get_python_files(self) -> List[Path]:
        """获取所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def _check_code_quality(self, files: List[Path]) -> Dict[str, Any]:
        """检查代码质量"""
        results = {
            'syntax_errors': [],
            'complex_functions': [],
            'long_files': [],
            'unused_imports': [],
            'code_style_issues': []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 语法检查
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    results['syntax_errors'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })

                # 复杂函数检查
                complex_funcs = self._find_complex_functions(content, str(file_path))
                results['complex_functions'].extend(complex_funcs)

                # 长文件检查
                if len(content.split('\n')) > 500:
                    results['long_files'].append({
                        'file': str(file_path),
                        'lines': len(content.split('\n'))
                    })

                # 未使用的导入检查
                unused_imports = self._find_unused_imports(content, str(file_path))
                results['unused_imports'].extend(unused_imports)

            except Exception as e:
                results['code_style_issues'].append({
                    'file': str(file_path),
                    'issue': f'文件读取错误: {e}'
                })

        return results

    def _find_complex_functions(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """查找复杂函数"""
        complex_functions = []

        # 简单的复杂度检查（行数）
        functions = re.findall(
            r'def\s+(\w+)\([^)]*\):(.*?)(?=\n\s*def|\n\s*@|\n\s*class|\nclass|\n@|\Z)', content, re.DOTALL)

        for func_name, func_body in functions:
            lines = len(func_body.split('\n'))
            if lines > 30:  # 超过30行的函数认为复杂
                complex_functions.append({
                    'file': file_path,
                    'function': func_name,
                    'lines': lines,
                    'complexity': 'high' if lines > 50 else 'medium'
                })

        return complex_functions

    def _find_unused_imports(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """查找未使用的导入"""
        unused_imports = []

        # 解析AST来检查导入使用情况
        try:
            tree = ast.parse(content)

            # 获取所有导入
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            # 检查使用情况（简化检查）
            used_imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_imports.add(node.id)

            # 找出可能的未使用导入
            potentially_unused = imports - used_imports
            if potentially_unused:
                unused_imports.append({
                    'file': file_path,
                    'potentially_unused': list(potentially_unused)
                })

        except:
            pass  # AST解析失败时跳过

        return unused_imports

    def _check_architecture_consistency(self, files: List[Path]) -> Dict[str, Any]:
        """检查架构一致性"""
        results = {
            'missing_base_classes': [],
            'inconsistent_naming': [],
            'circular_dependencies': [],
            'interface_compliance': []
        }

        # 检查基类继承
        base_classes = ['BaseInterface', 'BaseFactory', 'BaseManager', 'BaseService']

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否有类定义但缺少基类继承
                class_defs = re.findall(r'class\s+(\w+)\s*\(', content)
                for class_name in class_defs:
                    if not any(base in content for base in base_classes):
                        if 'Interface' in class_name or 'Manager' in class_name or 'Factory' in class_name:
                            results['missing_base_classes'].append({
                                'file': str(file_path),
                                'class': class_name,
                                'suggested_bases': self._suggest_base_class(class_name)
                            })

                # 检查命名一致性
                naming_issues = self._check_naming_consistency(content, str(file_path))
                results['inconsistent_naming'].extend(naming_issues)

            except Exception as e:
                continue

        return results

    def _suggest_base_class(self, class_name: str) -> List[str]:
        """建议基类"""
        suggestions = []
        if 'Factory' in class_name:
            suggestions.append('BaseFactory')
        if 'Manager' in class_name:
            suggestions.append('BaseManager')
        if 'Service' in class_name:
            suggestions.append('BaseService')
        if 'Interface' in class_name:
            suggestions.append('BaseInterface')
        return suggestions

    def _check_naming_consistency(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查命名一致性"""
        issues = []

        # 检查类名是否符合命名规范
        class_defs = re.findall(r'class\s+(\w+)', content)
        for class_name in class_defs:
            if not (class_name[0].isupper() and '_' not in class_name):
                issues.append({
                    'file': file_path,
                    'class': class_name,
                    'issue': '类名不符合驼峰命名规范'
                })

        # 检查方法名
        method_defs = re.findall(r'\s+def\s+(\w+)\s*\(', content)
        for method_name in method_defs:
            if not method_name.startswith('_') and '_' in method_name:
                issues.append({
                    'file': file_path,
                    'method': method_name,
                    'issue': '公共方法名不应使用下划线'
                })

        return issues

    def _check_security_issues(self, files: List[Path]) -> Dict[str, Any]:
        """检查安全问题"""
        results = {
            'hardcoded_secrets': [],
            'sql_injection_risks': [],
            'weak_encryption': [],
            'unsafe_eval': []
        }

        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']*["\']',
                r'secret\s*=\s*["\'][^"\']*["\']',
                r'api_key\s*=\s*["\'][^"\']*["\']'
            ],
            'sql_injection_risks': [
                r'execute\s*\(\s*f?["\']',
                r'cursor\.execute'
            ],
            'unsafe_eval': [
                r'eval\s*\(',
                r'exec\s*\('
            ]
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for issue_type, patterns in security_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            results[issue_type].append({
                                'file': str(file_path),
                                'pattern': pattern,
                                'matches': len(matches)
                            })

            except Exception:
                continue

        return results

    def _check_performance_concerns(self, files: List[Path]) -> Dict[str, Any]:
        """检查性能问题"""
        results = {
            'large_loops': [],
            'memory_inefficient_operations': [],
            'blocking_operations': []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查大循环
                loops = re.findall(r'for\s+.*in\s+.*:', content)
                for loop in loops:
                    # 简单的启发式检查
                    if len(loop) > 100:  # 很长的循环条件
                        results['large_loops'].append({
                            'file': str(file_path),
                            'loop': loop[:100] + '...'
                        })

                # 检查可能的问题操作
                if 'time.sleep(' in content:
                    results['blocking_operations'].append({
                        'file': str(file_path),
                        'operation': 'time.sleep'
                    })

            except Exception:
                continue

        return results

    def _check_maintainability_metrics(self, files: List[Path]) -> Dict[str, Any]:
        """检查可维护性指标"""
        results = {
            'cyclomatic_complexity': [],
            'documentation_coverage': [],
            'code_duplication': [],
            'test_coverage_estimate': []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                code_lines = [line for line in lines if line.strip(
                ) and not line.strip().startswith('#')]

                # 估算圈复杂度（简化）
                complexity_indicators = ['if ', 'elif ', 'for ',
                                         'while ', 'try:', 'except:', 'and ', 'or ']
                complexity_score = sum(content.count(indicator)
                                       for indicator in complexity_indicators)

                if complexity_score > 20:
                    results['cyclomatic_complexity'].append({
                        'file': str(file_path),
                        'complexity_score': complexity_score,
                        'severity': 'high' if complexity_score > 40 else 'medium'
                    })

                # 检查文档覆盖率
                docstring_count = content.count('"""') + content.count("'''")
                function_count = content.count('def ')
                if function_count > 0:
                    doc_coverage = docstring_count / function_count
                    if doc_coverage < 0.5:
                        results['documentation_coverage'].append({
                            'file': str(file_path),
                            'functions': function_count,
                            'docstrings': docstring_count,
                            'coverage': doc_coverage
                        })

            except Exception:
                continue

        return results

    def _check_best_practices(self, files: List[Path]) -> Dict[str, Any]:
        """检查最佳实践"""
        results = {
            'code_style_violations': [],
            'anti_patterns': [],
            'deprecated_usage': [],
            'missing_error_handling': []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查代码风格
                if '\t' in content:  # 使用制表符
                    results['code_style_violations'].append({
                        'file': str(file_path),
                        'issue': '使用制表符而非空格缩进'
                    })

                # 检查反模式
                if 'from module import *' in content:
                    results['anti_patterns'].append({
                        'file': str(file_path),
                        'pattern': '通配符导入',
                        'suggestion': '使用显式导入'
                    })

                # 检查异常处理
                try_blocks = content.count('try:')
                except_blocks = content.count('except')
                if try_blocks > except_blocks:
                    results['missing_error_handling'].append({
                        'file': str(file_path),
                        'try_blocks': try_blocks,
                        'except_blocks': except_blocks
                    })

            except Exception:
                continue

        return results

    def _generate_summary(self, review_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总结"""
        summary = {
            'total_issues': 0,
            'issues_by_category': {},
            'severity_distribution': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'most_problematic_files': []
        }

        # 计算各类问题数量
        for category, results in review_results.items():
            category_issues = 0
            for issue_list in results.values():
                if isinstance(issue_list, list):
                    category_issues += len(issue_list)

            summary['issues_by_category'][category] = category_issues
            summary['total_issues'] += category_issues

        # 确定严重程度分布
        summary['severity_distribution'] = {
            'critical': sum(1 for cat in review_results.values()
                            for issues in cat.values()
                            for issue in (issues if isinstance(issues, list) else [issues])
                            if isinstance(issue, dict) and issue.get('severity') == 'critical'),
            'high': sum(1 for cat in review_results.values()
                        for issues in cat.values()
                        for issue in (issues if isinstance(issues, list) else [issues])
                        if isinstance(issue, dict) and issue.get('severity') in ['high', None]),
            'medium': sum(1 for cat in review_results.values()
                          for issues in cat.values()
                          for issue in (issues if isinstance(issues, list) else [issues])
                          if isinstance(issue, dict) and issue.get('severity') == 'medium'),
            'low': sum(1 for cat in review_results.values()
                       for issues in cat.values()
                       for issue in (issues if isinstance(issues, list) else [issues])
                       if isinstance(issue, dict) and issue.get('severity') == 'low')
        }

        return summary

    def _generate_recommendations(self, review_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议"""
        recommendations = []

        # 基于各类问题生成建议
        if review_results['code_quality']['syntax_errors']:
            recommendations.append({
                'priority': 'critical',
                'category': 'code_quality',
                'title': '修复语法错误',
                'description': f'发现 {len(review_results["code_quality"]["syntax_errors"])} 个语法错误需要立即修复',
                'action_items': ['运行语法检查', '修复所有语法错误', '添加自动化语法检查']
            })

        if review_results['architecture_consistency']['missing_base_classes']:
            recommendations.append({
                'priority': 'high',
                'category': 'architecture',
                'title': '完善基类继承',
                'description': f'{len(review_results["architecture_consistency"]["missing_base_classes"])} 个类缺少适当的基类继承',
                'action_items': ['添加缺失的基类继承', '统一架构模式', '建立继承规范']
            })

        if review_results['security_issues']['hardcoded_secrets']:
            recommendations.append({
                'priority': 'critical',
                'category': 'security',
                'title': '移除硬编码密钥',
                'description': '发现硬编码的敏感信息',
                'action_items': ['使用环境变量', '实现密钥管理', '添加安全审计']
            })

        # 通用改进建议
        recommendations.extend([
            {
                'priority': 'medium',
                'category': 'performance',
                'title': '优化性能瓶颈',
                'description': '识别和优化潜在的性能问题',
                'action_items': ['性能分析', '代码优化', '添加性能监控']
            },
            {
                'priority': 'medium',
                'category': 'maintainability',
                'title': '提升可维护性',
                'description': '改进代码的可维护性和可读性',
                'action_items': ['重构复杂函数', '完善文档', '统一代码风格']
            }
        ])

        return recommendations

    def _calculate_quality_score(self, review_results: Dict[str, Any]) -> float:
        """计算质量分数"""
        base_score = 100.0

        # 各类问题的扣分标准
        penalties = {
            'syntax_errors': 10,  # 每个语法错误扣10分
            'missing_base_classes': 5,  # 每个缺少基类的类扣5分
            'hardcoded_secrets': 20,  # 每个硬编码密钥扣20分
            'complex_functions': 2,  # 每个复杂函数扣2分
            'security_issues': 15,  # 每个安全问题扣15分
        }

        for category, results in review_results.items():
            for issue_type, issues in results.items():
                if isinstance(issues, list):
                    penalty = penalties.get(issue_type, 1)
                    base_score -= len(issues) * penalty

        return max(0.0, base_score)

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_enhanced_summary(self, report: Dict[str, Any]):
        """打印增强审查摘要"""
        summary = report['summary']

        print('\\n🔍 增强代码审查摘要:')
        print('-' * 40)
        print(f'📊 总问题数: {summary["total_issues"]}')
        print(f'🏆 质量分数: {report["quality_score"]:.1f}/100')

        print('\\n📈 问题分类:')
        for category, count in summary['issues_by_category'].items():
            print(f'   {category}: {count}')

        print('\\n⚠️ 严重程度分布:')
        for severity, count in summary['severity_distribution'].items():
            print(f'   {severity}: {count}')

        print('\\n💡 关键建议:')
        for rec in report['recommendations'][:5]:  # 显示前5条
            print(f'   {rec["priority"].upper()}: {rec["title"]}')

        print('\\n📄 详细报告已保存: enhanced_code_review_report.json')


def main():
    """主函数"""
    reviewer = EnhancedCodeReviewer()
    report = reviewer.perform_enhanced_review()


if __name__ == "__main__":
    main()
