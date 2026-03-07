#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码质量检查和静态分析测试
测试RQA2025系统的代码质量、静态分析和质量指标
验证代码质量标准和最佳实践的遵循情况
"""

import pytest
import unittest
import ast
import inspect
import os
import re
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TestCodeQualityAnalysis(unittest.TestCase):
    """代码质量检查和静态分析测试"""

    def setUp(self):
        """测试前准备"""
        import time
        self.test_start_time = time.time()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.source_dirs = [
            'src',
            'tests'
        ]
        self.quality_metrics = {}
        self.analysis_results = {}

    def test_code_complexity_analysis(self):
        """测试代码复杂度分析"""
        logger.info("开始测试代码复杂度分析")

        complexity_results = {}
        total_files = 0
        total_functions = 0
        high_complexity_functions = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                files_results = self._analyze_directory_complexity(dir_path)
                complexity_results[source_dir] = files_results

                # 统计指标
                for file_result in files_results.values():
                    total_files += 1
                    for func_result in file_result.get('functions', []):
                        total_functions += 1
                        if func_result.get('complexity', 0) > 10:
                            high_complexity_functions += 1

        # 计算复杂度指标
        avg_complexity_per_file = sum(
            sum(func['complexity'] for func in file_result.get('functions', []))
            for dir_results in complexity_results.values()
            for file_result in dir_results.values()
        ) / max(total_functions, 1)

        complexity_metrics = {
            'total_files_analyzed': total_files,
            'total_functions_analyzed': total_functions,
            'high_complexity_functions': high_complexity_functions,
            'avg_complexity_per_file': avg_complexity_per_file,
            'complexity_distribution': {
                'low': sum(1 for dir_results in complexity_results.values()
                          for file_result in dir_results.values()
                          for func in file_result.get('functions', [])
                          if func['complexity'] <= 5),
                'medium': sum(1 for dir_results in complexity_results.values()
                             for file_result in dir_results.values()
                             for func in file_result.get('functions', [])
                             if 6 <= func['complexity'] <= 10),
                'high': sum(1 for dir_results in complexity_results.values()
                           for file_result in dir_results.values()
                           for func in file_result.get('functions', [])
                           if 11 <= func['complexity'] <= 20),
                'very_high': sum(1 for dir_results in complexity_results.values()
                                for file_result in dir_results.values()
                                for func in file_result.get('functions', [])
                                if func['complexity'] > 20)
            }
        }

        self.quality_metrics['complexity'] = complexity_metrics

        # 验证复杂度标准
        self.assertLess(complexity_metrics['avg_complexity_per_file'], 15.0, "平均函数复杂度过高")
        self.assertLess(complexity_metrics['high_complexity_functions'] / max(total_functions, 1), 0.1,
                       "高复杂度函数占比过高")

        logger.info(f"代码复杂度分析完成，平均复杂度: {complexity_metrics['avg_complexity_per_file']:.2f}")

    def test_code_quality_metrics(self):
        """测试代码质量指标"""
        logger.info("开始测试代码质量指标")

        quality_results = {}
        total_lines = 0
        total_functions = 0
        total_classes = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                dir_quality = self._analyze_directory_quality(dir_path)
                quality_results[source_dir] = dir_quality

                # 累积统计
                for file_quality in dir_quality.values():
                    total_lines += file_quality.get('lines_of_code', 0)
                    total_functions += file_quality.get('function_count', 0)
                    total_classes += file_quality.get('class_count', 0)

        # 计算质量指标
        quality_metrics = {
            'total_lines_of_code': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'avg_lines_per_function': total_lines / max(total_functions, 1),
            'avg_functions_per_class': total_functions / max(total_classes, 1),
            'quality_distribution': {
                'excellent': sum(1 for dir_quality in quality_results.values()
                               for file_quality in dir_quality.values()
                               if file_quality.get('quality_score', 0) >= 9.0),
                'good': sum(1 for dir_quality in quality_results.values()
                           for file_quality in dir_quality.values()
                           if 7.0 <= file_quality.get('quality_score', 0) < 9.0),
                'fair': sum(1 for dir_quality in quality_results.values()
                           for file_quality in dir_quality.values()
                           if 5.0 <= file_quality.get('quality_score', 0) < 7.0),
                'poor': sum(1 for dir_quality in quality_results.values()
                           for file_quality in dir_quality.values()
                           if file_quality.get('quality_score', 0) < 5.0)
            }
        }

        self.quality_metrics['code_quality'] = quality_metrics

        # 验证质量标准
        self.assertGreater(quality_metrics['quality_distribution']['excellent'] +
                          quality_metrics['quality_distribution']['good'], 0.8,
                          "高质量代码占比不足80%")
        self.assertLess(quality_metrics['avg_lines_per_function'], 50, "平均函数行数过长")

        logger.info(f"代码质量指标分析完成，高质量代码占比: {(quality_metrics['quality_distribution']['excellent'] + quality_metrics['quality_distribution']['good']) / max(sum(quality_metrics['quality_distribution'].values()), 1):.2%}")

    def test_static_analysis_issues(self):
        """测试静态分析问题"""
        logger.info("开始测试静态分析问题")

        static_issues = {}
        total_issues = 0
        critical_issues = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                dir_issues = self._analyze_directory_static_issues(dir_path)
                static_issues[source_dir] = dir_issues

                # 统计问题
                for file_issues in dir_issues.values():
                    for issue in file_issues:
                        total_issues += 1
                        if issue.get('severity') == 'critical':
                            critical_issues += 1

        # 计算问题统计
        issue_metrics = {
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'issues_by_type': {},
            'issues_by_severity': {
                'critical': critical_issues,
                'high': sum(1 for dir_issues in static_issues.values()
                           for file_issues in dir_issues.values()
                           for issue in file_issues
                           if issue.get('severity') == 'high'),
                'medium': sum(1 for dir_issues in static_issues.values()
                             for file_issues in dir_issues.values()
                             for issue in file_issues
                             if issue.get('severity') == 'medium'),
                'low': sum(1 for dir_issues in static_issues.values()
                          for file_issues in dir_issues.values()
                          for issue in file_issues
                          if issue.get('severity') == 'low')
            }
        }

        # 按类型统计问题
        for dir_issues in static_issues.values():
            for file_issues in dir_issues.values():
                for issue in file_issues:
                    issue_type = issue.get('type', 'unknown')
                    issue_metrics['issues_by_type'][issue_type] = issue_metrics['issues_by_type'].get(issue_type, 0) + 1

        self.quality_metrics['static_analysis'] = issue_metrics

        # 验证静态分析标准
        self.assertEqual(issue_metrics['critical_issues'], 0, "存在严重静态分析问题")
        self.assertLess(issue_metrics['issues_by_severity']['high'], 10, "高严重性问题过多")

        logger.info(f"静态分析问题检查完成，严重问题: {issue_metrics['critical_issues']}个")

    def test_code_style_compliance(self):
        """测试代码风格符合性"""
        logger.info("开始测试代码风格符合性")

        style_results = {}
        total_files = 0
        compliant_files = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                dir_style = self._analyze_directory_style(dir_path)
                style_results[source_dir] = dir_style

                # 统计符合性
                for file_style in dir_style.values():
                    total_files += 1
                    if file_style.get('style_compliant', False):
                        compliant_files += 1

        # 计算风格指标
        style_metrics = {
            'total_files_checked': total_files,
            'compliant_files': compliant_files,
            'compliance_rate': compliant_files / max(total_files, 1),
            'style_violations': {
                'naming': sum(1 for dir_style in style_results.values()
                             for file_style in dir_style.values()
                             for violation in file_style.get('violations', [])
                             if 'naming' in violation.lower()),
                'formatting': sum(1 for dir_style in style_results.values()
                                for file_style in dir_style.values()
                                for violation in file_style.get('violations', [])
                                if 'format' in violation.lower()),
                'imports': sum(1 for dir_style in style_results.values()
                              for file_style in dir_style.values()
                              for violation in file_style.get('violations', [])
                              if 'import' in violation.lower()),
                'documentation': sum(1 for dir_style in style_results.values()
                                   for file_style in dir_style.values()
                                   for violation in file_style.get('violations', [])
                                   if 'doc' in violation.lower())
            }
        }

        self.quality_metrics['code_style'] = style_metrics

        # 验证风格标准
        self.assertGreater(style_metrics['compliance_rate'], 0.9, "代码风格符合率不足90%")
        self.assertLess(sum(style_metrics['style_violations'].values()), 50, "代码风格违规过多")

        logger.info(f"代码风格符合性检查完成，符合率: {style_metrics['compliance_rate']:.2%}")

    def test_documentation_coverage(self):
        """测试文档覆盖率"""
        logger.info("开始测试文档覆盖率")

        documentation_results = {}
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                dir_docs = self._analyze_directory_documentation(dir_path)
                documentation_results[source_dir] = dir_docs

                # 统计文档覆盖
                for file_docs in dir_docs.values():
                    total_functions += file_docs.get('total_functions', 0)
                    documented_functions += file_docs.get('documented_functions', 0)
                    total_classes += file_docs.get('total_classes', 0)
                    documented_classes += file_docs.get('documented_classes', 0)

        # 计算文档指标
        documentation_metrics = {
            'function_documentation_coverage': documented_functions / max(total_functions, 1),
            'class_documentation_coverage': documented_classes / max(total_classes, 1),
            'overall_documentation_coverage': (documented_functions + documented_classes) / max(total_functions + total_classes, 1),
            'documentation_quality': {
                'excellent': sum(1 for dir_docs in documentation_results.values()
                               for file_docs in dir_docs.values()
                               if file_docs.get('avg_doc_quality', 0) >= 8.0),
                'good': sum(1 for dir_docs in documentation_results.values()
                           for file_docs in dir_docs.values()
                           if 6.0 <= file_docs.get('avg_doc_quality', 0) < 8.0),
                'fair': sum(1 for dir_docs in documentation_results.values()
                           for file_docs in dir_docs.values()
                           if 4.0 <= file_docs.get('avg_doc_quality', 0) < 6.0),
                'poor': sum(1 for dir_docs in documentation_results.values()
                           for file_docs in dir_docs.values()
                           if file_docs.get('avg_doc_quality', 0) < 4.0)
            }
        }

        self.quality_metrics['documentation'] = documentation_metrics

        # 验证文档标准
        self.assertGreater(documentation_metrics['overall_documentation_coverage'], 0.8,
                          "文档覆盖率不足80%")
        self.assertGreater(documentation_metrics['function_documentation_coverage'], 0.85,
                          "函数文档覆盖率不足85%")

        logger.info(f"文档覆盖率分析完成，整体覆盖率: {documentation_metrics['overall_documentation_coverage']:.2%}")

    def test_code_maintainability_index(self):
        """测试代码可维护性指数"""
        logger.info("开始测试代码可维护性指数")

        maintainability_results = {}
        total_files = 0

        for source_dir in self.source_dirs:
            dir_path = os.path.join(self.project_root, source_dir)
            if os.path.exists(dir_path):
                dir_maintainability = self._analyze_directory_maintainability(dir_path)
                maintainability_results[source_dir] = dir_maintainability

        # 计算可维护性指标
        all_scores = []
        for dir_results in maintainability_results.values():
            for file_result in dir_results.values():
                if 'maintainability_index' in file_result:
                    all_scores.append(file_result['maintainability_index'])
                    total_files += 1

        maintainability_metrics = {
            'avg_maintainability_index': sum(all_scores) / max(len(all_scores), 1),
            'maintainability_distribution': {
                'excellent': sum(1 for score in all_scores if score >= 85),
                'good': sum(1 for score in all_scores if 65 <= score < 85),
                'fair': sum(1 for score in all_scores if 45 <= score < 65),
                'poor': sum(1 for score in all_scores if score < 45)
            },
            'files_analyzed': total_files
        }

        self.quality_metrics['maintainability'] = maintainability_metrics

        # 验证可维护性标准
        self.assertGreater(maintainability_metrics['avg_maintainability_index'], 60,
                          "平均可维护性指数过低")
        excellent_plus_good = (maintainability_metrics['maintainability_distribution']['excellent'] +
                              maintainability_metrics['maintainability_distribution']['good'])
        self.assertGreater(excellent_plus_good / max(total_files, 1), 0.7,
                          "高质量可维护性文件占比不足70%")

        logger.info(f"代码可维护性指数分析完成，平均指数: {maintainability_metrics['avg_maintainability_index']:.1f}")

    def tearDown(self):
        """测试后清理"""
        # 输出代码质量分析总结
        import time
        test_duration = time.time() - self.test_start_time

        logger.info("=" * 70)
        logger.info("代码质量分析总结")
        logger.info("=" * 70)
        logger.info(f"总分析时间: {test_duration:.2f}秒")
        logger.info(f"质量指标数量: {len(self.quality_metrics)}")

        # 输出关键质量指标
        for metric_name, metrics in self.quality_metrics.items():
            logger.info(f"\n• {metric_name.upper()}:")
            if metric_name == 'complexity':
                logger.info(f"  平均复杂度: {metrics.get('avg_complexity_per_file', 0):.2f}")
                logger.info(f"  高复杂度函数: {metrics.get('high_complexity_functions', 0)}")
            elif metric_name == 'code_quality':
                logger.info(f"  总代码行数: {metrics.get('total_lines_of_code', 0)}")
                logger.info(f"  高质量文件占比: {(metrics.get('quality_distribution', {}).get('excellent', 0) + metrics.get('quality_distribution', {}).get('good', 0)) / max(sum(metrics.get('quality_distribution', {}).values()), 1):.2%}")
            elif metric_name == 'static_analysis':
                logger.info(f"  总问题数: {metrics.get('total_issues', 0)}")
                logger.info(f"  严重问题: {metrics.get('critical_issues', 0)}")
            elif metric_name == 'documentation':
                logger.info(f"  文档覆盖率: {metrics.get('overall_documentation_coverage', 0):.2%}")

        logger.info("=" * 70)

    def _analyze_directory_complexity(self, dir_path: str) -> Dict[str, Any]:
        """分析目录代码复杂度"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 使用AST分析复杂度
                        tree = ast.parse(content)

                        functions = []
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                complexity = self._calculate_function_complexity(node)
                                functions.append({
                                    'name': node.name,
                                    'complexity': complexity,
                                    'line': node.lineno,
                                    'type': 'function'
                                })
                            elif isinstance(node, ast.ClassDef):
                                # 简化类复杂度计算
                                class_complexity = len([n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                                functions.append({
                                    'name': node.name,
                                    'complexity': class_complexity,
                                    'line': node.lineno,
                                    'type': 'class'
                                })

                        results[file] = {
                            'file_path': file_path,
                            'functions': functions,
                            'avg_complexity': sum(f['complexity'] for f in functions) / max(len(functions), 1),
                            'max_complexity': max([f['complexity'] for f in functions]) if functions else 0
                        }

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = {'error': str(e)}

        return results

    def _analyze_directory_quality(self, dir_path: str) -> Dict[str, Any]:
        """分析目录代码质量"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')
                        tree = ast.parse(content)

                        # 统计基本指标
                        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

                        # 计算质量评分
                        quality_score = self._calculate_quality_score(lines_of_code, function_count, class_count)

                        results[file] = {
                            'file_path': file_path,
                            'lines_of_code': lines_of_code,
                            'function_count': function_count,
                            'class_count': class_count,
                            'quality_score': quality_score
                        }

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = {'error': str(e)}

        return results

    def _analyze_directory_static_issues(self, dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """分析目录静态问题"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        issues = self._detect_static_issues(content, file_path)
                        results[file] = issues

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = [{'type': 'parse_error', 'severity': 'high', 'message': str(e)}]

        return results

    def _analyze_directory_style(self, dir_path: str) -> Dict[str, Any]:
        """分析目录代码风格"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        violations = self._check_style_violations(content, file_path)
                        compliant = len(violations) == 0

                        results[file] = {
                            'file_path': file_path,
                            'style_compliant': compliant,
                            'violations': violations,
                            'violation_count': len(violations)
                        }

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = {'error': str(e)}

        return results

    def _analyze_directory_documentation(self, dir_path: str) -> Dict[str, Any]:
        """分析目录文档"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        tree = ast.parse(content)

                        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

                        documented_functions = sum(1 for func in functions if ast.get_docstring(func))
                        documented_classes = sum(1 for cls in classes if ast.get_docstring(cls))

                        # 计算文档质量
                        doc_qualities = []
                        for func in functions:
                            if ast.get_docstring(func):
                                quality = self._assess_docstring_quality(ast.get_docstring(func))
                                doc_qualities.append(quality)

                        avg_doc_quality = sum(doc_qualities) / max(len(doc_qualities), 1)

                        results[file] = {
                            'file_path': file_path,
                            'total_functions': len(functions),
                            'documented_functions': documented_functions,
                            'total_classes': len(classes),
                            'documented_classes': documented_classes,
                            'avg_doc_quality': avg_doc_quality
                        }

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = {'error': str(e)}

        return results

    def _analyze_directory_maintainability(self, dir_path: str) -> Dict[str, Any]:
        """分析目录可维护性"""
        results = {}

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 计算可维护性指数（简化版本）
                        tree = ast.parse(content)
                        lines_of_code = len([line for line in content.split('\n') if line.strip()])

                        # 计算平均复杂度
                        functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        complexities = [self._calculate_function_complexity(func) for func in functions]
                        avg_complexity = sum(complexities) / max(len(complexities), 1)

                        # 计算可维护性指数（基于行业标准公式）
                        maintainability_index = max(0, min(100,
                            171 - 5.2 * avg_complexity - 0.23 * lines_of_code + 16.2 * 0.01  # 假设10%的注释
                        ))

                        results[file] = {
                            'file_path': file_path,
                            'maintainability_index': maintainability_index,
                            'lines_of_code': lines_of_code,
                            'avg_complexity': avg_complexity
                        }

                    except Exception as e:
                        logger.warning(f"分析文件 {file_path} 时出错: {e}")
                        results[file] = {'error': str(e)}

        return results

    def _calculate_function_complexity(self, func_node) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Try):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1

        return complexity

    def _calculate_quality_score(self, lines_of_code: int, function_count: int, class_count: int) -> float:
        """计算质量评分"""
        # 基于代码行数、函数数、类数计算质量评分
        lines_per_function = lines_of_code / max(function_count, 1)
        functions_per_class = function_count / max(class_count, 1)

        # 理想范围
        ideal_lines_per_function = 20
        ideal_functions_per_class = 5

        # 计算偏差
        lines_deviation = abs(lines_per_function - ideal_lines_per_function) / ideal_lines_per_function
        functions_deviation = abs(functions_per_class - ideal_functions_per_class) / ideal_functions_per_class

        # 计算质量评分 (0-10)
        quality_score = 10 * (1 - (lines_deviation + functions_deviation) / 2)
        return max(0, min(10, quality_score))

    def _detect_static_issues(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检测静态问题"""
        issues = []

        lines = content.split('\n')

        # 检查常见的静态问题
        for i, line in enumerate(lines, 1):
            # 检查过长的行
            if len(line) > 120:
                issues.append({
                    'type': 'line_too_long',
                    'severity': 'low',
                    'line': i,
                    'message': f'行长度过长: {len(line)} > 120'
                })

            # 检查TODO注释
            if 'TODO' in line.upper():
                issues.append({
                    'type': 'todo_comment',
                    'severity': 'low',
                    'line': i,
                    'message': '发现TODO注释，需要处理'
                })

            # 检查print语句（在生产代码中）
            if 'print(' in line and not file_path.endswith('test_'):
                issues.append({
                    'type': 'print_statement',
                    'severity': 'medium',
                    'line': i,
                    'message': '生产代码中发现print语句'
                })

        return issues

    def _check_style_violations(self, content: str, file_path: str) -> List[str]:
        """检查风格违规"""
        violations = []

        # 检查命名约定
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    violations.append(f"函数名不符合命名约定: {node.name}")
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    violations.append(f"类名不符合命名约定: {node.name}")
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store) and node.id.isupper():
                    violations.append(f"常量名不符合约定: {node.id}")

        return violations

    def _assess_docstring_quality(self, docstring: str) -> float:
        """评估文档字符串质量"""
        if not docstring:
            return 0

        score = 5  # 基础分数

        # 检查是否有描述
        if len(docstring.strip()) > 10:
            score += 2

        # 检查是否有参数描述
        if 'Args:' in docstring or 'Parameters:' in docstring:
            score += 1.5

        # 检查是否有返回值描述
        if 'Returns:' in docstring or 'Return:' in docstring:
            score += 1.5

        # 检查是否有异常描述
        if 'Raises:' in docstring or 'Exceptions:' in docstring:
            score += 1

        return min(10, score)


if __name__ == "__main__":
    unittest.main()
