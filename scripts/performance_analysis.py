#!/usr/bin/env python3
"""
RQA2025 基础设施层性能分析脚本

分析基础设施层代码的性能瓶颈，识别热点方法
"""

import os
import sys
import ast
import time
from typing import Dict, List, Any


class PerformanceAnalyzer:
    def __init__(self):
        self.hotspots: List[Dict[str, Any]] = []
        self.complexity_scores: Dict[str, int] = {}

    def analyze_file_complexity(self, filepath: str) -> Dict[str, Any]:
        """分析文件中的方法复杂度"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filepath)
            methods = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_complexity(node)
                    methods.append({
                        'name': node.name,
                        'complexity': complexity,
                        'line': node.lineno,
                        'file': filepath
                    })

            return {
                'file': filepath,
                'methods': methods,
                'total_methods': len(methods),
                'avg_complexity': sum(m['complexity'] for m in methods) / len(methods) if methods else 0,
                'high_complexity_methods': len([m for m in methods if m['complexity'] > 20])
            }

        except Exception as e:
            return {
                'file': filepath,
                'error': str(e),
                'methods': [],
                'total_methods': 0,
                'avg_complexity': 0,
                'high_complexity_methods': 0
            }

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数的圈复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            # 条件语句
            if isinstance(child, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            # 布尔操作
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            # Try语句
            elif isinstance(child, ast.Try):
                complexity += 1
            # Except语句
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity

    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """分析整个目录的性能特征"""
        results = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    result = self.analyze_file_complexity(filepath)
                    results.append(result)

        # 汇总分析
        total_methods = sum(r['total_methods'] for r in results)
        high_complexity_count = sum(r['high_complexity_methods'] for r in results)

        # 找出复杂度最高的10个方法
        all_methods = []
        for result in results:
            all_methods.extend(result['methods'])

        top_complex_methods = sorted(
            [m for m in all_methods if m['complexity'] > 15],
            key=lambda x: x['complexity'],
            reverse=True
        )[:10]

        return {
            'total_files': len(results),
            'total_methods': total_methods,
            'high_complexity_methods': high_complexity_count,
            'avg_complexity_per_file': sum(r['avg_complexity'] for r in results) / len(results) if results else 0,
            'top_complex_methods': top_complex_methods,
            'complexity_distribution': self._analyze_complexity_distribution(all_methods)
        }

    def _analyze_complexity_distribution(self, methods: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析复杂度分布"""
        distribution = {
            '1-5': 0,
            '6-10': 0,
            '11-15': 0,
            '16-20': 0,
            '21-30': 0,
            '30+': 0
        }

        for method in methods:
            complexity = method['complexity']
            if complexity <= 5:
                distribution['1-5'] += 1
            elif complexity <= 10:
                distribution['6-10'] += 1
            elif complexity <= 15:
                distribution['11-15'] += 1
            elif complexity <= 20:
                distribution['16-20'] += 1
            elif complexity <= 30:
                distribution['21-30'] += 1
            else:
                distribution['30+'] += 1

        return distribution

    def identify_performance_bottlenecks(self, directory: str) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        analysis = self.analyze_directory(directory)
        bottlenecks = []

        # 高复杂度方法
        for method in analysis['top_complex_methods']:
            bottlenecks.append({
                'type': 'high_complexity',
                'description': f"方法复杂度过高: {method['complexity']}",
                'file': method['file'],
                'method': method['name'],
                'line': method['line'],
                'severity': 'high' if method['complexity'] > 25 else 'medium',
                'suggested_fix': '拆分方法，提取辅助函数，减少嵌套条件'
            })

        # 文件级别的复杂度问题
        files_with_high_complexity = [
            r for r in [self.analyze_file_complexity(os.path.join(directory, f))
                        for f in os.listdir(directory) if f.endswith('.py')]
            if r['high_complexity_methods'] > 3
        ]

        for file_result in files_with_high_complexity:
            bottlenecks.append({
                'type': 'file_high_complexity',
                'description': f"文件复杂度过高: {file_result['high_complexity_methods']}个高复杂度方法",
                'file': file_result['file'],
                'severity': 'medium',
                'suggested_fix': '重构文件，将相关方法提取到专门的模块中'
            })

        return bottlenecks

    def generate_report(self, directory: str) -> Dict[str, Any]:
        """生成性能分析报告"""
        analysis = self.analyze_directory(directory)
        bottlenecks = self.identify_performance_bottlenecks(directory)

        return {
            'timestamp': time.time(),
            'target_directory': directory,
            'summary': {
                'total_files': analysis['total_files'],
                'total_methods': analysis['total_methods'],
                'high_complexity_methods': analysis['high_complexity_methods'],
                'avg_complexity_per_file': analysis['avg_complexity_per_file']
            },
            'complexity_distribution': analysis['complexity_distribution'],
            'top_complex_methods': analysis['top_complex_methods'],
            'performance_bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(bottlenecks)
        }

    def _generate_recommendations(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        high_complexity_count = len([b for b in bottlenecks if b['type'] == 'high_complexity'])
        file_complexity_count = len([b for b in bottlenecks if b['type'] == 'file_high_complexity'])

        if high_complexity_count > 0:
            recommendations.append(
                f"🔴 高优先级: 重构{high_complexity_count}个高复杂度方法，采用单一职责原则拆分方法"
            )

        if file_complexity_count > 0:
            recommendations.append(
                f"🟡 中优先级: 重构{file_complexity_count}个高复杂度文件，进行模块化拆分"
            )

        recommendations.extend([
            "🔵 低优先级: 实施代码审查，对新增代码进行复杂度检查",
            "🔵 低优先级: 建立性能基准测试，持续监控性能指标",
            "🔵 低优先级: 优化热点路径，减少不必要的计算"
        ])

        return recommendations


def main():
    """主函数"""
    print('⚡ RQA2025 基础设施层性能分析')
    print('=' * 50)

    if len(sys.argv) < 2:
        print("用法: python performance_analysis.py <目录路径>")
        print("示例: python performance_analysis.py src/infrastructure")
        sys.exit(1)

    target_directory = sys.argv[1]

    if not os.path.exists(target_directory):
        print(f"❌ 目录不存在: {target_directory}")
        sys.exit(1)

    analyzer = PerformanceAnalyzer()
    report = analyzer.generate_report(target_directory)

    # 输出摘要
    summary = report['summary']
    print('📊 分析摘要:')
    print(f'   文件总数: {summary["total_files"]}')
    print(f'   方法总数: {summary["total_methods"]}')
    print(f'   高复杂度方法: {summary["high_complexity_methods"]}')
    print(f'   文件平均复杂度: {summary["avg_complexity_per_file"]:.2f}')
    print('🧩 复杂度分布:')
    for range_name, count in report['complexity_distribution'].items():
        print(f'   {range_name}: {count}')

    print(f'\\n🎯 性能瓶颈 ({len(report["performance_bottlenecks"])}个):')
    for i, bottleneck in enumerate(report['performance_bottlenecks'][:5], 1):
        severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(bottleneck['severity'], '❓')
        print(f'   {i}. {severity_icon} {bottleneck["description"]}')
        print(f'      文件: {os.path.basename(bottleneck["file"])}')

    print('\\n💡 优化建议:')
    for rec in report['recommendations']:
        print(f'   {rec}')

    # 保存详细报告
    import json
    report_file = f"performance_analysis_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'\\n✅ 详细报告已保存: {report_file}')


if __name__ == "__main__":
    main()
