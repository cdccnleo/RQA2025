#!/usr/bin/env python3
"""
复杂度治理工具

专门用于识别、分析和重构高复杂度代码的工具。
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ComplexityMetrics:
    """复杂度指标"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    lines_of_code: int
    parameter_count: int
    nested_depth: int
    halstead_volume: float = 0.0


@dataclass
class FunctionComplexity:
    """函数复杂度信息"""
    name: str
    file_path: str
    line_number: int
    metrics: ComplexityMetrics
    risk_level: str  # 'low', 'medium', 'high', 'very_high'
    suggestions: List[str]


class ComplexityAnalyzer:
    """
    复杂度分析器

    分析代码的各种复杂度指标，为重构提供科学依据。
    """

    def __init__(self):
        self.complexity_thresholds = {
            'cyclomatic': 15,  # 圈复杂度阈值
            'cognitive': 10,   # 认知复杂度阈值
            'lines': 50,       # 行数阈值
            'parameters': 5,   # 参数数量阈值
            'nesting': 4       # 嵌套深度阈值
        }

    def analyze_file(self, file_path: Path) -> List[FunctionComplexity]:
        """分析单个文件的复杂度"""
        complexities = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._analyze_function_complexity(node, file_path)
                    if complexity:
                        complexities.append(complexity)

        except Exception as e:
            print(f"⚠️ 分析文件失败 {file_path}: {e}")

        return complexities

    def _analyze_function_complexity(self, node: ast.FunctionDef, file_path: Path) -> Optional[FunctionComplexity]:
        """分析函数复杂度"""
        # 计算各种复杂度指标
        cyclomatic = self._calculate_cyclomatic_complexity(node)
        cognitive = self._calculate_cognitive_complexity(node)
        lines = node.end_lineno - node.lineno + 1
        parameters = len(node.args.args) + len(node.args.kwonlyargs)
        nesting = self._calculate_nesting_depth(node)

        metrics = ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            lines_of_code=lines,
            parameter_count=parameters,
            nested_depth=nesting
        )

        # 确定风险等级
        risk_level = self._assess_risk_level(metrics)

        # 生成建议
        suggestions = self._generate_suggestions(metrics, risk_level)

        return FunctionComplexity(
            name=node.name,
            file_path=str(file_path),
            line_number=node.lineno,
            metrics=metrics,
            risk_level=risk_level,
            suggestions=suggestions
        )

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.Or):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.FunctionDef) -> int:
        """计算认知复杂度"""
        complexity = 1  # 基础复杂度
        nesting_level = 0

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While)):
                if nesting_level > 0:
                    complexity += nesting_level
                complexity += 1
                nesting_level += 1
            elif isinstance(child, ast.Try):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1

        return complexity

    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """计算嵌套深度"""
        max_depth = 0
        current_depth = 0

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(child, ast.FunctionDef):
                # 遇到内嵌函数，重置深度计算
                if child != node:
                    current_depth = 0

        return max_depth

    def _assess_risk_level(self, metrics: ComplexityMetrics) -> str:
        """评估风险等级"""
        risk_score = 0

        # 圈复杂度评分
        if metrics.cyclomatic_complexity > self.complexity_thresholds['cyclomatic']:
            risk_score += 3
        elif metrics.cyclomatic_complexity > 10:
            risk_score += 1

        # 认知复杂度评分
        if metrics.cognitive_complexity > self.complexity_thresholds['cognitive']:
            risk_score += 3
        elif metrics.cognitive_complexity > 7:
            risk_score += 1

        # 行数评分
        if metrics.lines_of_code > self.complexity_thresholds['lines']:
            risk_score += 2
        elif metrics.lines_of_code > 30:
            risk_score += 1

        # 参数数量评分
        if metrics.parameter_count > self.complexity_thresholds['parameters']:
            risk_score += 2

        # 嵌套深度评分
        if metrics.nested_depth > self.complexity_thresholds['nesting']:
            risk_score += 2

        # 根据总分确定风险等级
        if risk_score >= 8:
            return 'very_high'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'

    def _generate_suggestions(self, metrics: ComplexityMetrics, risk_level: str) -> List[str]:
        """生成重构建议"""
        suggestions = []

        if metrics.cyclomatic_complexity > self.complexity_thresholds['cyclomatic']:
            suggestions.append(f"圈复杂度过高 ({metrics.cyclomatic_complexity})，建议拆分为多个函数")

        if metrics.cognitive_complexity > self.complexity_thresholds['cognitive']:
            suggestions.append(f"认知复杂度过高 ({metrics.cognitive_complexity})，建议简化条件逻辑")

        if metrics.lines_of_code > self.complexity_thresholds['lines']:
            suggestions.append(f"函数过长 ({metrics.lines_of_code}行)，建议拆分为多个职责单一的函数")

        if metrics.parameter_count > self.complexity_thresholds['parameters']:
            suggestions.append(f"参数过多 ({metrics.parameter_count}个)，建议使用参数对象")

        if metrics.nested_depth > self.complexity_thresholds['nesting']:
            suggestions.append(f"嵌套过深 ({metrics.nested_depth}层)，建议提取嵌套逻辑或使用早期返回")

        if not suggestions:
            suggestions.append("代码复杂度在可接受范围内")

        return suggestions


class ComplexityGovernor:
    """
    复杂度治理器

    提供复杂度治理的具体操作和重构建议。
    """

    def __init__(self):
        self.analyzer = ComplexityAnalyzer()

    def analyze_project(self, target_path: str) -> Dict[str, Any]:
        """
        分析项目的复杂度情况

        Args:
            target_path: 分析目标路径

        Returns:
            Dict[str, Any]: 复杂度分析报告
        """
        print("🔍 开始复杂度分析...")

        target = Path(target_path)
        all_complexities = []

        # 扫描Python文件
        python_files = []
        if target.is_file():
            if target.suffix == '.py':
                python_files.append(target)
        else:
            for file_path in target.rglob('*.py'):
                if not any(part.startswith('__') or part in {'node_modules', '.git'}
                           for part in file_path.parts):
                    python_files.append(file_path)

        print(f"📊 发现 {len(python_files)} 个Python文件")

        # 分析每个文件
        for file_path in python_files:
            complexities = self.analyzer.analyze_file(file_path)
            all_complexities.extend(complexities)

        print(f"🎯 分析完成，发现 {len(all_complexities)} 个函数")

        # 生成报告
        report = self._generate_complexity_report(all_complexities)

        return report

    def _generate_complexity_report(self, complexities: List[FunctionComplexity]) -> Dict[str, Any]:
        """生成复杂度报告"""
        if not complexities:
            return {"error": "未发现任何函数"}

        # 统计信息
        total_functions = len(complexities)

        # 按风险等级分组
        risk_groups = {
            'very_high': [c for c in complexities if c.risk_level == 'very_high'],
            'high': [c for c in complexities if c.risk_level == 'high'],
            'medium': [c for c in complexities if c.risk_level == 'medium'],
            'low': [c for c in complexities if c.risk_level == 'low']
        }

        # 计算平均复杂度
        avg_cyclomatic = statistics.mean(c.metrics.cyclomatic_complexity for c in complexities)
        avg_cognitive = statistics.mean(c.metrics.cognitive_complexity for c in complexities)
        avg_lines = statistics.mean(c.metrics.lines_of_code for c in complexities)

        # 识别最复杂的函数
        most_complex = max(complexities, key=lambda c: c.metrics.cyclomatic_complexity)

        report = {
            "summary": {
                "total_functions": total_functions,
                "avg_cyclomatic_complexity": round(avg_cyclomatic, 2),
                "avg_cognitive_complexity": round(avg_cognitive, 2),
                "avg_lines_of_code": round(avg_lines, 2),
                "risk_distribution": {level: len(funcs) for level, funcs in risk_groups.items()}
            },
            "high_risk_functions": [
                {
                    "name": f"{c.name} ({Path(c.file_path).name}:{c.line_number})",
                    "cyclomatic": c.metrics.cyclomatic_complexity,
                    "cognitive": c.metrics.cognitive_complexity,
                    "lines": c.metrics.lines_of_code,
                    "risk_level": c.risk_level,
                    "suggestions": c.suggestions[:2]  # 只显示前2个建议
                }
                for c in risk_groups['very_high'] + risk_groups['high']
            ],
            "most_complex_function": {
                "name": f"{most_complex.name} ({Path(most_complex.file_path).name}:{most_complex.line_number})",
                "cyclomatic": most_complex.metrics.cyclomatic_complexity,
                "cognitive": most_complex.metrics.cognitive_complexity,
                "lines": most_complex.metrics.lines_of_code,
                "suggestions": most_complex.suggestions
            },
            "recommendations": self._generate_recommendations(risk_groups)
        }

        return report

    def _generate_recommendations(self, risk_groups: Dict[str, List[FunctionComplexity]]) -> List[str]:
        """生成治理建议"""
        recommendations = []

        very_high_count = len(risk_groups['very_high'])
        high_count = len(risk_groups['high'])

        if very_high_count > 0:
            recommendations.append(f"🚨 紧急处理 {very_high_count} 个极高复杂度函数，这些函数风险很高")
        if high_count > 0:
            recommendations.append(f"⚠️ 优先处理 {high_count} 个高复杂度函数，这些函数需要重构")

        total_high_risk = very_high_count + high_count
        if total_high_risk > 10:
            recommendations.append("📊 建议分阶段处理：第一阶段处理前30%的最高风险函数")
        elif total_high_risk > 5:
            recommendations.append("📅 建议2-3周内完成所有高风险函数的重构")

        # 复杂度分布建议
        total = sum(len(funcs) for funcs in risk_groups.values())
        medium_ratio = len(risk_groups['medium']) / total if total > 0 else 0
        if medium_ratio > 0.3:
            recommendations.append("🔄 中等复杂度函数较多，建议建立代码审查机制防止进一步恶化")

        recommendations.append("🛠️ 推荐使用Extract Method、Replace Conditional with Polymorphism等重构手法")
        recommendations.append("📏 建立复杂度监控机制，定期检查代码质量指标")

        return recommendations


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="复杂度治理工具")
    parser.add_argument('target', help='分析目标路径')
    parser.add_argument('--output', '-o', default='complexity_analysis.json',
                        help='输出文件路径')
    parser.add_argument('--threshold', '-t', type=int, default=15,
                        help='复杂度阈值 (默认: 15)')

    args = parser.parse_args()

    # 创建治理器
    governor = ComplexityGovernor()
    if args.threshold != 15:
        governor.analyzer.complexity_thresholds['cyclomatic'] = args.threshold

    # 执行分析
    report = governor.analyze_project(args.target)

    # 保存报告
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 显示关键结果
    summary = report['summary']
    print("\n🎯 复杂度分析结果:")
    print(f"  • 总函数数: {summary['total_functions']}")
    print(f"  • 平均圈复杂度: {summary['avg_cyclomatic_complexity']}")
    print(f"  • 平均认知复杂度: {summary['avg_cognitive_complexity']}")
    print(".1f")
    risk_dist = summary['risk_distribution']
    print(f"  • 高风险函数: {risk_dist.get('very_high', 0) + risk_dist.get('high', 0)}")
    print(f"  • 中风险函数: {risk_dist.get('medium', 0)}")

    if 'most_complex_function' in report:
        most_complex = report['most_complex_function']
        print("\n🏆 最复杂函数:")
        print(f"  • {most_complex['name']}")
        print(f"  • 圈复杂度: {most_complex['cyclomatic']}")
        print(f"  • 行数: {most_complex['lines']}")

    if report.get('recommendations'):
        print("\n💡 治理建议:")
        for rec in report['recommendations'][:3]:  # 显示前3条建议
            print(f"  • {rec}")

    print(f"\n📄 详细报告已保存到: {args.output}")
    print("🎉 复杂度分析完成！")


if __name__ == '__main__':
    main()
