"""
代码复杂度分析器

分析代码的复杂度指标，为重构提供依据。
"""

import ast
from typing import Dict, Any

from ..core.code_fragment import CodeFragment


class ComplexityAnalyzer:
    """
    代码复杂度分析器

    计算各种复杂度指标：
    - 圈复杂度 (Cyclomatic Complexity)
    - 认知复杂度 (Cognitive Complexity)
    - 嵌套深度
    - 方法长度
    - 参数数量
    """

    def analyze_fragment(self, fragment: CodeFragment) -> Dict[str, Any]:
        """
        分析代码片段的复杂度

        Args:
            fragment: 代码片段

        Returns:
            Dict[str, Any]: 复杂度分析结果
        """
        if not fragment.ast_node:
            return self._empty_result()

        # 计算各种复杂度指标
        result = {
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(fragment.ast_node),
            'cognitive_complexity': self._calculate_cognitive_complexity(fragment.ast_node),
            'nesting_depth': self._calculate_nesting_depth(fragment.ast_node),
            'line_count': len(fragment.raw_content.split('\n')),
            'parameter_count': self._count_parameters(fragment.ast_node),
            'branch_count': self._count_branches(fragment.ast_node),
            'complexity_score': 0.0,  # 综合复杂度评分
        }

        # 计算综合复杂度评分
        result['complexity_score'] = self._calculate_complexity_score(result)

        # 复杂度等级判断
        result['complexity_level'] = self._get_complexity_level(result['complexity_score'])

        return result

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        计算圈复杂度

        基于McCabe的圈复杂度算法，计算控制流复杂度。
        """
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # 布尔操作增加复杂度
                complexity += len(child.op) if hasattr(child, 'op') else 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1

        return complexity

    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """
        计算认知复杂度

        基于SonarSource的认知复杂度算法，更符合人类的理解难度。
        """
        complexity = 0

        def visit_node(node: ast.AST, nesting: int = 0) -> None:
            nonlocal complexity

            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + nesting  # 嵌套增加复杂度
                nesting += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.op) if hasattr(node, 'op') else 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1

            for child in ast.iter_child_nodes(node):
                visit_node(child, nesting)

        visit_node(node)
        return complexity

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """
        计算最大嵌套深度
        """
        max_depth = 0

        def visit_node(node: ast.AST, depth: int = 0) -> None:
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                 ast.Try, ast.With, ast.AsyncWith)):
                depth += 1

            for child in ast.iter_child_nodes(node):
                visit_node(child, depth)

        visit_node(node)
        return max_depth

    def _count_parameters(self, node: ast.AST) -> int:
        """
        统计参数数量
        """
        if not node:
            return 0

        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = child.args
                if not args:
                    return 0

                param_count = len(args.args or [])
                param_count += len(args.vararg or [])
                param_count += len(args.kwarg or [])
                if args.defaults:
                    param_count += len(args.defaults)
                return param_count
        return 0

    def _count_branches(self, node: ast.AST) -> int:
        """
        统计分支数量
        """
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                count += 1
        return count

    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """
        计算综合复杂度评分

        基于多个指标计算总体复杂度评分。
        """
        # 归一化各项指标
        cc_norm = min(metrics['cyclomatic_complexity'] / 10.0, 1.0)
        cog_norm = min(metrics['cognitive_complexity'] / 15.0, 1.0)
        nest_norm = min(metrics['nesting_depth'] / 5.0, 1.0)
        line_norm = min(metrics['line_count'] / 50.0, 1.0)
        param_norm = min(metrics['parameter_count'] / 5.0, 1.0)

        # 加权计算综合评分
        score = (cc_norm * 0.3 +
                 cog_norm * 0.3 +
                 nest_norm * 0.2 +
                 line_norm * 0.1 +
                 param_norm * 0.1)

        return score

    def _get_complexity_level(self, score: float) -> str:
        """
        根据复杂度评分确定等级
        """
        if score < 0.2:
            return "low"
        elif score < 0.4:
            return "medium"
        elif score < 0.6:
            return "high"
        else:
            return "very_high"

    def _empty_result(self) -> Dict[str, Any]:
        """
        返回空的分析结果
        """
        return {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'nesting_depth': 0,
            'line_count': 0,
            'parameter_count': 0,
            'branch_count': 0,
            'complexity_score': 0.0,
            'complexity_level': 'unknown'
        }
