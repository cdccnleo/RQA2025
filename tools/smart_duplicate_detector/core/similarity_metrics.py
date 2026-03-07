"""
相似度度量算法

提供多种代码相似度计算方法，包括文本、AST结构、语义等。
"""

import ast
import difflib
import tokenize
import io
from typing import Dict, Any, List, Tuple
from collections import Counter
import re
from .code_fragment import CodeFragment


class SimilarityMetrics:
    """
    相似度度量计算器

    支持多种相似度计算算法：
    - 文本相似度
    - AST结构相似度
    - 语义相似度
    - 综合相似度
    """

    @staticmethod
    def text_similarity(code1: str, code2: str) -> float:
        """
        计算文本相似度

        Args:
            code1: 第一段代码
            code2: 第二段代码

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        # 规范化代码
        norm1 = SimilarityMetrics._normalize_for_text(code1)
        norm2 = SimilarityMetrics._normalize_for_text(code2)

        # 使用序列匹配器
        matcher = difflib.SequenceMatcher(None, norm1.split(), norm2.split())
        return matcher.ratio()

    @staticmethod
    def token_similarity(code1: str, code2: str) -> float:
        """
        计算基于token的相似度

        通过分析代码的token序列来计算相似度，比纯文本相似度更准确。

        Args:
            code1: 第一段代码
            code2: 第二段代码

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        try:
            # 提取token序列
            tokens1 = SimilarityMetrics._extract_tokens(code1)
            tokens2 = SimilarityMetrics._extract_tokens(code2)

            if not tokens1 or not tokens2:
                return 0.0

            # 计算token序列相似度
            matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
            return matcher.ratio()

        except Exception:
            # 如果token化失败，回退到文本相似度
            return SimilarityMetrics.text_similarity(code1, code2)

    @staticmethod
    def _extract_tokens(code: str) -> List[str]:
        """
        从代码中提取token序列

        Args:
            code: 源代码

        Returns:
            List[str]: token类型和值的组合列表
        """
        tokens = []
        try:
            # 使用Python的tokenize模块
            code_bytes = code.encode('utf-8')
            token_iter = tokenize.tokenize(io.BytesIO(code_bytes).readline)

            for tok in token_iter:
                if tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING,
                                tokenize.OP, tokenize.ERRORTOKEN):
                    # 只保留有意义的token
                    if tok.string.strip():  # 跳过空白token
                        tokens.append(f"{tok.type}:{tok.string}")
                elif tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER):
                    # 忽略这些结构token
                    continue
                else:
                    # 其他token类型也保留
                    tokens.append(f"{tok.type}:{tok.string}")

        except Exception:
            # 如果token化失败，返回空列表
            pass

        return tokens

    @staticmethod
    def ast_similarity(node1: ast.AST, node2: ast.AST) -> float:
        """
        计算AST结构相似度

        Args:
            node1: 第一个AST节点
            node2: 第二个AST节点

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        try:
            # 将AST转换为字符串表示
            ast_str1 = ast.dump(node1, annotate_fields=False)
            ast_str2 = ast.dump(node2, annotate_fields=False)

            # 计算字符串相似度
            matcher = difflib.SequenceMatcher(None, ast_str1, ast_str2)
            return matcher.ratio()
        except:
            return 0.0

    @staticmethod
    def semantic_similarity(fragment1: CodeFragment, fragment2: CodeFragment) -> float:
        """
        计算语义相似度

        基于变量使用模式、控制结构、函数调用等语义特征。

        Args:
            fragment1: 第一个代码片段
            fragment2: 第二个代码片段

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        features1 = SimilarityMetrics._extract_semantic_features(fragment1)
        features2 = SimilarityMetrics._extract_semantic_features(fragment2)

        return SimilarityMetrics._compare_semantic_features(features1, features2)

    @staticmethod
    def comprehensive_similarity(fragment1: CodeFragment, fragment2: CodeFragment,
                                 thresholds: 'SimilarityThresholds' = None) -> Dict[str, float]:
        """
        综合相似度计算

        返回多种相似度度量的组合结果。

        Args:
            fragment1: 第一个代码片段
            fragment2: 第二个代码片段
            thresholds: 相似度阈值配置

        Returns:
            Dict[str, float]: 各种相似度分数
        """
        results = {}

        # 文本相似度
        results['text'] = SimilarityMetrics.text_similarity(
            fragment1.raw_content, fragment2.raw_content
        )

        # Token相似度（新增）
        results['token'] = SimilarityMetrics.token_similarity(
            fragment1.raw_content, fragment2.raw_content
        )

        # AST相似度
        if fragment1.ast_node and fragment2.ast_node:
            results['ast'] = SimilarityMetrics.ast_similarity(
                fragment1.ast_node, fragment2.ast_node
            )
        else:
            results['ast'] = 0.0

        # 语义相似度
        results['semantic'] = SimilarityMetrics.semantic_similarity(fragment1, fragment2)

        # 标准化内容相似度
        results['normalized'] = SimilarityMetrics.text_similarity(
            fragment1.normalized_content, fragment2.normalized_content
        )

        # 计算综合得分
        if thresholds:
            weights = {
                'text': thresholds.text_weight,
                'token': thresholds.token_weight,
                'ast': thresholds.ast_weight,
                'semantic': thresholds.semantic_weight,
                'normalized': thresholds.normalized_weight
            }
        else:
            # 默认权重
            weights = {
                'text': 0.15,
                'token': 0.25,
                'ast': 0.35,
                'semantic': 0.2,
                'normalized': 0.05
            }

        results['comprehensive'] = sum(
            results[metric] * weight for metric, weight in weights.items()
        )

        return results

    @staticmethod
    def _normalize_for_text(code: str) -> str:
        """文本相似度计算的代码规范化"""
        lines = []
        for line in code.split('\n'):
            # 移除注释
            line = re.sub(r'#.*$', '', line)
            # 移除前后空白
            line = line.strip()
            # 跳过空行
            if line:
                lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def _extract_semantic_features(fragment: CodeFragment) -> Dict[str, Any]:
        """
        提取语义特征

        Args:
            fragment: 代码片段

        Returns:
            Dict[str, Any]: 语义特征字典
        """
        features = {
            'control_structures': [],
            'function_calls': [],
            'variable_usage': [],
            'literals': [],
            'complexity': fragment.complexity_score,
            'line_count': len(fragment),
        }

        if fragment.ast_node:
            SimilarityMetrics._analyze_ast_node(fragment.ast_node, features)

        return features

    @staticmethod
    def _analyze_ast_node(node: ast.AST, features: Dict[str, Any]) -> None:
        """递归分析AST节点提取特征"""
        # 控制结构
        if isinstance(node, ast.If):
            features['control_structures'].append('if')
        elif isinstance(node, ast.For):
            features['control_structures'].append('for')
        elif isinstance(node, ast.While):
            features['control_structures'].append('while')
        elif isinstance(node, ast.Try):
            features['control_structures'].append('try')

        # 函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                features['function_calls'].append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                features['function_calls'].append(f"{node.func.attr}")

        # 变量使用
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                features['variable_usage'].append(('load', node.id))
            elif isinstance(node.ctx, ast.Store):
                features['variable_usage'].append(('store', node.id))

        # 字面量
        if isinstance(node, (ast.Str, ast.Num)):
            features['literals'].append(type(node).__name__)

        # 递归处理子节点
        for child in ast.iter_child_nodes(node):
            SimilarityMetrics._analyze_ast_node(child, features)

    @staticmethod
    def _compare_semantic_features(features1: Dict[str, Any],
                                   features2: Dict[str, Any]) -> float:
        """
        比较语义特征

        Args:
            features1: 第一个片段的特征
            features2: 第二个片段的特征

        Returns:
            float: 相似度分数
        """
        scores = []

        # 比较复杂度（越接近越相似）
        if features1['complexity'] > 0 and features2['complexity'] > 0:
            complexity_ratio = min(features1['complexity'], features2['complexity']) / \
                max(features1['complexity'], features2['complexity'])
            scores.append(complexity_ratio)

        # 比较控制结构
        control_sim = SimilarityMetrics._compare_lists(
            features1['control_structures'],
            features2['control_structures']
        )
        scores.append(control_sim)

        # 比较函数调用
        call_sim = SimilarityMetrics._compare_lists(
            features1['function_calls'],
            features2['function_calls']
        )
        scores.append(call_sim)

        # 比较变量使用模式
        var_sim = SimilarityMetrics._compare_variable_patterns(
            features1['variable_usage'],
            features2['variable_usage']
        )
        scores.append(var_sim)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _compare_lists(list1: List, list2: List) -> float:
        """比较两个列表的相似度"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0

        # 使用Jaccard相似度
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _compare_variable_patterns(patterns1: List[Tuple], patterns2: List[Tuple]) -> float:
        """比较变量使用模式"""
        if not patterns1 and not patterns2:
            return 1.0

        # 统计操作类型
        ops1 = Counter(op for op, _ in patterns1)
        ops2 = Counter(op for op, _ in patterns2)

        # 计算操作类型相似度
        all_ops = set(ops1.keys()) | set(ops2.keys())
        similarity = 0.0

        for op in all_ops:
            count1 = ops1.get(op, 0)
            count2 = ops2.get(op, 0)
            if count1 > 0 and count2 > 0:
                similarity += min(count1, count2) / max(count1, count2)

        return similarity / len(all_ops) if all_ops else 0.0

    @staticmethod
    def token_based_similarity(code1: str, code2: str) -> float:
        """
        基于token的相似度计算

        使用更细粒度的token分析。
        """
        tokens1 = SimilarityMetrics._tokenize(code1)
        tokens2 = SimilarityMetrics._tokenize(code2)

        # 使用序列匹配器计算token级相似度
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        return matcher.ratio()

    @staticmethod
    def _tokenize(code: str) -> List[str]:
        """将代码分解为token"""
        # 简单的token化（生产环境中应该使用更复杂的解析器）
        tokens = []
        current_token = ""

        for char in code:
            if char.isalnum() or char == '_':
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char not in ' \t\n':
                    tokens.append(char)

        if current_token:
            tokens.append(current_token)

        return tokens
