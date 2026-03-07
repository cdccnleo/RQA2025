"""
代码片段表示和分析

定义代码片段的数据结构，支持AST分析和标准化。
"""

import ast
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import re


class FragmentType(Enum):
    """代码片段类型"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    BLOCK = "block"
    STATEMENT = "statement"
    EXPRESSION = "expression"


@dataclass
class CodeFragment:
    """
    代码片段表示

    包含原始代码、AST信息、标准化版本等。
    """

    file_path: str
    start_line: int
    end_line: int
    raw_content: str
    fragment_type: FragmentType

    # AST相关信息
    ast_node: Optional[ast.AST] = None
    parent_fragment: Optional['CodeFragment'] = None
    child_fragments: List['CodeFragment'] = field(default_factory=list)

    # 标准化和分析结果
    normalized_content: str = ""
    ast_hash: str = ""
    semantic_hash: str = ""
    complexity_score: float = 0.0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        self._analyze_fragment()

    def _analyze_fragment(self) -> None:
        """分析代码片段"""
        try:
            # 标准化内容
            self.normalized_content = self._normalize_code(self.raw_content)

            # 计算哈希值
            self.ast_hash = self._calculate_ast_hash()
            self.semantic_hash = self._calculate_semantic_hash()

            # 计算复杂度
            self.complexity_score = self._calculate_complexity()

            # 提取元数据
            self._extract_metadata()

        except Exception as e:
            self.metadata['analysis_error'] = str(e)

    def _normalize_code(self, code: str) -> str:
        """
        标准化代码用于比较

        移除注释、空行、标准化缩进、变量名标准化等。
        """
        lines = []

        for line in code.split('\n'):
            # 移除注释
            line = re.sub(r'#.*$', '', line)
            # 移除前后空白
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 标准化缩进（移除前导缩进）
            line = line.lstrip()

            # 变量名标准化（简单的启发式方法）
            line = self._standardize_variables(line)

            lines.append(line)

        return '\n'.join(lines)

    def _standardize_variables(self, line: str) -> str:
        """
        变量名标准化

        将变量名替换为占位符，基于变量类型和使用模式。
        """
        # 简单的变量名替换策略
        # 这是一个基础实现，更复杂的实现需要AST分析

        # 替换常见的变量名模式
        patterns = [
            (r'\bself\b', 'VAR_SELF'),
            (r'\bcls\b', 'VAR_CLS'),
            (r'\b[a-z]+_[a-z_]*\b', 'VAR_LOCAL'),  # 蛇形命名本地变量
            (r'\b[A-Z][a-zA-Z0-9_]*\b', 'VAR_CLASS'),  # 类名
            (r'\b_[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR_PRIVATE'),  # 私有变量
        ]

        result = line
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        return result

    def _calculate_ast_hash(self) -> str:
        """基于AST结构的哈希值"""
        try:
            if self.ast_node:
                # 将AST转换为标准化字符串
                ast_str = ast.dump(self.ast_node, annotate_fields=False)
                return hashlib.md5(ast_str.encode('utf-8')).hexdigest()
            else:
                # 尝试解析代码为AST
                tree = ast.parse(self.raw_content)
                ast_str = ast.dump(tree, annotate_fields=False)
                return hashlib.md5(ast_str.encode('utf-8')).hexdigest()
        except:
            # 如果AST解析失败，返回内容哈希
            return hashlib.md5(self.normalized_content.encode('utf-8')).hexdigest()

    def _calculate_semantic_hash(self) -> str:
        """基于语义的哈希值"""
        # 结合多个特征计算语义哈希
        features = [
            self.normalized_content,
            str(self.fragment_type.value),
            str(self.complexity_score),
            str(len(self.raw_content.split('\n'))),
        ]

        combined = '|'.join(features)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _calculate_complexity(self) -> float:
        """计算代码复杂度"""
        try:
            if self.ast_node:
                return self._calculate_ast_complexity(self.ast_node)
            else:
                # 基于文本的复杂度估算
                lines = [line for line in self.raw_content.split('\n') if line.strip()]
                return len(lines) * 0.1  # 简单估算
        except:
            return 0.0

    def _calculate_ast_complexity(self, node: ast.AST) -> float:
        """基于AST的复杂度计算"""
        complexity = 0.0

        # 递归遍历AST节点
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1.0
            elif isinstance(child, ast.FunctionDef):
                complexity += 0.5
            elif isinstance(child, ast.Call):
                complexity += 0.2

        return complexity

    def _extract_metadata(self) -> None:
        """提取元数据"""
        self.metadata.update({
            'line_count': self.end_line - self.start_line + 1,
            'char_count': len(self.raw_content),
            'normalized_line_count': len(self.normalized_content.split('\n')),
            'has_comments': '#' in self.raw_content,
            'has_docstring': '"""' in self.raw_content or "'''" in self.raw_content,
            'fragment_type': self.fragment_type.value,
            'complexity_score': self.complexity_score,
        })

    def get_similarity_score(self, other: 'CodeFragment') -> float:
        """
        计算与另一个片段的相似度

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        if not isinstance(other, CodeFragment):
            return 0.0

        # 多维度相似度计算
        scores = []

        # AST结构相似度
        if self.ast_hash and other.ast_hash:
            scores.append(1.0 if self.ast_hash == other.ast_hash else 0.0)

        # 语义相似度
        if self.semantic_hash and other.semantic_hash:
            scores.append(1.0 if self.semantic_hash == other.semantic_hash else 0.0)

        # 标准化内容相似度
        if self.normalized_content and other.normalized_content:
            import difflib
            matcher = difflib.SequenceMatcher(None,
                                              self.normalized_content.split(),
                                              other.normalized_content.split())
            scores.append(matcher.ratio())

        # 返回平均相似度
        return sum(scores) / len(scores) if scores else 0.0

    def __len__(self) -> int:
        """返回代码行数"""
        return self.end_line - self.start_line + 1

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line} ({self.fragment_type.value})"

    def __repr__(self) -> str:
        return f"CodeFragment({self.file_path}:{self.start_line}-{self.end_line}, type={self.fragment_type.value})"
