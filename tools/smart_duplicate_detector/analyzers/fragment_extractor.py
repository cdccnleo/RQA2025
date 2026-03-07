"""
代码片段提取器

从Python文件中提取各种类型的代码片段。
"""

import ast
from typing import List, Optional, Tuple
from ..core.code_fragment import CodeFragment, FragmentType
from ..core.config import SmartDuplicateConfig
from .base_analyzer import BaseAnalyzer


class FragmentExtractor(BaseAnalyzer):
    """
    代码片段提取器

    从Python代码中提取函数、类、方法等代码片段。
    """

    def __init__(self, config: SmartDuplicateConfig):
        super().__init__(config)

    def analyze(self, target_path: str) -> List[CodeFragment]:
        """
        提取代码片段

        Args:
            target_path: 目标路径

        Returns:
            List[CodeFragment]: 代码片段列表
        """
        python_files = self.get_python_files(target_path)
        fragments = []

        for file_path in python_files:
            file_fragments = self._extract_from_file(file_path)
            fragments.extend(file_fragments)

            # 限制每个文件的片段数量
            if len(file_fragments) > self.config.performance.max_fragments_per_file:
                self.logger.warning(f"文件{file_path}片段数量过多({len(file_fragments)})，"
                                    f"限制为{self.config.performance.max_fragments_per_file}个")
                fragments = fragments[:self.config.performance.max_fragments_per_file]

        self.logger.info(f"共提取{len(fragments)}个代码片段")
        return fragments

    def _extract_from_file(self, file_path: str) -> List[CodeFragment]:
        """
        从单个文件中提取片段

        Args:
            file_path: 文件路径

        Returns:
            List[CodeFragment]: 代码片段列表
        """
        content = self.read_file_content(file_path)
        if content is None:
            return []

        tree = self.parse_ast(file_path)
        if tree is None:
            return []

        fragments = []
        lines = content.split('\n')

        # 遍历AST节点提取片段
        for node in ast.walk(tree):
            fragment = self._extract_fragment_from_node(file_path, content, lines, node)
            if fragment:
                fragments.append(fragment)

        return fragments

    def _extract_fragment_from_node(self, file_path: str, content: str,
                                    lines: List[str], node: ast.AST) -> Optional[CodeFragment]:
        """
        从AST节点提取代码片段

        Args:
            file_path: 文件路径
            content: 文件内容
            lines: 文件行列表
            node: AST节点

        Returns:
            Optional[CodeFragment]: 代码片段
        """
        # 根据节点类型确定片段类型
        fragment_info = self._get_fragment_info(node)
        if not fragment_info:
            return None

        fragment_type, start_attr, end_attr = fragment_info

        # 获取行号信息
        start_line = getattr(node, start_attr, None)
        if start_line is None:
            return None

        # 计算结束行
        end_line = self._calculate_end_line(node, lines, start_line, end_attr)

        # 验证片段大小
        line_count = end_line - start_line + 1
        if not (self.config.analysis.min_fragment_size <= line_count <=
                self.config.analysis.max_fragment_size):
            return None

        # 提取代码内容
        fragment_content = '\n'.join(lines[start_line-1:end_line])

        # 过滤不需要的内容
        if self._should_skip_fragment(fragment_content):
            return None

        # 创建代码片段
        fragment = CodeFragment(
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            raw_content=fragment_content,
            fragment_type=fragment_type,
            ast_node=node
        )

        return fragment

    def _get_fragment_info(self, node: ast.AST) -> Optional[Tuple[FragmentType, str, str]]:
        """
        获取节点对应的片段信息

        Args:
            node: AST节点

        Returns:
            Optional[Tuple[FragmentType, str, str]]: (类型, 开始属性, 结束属性)
        """
        if isinstance(node, ast.FunctionDef) and self.config.analysis.extract_functions:
            return (FragmentType.FUNCTION, 'lineno', 'end_lineno')
        elif isinstance(node, ast.AsyncFunctionDef) and self.config.analysis.extract_functions:
            return (FragmentType.FUNCTION, 'lineno', 'end_lineno')
        elif isinstance(node, ast.ClassDef) and self.config.analysis.extract_classes:
            return (FragmentType.CLASS, 'lineno', 'end_lineno')
        elif self._is_method_node(node) and self.config.analysis.extract_methods:
            return (FragmentType.METHOD, 'lineno', 'end_lineno')
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)) and self.config.analysis.extract_blocks:
            return (FragmentType.BLOCK, 'lineno', 'end_lineno')
        elif self.config.analysis.extract_statements:
            # 简单的语句提取（可以扩展）
            return (FragmentType.STATEMENT, 'lineno', 'end_lineno')

        return None

    def _is_method_node(self, node: ast.AST) -> bool:
        """判断是否为方法节点"""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False

        # 检查是否在类中
        parent = getattr(node, '_parent', None)
        while parent:
            if isinstance(parent, ast.ClassDef):
                return True
            parent = getattr(parent, '_parent', None)

        return False

    def _calculate_end_line(self, node: ast.AST, lines: List[str],
                            start_line: int, end_attr: str) -> int:
        """
        计算片段结束行

        Args:
            node: AST节点
            lines: 文件行列表
            start_line: 开始行
            end_attr: 结束行属性名

        Returns:
            int: 结束行号
        """
        # 尝试使用AST提供的end_lineno
        end_line = getattr(node, end_attr, None)
        if end_line is not None:
            return end_line

        # 如果没有end_lineno，手动计算
        end_line = start_line

        # 递归查找所有子节点的行号
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                end_line = max(end_line, child.lineno)

        # 确保不超过文件长度
        return min(end_line, len(lines))

    def _should_skip_fragment(self, content: str) -> bool:
        """
        判断是否应该跳过这个片段

        Args:
            content: 代码内容

        Returns:
            bool: 是否跳过
        """
        # 检查是否只包含import语句
        if self.config.analysis.ignore_imports:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if all(line.startswith('import ') or line.startswith('from ') for line in lines):
                return True

        # 检查是否只包含注释
        if self.config.analysis.ignore_comments:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if all(line.startswith('#') or not line for line in lines):
                return True

        # 检查是否只包含docstring
        if self.config.analysis.ignore_docstrings:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if lines and lines[0].startswith('"""') and lines[-1].endswith('"""'):
                return True

        return False

    def extract_custom_fragments(self, file_path: str,
                                 custom_extractors: List[callable]) -> List[CodeFragment]:
        """
        使用自定义提取器提取片段

        Args:
            file_path: 文件路径
            custom_extractors: 自定义提取器函数列表

        Returns:
            List[CodeFragment]: 代码片段列表
        """
        content = self.read_file_content(file_path)
        if content is None:
            return []

        fragments = []
        for extractor in custom_extractors:
            try:
                custom_fragments = extractor(file_path, content)
                fragments.extend(custom_fragments)
            except Exception as e:
                self.logger.error(f"自定义提取器失败: {e}")

        return fragments
