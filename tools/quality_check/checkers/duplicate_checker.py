"""
代码重复检测检查器

检测代码中的重复模式和相似代码块。
"""

import ast
import hashlib
from typing import Dict, Any, List, Optional
from collections import defaultdict
import difflib
import re

from ..core.base_checker import BaseChecker
from ..core.check_result import IssueSeverity


class CodeBlock:
    """代码块表示"""

    def __init__(self, file_path: str, start_line: int, end_line: int, content: str):
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.content = content
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """计算代码块的哈希值"""
        # 规范化代码（移除注释、空行、缩进差异）
        normalized = self._normalize_code(self.content)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _normalize_code(self, code: str) -> str:
        """规范化代码用于比较"""
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

    def __len__(self) -> int:
        return self.end_line - self.start_line + 1

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class DuplicateCodeChecker(BaseChecker):
    """
    代码重复检测检查器

    使用AST解析和相似度算法检测重复代码。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def _setup_default_config(self) -> None:
        """设置默认配置"""
        defaults = {
            'min_lines': 5,  # 最少行数
            'max_lines': 50,  # 最大行数
            'similarity_threshold': 0.8,  # 相似度阈值
            'duplicate_threshold': 3,  # 重复次数阈值
            'ignore_imports': True,  # 忽略import语句
            'ignore_comments': True,  # 忽略注释
            'ignore_docstrings': True,  # 忽略文档字符串
        }

        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    @property
    def checker_name(self) -> str:
        return "duplicate_code_checker"

    @property
    def checker_description(self) -> str:
        return "检测代码中的重复模式和相似代码块"

    def check(self, target_path: str) -> 'CheckResult':
        """
        执行代码重复检测

        Args:
            target_path: 检查目标路径

        Returns:
            CheckResult: 检查结果
        """
        result = self._create_result()

        try:
            # 收集所有Python文件
            python_files = self._collect_python_files(target_path)

            if not python_files:
                result.metadata['message'] = "未找到Python文件"
                result.set_end_time()
                return result

            # 提取代码块
            code_blocks = self._extract_code_blocks(python_files)

            # 检测重复
            duplicates = self._find_duplicates(code_blocks)

            # 检测相似代码
            similar_groups = self._find_similar_code(code_blocks)

            # 生成问题报告
            self._generate_duplicate_issues(result, duplicates)
            self._generate_similar_issues(result, similar_groups)

            # 设置元数据
            result.metadata.update({
                'total_files': len(python_files),
                'total_blocks': len(code_blocks),
                'duplicate_groups': len(duplicates),
                'similar_groups': len(similar_groups)
            })

        except Exception as e:
            self.logger.error(f"代码重复检测失败: {e}")
            result.add_issue(self._create_issue(
                file_path=target_path,
                message=f"代码重复检测失败: {e}",
                severity=IssueSeverity.ERROR,
                rule_id="DUPLICATE_CHECK_FAILED"
            ))

        result.set_end_time()
        return result

    def _extract_code_blocks(self, python_files: List[str]) -> List[CodeBlock]:
        """
        从Python文件中提取代码块

        Args:
            python_files: Python文件列表

        Returns:
            List[CodeBlock]: 代码块列表
        """
        code_blocks = []

        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析AST
                tree = ast.parse(content, filename=file_path)

                # 提取函数定义
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        block = self._extract_function_block(file_path, content, node)
                        if block and len(block) >= self.config['min_lines']:
                            code_blocks.append(block)

            except Exception as e:
                self.logger.warning(f"解析文件失败 {file_path}: {e}")

        return code_blocks

    def _extract_function_block(self, file_path: str, content: str, node: ast.AST) -> Optional[CodeBlock]:
        """
        提取函数或类的代码块

        Args:
            file_path: 文件路径
            content: 文件内容
            node: AST节点

        Returns:
            Optional[CodeBlock]: 代码块
        """
        lines = content.split('\n')
        start_line = node.lineno - 1  # AST lineno从1开始

        # 找到结束行
        end_line = start_line
        if hasattr(node, 'body') and node.body:
            # 找到最后一个子节点的行号
            for child in ast.walk(node):
                if hasattr(child, 'lineno'):
                    end_line = max(end_line, child.lineno - 1)

        # 确保行数在合理范围内
        if end_line - start_line + 1 > self.config['max_lines']:
            end_line = start_line + self.config['max_lines'] - 1

        # 提取代码内容
        block_content = '\n'.join(lines[start_line:end_line + 1])

        return CodeBlock(file_path, start_line + 1, end_line + 1, block_content)

    def _find_duplicates(self, code_blocks: List[CodeBlock]) -> List[List[CodeBlock]]:
        """
        查找完全相同的重复代码块

        Args:
            code_blocks: 代码块列表

        Returns:
            List[List[CodeBlock]]: 重复组列表
        """
        hash_groups = defaultdict(list)

        for block in code_blocks:
            hash_groups[block.hash].append(block)

        # 过滤出重复的组
        duplicates = []
        for blocks in hash_groups.values():
            if len(blocks) >= self.config['duplicate_threshold']:
                duplicates.append(blocks)

        return duplicates

    def _find_similar_code(self, code_blocks: List[CodeBlock]) -> List[List[CodeBlock]]:
        """
        查找相似的代码块

        Args:
            code_blocks: 代码块列表

        Returns:
            List[List[CodeBlock]]: 相似组列表
        """
        similar_groups = []

        # 对于每个代码块，查找相似的其他块
        checked = set()

        for i, block1 in enumerate(code_blocks):
            if block1 in checked:
                continue

            similar_blocks = [block1]

            for j, block2 in enumerate(code_blocks):
                if i != j and block2 not in checked:
                    similarity = self._calculate_similarity(block1.content, block2.content)
                    if similarity >= self.config['similarity_threshold']:
                        similar_blocks.append(block2)
                        checked.add(block2)

            if len(similar_blocks) >= 2:
                similar_groups.append(similar_blocks)
                checked.update(similar_blocks)

        return similar_groups

    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """
        计算两段代码的相似度

        Args:
            code1: 第一段代码
            code2: 第二段代码

        Returns:
            float: 相似度 (0.0-1.0)
        """
        # 规范化代码
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)

        # 使用序列匹配器计算相似度
        matcher = difflib.SequenceMatcher(None, norm1.split(), norm2.split())
        return matcher.ratio()

    def _normalize_code(self, code: str) -> str:
        """
        规范化代码用于相似度比较

        Args:
            code: 原始代码

        Returns:
            str: 规范化后的代码
        """
        lines = []
        for line in code.split('\n'):
            # 移除注释
            if self.config.get('ignore_comments', True):
                line = re.sub(r'#.*$', '', line)

            # 移除前后空白
            line = line.strip()

            # 跳过空行和纯注释行
            if line and not line.startswith('#'):
                lines.append(line)

        return '\n'.join(lines)

    def _generate_duplicate_issues(self, result: 'CheckResult',
                                   duplicates: List[List[CodeBlock]]) -> None:
        """
        生成重复代码问题报告

        Args:
            result: 检查结果
            duplicates: 重复组列表
        """
        for group in duplicates:
            # 创建主要问题
            primary_block = group[0]
            duplicate_count = len(group)

            message = f"发现重复代码块，共{duplicate_count}处出现，" \
                f"代码长度{len(primary_block)}行"

            issue = self._create_issue(
                file_path=primary_block.file_path,
                message=message,
                severity=IssueSeverity.WARNING,
                rule_id="DUPLICATE_CODE",
                line_number=primary_block.start_line,
                details={
                    'duplicate_count': duplicate_count,
                    'block_length': len(primary_block),
                    'locations': [str(block) for block in group]
                }
            )
            result.add_issue(issue)

            # 为其他重复位置创建次要问题
            for block in group[1:]:
                issue = self._create_issue(
                    file_path=block.file_path,
                    message=f"重复代码块（与{primary_block}相同）",
                    severity=IssueSeverity.INFO,
                    rule_id="DUPLICATE_CODE_INSTANCE",
                    line_number=block.start_line,
                    details={
                        'primary_location': str(primary_block),
                        'block_length': len(block)
                    }
                )
                result.add_issue(issue)

    def _generate_similar_issues(self, result: 'CheckResult',
                                 similar_groups: List[List[CodeBlock]]) -> None:
        """
        生成相似代码问题报告

        Args:
            result: 检查结果
            similar_groups: 相似组列表
        """
        for group in similar_groups:
            primary_block = group[0]
            similar_count = len(group)

            message = f"发现相似代码块，共{similar_count}处出现，" \
                f"建议提取公共逻辑"

            issue = self._create_issue(
                file_path=primary_block.file_path,
                message=message,
                severity=IssueSeverity.INFO,
                rule_id="SIMILAR_CODE",
                line_number=primary_block.start_line,
                details={
                    'similar_count': similar_count,
                    'block_length': len(primary_block),
                    'locations': [str(block) for block in group]
                }
            )
            result.add_issue(issue)
