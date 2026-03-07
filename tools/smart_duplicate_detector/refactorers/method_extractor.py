"""
方法提取器

将重复代码提取为公共方法的自动化重构器。
"""

from typing import Optional
from ..core.code_fragment import CodeFragment, FragmentType
from ..core.detection_result import CloneGroup
from .base_refactorer import BaseRefactorer, RefactoringSuggestion


class MethodExtractor(BaseRefactorer):
    """
    方法提取重构器

    将相似的代码片段提取为公共方法。
    """

    def __init__(self):
        super().__init__()

    def can_refactor(self, clone_group: CloneGroup) -> bool:
        """
        判断是否可以提取方法

        Args:
            clone_group: 克隆组

        Returns:
            bool: 是否可以重构
        """
        # 检查条件：
        # 1. 至少有3个相似片段
        # 2. 相似度足够高
        # 3. 都是函数或方法片段
        if len(clone_group) < 3:
            return False

        if clone_group.similarity_score < 0.8:
            return False

        # 检查是否都是函数/方法片段
        valid_types = {FragmentType.FUNCTION, FragmentType.METHOD}
        for fragment in clone_group.fragments:
            if fragment.fragment_type not in valid_types:
                return False

        return True

    def generate_suggestion(self, clone_group: CloneGroup) -> Optional[RefactoringSuggestion]:
        """
        生成方法提取建议

        Args:
            clone_group: 克隆组

        Returns:
            Optional[RefactoringSuggestion]: 重构建议
        """
        if not self.can_refactor(clone_group):
            return None

        # 创建建议
        suggestion = RefactoringSuggestion(
            suggestion_type='extract_method',
            description=f'将{len(clone_group)}处重复代码提取为公共方法',
            impact='high',
            complexity='medium'
        )

        # 确定目标文件（选择包含最多片段的文件）
        target_file = self._select_target_file(clone_group)
        suggestion.target_files.append(target_file)

        # 生成新方法
        method_code = self._generate_method_code(clone_group)
        method_name = self._generate_method_name(clone_group)

        # 添加方法定义变更
        suggestion.add_change(
            file_path=target_file,
            change_type='insert',
            new_code=method_code,
            line_number=self._find_insertion_point(target_file),
            method_name=method_name
        )

        # 为每个重复片段添加替换变更
        for fragment in clone_group.fragments:
            replacement_call = self._generate_method_call(method_name, fragment)

            suggestion.add_change(
                file_path=fragment.file_path,
                change_type='replace',
                old_code=fragment.raw_content,
                new_code=replacement_call,
                line_number=fragment.start_line
            )

        # 计算置信度
        suggestion.confidence_score = self._calculate_confidence(clone_group)

        return suggestion

    def _select_target_file(self, clone_group: CloneGroup) -> str:
        """
        选择目标文件

        选择包含最多片段的文件作为目标文件。

        Args:
            clone_group: 克隆组

        Returns:
            str: 目标文件路径
        """
        from collections import Counter

        file_counts = Counter(frag.file_path for frag in clone_group.fragments)
        return file_counts.most_common(1)[0][0]

    def _generate_method_name(self, clone_group: CloneGroup) -> str:
        """
        生成方法名

        Args:
            clone_group: 克隆组

        Returns:
            str: 方法名
        """
        # 使用第一个片段的上下文生成有意义的方法名
        first_fragment = clone_group.fragments[0]

        # 简单的启发式方法名生成
        if 'process' in first_fragment.raw_content.lower():
            return '_extracted_process_method'
        elif 'calculate' in first_fragment.raw_content.lower():
            return '_extracted_calculate_method'
        elif 'validate' in first_fragment.raw_content.lower():
            return '_extracted_validate_method'
        else:
            return f'_extracted_common_method_{clone_group.group_id[:4]}'

    def _generate_method_code(self, clone_group: CloneGroup) -> str:
        """
        生成新方法代码

        Args:
            clone_group: 克隆组

        Returns:
            str: 方法代码
        """
        method_name = self._generate_method_name(clone_group)
        first_fragment = clone_group.fragments[0]

        # 提取方法的缩进级别
        lines = first_fragment.raw_content.split('\n')
        if lines:
            # 找到最小缩进
            indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
            min_indent = min(indents) if indents else 0
        else:
            min_indent = 0

        # 规范化代码（减少缩进）
        normalized_lines = []
        for line in lines:
            if line.strip():  # 只处理非空行
                # 减少缩进
                if len(line) > min_indent:
                    normalized_lines.append(line[min_indent:])
                else:
                    normalized_lines.append(line.lstrip())
            else:
                normalized_lines.append('')

        method_body = '\n'.join(normalized_lines)

        # 生成完整的方法定义
        method_code = f'''
    def {method_name}(self):
        """
        提取的公共方法
        从{len(clone_group)}处重复代码中提取
        """
{method_body}
'''

        return method_code

    def _generate_method_call(self, method_name: str, fragment: CodeFragment) -> str:
        """
        生成方法调用代码

        Args:
            method_name: 方法名
            fragment: 原始片段

        Returns:
            str: 方法调用代码
        """
        # 确定正确的缩进
        lines = fragment.raw_content.split('\n')
        if lines:
            indent = len(lines[0]) - len(lines[0].lstrip())
            indent_str = ' ' * indent
        else:
            indent_str = ''

        return f"{indent_str}self.{method_name}()"

    def _find_insertion_point(self, file_path: str) -> int:
        """
        找到合适的方法插入点

        Args:
            file_path: 文件路径

        Returns:
            int: 插入行号
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 查找类定义的末尾
            for i, line in enumerate(lines):
                if line.strip().startswith('class '):
                    # 找到类的结束
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith('def ') and not lines[j].startswith('    '):
                            # 找到了类外的第一个方法，插入到类定义后
                            return j
                        elif lines[j].strip() == '' and j > i + 2:
                            # 找到合适的位置
                            return j + 1

            # 如果没找到合适位置，插入到文件末尾
            return len(lines) + 1

        except Exception:
            return 1

    def _calculate_confidence(self, clone_group: CloneGroup) -> float:
        """
        计算重构置信度

        Args:
            clone_group: 克隆组

        Returns:
            float: 置信度分数 (0.0-1.0)
        """
        confidence = 0.5  # 基础置信度

        # 相似度越高，置信度越高
        confidence += clone_group.similarity_score * 0.3

        # 片段数量越多，置信度越高
        if len(clone_group) >= 5:
            confidence += 0.2
        elif len(clone_group) >= 3:
            confidence += 0.1

        # 如果都是相同类型的片段，置信度更高
        types = set(frag.fragment_type for frag in clone_group.fragments)
        if len(types) == 1:
            confidence += 0.1

        return min(confidence, 1.0)
