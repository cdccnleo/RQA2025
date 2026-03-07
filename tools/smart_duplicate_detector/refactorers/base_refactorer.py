"""
基础重构器

提供代码重构的基础设施和通用功能。
"""

import ast
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from ..core.detection_result import CloneGroup


class RefactoringSuggestion:
    """
    重构建议

    包含重构的具体信息和执行步骤。
    """

    def __init__(self, suggestion_type: str, description: str,
                 impact: str = 'medium', complexity: str = 'medium'):
        self.suggestion_type = suggestion_type
        self.description = description
        self.impact = impact  # 'low', 'medium', 'high'
        self.complexity = complexity  # 'low', 'medium', 'high'
        self.target_files: List[str] = []
        self.changes: List[Dict[str, Any]] = []
        self.confidence_score: float = 0.0

    def add_change(self, file_path: str, change_type: str,
                   old_code: str = None, new_code: str = None,
                   line_number: int = None, **kwargs):
        """
        添加代码变更

        Args:
            file_path: 文件路径
            change_type: 变更类型 ('insert', 'replace', 'delete', 'extract')
            old_code: 原始代码
            new_code: 新代码
            line_number: 行号
            **kwargs: 其他参数
        """
        change = {
            'file_path': file_path,
            'change_type': change_type,
            'old_code': old_code,
            'new_code': new_code,
            'line_number': line_number,
            **kwargs
        }
        self.changes.append(change)
        self.target_files.append(file_path)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'type': self.suggestion_type,
            'description': self.description,
            'impact': self.impact,
            'complexity': self.complexity,
            'target_files': list(set(self.target_files)),
            'changes': self.changes,
            'confidence_score': self.confidence_score
        }


class BaseRefactorer(ABC):
    """
    基础重构器

    提供重构的基础功能和模板方法。
    """

    def __init__(self):
        self.logger = None  # 将在子类中设置

    @abstractmethod
    def can_refactor(self, clone_group: CloneGroup) -> bool:
        """
        判断是否可以对该克隆组进行重构

        Args:
            clone_group: 克隆组

        Returns:
            bool: 是否可以重构
        """

    @abstractmethod
    def generate_suggestion(self, clone_group: CloneGroup) -> Optional[RefactoringSuggestion]:
        """
        生成重构建议

        Args:
            clone_group: 克隆组

        Returns:
            Optional[RefactoringSuggestion]: 重构建议
        """

    def apply_suggestion(self, suggestion: RefactoringSuggestion,
                         dry_run: bool = True) -> Dict[str, Any]:
        """
        应用重构建议

        Args:
            suggestion: 重构建议
            dry_run: 是否仅预览变更

        Returns:
            Dict[str, Any]: 应用结果
        """
        results = {
            'applied': False,
            'changes_made': [],
            'errors': [],
            'backup_files': []
        }

        if dry_run:
            results['preview'] = [change for change in suggestion.changes]
            return results

        try:
            # 创建备份
            for file_path in set(suggestion.target_files):
                backup_path = self._create_backup(file_path)
                if backup_path:
                    results['backup_files'].append(backup_path)

            # 应用变更
            for change in suggestion.changes:
                result = self._apply_change(change)
                if result['success']:
                    results['changes_made'].append(result)
                else:
                    results['errors'].append(result)

            results['applied'] = len(results['errors']) == 0

        except Exception as e:
            results['errors'].append({'error': str(e)})

        return results

    def _create_backup(self, file_path: str) -> Optional[str]:
        """
        创建文件备份

        Args:
            file_path: 原始文件路径

        Returns:
            Optional[str]: 备份文件路径
        """
        try:
            backup_path = f"{file_path}.backup"
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            return backup_path
        except Exception:
            return None

    def _apply_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用单个变更

        Args:
            change: 变更信息

        Returns:
            Dict[str, Any]: 应用结果
        """
        try:
            file_path = change['file_path']
            change_type = change['change_type']

            if change_type == 'replace':
                return self._apply_replace_change(file_path, change)
            elif change_type == 'insert':
                return self._apply_insert_change(file_path, change)
            elif change_type == 'delete':
                return self._apply_delete_change(file_path, change)
            elif change_type == 'extract':
                return self._apply_extract_change(file_path, change)
            else:
                return {
                    'success': False,
                    'error': f'不支持的变更类型: {change_type}',
                    'change': change
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'change': change
            }

    def _apply_replace_change(self, file_path: str, change: Dict[str, Any]) -> Dict[str, Any]:
        """应用替换变更"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            old_code = change['old_code']
            new_code = change['new_code']

            if old_code not in content:
                return {
                    'success': False,
                    'error': '未找到要替换的代码',
                    'change': change
                }

            new_content = content.replace(old_code, new_code, 1)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return {
                'success': True,
                'change': change,
                'description': f'替换了 {len(old_code)} 个字符'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'change': change
            }

    def _apply_insert_change(self, file_path: str, change: Dict[str, Any]) -> Dict[str, Any]:
        """应用插入变更"""
        # 简化实现，实际需要更复杂的行号处理
        return {
            'success': False,
            'error': '插入变更暂未实现',
            'change': change
        }

    def _apply_delete_change(self, file_path: str, change: Dict[str, Any]) -> Dict[str, Any]:
        """应用删除变更"""
        return {
            'success': False,
            'error': '删除变更暂未实现',
            'change': change
        }

    def _apply_extract_change(self, file_path: str, change: Dict[str, Any]) -> Dict[str, Any]:
        """应用提取变更"""
        return {
            'success': False,
            'error': '提取变更暂未实现',
            'change': change
        }

    def validate_suggestion(self, suggestion: RefactoringSuggestion) -> Dict[str, Any]:
        """
        验证重构建议的有效性

        Args:
            suggestion: 重构建议

        Returns:
            Dict[str, Any]: 验证结果
        """
        issues = []

        # 检查文件是否存在
        for file_path in suggestion.target_files:
            if not os.path.exists(file_path):
                issues.append(f'目标文件不存在: {file_path}')

        # 检查变更的合理性
        for change in suggestion.changes:
            file_path = change['file_path']
            if not os.path.exists(file_path):
                issues.append(f'变更目标文件不存在: {file_path}')
                continue

            # 检查代码变更的语法有效性
            if 'new_code' in change and change['new_code']:
                if not self._is_valid_python_code(change['new_code']):
                    issues.append(f'新代码语法无效: {change["new_code"][:50]}...')

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'confidence_score': suggestion.confidence_score
        }

    def _is_valid_python_code(self, code: str) -> bool:
        """
        检查代码语法是否有效

        Args:
            code: Python代码

        Returns:
            bool: 是否有效
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def estimate_effort(self, suggestion: RefactoringSuggestion) -> Dict[str, Any]:
        """
        估算重构工作量

        Args:
            suggestion: 重构建议

        Returns:
            Dict[str, Any]: 工作量估算
        """
        base_effort = {
            'time_estimate': 0,  # 分钟
            'risk_level': 'low',
            'affected_lines': 0,
            'files_to_modify': len(set(suggestion.target_files))
        }

        # 根据变更类型调整估算
        for change in suggestion.changes:
            change_type = change['change_type']
            if change_type == 'extract':
                base_effort['time_estimate'] += 30  # 提取方法大约30分钟
                base_effort['affected_lines'] += change.get('lines_affected', 10)
            elif change_type == 'replace':
                base_effort['time_estimate'] += 15  # 替换大约15分钟
                old_code = change.get('old_code', '')
                base_effort['affected_lines'] += len(old_code.split('\n'))

        # 根据复杂度调整风险
        if suggestion.complexity == 'high':
            base_effort['risk_level'] = 'high'
            base_effort['time_estimate'] *= 1.5
        elif suggestion.complexity == 'medium':
            base_effort['risk_level'] = 'medium'
            base_effort['time_estimate'] *= 1.2

        return base_effort
