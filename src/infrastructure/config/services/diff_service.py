from typing import Dict, Any, Union
from deepdiff import DeepDiff
from ..interfaces.diff_service import IDiffService

class DictDiffService(IDiffService):
    """字典差异比较服务"""

    def __init__(self, ignore_types: tuple = (float, int)):
        self.ignore_types = ignore_types

    def compare_dicts(self,
                     d1: Dict[str, Any],
                     d2: Dict[str, Any],
                     **kwargs) -> Dict[str, Any]:
        """深度比较两个字典的差异

        Args:
            d1: 旧版本字典
            d2: 新版本字典
            kwargs: 传递给DeepDiff的参数

        Returns:
            差异报告字典，包含以下可能键:
            - type_changes: 类型变化的字段
            - values_changed: 值变化的字段
            - dictionary_item_added: 新增的字段
            - dictionary_item_removed: 删除的字段
        """
        diff = DeepDiff(d1, d2,
                       ignore_type_in_groups=[self.ignore_types],
                       **kwargs)
        return self._format_diff(diff)

    def _format_diff(self, diff: Dict[str, Any]) -> Dict[str, Any]:
        """格式化差异结果"""
        result = {}
        for change_type, changes in diff.items():
            if change_type.startswith('values_changed'):
                result['values_changed'] = [
                    {'path': k, 'old': v['old_value'], 'new': v['new_value']}
                    for k, v in changes.items()
                ]
            elif change_type.startswith('type_changes'):
                result['type_changes'] = [
                    {'path': k, 'old_type': str(v['old_type']),
                     'new_type': str(v['new_type'])}
                    for k, v in changes.items()
                ]
            elif change_type == 'dictionary_item_added':
                result['added'] = list(changes)
            elif change_type == 'dictionary_item_removed':
                result['removed'] = list(changes)

        return result
