from deepdiff import DeepDiff
from typing import Dict, Any

"""
基础设施层 - 工具组件组件

diff_service 模块

通用工具组件
提供工具组件相关的功能实现。
"""


class DictDiffService:
    """
    diff_service - 配置管理

    职责说明：
    负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

    核心职责：
    - 配置文件的读取和解析
    - 配置参数的验证
    - 配置的热重载
    - 配置的分发和同步
    - 环境变量管理
    - 配置加密和安全

    相关接口：
    - IConfigComponent
    - IConfigManager
    - IConfigValidator
    """
    """字典差异比较服务"""

    def __init__(self, ignore_types: tuple = (float, int)):
        self.ignore_types = ignore_types

    def compare_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any], **kwargs):
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
                # DeepDiff返回的是字典，键是路径
                result['values_changed'] = [
                    {'path': k, 'old': v['old_value'], 'new': v['new_value']}
                    for k, v in changes.items()
                ]
            elif change_type.startswith('type_changes'):
                # DeepDiff返回的是字典，键是路径
                result['type_changes'] = [
                    {'path': k, 'old_type': v['old_type'],
                     'new_type': v['new_type'], 'old': v['old_value'], 'new': v['new_value']}
                    for k, v in changes.items()
                ]
            elif change_type == 'dictionary_item_added':
                # DeepDiff返回的是列表
                result['added'] = list(changes)
            elif change_type == 'dictionary_item_removed':
                # DeepDiff返回的是列表
                result['removed'] = list(changes)

        return result

    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个配置的差异（接口方法）"""
        return self.compare_dicts(config1, config2)

    def get_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """获取配置变更（接口方法）"""
        return self.compare_dicts(old_config, new_config)

    def apply_diff(self, base_config: Dict[str, Any], diff: Dict[str, Any]) -> Dict[str, Any]:
        """应用配置差异（接口方法）"""
        result_config = base_config.copy()

        # 应用新增的配置
        if 'added' in diff:
            for key_path in diff['added']:
                # 从路径中提取键名，例如 "root['key']" -> "key"
                key = self._extract_key_from_path(key_path)
                if key and key not in result_config:
                    result_config[key] = None

        # 应用删除的配置
        if 'removed' in diff:
            for key_path in diff['removed']:
                key = self._extract_key_from_path(key_path)
                if key and key in result_config:
                    del result_config[key]

        # 应用值变更
        if 'values_changed' in diff:
            for change in diff['values_changed']:
                if 'path' in change and 'new' in change:
                    key = self._extract_key_from_path(change['path'])
                    if key and key in result_config:
                        result_config[key] = change['new']

        return result_config

    def _extract_key_from_path(self, path: str) -> str:
        """从DeepDiff路径中提取键名

        Args:
            path: DeepDiff路径，例如 "root['key']" 或 "root['level1']['level2']['key']"

        Returns:
            提取的键名，例如 "key"
        """
        if not path or not path.startswith("root["):
            return None

        # 移除 'root[' 前缀
        path = path[6:]

        # 处理单引号和双引号的情况
        if "']['" in path:
            # 分割嵌套路径并取最后一个
            parts = path.split("']['")
            last_part = parts[-1]
            # 移除结尾的 ]
            if last_part.endswith("']"):
                last_part = last_part[:-2]
            return last_part
        elif "\"][\"" in path:
            # 处理双引号的情况
            parts = path.split("\"][\"")
            last_part = parts[-1]
            # 移除结尾的 "]
            if last_part.endswith("\"]"):
                last_part = last_part[:-2]
            return last_part
        else:
            # 简单路径，移除结尾的 ]
            if path.endswith("']"):
                path = path[:-2]
            elif path.endswith("\"]"):
                path = path[:-2]
            return path