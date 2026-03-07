"""
版本策略模块

提供版本策略管理和验证功能。
"""

from typing import Dict, List, Callable, Optional, Union
from ..core.version import Version


class VersionPolicy:
    """
    版本策略管理器

    提供版本策略的定义、验证和管理功能。
    """

    def __init__(self):
        """初始化版本策略管理器"""
        self._policies: Dict[str, Callable[[Version], bool]] = {}

        # 默认策略函数
        self._default_policy = lambda v: v.is_stable() and v.major >= 1

        # 预定义常用策略
        self._init_default_policies()

    def _init_default_policies(self):
        """初始化默认策略"""
        # 稳定版本策略
        self._policies['stable'] = lambda v: v.is_stable()

        # 预发布版本策略
        self._policies['prerelease'] = lambda v: v.is_prerelease()

        # 主要版本策略 (1.x.x)
        self._policies['major'] = lambda v: v.major >= 1

        # 次要版本策略 (允许0.x.x但要求稳定)
        self._policies['minor'] = lambda v: v.is_stable()

        # 补丁版本策略 (任何版本)
        self._policies['patch'] = lambda v: True

        # 最新版本策略 (总是允许)
        self._policies['latest'] = lambda v: True

        # 企业级稳定版本策略
        self._policies['enterprise'] = lambda v: (
            v.is_stable() and
            v.major >= 1 and
            v.patch >= 0  # 至少有补丁版本
        )

    def add_policy(self, name: str, policy_func: Callable[[Version], bool]) -> None:
        """
        添加自定义策略

        Args:
            name: 策略名称
            policy_func: 策略函数，接受Version对象，返回bool
        """
        if not callable(policy_func):
            raise ValueError("策略函数必须是可调用的")
        self._policies[name] = policy_func

    def remove_policy(self, name: str) -> bool:
        """
        移除策略

        Args:
            name: 策略名称

        Returns:
            是否成功移除
        """
        if name in self._policies and name not in ['stable', 'prerelease', 'major', 'minor', 'patch', 'latest', 'enterprise']:
            del self._policies[name]
            return True
        return False

    def validate_version(self, version: Union[str, Version], policy_name: str = None) -> bool:
        """
        验证版本是否符合策略

        Args:
            version: 要验证的版本
            policy_name: 策略名称，None表示使用默认策略

        Returns:
            版本是否符合策略
        """
        if isinstance(version, str):
            version = Version(version)

        if policy_name and policy_name in self._policies:
            # 使用指定的策略
            return self._policies[policy_name](version)
        else:
            # 使用默认策略
            return self._default_policy(version)

    def list_policies(self) -> List[str]:
        """
        列出所有可用策略

        Returns:
            策略名称列表
        """
        return list(self._policies.keys()) + ['default']

    def get_policy_description(self, policy_name: str) -> Optional[str]:
        """
        获取策略描述

        Args:
            policy_name: 策略名称

        Returns:
            策略描述，不存在返回None
        """
        descriptions = {
            'stable': '只允许稳定版本 (主版本 >= 1，无预发布标识符)',
            'prerelease': '只允许预发布版本 (包含预发布标识符)',
            'major': '只允许主要版本 (主版本 >= 1)',
            'minor': '只允许稳定版本 (无预发布标识符)',
            'patch': '允许任何版本',
            'latest': '允许任何版本 (等同patch)',
            'enterprise': '企业级稳定版本 (稳定 + 主版本 >= 1 + 至少一个补丁)',
            'default': '默认策略 (稳定版本且主版本 >= 1)'
        }

        if policy_name in descriptions:
            return descriptions[policy_name]

        if policy_name in self._policies:
            return f"自定义策略: {policy_name}"

        return None

    def clear_policies(self) -> None:
        """清空所有自定义策略（保留预定义策略）"""
        # 保留预定义策略
        default_policies = ['stable', 'prerelease', 'major',
                            'minor', 'patch', 'latest', 'enterprise']
        self._policies = {k: v for k, v in self._policies.items() if k in default_policies}

    def validate_version_range(self, versions: List[Union[str, Version]],
                               policy_name: str = None) -> List[bool]:
        """
        批量验证版本范围

        Args:
            versions: 版本列表
            policy_name: 策略名称

        Returns:
            验证结果列表
        """
        return [self.validate_version(v, policy_name) for v in versions]

    def find_compliant_versions(self, versions: List[Union[str, Version]],
                                policy_name: str = None) -> List[Version]:
        """
        查找符合策略的版本

        Args:
            versions: 版本列表
            policy_name: 策略名称

        Returns:
            符合策略的版本列表
        """
        compliant = []
        for v in versions:
            version_obj = Version(v) if isinstance(v, str) else v
            if self.validate_version(version_obj, policy_name):
                compliant.append(version_obj)
        return compliant

    def get_policy_stats(self) -> Dict[str, int]:
        """
        获取策略使用统计

        Returns:
            策略统计信息
        """
        return {
            'total_policies': len(self._policies) + 1,  # 包括默认策略
            'custom_policies': len([p for p in self._policies.keys()
                                   if p not in ['stable', 'prerelease', 'major', 'minor', 'patch', 'latest', 'enterprise']]),
            'predefined_policies': 7  # 固定的预定义策略数量
        }

    # ===== 新增的便捷方法，提供测试和旧接口兼容 =====

    def _ensure_version(self, version: Union[str, Version]) -> Version:
        """内部工具：统一将输入转换为Version对象"""
        return Version(version) if isinstance(version, str) else version

    def get_next_version(self, version: Union[str, Version],
                         bump_type: str = "patch",
                         increment_type: Optional[str] = None) -> Version:
        """
        根据指定的版本增量类型返回下一个版本。

        Args:
            version: 当前版本
            bump_type: 版本递增类型，支持 'major'、'minor'、'patch'

        Returns:
            递增后的版本对象
        """
        input_is_str = isinstance(version, str)
        version = self._ensure_version(version)
        # 复制一份，避免修改传入对象
        next_version = Version(str(version))

        bump = (increment_type or bump_type or "patch").lower()
        if bump == "major":
            next_version.increment_major()
        elif bump == "minor":
            next_version.increment_minor()
        elif bump == "patch":
            next_version.increment_patch()
        else:
            raise ValueError(f"不支持的版本递增类型: {bump}")
        return str(next_version) if input_is_str else next_version

    def allows_upgrade(self, old_version: Union[str, Version],
                       new_version: Union[str, Version]) -> bool:
        """判断是否允许从旧版本升级到新版本"""
        old_version = self._ensure_version(old_version)
        new_version = self._ensure_version(new_version)
        return new_version > old_version

    def allows_downgrade(self, current_version: Union[str, Version],
                         target_version: Union[str, Version]) -> bool:
        """判断是否允许从当前版本降级到目标版本"""
        current_version = self._ensure_version(current_version)
        target_version = self._ensure_version(target_version)
        return target_version < current_version

    def is_compatible(self, version_a: Union[str, Version],
                      version_b: Union[str, Version]) -> bool:
        """判断两个版本是否兼容（主版本号一致即视为兼容）"""
        version_a = self._ensure_version(version_a)
        version_b = self._ensure_version(version_b)
        return version_a.major == version_b.major