"""配置版本控制服务

提供配置版本管理功能：
1. 版本记录
2. 版本比较
3. 版本回滚
"""
from typing import Dict, Any, List
import time
from copy import deepcopy

class VersionService:
    """配置版本服务基础实现"""

    def __init__(self):
        self._versions = {}  # {env: [version_history]}
        self._max_versions = 50  # 默认保留50个版本

    def add_version(self, env: str, config: Dict[str, Any], metadata: Dict[str, Any] = None) -> int:
        """添加新版本

        Args:
            env: 环境名称
            config: 配置字典
            metadata: 版本元数据

        Returns:
            版本号(从0开始)
        """
        if env not in self._versions:
            self._versions[env] = []

        version_data = {
            'config': deepcopy(config),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self._versions[env].append(version_data)

        # 保持版本数量限制
        if len(self._versions[env]) > self._max_versions:
            self._versions[env].pop(0)

        return len(self._versions[env]) - 1

    def get_version(self, env: str, version: int) -> Dict[str, Any]:
        """获取指定版本配置

        Args:
            env: 环境名称
            version: 版本号(负数表示从最新开始)

        Returns:
            版本数据字典
        """
        if env not in self._versions or not self._versions[env]:
            raise ValueError(f"No versions available for env: {env}")

        if version < 0:
            version = len(self._versions[env]) + version

        return deepcopy(self._versions[env][version])

    def get_version_history(self, env: str) -> List[Dict[str, Any]]:
        """获取版本历史

        Args:
            env: 环境名称

        Returns:
            版本历史列表(最新最后)
        """
        return deepcopy(self._versions.get(env, []))

    def set_max_versions(self, max_versions: int):
        """设置最大保留版本数"""
        self._max_versions = max_versions
