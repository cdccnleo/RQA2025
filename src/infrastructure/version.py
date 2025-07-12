from typing import Dict, Optional
import threading

class VersionProxy:
    """版本控制代理，管理配置版本历史"""

    def __init__(self):
        self._versions: Dict[str, Dict[int, Dict]] = {}
        self._latest_versions: Dict[str, int] = {}
        self._lock = threading.Lock()

    def add_version(self, env: str, config: Dict) -> int:
        """添加新版本配置

        Args:
            env: 环境名称
            config: 配置字典

        Returns:
            int: 新版本号
        """
        with self._lock:
            if env not in self._versions:
                self._versions[env] = {}
                self._latest_versions[env] = 0

            new_version = self._latest_versions[env] + 1
            self._versions[env][new_version] = config.copy()
            self._latest_versions[env] = new_version
            return new_version

    def get_version(self, env: str, version: int) -> Optional[Dict]:
        """获取特定版本配置

        Args:
            env: 环境名称
            version: 版本号，-1表示最新版本

        Returns:
            Optional[Dict]: 配置字典，不存在返回None
        """
        with self._lock:
            if env not in self._versions:
                return None

            if version == -1:
                version = self._latest_versions.get(env, 0)

            return self._versions[env].get(version, None)

    def get_latest_version(self, env: str) -> int:
        """获取最新版本号

        Args:
            env: 环境名称

        Returns:
            int: 最新版本号
        """
        with self._lock:
            return self._latest_versions.get(env, 0)

    def diff_versions(self, env: str, v1: int, v2: int) -> Dict:
        """比较两个版本差异

        Args:
            env: 环境名称
            v1: 版本号1
            v2: 版本号2

        Returns:
            Dict: 差异结果
        """
        config1 = self.get_version(env, v1)
        config2 = self.get_version(env, v2)

        if not config1 or not config2:
            return {}

        # 简单实现 - 实际项目中应该使用更专业的diff算法
        diff = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        for key in all_keys:
            if config1.get(key) != config2.get(key):
                diff[key] = {
                    'old': config1.get(key),
                    'new': config2.get(key)
                }
        return diff

def get_default_version_proxy() -> VersionProxy:
    """获取默认版本代理单例"""
    if not hasattr(get_default_version_proxy, "_instance"):
        get_default_version_proxy._instance = VersionProxy()
    return get_default_version_proxy._instance
