
"""配置文件存储相关类"""

from typing import Dict, Any, Optional, List
from .configscope import ConfigScope
from .fileconfigstorage import FileConfigStorage
from .storageconfig import StorageConfig
from .storagetype import StorageType


class ConfigStorage(FileConfigStorage):
    """配置存储（向后兼容简单接口）"""

    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """初始化配置存储"""
        if storage_config is None:
            storage_config = {}

        # 默认配置
        default_path = storage_config.get('path', 'config/app_config.json')

        config = StorageConfig()
        config.type = StorageType.FILE
        config.path = default_path
        config.backup_enabled = storage_config.get('backup_enabled', True)
        config.max_backups = storage_config.get('max_backups', 5)

        super().__init__(config)
        self.storage = self._data  # 兼容原有接口
        self._configs = self._data  # 兼容测试期望的属性

    def set_config(self, name: str, config: Dict[str, Any]) -> bool:
        """设置配置（兼容接口）"""
        return self.set(name, config)

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """获取配置（兼容接口）"""
        return self.get(name)

    def list_configs(self) -> List[str]:
        """列出所有配置（兼容接口）"""
        return self.list_keys()

    # 实现IConfigStorage接口的额外方法（如果FileConfigStorage没有完全实现）
    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""
        return super().exists(key, scope)

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""
        return super().list_keys(scope)

# ==================== 对外接口 ====================

# 向后兼容的类型别名




