
import shutil
import logging
import threading
import os
import time
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List

"""配置文件存储相关类"""

from .configitem import ConfigItem
from .configscope import ConfigScope
from .iconfigstorage import IConfigStorage
from .storageconfig import StorageConfig
from .storagetype import StorageType

logger = logging.getLogger(__name__)


class FileConfigStorage(IConfigStorage):
    """文件配置存储实现"""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._data: Dict[ConfigScope, Dict[str, ConfigItem]] = defaultdict(dict)
        self._lock = threading.RLock()

        # 确保配置目录存在
        if config.path:
            Path(config.path).parent.mkdir(parents=True, exist_ok=True)

        # 加载现有配置
        self.load()

    def get(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> Optional[Any]:
        """获取配置值"""
        item = self._get_item(key, scope)
        return item.value if item else None

    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """设置配置值"""
        try:
            with self._lock:
                timestamp = time.time()
                version = self._generate_version(key, value, timestamp)

                item = ConfigItem(
                    key=key,
                    value=value,
                    scope=scope,
                    timestamp=timestamp,
                    version=version
                )

                self._data[scope][key] = item

                # 自动保存
                if self.config.type == StorageType.FILE:
                    self.save()

                return True
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False

    def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """删除配置"""
        try:
            with self._lock:
                if scope in self._data and key in self._data[scope]:
                    del self._data[scope][key]

                    # 自动保存
                    if self.config.type == StorageType.FILE:
                        self.save()

                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete config {key}: {e}")
            return False

    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""
        with self._lock:
            return scope in self._data and key in self._data[scope]

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""
        with self._lock:
            if scope:
                return list(self._data.get(scope, {}).keys())
            else:
                all_keys = []
                for scope_data in self._data.values():
                    all_keys.extend(scope_data.keys())
                return all_keys

    def save(self) -> bool:
        """保存到文件"""
        try:
            if not self.config.path:
                return True

            # 转换为可序列化的格式
            serializable_data = {}
            for scope, items in self._data.items():
                serializable_data[scope.value] = {}
                for key, item in items.items():
                    serializable_data[scope.value][key] = {
                        'value': item.value,
                        'timestamp': item.timestamp,
                        'version': item.version,
                        'metadata': item.metadata
                    }

            # 备份现有文件
            if self.config.backup_enabled and os.path.exists(self.config.path):
                self._create_backup()

            # 保存到文件
            with open(self.config.path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config.path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def load(self) -> bool:
        """从文件加载"""
        try:
            if not self.config.path or not os.path.exists(self.config.path):
                return True

            with open(self.config.path, 'r', encoding='utf-8') as f:
                serializable_data = json.load(f)

            # 转换回内部格式
            for scope_str, items in serializable_data.items():
                scope = ConfigScope(scope_str)
                for key, item_data in items.items():
                    item = ConfigItem(
                        key=key,
                        value=item_data['value'],
                        scope=scope,
                        timestamp=item_data['timestamp'],
                        version=item_data.get('version', '1.0'),
                        metadata=item_data.get('metadata', {})
                    )
                    self._data[scope][key] = item

            logger.info(f"Configuration loaded from {self.config.path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    def _generate_version(self, key: str, value: Any, timestamp: float) -> str:
        """生成版本号"""
        content = f"{key}:{value}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def _create_backup(self):
        """创建备份"""
        try:
            if not self.config.path:
                return

            backup_dir = Path(self.config.path).parent / "backups"
            backup_dir.mkdir(exist_ok=True)

            # 清理旧备份
            existing_backups = sorted(backup_dir.glob("*.bak"))
            if len(existing_backups) >= self.config.max_backups:
                for old_backup in existing_backups[:-self.config.max_backups + 1]:
                    old_backup.unlink()

            # 创建新备份
            timestamp = int(time.time())
            backup_path = backup_dir / f"config_{timestamp}.bak"

            # 使用shutil.move来处理跨平台兼容性
            shutil.move(str(self.config.path), str(backup_path))
            logger.info(f"Backup created: {backup_path}")

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _get_item(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION):
        """获取配置项对象（内部方法）"""
        with self._lock:
            if scope in self._data and key in self._data[scope]:
                return self._data[scope][key]
            return None




