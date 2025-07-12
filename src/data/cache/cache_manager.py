import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import lz4.frame
import pickle
from datetime import datetime
from .memory_cache import MemoryCache
from .disk_cache import DiskCache

@dataclass
class CacheConfig:
    """缓存配置项"""
    strategy: str = 'default'
    ttl: int = 3600
    compression: Optional[str] = None
    storage: List[str] = None
    version_control: bool = False

class CacheManager:
    """增强版缓存管理器"""

    def __init__(self):
        self.memory_cache = MemoryCache()
        self.disk_cache = DiskCache()
        self.lock = threading.Lock()
        self.version_store = {}  # 版本控制存储
        self.default_config = CacheConfig()
        self.configs: Dict[str, CacheConfig] = {}

    def configure(self, config: Dict[str, Dict]) -> None:
        """更新缓存配置"""
        with self.lock:
            for data_type, params in config.items():
                self.configs[data_type] = CacheConfig(
                    strategy=params.get('strategy', 'default'),
                    ttl=params.get('ttl', 3600),
                    compression=params.get('compression'),
                    storage=params.get('storage', ['memory']),
                    version_control=params.get('version_control', False)
                )

    def get(self, key: str, data_type: str) -> Optional[Any]:
        """获取缓存数据（支持版本控制）"""
        config = self._get_config(data_type)

        with self.lock:
            # 检查版本控制
            if config.version_control:
                version_key = f"{key}_versions"
                versions = self.disk_cache.get(version_key)
                if versions:
                    latest = max(versions.keys())
                    key = versions[latest]  # 获取最新版本的实际键名

            # 按存储策略顺序查找
            for storage in config.storage:
                if storage == 'memory':
                    data = self.memory_cache.get(key)
                else:
                    data = self.disk_cache.get(key)

                if data is not None:
                    # 解压缩处理
                    if config.compression == 'lz4':
                        data = pickle.loads(lz4.frame.decompress(data))
                    return data

        return None

    def set(self, key: str, value: Any, data_type: str) -> None:
        """设置缓存数据（支持版本控制）"""
        config = self._get_config(data_type)

        with self.lock:
            # 处理数据压缩
            processed_value = value
            if config.compression == 'lz4':
                processed_value = lz4.frame.compress(pickle.dumps(value))

            # 版本控制处理
            if config.version_control:
                version_key = f"{key}_versions"
                versions = self.disk_cache.get(version_key) or {}
                version = datetime.now().timestamp()
                versioned_key = f"{key}_v{version}"
                versions[version] = versioned_key
                self.disk_cache.set(version_key, versions)
                key = versioned_key

            # 按存储策略存储
            for storage in config.storage:
                if storage == 'memory':
                    self.memory_cache.set(key, processed_value, config.ttl)
                else:
                    self.disk_cache.set(key, processed_value, config.ttl)

    def clear(self, data_type: str = None) -> None:
        """清理缓存"""
        with self.lock:
            if data_type is None:
                self.memory_cache.clear()
                self.disk_cache.clear()
                self.version_store.clear()
            else:
                # 按类型清理的逻辑
                pass

    def _get_config(self, data_type: str) -> CacheConfig:
        """获取数据类型的缓存配置"""
        return self.configs.get(data_type, self.default_config)

    def _apply_strategy(self, key: str, value: Any, config: CacheConfig) -> Any:
        """应用缓存策略"""
        # 策略相关处理
        if config.strategy == 'aggressive':
            # 激进策略：预加载相关数据
            pass
        elif config.strategy == 'incremental':
            # 增量策略：只更新变化部分
            pass
        return value
