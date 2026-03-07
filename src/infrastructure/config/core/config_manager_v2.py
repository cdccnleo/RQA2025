
#!/usr/bin/env python3
"""
配置管理器V2 - 完全重构版本

Phase 1重构：彻底消除循环依赖，使用纯组合模式
独立于现有manager类的全新实现
"""

import logging
import time
import threading

from ..storage.types.fileconfigstorage import FileConfigStorage
from ..storage.types.storageconfig import StorageConfig
from ..storage.types.storagetype import StorageType
from .common_mixins import BatchOperationsMixin
from .imports import (
    Dict, Any, List, Callable
)
from ..services.config_operations_service import ConfigOperationsService
from ..services.config_storage_service import ConfigStorageService
from ..services.iconfig_service import BaseConfigService
logger = logging.getLogger(__name__)


class ConfigManagerV2(BaseConfigService, BatchOperationsMixin):
    """配置管理器V2 - 完全重构版本

    使用纯组合模式，彻底消除循环依赖
    独立于原有manager类的新实现
    """

    def __init__(self, storage_backend=None, cache_enabled: bool = True,
                 service_name: str = "config: config_manager_v2"):
        """初始化配置管理器V2

        Args:
            storage_backend: 存储后端实现
            cache_enabled: 是否启用缓存
            service_name: 服务名称
        """
        super().__init__(service_name)

        # 使用组合模式组合各个服务
        self._storage_service = ConfigStorageService(cache_enabled=cache_enabled)
        self._operations_service = ConfigOperationsService(self._storage_service)

        # 如果提供了存储后端，设置它
        if storage_backend:
            self._storage_service.set_storage_backend(storage_backend)

        # 配置数据存储（临时解决方案，直到完全实现存储服务）
        self._config_data: Dict[str, Any] = {}
        self._config_lock = threading.RLock()

        # 性能统计
        self._stats = {
            'operations': 0,
            'cache_hits': 0,
            'errors': 0
        }

    def _initialize(self):
        """初始化服务"""
        self._start_time = time.time()
        logger.info(f"{self._service_name} 已初始化（V2纯组合模式）")

    # ========== 核心配置操作接口 ==========

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        self._ensure_initialized()

        try:
            self._stats['operations'] += 1

            # 首先尝试从本地存储获取（临时实现）
            with self._config_lock:
                if key in self._config_data:
                    return self._config_data[key]

            # 返回默认值
            return default

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"获取配置失败 {key}: {e}")
            return default

    def set(self, key: str, value: Any) -> bool:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值

        Returns:
            设置是否成功
        """
        self._ensure_initialized()

        try:
            self._stats['operations'] += 1

            # 临时实现：直接存储到本地
            with self._config_lock:
                self._config_data[key] = value

            logger.debug(f"配置已设置: {key}")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"设置配置失败 {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除配置项

        Args:
            key: 配置键

        Returns:
            删除是否成功
        """
        self._ensure_initialized()

        try:
            self._stats['operations'] += 1

            with self._config_lock:
                if key in self._config_data:
                    del self._config_data[key]
                    logger.debug(f"配置已删除: {key}")
                    return True

            return False

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"删除配置失败 {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查配置项是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """
        self._ensure_initialized()

        try:
            self._stats['operations'] += 1

            with self._config_lock:
                return key in self._config_data

        except Exception as e:
            logger.error(f"检查配置存在性失败 {key}: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的配置键

        Args:
            pattern: 匹配模式

        Returns:
            配置键列表
        """
        self._ensure_initialized()

        try:
            with self._config_lock:
                if pattern == "*":
                    return list(self._config_data.keys())
                else:
                    # 简单模式匹配实现
                    return [k for k in self._config_data.keys() if pattern in k]

        except Exception as e:
            logger.error(f"获取配置键失败 {pattern}: {e}")
            return []

    def clear(self) -> bool:
        """清空所有配置

        Returns:
            清空是否成功
        """
        self._ensure_initialized()

        try:
            with self._config_lock:
                self._config_data.clear()

            logger.info("配置已清空")
            return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"清空配置失败: {e}")
            return False

    # ========== 存储管理接口 ==========

    def load_config(self, source: str) -> Dict[str, Any]:
        """从指定源加载配置"""
        try:
            # 使用存储服务加载
            config = self._storage_service.load(source)

            # 合并到本地存储
            with self._config_lock:
                self._config_data.update(config)

            logger.info(f"配置已加载: {source}")
            return config

        except Exception as e:
            logger.error(f"加载配置失败 {source}: {e}")
            raise

    def save_config(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置到指定目标"""
        try:
            return self._storage_service.save(config, target)

        except Exception as e:
            logger.error(f"保存配置失败 {target}: {e}")
            raise

    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            return self._storage_service.reload()

        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
            raise

    # ========== 服务管理接口 ==========

    def add_validator(self, validator: Callable) -> None:
        """添加验证器"""
        self._operations_service.add_validator(validator)

    def add_listener(self, listener: Callable) -> None:
        """添加变更监听器"""
        self._operations_service.add_listener(listener)

    # ========== 监控和统计接口 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'basic': dict(self._stats),
            'storage': self._storage_service.get_storage_stats(),
            'operations': self._operations_service.get_operation_stats(),
            'config_count': len(self._config_data),
            'uptime': time.time() - (self._start_time or time.time())
        }

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            return {
                'healthy': True,
                'services': {
                    'storage': {'healthy': self._storage_service._initialized},
                    'operations': {'healthy': self._operations_service._initialized}
                },
                'config_count': len(self._config_data),
                'uptime': time.time() - (self._start_time or time.time()),
                'timestamp': time.time()
            }

        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }

    # ========== 批量操作接口 ==========

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._config_lock:
            return self._config_data.copy()

    # ========== 兼容性接口 ==========

    # 为保持兼容性，提供别名
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置（兼容性方法）"""
        return self.get(key, default)

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置（兼容性方法）"""
        return self.set(key, value)

    def load(self, source: str) -> Dict[str, Any]:
        """加载配置（兼容性方法）"""
        return self.load_config(source)

    def save(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置（兼容性方法）"""
        return self.save_config(config, target)

    # ========== 清理接口 ==========

    def cleanup(self):
        """清理资源"""
        try:
            self._operations_service.cleanup()
            self._storage_service.cleanup()

            with self._config_lock:
                self._config_data.clear()

            logger.info(f"{self._service_name} 已清理")

        except Exception as e:
            logger.error(f"清理失败: {e}")

    # ========== 内部方法 ==========

    def _get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            'service_name': self._service_name,
            'version': '2.0',
            'architecture': 'composition_pattern',
            'config_count': len(self._config_data),
            'initialized': self._initialized,
            'start_time': self._start_time
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"ConfigManagerV2(configs={len(self._config_data)}, initialized={self._initialized})"

    def __repr__(self) -> str:
        """详细表示"""
        return (f"ConfigManagerV2("
                f"service='{self._service_name}', "
                f"configs={len(self._config_data)}, "
                f"initialized={self._initialized})")

    # ========== 类方法 ==========

    @classmethod
    def create_with_file_storage(cls, config_file: str, **kwargs) -> 'ConfigManagerV2':
        """创建带有文件存储的配置管理器"""
        try:
            storage_config = StorageConfig(
                type=StorageType.FILE,
                path=config_file
            )

            storage_backend = FileConfigStorage(storage_config)
            return cls(storage_backend=storage_backend, **kwargs)

        except Exception as e:
            logger.error(f"创建文件存储配置管理器失败: {e}")
            # 返回无存储后端的实例
            return cls(**kwargs)

    @classmethod
    def create_simple(cls, **kwargs) -> 'ConfigManagerV2':
        """创建简单的配置管理器（仅内存存储）"""
        return cls(cache_enabled=False, **kwargs)




