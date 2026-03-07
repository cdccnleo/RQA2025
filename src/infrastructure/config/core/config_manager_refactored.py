
#!/usr/bin/env python3
"""
重构后的统一配置管理器

Phase 1重构：使用组合模式替代多重继承
消除循环依赖，提高代码可维护性
"""

import logging
import time

from .common_mixins import BatchOperationsMixin
from .imports import (
    Dict, Any, List
)
from ..services.config_operations_service import ConfigOperationsService
from ..services.config_storage_service import ConfigStorageService
from ..services.iconfig_service import BaseConfigService
logger = logging.getLogger(__name__)


class UnifiedConfigManager(BaseConfigService, BatchOperationsMixin):
    """重构后的统一配置管理器

    使用组合模式替代多重继承，消除循环依赖问题
    """

    def __init__(self, storage_backend=None, cache_enabled: bool = True):
        """初始化配置管理器

        Args:
            storage_backend: 存储后端实现
            cache_enabled: 是否启用缓存
        """
        super().__init__("unified_config_manager")

        # 组合各个服务
        self._storage_service = ConfigStorageService(cache_enabled=cache_enabled)
        self._operations_service = ConfigOperationsService(self._storage_service)

        # 如果提供了存储后端，设置它
        if storage_backend:
            self._storage_service.set_storage_backend(storage_backend)

        # 监控和统计
        self._performance_stats = {
            'total_operations': 0,
            'cache_hit_rate': 0.0,
            'error_count': 0,
            'uptime': 0
        }

    def _initialize(self):
        """初始化管理器"""
        self._start_time = time.time()
        logger.info("统一配置管理器已初始化（组合模式）")

    # ========== 配置操作接口 ==========

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            return self._operations_service.get(key, default)
        except Exception as e:
            # 返回错误信息而不是抛出异常
            return {"error": str(e), "key": key, "default": default}

    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        return self._operations_service.set(key, value)

    def delete(self, key: str) -> bool:
        """删除配置项"""
        return self._operations_service.delete(key)

    def exists(self, key: str) -> bool:
        """检查配置项是否存在"""
        return self._operations_service.exists(key)

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的配置键"""
        return self._operations_service.keys(pattern)

    def clear(self) -> bool:
        """清空所有配置"""
        return self._operations_service.clear()

    # ========== 存储管理接口 ==========

    def load_config(self, source: str) -> Dict[str, Any]:
        """从指定源加载配置"""
        return self._storage_service.load(source)

    def save_config(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置到指定目标"""
        return self._storage_service.save(config, target)

    def reload_config(self) -> bool:
        """重新加载配置"""
        return self._storage_service.reload()

    def set_storage_backend(self, storage_backend):
        """设置存储后端"""
        self._storage_service.set_storage_backend(storage_backend)

    # ========== 服务管理接口 ==========

    def add_validator(self, validator):
        """添加验证器"""
        self._operations_service.add_validator(validator)

    def add_listener(self, listener):
        """添加变更监听器"""
        self._operations_service.add_listener(listener)

    def add_preprocessor(self, preprocessor):
        """添加预处理器"""
        self._operations_service.add_preprocessor(preprocessor)

    def add_postprocessor(self, postprocessor):
        """添加后处理器"""
        self._operations_service.add_postprocessor(postprocessor)

    # ========== 监控和统计接口 ==========

    def get_operation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取操作历史"""
        return self._operations_service.get_operation_history(limit)

    def get_operation_stats(self) -> Dict[str, Any]:
        """获取操作统计信息"""
        ops_stats = self._operations_service.get_operation_stats()
        storage_stats = self._storage_service.get_storage_stats()

        return {
            'operations': ops_stats,
            'storage': storage_stats,
            'performance': self._performance_stats,
            'services': {
                'storage': self._storage_service.get_service_info(),
                'operations': self._operations_service.get_service_info()
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            # 检查各个服务的状态
            storage_healthy = self._storage_service._initialized
            operations_healthy = self._operations_service._initialized

            overall_healthy = storage_healthy and operations_healthy

            return {
                'healthy': overall_healthy,
                'services': {
                    'storage': {'healthy': storage_healthy},
                    'operations': {'healthy': operations_healthy}
                },
                'uptime': time.time() - (self._start_time or time.time()),
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }

    # ========== 兼容性接口 ==========

    # 保持与原接口的兼容性
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

    # ========== 高级功能接口 ==========

    def get_config_snapshot(self) -> Dict[str, Any]:
        """获取配置快照"""
        # 获取所有配置键
        all_keys = self.keys()
        return self.batch_get(all_keys)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置字典"""
        results = {'valid': True, 'errors': []}

        for key, value in config.items():
            # 这里可以添加更复杂的验证逻辑
            if not isinstance(key, str):
                results['errors'].append(f"键必须是字符串: {key}")
                results['valid'] = False

        return results

    # ========== 清理和维护接口 ==========

    def cleanup(self):
        """清理资源"""
        try:
            self._operations_service.cleanup()
            self._storage_service.cleanup()
            logger.info("配置管理器已清理")
        except Exception as e:
            logger.error(f"清理失败: {e}")

    def reset(self):
        """重置管理器状态"""
        try:
            self.cleanup()
            # 重新初始化
            self._initialize()
            logger.info("配置管理器已重置")
        except Exception as e:
            logger.error(f"重置失败: {e}")
            raise

    # ========== 内部方法 ==========

    def _update_performance_stats(self):
        """更新性能统计"""
        try:
            stats = self.get_operation_stats()
            total_ops = sum(stats['operations']['operations'].values())

            self._performance_stats.update({
                'total_operations': total_ops,
                'uptime': time.time() - (self._start_time or time.time())
            })

            # 计算缓存命中率
            storage_stats = stats['storage']
            cache_hits = storage_stats.get('cache_hits', 0)
            cache_misses = storage_stats.get('cache_misses', 0)
            total_cache_reqs = cache_hits + cache_misses

            if total_cache_reqs > 0:
                self._performance_stats['cache_hit_rate'] = cache_hits / total_cache_reqs

        except Exception as e:
            logger.warning(f"更新性能统计失败: {e}")

    def __str__(self) -> str:
        """字符串表示"""
        return f"UnifiedConfigManager(services=2, initialized={self._initialized})"

    def __repr__(self) -> str:
        """详细表示"""
        return (f"UnifiedConfigManager("
                f"storage={self._storage_service.__class__.__name__}, "
                f"operations={self._operations_service.__class__.__name__}, "
                f"initialized={self._initialized})")




