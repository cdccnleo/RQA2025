"""
cache_factory 模块

提供 cache_factory 相关功能和接口。
"""

import logging
import threading
from typing import Any, Dict, Optional

from . import cache_configs as cache_configs_module
from .cache_manager import UnifiedCacheManager

"""
RQA2025 Cache Factory

缓存工厂实现
"""

logger = logging.getLogger(__name__)


class CacheFactory:

    """缓存工厂"""

    _instances = {}
    _lock = threading.Lock()  # 保护实例字典的线程锁

    def __init__(self):
        """初始化缓存工厂"""
        self.cache_types = {
            'memory': '内存缓存',
            'redis': 'Redis缓存',
            'file': '文件缓存'
        }
        self.configurations = {}

    def create_cache(self, cache_type: str = 'memory', **kwargs):
        """创建缓存实例"""
        try:
            # 导入配置类
            config_dict = {
                'enabled': kwargs.get('enabled', True),
                'max_size': kwargs.get('max_size', 1000),
                'ttl': kwargs.get('ttl', 3600),
                'strict_validation': kwargs.get('strict_validation', False),
                'basic': {
                    'max_size': kwargs.get('max_size', 1000),
                    'ttl': kwargs.get('ttl', 3600)
                },
                'multi_level': {},
                'advanced': {},
                'smart': {},
                'distributed': {}
            }

            # 根据缓存类型设置级别
            if cache_type == 'redis':
                config_dict['multi_level']['level'] = 'redis'
            elif cache_type == 'file':
                config_dict['multi_level']['level'] = 'file'
            elif cache_type == 'hybrid':
                config_dict['multi_level']['level'] = 'hybrid'
            else:
                config_dict['multi_level']['level'] = 'memory'

            # 配置Redis参数
            if cache_type in ['redis', 'hybrid']:
                config_dict['distributed']['distributed'] = True
                config_dict['distributed']['redis_host'] = kwargs.get('host', 'localhost')
                config_dict['distributed']['redis_port'] = kwargs.get('port', 6379)
            else:
                config_dict['distributed']['distributed'] = False

            # 使用from_dict创建配置
            config = cache_configs_module.CacheConfig.from_dict(config_dict)

            return UnifiedCacheManager(config)

        except Exception as e:
            logger.error(f"创建缓存失败: {e}")
            # 返回默认配置的缓存管理器
            return UnifiedCacheManager()

    def add_configuration(self, config_name: str, config_data: Dict[str, Any]):
        """添加配置"""
        self.configurations[config_name] = config_data

    def get_configuration(self, config_name: str) -> Dict[str, Any]:
        """获取配置"""
        return self.configurations.get(config_name, {})

    @classmethod
    def create_cache_manager(cls, cache_type: str = 'memory', config: Optional[Dict[str, Any]] = None, **kwargs) -> UnifiedCacheManager:
        """创建缓存管理器"""
        config_data: Dict[str, Any] = dict(config or {})
        config_data.update(kwargs)
        basic_dict = config_data.setdefault('basic', {})
        if 'max_size' in config_data:
            basic_dict.setdefault('max_size', config_data['max_size'])
        if 'ttl' in config_data:
            basic_dict.setdefault('ttl', config_data['ttl'])
        multi_level_dict = config_data.setdefault('multi_level', {})
        if 'level' not in multi_level_dict:
            if cache_type == 'redis':
                multi_level_dict['level'] = 'redis'
            elif cache_type == 'file':
                multi_level_dict['level'] = 'file'
            elif cache_type == 'hybrid':
                multi_level_dict['level'] = 'hybrid'
            else:
                multi_level_dict['level'] = 'memory'
        if cache_type in {'redis', 'hybrid'}:
            distributed_dict = config_data.setdefault('distributed', {})
            distributed_dict.setdefault('distributed', True)
            distributed_dict.setdefault('redis_host', config_data.get('host', 'localhost'))
            distributed_dict.setdefault('redis_port', config_data.get('port', 6379))
        config_data.setdefault('strict_validation', config_data.get('strict_validation', False))

        try:
            cache_config = cache_configs_module.CacheConfig.from_dict(config_data)
            return UnifiedCacheManager(cache_config)
        except Exception as e:
            logger.error(f"创建缓存管理器失败: {e}")
            return UnifiedCacheManager()

    @classmethod
    def create_cache_service(cls, config: Optional[Dict[str, Any]] = None) -> UnifiedCacheManager:
        """创建缓存服务"""
        try:
            config_data = dict(config or {})
            config_data.setdefault('strict_validation', config_data.get('strict_validation', False))
            cache_config = cache_configs_module.CacheConfig.from_dict(config_data)
            service = UnifiedCacheManager(cache_config)
            return service
        except Exception as e:
            logger.error(f"创建缓存服务失败: {e}")
            return UnifiedCacheManager()

    @classmethod
    def get_cache_service(cls, service_name: str = 'default', config: Optional[Dict[str, Any]] = None) -> UnifiedCacheManager:
        """获取缓存服务实例（单例模式）"""
        # 双重检查锁定模式，确保线程安全
        # 单例模式只使用第一次调用时的配置，后续调用忽略配置参数
        if service_name not in cls._instances:
            with cls._lock:
                # 再次检查，避免在锁等待期间其他线程已创建实例
                if service_name not in cls._instances:
                    cls._instances[service_name] = cls.create_cache_service(config)

        return cls._instances[service_name]
