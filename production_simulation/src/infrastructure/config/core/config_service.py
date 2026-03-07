
# 导入CacheService

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging
import threading
import time
from ..interfaces.unified_interface import ServiceStatus
from ..services.cache_service import CacheService
#!/usr/bin/env python3
"""
统一配置服务 (优化版)

整合所有配置服务功能，提供统一的配置服务框架
合并了config_service.py, config_service_components.py, unified_service.py的功能

支持:
- 配置加载和管理
- 缓存服务集成
- 验证器集成
- 热重载功能
- 服务组件化架构
- 性能监控
- 错误处理和恢复
"""

logger = logging.getLogger(__name__)

# ==================== 枚举定义 ====================


class ServiceHealth(Enum):
    """服务健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

# ==================== 服务接口 ====================


class IConfigServiceComponent(ABC):
    """配置服务组件接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""
        ...

    @abstractmethod
    def start(self) -> bool:
        """启动组件"""
        ...

    @abstractmethod
    def stop(self) -> bool:
        """停止组件"""
        ...

    @abstractmethod
    def get_status(self) -> ServiceStatus:
        """获取组件状态"""
        ...

    @abstractmethod
    def get_health(self) -> ServiceHealth:
        """获取组件健康状态"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称"""
        ...


class IConfigService(ABC):
    """配置服务接口"""

    @abstractmethod
    def load_config(self, config_path: str) -> bool:
        """加载配置"""
        ...

    @abstractmethod
    def reload_config(self) -> bool:
        """重载配置"""
        ...

    @abstractmethod
    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置"""
        ...

    @abstractmethod
    def set_config(self, key: str, value: Any) -> bool:
        """设置配置"""
        ...

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        ...

    @abstractmethod
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        ...

# ==================== 缓存服务 ====================
# 注意：缓存服务已移动到services/cache_service.py中，但这里需要导入以保持兼容性

# ==================== 统一配置服务 ====================


class UnifiedConfigService(IConfigService, IConfigServiceComponent):
    """
    统一配置服务

    整合所有配置服务功能：
    - 配置加载和管理
    - 缓存集成
    - 验证器集成
    - 热重载
    - 性能监控
    - 健康检查
    """

    def __init__(self):
        self._status = ServiceStatus.INITIALIZING
        self._health = ServiceHealth.HEALTHY

        # 核心组件
        self._config: Dict[str, Any] = {}
        self._loaders: Dict[str, Any] = {}
        self._validators: List[Any] = []
        # 确保缓存服务正确初始化
        try:
            self._cache_service = CacheService()
            if self._cache_service:
                self._cache_service.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize cache service: {e}")
            self._cache_service = None

        # 服务管理
        self._config_path: Optional[str] = None
        self._last_reload_time = 0
        self._start_time = time.time()

        # 线程管理
        self._running = False
        self._lock = threading.RLock()

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化服务"""
        try:
            with self._lock:
                self._config.update(config)
                self._status = ServiceStatus.RUNNING
                self._health = ServiceHealth.HEALTHY
                logger.info("UnifiedConfigService initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            self._status = ServiceStatus.ERROR
            self._health = ServiceHealth.UNHEALTHY
            return False

    def start(self) -> bool:
        """启动服务"""
        try:
            with self._lock:
                if self._status == ServiceStatus.RUNNING:
                    return True

                self._running = True
                self._status = ServiceStatus.RUNNING
                self._health = ServiceHealth.HEALTHY
                logger.info("UnifiedConfigService started")
                return True
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self._status = ServiceStatus.ERROR
            self._health = ServiceHealth.UNHEALTHY
            return False

    def stop(self) -> bool:
        """停止服务"""
        try:
            with self._lock:
                self._running = False
                self._status = ServiceStatus.STOPPED
                logger.info("UnifiedConfigService stopped")
                return True
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            return False

    def load_config(self, config_path: str) -> bool:
        """加载配置文件"""
        try:
            with self._lock:
                self._config_path = config_path

                if Path(config_path).exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                        self._config.update(loaded_config)
                        self._last_reload_time = time.time()
                        logger.info(f"Configuration loaded from {config_path}")
                        return True
                else:
                    logger.warning(f"Configuration file not found: {config_path}")
                    return False
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def reload_config(self) -> bool:
        """重载配置"""
        if not self._config_path:
            logger.warning("No configuration path set for reload")
            return False
        return self.load_config(self._config_path)

    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置"""
        try:
            with self._lock:
                if key is None:
                    return self._config.copy()
                else:
                    # 尝试从缓存获取
                    if self._cache_service:
                        cached_value = self._cache_service.get(key)
                        if cached_value is not None:
                            return cached_value

                    # 从配置获取
                    result = self._get_nested_value(self._config, key)

                    # 缓存结果
                    if result is not None and self._cache_service:
                        self._cache_service.set(key, result, ttl=300)

                    return result
        except Exception as e:
            logger.error(f"Failed to get config for key {key}: {e}")
            return None

    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        # 首先尝试直接查找（处理包含点号的键）
        if key in config:
            return config[key]

        # 如果找不到，则尝试嵌套查找
        keys = key.split('.')
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置"""
        try:
            with self._lock:
                self._config[key] = value

                # 清除相关缓存
                if self._cache_service:
                    self._cache_service.delete(key)

                return True
        except Exception as e:
            logger.error(f"Failed to set config for key {key}: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        try:
            result = {'is_valid': True, 'errors': [], 'warnings': []}

            for validator in self._validators:
                try:
                    validator_result = validator.validate(config)
                    if hasattr(validator_result, 'is_valid') and not validator_result.is_valid:
                        result['is_valid'] = False
                        result['errors'].extend(getattr(validator_result, 'errors', []))
                    if hasattr(validator_result, 'warnings'):
                        result['warnings'].extend(validator_result.warnings)
                except Exception as e:
                    result['is_valid'] = False
                    result['errors'].append(str(e))

            return result
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': []}

    def register_loader(self, name: str, loader: Any) -> None:
        """注册配置加载器"""
        with self._lock:
            self._loaders[name] = loader
            logger.info(f"Registered config loader: {name}")

    def register_validator(self, validator: Any) -> None:
        """注册配置验证器"""
        with self._lock:
            self._validators.append(validator)
            logger.info(
                f"Registered config validator: {validator.name if hasattr(validator, 'name') else str(validator)}")

    def get_status(self) -> ServiceStatus:
        """获取服务状态"""
        return self._status

    def get_health(self) -> ServiceHealth:
        """获取服务健康状态"""
        return self._health

    @property
    def name(self) -> str:
        """服务名称"""
        return "UnifiedConfigService"

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态详情"""
        with self._lock:
            return {
                'status': self._status.value,
                'health': self._health.value,
                'uptime_seconds': int(time.time() - self._start_time),
                'config_path': self._config_path,
                'last_reload_time': self._last_reload_time,
                'loaders_count': len(self._loaders),
                'validators_count': len(self._validators),
                'cache_enabled': self._cache_service is not None,
                'config_keys_count': len(self._config)
            }

# ==================== 服务工厂 ====================


class ConfigServiceFactory:
    """配置服务工厂"""

    def __init__(self):
        self._service_classes: Dict[str, type] = {
            'unified': UnifiedConfigService
        }

    def create_service(self, service_type: str = 'unified') -> IConfigService:
        """
        创建配置服务

        Args:
            service_type: 服务类型

        Returns:
            配置服务实例
        """
        if service_type not in self._service_classes:
            raise ValueError(f"Unknown service type: {service_type}")

        service_class = self._service_classes[service_type]
        service = service_class()

        # 初始化服务
        init_config = {'service_name': service_type}
        if service.initialize(init_config):
            service.start()
            return service
        else:
            raise RuntimeError(f"Failed to initialize {service_type} service")

    def register_service_type(self, service_type: str, service_class: type) -> None:
        """
        注册服务类型

        Args:
            service_type: 服务类型名称
            service_class: 服务类
        """
        if not issubclass(service_class, IConfigService):
            raise ValueError("Service class must implement IConfigService interface")

        self._service_classes[service_type] = service_class

# ==================== 便捷函数 ====================


def create_config_service(service_type: str = 'unified') -> IConfigService:
    """创建配置服务 (便捷函数)"""
    factory = ConfigServiceFactory()
    return factory.create_service(service_type)

# ==================== 向后兼容性 ====================


class ConfigService:
    """向后兼容的配置服务类"""

    def __init__(self, cache_service: Optional[CacheService] = None):
        """
        初始化配置服务 (向后兼容)

        Args:
            cache_service: 缓存服务实例
        """
        self._unified_service = UnifiedConfigService()
        self.cache_service = cache_service or CacheService()

        # 初始化兼容性属性
        self._config = {}
        self._loaders = {}
        self._validators = []
        self._config_path = None

    def register_loader(self, name: str, loader: Any) -> None:
        """注册加载器 (向后兼容)"""
        self._unified_service.register_loader(name, loader)

    def register_validator(self, validator: Any) -> None:
        """注册验证器 (向后兼容)"""
        self._unified_service.register_validator(validator)

    def load_config(self, config_path: str) -> bool:
        """加载配置 (向后兼容)"""
        return self._unified_service.load_config(config_path)

    def reload_config(self) -> bool:
        """重载配置 (向后兼容)"""
        return self._unified_service.reload_config()

    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置 (向后兼容)"""
        return self._unified_service.get_config(key)

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置 (向后兼容)"""
        return self._unified_service.set_config(key, value)

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置 (向后兼容)"""
        return self._unified_service.validate_config(config)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态 (向后兼容)"""
        return self._unified_service.get_service_status()




