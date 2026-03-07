"""
基础设施层 - RQA2025量化交易系统核心支撑

提供企业级的配置管理、缓存系统、安全监控、日志处理等基础设施服务。
采用统一接口设计，支持模块化扩展和高可用部署。

版本: 2.1.0
"""

import logging
from typing import Optional, Any

__version__ = "2.1.0"
__author__ = "RQA2025 Team"

# 条件导入核心组件
try:
    from .config.core.config_manager_complete import UnifiedConfigManager
except Exception:
    UnifiedConfigManager = None

try:
    from .cache.core.cache_manager import UnifiedCacheManager
    # 别名：BaseCacheManager = UnifiedCacheManager（向后兼容）
    BaseCacheManager = UnifiedCacheManager
except Exception:
    UnifiedCacheManager = None
    BaseCacheManager = None

try:
    from .init_infrastructure import InfrastructureInitializer
except Exception:
    InfrastructureInitializer = None

try:
    from .health import EnhancedHealthChecker
except Exception:
    EnhancedHealthChecker = None

try:
    from .logging.core.unified_logging_interface import UnifiedLogger
except Exception:
    UnifiedLogger = None

# 配置延迟导入，避免循环依赖
_config_manager = None
_cache_manager = None
_health_checker = None
_monitor = None


def get_config_manager() -> Optional[Any]:
    """延迟加载配置管理器"""
    global _config_manager
    if _config_manager is None:
        try:
            _config_manager = UnifiedConfigManager()
        except ImportError:
            logging.warning("UnifiedConfigManager不可用")
            _config_manager = None
    return _config_manager


def get_cache_manager() -> Optional[Any]:
    """延迟加载缓存管理器"""
    global _cache_manager
    if _cache_manager is None:
        try:
            _cache_manager = UnifiedCacheManager()
        except ImportError:
            logging.warning("UnifiedCacheManager不可用")
            _cache_manager = None
    return _cache_manager


def get_health_checker() -> Optional[Any]:
    """延迟加载健康检查器"""
    global _health_checker
    if _health_checker is None:
        try:
            _health_checker = EnhancedHealthChecker()
        except ImportError:
            logging.warning("EnhancedHealthChecker不可用")
            _health_checker = None
    return _health_checker


def get_monitor() -> Optional[Any]:
    """延迟加载监控器"""
    global _monitor
    if _monitor is None:
        try:
            _monitor = UnifiedLogger()
        except ImportError:
            logging.warning("UnifiedLogger不可用")
            _monitor = None
    return _monitor

# 便捷工厂函数


def create_config_manager(**kwargs):
    """创建配置管理器"""
    manager_class = get_config_manager()
    return manager_class(**kwargs) if manager_class else None


def create_cache_manager(**kwargs):
    """创建缓存管理器"""
    manager_class = get_cache_manager()
    return manager_class(**kwargs) if manager_class else None


def create_health_checker(**kwargs):
    """创建健康检查器"""
    checker_class = get_health_checker()
    return checker_class(**kwargs) if checker_class else None


# 兼容性别名
ConfigManager = get_config_manager
CacheManager = get_cache_manager
HealthChecker = get_health_checker
Monitor = get_monitor

# 导入SystemMonitor
try:
    from .monitoring.system_monitor import SystemMonitor
except ImportError:
    SystemMonitor = None

# 导入LRUCache（如果需要）
try:
    from .cache.core.cache_manager import UnifiedCacheManager
    # 提供LRUCache别名（如果需要）
    LRUCache = UnifiedCacheManager
except Exception:
    LRUCache = None

# 为保持基础设施层独立性，这里提供最小化的容器占位实现，避免直接导入核心服务层。
class _InfrastructureContainerStub:
    def __init__(self, *_, **__):
        raise RuntimeError("DependencyContainer 不可用：基础设施层不直接依赖核心服务层")

    def has(self, *_args, **_kwargs) -> bool:
        return False

    def get(self, *_args, **_kwargs):
        raise KeyError("Service not registered in infrastructure container stub")


UnifiedContainer = None
BaseContainer = None

# 导入MonitorFactory
try:
    from .monitoring.monitor_factory import MonitorFactory
except Exception:
    try:
        from .monitoring import MonitorFactory
    except Exception:
        # 提供基础实现
        class MonitorFactory:
            @staticmethod
            def create_monitor(monitor_type: str = "system"):
                return None
        
        MonitorFactory = MonitorFactory


def get_default_monitor():
    """获取默认监控器"""
    return SystemMonitor() if SystemMonitor else None
