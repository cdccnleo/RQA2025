"""
健康基础设施管理

管理健康检查相关的基础设施服务。
"""

import logging
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from src.infrastructure.logging.core.unified_logger import get_unified_logger

# 可选导入基础设施服务
try:
    from src.infrastructure.cache.core.unified_cache import UnifiedCacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    UnifiedCacheManager = None

try:
    from src.infrastructure.health.core.enhanced_health_checker import EnhancedHealthChecker
    HEALTH_CHECKER_AVAILABLE = True
except ImportError:
    HEALTH_CHECKER_AVAILABLE = False
    EnhancedHealthChecker = None

try:
    from src.infrastructure.monitoring.core.unified_monitoring import UnifiedMonitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    UnifiedMonitoring = None

try:
    from src.infrastructure.config.core.unified_config import UnifiedConfigManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    UnifiedConfigManager = None

logger = get_unified_logger(__name__)


@dataclass
class HealthInfrastructureConfig:
    """健康基础设施配置"""
    enable_cache: bool = True
    enable_monitoring: bool = True
    enable_config_manager: bool = True
    enable_health_checker: bool = True
    max_workers: int = 10


class HealthInfrastructureManagerImpl:
    """健康基础设施管理器实现 - 职责：管理基础设施服务"""

    def __init__(self, config: HealthInfrastructureConfig):
        self.config = config
        self._cache_manager: Optional[Any] = None
        self._monitoring: Optional[Any] = None
        self._config_manager: Optional[Any] = None
        self._health_checker: Optional[Any] = None
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._logger = get_unified_logger(f"{__name__}.HealthInfrastructureManager")

    def initialize_infrastructure(self) -> Dict[str, Any]:
        """初始化基础设施服务"""
        try:
            results = {}

            # 可选初始化基础设施服务
            if self.config.enable_cache and CACHE_AVAILABLE:
                self._cache_manager = UnifiedCacheManager()
                results['cache_manager'] = 'initialized'
                self._logger.info("缓存管理器初始化成功")
            else:
                self._cache_manager = None
                results['cache_manager'] = 'unavailable'

            if self.config.enable_monitoring and MONITORING_AVAILABLE:
                self._monitoring = UnifiedMonitoring()
                results['monitoring'] = 'initialized'
                self._logger.info("监控服务初始化成功")
            else:
                self._monitoring = None
                results['monitoring'] = 'unavailable'

            if self.config.enable_config_manager and CONFIG_AVAILABLE:
                self._config_manager = UnifiedConfigManager()
                results['config_manager'] = 'initialized'
                self._logger.info("配置管理器初始化成功")
            else:
                self._config_manager = None
                results['config_manager'] = 'unavailable'

            if self.config.enable_health_checker and HEALTH_CHECKER_AVAILABLE:
                self._health_checker = EnhancedHealthChecker()
                results['health_checker'] = 'initialized'
                self._logger.info("健康检查器初始化成功")
            else:
                self._health_checker = None
                results['health_checker'] = 'unavailable'

            return results

        except Exception as e:
            self._logger.error(f"基础设施初始化失败: {e}")
            return {'error': str(e)}

    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        return {
            'cache_manager': self._cache_manager,
            'monitoring': self._monitoring,
            'config_manager': self._config_manager,
            'health_checker': self._health_checker,
            'executor': self._executor,
            'logger': self._logger
        }

