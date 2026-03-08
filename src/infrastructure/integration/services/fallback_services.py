#!/usr/bin/env python3
"""
RQA2025 基础设施降级服务

当基础设施服务不可用时，提供降级的备用实现，
确保系统能够继续运行并提供基本功能。
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging
import time

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = logging.getLogger(__name__)


class FallbackService(ABC):

    """降级服务基类"""

    def __init__(self, service_name: str):

        self.service_name = service_name
        self._logger = get_unified_logger(f"fallback_{service_name}")

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'service_name': self.service_name,
            'status': 'degraded' if not self.is_available() else 'healthy',
            'service_type': 'fallback',
            'timestamp': '2025 - 01 - 27T10:00:00Z'
        }


class FallbackConfigManager(FallbackService):

    """降级配置管理器"""

    def __init__(self):

        super().__init__('config_manager')
        self._configs: Dict[str, Any] = {}

    def is_available(self) -> bool:

        return True  # 降级配置管理器总是可用的

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._configs.get(key, default)

    def set_config(self, key: str, value: Any) -> bool:
        """设置配置值"""
        try:
            self._configs[key] = value
            return True
        except Exception as e:
            self._logger.error(f"设置配置失败: {e}")
            return False

    def has_config(self, key: str) -> bool:
        """检查配置是否存在"""
        return key in self._configs

    def delete_config(self, key: str) -> bool:
        """删除配置"""
        try:
            if key in self._configs:
                del self._configs[key]
                return True
            return False
        except Exception as e:
            self._logger.error(f"删除配置失败: {e}")
            return False

    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._configs.copy()


class FallbackCacheManager(FallbackService):

    """降级缓存管理器"""

    def __init__(self):

        super().__init__('cache_manager')
        self._cache: Dict[str, Dict[str, Any]] = {}

    def is_available(self) -> bool:

        return True  # 降级缓存管理器总是可用的

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值（兼容接口）"""
        return self.get_cache(key, default)

    def get_cache(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                return entry['value']
            else:
                # 删除过期条目
                del self._cache[key]
        return default

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值（兼容接口）"""
        return self.set_cache(key, value, ttl)

    def set_cache(self, key: str, value: Any, expire: int = 3600) -> bool:
        """设置缓存值"""
        try:
            self._cache[key] = {
                'value': value,
                'expire_time': time.time() + expire,
                'created_at': time.time()
            }
            return True
        except Exception as e:
            self._logger.error(f"设置缓存失败: {e}")
            return False

    def delete_cache(self, key: str) -> bool:
        """删除缓存"""
        try:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
        except Exception as e:
            self._logger.error(f"删除缓存失败: {e}")
            return False

    def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            self._cache.clear()
            return True
        except Exception as e:
            self._logger.error(f"清空缓存失败: {e}")
            return False

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """检查条目是否过期"""
        return time.time() > entry['expire_time']


class FallbackLogger(FallbackService):

    """降级日志器"""

    def __init__(self):

        super().__init__('logger')
        self._logs: List[Dict[str, Any]] = []

    def is_available(self) -> bool:

        return True  # 降级日志器总是可用的

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """调试日志"""
        self._log('DEBUG', message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """信息日志"""
        self._log('INFO', message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """警告日志"""
        self._log('WARNING', message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """错误日志"""
        self._log('ERROR', message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """严重错误日志"""
        self._log('CRITICAL', message, extra)

    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录日志"""
        try:
            log_entry = {
                'timestamp': '2025 - 01 - 27T10:00:00Z',
                'level': level,
                'message': message,
                'extra': extra or {},
                'service': 'fallback_logger'
            }
            self._logs.append(log_entry)

            # 同时输出到标准输出（用于调试）
            print(f"[FALLBACK {level}] {message}")

            # 限制日志条目数量
            if len(self._logs) > 1000:
                self._logs = self._logs[-500:]  # 保留最新的500条

        except Exception as e:
            print(f"[FALLBACK ERROR] 日志记录失败: {e}")

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取日志"""
        return self._logs[-limit:]


class FallbackMonitoring(FallbackService):

    """降级监控器"""

    def __init__(self):

        super().__init__('monitoring')
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}

    def is_available(self) -> bool:

        return True  # 降级监控器总是可用的

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录指标"""
        try:
            if name not in self._metrics:
                self._metrics[name] = []

            metric_entry = {
                'timestamp': '2025 - 01 - 27T10:00:00Z',
                'name': name,
                'value': value,
                'tags': tags or {}
            }

            self._metrics[name].append(metric_entry)

            # 限制每个指标的条目数量
            if len(self._metrics[name]) > 100:
                self._metrics[name] = self._metrics[name][-50:]

        except Exception as e:
            self._logger.error(f"记录指标失败: {e}")

    def record_alert(self, level: str, message: str, tags: Optional[Dict[str, str]] = None):
        """记录告警"""
        try:
            alert_entry = {
                'timestamp': '2025 - 01 - 27T10:00:00Z',
                'level': level,
                'message': message,
                'tags': tags or {},
                'service': 'fallback_monitoring'
            }

            # 记录为特殊指标
            if 'alerts' not in self._metrics:
                self._metrics['alerts'] = []

            self._metrics['alerts'].append(alert_entry)

            # 同时输出到标准输出
            print(f"[FALLBACK ALERT {level}] {message}")

        except Exception as e:
            self._logger.error(f"记录告警失败: {e}")

    def get_metrics(self, name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取指标"""
        return self._metrics.get(name, [])[-limit:]


class FallbackHealthChecker(FallbackService):

    """降级健康检查器"""

    def __init__(self):

        super().__init__('health_checker')
        self._health_status = 'healthy'

    def is_available(self) -> bool:

        return True  # 降级健康检查器总是可用的

    def check_health(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """检查健康状态"""
        return {
            'service_name': service_name or 'fallback_health_checker',
            'status': self._health_status,
            'timestamp': '2025 - 01 - 27T10:00:00Z',
            'service_type': 'fallback',
            'checks': {
                'memory': {'status': 'healthy', 'usage': 'N / A'},
                'cpu': {'status': 'healthy', 'usage': 'N / A'},
                'disk': {'status': 'healthy', 'usage': 'N / A'}
            }
        }

    def set_health_status(self, status: str):
        """设置健康状态"""
        self._health_status = status


# 全局降级服务实例
_fallback_services = {
    'config_manager': FallbackConfigManager(),
    'cache_manager': FallbackCacheManager(),
    'logger': FallbackLogger(),
    'monitoring': FallbackMonitoring(),
    'health_checker': FallbackHealthChecker()
}


def get_fallback_service(service_name: str) -> Optional[FallbackService]:
    """获取降级服务"""
    return _fallback_services.get(service_name)


def get_all_fallback_services() -> Dict[str, FallbackService]:
    """获取所有降级服务"""
    return _fallback_services.copy()


def health_check_fallback_services() -> Dict[str, Any]:
    """检查所有降级服务的健康状态"""
    overall_health = {
        'timestamp': '2025 - 01 - 27T10:00:00Z',
        'service_type': 'fallback_services',
        'services': {},
        'overall_status': 'healthy'
    }

    for service_name, service in _fallback_services.items():
        health_info = service.health_check()
        overall_health['services'][service_name] = health_info

        if health_info.get('status') != 'healthy':
            overall_health['overall_status'] = 'degraded'

    return overall_health


# 便捷函数

def get_fallback_config_manager() -> FallbackConfigManager:
    """获取降级配置管理器"""
    return _fallback_services['config_manager']


def get_fallback_cache_manager() -> FallbackCacheManager:
    """获取降级缓存管理器"""
    return _fallback_services['cache_manager']


def get_fallback_logger() -> FallbackLogger:
    """获取降级日志器"""
    return _fallback_services['logger']


def get_fallback_monitoring() -> FallbackMonitoring:
    """获取降级监控器"""
    return _fallback_services['monitoring']


def get_fallback_health_checker() -> FallbackHealthChecker:
    """获取降级健康检查器"""
    return _fallback_services['health_checker']
