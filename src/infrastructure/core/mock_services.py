"""
基础设施层Mock服务基类

提供简化的Mock服务实现，减少代码重复，便于测试和开发

作者: RQA2025团队
创建时间: 2025-10-23
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseMockService(ABC):
    """Mock服务基类
    
    提供所有Mock服务的通用功能：
    - 健康检查
    - 调用跟踪
    - 状态管理
    - 异常模拟
    """
    
    def __init__(self, service_name: Optional[str] = None):
        self._service_name = service_name or self.__class__.__name__
        self._is_healthy = True
        self._call_history: List[tuple] = []  # 调用历史
        self._failure_mode = False  # 失败模式
        self._failure_exception = None  # 失败时抛出的异常
    
    def is_healthy(self) -> bool:
        """检查服务健康状态"""
        return self._is_healthy
    
    def check_health(self) -> Dict[str, Any]:
        """执行健康检查"""
        from .health_check_interface import HealthCheckResult
        
        return HealthCheckResult(
            service_name=self._service_name,
            healthy=self._is_healthy,
            status="healthy" if self._is_healthy else "unhealthy",
            timestamp=datetime.now(),
            version="1.0.0-mock",
            details={
                "call_count": len(self._call_history),
                "failure_mode": self._failure_mode
            }
        ).to_dict()
    
    def _record_call(self, method_name: str, *args, **kwargs):
        """记录方法调用"""
        self._call_history.append((
            datetime.now(),
            method_name,
            args,
            kwargs
        ))
    
    def _check_failure_mode(self):
        """检查是否应该抛出异常"""
        if self._failure_mode and self._failure_exception:
            raise self._failure_exception
    
    def set_failure_mode(self, enable: bool = True, exception: Optional[Exception] = None):
        """设置失败模式（用于测试）"""
        self._failure_mode = enable
        self._failure_exception = exception or Exception("Mock service failure")
    
    def set_healthy(self, healthy: bool = True):
        """设置健康状态（用于测试）"""
        self._is_healthy = healthy
    
    def get_call_history(self) -> List[tuple]:
        """获取调用历史"""
        return self._call_history.copy()
    
    def reset_call_history(self):
        """重置调用历史"""
        self._call_history.clear()
    
    @property
    def call_count(self) -> int:
        """获取调用次数"""
        return len(self._call_history)


class SimpleMockDict(BaseMockService):
    """简化的字典型Mock服务
    
    提供基于字典的简单Mock实现，
    适用于配置管理、缓存等需要键值对存储的服务
    """
    
    def __init__(self, service_name: Optional[str] = None, initial_data: Optional[Dict[str, Any]] = None):
        super().__init__(service_name)
        self._data = initial_data.copy() if initial_data else {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取值"""
        self._record_call('get', key, default)
        self._check_failure_mode()
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any, **kwargs) -> bool:
        """设置值"""
        self._record_call('set', key, value, **kwargs)
        self._check_failure_mode()
        self._data[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """删除值"""
        self._record_call('delete', key)
        self._check_failure_mode()
        return self._data.pop(key, None) is not None
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        self._record_call('exists', key)
        return key in self._data
    
    def clear(self) -> bool:
        """清空数据"""
        self._record_call('clear')
        self._check_failure_mode()
        self._data.clear()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_keys": len(self._data),
            "call_count": self.call_count,
            "is_healthy": self._is_healthy
        }
    
    def get_all_data(self) -> Dict[str, Any]:
        """获取所有数据（用于测试验证）"""
        return self._data.copy()


class SimpleMockLogger(BaseMockService):
    """简化的日志Mock服务
    
    提供基础的日志记录Mock实现，
    支持调用跟踪和日志级别过滤
    """
    
    def __init__(self, service_name: Optional[str] = None, enabled_levels: Optional[List[str]] = None):
        super().__init__(service_name)
        self._enabled_levels = enabled_levels or ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self._logs: List[Dict[str, Any]] = []
    
    def _log(self, level: str, message: str, **kwargs):
        """内部日志记录"""
        self._record_call('log', level, message, **kwargs)
        
        if level.upper() in self._enabled_levels:
            log_entry = {
                'timestamp': datetime.now(),
                'level': level.upper(),
                'message': message,
                'kwargs': kwargs
            }
            self._logs.append(log_entry)
            print(f"[{level.upper()}] {message}")
    
    def debug(self, message: str, **kwargs) -> None:
        """记录DEBUG日志"""
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """记录INFO日志"""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录WARNING日志"""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """记录ERROR日志"""
        if exc:
            kwargs['exception'] = str(exc)
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """记录CRITICAL日志"""
        if exc:
            kwargs['exception'] = str(exc)
        self._log('CRITICAL', message, **kwargs)
    
    def log(self, level, message: str, **kwargs) -> None:
        """通用日志记录"""
        self._log(level, message, **kwargs)
    
    def is_enabled_for(self, level) -> bool:
        """检查日志级别是否启用"""
        return level.upper() in self._enabled_levels
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取日志记录（用于测试验证）"""
        if level:
            return [log for log in self._logs if log['level'] == level.upper()]
        return self._logs.copy()
    
    def clear_logs(self):
        """清空日志记录"""
        self._logs.clear()


class SimpleMockMonitor(BaseMockService):
    """简化的监控Mock服务
    
    提供基础的监控指标Mock实现，
    支持指标收集和统计
    """
    
    def __init__(self, service_name: Optional[str] = None):
        super().__init__(service_name)
        self._metrics: Dict[str, List[Any]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
    
    def record_metric(self, name: str, value, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        self._record_call('record_metric', name, value, tags)
        self._metrics[name].append({
            'value': value,
            'tags': tags or {},
            'timestamp': datetime.now()
        })
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        self._record_call('increment_counter', name, value, tags)
        self._counters[name] += value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录直方图"""
        self._record_call('record_histogram', name, value, tags)
        self._metrics[f"{name}_histogram"].append({
            'value': value,
            'tags': tags or {},
            'timestamp': datetime.now()
        })
    
    def get_metric_values(self, name: str) -> List[Any]:
        """获取指标值（用于测试验证）"""
        return [m['value'] for m in self._metrics.get(name, [])]
    
    def get_counter_value(self, name: str) -> int:
        """获取计数器值（用于测试验证）"""
        return self._counters.get(name, 0)
    
    def reset_metrics(self):
        """重置所有指标"""
        self._metrics.clear()
        self._counters.clear()


# ==================== 导出列表 ====================

__all__ = [
    'BaseMockService',
    'SimpleMockDict',
    'SimpleMockLogger',
    'SimpleMockMonitor',
]

