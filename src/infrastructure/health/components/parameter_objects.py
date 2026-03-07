"""
参数对象定义

用于优化长参数列表函数，使用参数对象模式提高代码可读性和可维护性。
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional


@dataclass
class HealthCheckConfig:
    """
    健康检查配置参数对象
    用于替代长参数列表的配置参数
    """
    name: str
    check_func: Callable
    config: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    retry_count: Optional[int] = None
    retry_delay: Optional[float] = None


@dataclass
class SystemHealthInfo:
    """
    系统健康信息参数对象
    用于系统健康检查相关函数的参数传递
    """
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    disk_info: Dict[str, Any]
    
    def get_all_statuses(self) -> list:
        """获取所有组件状态"""
        statuses = []
        for info in [self.cpu_info, self.memory_info, self.disk_info]:
            if isinstance(info, dict) and "status" in info:
                statuses.append(info["status"])
        return statuses


@dataclass
class ExecutorConfig:
    """
    执行器配置参数对象
    用于HealthCheckExecutor的初始化参数
    """
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    concurrent_limit: int = 10


@dataclass
class HealthCheckResult:
    """
    健康检查结果参数对象
    用于标准化健康检查结果的格式
    """
    service_name: str
    status: str
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "service": self.service_name,
            "status": self.status,
            "timestamp": self.timestamp
        }
        
        if self.response_time is not None:
            result["response_time"] = self.response_time
            
        if self.error_message:
            result["error"] = self.error_message
            
        if self.metadata:
            result.update(self.metadata)
            
        return result


@dataclass
class DependencyConfig:
    """
    依赖服务配置参数对象
    用于依赖服务的注册和配置
    """
    name: str
    check_func: Callable
    config: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    enabled: bool = True
    
    def get_effective_config(self) -> Dict[str, Any]:
        """获取有效的配置，合并默认配置和用户配置"""
        effective_config = self.config.copy() if self.config else {}
        
        if self.timeout is not None:
            effective_config["timeout"] = self.timeout
            
        return effective_config


@dataclass
class MonitoringConfig:
    """
    监控配置参数对象
    用于监控系统的配置参数
    """
    interval: float = 60.0
    retention_days: int = 30
    max_entries: int = 1000
    alert_timeout: float = 300.0
    enabled: bool = True


@dataclass
class AlertRuleConfig:
    """
    告警规则配置参数对象
    用于告警规则的创建和配置
    """
    name: str
    query: str
    threshold: float
    severity: str = "warning"
    duration: float = 0.0
    enabled: bool = True
    description: Optional[str] = None

