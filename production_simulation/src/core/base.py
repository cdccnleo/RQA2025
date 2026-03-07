"""
Core Base模块 - 核心基础组件

提供所有核心服务层组件的基类和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """组件状态枚举"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


class ComponentHealth(Enum):
    """组件健康状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class BaseComponent(ABC):
    """
    核心基础组件抽象基类
    
    所有核心服务层组件的基类，提供统一的生命周期管理和状态监控
    """
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        """
        初始化基础组件
        
        Args:
            name: 组件名称
            version: 组件版本
            description: 组件描述
        """
        self.name = name
        self.version = version
        self.description = description
        self._status = ComponentStatus.INITIALIZING
        self._health = ComponentHealth.UNKNOWN
        self._created_at = datetime.now()
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None
        self._error_count = 0
        self._metrics: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    @property
    def status(self) -> ComponentStatus:
        """获取组件状态"""
        return self._status
    
    @status.setter
    def status(self, value: ComponentStatus):
        """设置组件状态"""
        self._status = value
    
    @property
    def health(self) -> ComponentHealth:
        """获取组件健康状态"""
        return self._health
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        初始化组件
        
        Args:
            config: 配置信息
            
        Returns:
            是否初始化成功
        """
        try:
            self._config = config or {}
            self._status = ComponentStatus.INITIALIZING
            result = self._do_initialize(config)
            if result:
                self._status = ComponentStatus.STOPPED
                self._health = ComponentHealth.HEALTHY
            else:
                self._status = ComponentStatus.ERROR
                self._health = ComponentHealth.UNHEALTHY
            return result
        except Exception as e:
            logger.error(f"组件 {self.name} 初始化失败: {e}")
            self._status = ComponentStatus.ERROR
            self._health = ComponentHealth.UNHEALTHY
            return False
    
    def start(self) -> bool:
        """
        启动组件
        
        Returns:
            是否启动成功
        """
        try:
            if self._status == ComponentStatus.RUNNING:
                logger.warning(f"组件 {self.name} 已在运行")
                return True
            
            result = self._do_start()
            if result:
                self._status = ComponentStatus.RUNNING
                self._started_at = datetime.now()
                self._health = ComponentHealth.HEALTHY
            else:
                self._status = ComponentStatus.ERROR
                self._health = ComponentHealth.UNHEALTHY
            return result
        except Exception as e:
            logger.error(f"组件 {self.name} 启动失败: {e}")
            self._status = ComponentStatus.ERROR
            self._health = ComponentHealth.UNHEALTHY
            return False
    
    def stop(self) -> bool:
        """
        停止组件
        
        Returns:
            是否停止成功
        """
        try:
            if self._status == ComponentStatus.STOPPED:
                logger.warning(f"组件 {self.name} 已停止")
                return True
            
            result = self._do_stop()
            if result:
                self._status = ComponentStatus.STOPPED
                self._stopped_at = datetime.now()
            return result
        except Exception as e:
            logger.error(f"组件 {self.name} 停止失败: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康检查结果
        """
        try:
            check_result = self._do_health_check()
            return {
                'component': self.name,
                'status': self._status.value,
                'health': self._health.value,
                'uptime': (datetime.now() - self._created_at).total_seconds(),
                'error_count': self._error_count,
                **check_result
            }
        except Exception as e:
            logger.error(f"组件 {self.name} 健康检查失败: {e}")
            return {
                'component': self.name,
                'status': 'error',
                'health': 'unhealthy',
                'error': str(e)
            }
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        获取状态信息
        
        Returns:
            状态信息字典
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'status': self._status.value,
            'health': self._health.value,
            'created_at': self._created_at.isoformat() if self._created_at else None,
            'started_at': self._started_at.isoformat() if self._started_at else None,
            'stopped_at': self._stopped_at.isoformat() if self._stopped_at else None,
            'error_count': self._error_count,
            'metrics': self._metrics
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        return self._metrics.copy()
    
    def update_metric(self, key: str, value: Any):
        """
        更新性能指标
        
        Args:
            key: 指标名称
            value: 指标值
        """
        self._metrics[key] = value
    
    def record_error(self):
        """记录错误"""
        self._error_count += 1
    
    # 子类需要实现的方法
    @abstractmethod
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """执行初始化的具体逻辑"""
        pass
    
    @abstractmethod
    def _do_start(self) -> bool:
        """执行启动的具体逻辑"""
        pass
    
    @abstractmethod
    def _do_stop(self) -> bool:
        """执行停止的具体逻辑"""
        pass
    
    @abstractmethod
    def _do_health_check(self) -> Dict[str, Any]:
        """执行健康检查的具体逻辑"""
        pass


class BaseService(BaseComponent):
    """
    基础服务类
    
    提供通用的服务功能，如配置管理、日志记录等
    """
    
    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        super().__init__(name, version, description)
        self.logger = logging.getLogger(name)
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """默认初始化实现"""
        self.logger.info(f"服务 {self.name} 初始化")
        return True
    
    def _do_start(self) -> bool:
        """默认启动实现"""
        self.logger.info(f"服务 {self.name} 启动")
        return True
    
    def _do_stop(self) -> bool:
        """默认停止实现"""
        self.logger.info(f"服务 {self.name} 停止")
        return True
    
    def _do_health_check(self) -> Dict[str, Any]:
        """默认健康检查实现"""
        return {
            'healthy': True,
            'message': 'Service is running normally'
        }


__all__ = [
    'BaseComponent',
    'BaseService',
    'ComponentStatus',
    'ComponentHealth'
]

