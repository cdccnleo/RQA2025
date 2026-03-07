"""
logging_service_components 模块

提供 logging_service_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, List, Optional
"""
基础设施层 - Logging_Service组件统一实现

使用统一的ComponentFactory基类，提供Logging_Service组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class ILoggingServiceComponent(ABC):

    """Logging Service组件接口"""

    def get_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            组件基本信息字典
        """
        return {
            "component_type": "logging_service_component",
            "interface": "ILoggingServiceComponent",
            "version": "2.0.0",
            "description": "Logging Service组件基础接口"
        }

    @abstractmethod
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_service_id(self) -> int:
        """获取服务ID"""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标"""


class BaseService:

    """基础服务类"""

    def __init__(self, service_name: str = "Logging_BaseService", service_id: int = None):
        """初始化服务"""
        self.service_name = service_name
        self.service_id = service_id
        self.is_running = False
        self.start_time = None
        self.request_count = 0
        self._status = "stopped"
        self._start_time = None
        self._metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "response_time_avg": 0.0,
            "avg_response_time": 0.0,  # 兼容性字段
            "uptime_seconds": 0
        }

    async def start(self) -> bool:
        """启动服务"""
        try:
            self.is_running = True
            self.start_time = datetime.now()
            print(f"服务 {self.service_name} 已启动")
            return True
        except Exception as e:
            print(f"启动服务失败: {e}")
            return False

    async def stop(self) -> bool:
        """停止服务"""
        try:
            self.is_running = False
            print(f"服务 {self.service_name} 已停止")
            return True
        except Exception as e:
            print(f"停止服务失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        uptime_seconds = self._calculate_uptime()
        status_info = {
            "service_name": self.service_name,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": f"{uptime_seconds:.2f}s" if uptime_seconds > 0 else None,
            "timestamp": datetime.now().isoformat()
        }
        if hasattr(self, '_status'):
            status_info["status"] = self._status
        return status_info

    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        return self._metrics.copy()

    def _update_metrics(self, response_time: float = 0.0, error: bool = False):
        """更新指标"""
        self._metrics["requests_total"] += 1
        if error:
            self._metrics["errors_total"] += 1

        # 更新平均响应时间
        if response_time > 0:
            current_avg = self._metrics["response_time_avg"]
            total_requests = self._metrics["requests_total"]
            new_avg = (current_avg * (total_requests - 1) + response_time) / total_requests
            self._metrics["response_time_avg"] = new_avg
            self._metrics["avg_response_time"] = new_avg  # 同步更新兼容性字段

    def _calculate_uptime(self) -> float:
        """计算运行时间（秒）"""
        if not self.start_time:
            return 0.0
        uptime = datetime.now() - self.start_time
        return uptime.total_seconds()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "service": self.service_name,
            "status": "healthy" if self.is_running else "unhealthy",
            "timestamp": datetime.now().isoformat()
        }


class LoggingServiceComponent(BaseService, ILoggingServiceComponent):

    """统一Logging Service组件实现"""

    def __init__(self, service_id: int):
        """初始化组件"""
        super().__init__(f"日志管理_Service_{service_id}")
        self.service_id = service_id
        self.component_name = f"LoggingService_Component_{service_id}"
        self.creation_time = datetime.now()
        self.processed_requests = 0
        self.error_count = 0

    def get_service_id(self) -> int:
        """获取服务ID"""
        return self.service_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "service_id": self.service_id,
            "component_name": self.component_name,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一Logging Service组件实现",
            "version": "2.0.0",
            "component_type": "logging_service_component",
            "interface": "ILoggingServiceComponent",
            "type": "unified_logging_service_component"
        }

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标"""
        return {
            "processed_requests": self.processed_requests,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.processed_requests * 100) if self.processed_requests > 0 else 0,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None,
            "service_id": self.service_id
        }

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        try:
            self.processed_requests += 1

            action = request_data.get('action', 'unknown')

            if action == 'get_status':
                # 获取状态
                status_data = self.get_status()
                result = {
                    "request_id": f"req_{self.processed_requests}",
                    "service_id": self.service_id,
                    "component_name": self.component_name,
                    "status": "success",
                    "processed_at": datetime.now().isoformat(),
                    "data": status_data
                }
            elif action == 'get_metrics':
                # 获取指标
                metrics_data = self.get_metrics()
                result = {
                    "request_id": f"req_{self.processed_requests}",
                    "service_id": self.service_id,
                    "component_name": self.component_name,
                    "status": "success",
                    "processed_at": datetime.now().isoformat(),
                    "data": metrics_data
                }
            elif action == 'log_entry':
                # 处理日志条目
                result = {
                    "request_id": f"req_{self.processed_requests}",
                    "service_id": self.service_id,
                    "component_name": self.component_name,
                    "status": "success",
                    "processed_at": datetime.now().isoformat(),
                    "message": f"Logged entry: {request_data.get('message', 'N/A')}",
                    "level": request_data.get('level', 'INFO')
                }
            else:
                # 无效的action
                if not request_data.get('action'):
                    raise ValueError("Missing required field: action")
                else:
                    raise ValueError(f"Unsupported action: {action}")

            return result

        except Exception as e:
            self.error_count += 1
            return {
                "request_id": f"req_{self.processed_requests}",
                "service_id": self.service_id,
                "component_name": self.component_name,
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "service_id": self.service_id,
            "component_name": self.component_name,
            "status": "running" if self.is_running else "stopped",
            "processed_requests": self.processed_requests,
            "error_count": self.error_count,
            "creation_time": self.creation_time.isoformat(),
            "health": "healthy" if self.error_count == 0 else "warning"
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "service_name": self.service_name,
            "service_id": self.service_id,
            "component_name": self.component_name,
            "processed_requests": self.processed_requests,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.processed_requests * 100) if self.processed_requests > 0 else 0,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None
        }


class LoggingServiceComponentFactory(ComponentFactory):

    """Logging Service组件工厂"""

    # 支持的服务ID列表
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数
        self._components = {}
        self._configs = {}

    SUPPORTED_SERVICE_IDS = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70]

    @staticmethod
    def create_component(service_id: int) -> LoggingServiceComponent:
        """创建指定ID的服务组件"""
        if service_id not in LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS:
            raise ValueError(
                f"不支持的服务ID: {service_id}。支持的ID: {LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS}")

        return LoggingServiceComponent(service_id)

    @staticmethod
    def get_available_services() -> List[int]:
        """获取所有可用的服务ID"""
        return sorted(list(LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS))

    def create_component_instance(self, service_type: str, config: Dict[str, Any] = None) -> LoggingServiceComponent:
        """创建组件实例（兼容性方法）"""
        if service_type == 'logging_service':
            # 使用第一个可用的服务ID
            service_id = self.SUPPORTED_SERVICE_IDS[0]
            return self.create_component(service_id)
        raise ValueError(f"不支持的服务类型: {service_type}")

    def register_component(self, component_type: str, component_class):
        """注册组件"""
        self._components[component_type] = component_class

    def get_available_components(self) -> List[str]:
        """获取可用组件"""
        return list(self._components.keys())

    def validate_config(self, component_type: str, config: Dict[str, Any]) -> bool:
        """验证配置"""
        if not isinstance(config, dict):
            return False
        return True

    def get_component_info(self, component_type: str) -> Optional[Dict[str, Any]]:
        """获取组件信息"""
        if component_type == 'logging_service':
            return {
                "type": "logging_service",
                "description": "Logging Service Component",
                "version": "2.0.0"
            }
        return None

    @staticmethod
    def create_all_services() -> Dict[int, LoggingServiceComponent]:
        """创建所有可用服务"""
        return {
            service_id: LoggingServiceComponent(service_id)
            for service_id in LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "LoggingServiceComponentFactory",
            "version": "2.0.0",
            "total_services": len(LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS),
            "supported_ids": sorted(list(LoggingServiceComponentFactory.SUPPORTED_SERVICE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Logging Service组件工厂"
        }

# 向后兼容：创建旧的组件实例


def create_logging_service_component_4():

    return LoggingServiceComponentFactory.create_component(4)


def create_logging_service_component_10():

    return LoggingServiceComponentFactory.create_component(10)


def create_logging_service_component_16():

    return LoggingServiceComponentFactory.create_component(16)


def create_logging_service_component_22():

    return LoggingServiceComponentFactory.create_component(22)


def create_logging_service_component_28():

    return LoggingServiceComponentFactory.create_component(28)


def create_logging_service_component_34():

    return LoggingServiceComponentFactory.create_component(34)


def create_logging_service_component_40():

    return LoggingServiceComponentFactory.create_component(40)


def create_logging_service_component_46():

    return LoggingServiceComponentFactory.create_component(46)


def create_logging_service_component_52():

    return LoggingServiceComponentFactory.create_component(52)


def create_logging_service_component_58():

    return LoggingServiceComponentFactory.create_component(58)


def create_logging_service_component_64():

    return LoggingServiceComponentFactory.create_component(64)


def create_logging_service_component_70():

    return LoggingServiceComponentFactory.create_component(70)


__all__ = [
    "ILoggingServiceComponent",
    "BaseService",
    "LoggingServiceComponent",
    "LoggingServiceComponentFactory",
    "create_logging_service_component_4",
    "create_logging_service_component_10",
    "create_logging_service_component_16",
    "create_logging_service_component_22",
    "create_logging_service_component_28",
    "create_logging_service_component_34",
    "create_logging_service_component_40",
    "create_logging_service_component_46",
    "create_logging_service_component_52",
    "create_logging_service_component_58",
    "create_logging_service_component_64",
    "create_logging_service_component_70",
]

# Logger setup
logger = logging.getLogger(__name__)
