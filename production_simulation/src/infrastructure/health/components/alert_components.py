"""
alert_components 模块

提供 alert_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类 - 延迟导入避免循环依赖
import asyncio

from src.infrastructure.utils.components.core.base_components import ComponentFactory, IComponentFactory
from abc import ABC, abstractmethod
from datetime import datetime
from ..core.interfaces import IUnifiedInfrastructureInterface
from typing import Dict, Any, Optional, List
"""
基础设施层 - Alert组件统一实现

使用统一的ComponentFactory基类，提供Alert组件的工厂模式实现。
"""

try:
    COMPONENT_FACTORY_AVAILABLE = True
except ImportError:
    # 如果导入失败，使用本地定义
    COMPONENT_FACTORY_AVAILABLE = False

    class ComponentFactory:
        """本地ComponentFactory定义避免循环导入"""

    class IComponentFactory:
        """本地IComponentFactory定义避免循环导入"""
# 导入统一基础设施接口
logger = logging.getLogger(__name__)

# 常量定义 - 清理魔法数字
ALERT_ID_SYSTEM_ERROR = 6
ALERT_ID_DATABASE_ERROR = 12
ALERT_ID_NETWORK_ERROR = 18
ALERT_ID_SECURITY_ALERT = 24
ALERT_ID_PERFORMANCE_WARNING = 30
ALERT_ID_RESOURCE_WARNING = 36
ALERT_ID_DEPENDENCY_ERROR = 42
ALERT_ID_CONFIG_ERROR = 48
ALERT_ID_BACKUP_ERROR = 54
ALERT_ID_MONITORING_ERROR = 60
ALERT_ID_CUSTOM_ALERT = 66

# 支持的告警ID列表
SUPPORTED_ALERT_IDS = [
    ALERT_ID_SYSTEM_ERROR,
    ALERT_ID_DATABASE_ERROR,
    ALERT_ID_NETWORK_ERROR,
    ALERT_ID_SECURITY_ALERT,
    ALERT_ID_PERFORMANCE_WARNING,
    ALERT_ID_RESOURCE_WARNING,
    ALERT_ID_DEPENDENCY_ERROR,
    ALERT_ID_CONFIG_ERROR,
    ALERT_ID_BACKUP_ERROR,
    ALERT_ID_MONITORING_ERROR,
    ALERT_ID_CUSTOM_ALERT
]


class IAlertComponent(ABC):

    """Alert组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_alert_id(self) -> int:
        """获取alert ID"""


class AlertComponent(IAlertComponent):

    """统一Alert组件实现"""

    def __init__(self, alert_id: int, component_type: str = "Alert"):
        """初始化组件"""
        self.alert_id = alert_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{alert_id}"
        self.creation_time = datetime.now()

    def get_alert_id(self) -> int:
        """获取alert ID"""
        return self.alert_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        try:
            logger.debug(f"获取Alert组件信息，ID: {self.alert_id}")

            info = {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "creation_time": self.creation_time.isoformat(),
                "description": f"统一{self.component_type}组件实现",
                "version": "2.0.0",
                "type": "unified_health_monitoring_component"
            }

            logger.debug(f"Alert组件信息获取成功，ID: {self.alert_id}")
            return info

        except Exception as e:
            logger.error(f"获取Alert组件信息失败，ID: {self.alert_id}, 错误: {str(e)}", exc_info=True)
            # 返回基本的错误信息
            return {
                "alert_id": self.alert_id,
                "error": str(e),
                "component_name": getattr(self, 'component_name', 'unknown'),
                "status": "error"
            }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            logger.info(f"开始处理Alert数据，ID: {self.alert_id}, 数据键数: {len(data) if data else 0}")

            # 数据验证
            if not isinstance(data, dict):
                logger.warning(f"Alert组件接收到非字典数据类型: {type(data)}")
                data = {"raw_data": data, "converted": True}

            # 处理逻辑
            processed_at = datetime.now()
            result = {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": processed_at.isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_alert_processing",
                "processing_duration": (datetime.now() - processed_at).total_seconds()
            }

            logger.info(
                f"Alert数据处理成功，ID: {self.alert_id}, 处理耗时: {result['processing_duration']:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Alert数据处理失败，ID: {self.alert_id}, 错误: {str(e)}", exc_info=True)
            return {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "alert_id": self.alert_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }

    # =========================================================================
    # 异步处理能力扩展
    # =========================================================================

    async def get_info_async(self) -> Dict[str, Any]:
        """异步获取组件信息"""
        try:
            logger.debug(f"异步获取Alert组件信息，ID: {self.alert_id}")

            info = {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "creation_time": self.creation_time.isoformat(),
                "description": f"统一{self.component_type}组件实现",
                "version": "2.0.0",
                "type": "unified_health_monitoring_component",
                "async_support": True
            }

            logger.debug(f"Alert组件异步信息获取成功，ID: {self.alert_id}")
            return info

        except Exception as e:
            logger.error(f"异步获取Alert组件信息失败，ID: {self.alert_id}, 错误: {str(e)}", exc_info=True)
            return {
                "alert_id": self.alert_id,
                "error": str(e),
                "component_name": getattr(self, 'component_name', 'unknown'),
                "status": "error"
            }

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理数据"""
        try:
            logger.info(f"开始异步处理Alert数据，ID: {self.alert_id}, 数据键数: {len(data) if data else 0}")

            # 数据验证
            if not isinstance(data, dict):
                logger.warning(f"Alert组件接收到非字典数据类型: {type(data)}")
                data = {"raw_data": data, "converted": True}

            # 模拟异步处理（可以替换为实际的异步操作）
            await asyncio.sleep(0.01)  # 短暂延迟模拟异步操作

            # 处理逻辑
            processed_at = datetime.now()
            result = {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": processed_at.isoformat(),
                "status": "success",
                "result": f"Asynchronously processed by {self.component_name}",
                "processing_type": "async_unified_alert_processing",
                "processing_duration": (datetime.now() - processed_at).total_seconds()
            }

            logger.info(
                f"Alert数据异步处理成功，ID: {self.alert_id}, 处理耗时: {result['processing_duration']:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Alert数据异步处理失败，ID: {self.alert_id}, 错误: {str(e)}", exc_info=True)
            return {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def get_status_async(self) -> Dict[str, Any]:
        """异步获取组件状态"""
        try:
            # 模拟异步状态检查
            await asyncio.sleep(0.005)

            return {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "status": "active",
                "creation_time": self.creation_time.isoformat(),
                "health": "good",
                "async_support": True,
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"异步获取Alert组件状态失败，ID: {self.alert_id}, 错误: {str(e)}", exc_info=True)
            return {
                "alert_id": self.alert_id,
                "status": "error",
                "error": str(e),
                "component_name": getattr(self, 'component_name', 'unknown')
            }


class AlertComponentFactory(IUnifiedInfrastructureInterface, ComponentFactory):

    """Alert组件工厂"""

    # 常量定义 - 与模块级常量同步
    ALERT_ID_SYSTEM_ERROR = ALERT_ID_SYSTEM_ERROR
    ALERT_ID_DATABASE_ERROR = ALERT_ID_DATABASE_ERROR
    ALERT_ID_NETWORK_ERROR = ALERT_ID_NETWORK_ERROR
    ALERT_ID_SECURITY_ALERT = ALERT_ID_SECURITY_ALERT
    ALERT_ID_PERFORMANCE_WARNING = ALERT_ID_PERFORMANCE_WARNING
    ALERT_ID_RESOURCE_WARNING = ALERT_ID_RESOURCE_WARNING
    ALERT_ID_DEPENDENCY_ERROR = ALERT_ID_DEPENDENCY_ERROR
    ALERT_ID_CONFIG_ERROR = ALERT_ID_CONFIG_ERROR
    ALERT_ID_BACKUP_ERROR = ALERT_ID_BACKUP_ERROR
    ALERT_ID_MONITORING_ERROR = ALERT_ID_MONITORING_ERROR
    ALERT_ID_CUSTOM_ALERT = ALERT_ID_CUSTOM_ALERT
    ALERT_ID_MEMORY_WARNING = ALERT_ID_RESOURCE_WARNING  # 别名
    ALERT_ID_CPU_WARNING = ALERT_ID_PERFORMANCE_WARNING  # 别名

    # 支持的alert ID列表
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    # 使用常量定义替换魔法数字
    SUPPORTED_ALERT_IDS = SUPPORTED_ALERT_IDS

    @staticmethod
    def create_component(alert_id: int) -> AlertComponent:
        """创建指定ID的alert组件"""
        try:
            logger.info(f"开始创建Alert组件，ID: {alert_id}")

            # 参数验证
            if not isinstance(alert_id, int):
                logger.error(f"Alert ID必须是整数类型，收到: {type(alert_id)}")
                raise TypeError(f"Alert ID必须是整数类型，收到: {type(alert_id)}")

            if alert_id not in AlertComponentFactory.SUPPORTED_ALERT_IDS:
                logger.error(
                    f"不支持的alert ID: {alert_id}。支持的ID: {AlertComponentFactory.SUPPORTED_ALERT_IDS}")
                raise ValueError(
                    f"不支持的alert ID: {alert_id}。支持的ID: {AlertComponentFactory.SUPPORTED_ALERT_IDS}")

            # 创建组件
            component = AlertComponent(alert_id, "Alert")
            logger.info(f"Alert组件创建成功，ID: {alert_id}, 类型: {component.component_type}")
            return component

        except Exception as e:
            logger.error(f"创建Alert组件失败，ID: {alert_id}, 错误: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def get_available_alerts() -> List[int]:
        """获取所有可用的alert ID"""
        return sorted(list(AlertComponentFactory.SUPPORTED_ALERT_IDS))

    @staticmethod
    def create_all_alerts() -> Dict[int, AlertComponent]:
        """创建所有可用alert"""
        return {
            alert_id: AlertComponent(alert_id, "Alert")
            for alert_id in AlertComponentFactory.SUPPORTED_ALERT_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "AlertComponentFactory",
            "version": "2.0.0",
            "total_alerts": len(AlertComponentFactory.SUPPORTED_ALERT_IDS),
            "supported_ids": sorted(list(AlertComponentFactory.SUPPORTED_ALERT_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Alert组件工厂，替代原有的模板化文件",
            "async_support": True
        }

    # =========================================================================
    # 异步处理能力扩展
    # =========================================================================

    @staticmethod
    async def create_component_async(alert_id: int) -> AlertComponent:
        """异步创建指定ID的alert组件"""
        try:
            logger.info(f"开始异步创建Alert组件，ID: {alert_id}")

            # 模拟异步操作（可以替换为实际的异步操作）
            await asyncio.sleep(0.01)

            # 使用同步方法创建组件
            component = AlertComponentFactory.create_component(alert_id)
            logger.info(f"Alert组件异步创建成功，ID: {alert_id}, 类型: {component.component_type}")
            return component

        except Exception as e:
            logger.error(f"异步创建Alert组件失败，ID: {alert_id}, 错误: {str(e)}", exc_info=True)
            raise

    @staticmethod
    async def get_available_alerts_async() -> List[int]:
        """异步获取所有可用的alert ID"""
        try:
            # 模拟异步操作
            await asyncio.sleep(0.005)
            return AlertComponentFactory.get_available_alerts()
        except Exception as e:
            logger.error(f"异步获取可用alerts失败: {str(e)}", exc_info=True)
            return []

    @staticmethod
    async def create_all_alerts_async() -> Dict[int, AlertComponent]:
        """异步创建所有可用alert"""
        try:
            logger.info("开始异步创建所有Alert组件")

            # 模拟异步操作
            await asyncio.sleep(0.02)

            alerts = AlertComponentFactory.create_all_alerts()
            logger.info(f"异步创建所有Alert组件成功，共{len(alerts)}个组件")
            return alerts

        except Exception as e:
            logger.error(f"异步创建所有Alert组件失败: {str(e)}", exc_info=True)
            return {}

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """获取工厂信息"""
        try:
            return AlertComponentFactory.get_factory_info()
        except Exception as e:
            logger.error(f"获取工厂信息失败: {str(e)}", exc_info=True)
            return {
                "factory_name": "AlertComponentFactory",
                "error": str(e),
                "supported_alerts_count": len(AlertComponentFactory.SUPPORTED_ALERT_IDS)
            }

    @staticmethod
    async def get_info_async() -> Dict[str, Any]:
        """异步获取工厂信息"""
        try:
            # 模拟异步操作
            await asyncio.sleep(0.005)
            return AlertComponentFactory.get_info()
        except Exception as e:
            logger.error(f"异步获取工厂信息失败: {str(e)}", exc_info=True)
            return {
                "factory_name": "AlertComponentFactory",
                "error": str(e),
                "async_support": True
            }

    @staticmethod
    async def get_factory_info_async() -> Dict[str, Any]:
        """异步获取工厂信息"""
        try:
            # 模拟异步操作
            await asyncio.sleep(0.005)

            info = AlertComponentFactory.get_factory_info()
            info["async_check_time"] = datetime.now().isoformat()
            return info

        except Exception as e:
            logger.error(f"异步获取工厂信息失败: {str(e)}", exc_info=True)
            return {
                "factory_name": "AlertComponentFactory",
                "error": str(e),
                "async_support": True
            }

    # ============================================================================
    # IUnifiedInfrastructureInterface 实现
    # ============================================================================

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化Alert组件工厂

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化AlertComponentFactory")
            logger.debug(f"初始化配置参数: {config.keys() if config else 'None'}")

            # 如果提供了配置，更新现有配置
            if config:
                logger.debug("应用配置更新")
                # 这里可以根据需要更新工厂配置

            logger.info("AlertComponentFactory 初始化完成")
            return True
        except Exception as e:
            logger.error(f"AlertComponentFactory 初始化失败: {e}", exc_info=True)
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "component_type": "AlertComponentFactory",
            "description": "Alert组件工厂",
            "version": "1.0.0",
            "capabilities": ["create_alert_components", "manage_alert_types"],
            "supported_alert_ids": list(self.SUPPORTED_ALERT_IDS),
            "total_supported": len(self.SUPPORTED_ALERT_IDS)
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态"""
        try:
            # 工厂总是健康的，只要类定义正确
            return True
        except Exception as e:
            logger.error(f"AlertComponentFactory健康检查失败: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标"""
        return {
            "factory_name": "AlertComponentFactory",
            "supported_alert_types": len(self.SUPPORTED_ALERT_IDS),
            "status": "operational",
            "timestamp": datetime.now().isoformat()
        }

    def cleanup(self) -> bool:
        """清理组件资源"""
        try:
            # 工厂类通常不需要清理资源
            logger.info("AlertComponentFactory资源清理完成")
            return True
        except Exception as e:
            logger.error(f"AlertComponentFactory资源清理失败: {e}")
            return False


def create_alert_alert_component_6():
    return AlertComponentFactory.create_component(ALERT_ID_SYSTEM_ERROR)


def create_alert_alert_component_12():
    return AlertComponentFactory.create_component(ALERT_ID_DATABASE_ERROR)


def create_alert_alert_component_18():
    return AlertComponentFactory.create_component(ALERT_ID_NETWORK_ERROR)


def create_alert_alert_component_24():
    return AlertComponentFactory.create_component(ALERT_ID_SECURITY_ALERT)


def create_alert_alert_component_30():
    return AlertComponentFactory.create_component(ALERT_ID_PERFORMANCE_WARNING)


def create_alert_alert_component_36():
    return AlertComponentFactory.create_component(ALERT_ID_RESOURCE_WARNING)


def create_alert_alert_component_42():
    return AlertComponentFactory.create_component(ALERT_ID_DEPENDENCY_ERROR)


def create_alert_alert_component_48():
    return AlertComponentFactory.create_component(ALERT_ID_CONFIG_ERROR)


def create_alert_alert_component_54():
    return AlertComponentFactory.create_component(ALERT_ID_BACKUP_ERROR)


def create_alert_alert_component_60():
    return AlertComponentFactory.create_component(ALERT_ID_MONITORING_ERROR)


def create_alert_alert_component_66():
    return AlertComponentFactory.create_component(ALERT_ID_CUSTOM_ALERT)


__all__ = [
    "IAlertComponent",
    "AlertComponent",
    "AlertComponentFactory",
    "create_alert_alert_component_6",
    "create_alert_alert_component_12",
    "create_alert_alert_component_18",
    "create_alert_alert_component_24",
    "create_alert_alert_component_30",
    "create_alert_alert_component_36",
    "create_alert_alert_component_42",
    "create_alert_alert_component_48",
    "create_alert_alert_component_54",
    "create_alert_alert_component_60",
    "create_alert_alert_component_66",
]
