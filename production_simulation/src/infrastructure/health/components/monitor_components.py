"""
monitor_components 模块

提供 monitor_components 相关功能和接口。
"""

import logging

# 导入统一基础设施接口
# 导入统一的ComponentFactory基类
import asyncio

from abc import ABC, abstractmethod
from datetime import datetime
from ..core.interfaces import IUnifiedInfrastructureInterface
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, Optional, List
"""
基础设施层 - Monitor组件统一实现

使用统一的ComponentFactory基类，提供Monitor组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IMonitorComponent(ABC):

    """Monitor组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    async def get_info_async(self) -> Dict[str, Any]:
        """异步获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    async def get_status_async(self) -> Dict[str, Any]:
        """异步获取组件状态"""

    @abstractmethod
    def get_monitor_id(self) -> int:
        """获取monitor ID"""


class MonitorComponent(IMonitorComponent):

    """统一Monitor组件实现"""

    def __init__(self, monitor_id: int, component_type: str = "Monitor"):
        """初始化组件"""
        self.monitor_id = monitor_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{monitor_id}"
        self.creation_time = datetime.now()

    def get_monitor_id(self) -> int:
        """获取monitor ID"""
        return self.monitor_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "monitor_id": self.monitor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_health_monitoring_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_monitor_processing"
            }
            return result
        except Exception as e:
            return {
                "monitor_id": self.monitor_id,
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
            "monitor_id": self.monitor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }

    # ============================================================================
    # 异步方法实现
    # ============================================================================

    async def get_info_async(self) -> Dict[str, Any]:
        """异步获取组件信息"""
        try:
            # 模拟异步操作
            await asyncio.sleep(0.001)

            return {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "creation_time": self.creation_time.isoformat(),
                "description": f"统一{self.component_type}组件异步实现",
                "version": "2.0.0",
                "type": "unified_health_monitoring_component_async",
                "async_support": True
            }
        except Exception as e:
            logger.error(f"异步获取组件信息失败: {e}")
            return {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "error": str(e),
                "async_support": True
            }

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理数据"""
        try:
            # 模拟异步处理操作
            await asyncio.sleep(0.01)

            result = {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Asynchronously processed by {self.component_name}",
                "processing_type": "unified_monitor_async_processing",
                "async_support": True
            }
            return result
        except Exception as e:
            logger.error(f"异步处理数据失败: {e}")
            return {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "async_support": True
            }

    async def get_status_async(self) -> Dict[str, Any]:
        """异步获取组件状态"""
        try:
            # 模拟异步状态检查
            await asyncio.sleep(0.001)

            return {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "status": "active",
                "creation_time": self.creation_time.isoformat(),
                "health": "good",
                "type": "unified_monitor_async_status",
                "async_support": True,
                "performance": {
                    "response_time_ms": 1.0,
                    "throughput": "high"
                }
            }
        except Exception as e:
            logger.error(f"异步获取组件状态失败: {e}")
            return {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "status": "error",
                "error": str(e),
                "async_support": True
            }


class MonitorComponentFactory(ComponentFactory, IUnifiedInfrastructureInterface):

    """Monitor组件工厂"""

    # 支持的monitor ID列表
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    SUPPORTED_MONITOR_IDS = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69]

    @staticmethod
    def create_component(monitor_id: int) -> MonitorComponent:
        """创建指定ID的monitor组件"""
        if monitor_id not in MonitorComponentFactory.SUPPORTED_MONITOR_IDS:
            raise ValueError(
                f"不支持的monitor ID: {monitor_id}。支持的ID: {MonitorComponentFactory.SUPPORTED_MONITOR_IDS}")

        return MonitorComponent(monitor_id, "Monitor")

    @staticmethod
    def get_available_monitors() -> List[int]:
        """获取所有可用的monitor ID"""
        return sorted(list(MonitorComponentFactory.SUPPORTED_MONITOR_IDS))

    @staticmethod
    def create_all_monitors() -> Dict[int, MonitorComponent]:
        """创建所有可用monitor"""
        return {
            monitor_id: MonitorComponent(monitor_id, "Monitor")
            for monitor_id in MonitorComponentFactory.SUPPORTED_MONITOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "MonitorComponentFactory",
            "version": "2.0.0",
            "total_monitors": len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS),
            "supported_ids": sorted(list(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }

    @staticmethod
    async def get_factory_info_async() -> Dict[str, Any]:
        """异步获取工厂信息"""
        try:
            # 模拟异步操作
            await asyncio.sleep(0.005)

            info = MonitorComponentFactory.get_factory_info()
            info["async_check_time"] = datetime.now().isoformat()
            info["async_support"] = True
            return info
        except Exception as e:
            logger.error(f"异步获取工厂信息失败: {str(e)}", exc_info=True)
            return {
                "factory_name": "MonitorComponentFactory",
                "error": str(e),
                "async_support": True
            }

    @staticmethod
    async def create_component_async(monitor_id: int) -> Optional[MonitorComponent]:
        """异步创建组件"""
        try:
            # 模拟异步创建操作
            await asyncio.sleep(0.01)

            component = MonitorComponentFactory.create_component(monitor_id)
            logger.info(f"异步创建Monitor组件成功，ID: {monitor_id}")
            return component
        except Exception as e:
            logger.error(f"异步创建Monitor组件失败，ID: {monitor_id}, 错误: {str(e)}")
            return None

    @staticmethod
    async def get_available_monitors_async() -> Dict[int, MonitorComponent]:
        """异步获取所有可用monitor"""
        try:
            logger.info("开始异步获取所有Monitor组件")

            # 模拟异步操作
            await asyncio.sleep(0.02)

            monitors = MonitorComponentFactory.get_available_monitors()
            logger.info(f"异步获取所有Monitor组件成功，共{len(monitors)}个组件")
            return monitors

        except Exception as e:
            logger.error(f"异步获取所有Monitor组件失败: {str(e)}", exc_info=True)
            return {}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化组件

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化MonitorComponentFactory")

            # 设置初始化状态
            self._initialized = True
            self._config = config or {}
            self._creation_count = 0
            self._last_creation_time = None

            logger.info("MonitorComponentFactory初始化完成")
            return True

        except Exception as e:
            logger.error(f"MonitorComponentFactory初始化失败: {str(e)}", exc_info=True)
            self._initialized = False
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        try:
            return {
                "component_name": "MonitorComponentFactory",
                "component_type": "Factory",
                "version": "2.0.0",
                "supported_monitor_ids": self.SUPPORTED_MONITOR_IDS,
                "total_supported_monitors": len(self.SUPPORTED_MONITOR_IDS),
                "initialized": getattr(self, '_initialized', False),
                "creation_count": getattr(self, '_creation_count', 0),
                "last_creation_time": getattr(self, '_last_creation_time', None),
                "async_support": True
            }
        except Exception as e:
            logger.error(f"获取Monitor组件工厂信息失败: {e}")
            return {"error": str(e)}

    def is_healthy(self) -> bool:
        """检查组件是否健康

        Returns:
            bool: 健康状态
        """
        try:
            # 检查基本状态
            is_initialized = getattr(self, '_initialized', False)
            has_supported_ids = len(self.SUPPORTED_MONITOR_IDS) > 0

            # 检查是否可以创建组件
            try:
                test_component = self.create_component(self.SUPPORTED_MONITOR_IDS[0])
                creation_works = test_component is not None
            except Exception:
                creation_works = False

            return is_initialized and has_supported_ids and creation_works

        except Exception as e:
            logger.error(f"Monitor组件工厂健康检查失败: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标
        """
        try:
            return {
                "factory_metrics": {
                    "total_supported_monitors": len(self.SUPPORTED_MONITOR_IDS),
                    "supported_monitor_ids": self.SUPPORTED_MONITOR_IDS,
                    "creation_count": getattr(self, '_creation_count', 0),
                    "last_creation_time": getattr(self, '_last_creation_time', None),
                    "factory_status": "active" if getattr(self, '_initialized', False) else "inactive"
                },
                "performance_metrics": {
                    "async_support": True,
                    "factory_efficiency": 1.0 if self.is_healthy() else 0.0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取Monitor组件工厂指标失败: {e}")
            return {"error": str(e)}

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            logger.info("开始清理MonitorComponentFactory资源")

            # 清理实例变量
            if hasattr(self, '_config'):
                self._config.clear()
            if hasattr(self, '_adapters'):
                self._adapters.clear()

            # 重置状态
            self._initialized = False
            self._creation_count = 0
            self._last_creation_time = None

            logger.info("MonitorComponentFactory资源清理完成")
            return True

        except Exception as e:
            logger.error(f"MonitorComponentFactory资源清理失败: {str(e)}", exc_info=True)
            return False

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始Monitor组件工厂健康检查")

            health_checks = {
                "factory_status": self.check_factory_health(),
                "component_creation": self.check_component_creation_health(),
                "configuration": self.check_configuration_health()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": "monitor_component_factory",
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning("Monitor组件工厂健康检查发现问题")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"Monitor组件工厂健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"Monitor组件工厂健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": "monitor_component_factory",
                "error": str(e)
            }

    def check_factory_health(self) -> Dict[str, Any]:
        """检查工厂运行状态

        Returns:
            Dict[str, Any]: 工厂健康状态检查结果
        """
        try:
            # 检查工厂基本状态
            is_initialized = getattr(self, '_initialized', False)

            # 检查支持的monitor ID数量
            supported_count = len(self.SUPPORTED_MONITOR_IDS)

            return {
                "healthy": is_initialized and supported_count > 0,
                "initialized": is_initialized,
                "supported_monitors_count": supported_count,
                "supported_monitor_ids": self.SUPPORTED_MONITOR_IDS
            }
        except Exception as e:
            logger.error(f"工厂健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_component_creation_health(self) -> Dict[str, Any]:
        """检查组件创建功能

        Returns:
            Dict[str, Any]: 组件创建健康检查结果
        """
        try:
            # 尝试创建一个测试组件
            test_component = None
            creation_successful = False

            try:
                test_component = self.create_component(self.SUPPORTED_MONITOR_IDS[0])
                creation_successful = test_component is not None
                if test_component and hasattr(test_component, 'component_type'):
                    creation_successful = creation_successful and test_component.component_type == "Monitor"
            except Exception:
                creation_successful = False

            return {
                "healthy": creation_successful,
                "component_creation_test": creation_successful,
                "test_component_type": test_component.component_type if test_component else None,
                "test_component_id": test_component.get_monitor_id() if test_component else None
            }
        except Exception as e:
            logger.error(f"组件创建健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_configuration_health(self) -> Dict[str, Any]:
        """检查配置健康状态

        Returns:
            Dict[str, Any]: 配置健康检查结果
        """
        try:
            # 检查常量定义
            constants_defined = hasattr(self, 'SUPPORTED_MONITOR_IDS')

            # 检查monitor ID的有效性
            supported_ids = self.SUPPORTED_MONITOR_IDS
            ids_valid = all(
                isinstance(monitor_id, int) and monitor_id > 0
                for monitor_id in supported_ids
            )

            return {
                "healthy": constants_defined and ids_valid,
                "constants_defined": constants_defined,
                "ids_valid": ids_valid,
                "total_monitor_types": len(supported_ids)
            }
        except Exception as e:
            logger.error(f"配置健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            health_check = self.check_health()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "factory_info": self.get_component_info(),
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            available_monitors = self.get_available_monitors()

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "monitor_system_status": {
                    "total_monitor_types": len(available_monitors),
                    "supported_monitor_ids": available_monitors,
                    "factory_operational": health_check["healthy"]
                },
                "configuration": {
                    "monitor_constants_defined": hasattr(self, 'SUPPORTED_MONITOR_IDS'),
                    "factory_initialized": getattr(self, '_initialized', False)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "error": str(e)}

    def monitor_monitor_factory(self) -> Dict[str, Any]:
        """监控Monitor工厂状态

        Returns:
            Dict[str, Any]: 工厂监控结果
        """
        try:
            factory_info = self.get_component_info()
            health_check = self.check_health()

            # 计算工厂效率指标
            supported_monitors = len(factory_info.get("supported_monitor_ids", []))
            factory_efficiency = 1.0 if health_check["healthy"] else 0.0

            return {
                "healthy": health_check["healthy"],
                "factory_metrics": {
                    "supported_monitors": supported_monitors,
                    "factory_efficiency": factory_efficiency,
                    "operational_status": "active" if health_check["healthy"] else "inactive"
                },
                "monitor_coverage": {
                    "system_monitors": any(id >= 3 and id <= 9 for id in factory_info.get("supported_monitor_ids", [])),
                    "database_monitors": any(id >= 15 and id <= 21 for id in factory_info.get("supported_monitor_ids", [])),
                    "network_monitors": any(id >= 27 and id <= 33 for id in factory_info.get("supported_monitor_ids", [])),
                    "performance_monitors": any(id >= 39 and id <= 45 for id in factory_info.get("supported_monitor_ids", []))
                }
            }
        except Exception as e:
            logger.error(f"Monitor工厂监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def validate_monitor_factory_config(self) -> Dict[str, Any]:
        """验证Monitor工厂配置

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            validation_results = {
                "constants_validation": self._validate_monitor_constants(),
                "factory_initialization": self._validate_factory_initialization(),
                "component_creation": self._validate_component_creation()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Monitor工厂配置验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    def _validate_monitor_constants(self) -> Dict[str, Any]:
        """验证Monitor常量定义"""
        try:
            required_constants = ['SUPPORTED_MONITOR_IDS']

            constants_defined = all(
                hasattr(self, name) for name in required_constants
            )

            # 检查monitor ID是否有效
            if constants_defined:
                supported_ids = self.SUPPORTED_MONITOR_IDS
                ids_valid = all(
                    isinstance(monitor_id, int) and monitor_id > 0
                    for monitor_id in supported_ids
                )
                ids_unique = len(set(supported_ids)) == len(supported_ids)
            else:
                ids_valid = False
                ids_unique = False

            return {
                "valid": constants_defined and ids_valid and ids_unique,
                "constants_defined": constants_defined,
                "ids_valid": ids_valid,
                "ids_unique": ids_unique,
                "defined_constants": len([name for name in required_constants if hasattr(self, name)])
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_factory_initialization(self) -> Dict[str, Any]:
        """验证工厂初始化"""
        try:
            # 检查工厂是否可以实例化
            factory_instance = MonitorComponentFactory()
            initialization_successful = factory_instance is not None

            # 检查是否继承了正确的基类
            inherits_correct_base = (
                isinstance(factory_instance, ComponentFactory) and
                isinstance(factory_instance, IUnifiedInfrastructureInterface)
            )

            return {
                "valid": initialization_successful and inherits_correct_base,
                "initialization_successful": initialization_successful,
                "inherits_correct_base": inherits_correct_base
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_component_creation(self) -> Dict[str, Any]:
        """验证组件创建功能"""
        try:
            # 测试创建多个组件
            test_monitor_ids = self.SUPPORTED_MONITOR_IDS[:3]  # 测试前3个
            creation_results = {}

            for monitor_id in test_monitor_ids:
                try:
                    component = self.create_component(monitor_id)
                    creation_results[monitor_id] = {
                        "success": True,
                        "component_type": component.component_type if hasattr(component, 'component_type') else None
                    }
                except Exception as e:
                    creation_results[monitor_id] = {
                        "success": False,
                        "error": str(e)
                    }

            all_successful = all(result["success"] for result in creation_results.values())

            return {
                "valid": all_successful,
                "creation_results": creation_results,
                "tested_monitor_ids": test_monitor_ids
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

# 向后兼容：创建旧的组件实例


def create_monitor_monitor_component_3():

    return MonitorComponentFactory.create_component(3)


def create_monitor_monitor_component_9():

    return MonitorComponentFactory.create_component(9)


def create_monitor_monitor_component_15():

    return MonitorComponentFactory.create_component(15)


def create_monitor_monitor_component_21():

    return MonitorComponentFactory.create_component(21)


def create_monitor_monitor_component_27():

    return MonitorComponentFactory.create_component(27)


def create_monitor_monitor_component_33():

    return MonitorComponentFactory.create_component(33)


def create_monitor_monitor_component_39():

    return MonitorComponentFactory.create_component(39)


def create_monitor_monitor_component_45():

    return MonitorComponentFactory.create_component(45)


def create_monitor_monitor_component_51():

    return MonitorComponentFactory.create_component(51)


def create_monitor_monitor_component_57():

    return MonitorComponentFactory.create_component(57)


def create_monitor_monitor_component_63():

    return MonitorComponentFactory.create_component(63)


def create_monitor_monitor_component_69():

    return MonitorComponentFactory.create_component(69)


__all__ = [
    "IMonitorComponent",
    "MonitorComponent",
    "MonitorComponentFactory",
    "create_monitor_monitor_component_3",
    "create_monitor_monitor_component_9",
    "create_monitor_monitor_component_15",
    "create_monitor_monitor_component_21",
    "create_monitor_monitor_component_27",
    "create_monitor_monitor_component_33",
    "create_monitor_monitor_component_39",
    "create_monitor_monitor_component_45",
    "create_monitor_monitor_component_51",
    "create_monitor_monitor_component_57",
    "create_monitor_monitor_component_63",
    "create_monitor_monitor_component_69",
]
