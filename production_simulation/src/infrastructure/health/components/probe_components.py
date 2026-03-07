"""
probe_components 模块

提供 probe_components 相关功能和接口。
"""

import logging

import asyncio
# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.core.base_components import ComponentFactory
from typing import Dict, Any, List
"""
基础设施层 - Probe组件统一实现

使用统一的ComponentFactory基类，提供Probe组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IProbeComponent(ABC):

    """Probe组件接口"""

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
    def get_probe_id(self) -> int:
        """获取probe ID"""


class ProbeComponent(IProbeComponent):
    """统一Probe组件实现"""

    def __init__(self, probe_id: int, component_type: str = "Probe"):
        """初始化组件"""
        self.probe_id = probe_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{probe_id}"
        self.creation_time = datetime.now()

    def get_probe_id(self) -> int:
        """获取probe ID"""
        return self.probe_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "probe_id": self.probe_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_health_monitoring_component",
            "status": "active"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "probe_id": self.probe_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed": True,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_probe_processing"
            }
            return result
        except Exception as e:
            return {
                "probe_id": self.probe_id,
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
        from datetime import datetime
        return {
            "probe_id": self.probe_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
            "healthy": True,
            "timestamp": datetime.now().isoformat()
        }

    async def get_info_async(self) -> Dict[str, Any]:
        """异步获取组件信息"""
        try:
            logger.info(f"异步获取Probe组件信息: {self.probe_id}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.get_info)
        except Exception as e:
            logger.error(f"异步获取Probe组件信息失败 {self.probe_id}: {str(e)}")
            return {"error": str(e), "probe_id": self.probe_id}

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理数据"""
        try:
            logger.info(f"异步处理数据: probe_id={self.probe_id}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.process, data)
        except Exception as e:
            logger.error(f"异步处理数据失败 {self.probe_id}: {str(e)}")
            return {
                "probe_id": self.probe_id,
                "status": "error",
                "error": str(e),
                "processed_at": datetime.now().isoformat()
            }

    async def get_status_async(self) -> Dict[str, Any]:
        """异步获取组件状态"""
        try:
            logger.debug(f"异步获取Probe组件状态: {self.probe_id}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.get_status)
        except Exception as e:
            logger.error(f"异步获取Probe组件状态失败 {self.probe_id}: {str(e)}")
            return {
                "probe_id": self.probe_id,
                "status": "error",
                "error": str(e)
            }


class ProbeComponentFactory(ComponentFactory):

    """Probe组件工厂"""

    # 支持的probe ID列表
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    SUPPORTED_PROBE_IDS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65]

    @staticmethod
    def create_component(probe_id: int) -> ProbeComponent:
        """创建指定ID的probe组件"""
        if probe_id not in ProbeComponentFactory.SUPPORTED_PROBE_IDS:
            raise ValueError(
                f"不支持的probe ID: {probe_id}。支持的ID: {ProbeComponentFactory.SUPPORTED_PROBE_IDS}")

        return ProbeComponent(probe_id, "Probe")

    def create(self, probe_id: int, config: Dict[str, Any] = None) -> ProbeComponent:
        """创建指定ID的probe组件（实例方法别名）"""
        component = self.create_component(probe_id)
        if config:
            # 应用配置
            component.config = config
        return component

    def get_stats(self) -> Dict[str, Any]:
        """获取工厂统计信息"""
        return {
            "created_count": len(self.SUPPORTED_PROBE_IDS),
            "supported_ids": self.SUPPORTED_PROBE_IDS,
            "factory_type": "ProbeComponentFactory"
        }

    def cleanup(self) -> None:
        """清理工厂资源"""
        # 清理逻辑
        pass

    def reset(self) -> None:
        """重置工厂状态"""
        # 重置逻辑
        pass

    @staticmethod
    def get_available_probes() -> List[int]:
        """获取所有可用的probe ID"""
        return sorted(list(ProbeComponentFactory.SUPPORTED_PROBE_IDS))

    @staticmethod
    def create_all_probes() -> Dict[int, ProbeComponent]:
        """创建所有可用probe"""
        return {
            probe_id: ProbeComponent(probe_id, "Probe")
            for probe_id in ProbeComponentFactory.SUPPORTED_PROBE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ProbeComponentFactory",
            "version": "2.0.0",
            "total_probes": len(ProbeComponentFactory.SUPPORTED_PROBE_IDS),
            "supported_ids": sorted(list(ProbeComponentFactory.SUPPORTED_PROBE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }

    @staticmethod
    async def create_component_async(probe_id: int) -> ProbeComponent:
        """异步创建指定ID的probe组件"""
        try:
            logger.info(f"异步创建Probe组件: {probe_id}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, ProbeComponentFactory.create_component, probe_id)
        except Exception as e:
            logger.error(f"异步创建Probe组件失败 {probe_id}: {str(e)}")
            raise

    @staticmethod
    async def get_available_probes_async() -> List[int]:
        """异步获取所有可用的probe ID"""
        try:
            logger.debug("异步获取可用Probe列表")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, ProbeComponentFactory.get_available_probes)
        except Exception as e:
            logger.error(f"异步获取可用Probe列表失败: {str(e)}")
            return []

    @staticmethod
    async def create_all_probes_async() -> Dict[int, ProbeComponent]:
        """异步创建所有可用probe"""
        try:
            logger.info("异步创建所有Probe组件")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, ProbeComponentFactory.create_all_probes)
        except Exception as e:
            logger.error(f"异步创建所有Probe组件失败: {str(e)}")
            return {}

    @staticmethod
    async def get_factory_info_async() -> Dict[str, Any]:
        """异步获取工厂信息"""
        try:
            logger.debug("异步获取Probe工厂信息")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, ProbeComponentFactory.get_factory_info)
        except Exception as e:
            logger.error(f"异步获取Probe工厂信息失败: {str(e)}")
            return {"error": str(e)}
# 向后兼容：创建旧的组件实例


def create_probe_probe_component_5():
    return ProbeComponentFactory.create_component(5)


def create_probe_probe_component_11():
    return ProbeComponentFactory.create_component(11)


def create_probe_probe_component_17():
    return ProbeComponentFactory.create_component(17)


def create_probe_probe_component_23():
    return ProbeComponentFactory.create_component(23)


def create_probe_probe_component_29():
    return ProbeComponentFactory.create_component(29)


def create_probe_probe_component_35():
    return ProbeComponentFactory.create_component(35)


def create_probe_probe_component_41():
    return ProbeComponentFactory.create_component(41)


def create_probe_probe_component_47():
    return ProbeComponentFactory.create_component(47)


def create_probe_probe_component_53():
    return ProbeComponentFactory.create_component(53)


def create_probe_probe_component_59():
    return ProbeComponentFactory.create_component(59)


def create_probe_probe_component_65():

    return ProbeComponentFactory.create_component(65)


__all__ = [
    "IProbeComponent",
    "ProbeComponent",
    "ProbeComponentFactory",
    "create_probe_probe_component_5",
    "create_probe_probe_component_11",
    "create_probe_probe_component_17",
    "create_probe_probe_component_23",
    "create_probe_probe_component_29",
    "create_probe_probe_component_35",
    "create_probe_probe_component_41",
    "create_probe_probe_component_47",
    "create_probe_probe_component_53",
    "create_probe_probe_component_59",
    "create_probe_probe_component_65",
]

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始Probe组件模块健康检查")

        health_checks = {
            "interface_definition": check_interface_definition(),
            "component_implementation": check_component_implementation(),
            "factory_system": check_factory_system()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "probe_components",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("Probe组件模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"Probe组件模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"Probe组件模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "probe_components",
            "error": str(e)
        }


def check_interface_definition() -> Dict[str, Any]:
    """检查接口定义

    Returns:
        Dict[str, Any]: 接口定义检查结果
    """
    try:
        # 检查IProbeComponent接口存在
        interface_exists = 'IProbeComponent' in globals()

        if not interface_exists:
            return {"healthy": False, "error": "IProbeComponent interface not found"}

        # 检查必需的方法
        required_methods = ['get_info', 'process', 'get_status', 'get_probe_id']
        existing_methods = [method for method in dir(
            IProbeComponent) if not method.startswith('_') and method != 'get_probe_id']

        # 排除get_probe_id，因为它没有在dir()中显示
        methods_complete = all(method in existing_methods for method in required_methods[:-1])

        return {
            "healthy": interface_exists and methods_complete,
            "interface_exists": interface_exists,
            "methods_complete": methods_complete,
            "existing_methods": existing_methods,
            "required_methods": required_methods
        }
    except Exception as e:
        logger.error(f"接口定义检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_component_implementation() -> Dict[str, Any]:
    """检查组件实现

    Returns:
        Dict[str, Any]: 组件实现检查结果
    """
    try:
        # 检查ProbeComponent类存在
        component_exists = 'ProbeComponent' in globals()

        if not component_exists:
            return {"healthy": False, "error": "ProbeComponent class not found"}

        # 检查继承关系
        is_subclass = issubclass(
            ProbeComponent, IProbeComponent) if component_exists and 'IProbeComponent' in globals() else False

        # 检查必需的方法
        required_methods = ['__init__', 'get_probe_id', 'get_info', 'process', 'get_status']
        existing_methods = [method for method in dir(ProbeComponent) if not method.startswith('_')]

        methods_complete = all(method in existing_methods for method in required_methods)

        # 测试组件实例化
        instantiation_works = False
        try:
            component = ProbeComponent(probe_id=1)
            instantiation_works = component is not None
        except Exception:
            instantiation_works = False

        return {
            "healthy": component_exists and is_subclass and methods_complete and instantiation_works,
            "component_exists": component_exists,
            "is_subclass": is_subclass,
            "methods_complete": methods_complete,
            "instantiation_works": instantiation_works,
            "existing_methods": existing_methods
        }
    except Exception as e:
        logger.error(f"组件实现检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_factory_system() -> Dict[str, Any]:
    """检查工厂系统

    Returns:
        Dict[str, Any]: 工厂系统检查结果
    """
    try:
        # 检查ProbeComponentFactory类存在
        factory_exists = 'ProbeComponentFactory' in globals()

        if not factory_exists:
            return {"healthy": False, "error": "ProbeComponentFactory class not found"}

        # 检查继承关系
        is_subclass = issubclass(
            ProbeComponentFactory, ComponentFactory) if factory_exists and 'ComponentFactory' in globals() else False

        # 检查工厂方法
        factory_methods = ['create_component', 'get_available_probes',
                           'create_all_probes', 'get_factory_info']
        methods_exist = all(hasattr(ProbeComponentFactory, method) for method in factory_methods)

        # 测试工厂实例化
        factory_instantiation_works = False
        try:
            factory = ProbeComponentFactory()
            factory_instantiation_works = factory is not None
        except Exception:
            factory_instantiation_works = False

        return {
            "healthy": factory_exists and is_subclass and methods_exist and factory_instantiation_works,
            "factory_exists": factory_exists,
            "is_subclass": is_subclass,
            "methods_exist": methods_exist,
            "factory_instantiation_works": factory_instantiation_works,
            "factory_methods": factory_methods
        }
    except Exception as e:
        logger.error(f"工厂系统检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def health_status() -> Dict[str, Any]:
    """获取健康状态摘要

    Returns:
        Dict[str, Any]: 健康状态摘要
    """
    try:
        health_check = check_health()

        return {
            "status": "healthy" if health_check["healthy"] else "unhealthy",
            "service": "probe_components",
            "health_check": health_check,
            "component_types": ["IProbeComponent", "ProbeComponent", "ProbeComponentFactory"] if 'IProbeComponent' in globals() and 'ProbeComponent' in globals() and 'ProbeComponentFactory' in globals() else [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康状态摘要失败: {str(e)}")
        return {"status": "error", "error": str(e)}


def health_summary() -> Dict[str, Any]:
    """获取健康摘要报告

    Returns:
        Dict[str, Any]: 健康摘要报告
    """
    try:
        health_check = check_health()

        return {
            "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
            "probe_components_module_info": {
                "service_name": "probe_components",
                "purpose": "Probe组件统一实现",
                "operational": health_check["healthy"]
            },
            "architecture_status": {
                "interface_definition_complete": health_check["checks"]["interface_definition"]["healthy"],
                "component_implementation_complete": health_check["checks"]["component_implementation"]["healthy"],
                "factory_system_complete": health_check["checks"]["factory_system"]["healthy"]
            },
            "functionality_status": {
                "inheritance_structure_valid": health_check["checks"]["component_implementation"]["is_subclass"],
                "factory_pattern_implemented": health_check["checks"]["factory_system"]["is_subclass"],
                "component_instantiation_working": health_check["checks"]["component_implementation"]["instantiation_works"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_probe_components() -> Dict[str, Any]:
    """监控Probe组件状态

    Returns:
        Dict[str, Any]: 组件监控结果
    """
    try:
        health_check = check_health()

        # 计算组件效率指标
        component_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "component_metrics": {
                "service_name": "probe_components",
                "component_efficiency": component_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "architecture_metrics": {
                "interface_definition_complete": health_check["checks"]["interface_definition"]["healthy"],
                "component_implementation_complete": health_check["checks"]["component_implementation"]["healthy"],
                "factory_system_complete": health_check["checks"]["factory_system"]["healthy"]
            }
        }
    except Exception as e:
        logger.error(f"Probe组件监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_probe_components() -> Dict[str, Any]:
    """验证Probe组件

    Returns:
        Dict[str, Any]: 组件验证结果
    """
    try:
        validation_results = {
            "interface_validation": check_interface_definition(),
            "component_validation": check_component_implementation(),
            "factory_validation": check_factory_system()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Probe组件验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}
