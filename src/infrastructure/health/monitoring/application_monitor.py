"""
application_monitor 模块

提供 application_monitor 相关功能和接口。
"""

import logging

# 提取参数

from .application_monitor_config import ApplicationMonitorConfig, AlertHandler, InfluxDBConfig, PrometheusConfig
from .application_monitor_core import ApplicationMonitor as ApplicationMonitorCore
from .application_monitor_metrics import ApplicationMonitorMetricsMixin
from .application_monitor_monitoring import ApplicationMonitorMonitoringMixin
from datetime import datetime
from typing import Dict, Any
"""
基础设施层 - 应用监控组件

application_monitor 模块

应用性能监控器的主要入口模块，组合所有功能组件。
"""

logger = logging.getLogger(__name__)


class ApplicationMonitor(
    ApplicationMonitorCore,
    ApplicationMonitorMonitoringMixin,
    ApplicationMonitorMetricsMixin
):
    """
    应用性能监控器

    组合了核心功能、监控功能和指标管理功能。
    提供应用性能监控、错误跟踪、指标收集等功能。
    支持Prometheus和InfluxDB集成。
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        初始化应用监控器

        Args:
            config: ApplicationMonitorConfig对象
            **kwargs: 向后兼容的传统参数
        """
        # 如果提供了传统参数，转换为配置对象
        if config is None and kwargs:
            config = self._create_config_from_kwargs(kwargs)
        elif config is None:
            config = ApplicationMonitorConfig.create_default()

        # 调用父类的初始化方法
        super().__init__(config)

    def _create_config_from_kwargs(self, kwargs):
        """从传统参数创建配置对象"""
        app_name = kwargs.get('app_name', 'rqa2025')
        alert_handlers = kwargs.get('alert_handlers', [])
        influx_config = kwargs.get('influx_config')
        sample_rate = kwargs.get('sample_rate', 1.0)
        retention_policy = kwargs.get('retention_policy', '30d')
        influx_client_mock = kwargs.get('influx_client_mock')
        skip_thread = kwargs.get('skip_thread', False)
        registry = kwargs.get('registry')

        # 转换为配置对象
        alert_handler_objects = [
            AlertHandler(name=f"handler_{i}", handler=handler)
            for i, handler in enumerate(alert_handlers)
        ]

        influx_config_obj = None
        if influx_config:
            influx_config_obj = InfluxDBConfig()
            field_map = {
                'host': 'host',
                'port': 'port',
                'database': 'database',
                'username': 'username',
                'password': 'password',
                'ssl': 'ssl',
                'timeout': 'timeout',
                'enabled': 'enabled'
            }
            for source_key, target_field in field_map.items():
                if source_key in influx_config:
                    setattr(influx_config_obj, target_field, influx_config[source_key])

            # 兼容旧参数命名
            if 'url' in influx_config:
                influx_config_obj.host = influx_config['url']
            if 'bucket' in influx_config:
                influx_config_obj.database = influx_config['bucket']
            if 'token' in influx_config:
                influx_config_obj.password = influx_config['token']
            if 'org' in influx_config:
                influx_config_obj.username = influx_config['org']

        prometheus_config = PrometheusConfig(registry=registry)

        return ApplicationMonitorConfig(
            app_name=app_name,
            service_name=kwargs.get('service_name', app_name),
            alert_handlers=alert_handler_objects,
            influx_config=influx_config_obj,
            sample_rate=sample_rate,
            retention_policy=retention_policy,
            influx_client_mock=influx_client_mock,
            skip_thread=skip_thread,
            prometheus_config=prometheus_config
        )

# 模块级健康检查函数


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始应用监控器模块健康检查")

        health_checks = {
            "monitor_class": check_monitor_class(),
            "mixin_integration": check_mixin_integration(),
            "config_system": check_config_system()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "application_monitor",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("应用监控器模块健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"应用监控器模块健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"应用监控器模块健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "application_monitor",
            "error": str(e)
        }


def check_monitor_class() -> Dict[str, Any]:
    """检查监控器类定义

    Returns:
        Dict[str, Any]: 监控器类检查结果
    """
    try:
        # 检查ApplicationMonitor类存在
        monitor_class_exists = 'ApplicationMonitor' in globals()

        if not monitor_class_exists:
            return {"healthy": False, "error": "ApplicationMonitor class not found"}

        # 检查必需的方法（继承自父类）
        required_methods = ['__init__', 'initialize', 'get_component_info', 'is_healthy']
        existing_methods = [method for method in dir(
            ApplicationMonitor) if not method.startswith('_')]

        methods_complete = all(method in existing_methods for method in required_methods)

        # 测试类实例化
        instantiation_works = False
        try:
            monitor = ApplicationMonitor()
            instantiation_works = monitor is not None
        except Exception:
            instantiation_works = False

        # 检查继承关系
        base_classes = ['ApplicationMonitorCore',
                        'ApplicationMonitorMonitoringMixin', 'ApplicationMonitorMetricsMixin']
        inheritance_correct = all(
            hasattr(ApplicationMonitor, attr) for attr in ['initialize', 'get_component_info', 'is_healthy']
        )

        return {
            "healthy": monitor_class_exists and methods_complete and instantiation_works and inheritance_correct,
            "monitor_class_exists": monitor_class_exists,
            "methods_complete": methods_complete,
            "instantiation_works": instantiation_works,
            "inheritance_correct": inheritance_correct,
            "existing_methods": existing_methods
        }
    except Exception as e:
        logger.error(f"监控器类检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_mixin_integration() -> Dict[str, Any]:
    """检查Mixin集成

    Returns: Dict[str, Any]ixin集成检查结果
    """
    try:
        # 检查Mixin类存在
        mixin_classes = [
            'ApplicationMonitorCore',
            'ApplicationMonitorMonitoringMixin',
            'ApplicationMonitorMetricsMixin'
        ]

        mixins_exist = all(cls in globals() for cls in mixin_classes)

        if not mixins_exist:
            return {"healthy": False, "error": f"Missing mixin classes: {[cls for cls in mixin_classes if cls not in globals()]} "}

        # 检查多重继承
        mro = ApplicationMonitor.__mro__
        expected_in_mro = [ApplicationMonitor, ApplicationMonitorCore,
                           ApplicationMonitorMonitoringMixin, ApplicationMonitorMetricsMixin]
        inheritance_structure_correct = all(cls in mro for cls in expected_in_mro)

        # 测试方法解析
        method_resolution_works = False
        try:
            monitor = ApplicationMonitor()
            # 测试核心方法
            has_initialize = hasattr(monitor, 'initialize')
            has_get_component_info = hasattr(monitor, 'get_component_info')
            has_is_healthy = hasattr(monitor, 'is_healthy')
            has_get_metrics = hasattr(monitor, 'get_metrics')
            method_resolution_works = all(
                [has_initialize, has_get_component_info, has_is_healthy, has_get_metrics])
        except Exception:
            method_resolution_works = False

        return {
            "healthy": mixins_exist and inheritance_structure_correct and method_resolution_works,
            "mixins_exist": mixins_exist,
            "inheritance_structure_correct": inheritance_structure_correct,
            "method_resolution_works": method_resolution_works,
            "mixin_classes": mixin_classes,
            "mro_length": len(mro)
        }
    except Exception as e:
        logger.error(f"Mixin集成检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_config_system() -> Dict[str, Any]:
    """检查配置系统

    Returns:
        Dict[str, Any]: 配置系统检查结果
    """
    try:
        # 检查配置类存在
        config_class_exists = 'ApplicationMonitorConfig' in globals()

        if not config_class_exists:
            return {"healthy": False, "error": "ApplicationMonitorConfig class not found"}

        # 测试配置创建
        config_creation_works = False
        default_config_works = False
        kwargs_config_works = False

        try:
            # 测试默认配置
            default_config = ApplicationMonitorConfig.create_default()
            default_config_works = default_config is not None
        except Exception:
            default_config_works = False

        try:
            # 测试从参数创建配置
            config = ApplicationMonitor(app_name="test_app")
            config_creation_works = config is not None
        except Exception:
            config_creation_works = False

        try:
            # 测试传统参数转换
            monitor = ApplicationMonitor(app_name="test_app", sample_rate=0.5)
            kwargs_config_works = monitor is not None
        except Exception:
            kwargs_config_works = False

        return {
            "healthy": config_class_exists and default_config_works and config_creation_works and kwargs_config_works,
            "config_class_exists": config_class_exists,
            "default_config_works": default_config_works,
            "config_creation_works": config_creation_works,
            "kwargs_config_works": kwargs_config_works
        }
    except Exception as e:
        logger.error(f"配置系统检查失败: {str(e)}")
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
            "service": "application_monitor",
            "health_check": health_check,
            "mixin_count": 3,  # Core, Monitoring, Metrics
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
            "application_monitor_module_info": {
                "service_name": "application_monitor",
                "purpose": "应用性能监控器",
                "operational": health_check["healthy"]
            },
            "architecture_status": {
                "mixin_integration_complete": health_check["checks"]["mixin_integration"]["healthy"],
                "inheritance_structure_valid": health_check["checks"]["mixin_integration"]["inheritance_structure_correct"],
                "monitor_class_working": health_check["checks"]["monitor_class"]["healthy"]
            },
            "configuration_status": {
                "config_system_functional": health_check["checks"]["config_system"]["healthy"],
                "default_config_available": health_check["checks"]["config_system"]["default_config_works"],
                "parameter_conversion_working": health_check["checks"]["config_system"]["kwargs_config_works"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_application_monitor_module() -> Dict[str, Any]:
    """监控应用监控器模块状态

    Returns:
        Dict[str, Any]: 模块监控结果
    """
    try:
        health_check = check_health()

        # 计算模块效率指标
        module_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "module_metrics": {
                "service_name": "application_monitor",
                "module_efficiency": module_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "architecture_metrics": {
                "mixin_classes_integrated": len(ApplicationMonitor.__mro__) - 1,  # 排除object类
                "inheritance_structure_valid": health_check["checks"]["mixin_integration"]["inheritance_structure_correct"],
                "method_resolution_working": health_check["checks"]["mixin_integration"]["method_resolution_works"]
            }
        }
    except Exception as e:
        logger.error(f"应用监控器模块监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_application_monitor_config() -> Dict[str, Any]:
    """验证应用监控器配置

    Returns:
        Dict[str, Any]: 配置验证结果
    """
    try:
        validation_results = {
            "class_validation": _validate_monitor_classes(),
            "mixin_validation": _validate_mixin_system(),
            "config_validation": _validate_configuration_system()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"应用监控器配置验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_monitor_classes() -> Dict[str, Any]:
    """验证监控器类"""
    try:
        # 检查必需的类
        required_classes = ['ApplicationMonitor', 'ApplicationMonitorCore',
                            'ApplicationMonitorMonitoringMixin', 'ApplicationMonitorMetricsMixin', 'ApplicationMonitorConfig']
        classes_exist = all(cls in globals() for cls in required_classes)

        # 检查类是否可以实例化
        instantiation_tests = {}
        for cls_name in required_classes:
            if cls_name in globals():
                try:
                    cls = globals()[cls_name]
                    if cls_name == 'ApplicationMonitor':
                        instance = cls()
                        instantiation_tests[cls_name] = {"success": True}
                    elif cls_name == 'ApplicationMonitorConfig':
                        instance = cls.create_default()
                        instantiation_tests[cls_name] = {"success": True}
                    else:
                        # Mixin类不需要实例化
                        instantiation_tests[cls_name] = {"success": True}
                except Exception as e:
                    instantiation_tests[cls_name] = {"success": False, "error": str(e)}
            else:
                instantiation_tests[cls_name] = {"success": False, "error": "Class not found"}

        all_instantiable = all(test["success"] for test in instantiation_tests.values())

        return {
            "valid": classes_exist and all_instantiable,
            "classes_exist": classes_exist,
            "all_instantiable": all_instantiable,
            "instantiation_tests": instantiation_tests,
            "required_classes": required_classes
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_mixin_system() -> Dict[str, Any]:
    """验证Mixin系统"""
    try:
        # 检查MRO（方法解析顺序）
        mro = ApplicationMonitor.__mro__

        # 验证继承层次结构
        expected_classes = [ApplicationMonitor, ApplicationMonitorCore,
                            ApplicationMonitorMonitoringMixin, ApplicationMonitorMetricsMixin]
        inheritance_correct = all(cls in mro for cls in expected_classes)

        # 验证方法继承
        inherited_methods = set()
        for cls in mro:
            if cls != object:
                inherited_methods.update(
                    [method for method in dir(cls) if not method.startswith('_')])

        # 检查关键方法是否被继承
        key_methods = ['initialize', 'get_component_info', 'is_healthy', 'get_metrics', 'cleanup']
        methods_inherited = all(method in inherited_methods for method in key_methods)

        return {
            "valid": inheritance_correct and methods_inherited,
            "inheritance_correct": inheritance_correct,
            "methods_inherited": methods_inherited,
            "mro_length": len(mro),
            "inherited_method_count": len(inherited_methods),
            "key_methods": key_methods
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_configuration_system() -> Dict[str, Any]:
    """验证配置系统"""
    try:
        # 检查配置创建方法
        config_creation_methods = ['create_default']
        methods_exist = all(hasattr(ApplicationMonitorConfig, method)
                            for method in config_creation_methods)

        # 测试配置创建
        config_tests = {}
        try:
            default_config = ApplicationMonitorConfig.create_default()
            config_tests["default_config"] = {
                "success": True, "type": type(default_config).__name__}
        except Exception as e:
            config_tests["default_config"] = {"success": False, "error": str(e)}

        # 测试参数转换
        try:
            monitor = ApplicationMonitor(app_name="test", sample_rate=0.8)
            config_tests["parameter_conversion"] = {"success": True}
        except Exception as e:
            config_tests["parameter_conversion"] = {"success": False, "error": str(e)}

        all_configs_work = all(test["success"] for test in config_tests.values())

        return {
            "valid": methods_exist and all_configs_work,
            "methods_exist": methods_exist,
            "all_configs_work": all_configs_work,
            "config_tests": config_tests,
            "creation_methods": config_creation_methods
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}
