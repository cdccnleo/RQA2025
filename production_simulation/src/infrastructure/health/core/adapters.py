"""
adapters 模块

提供 adapters 相关功能和接口。
"""

import logging

from ..components.alert_components import AlertComponentFactory
from ..components.checker_components import CheckerComponentFactory
from ..components.health_checker import AsyncHealthCheckerComponent
from ..components.health_components import HealthComponentFactory
from ..components.monitor_components import MonitorComponentFactory
from ..database.database_health_monitor import DatabaseHealthMonitor
from .interfaces import IInfrastructureAdapter, IUnifiedInfrastructureInterface, InfrastructureAdapterFactory
from datetime import datetime
from typing import Any, Dict, Optional
"""
基础设施层 - 适配器模式实现

提供基础设施服务的适配器实现，支持统一访问不同的基础设施组件。
"""

logger = logging.getLogger(__name__)

# 导入健康检查组件


class BaseInfrastructureAdapter(IInfrastructureAdapter, IUnifiedInfrastructureInterface):
    """基础设施适配器基类

    提供基础设施适配器的通用实现。
    """

    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self._last_check_time = None
        self._service_status = "unknown"
        self._initialized = False
        logger.info(f"初始化基础设施适配器: {service_name}")

    def get_service_name(self) -> str:
        """获取服务名称"""
        return self.service_name

    def is_service_available(self) -> bool:
        """检查服务是否可用"""
        try:
            status = self.get_service_status()
            return status.get("available", False)
        except Exception as e:
            logger.error(f"检查服务可用性失败 {self.service_name}: {str(e)}")
            return False

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        try:
            # 基础状态信息
            status = {
                "service_name": self.service_name,
                "available": True,
                "status": self._service_status,
                "last_check": self._last_check_time.isoformat() if self._last_check_time else None,
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            }

            self._last_check_time = datetime.now()
            return status

        except Exception as e:
            logger.error(f"获取服务状态失败 {self.service_name}: {str(e)}")
            return {
                "service_name": self.service_name,
                "available": False,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行服务操作"""
        try:
            logger.info(f"执行操作 {self.service_name}.{operation}")
            # 合并参数
            all_params = {}
            if params:
                all_params.update(params)
            all_params.update(kwargs)
            
            # 基础实现，子类应该重写此方法
            result = {
                "operation": operation, 
                "status": "success",
                "success": True
            }
            result.update(all_params)
            return result
        except Exception as e:
            logger.error(f"执行操作失败 {self.service_name}.{operation}: {str(e)}")
            raise

    async def execute_operation_async(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """异步执行服务操作"""
        try:
            logger.info(f"异步执行操作 {self.service_name}.{operation}")
            # 合并参数
            all_params = {}
            if params:
                all_params.update(params)
            all_params.update(kwargs)
            
            # 基础实现，子类应该重写此方法
            result = {
                "operation": operation, 
                "status": "success",
                "success": True,
                "async": True
            }
            result.update(all_params)
            return result
        except Exception as e:
            logger.error(f"异步执行操作失败 {self.service_name}.{operation}: {str(e)}")
            raise

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info(f"开始基础设施适配器健康检查: {self.service_name}")

            health_checks = {
                "service_availability": self.check_service_availability(),
                "service_status": self.check_service_status_health(),
                "configuration": self.check_adapter_configuration()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": self.service_name,
                "adapter_type": self.__class__.__name__,
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning(f"基础设施适配器健康检查发现问题: {self.service_name}")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"基础设施适配器健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"基础设施适配器健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": self.service_name,
                "adapter_type": self.__class__.__name__,
                "error": str(e)
            }

    def check_service_availability(self) -> Dict[str, Any]:
        """检查服务可用性

        Returns:
            Dict[str, Any]: 服务可用性检查结果
        """
        try:
            available = self.is_service_available()
            return {
                "healthy": available,
                "service_available": available,
                "service_name": self.service_name
            }
        except Exception as e:
            logger.error(f"服务可用性检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_service_status_health(self) -> Dict[str, Any]:
        """检查服务状态健康

        Returns:
            Dict[str, Any]: 服务状态健康检查结果
        """
        try:
            status = self.get_service_status()
            is_healthy = status.get("available", False) and status.get("status") != "error"

            return {
                "healthy": is_healthy,
                "service_status": status.get("status"),
                "last_check": status.get("last_check"),
                "status_details": status
            }
        except Exception as e:
            logger.error(f"服务状态健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_adapter_configuration(self) -> Dict[str, Any]:
        """检查适配器配置

        Returns:
            Dict[str, Any]: 配置健康检查结果
        """
        try:
            # 检查基本配置
            config_exists = self.config is not None
            service_name_valid = bool(self.service_name and len(self.service_name) > 0)

            return {
                "healthy": config_exists and service_name_valid,
                "config_exists": config_exists,
                "service_name_valid": service_name_valid,
                "config_keys": list(self.config.keys()) if self.config else []
            }
        except Exception as e:
            logger.error(f"适配器配置检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            health_check = self.check_health()
            service_status = self.get_service_status()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "service_name": self.service_name,
                "adapter_type": self.__class__.__name__,
                "service_status": service_status.get("status"),
                "last_check": service_status.get("last_check"),
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "service": self.service_name, "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            service_status = self.get_service_status()

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "adapter_info": {
                    "service_name": self.service_name,
                    "adapter_type": self.__class__.__name__,
                    "operational": health_check["healthy"]
                },
                "service_status": {
                    "current_status": service_status.get("status"),
                    "last_check_time": service_status.get("last_check"),
                    "availability": service_status.get("available", False)
                },
                "configuration": {
                    "config_provided": self.config is not None,
                    "config_keys_count": len(self.config) if self.config else 0
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "service": self.service_name, "error": str(e)}

    def monitor_adapter(self) -> Dict[str, Any]:
        """监控适配器状态

        Returns:
            Dict[str, Any]: 适配器监控结果
        """
        try:
            health_check = self.check_health()
            service_status = self.get_service_status()

            # 计算适配器效率指标
            adapter_efficiency = 1.0 if health_check["healthy"] else 0.0

            return {
                "healthy": health_check["healthy"],
                "adapter_metrics": {
                    "service_name": self.service_name,
                    "adapter_type": self.__class__.__name__,
                    "adapter_efficiency": adapter_efficiency,
                    "operational_status": "active" if health_check["healthy"] else "inactive"
                },
                "service_metrics": {
                    "service_available": service_status.get("available", False),
                    "service_status": service_status.get("status"),
                    "last_check_time": service_status.get("last_check")
                }
            }
        except Exception as e:
            logger.error(f"适配器监控失败: {str(e)}")
            return {"healthy": False, "service": self.service_name, "error": str(e)}

    def validate_adapter_config(self) -> Dict[str, Any]:
        """验证适配器配置

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            validation_results = {
                "service_name_validation": self._validate_service_name(),
                "config_validation": self._validate_config(),
                "adapter_initialization": self._validate_adapter_initialization()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"适配器配置验证失败: {str(e)}")
            return {"valid": False, "service": self.service_name, "error": str(e)}

    def _validate_service_name(self) -> Dict[str, Any]:
        """验证服务名称"""
        try:
            name_exists = bool(self.service_name)
            name_valid = name_exists and len(self.service_name.strip()) > 0
            name_not_empty = name_exists and self.service_name.strip() == self.service_name

            return {
                "valid": name_valid and name_not_empty,
                "name_exists": name_exists,
                "name_not_empty": name_not_empty,
                "service_name": self.service_name
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_config(self) -> Dict[str, Any]:
        """验证配置"""
        try:
            config_is_dict = isinstance(self.config, dict)
            config_not_none = self.config is not None

            return {
                "valid": config_not_none and config_is_dict,
                "config_not_none": config_not_none,
                "config_is_dict": config_is_dict,
                "config_keys_count": len(self.config) if config_is_dict else 0
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_adapter_initialization(self) -> Dict[str, Any]:
        """验证适配器初始化"""
        try:
            # 检查必需属性
            has_service_name = hasattr(self, 'service_name')
            has_config = hasattr(self, 'config')
            has_last_check_time = hasattr(self, '_last_check_time')
            has_service_status = hasattr(self, '_service_status')

            all_attributes_present = all([
                has_service_name, has_config, has_last_check_time, has_service_status
            ])

            return {
                "valid": all_attributes_present,
                "has_service_name": has_service_name,
                "has_config": has_config,
                "has_last_check_time": has_last_check_time,
                "has_service_status": has_service_status
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    # IUnifiedInfrastructureInterface 实现
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化组件

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            if config:
                self.config.update(config)

            # 初始化基础属性
            self._initialized = True
            self._start_time = datetime.now()

            logger.info(f"基础设施适配器 {self.service_name} 初始化成功")
            return True
        except Exception as e:
            logger.error(f"基础设施适配器 {self.service_name} 初始化失败: {str(e)}")
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        return {
            "component_type": f"{self.__class__.__name__}",
            "description": f"基础设施适配器 - {self.service_name}",
            "version": "1.0.0",
            "service_name": self.service_name,
            "initialized": getattr(self, '_initialized', False),
            "start_time": getattr(self, '_start_time', datetime.now()).isoformat(),
            "uptime_seconds": (datetime.now() - getattr(self, '_start_time', datetime.now())).total_seconds()
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """
        try:
            # 检查基本初始化状态
            if not getattr(self, '_initialized', False):
                return False

            # 检查服务可用性
            return self.is_service_available()
        except Exception as e:
            logger.error(f"检查适配器 {self.service_name} 健康状态失败: {str(e)}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """
        current_time = datetime.now()

        return {
            "service_name": self.service_name,
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "service_status": self._service_status,
            "uptime_seconds": (current_time - getattr(self, '_start_time', current_time)).total_seconds(),
            "component_status": {
                "initialized": getattr(self, '_initialized', False),
                "healthy": self.is_healthy()
            },
            "config_info": {
                "has_config": bool(self.config),
                "config_keys": list(self.config.keys()) if self.config else []
            }
        }

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            # 重置状态
            self._last_check_time = None
            self._service_status = "unknown"

            # 保持初始化状态，但清理运行时数据
            logger.info(f"基础设施适配器 {self.service_name} 资源清理完成")
            return True
        except Exception as e:
            logger.error(f"基础设施适配器 {self.service_name} 资源清理失败: {str(e)}")
            return False


class CacheAdapter(BaseInfrastructureAdapter):
    """缓存服务适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("cache", config)
        self._cache_backend = self.config.get("backend", "memory")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行缓存操作"""
        if operation == "get":
            return self._cache_get(kwargs.get("key"))
        elif operation == "set":
            return self._cache_set(kwargs.get("key"), kwargs.get("value"), kwargs.get("ttl"))
        elif operation == "delete":
            return self._cache_delete(kwargs.get("key"))
        elif operation == "clear":
            return self._cache_clear()
        else:
            return super().execute_operation(operation, **kwargs)

    def _cache_get(self, key: str) -> Any:
        """获取缓存值"""
        # 模拟缓存操作，实际实现应连接真实的缓存服务
        logger.debug(f"缓存获取: {key}")
        return {"key": key, "value": None, "found": False}

    def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        logger.debug(f"缓存设置: {key} = {value}, ttl={ttl}")
        return True

    def _cache_delete(self, key: str) -> bool:
        """删除缓存值"""
        logger.debug(f"缓存删除: {key}")
        return True

    def _cache_clear(self) -> bool:
        """清空缓存"""
        logger.debug("缓存清空")
        return True


class DatabaseAdapter(BaseInfrastructureAdapter):
    """数据库服务适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("database", config)
        self._db_type = self.config.get("type", "postgresql")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行数据库操作"""
        if operation == "query":
            return self._db_query(kwargs.get("sql"), kwargs.get("params", []))
        elif operation == "execute":
            return self._db_execute(kwargs.get("sql"), kwargs.get("params", []))
        elif operation == "health_check":
            return self._db_health_check()
        else:
            return super().execute_operation(operation, **kwargs)

    def _db_query(self, sql: str, params: list) -> Dict[str, Any]:
        """执行查询"""
        logger.debug(f"数据库查询: {sql}")
        return {"sql": sql, "params": params, "rows": [], "count": 0}

    def _db_execute(self, sql: str, params: list) -> Dict[str, Any]:
        """执行语句"""
        logger.debug(f"数据库执行: {sql}")
        return {"sql": sql, "params": params, "affected_rows": 0}

    def _db_health_check(self) -> Dict[str, Any]:
        """数据库健康检查"""
        logger.debug("数据库健康检查")
        return {
            "status": "healthy",
            "connection_count": 1,
            "response_time": 0.001,
            "timestamp": datetime.now().isoformat()
        }


class MonitoringAdapter(BaseInfrastructureAdapter):
    """监控服务适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("monitoring", config)
        self._metrics_collected = 0

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行监控操作"""
        if operation == "record_metric":
            return self._record_metric(kwargs.get("name"), kwargs.get("value"), kwargs.get("tags", {}))
        elif operation == "get_metrics":
            return self._get_metrics(kwargs.get("name"), kwargs.get("time_range"))
        elif operation == "alert":
            return self._send_alert(kwargs.get("message"), kwargs.get("level", "info"))
        else:
            return super().execute_operation(operation, **kwargs)

    def _record_metric(self, name: str, value: float, tags: Dict[str, str]) -> bool:
        """记录指标"""
        self._metrics_collected += 1
        logger.debug(f"记录指标: {name} = {value}, tags={tags}")
        return True

    def _get_metrics(self, name: str, time_range: Optional[tuple]) -> Dict[str, Any]:
        """获取指标数据"""
        logger.debug(f"获取指标: {name}, time_range={time_range}")
        return {
            "name": name,
            "data_points": [],
            "time_range": time_range,
            "count": 0
        }

    def _send_alert(self, message: str, level: str) -> bool:
        """发送告警"""
        logger.info(f"发送告警 [{level}]: {message}")
        return True


class LoggingAdapter(BaseInfrastructureAdapter):
    """日志服务适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("logging", config)
        self._log_level = self.config.get("level", "INFO")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行日志操作"""
        if operation == "log":
            return self._log_message(kwargs.get("level", "info"), kwargs.get("message"), kwargs.get("extra", {}))
        elif operation == "query_logs":
            return self._query_logs(kwargs.get("filters", {}))
        elif operation == "rotate":
            return self._rotate_logs()
        else:
            return super().execute_operation(operation, **kwargs)

    def _log_message(self, level: str, message: str, extra: Dict[str, Any]) -> bool:
        """记录日志消息"""
        logger.log(getattr(logging, level.upper(), logging.INFO), f"{message} {extra}")
        return True

    def _query_logs(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """查询日志"""
        logger.debug(f"查询日志: {filters}")
        return {
            "filters": filters,
            "logs": [],
            "count": 0
        }

    def _rotate_logs(self) -> bool:
        """轮转日志"""
        logger.info("执行日志轮转")
        return True


class HealthCheckerAdapter(BaseInfrastructureAdapter):
    """健康检查器适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("health_checker", config)
        self._checker = None

    def connect(self) -> bool:
        """连接到健康检查器"""
        try:
            self._checker = AsyncHealthCheckerComponent(self.config)
            self._service_status = "connected"
            logger.info("健康检查器适配器连接成功")
            return True
        except Exception as e:
            logger.error(f"健康检查器适配器连接失败: {e}")
            self._service_status = "error"
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self._checker:
                self._checker.cleanup()
            self._service_status = "disconnected"
            logger.info("健康检查器适配器断开连接")
            return True
        except Exception as e:
            logger.error(f"健康检查器适配器断开连接失败: {e}")
            return False

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行操作"""
        # 合并参数
        all_params = {}
        if params:
            all_params.update(params)
        all_params.update(kwargs)
        
        if not self._checker:
            # 如果未连接，返回模拟结果而不是抛出异常
            if operation == "check_health":
                return {"status": "mock", "healthy": True, "message": "模拟健康检查结果"}
            elif operation == "check_service":
                service_name = all_params.get("service_name", "")
                return {"service": service_name, "status": "mock", "healthy": True}
            elif operation == "get_metrics":
                return {"metrics": "mock_data", "status": "simulated"}
            else:
                return {"operation": operation, "status": "not_connected", "message": "健康检查器未连接"}

        try:
            if operation == "check_health":
                return self._checker.check_health_async()
            elif operation == "check_service":
                service_name = all_params.get("service_name", "")
                return self._checker.check_service_async(service_name)
            elif operation == "get_metrics":
                return self._checker.get_metrics()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行健康检查器操作失败: {operation}, 错误: {e}")
            raise


class DatabaseMonitorAdapter(BaseInfrastructureAdapter):
    """数据库监控器适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("database_monitor", config)
        self._monitor = None

    def connect(self) -> bool:
        """连接到数据库监控器"""
        try:
            # 这里需要一个数据管理器实例，暂时使用None
            # 在实际使用时应该从配置或依赖注入获取
            data_manager = self.config.get("data_manager")
            monitor_config = self.config.get("monitor")
            self._monitor = DatabaseHealthMonitor(data_manager, monitor_config)
            self._service_status = "connected"
            logger.info("数据库监控器适配器连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库监控器适配器连接失败: {e}")
            self._service_status = "error"
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self._monitor:
                self._monitor.cleanup()
            self._service_status = "disconnected"
            logger.info("数据库监控器适配器断开连接")
            return True
        except Exception as e:
            logger.error(f"数据库监控器适配器断开连接失败: {e}")
            return False

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行操作"""
        if not self._monitor:
            raise RuntimeError("数据库监控器未连接")

        try:
            if operation == "check_health":
                return self._monitor.check_health_async()
            elif operation == "get_metrics":
                return self._monitor.get_metrics()
            elif operation == "get_health_report":
                return self._monitor.get_health_report()
            elif operation == "start_monitoring":
                return self._monitor.start_monitoring_async()
            elif operation == "stop_monitoring":
                return self._monitor.stop_monitoring_async()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行数据库监控器操作失败: {operation}, 错误: {e}")
            raise


class AlertComponentAdapter(BaseInfrastructureAdapter):
    """Alert组件适配器

    提供Alert组件工厂的统一访问接口。
    """

    def __init__(self, service_name: str = "alert_factory", config: Optional[Dict[str, Any]] = None):
        super().__init__(service_name, config)
        self._alert_factory = AlertComponentFactory()
        logger.info(f"初始化Alert组件适配器: {service_name}")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行Alert组件操作"""
        try:
            logger.debug(f"执行Alert组件操作: {operation}")
            # 合并参数
            all_params = {}
            if params:
                all_params.update(params)
            all_params.update(kwargs)

            if operation == "create_component":
                alert_id = all_params.get("alert_id")
                if alert_id is None:
                    raise ValueError("创建Alert组件需要alert_id参数")
                return self._alert_factory.create_component(alert_id)
            elif operation == "get_available_alerts":
                return self._alert_factory.get_available_alerts()
            elif operation == "get_factory_info":
                return self._alert_factory.get_factory_info()
            elif operation == "create_all_alerts":
                return self._alert_factory.create_all_alerts()
            elif operation == "get_factory_info_async":
                return self._alert_factory.get_factory_info_async()
            elif operation == "create_component_async":
                alert_id = all_params.get("alert_id")
                if alert_id is None:
                    raise ValueError("异步创建Alert组件需要alert_id参数")
                return self._alert_factory.create_component_async(alert_id)
            elif operation == "get_available_alerts_async":
                return self._alert_factory.get_available_alerts_async()
            elif operation == "get_info":
                # 提供工厂信息作为info
                return self._alert_factory.get_factory_info()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行Alert组件操作失败: {operation}, 错误: {e}")
            raise

    def get_service_metrics(self) -> Dict[str, Any]:
        """获取Alert组件指标"""
        try:
            factory_info = self._alert_factory.get_factory_info()
            return {
                "supported_alert_types": factory_info.get("total_alerts", 0),
                "factory_status": "active",
                "alert_ids": factory_info.get("supported_ids", []),
                "last_operation_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取Alert组件指标失败: {e}")
            return {"error": str(e)}


class CheckerComponentAdapter(BaseInfrastructureAdapter):
    """Checker组件适配器

    提供Checker组件工厂的统一访问接口。
    """

    def __init__(self, service_name: str = "checker_factory", config: Optional[Dict[str, Any]] = None):
        super().__init__(service_name, config)
        self._checker_factory = CheckerComponentFactory()
        logger.info(f"初始化Checker组件适配器: {service_name}")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行Checker组件操作"""
        try:
            logger.debug(f"执行Checker组件操作: {operation}")

            if operation == "create_component":
                checker_id = kwargs.get("checker_id")
                if checker_id is None:
                    raise ValueError("创建Checker组件需要checker_id参数")
                return self._checker_factory.create_component(checker_id)
            elif operation == "get_available_checkers":
                return self._checker_factory.get_available_checkers()
            elif operation == "get_factory_info":
                return self._checker_factory.get_factory_info()
            elif operation == "create_all_checkers":
                return self._checker_factory.create_all_checkers()
            elif operation == "get_factory_info_async":
                return self._checker_factory.get_factory_info_async()
            elif operation == "create_component_async":
                checker_id = kwargs.get("checker_id")
                if checker_id is None:
                    raise ValueError("异步创建Checker组件需要checker_id参数")
                return self._checker_factory.create_component_async(checker_id)
            elif operation == "get_available_checkers_async":
                return self._checker_factory.get_available_checkers_async()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行Checker组件操作失败: {operation}, 错误: {e}")
            raise

    def get_service_metrics(self) -> Dict[str, Any]:
        """获取Checker组件指标"""
        try:
            factory_info = self._checker_factory.get_factory_info()
            return {
                "supported_checker_types": factory_info.get("total_checkers", 0),
                "factory_status": "active",
                "checker_ids": factory_info.get("supported_ids", []),
                "last_operation_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取Checker组件指标失败: {e}")
            return {"error": str(e)}


class HealthComponentAdapter(BaseInfrastructureAdapter):
    """Health组件适配器

    提供Health组件工厂的统一访问接口。
    """

    def __init__(self, service_name: str = "health_factory", config: Optional[Dict[str, Any]] = None):
        super().__init__(service_name, config)
        self._health_factory = HealthComponentFactory()
        logger.info(f"初始化Health组件适配器: {service_name}")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行Health组件操作"""
        try:
            logger.debug(f"执行Health组件操作: {operation}")

            if operation == "create_component":
                health_id = kwargs.get("health_id")
                if health_id is None:
                    raise ValueError("创建Health组件需要health_id参数")
                return self._health_factory.create_component(health_id)
            elif operation == "get_available_healths":
                return self._health_factory.get_available_healths()
            elif operation == "get_factory_info":
                return self._health_factory.get_factory_info()
            elif operation == "create_all_healths":
                return self._health_factory.create_all_healths()
            elif operation == "get_factory_info_async":
                return self._health_factory.get_factory_info_async()
            elif operation == "create_component_async":
                health_id = kwargs.get("health_id")
                if health_id is None:
                    raise ValueError("异步创建Health组件需要health_id参数")
                return self._health_factory.create_component_async(health_id)
            elif operation == "get_available_healths_async":
                return self._health_factory.get_available_healths_async()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行Health组件操作失败: {operation}, 错误: {e}")
            raise

    def get_service_metrics(self) -> Dict[str, Any]:
        """获取Health组件指标"""
        try:
            factory_info = self._health_factory.get_factory_info()
            return {
                "supported_health_types": factory_info.get("total_healths", 0),
                "factory_status": "active",
                "health_ids": factory_info.get("supported_ids", []),
                "last_operation_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取Health组件指标失败: {e}")
            return {"error": str(e)}


class MonitorComponentAdapter(BaseInfrastructureAdapter):
    """Monitor组件适配器

    提供Monitor组件工厂的统一访问接口。
    """

    def __init__(self, service_name: str = "monitor_factory", config: Optional[Dict[str, Any]] = None):
        super().__init__(service_name, config)
        self._monitor_factory = MonitorComponentFactory()
        logger.info(f"初始化Monitor组件适配器: {service_name}")

    def execute_operation(self, operation: str, params: Dict[str, Any] = None, **kwargs) -> Any:
        """执行Monitor组件操作"""
        try:
            logger.debug(f"执行Monitor组件操作: {operation}")

            if operation == "create_component":
                monitor_id = kwargs.get("monitor_id")
                if monitor_id is None:
                    raise ValueError("创建Monitor组件需要monitor_id参数")
                return self._monitor_factory.create_component(monitor_id)
            elif operation == "get_available_monitors":
                return self._monitor_factory.get_available_monitors()
            elif operation == "get_factory_info":
                return self._monitor_factory.get_factory_info()
            elif operation == "create_all_monitors":
                return self._monitor_factory.create_all_monitors()
            elif operation == "get_factory_info_async":
                return self._monitor_factory.get_factory_info_async()
            elif operation == "create_component_async":
                monitor_id = kwargs.get("monitor_id")
                if monitor_id is None:
                    raise ValueError("异步创建Monitor组件需要monitor_id参数")
                return self._monitor_factory.create_component_async(monitor_id)
            elif operation == "get_available_monitors_async":
                return self._monitor_factory.get_available_monitors_async()
            else:
                raise ValueError(f"不支持的操作: {operation}")
        except Exception as e:
            logger.error(f"执行Monitor组件操作失败: {operation}, 错误: {e}")
            raise

    def get_service_metrics(self) -> Dict[str, Any]:
        """获取Monitor组件指标"""
        try:
            factory_info = self._monitor_factory.get_factory_info()
            return {
                "supported_monitor_types": factory_info.get("total_monitors", 0),
                "factory_status": "active",
                "monitor_ids": factory_info.get("supported_ids", []),
                "last_operation_time": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取Monitor组件指标失败: {e}")
            return {"error": str(e)}


# 注册基础适配器
InfrastructureAdapterFactory.register_adapter("cache", CacheAdapter)
InfrastructureAdapterFactory.register_adapter("database", DatabaseAdapter)
InfrastructureAdapterFactory.register_adapter("monitoring", MonitoringAdapter)
InfrastructureAdapterFactory.register_adapter("logging", LoggingAdapter)

# 注册健康基础设施适配器
InfrastructureAdapterFactory.register_adapter("health_checker", HealthCheckerAdapter)
InfrastructureAdapterFactory.register_adapter("database_monitor", DatabaseMonitorAdapter)

# 注册组件适配器
InfrastructureAdapterFactory.register_adapter("alert_factory", AlertComponentAdapter)
InfrastructureAdapterFactory.register_adapter("checker_factory", CheckerComponentAdapter)
InfrastructureAdapterFactory.register_adapter("health_factory", HealthComponentAdapter)
InfrastructureAdapterFactory.register_adapter("monitor_factory", MonitorComponentAdapter)

logger.info("基础设施适配器注册完成")
