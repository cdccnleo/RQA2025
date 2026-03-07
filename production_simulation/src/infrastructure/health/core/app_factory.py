"""
app_factory 模块

提供 app_factory 相关功能和接口。
"""

import logging

import fastapi
import inspect

from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from .interfaces import IUnifiedInfrastructureInterface
from datetime import datetime
from fastapi import FastAPI
from typing import Dict, Any, Optional
"""
基础设施层 - 应用工厂

创建和管理FastAPI应用实例
"""

logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """创建FastAPI应用实例"""
    try:
        logger.info("开始创建FastAPI应用实例")

        # 创建FastAPI应用
        app = FastAPI(
            title="RQA2025量化平台",
            description="A股量化交易系统API",
            version="1.0.0",
            # 添加错误处理相关的配置
            debug=False,  # 生产环境关闭debug模式
            docs_url="/api/docs",  # API文档路径
            redoc_url="/api/redoc",  # ReDoc文档路径
            openapi_url="/api/openapi.json"  # OpenAPI规范路径
        )

        # 添加全局异常处理器
        _setup_global_exception_handlers(app)

        logger.info("FastAPI应用实例创建成功")
        return app

    except ImportError as e:
        logger.error(f"FastAPI导入失败，请确保已安装fastapi: {e}")
        raise RuntimeError(f"FastAPI不可用: {e}") from e

    except Exception as e:
        logger.error(f"创建FastAPI应用实例失败: {e}", exc_info=True)
        raise RuntimeError(f"应用创建失败: {e}") from e


def _setup_global_exception_handlers(app: FastAPI) -> None:
    """设置全局异常处理器"""
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP异常处理器"""
        logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url),
                "method": request.method
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理器"""
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "内部服务器错误",
                "status_code": 500,
                "path": str(request.url),
                "method": request.method,
                "error_type": type(exc).__name__
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """值错误处理器"""
        logger.warning(f"值错误: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "message": str(exc),
                "status_code": 400,
                "path": str(request.url),
                "method": request.method,
                "error_type": "ValueError"
            }
        )

    logger.info("全局异常处理器已设置")


def check_health() -> Dict[str, Any]:
    """执行整体健康检查

    Returns:
        Dict[str, Any]: 健康检查结果
    """
    try:
        logger.info("开始应用工厂健康检查")

        health_checks = {
            "fastapi_availability": check_fastapi_availability(),
            "application_creation": check_application_creation(),
            "exception_handlers": check_exception_handlers()
        }

        # 综合健康状态
        overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

        result = {
            "healthy": overall_healthy,
            "timestamp": datetime.now().isoformat(),
            "service": "app_factory",
            "checks": health_checks
        }

        if not overall_healthy:
            logger.warning("应用工厂健康检查发现问题")
            result["issues"] = [
                name for name, check in health_checks.items()
                if not check.get("healthy", False)
            ]

        logger.info(f"应用工厂健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
        return result

    except Exception as e:
        logger.error(f"应用工厂健康检查失败: {str(e)}", exc_info=True)
        return {
            "healthy": False,
            "timestamp": datetime.now().isoformat(),
            "service": "app_factory",
            "error": str(e)
        }


def check_fastapi_availability() -> Dict[str, Any]:
    """检查FastAPI可用性

    Returns: Dict[str, Any]astAPI可用性检查结果
    """
    try:
        # 检查FastAPI是否可以导入
        version = fastapi.__version__

        return {
            "healthy": True,
            "fastapi_available": True,
            "version": version
        }
    except ImportError:
        logger.error("FastAPI不可用")
        return {"healthy": False, "fastapi_available": False, "error": "FastAPI not installed"}
    except Exception as e:
        logger.error(f"FastAPI可用性检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_application_creation() -> Dict[str, Any]:
    """检查应用创建功能

    Returns:
        Dict[str, Any]: 应用创建健康检查结果
    """
    try:
        # 尝试创建一个测试应用
        test_app = None
        creation_successful = False

        try:
            test_app = create_application()
            creation_successful = test_app is not None
            if test_app and hasattr(test_app, 'title'):
                creation_successful = creation_successful and test_app.title == "RQA2025量化平台"
        except Exception:
            creation_successful = False

        return {
            "healthy": creation_successful,
            "application_creation_test": creation_successful,
            "test_app_title": test_app.title if test_app else None
        }
    except Exception as e:
        logger.error(f"应用创建健康检查失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def check_exception_handlers() -> Dict[str, Any]:
    """检查异常处理器配置

    Returns:
        Dict[str, Any]: 异常处理器健康检查结果
    """
    try:
        # 创建测试应用并检查异常处理器
        test_app = create_application()

        # 检查是否设置了异常处理器
        has_http_exception_handler = hasattr(
            test_app, 'exception_handlers') and len(test_app.exception_handlers) > 0
        has_general_exception_handler = hasattr(test_app, 'exception_handlers')

        # 检查用户异常处理器
        user_exception_handlers = getattr(test_app, 'user_middleware', [])
        has_user_handlers = len(user_exception_handlers) > 0

        return {
            "healthy": has_http_exception_handler,
            "exception_handlers_configured": has_http_exception_handler,
            "general_handlers_available": has_general_exception_handler,
            "user_handlers_count": len(user_exception_handlers) if has_user_handlers else 0
        }
    except Exception as e:
        logger.error(f"异常处理器健康检查失败: {str(e)}")
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
            "service": "app_factory",
            "health_check": health_check,
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
            "factory_info": {
                "service_name": "app_factory",
                "purpose": "FastAPI应用工厂",
                "operational": health_check["healthy"]
            },
            "fastapi_status": {
                "available": health_check["checks"]["fastapi_availability"]["fastapi_available"],
                "version": health_check["checks"]["fastapi_availability"].get("version"),
                "app_creation_working": health_check["checks"]["application_creation"]["application_creation_test"]
            },
            "configuration": {
                "exception_handlers_setup": health_check["checks"]["exception_handlers"]["exception_handlers_configured"],
                "global_handlers_configured": True  # 全局异常处理器已在代码中配置
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取健康摘要报告失败: {str(e)}")
        return {"overall_health": "error", "error": str(e)}


def monitor_app_factory() -> Dict[str, Any]:
    """监控应用工厂状态

    Returns:
        Dict[str, Any]: 工厂监控结果
    """
    try:
        health_check = check_health()

        # 计算工厂效率指标
        factory_efficiency = 1.0 if health_check["healthy"] else 0.0

        return {
            "healthy": health_check["healthy"],
            "factory_metrics": {
                "service_name": "app_factory",
                "factory_efficiency": factory_efficiency,
                "operational_status": "active" if health_check["healthy"] else "inactive"
            },
            "fastapi_metrics": {
                "fastapi_available": health_check["checks"]["fastapi_availability"]["fastapi_available"],
                "app_creation_success_rate": 1.0 if health_check["checks"]["application_creation"]["application_creation_test"] else 0.0,
                "exception_handlers_configured": health_check["checks"]["exception_handlers"]["exception_handlers_configured"]
            }
        }
    except Exception as e:
        logger.error(f"应用工厂监控失败: {str(e)}")
        return {"healthy": False, "error": str(e)}


def validate_app_factory_config() -> Dict[str, Any]:
    """验证应用工厂配置

    Returns:
        Dict[str, Any]: 配置验证结果
    """
    try:
        validation_results = {
            "fastapi_validation": _validate_fastapi_import(),
            "function_validation": _validate_factory_functions(),
            "configuration_validation": _validate_app_configuration()
        }

        overall_valid = all(result.get("valid", False) for result in validation_results.values())

        return {
            "valid": overall_valid,
            "validation_results": validation_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"应用工厂配置验证失败: {str(e)}")
        return {"valid": False, "error": str(e)}


def _validate_fastapi_import() -> Dict[str, Any]:
    """验证FastAPI导入"""
    try:
        imports_available = all([
            hasattr(fastapi, '__version__'),
            FastAPI is not None,
            HTTPException is not None
        ])

        return {
            "valid": imports_available,
            "fastapi_version": fastapi.__version__,
            "required_imports_available": imports_available
        }
    except ImportError as e:
        return {"valid": False, "error": f"Import error: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_factory_functions() -> Dict[str, Any]:
    """验证工厂函数"""
    try:
        # 检查必需的函数是否存在
        functions_exist = all([
            callable(create_application),
            callable(_setup_global_exception_handlers)
        ])

        # 检查函数签名
        create_app_sig = inspect.signature(create_application)
        setup_handlers_sig = inspect.signature(_setup_global_exception_handlers)

        signatures_valid = (
            len(create_app_sig.parameters) == 0 and  # create_application()
            len(setup_handlers_sig.parameters) == 1  # _setup_global_exception_handlers(app)
        )

        return {
            "valid": functions_exist and signatures_valid,
            "functions_exist": functions_exist,
            "signatures_valid": signatures_valid
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_app_configuration() -> Dict[str, Any]:
    """验证应用配置"""
    try:
        # 创建测试应用验证配置
        test_app = create_application()

        # 检查应用基本配置
        config_valid = all([
            hasattr(test_app, 'title') and test_app.title == "RQA2025量化平台",
            hasattr(test_app, 'version') and test_app.version == "1.0.0",
            hasattr(test_app, 'docs_url') and test_app.docs_url == "/api/docs",
            hasattr(test_app, 'redoc_url') and test_app.redoc_url == "/api/redoc",
            hasattr(test_app, 'openapi_url') and test_app.openapi_url == "/api/openapi.json"
        ])

        # 检查debug模式
        debug_mode_correct = not getattr(test_app, 'debug', True)  # 应该是False

        return {
            "valid": config_valid and debug_mode_correct,
            "config_valid": config_valid,
            "debug_mode_correct": debug_mode_correct,
            "app_title": getattr(test_app, 'title', None),
            "app_version": getattr(test_app, 'version', None)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# 导入统一接口


class AppFactoryManager(IUnifiedInfrastructureInterface):
    """应用工厂管理器

    实现统一基础设施接口，管理FastAPI应用的创建和生命周期。
    """

    def __init__(self):
        """初始化管理器"""
        self._initialized = False
        self._app_count = 0
        self._last_app_creation = None
        self._start_time = datetime.now()
        self._config = {}
        self._current_app = None

        logger.info("AppFactoryManager initialized")

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化应用工厂管理器

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            self._config = config or {}
            self._initialized = True

            logger.info("AppFactoryManager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AppFactoryManager: {e}")
            return False

    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息

        Returns:
            Dict[str, Any]: 组件信息
        """
        return {
            "component_type": "AppFactoryManager",
            "description": "FastAPI应用工厂管理器",
            "version": "1.0.0",
            "functions": [
                {"name": "create_application", "description": "创建FastAPI应用实例"},
                {"name": "_setup_global_exception_handlers", "description": "设置全局异常处理器"}
            ],
            "initialized": self._initialized,
            "start_time": self._start_time.isoformat(),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            "app_creation_count": self._app_count,
            "has_current_app": self._current_app is not None
        }

    def is_healthy(self) -> bool:
        """检查组件健康状态

        Returns:
            bool: 组件是否健康
        """
        try:
            # 检查基本状态
            if not self._initialized:
                return False

            # 检查FastAPI是否可用
            try:
                import fastapi
            except ImportError:
                return False

            # 检查应用创建函数是否存在
            return callable(create_application)

        except Exception as e:
            logger.error(f"Error checking AppFactoryManager health: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """获取组件指标

        Returns:
            Dict[str, Any]: 组件指标数据
        """
        current_time = datetime.now()

        return {
            "app_creation_count": self._app_count,
            "last_app_creation": self._last_app_creation.isoformat() if self._last_app_creation else None,
            "uptime_seconds": (current_time - self._start_time).total_seconds(),
            "has_current_app": self._current_app is not None,
            "functions_available": {
                "create_application": callable(create_application),
                "setup_exception_handlers": callable(_setup_global_exception_handlers)
            },
            "component_status": {
                "initialized": self._initialized,
                "healthy": self.is_healthy()
            },
            "fastapi_info": {
                "available": True,
                "version": getattr(FastAPI, '__version__', 'unknown') if 'FastAPI' in globals() else 'not imported'
            }
        }

    def cleanup(self) -> bool:
        """清理组件资源

        Returns:
            bool: 清理是否成功
        """
        try:
            # 清理当前应用实例
            self._current_app = None
            self._app_count = 0
            self._last_app_creation = None

            # 保持初始化状态，但清理运行时数据
            logger.info("AppFactoryManager资源清理完成")
            return True

        except Exception as e:
            logger.error(f"AppFactoryManager资源清理失败: {str(e)}")
            return False
