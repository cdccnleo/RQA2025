"""
health_checker_factory 模块

提供 health_checker_factory 相关功能和接口。
"""

import logging

import time

from ..api.fastapi_integration import FastAPIHealthChecker
from ..monitoring.basic_health_checker import BasicHealthChecker
from .enhanced_health_checker import EnhancedHealthChecker
from typing import Dict, Any, Optional
from ..core.interfaces import IHealthChecker
"""
基础设施层 - 日志系统组件

health_checker_factory 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查器工厂

遵循基础设施层架构设计，提供统一的健康检查器创建接口。
"""

# 常量定义 - 清理魔法数字
DEFAULT_CHECK_INTERVAL = 30
DEFAULT_TIMEOUT = 10
DEFAULT_RETRY_COUNT = 3
DEFAULT_ALERT_THRESHOLD = 0.8
SUPPORTED_COUNT_ZERO = 0

logger = logging.getLogger(__name__)


class HealthCheckerFactory:
    """
    健康检查器工厂

    遵循基础设施层架构设计，提供统一的健康检查器创建接口。
    支持标准化的健康检查方法命名规范。
    """

    # =========================================================================
    # 标准化健康检查方法 (check_*)
    # =========================================================================

    @classmethod
    def _validate_config_parameter(cls, config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """验证配置参数"""
        if config is not None and not isinstance(config, dict):
            logger.warning(f"配置参数类型无效: {type(config)}，使用默认配置")
            return None
        return config

    @classmethod
    def _check_supported_types_count(cls) -> int:
        """检查支持的类型数量"""
        supported_count = len(cls.SUPPORTED_TYPES)
        logger.debug(f"健康检查器工厂支持的类型数量: {supported_count}")
        return supported_count

    @classmethod
    def _create_factory_error_response(cls, message: str, start_time: float,
                                       error: Optional[str] = None) -> Dict[str, Any]:
        """创建工厂错误响应"""
        response = {
            "service": "health_checker_factory",
            "status": "critical",
            "message": message,
            "response_time": time.time() - start_time,
            "timestamp": time.time()
        }
        if error:
            response["error"] = error
        return response

    @classmethod
    def _create_factory_success_response(cls, supported_count: int, start_time: float) -> Dict[str, Any]:
        """创建工厂成功响应"""
        logger.info(f"健康检查器工厂运行正常，支持{supported_count}种检查器类型")
        return {
            "service": "health_checker_factory",
            "status": "healthy",
            "message": f"工厂运行正常，支持{supported_count}种检查器类型",
            "response_time": time.time() - start_time,
            "timestamp": time.time(),
            "details": {
                "supported_types": list(cls.SUPPORTED_TYPES.keys()),
                "total_types": supported_count
            }
        }

    @classmethod
    def _create_factory_warning_response(cls, message: str, start_time: float) -> Dict[str, Any]:
        """创建工厂警告响应"""
        logger.warning(message)
        return {
            "service": "health_checker_factory",
            "status": "warning",
            "message": message,
            "response_time": time.time() - start_time,
            "timestamp": time.time()
        }

    @classmethod
    async def _test_factory_creation(cls, test_config: Dict[str, Any]) -> Optional[Any]:
        """测试工厂创建能力"""
        logger.debug("尝试创建基础健康检查器进行测试")
        checker = cls.create_health_checker('basic', test_config)
        logger.debug(f"健康检查器创建结果: {type(checker)}")
        return checker

    @classmethod
    async def check_health_factory_async(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查健康检查器工厂健康状态"""
        start_time = time.time()

        try:
            logger.info("开始异步检查健康检查器工厂健康状态")

            # 参数验证
            config = cls._validate_config_parameter(config)

            # 检查支持的类型
            supported_count = cls._check_supported_types_count()

            if supported_count == SUPPORTED_COUNT_ZERO:
                logger.error("健康检查器工厂没有支持的类型")
                return cls._create_factory_error_response("没有支持的健康检查器类型", start_time)

            # 检查配置有效性
            test_config = config or {}
            logger.debug(f"使用测试配置: {list(test_config.keys()) if test_config else '默认配置'}")

            try:
                # 测试工厂创建能力
                checker = await cls._test_factory_creation(test_config)

                if checker:
                    return cls._create_factory_success_response(supported_count, start_time)
                else:
                    return cls._create_factory_warning_response("工厂可以创建检查器但返回为空", start_time)

            except Exception as create_error:
                logger.error(f"工厂无法创建健康检查器实例: {str(create_error)}", exc_info=True)
                return cls._create_factory_error_response("工厂无法创建健康检查器实例", start_time, str(create_error))

        except Exception as e:
            logger.error(f"健康检查器工厂健康检查失败: {str(e)}", exc_info=True)
            return cls._create_factory_error_response("健康检查器工厂健康检查失败", start_time, str(e))

    @classmethod
    async def check_basic_checker_async(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查基础健康检查器"""
        return await cls._check_specific_checker_async('basic', config)

    @classmethod
    async def check_enhanced_checker_async(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查增强健康检查器"""
        return await cls._check_specific_checker_async('enhanced', config)

    @classmethod
    async def check_fastapi_checker_async(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查FastAPI健康检查器"""
        return await cls._check_specific_checker_async('fastapi', config)

    @classmethod
    def _validate_checker_type(cls, checker_type: str, start_time: float) -> Optional[Dict[str, Any]]:
        """验证检查器类型"""
        if checker_type not in cls.SUPPORTED_TYPES:
            return {
                "service": f"health_checker_{checker_type}",
                "status": "critical",
                "message": f"不支持的检查器类型: {checker_type}",
                "response_time": time.time() - start_time,
                "timestamp": time.time()
            }
        return None

    @classmethod
    def _create_checker_instance(cls, checker_type: str, config: Optional[Dict[str, Any]],
                                 start_time: float) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """创建检查器实例"""
        checker = cls.create_health_checker(checker_type, config)
        if not checker:
            return None, {
                "service": f"health_checker_{checker_type}",
                "status": "critical",
                "message": f"无法创建{checker_type}类型的健康检查器",
                "response_time": time.time() - start_time,
                "timestamp": time.time()
            }
        return checker, None

    @classmethod
    def _execute_checker_health_check(cls, checker: Any, checker_type: str,
                                      start_time: float) -> Dict[str, Any]:
        """执行检查器健康检查"""
        if hasattr(checker, 'check_health'):
            result = checker.check_health()
            return {
                "service": f"health_checker_{checker_type}",
                "status": "healthy" if result.get('status') == 'healthy' else "warning",
                "message": f"{checker_type}检查器运行正常",
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
                "details": result
            }
        else:
            return {
                "service": f"health_checker_{checker_type}",
                "status": "warning",
                "message": f"{checker_type}检查器缺少check_health方法",
                "response_time": time.time() - start_time,
                "timestamp": time.time()
            }

    @classmethod
    def _create_checker_error_response(cls, checker_type: str, error: Exception,
                                       start_time: float) -> Dict[str, Any]:
        """创建检查器错误响应"""
        logger.error(f"{checker_type}健康检查器检查失败: {error}")
        return {
            "service": f"health_checker_{checker_type}",
            "status": "critical",
            "error": str(error),
            "response_time": time.time() - start_time,
            "timestamp": time.time()
        }

    @classmethod
    async def _check_specific_checker_async(
        cls,
        checker_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """检查特定类型的健康检查器"""
        start_time = time.time()

        try:
            # 验证检查器类型
            type_error = cls._validate_checker_type(checker_type, start_time)
            if type_error:
                return type_error

            # 创建检查器实例
            checker, create_error = cls._create_checker_instance(checker_type, config, start_time)
            if create_error:
                return create_error

            # 执行健康检查
            return cls._execute_checker_health_check(checker, checker_type, start_time)

        except Exception as e:
            return cls._create_checker_error_response(checker_type, e, start_time)

    # =========================================================================
    # 健康状态管理方法 (health_*)
    # =========================================================================

    @classmethod
    async def health_status_async(cls) -> Dict[str, Any]:
        """异步获取工厂健康状态摘要"""
        return {
            "component": "HealthCheckerFactory",
            "supported_types": list(cls.SUPPORTED_TYPES.keys()),
            "total_types": len(cls.SUPPORTED_TYPES),
            "timestamp": time.time()
        }

    @classmethod
    async def health_summary_async(cls) -> Dict[str, Any]:
        """异步获取健康状态汇总报告"""
        summary = await cls.health_status_async()

        # 执行所有检查器的健康检查
        all_checks = []
        for checker_type in cls.SUPPORTED_TYPES.keys():
            check_result = await cls._check_specific_checker_async(checker_type)
            all_checks.append(check_result)

        # 汇总结果
        healthy_count = sum(1 for r in all_checks if r.get("status") == "healthy")
        warning_count = sum(1 for r in all_checks if r.get("status") == "warning")
        critical_count = sum(1 for r in all_checks if r.get("status") == "critical")

        overall_status = "healthy"
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"

        summary.update({
            "overall_status": overall_status,
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "total_checks": len(all_checks),
            "check_results": all_checks
        })

        return summary

    # =========================================================================
    # 监控管理方法 (monitor_*)
    # =========================================================================

    @classmethod
    async def monitor_factory_async(cls) -> Dict[str, Any]:
        """异步监控工厂运行状态"""
        return {
            "component": "HealthCheckerFactory",
            "monitoring_status": "active",
            "supported_types": list(cls.SUPPORTED_TYPES.keys()),
            "factory_health": "healthy",
            "timestamp": time.time()
        }

    # =========================================================================
    # 验证方法 (validate_*)
    # =========================================================================

    @classmethod
    async def validate_factory_config_async(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证工厂配置"""
        start_time = time.time()

        try:
            # 检查必需的配置项
            required_keys = ['checker_type']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                return {
                    "service": "factory_config_validation",
                    "status": "critical",
                    "message": f"缺少必需配置项: {missing_keys}",
                    "response_time": time.time() - start_time,
                    "timestamp": time.time()
                }

            # 检查检查器类型是否支持
            checker_type = config.get('checker_type')
            if checker_type not in cls.SUPPORTED_TYPES:
                return {
                    "service": "factory_config_validation",
                    "status": "critical",
                    "message": f"不支持的检查器类型: {checker_type}",
                    "response_time": time.time() - start_time,
                    "timestamp": time.time()
                }

            return {
                "service": "factory_config_validation",
                "status": "healthy",
                "message": "工厂配置验证通过",
                "response_time": time.time() - start_time,
                "timestamp": time.time(),
                "details": {
                    "checker_type": checker_type,
                    "config_valid": True
                }
            }

        except Exception as e:
            logger.error(f"工厂配置验证失败: {e}")
            return {
                "service": "factory_config_validation",
                "status": "critical",
                "error": str(e),
                "response_time": time.time() - start_time,
                "timestamp": time.time()
            }

    # 支持的健康检查器类型
    SUPPORTED_TYPES = {
        'basic': BasicHealthChecker,
        'enhanced': EnhancedHealthChecker,
        'fastapi': FastAPIHealthChecker
    }

    @classmethod
    def create_health_checker(cls,
                              checker_type: str = 'basic',
                              config: Optional[Dict[str, Any]] = None):
        """
        创建健康检查器实例

        Args:
            checker_type: 检查器类型 ('basic', 'enhanced', 'fastapi')
            config: 配置字典

        Returns:
            健康检查器实例

        Raises:
            ValueError: 不支持的检查器类型
        """
        if checker_type not in cls.SUPPORTED_TYPES:
            supported_types = ', '.join(cls.SUPPORTED_TYPES.keys())
            raise ValueError(f"不支持的检查器类型: {checker_type}. 支持的类型: {supported_types}")

        checker_class = cls.SUPPORTED_TYPES[checker_type]

        try:
            if checker_type == 'fastapi':
                # FastAPI检查器需要先创建基础检查器
                base_checker = cls.create_health_checker('enhanced', config)
                checker = checker_class(base_checker, config)
            else:
                checker = checker_class()

            logger.info(f"已创建健康检查器: {checker_type}")
            return checker

        except Exception as e:
            logger.error(f"创建健康检查器失败: {checker_type}, 错误: {e}")
            raise

    @classmethod
    def create_basic_checker(cls, config: Optional[Dict[str, Any]] = None) -> BasicHealthChecker:
        """
        创建基础健康检查器

        Args:
            config: 配置字典

        Returns:
            基础健康检查器实例
        """
        return cls.create_health_checker('basic', config)

    @classmethod
    def create_enhanced_checker(cls, config: Optional[Dict[str, Any]] = None) -> EnhancedHealthChecker:
        """
        创建增强健康检查器

        Args:
            config: 配置字典

        Returns:
            增强健康检查器实例
        """
        return cls.create_health_checker('enhanced', config)

    @classmethod
    def create_fastapi_checker(cls,
                               base_checker: Optional[IHealthChecker] = None,
                               config: Optional[Dict[str, Any]] = None):
        """
        创建FastAPI健康检查器

        Args:
            base_checker: 基础健康检查器实例
            config: 配置字典

        Returns:
            FastAPI健康检查器实例
        """
        if base_checker is None:
            base_checker = cls.create_enhanced_checker(config)

        return FastAPIHealthChecker(base_checker, config)

    @classmethod
    def get_supported_types(cls) -> Dict[str, str]:
        """
        获取支持的健康检查器类型

        Returns:
            类型名称到描述的映射
        """
        return {
            'basic': '基础健康检查器 - 提供基本的健康检查功能',
            'enhanced': '增强健康检查器 - 提供性能监控和历史记录',
            'fastapi': 'FastAPI健康检查器 - 提供HTTP路由接口'
        }

    @classmethod
    def validate_config(cls, checker_type: str, config: Dict[str, Any]) -> bool:
        """
        验证配置是否有效

        Args:
            checker_type: 检查器类型
            config: 配置字典

        Returns:
            配置是否有效
        """
        if checker_type not in cls.SUPPORTED_TYPES:
            return False

        # 基础配置验证
        required_configs = {
            'basic': ['check_interval'],  # basic只需要check_interval
            'enhanced': ['check_interval', 'timeout', 'performance_tracking'],
            'fastapi': ['check_interval', 'timeout']
        }

        if checker_type in required_configs:
            required_keys = required_configs[checker_type]
            for key in required_keys:
                if key not in config:
                    logger.warning(f"缺少必需配置: {key}")
                    return False

        return True

    @classmethod
    def create_with_default_config(cls, checker_type: str = 'basic') -> IHealthChecker:
        """
        使用默认配置创建健康检查器

        Args:
            checker_type: 检查器类型

        Returns:
            健康检查器实例
        """

        default_configs = {
            'basic': {
                'check_interval': DEFAULT_CHECK_INTERVAL,
                'timeout': DEFAULT_TIMEOUT,
                'retry_count': DEFAULT_RETRY_COUNT
            },
            'enhanced': {
                'check_interval': DEFAULT_CHECK_INTERVAL,
                'timeout': DEFAULT_TIMEOUT,
                'retry_count': DEFAULT_RETRY_COUNT,
                'performance_tracking': True,
                'auto_monitoring': True,
                'alert_threshold': DEFAULT_ALERT_THRESHOLD
            },
            'fastapi': {
                'check_interval': DEFAULT_CHECK_INTERVAL,
                'timeout': DEFAULT_TIMEOUT,
                'retry_count': DEFAULT_RETRY_COUNT
            }
        }

        config = default_configs.get(checker_type, {})
        return cls.create_health_checker(checker_type, config)
