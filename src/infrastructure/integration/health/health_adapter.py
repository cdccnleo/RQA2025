"""
RQA2025 健康层基础设施适配器 - 入口点

这是health_adapter的简化入口点，重新导出所有组件。

拆分后的模块结构：
- health_models.py: 数据模型和枚举
- health_protocols.py: Protocol接口定义
- health_infrastructure.py: 基础设施管理器
- health_adapter.py: 简化入口（原822行拆分后）

基于适配器模式设计，为健康管理系统提供统一的基础设施服务访问接口。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from datetime import datetime

from ...business_adapters import IBusinessAdapter, BusinessLayerType
from src.infrastructure.logging.core.unified_logger import get_unified_logger

# 导入拆分后的模块
from .health_models import (
    HealthCheckType,
    HealthStatus,
    HealthCheckResult,
    AsyncHealthCheckConfig,
)
from .health_protocols import (
    HealthInfrastructureManager,
    HealthCheckExecutor,
    HealthConfigManager,
)
from .health_infrastructure import (
    HealthInfrastructureConfig,
    HealthInfrastructureManagerImpl,
)

logger = get_unified_logger(__name__)


# 简化的健康检查执行器实现
class HealthCheckExecutorImpl:
    """健康检查执行器实现 - 职责：执行健康检查"""

    def __init__(self, config: AsyncHealthCheckConfig, infrastructure: HealthInfrastructureManager):
        self.config = config
        self.infrastructure = infrastructure
        self._health_checks: Dict[str, Callable] = {}
        self._check_configs: Dict[str, Dict[str, Any]] = {}
        self._logger = get_unified_logger(f"{__name__}.HealthCheckExecutor")

    async def perform_health_check(self, service_name: str,
                                 check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
        """执行健康检查"""
        start_time = time.time()

        try:
            if service_name not in self._health_checks:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    message=f"健康检查方法未注册: {service_name}"
                )

            # 执行健康检查
            check_func = self._health_checks[service_name]
            result = await check_func(check_type)

            return HealthCheckResult(
                status=HealthStatus.HEALTHY if result.get('status') == 'healthy' else HealthStatus.UNHEALTHY,
                message=result.get('message', 'OK'),
                details=result.get('details', {})
            )

        except Exception as e:
            self._logger.error(f"健康检查失败: {service_name}, {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"检查失败: {str(e)}"
            )

    async def perform_batch_health_check(self, service_names: List[str],
                                       check_type: HealthCheckType = HealthCheckType.BASIC) -> Dict[str, HealthCheckResult]:
        """批量健康检查"""
        tasks = [self.perform_health_check(name, check_type) for name in service_names]
        results = await asyncio.gather(*tasks)
        return dict(zip(service_names, results))

    def register_health_check(self, service_name: str, check_func: Callable):
        """注册健康检查方法"""
        self._health_checks[service_name] = check_func
        self._logger.info(f"健康检查方法已注册: {service_name}")


class HealthConfigManagerImpl:
    """健康配置管理器实现"""

    def __init__(self):
        self._config = AsyncHealthCheckConfig()
        self._logger = get_unified_logger(f"{__name__}.HealthConfigManager")

    def get_config(self) -> AsyncHealthCheckConfig:
        """获取配置"""
        return self._config

    def update_config(self, config: Dict[str, Any]):
        """更新配置"""
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        self._logger.info("配置已更新")


# 健康层适配器接口
class IHealthAdapter(ABC):
    """健康层适配器接口"""

    @abstractmethod
    async def perform_async_health_check(self, service_name: str, check_type: HealthCheckType) -> HealthCheckResult:
        """执行异步健康检查"""
        pass

    @abstractmethod
    async def perform_batch_health_checks(self, service_names: List[str]) -> Dict[str, HealthCheckResult]:
        """批量健康检查"""
        pass


# 重构后的健康层适配器
class HealthLayerAdapterRefactored(IBusinessAdapter, IHealthAdapter):
    """健康层适配器 - 重构版：使用组合模式和专门的组件"""

    def __init__(self, config: Optional[AsyncHealthCheckConfig] = None):
        self.config = config or AsyncHealthCheckConfig()
        self.layer_type = BusinessLayerType.HEALTH
        self._logger = get_unified_logger(f"{__name__}.HealthLayerAdapter")

        # 初始化基础设施管理器
        infra_config = HealthInfrastructureConfig()
        self.infrastructure_manager = HealthInfrastructureManagerImpl(infra_config)
        self.infrastructure_manager.initialize_infrastructure()

        # 初始化健康检查执行器
        self.health_executor = HealthCheckExecutorImpl(self.config, self.infrastructure_manager)

        # 初始化配置管理器
        self.config_manager = HealthConfigManagerImpl()

        self._logger.info("健康层适配器初始化完成")

    async def perform_async_health_check(self, service_name: str,
                                       check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
        """执行异步健康检查"""
        return await self.health_executor.perform_health_check(service_name, check_type)

    async def perform_batch_health_checks(self, service_names: List[str]) -> Dict[str, HealthCheckResult]:
        """批量健康检查"""
        return await self.health_executor.perform_batch_health_check(service_names)

    def get_layer_type(self) -> BusinessLayerType:
        """获取层类型"""
        return self.layer_type

    def get_layer_info(self) -> Dict[str, Any]:
        """获取层信息"""
        return {
            'layer_type': self.layer_type.value,
            'config': {
                'enabled': self.config.enabled,
                'check_interval': self.config.check_interval,
                'timeout': self.config.timeout
            },
            'infrastructure': self.infrastructure_manager.get_infrastructure_services(),
            'status': 'active'
        }


# 为向后兼容，提供原有的类名
HealthLayerAdapter = HealthLayerAdapterRefactored


# 重新导出所有组件
__all__ = [
    # 模型和枚举
    'HealthCheckType',
    'HealthStatus',
    'HealthCheckResult',
    'AsyncHealthCheckConfig',
    
    # 协议
    'HealthInfrastructureManager',
    'HealthCheckExecutor',
    'HealthConfigManager',
    
    # 实现
    'HealthInfrastructureConfig',
    'HealthInfrastructureManagerImpl',
    'HealthCheckExecutorImpl',
    'HealthConfigManagerImpl',
    
    # 适配器
    'IHealthAdapter',
    'HealthLayerAdapter',
    'HealthLayerAdapterRefactored',
]
