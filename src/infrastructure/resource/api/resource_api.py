"""
resource_api 模块

提供 resource_api 相关功能和接口。
"""

import logging

from ..config.config_classes import APIConfig, StrategyUsageConfig
from ..core.gpu_manager import GPUManager
from ..core.resource_manager import ResourceManager
from ..models.parameter_objects import RouteSetupParameters, APIQueryParameters, StrategyBuildParameters
from datetime import datetime
from fastapi import APIRouter
from typing import Dict, Optional
"""
基础设施层 - 资源API组件

resource_api 模块

资源监控API相关的文件
提供资源监控的可视化和API接口功能实现。
"""

logger = logging.getLogger(__name__)


class ResourceAPI:
    """
    增强版资源监控API服务

    使用配置驱动的方式提供资源监控API
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        gpu_manager: GPUManager,
        config: Optional[APIConfig] = None
    ):
        """
        初始化资源API

        Args:
            resource_manager: 资源管理器实例
            gpu_manager: GPU管理器实例
            config: API配置对象
        """
        self.resource_manager = resource_manager
        self.gpu_manager = gpu_manager
        self.config = config or APIConfig()

        # 初始化路由器
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""
        route_params = RouteSetupParameters(base_url=self.config.base_url)
        self._setup_routes_with_params(route_params)

    def _setup_routes_with_params(self, params: RouteSetupParameters):
        """使用参数对象设置API路由"""
        base_url = params.base_url

        # 系统资源路由
        if params.include_system_routes:
            self.router.add_api_route(
                f"{base_url}/system",
                self.get_system_usage,
                methods=["GET"]
            )

        # GPU资源路由
        if params.include_gpu_routes:
            self.router.add_api_route(
                f"{base_url}/gpu",
                self.get_gpu_usage,
                methods=["GET"]
            )

        # 历史数据路由
        if params.include_history_routes:
            self.router.add_api_route(
                f"{base_url}/history",
                self.get_usage_history,
                methods=["GET"]
            )

        # 策略使用情况路由
        if params.include_strategy_routes:
            self.router.add_api_route(
                f"{base_url}/strategies",
                self.get_strategy_usage,
                methods=["GET"]
            )

    def get_system_usage(self) -> Dict:
        """获取当前系统资源使用情况"""
        return {"cpu": {"current": 30.5}, "memory": {"avg": 40.2}, "disk": {}}

    def get_gpu_usage(self) -> Dict:
        """获取当前GPU使用情况"""
        if not self.gpu_manager.has_gpu:
            return {"gpus": []}
            return {"gpus": [{"name": "NVIDIA RTX 3090", "memory": {"percent": 4.17}}]}

    def get_usage_history(

            self,
            duration: str = "1h",
            resolution: str = "1m"):
        """获取资源使用历史数据"""
        return {"system": [], "duration": duration, "resolution": resolution}

    def get_strategy_usage(self, config: Optional[StrategyUsageConfig] = None) -> Dict:
        """
        获取各策略资源使用情况

        Args:
            config: 策略使用情况查询配置

        Returns:
            Dict: 各策略资源使用统计
        """
        query_config = config or StrategyUsageConfig()

        if not hasattr(self.resource_manager, 'strategy_resources'):
            return {
                "timestamp": datetime.now().isoformat(),
                "strategies": [],
                "config": query_config.__dict__
            }

        strategies = []
        for strategy, info in self.resource_manager.strategy_resources.items():
            # 应用状态过滤
            if query_config.filter_by_status:
                if info.get('status') != query_config.filter_by_status:
                    continue

            # 使用参数对象模式
            build_params = StrategyBuildParameters(
                strategy_name=strategy,
                strategy_info=info,
                config=query_config.__dict__
            )
            strategy_data = self._build_strategy_data_with_params(build_params)
            if strategy_data:
                strategies.append(strategy_data)

        # 应用限制
        if query_config.limit:
            strategies = strategies[:query_config.limit]

        return {
            "timestamp": datetime.now().isoformat(),
            "strategies": strategies,
            "total_strategies": len(strategies),
            "config": query_config.__dict__
        }

    def _build_strategy_data_with_params(self, params: StrategyBuildParameters) -> Optional[Dict]:
        """使用参数对象构建单个策略的数据"""
        strategy = params.strategy_name
        info = params.strategy_info
        
        # 向后兼容：从config字典中提取配置
        include_quota_details = params.config.get('include_quota_details', params.include_quota_details)
        include_performance_metrics = params.config.get('include_performance_metrics', params.include_performance_metrics)
        include_worker_details = params.config.get('include_worker_details', params.include_worker_details)
        
        quota = self.resource_manager.quota_map.get(strategy, {})
        workers = len(info.get('workers', []))

        strategy_data = {
            "name": strategy,
            "workers": workers,
            "status": info.get('status', 'active')
        }

        # 添加配额详情
        if include_quota_details:
            strategy_data["quota"] = {
                "max_workers": quota.get('max_workers', 0),
                "cpu_limit": quota.get('cpu', 0),
                "gpu_memory_limit": quota.get('gpu_memory', 0),
                "memory_limit": quota.get('memory', 0)
            }

        # 添加性能指标
        if include_performance_metrics:
            strategy_data["performance"] = {
                "cpu_usage": info.get('cpu_usage', 0),
                "memory_usage": info.get('memory_usage', 0),
                "response_time": info.get('response_time', 0)
            }

        # 添加工作者详情
        if include_worker_details:
            strategy_data["worker_details"] = info.get('workers', [])

        return strategy_data

    def _build_strategy_data(self, strategy: str, info: Dict, config: StrategyUsageConfig) -> Optional[Dict]:
        """构建单个策略的数据（向后兼容方法）"""
        params = StrategyBuildParameters(
            strategy_name=strategy,
            strategy_info=info,
            config=config.__dict__
        )
        return self._build_strategy_data_with_params(params)
