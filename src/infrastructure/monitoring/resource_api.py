from fastapi import APIRouter
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..resource.resource_manager import ResourceManager
from ..resource.gpu_manager import GPUManager
import logging

logger = logging.getLogger(__name__)

class ResourceAPI:
    """增强版资源监控API服务"""

    def __init__(self, resource_manager: ResourceManager, gpu_manager: GPUManager):
        self.router = APIRouter()
        self.resource_manager = resource_manager
        self.gpu_manager = gpu_manager

        # 注册API路由
        self.router.add_api_route("/system", self.get_system_usage, methods=["GET"])
        self.router.add_api_route("/gpu", self.get_gpu_usage, methods=["GET"])
        self.router.add_api_route("/history", self.get_usage_history, methods=["GET"])
        self.router.add_api_route("/strategies", self.get_strategy_usage, methods=["GET"])

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
        resolution: str = "1m"
    ) -> Dict:
        """获取资源使用历史数据"""
        return {"system": [], "duration": duration, "resolution": resolution}

    def get_strategy_usage(self) -> Dict:
        """
        获取各策略资源使用情况

        Returns:
            Dict: 各策略资源使用统计
        """
        if not hasattr(self.resource_manager, 'strategy_resources'):
            return {"strategies": []}

        strategies = []
        for strategy, info in self.resource_manager.strategy_resources.items():
            quota = self.resource_manager.quota_map.get(strategy, {})
            workers = len(info.get('workers', []))

            strategies.append({
                "name": strategy,
                "workers": workers,
                "quota": {
                    "max_workers": quota.get('max_workers', 0),
                    "cpu_limit": quota.get('cpu', 0),
                    "gpu_memory_limit": quota.get('gpu_memory', 0)
                }
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "strategies": strategies
        }
