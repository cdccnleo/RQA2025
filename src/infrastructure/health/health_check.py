from fastapi import APIRouter, status
from typing import Dict, List
import psutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthCheck:
    """系统健康检查服务"""

    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/ready", self.ready, methods=["GET"])

        # 依赖服务检查列表
        self.dependencies: List[Dict] = []

    def add_dependency_check(self, name: str, check_func: callable):
        """添加依赖服务检查"""
        self.dependencies.append({
            "name": name,
            "check": check_func
        })

    async def health(self) -> Dict:
        """健康检查端点"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "system": self._get_system_health(),
            "dependencies": []
        }

        # 检查各依赖服务
        for dep in self.dependencies:
            try:
                result = dep["check"]()
                status["dependencies"].append({
                    "name": dep["name"],
                    "status": "healthy" if result else "unhealthy",
                    "details": str(result)
                })
            except Exception as e:
                status["dependencies"].append({
                    "name": dep["name"],
                    "status": "error",
                    "error": str(e)
                })

        # 如果有不健康的依赖，整体状态设为不健康
        if any(d["status"] != "healthy" for d in status["dependencies"]):
            status["status"] = "degraded"

        return status

    async def ready(self) -> Dict:
        """就绪检查端点"""
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "ready"
        }

    def _get_system_health(self) -> Dict:
        """获取系统健康状态"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu": f"{cpu}%",
                "memory": f"{mem.percent}%",
                "disk": f"{disk.percent}%",
                "process": {
                    "uptime": str(datetime.now() - datetime.fromtimestamp(psutil.Process().create_time()))
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "error": str(e)
            }
