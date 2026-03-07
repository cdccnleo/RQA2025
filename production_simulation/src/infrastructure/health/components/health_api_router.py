"""
健康检查API路由管理器

负责管理健康检查相关的API端点和路由配置。
"""

from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class HealthApiRouter:

    """
    健康检查API路由管理器
    
    职责：
    - API路由配置和管理
    - 健康检查端点处理
    - 就绪检查端点处理
    """
    
    def __init__(self, 
                 system_health_checker=None, 
                 dependency_checker=None):
        """
        初始化API路由管理器
        
        Args:
            system_health_checker: 系统健康检查器实例
            dependency_checker: 依赖检查器实例
        """
        self.router = APIRouter()
        self._system_health_checker = system_health_checker
        self._dependency_checker = dependency_checker
        
        # 注册路由
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置API路由"""
        self.router.add_api_route("/health", self.health_endpoint, methods=["GET"])
        self.router.add_api_route("/ready", self.ready_endpoint, methods=["GET"])
        self.router.add_api_route("/live", self.live_endpoint, methods=["GET"])
        
    async def health_endpoint(self) -> Dict[str, Any]:
        """
        健康检查端点
        
        Returns:
            Dict[str, Any]: 完整的健康状态信息
        """
        try:
            start_time = datetime.now()
            
            # 收集系统健康状态
            system_health = {}
            if self._system_health_checker:
                system_health = self._system_health_checker.get_system_health()
            
            # 收集依赖服务状态
            dependencies_health = []
            if self._dependency_checker:
                deps_result = self._dependency_checker.check_dependencies_health()
                dependencies_health = deps_result.get("dependencies", [])
            
            # 确定整体状态
            overall_status = self._determine_overall_status(system_health, dependencies_health)
            
            response = {
                "timestamp": datetime.now().isoformat(),
                "status": overall_status,
                "system": system_health,
                "dependencies": dependencies_health,
                "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            logger.debug(f"健康检查端点响应: {overall_status}")
            return response
            
        except Exception as e:
            logger.error(f"健康检查端点异常: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def ready_endpoint(self) -> Dict[str, Any]:
        """
        就绪检查端点
        
        Returns:
            Dict[str, Any]: 就绪状态信息
        """
        try:
            # 检查基本组件是否已初始化
            is_ready = self._check_readiness()
            
            response = {
                "timestamp": datetime.now().isoformat(),
                "status": "ready" if is_ready else "not_ready",
                "components": {
                    "system_checker": self._system_health_checker is not None,
                    "dependency_checker": self._dependency_checker is not None
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"就绪检查端点异常: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    async def live_endpoint(self) -> Dict[str, Any]:
        """
        存活检查端点（简单的心跳检查）
        
        Returns:
            Dict[str, Any]: 存活状态信息
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "alive",
            "message": "服务正在运行"
        }
    
    def _determine_overall_status(self, 
                                system_health: Dict[str, Any], 
                                dependencies_health: list) -> str:
        """
        确定整体健康状态
        
        Args:
            system_health: 系统健康状态
            dependencies_health: 依赖服务健康状态列表
            
        Returns:
            str: 整体状态
        """
        # 检查系统状态
        system_status = system_health.get("status", "healthy")
        if system_status in ["critical", "error"]:
            return "unhealthy"
        elif system_status == "warning":
            return "degraded"
        
        # 检查依赖服务状态
        if dependencies_health:
            unhealthy_deps = [dep for dep in dependencies_health 
                            if dep.get("status") not in ["healthy", "ok"]]
            
            if len(unhealthy_deps) == len(dependencies_health):
                return "unhealthy"
            elif unhealthy_deps:
                return "degraded"
        
        return "healthy"
    
    def _check_readiness(self) -> bool:
        """
        检查服务是否就绪
        
        Returns:
            bool: 是否就绪
        """
        try:
            # 检查必要的组件是否可用
            if not self._system_health_checker:
                logger.warning("系统健康检查器未初始化")
                return False
            
            # 可以添加更多就绪检查条件
            return True
            
        except Exception as e:
            logger.error(f"就绪检查失败: {e}")
            return False
    
    def check_router_health(self) -> Dict[str, Any]:
        """
        检查路由器健康状态
        
        Returns:
            Dict[str, Any]: 路由器健康状态
        """
        try:
            logger.debug("执行路由器健康检查")
            
            if not hasattr(self.router, 'routes') or not self.router.routes:
                return {
                    "status": "error",
                    "message": "路由器未正确配置",
                    "timestamp": datetime.now().isoformat()
                }
            
            route_count = len(self.router.routes)
            route_paths = [route.path for route in self.router.routes]
            
            return {
                "status": "success",
                "message": f"路由器健康，包含 {route_count} 个路由",
                "route_count": route_count,
                "route_paths": route_paths,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"路由器健康检查失败: {e}")
            return {
                "status": "error",
                "message": f"路由器健康检查失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_router_info(self) -> Dict[str, Any]:
        """获取路由器信息"""
        try:
            route_count = len(self.router.routes) if hasattr(self.router, 'routes') else 0
            
            return {
                "router_type": "FastAPI_APIRouter",
                "route_count": route_count,
                "endpoints": {
                    "/health": "健康检查端点",
                    "/ready": "就绪检查端点", 
                    "/live": "存活检查端点"
                },
                "component_status": {
                    "system_checker": self._system_health_checker is not None,
                    "dependency_checker": self._dependency_checker is not None
                }
            }
        except Exception as e:
            logger.error(f"获取路由器信息失败: {e}")
            return {"error": str(e)}

