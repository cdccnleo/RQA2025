"""
依赖服务检查器

负责管理依赖服务的健康检查和状态监控。
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .parameter_objects import DependencyConfig

logger = get_unified_logger(__name__)


class DependencyService:
    """依赖服务信息"""
    
    def __init__(self, name: str = "", check_func: Optional[Callable] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化依赖服务
        
        Args:
            name: 服务名称
            check_func: 检查函数
            config: 配置参数
        """
        self.name = name
        self.check_func = check_func or (lambda: {"status": "unknown"})
        self.config = config or {}
        self.last_check_result = None
        self.last_check_time = None
        self.check_count = 0
        self.error_count = 0


class DependencyChecker:
    """
    依赖服务检查器
    
    职责：
    - 依赖服务注册和管理
    - 依赖服务健康检查
    - 依赖服务状态跟踪
    """
    
    def __init__(self):
        """初始化依赖检查器"""
        self.dependencies: List[DependencyService] = []
        self._last_check_time = None
        
    def add_dependency_check(
        self,
        dep_config: Union[DependencyConfig, str],
        check_func: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加依赖服务检查
        
        Args:
            dep_config: 依赖服务配置参数对象
            
        Returns:
            bool: 是否添加成功
        """
        try:
            if isinstance(dep_config, DependencyConfig):
                config_obj = dep_config
            else:
                if check_func is None:
                    raise ValueError("check_func 必须提供")
                config_obj = DependencyConfig(
                    name=dep_config,
                    check_func=check_func,
                    config=config
                )

            # 检查是否已存在同名服务
            if any(dep.name == config_obj.name for dep in self.dependencies):
                logger.warning(f"依赖服务已存在: {config_obj.name}")
                return False
            
            service = DependencyService(
                name=config_obj.name,
                check_func=config_obj.check_func,
                config=config_obj.get_effective_config()
            )
            self.dependencies.append(service)
            
            logger.info(f"添加依赖服务检查: {config_obj.name}")
            return True
            
        except Exception as e:
            service_name = dep_config.name if isinstance(dep_config, DependencyConfig) else dep_config
            logger.error(f"添加依赖服务检查失败 {service_name}: {e}")
            return False
    
    def add_dependency_check_legacy(self, name: str, check_func: Callable, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加依赖服务检查（传统参数方式，向后兼容）
        
        Args:
            name: 服务名称
            check_func: 检查函数
            config: 配置参数
            
        Returns:
            bool: 是否添加成功
        """
        return self.add_dependency_check(name, check_func, config)
    
    def remove_dependency_check(self, name: str) -> bool:
        """
        移除依赖服务检查
        
        Args:
            name: 服务名称
            
        Returns:
            bool: 是否移除成功
        """
        try:
            original_count = len(self.dependencies)
            self.dependencies = [dep for dep in self.dependencies if dep.name != name]
            
            if len(self.dependencies) < original_count:
                logger.info(f"移除依赖服务检查: {name}")
                return True
            else:
                logger.warning(f"依赖服务不存在: {name}")
                return False
                
        except Exception as e:
            logger.error(f"移除依赖服务检查失败 {name}: {e}")
            return False
    
    async def check_all_dependencies(self) -> List[Dict[str, Any]]:
        """
        检查所有依赖服务
        
        Returns:
            List[Dict[str, Any]]: 所有依赖服务的检查结果
        """
        self._last_check_time = datetime.now()
        results = []
        
        for service in self.dependencies:
            try:
                result = await self._check_single_dependency(service)
                results.append(result)
            except Exception as e:
                logger.error(f"检查依赖服务失败 {service.name}: {e}")
                results.append({
                    "name": service.name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": self._last_check_time.isoformat()
                })
        
        return results
    
    async def _check_single_dependency(self, service: DependencyService) -> Dict[str, Any]:
        """
        检查单个依赖服务
        
        Args:
            service: 依赖服务对象
            
        Returns:
            Dict[str, Any]: 检查结果
        """
        start_time = datetime.now()
        
        try:
            service.check_count += 1
            
            # 执行检查函数
            result = service.check_func()
            
            # 如果返回异步对象，等待结果
            if asyncio.iscoroutine(result):
                result = await result
            
            # 判断健康状态
            is_healthy = self._evaluate_health_status(result)
            status = "healthy" if is_healthy else "unhealthy"
            
            # 更新服务状态
            service.last_check_result = result
            service.last_check_time = datetime.now()
            
            return {
                "name": service.name,
                "status": status,
                "details": str(result),
                "timestamp": start_time.isoformat(),
                "response_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "check_count": service.check_count,
                "error_count": service.error_count
            }
            
        except Exception as e:
            service.error_count += 1
            logger.error(f"依赖服务检查异常 {service.name}: {e}")
            
            return {
                "name": service.name,
                "status": "error",
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "check_count": service.check_count,
                "error_count": service.error_count
            }
    
    def _evaluate_health_status(self, result: Any) -> bool:
        """
        评估健康状态
        
        Args:
            result: 检查结果
            
        Returns:
            bool: 是否健康
        """
        try:
            # 如果结果是布尔值，直接返回
            if isinstance(result, bool):
                return result
            
            # 如果结果是字典，查找status字段
            if isinstance(result, dict):
                status = result.get("status", "").lower()
                return status in ["healthy", "ok", "success", "ready"]
            
            # 如果结果是字符串，检查关键词
            if isinstance(result, str):
                result_lower = result.lower()
                return any(keyword in result_lower for keyword in ["healthy", "ok", "success", "ready"])
            
            # 其他情况，如果结果非空且非None，认为健康
            return result is not None and result != ""
            
        except Exception as e:
            logger.error(f"评估健康状态失败: {e}")
            return False
    
    def check_dependencies_health(self) -> Dict[str, Any]:
        """
        检查依赖服务健康状态的同步方法
        
        Returns:
            Dict[str, Any]: 依赖服务健康状态
        """
        try:
            logger.debug("执行依赖服务健康检查")
            
            if not self.dependencies:
                return self._create_no_dependencies_response()
            
            results = self._check_all_dependencies_sync()
            return self._create_health_check_response(results)
            
        except Exception as e:
            logger.error(f"依赖服务健康检查失败: {e}")
            return self._create_error_response(str(e))

    def _create_no_dependencies_response(self) -> Dict[str, Any]:
        """创建无依赖服务的响应"""
        return {
            "status": "success",
            "message": "没有配置依赖服务",
            "dependencies": [],
            "total_count": 0,
            "healthy_count": 0,
            "timestamp": datetime.now().isoformat()
        }

    def _check_all_dependencies_sync(self) -> List[Dict[str, Any]]:
        """同步检查所有依赖服务"""
        results = []
        
        for service in self.dependencies:
            result = self._check_single_dependency_sync(service)
            results.append(result)
        
        return results

    def _check_single_dependency_sync(self, service) -> Dict[str, Any]:
        """检查单个依赖服务"""
        try:
            result = service.check_func()
            is_healthy = self._evaluate_health_status(result)
            
            return {
                "name": service.name,
                "status": "healthy" if is_healthy else "unhealthy",
                "details": str(result),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"检查依赖服务 {service.name} 失败: {e}")
            return {
                "name": service.name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_health_check_response(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建健康检查响应"""
        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        total_count = len(results)
        
        return {
            "status": "success",
            "message": f"检查了 {total_count} 个依赖服务",
            "dependencies": results,
            "total_count": total_count,
            "healthy_count": healthy_count,
            "unhealthy_count": total_count - healthy_count,
            "timestamp": datetime.now().isoformat()
        }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "status": "error",
            "message": f"依赖服务健康检查失败: {error_message}",
            "dependencies": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_dependencies_summary(self) -> Dict[str, Any]:
        """获取依赖服务摘要信息"""
        try:
            total_count = len(self.dependencies)
            total_checks = sum(dep.check_count for dep in self.dependencies)
            total_errors = sum(dep.error_count for dep in self.dependencies)
            
            return {
                "total_dependencies": total_count,
                "total_checks": total_checks,
                "total_errors": total_errors,
                "error_rate": round(total_errors / total_checks * 100, 2) if total_checks > 0 else 0,
                "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
                "dependencies": [
                    {
                        "name": dep.name,
                        "check_count": dep.check_count,
                        "error_count": dep.error_count,
                        "last_check_time": dep.last_check_time.isoformat() if dep.last_check_time else None
                    }
                    for dep in self.dependencies
                ]
            }
        except Exception as e:
            logger.error(f"获取依赖服务摘要失败: {e}")
            return {"error": str(e)}
    
    def get_dependency_by_name(self, name: str) -> Optional[DependencyService]:
        """根据名称获取依赖服务"""
        for dep in self.dependencies:
            if dep.name == name:
                return dep
        return None
