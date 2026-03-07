"""
异步健康检查助手

提供异步健康检查的辅助功能，包括数据库检查、服务检查和全面检查。
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Coroutine, Optional

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class AsyncHealthCheckHelper:
    """
    异步健康检查助手
    
    职责：
    - 异步数据库健康检查
    - 异步服务健康检查
    - 异步全面健康检查
    - 结果分析和状态确定
    """
    
    def __init__(self, health_checker=None):
        """
        初始化异步健康检查助手
        
        Args:
            health_checker: 基础健康检查器实例
        """
        self.health_checker = health_checker or self._create_default_health_checker()

    @staticmethod
    def _create_default_health_checker():
        """创建一个最简健康检查器以保持旧接口兼容"""

        class _DefaultHealthChecker:
            async def check_health_async(self):
                return {"status": "healthy", "details": {}}

            def check_health(self):
                return {"status": "healthy", "details": {}}

            async def check_service_async(self, service_name: str):
                return {"status": "healthy", "service": service_name}

            def check_service(self, service_name: str):
                return {"status": "healthy", "service": service_name}

        return _DefaultHealthChecker()
    
    async def check_database_async(self) -> Dict[str, Any]:
        """异步检查数据库健康状态"""
        try:
            # 如果健康检查器支持异步方法，优先使用异步方法
            if hasattr(self.health_checker, 'check_health_async'):
                result = await self.health_checker.check_health_async()
                return {
                    "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
                    "database_status": result,
                    "timestamp": datetime.now().isoformat(),
                    "check_type": "async_database_check"
                }
            else:
                # 回退到同步方法
                result = self.health_checker.check_health()
                return {
                    "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
                    "database_status": result,
                    "timestamp": datetime.now().isoformat(),
                    "check_type": "sync_database_check"
                }
        except Exception as e:
            logger.error(f"异步数据库健康检查失败: {e}")
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "check_type": "async_database_check"
            }

    async def check_service_async(self, service_name: str) -> Dict[str, Any]:
        """异步检查特定服务健康状态"""
        try:
            # 如果健康检查器支持异步方法，优先使用异步方法
            if hasattr(self.health_checker, 'check_service_async'):
                result = await self.health_checker.check_service_async(service_name)
                return {
                    "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
                    "service": service_name,
                    "service_status": result,
                    "timestamp": datetime.now().isoformat(),
                    "check_type": "async_service_check"
                }
            else:
                # 回退到同步方法
                result = self.health_checker.check_service(service_name)
                return {
                    "status": "healthy" if result.get("status") == "healthy" else "unhealthy",
                    "service": service_name,
                    "service_status": result,
                    "timestamp": datetime.now().isoformat(),
                    "check_type": "sync_service_check"
                }
        except Exception as e:
            logger.error(f"异步服务健康检查失败 {service_name}: {e}")
            return {
                "status": "critical",
                "service": service_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "check_type": "async_service_check"
            }

    def create_comprehensive_check_tasks(self, service_names: Optional[List[str]] = None) -> List[Coroutine]:
        """创建全面检查任务"""
        if service_names is None:
            service_names = ["cache", "monitoring"]
        
        tasks = [self.check_database_async()]
        
        for service_name in service_names:
            tasks.append(self.check_service_async(service_name))
        
        return tasks

    def analyze_comprehensive_results(self, results: List[Any]) -> tuple[Dict[str, Any], Dict[str, int]]:
        """分析全面检查结果"""
        healthy_count = 0
        unhealthy_count = 0
        critical_count = 0
        component_results = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                critical_count += 1
                component_results[f"task_{i}"] = {"status": "error", "error": str(result)}
            elif isinstance(result, dict):
                status = result.get("status", "unknown")
                component_results[f"component_{i}"] = result

                if status == "healthy":
                    healthy_count += 1
                elif status == "unhealthy":
                    unhealthy_count += 1
                elif status == "critical":
                    critical_count += 1

        counts = {
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "critical_count": critical_count
        }

        return component_results, counts

    def determine_comprehensive_status(self, counts: Dict[str, int]) -> str:
        """确定全面检查整体状态"""
        if counts.get("critical_count", 0) > 0:
            return "critical"
        elif counts.get("unhealthy_count", 0) > 0:
            return "warning"
        else:
            return "healthy"

    def create_comprehensive_success_response(self, 
                                            overall_status: str, 
                                            start_time: float,
                                            tasks: List[Coroutine], 
                                            counts: Dict[str, int],
                                            component_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建全面检查成功响应"""
        execution_time = time.time() - start_time
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "execution_time": f"{execution_time:.3f}s",
            "components_checked": len(tasks),
            "healthy_components": counts["healthy_count"],
            "unhealthy_components": counts["unhealthy_count"],
            "critical_components": counts["critical_count"],
            "component_details": component_results,
            "check_type": "async_comprehensive_check"
        }

    def create_comprehensive_error_response(self, error: Exception) -> Dict[str, Any]:
        """创建全面检查错误响应"""
        logger.error(f"异步全面健康检查失败: {error}")
        return {
            "status": "critical",
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "check_type": "async_comprehensive_check"
        }

    async def comprehensive_health_check_async(self, service_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """异步执行全面健康检查"""
        try:
            start_time = time.time()

            # 创建检查任务
            tasks = self.create_comprehensive_check_tasks(service_names)

            # 执行并发检查
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 分析结果
            component_results, counts = self.analyze_comprehensive_results(results)
            overall_status = self.determine_comprehensive_status(counts)

            return self.create_comprehensive_success_response(
                overall_status, 
                start_time, 
                tasks, 
                counts, 
                component_results
            )

        except Exception as e:
            return self.create_comprehensive_error_response(e)

