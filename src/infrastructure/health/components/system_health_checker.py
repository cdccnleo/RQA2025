"""
系统健康检查器

负责系统资源监控和健康状态检查，包括CPU、内存、磁盘等。
"""

import asyncio
import psutil
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .parameter_objects import SystemHealthInfo

logger = get_unified_logger(__name__)


class SystemHealthChecker:
    """
    系统健康检查器
    
    职责：
    - 系统资源监控
    - 系统健康状态评估
    - 系统性能指标收集
    """
    
    def __init__(self):
        """初始化系统健康检查器"""
        self._last_check_time = None
        self._check_count = 0
        
    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统健康状态
        
        Returns:
            Dict[str, Any]: 系统健康状态信息
        """
        try:
            self._check_count += 1
            self._last_check_time = datetime.now()
            
            # 获取系统资源信息
            cpu_info = self._get_cpu_info()
            memory_info = self._get_memory_info()
            disk_info = self._get_disk_info()
            process_info = self._get_process_info()
            
            # 评估整体健康状态
            health_info = SystemHealthInfo(
                cpu_info=cpu_info,
                memory_info=memory_info,
                disk_info=disk_info
            )
            overall_status = self._evaluate_overall_status(health_info)
            
            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "process": process_info,
                "check_count": self._check_count,
                "last_check_time": self._last_check_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取系统健康状态失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU使用信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            return {
                "usage_percent": round(cpu_percent, 2),
                "count": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
                "load_average": load_avg,
                "status": self._evaluate_cpu_status(cpu_percent)
            }
        except Exception as e:
            logger.error(f"获取CPU信息失败: {e}")
            return {"error": str(e)}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        try:
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            return {
                "total_gb": round(virtual_memory.total / (1024**3), 2),
                "used_gb": round(virtual_memory.used / (1024**3), 2),
                "available_gb": round(virtual_memory.available / (1024**3), 2),
                "usage_percent": round(virtual_memory.percent, 2),
                "swap_total_gb": round(swap_memory.total / (1024**3), 2),
                "swap_used_gb": round(swap_memory.used / (1024**3), 2),
                "swap_percent": round(swap_memory.percent, 2),
                "status": self._evaluate_memory_status(virtual_memory.percent)
            }
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {"error": str(e)}
    
    def _get_disk_info(self) -> Dict[str, Any]:
        """获取磁盘使用信息"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "usage_percent": round(disk_usage.percent, 2),
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0,
                "status": self._evaluate_disk_status(disk_usage.percent)
            }
        except Exception as e:
            logger.error(f"获取磁盘信息失败: {e}")
            return {"error": str(e)}
    
    def _get_process_info(self) -> Dict[str, Any]:
        """获取进程信息"""
        try:
            current_process = psutil.Process()
            create_time = current_process.create_time()
            uptime_seconds = time.time() - create_time
            
            return {
                "pid": current_process.pid,
                "cpu_percent": round(current_process.cpu_percent(), 2),
                "memory_percent": round(current_process.memory_percent(), 2),
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_formatted": str(datetime.now() - datetime.fromtimestamp(create_time)),
                "num_threads": current_process.num_threads(),
                "status": current_process.status()
            }
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            return {"error": str(e)}
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """检查CPU使用率"""
        try:
            cpu_percent = round(psutil.cpu_percent(interval=0.1), 2)
            status = self._evaluate_cpu_status(cpu_percent)
            message_map = {
                "healthy": f"CPU使用率正常: {cpu_percent:.1f}%",
                "warning": f"CPU使用率过高: {cpu_percent:.1f}%",
                "critical": f"CPU使用率严重过高: {cpu_percent:.1f}%",
            }
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "thresholds": {"warning": 80.0, "critical": 98.0},
                "message": message_map.get(status, ""),
            }
        except Exception as e:
            logger.error(f"获取CPU信息失败: {e}")
            return {
                "status": "unknown",
                "message": f"无法获取CPU信息: {e}",
                "error": str(e),
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用率"""
        try:
            virtual_memory = psutil.virtual_memory()
            available_gb = round(virtual_memory.available / (1024 ** 3), 2)
            usage_percent = round(virtual_memory.percent, 2)
            status = self._evaluate_memory_status(usage_percent)
            message_map = {
                "healthy": f"内存使用率正常: {usage_percent:.1f}%",
                "warning": f"内存使用率偏高: {usage_percent:.1f}%",
                "critical": f"内存使用率严重偏高: {usage_percent:.1f}%",
            }
            return {
                "status": status,
                "memory_percent": usage_percent,
                "available_gb": available_gb,
                "thresholds": {"warning": 85.0, "critical": 95.0},
                "message": message_map.get(status, ""),
            }
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {
                "status": "unknown",
                "message": f"无法获取内存信息: {e}",
                "error": str(e),
            }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """检查磁盘使用率"""
        try:
            disk_usage = psutil.disk_usage("/")
            free_gb = round(disk_usage.free / (1024 ** 3), 2)
            usage_percent = round(disk_usage.percent, 2)
            status = self._evaluate_disk_status(usage_percent)
            message_map = {
                "healthy": f"磁盘使用率正常: {usage_percent:.1f}%",
                "warning": f"磁盘使用率偏高: {usage_percent:.1f}%",
                "critical": f"磁盘使用率严重偏高: {usage_percent:.1f}%",
            }
            return {
                "status": status,
                "disk_percent": usage_percent,
                "free_gb": free_gb,
                "thresholds": {"warning": 85.0, "critical": 95.0},
                "message": message_map.get(status, ""),
            }
        except Exception as e:
            logger.error(f"获取磁盘信息失败: {e}")
            return {
                "status": "unknown",
                "message": f"无法获取磁盘信息: {e}",
                "error": str(e),
            }

    def _check_process_health(self) -> Dict[str, Any]:
        """检查关键进程健康状况"""
        critical_process_names = {"sshd", "nginx", "redis", "postgres", "mysql"}
        try:
            processes = list(psutil.process_iter(["pid", "name", "status"]))
            critical_processes: List[Dict[str, Any]] = []

            for proc in processes:
                info = proc.info or {}
                name = (info.get("name") or "").lower()
                status = (info.get("status") or "").lower()
                if name in critical_process_names and status not in {"running", "sleeping"}:
                    critical_processes.append(info)

            status = "warning" if critical_processes else "healthy"
            message = (
                "关键进程存在异常状态"
                if critical_processes
                else "关键进程运行正常"
            )
            return {
                "status": status,
                "total_processes": len(processes),
                "critical_processes": critical_processes,
                "message": message,
            }
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            return {
                "status": "unknown",
                "message": f"无法获取进程信息: {e}",
                "error": str(e),
            }

    def run_health_checks(self) -> Dict[str, Any]:
        """执行完整的系统健康检查"""
        cpu_result = self._check_cpu_usage()
        memory_result = self._check_memory_usage()
        disk_result = self._check_disk_usage()
        process_result = self._check_process_health()

        cpu_stub = {"status": cpu_result.get("status", "unknown")}
        memory_stub = {"status": memory_result.get("status", "unknown")}
        disk_stub = {"status": disk_result.get("status", "unknown")}
        overall_status = self._evaluate_overall_status_legacy(cpu_stub, memory_stub, disk_stub)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "checks": {
                "cpu": cpu_result,
                "memory": memory_result,
                "disk": disk_result,
                "process": process_result,
            },
            "metadata": {
                "check_count": self._check_count,
                "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            },
        }

    async def check_health_async(self) -> Dict[str, Any]:
        """异步执行系统健康检查"""
        try:
            if hasattr(asyncio, "to_thread"):
                return await asyncio.to_thread(self.run_health_checks)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.run_health_checks)
        except RuntimeError:
            # 没有事件循环时直接同步执行
            return self.run_health_checks()

    def check_health_sync(self) -> Dict[str, Any]:
        """同步执行系统健康检查（兼容旧接口）"""
        async_result = self.check_health_async()

        if isinstance(async_result, asyncio.Future):
            if async_result.done():
                return async_result.result()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("事件循环正在运行，无法同步等待Future结果")
            return loop.run_until_complete(async_result)

        if asyncio.iscoroutine(async_result):
            try:
                return asyncio.run(async_result)
            except RuntimeError:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = asyncio.ensure_future(async_result, loop=loop)
                    if task.done():
                        return task.result()
                    raise RuntimeError("事件循环正在运行，无法同步等待协程结果")
                return loop.run_until_complete(async_result)

        return async_result

    def run_health_checks_legacy(self) -> Dict[str, Any]:
        """向后兼容的健康检查方法"""
        return self.run_health_checks()

    def _evaluate_cpu_status(self, cpu_percent: float) -> str:
        """评估CPU状态"""
        if cpu_percent >= 98:
            return "critical"
        elif cpu_percent >= 80:
            return "warning"
        else:
            return "healthy"
    
    def _evaluate_memory_status(self, memory_percent: float) -> str:
        """评估内存状态"""
        if memory_percent >= 95:
            return "critical"
        elif memory_percent >= 85:
            return "warning"
        else:
            return "healthy"
    
    def _evaluate_disk_status(self, disk_percent: float) -> str:
        """评估磁盘状态"""
        if disk_percent >= 95:
            return "critical"
        elif disk_percent >= 85:
            return "warning"
        else:
            return "healthy"
    
    def _evaluate_overall_status(self, health_info: SystemHealthInfo) -> str:
        """评估整体健康状态"""
        statuses = health_info.get_all_statuses()
        
        # 确定整体状态
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"
    
    def _evaluate_overall_status_legacy(self, cpu_info: Dict, memory_info: Dict, disk_info: Dict) -> str:
        """评估整体健康状态（传统参数方式，向后兼容）"""
        health_info = SystemHealthInfo(
            cpu_info=cpu_info,
            memory_info=memory_info,
            disk_info=disk_info
        )
        return self._evaluate_overall_status(health_info)
    
    def check_system_health_status(self) -> Dict[str, Any]:
        """检查系统健康状态的详细方法"""
        try:
            logger.debug("执行系统健康状态检查")
            
            health_data = self.get_system_health()
            
            return {
                "status": "success",
                "message": "系统健康状态检查完成",
                "data": health_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"系统健康状态检查失败: {e}")
            return {
                "status": "error",
                "message": f"系统健康状态检查失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            health_data = self.get_system_health()
            
            return {
                "system_metrics": {
                    "check_count": self._check_count,
                    "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
                    "cpu_usage": health_data.get("cpu", {}).get("usage_percent"),
                    "memory_usage": health_data.get("memory", {}).get("usage_percent"),
                    "disk_usage": health_data.get("disk", {}).get("usage_percent"),
                    "overall_status": health_data.get("status")
                }
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {"error": str(e)}
