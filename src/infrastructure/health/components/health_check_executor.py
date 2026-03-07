"""
健康检查执行器

负责执行具体的健康检查逻辑，包括重试机制、超时控制、结果处理等功能。
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.infrastructure.logging.core.unified_logger import get_unified_logger
from .parameter_objects import ExecutorConfig, HealthCheckConfig, HealthCheckResult

logger = get_unified_logger(__name__)

# 常量定义
DEFAULT_SERVICE_TIMEOUT = 5.0
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_CONCURRENT_LIMIT = 10

# 阈值常量
RESPONSE_TIME_WARNING_THRESHOLD = 2.0
RESPONSE_TIME_CRITICAL_THRESHOLD = 5.0
CPU_USAGE_WARNING_THRESHOLD = 80
CPU_USAGE_CRITICAL_THRESHOLD = 95
MEMORY_USAGE_WARNING_THRESHOLD = 85
MEMORY_USAGE_CRITICAL_THRESHOLD = 95
DISK_USAGE_WARNING_THRESHOLD = 80
DISK_USAGE_CRITICAL_THRESHOLD = 95


class HealthCheckExecutor:
    """
    健康检查执行器
    
    职责：
    - 执行健康检查函数
    - 管理重试机制
    - 处理超时控制
    - 执行系统资源检查
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        """
        初始化执行器
        
        Args:
            config: 执行器配置参数对象，如果为None则使用默认配置
        """
        if config is None:
            config = ExecutorConfig()
            
        self._timeout = config.timeout
        self._retry_count = config.retry_count
        self._retry_delay = config.retry_delay
        self._concurrent_limit = config.concurrent_limit
        try:
            self._semaphore = asyncio.Semaphore(config.concurrent_limit)
        except RuntimeError:
            # 如果没有事件循环，使用Mock
            self._semaphore = None
        self._executor = ThreadPoolExecutor(max_workers=config.concurrent_limit)
    
    @classmethod
    def create_with_params(cls, 
                          timeout: float = DEFAULT_SERVICE_TIMEOUT,
                          retry_count: int = DEFAULT_RETRY_COUNT,
                          retry_delay: float = DEFAULT_RETRY_DELAY,
                          concurrent_limit: int = DEFAULT_CONCURRENT_LIMIT):
        """
        使用传统参数创建执行器实例（向后兼容）
        
        Args:
            timeout: 默认超时时间
            retry_count: 重试次数
            retry_delay: 重试延迟
            concurrent_limit: 并发限制
        """
        config = ExecutorConfig(
            timeout=timeout,
            retry_count=retry_count,
            retry_delay=retry_delay,
            concurrent_limit=concurrent_limit
        )
        return cls(config)
    
    async def execute_check_with_retry(self, check_config: HealthCheckConfig) -> Dict[str, Any]:
        """
        执行健康检查并支持重试
        
        Args:
            check_config: 健康检查配置参数对象
            
        Returns:
            检查结果
        """
        last_exception = None
        timeout = check_config.timeout or config.get('timeout', self._timeout) if (config := check_config.config) else self._timeout
        
        for attempt in range(self._retry_count + 1):
            try:
                async with self._semaphore:
                    return await self._execute_single_check(check_config, timeout)
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"健康检查失败 (尝试 {attempt + 1}/{self._retry_count + 1}): {check_config.name} - {e}")
                
                if attempt < self._retry_count:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))  # 指数退避
        
        # 所有重试都失败
        return self._create_error_result(check_config.name, last_exception, timeout)
    
    async def execute_check_with_retry_legacy(self, 
                                             name: str, 
                                             check_func: Callable, 
                                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行健康检查并支持重试（传统参数方式，向后兼容）
        
        Args:
            name: 检查名称
            check_func: 检查函数
            config: 配置参数
            
        Returns:
            检查结果
        """
        check_config = HealthCheckConfig(
            name=name,
            check_func=check_func,
            config=config
        )
        return await self.execute_check_with_retry(check_config)
    
    async def _execute_single_check(self, check_config: HealthCheckConfig, timeout: float) -> Dict[str, Any]:
        """执行单次健康检查"""
        start_time = time.time()
        
        config = check_config.config or {}
        
        try:
            # 检查函数是否是可等待的
            if asyncio.iscoroutinefunction(check_config.check_func):
                result = await asyncio.wait_for(check_config.check_func(**config), timeout=timeout)
            else:
                # 同步函数在线程池中执行
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, lambda: check_config.check_func(**config)),
                    timeout=timeout
                )
            
            response_time = time.time() - start_time
            
            # 格式化结果
            if isinstance(result, dict):
                result.setdefault('service', check_config.name)
                result.setdefault('response_time', response_time)
                result.setdefault('timestamp', datetime.now().isoformat())
                return result
            else:
                return {
                    'service': check_config.name,
                    'status': 'healthy' if result else 'warning',
                    'result': result,
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat()
                }
                
        except asyncio.TimeoutError:
            logger.error(f"健康检查超时: {check_config.name} ({timeout}s)")
            return {
                'service': check_config.name,
                'status': 'critical',
                'error': f'检查超时 ({timeout}s)',
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"健康检查异常: {check_config.name} - {e}")
            return {
                'service': check_config.name,
                'status': 'DOWN',
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_error_result(self, name: str, exception: Exception, timeout: float) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'service': name,
            'status': 'DOWN',
            'error': f'重试失败: {str(exception)}',
            'response_time': timeout,
            'timestamp': datetime.now().isoformat()
        }
    
    # =========================================================================
    # 系统资源检查方法
    # =========================================================================
    
    async def check_cpu_health(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查CPU健康状态"""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            details = self._get_cpu_usage_info(cpu_percent)
            status, message = self._evaluate_cpu_status(cpu_percent)
            
            return self._create_cpu_health_response(status, message, start_time, details)
            
        except Exception as e:
            logger.error(f"CPU健康检查失败: {e}")
            return {
                'status': 'critical',
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_memory_health(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查内存健康状态"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            details = self._get_memory_usage_info(memory)
            status, message = self._evaluate_memory_status(memory.percent)
            
            return self._create_memory_health_response(status, message, start_time, details)
            
        except Exception as e:
            logger.error(f"内存健康检查失败: {e}")
            return {
                'status': 'critical',
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def check_disk_health(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查磁盘健康状态"""
        start_time = time.time()
        
        try:
            # 检查主要磁盘分区
            partitions = psutil.disk_partitions()
            details = []
            overall_status = 'healthy'
            
            for partition in partitions:
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    usage_info = self._get_disk_usage_info(disk_usage)
                    usage_info['mountpoint'] = partition.mountpoint
                    details.append(usage_info)
                    
                    partition_status, _ = self._evaluate_disk_status(disk_usage.percent)
                    if partition_status == 'critical':
                        overall_status = 'critical'
                    elif partition_status == 'warning' and overall_status != 'critical':
                        overall_status = 'warning'
                        
                except PermissionError:
                    logger.warning(f"无法访问分区: {partition.mountpoint}")
                    continue
            
            message = f"检查了 {len(details)} 个磁盘分区"
            return self._create_disk_health_response(overall_status, message, start_time, {'partitions': details})
            
        except Exception as e:
            logger.error(f"磁盘健康检查失败: {e}")
            return {
                'status': 'critical',
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_cpu_usage_info(self, cpu_percent: float) -> Dict[str, Any]:
        """获取CPU使用信息"""
        return {
            'cpu_usage_percent': round(cpu_percent, 2),
            'cpu_count': psutil.cpu_count(),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }
    
    def _evaluate_cpu_status(self, usage_percent: float) -> tuple[str, str]:
        """评估CPU状态"""
        if usage_percent >= CPU_USAGE_CRITICAL_THRESHOLD:
            return 'critical', f'CPU使用率过高: {usage_percent:.1f}%'
        elif usage_percent >= CPU_USAGE_WARNING_THRESHOLD:
            return 'warning', f'CPU使用率较高: {usage_percent:.1f}%'
        else:
            return 'healthy', f'CPU使用率正常: {usage_percent:.1f}%'
    
    def _create_cpu_health_response(self, status: str, message: str, start_time: float,
                                   details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建CPU健康响应"""
        return {
            'status': status,
            'message': message,
            'response_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
    
    def _get_memory_usage_info(self, memory) -> Dict[str, Any]:
        """获取内存使用信息"""
        return {
            'memory_usage_percent': memory.percent,
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'memory_used_gb': round(memory.used / (1024**3), 2),
            'memory_available_gb': round(memory.available / (1024**3), 2)
        }
    
    def _evaluate_memory_status(self, usage_percent: float) -> tuple[str, str]:
        """评估内存状态"""
        if usage_percent >= MEMORY_USAGE_CRITICAL_THRESHOLD:
            return 'critical', f'内存使用率过高: {usage_percent:.1f}%'
        elif usage_percent >= MEMORY_USAGE_WARNING_THRESHOLD:
            return 'warning', f'内存使用率较高: {usage_percent:.1f}%'
        else:
            return 'healthy', f'内存使用率正常: {usage_percent:.1f}%'
    
    def _create_memory_health_response(self, status: str, message: str, start_time: float,
                                     details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建内存健康响应"""
        return {
            'status': status,
            'message': message,
            'response_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
    
    def _get_disk_usage_info(self, disk_usage) -> Dict[str, Any]:
        """获取磁盘使用信息"""
        return {
            'disk_usage_percent': disk_usage.percent,
            'disk_total_gb': round(disk_usage.total / (1024**3), 2),
            'disk_used_gb': round(disk_usage.used / (1024**3), 2),
            'disk_free_gb': round(disk_usage.free / (1024**3), 2)
        }
    
    def _evaluate_disk_status(self, usage_percent: float) -> tuple[str, str]:
        """评估磁盘状态"""
        if usage_percent >= DISK_USAGE_CRITICAL_THRESHOLD:
            return 'critical', f'磁盘使用率过高: {usage_percent:.1f}%'
        elif usage_percent >= DISK_USAGE_WARNING_THRESHOLD:
            return 'warning', f'磁盘使用率较高: {usage_percent:.1f}%'
        else:
            return 'healthy', f'磁盘使用率正常: {usage_percent:.1f}%'
    
    def _create_disk_health_response(self, status: str, message: str, start_time: float,
                                   details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建磁盘健康响应"""
        return {
            'status': status,
            'message': message,
            'response_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
    
    async def batch_execute_checks(self, 
                                 health_checks: Dict[str, Callable],
                                 check_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """批量执行健康检查"""
        tasks = []
        
        for name, check_func in health_checks.items():
            config = check_configs.get(name, {})
            task = asyncio.create_task(
                self.execute_check_with_retry(name, check_func, config)
            )
            tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"批量检查任务失败: {name} - {e}")
                results[name] = {
                    'service': name,
                    'status': 'critical',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def cleanup(self):
        """清理资源"""
        if self._executor:
            self._executor.shutdown(wait=True)
