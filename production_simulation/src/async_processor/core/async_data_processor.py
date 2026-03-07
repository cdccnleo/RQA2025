#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层异步处理系统

实现异步数据加载和处理，提升数据处理并发能力。
"""

from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import asyncio
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable

# 导入本地模块
from .async_models import (
    TaskPriority,
    AsyncProcessorEventType,
    AsyncConfig,
    ProcessingStats
)
from .async_event_handler import AsyncEventHandler

# 创建日志记录器
logger = logging.getLogger(__name__)

# 使用统一基础设施集成管理器
try:
    from src.core.integration import (
        get_data_layer_adapter,
        log_data_operation, record_data_metric, get_data_config
    )
    # 获取基础设施服务
    _data_adapter = None

    def get_data_integration_manager() -> Any:
        """get_data_integration_manager 函数的文档字符串"""

        global _data_adapter
        if _data_adapter is None:
            _data_adapter = get_data_layer_adapter()
        return _data_adapter

    def get_data_cache(key, default=None) -> Any:
        """get_data_cache 函数的文档字符串"""

        adapter = get_data_integration_manager()
        if adapter and hasattr(adapter, 'get_cache_manager'):
            cache_mgr = adapter.get_cache_manager()
            return cache_mgr.get(key, default) if cache_mgr else default
        return default

    def set_data_cache(key, value, ttl=None) -> Any:
        """set_data_cache 函数的文档字符串"""

        adapter = get_data_integration_manager()
        if adapter and hasattr(adapter, 'get_cache_manager'):
            cache_mgr = adapter.get_cache_manager()
            if cache_mgr:
                cache_mgr.set(key, value, ttl)

    def publish_data_event(event_type, data) -> Any:
        """publish_data_event 函数的文档字符串"""

        adapter = get_data_integration_manager()
        if adapter and hasattr(adapter, 'get_event_bus'):
            event_bus = adapter.get_event_bus()
            if event_bus:
                event_bus.publish_sync({
                    'event_type': event_type,
                    'data': data,
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                })

    def perform_data_health_check() -> Any:
        """perform_data_health_check 函数的文档字符串"""

        adapter = get_data_integration_manager()
        if adapter and hasattr(adapter, 'health_check'):
            return adapter.health_check()
        return True

except ImportError as e:
    # 如果统一集成管理器不存在，使用默认实现
    logger.warning(
        f"Unified infrastructure integration manager not found ({e}), using default implementations")

    class DefaultIntegrationManager:
        def __init__(self):
            self._initialized = True

        def initialize(self):
            pass

        def get_config(self, key=None, default=None):
            return default

        def get_cache(self, key=None, default=None):
            return default

        def set_cache(self, key, value, ttl=None):
            pass

        def log_operation(self, operation, **kwargs):
            pass

        def record_metric(self, name, value, **kwargs):
            pass

        def publish_event(self, event_type, data):
            pass

        def health_check(self):
            return True

    def get_data_integration_manager(): return DefaultIntegrationManager()
    def get_data_cache(key, default=None): return default
    def set_data_cache(key, value, ttl=None): return None
    def get_data_config(key, default=None): return default
    def log_data_operation(operation, *args, **kwargs): pass
    record_data_metric = lambda name, value, **kwargs: None
    def publish_data_event(event_type, data): return None
    def perform_data_health_check(): return True

# 导入标准接口
try:
    from ..interfaces.standard_interfaces import (
        DataRequest, DataResponse, DataSourceType, IDataAdapter
    )
except ImportError:
    # 如果接口不存在，使用默认实现
    logger.warning("Standard interfaces not found, using default implementations")

    @dataclass
    class DataRequest:
        data_type: str = ""
        parameters: Dict[str, Any] = field(default_factory=dict)
        timeout: float = 30.0

    @dataclass
    class DataResponse:
        success: bool = True
        data: Any = None
        error: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    class DataSourceType:
        STOCK = "stock"
        CRYPTO = "crypto"
        FOREX = "forex"
        NEWS = "news"

    class IDataAdapter:
        pass

# 确保类在模块级别可用
if 'DataSourceType' not in globals():
    class DataSourceType:
        STOCK = "stock"
        CRYPTO = "crypto"
        FOREX = "forex"
        NEWS = "news"

# 导入任务调度器
try:
    from .task_scheduler import TaskScheduler as AsyncTaskScheduler
    from .task_scheduler import TaskPriority
except ImportError:
    # 如果调度器不存在，使用默认实现
    logger.warning("Async task scheduler not found, using default implementations")

# 数据模型已移至async_models.py
# 事件处理已移至async_event_handler.py


class AsyncDataProcessor:

    """
    异步数据处理器

    提供异步数据加载和批量处理能力，提升数据处理并发性能。
    """

    def __init__(self, config: Optional[AsyncConfig] = None):
        """__init__ 函数的文档字符串"""

        # 使用基础设施集成管理器获取配置
        self.config_obj = config or AsyncConfig()
        merged_config = self._load_config_from_integration_manager()
        self.config = AsyncConfig(**merged_config)

        # 初始化基础设施集成管理器
        self.integration_manager = get_data_integration_manager()
        if not self.integration_manager._initialized:
            self.integration_manager.initialize()

        self.semaphore = Semaphore(self.config.max_concurrent_requests)
        self.stats = ProcessingStats()

        # 创建线程池和进程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        if self.config.enable_process_pool:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_processes)
        else:
            self.process_pool = None

        # 事件循环管理
        self.loop = None
        self.loop_thread = None

        # 智能调度器
        self.task_scheduler = self._create_task_scheduler()

        # 事件驱动架构
        self.event_handlers = {}
        self.event_queue = asyncio.Queue() if asyncio else None
        self.event_processor_task = None

        # 注册事件处理器
        self._register_event_handlers()

        # 健康检查注册
        self._register_health_checks()

        log_data_operation("async_processor_init", "stock",
                           {"max_workers": self.config.max_workers, "max_concurrent": self.config.max_concurrent_requests}, "info")

    def _register_event_handlers(self) -> Any:
        """注册事件处理器"""
        self.event_handlers = {
            AsyncProcessorEventType.TASK_STARTED: self._handle_task_started,
            AsyncProcessorEventType.TASK_COMPLETED: self._handle_task_completed,
            AsyncProcessorEventType.TASK_FAILED: self._handle_task_failed,
            AsyncProcessorEventType.BATCH_STARTED: self._handle_batch_started,
            AsyncProcessorEventType.BATCH_COMPLETED: self._handle_batch_completed,
            AsyncProcessorEventType.PROCESSOR_OVERLOADED: self._handle_processor_overloaded,
            AsyncProcessorEventType.CONFIG_UPDATED: self._handle_config_updated,
        }

        # 启动事件处理器任务
        if self.loop and asyncio:
            self.event_processor_task = asyncio.run_coroutine_threadsafe(
                self._process_events(), self.loop
            )

    async def _process_events(self):
        """异步处理事件队列"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._dispatch_event(event)
                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"事件处理异常: {e}")

    async def _dispatch_event(self, event):
        """分发事件到对应处理器"""
        event_type = event.get('type')
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"事件处理器异常 ({event_type}): {e}")
        else:
            logger.debug(f"未找到事件处理器: {event_type}")

    def publish_event(self, event_type: str, data: Dict[str, Any], priority: str = "normal"):
        """发布异步事件"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'source': 'AsyncDataProcessor'
        }

        # 添加到异步队列
        if self.event_queue and asyncio:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(event), self.loop
                )
            except Exception as e:
                logger.error(f"发布事件失败: {e}")

        # 同时发布到统一事件总线
        if event_bus_available:
            try:
                event_obj = Event(
                    type=EventType.CUSTOM,
                    data={
                        'async_event_type': event_type,
                        'async_data': data,
                        'source': 'AsyncDataProcessor'
                    },
                    priority=EventPriority.NORMAL if priority == "normal" else EventPriority.HIGH
                )
                # 假设有全局事件总线实例
                # global_event_bus.publish(event_obj)
            except Exception as e:
                logger.debug(f"事件总线发布失败: {e}")

    # 事件处理器方法
    async def _handle_task_started(self, event):
        """处理任务开始事件"""
        task_id = event['data'].get('task_id')
        logger.info(f"任务开始处理: {task_id}")
        record_data_metric("async_task_started", 1, "count", {"task_id": task_id})

    async def _handle_task_completed(self, event):
        """处理任务完成事件"""
        task_id = event['data'].get('task_id')
        duration = event['data'].get('duration', 0)
        logger.info(f"任务完成: {task_id}, 耗时: {duration:.2f}s")
        record_data_metric("async_task_completed", 1, "count", {
                           "task_id": task_id, "duration": duration})

    async def _handle_task_failed(self, event):
        """处理任务失败事件"""
        task_id = event['data'].get('task_id')
        error = event['data'].get('error', 'Unknown error')
        logger.warning(f"任务失败: {task_id}, 错误: {error}")
        record_data_metric("async_task_failed", 1, "count", {
                           "task_id": task_id, "error": str(error)})

    async def _handle_batch_started(self, event):
        """处理批量任务开始事件"""
        batch_id = event['data'].get('batch_id')
        task_count = event['data'].get('task_count', 0)
        logger.info(f"批量任务开始: {batch_id}, 任务数: {task_count}")
        record_data_metric("async_batch_started", task_count,
                           "count", {"batch_id": batch_id})

    async def _handle_batch_completed(self, event):
        """处理批量任务完成事件"""
        batch_id = event['data'].get('batch_id')
        total_duration = event['data'].get('total_duration', 0)
        success_count = event['data'].get('success_count', 0)
        logger.info(
            f"批量任务完成: {batch_id}, 总耗时: {total_duration:.2f}s, 成功: {success_count}")
        record_data_metric("async_batch_completed", success_count,
                           "count", {"batch_id": batch_id})

    async def _handle_processor_overloaded(self, event):
        """处理处理器过载事件"""
        load_factor = event['data'].get('load_factor', 0)
        logger.warning(f"异步处理器过载，负载因子: {load_factor}")
        record_data_metric("async_processor_overloaded", load_factor, "gauge")

        # 触发降级措施
        await self._apply_overload_protection()

    async def _handle_config_updated(self, event):
        """处理配置更新事件"""
        config_changes = event['data'].get('changes', {})
        logger.info(f"异步处理器配置更新: {config_changes}")

        # 重新配置组件
        await self._reconfigure_components(config_changes)

    async def _apply_overload_protection(self):
        """应用过载保护措施"""
        # 减少并发数
        original_limit = self.config.max_concurrent_requests
        self.config.max_concurrent_requests = max(1, original_limit // 2)
        self.semaphore = Semaphore(self.config.max_concurrent_requests)

        logger.info(
            f"过载保护激活，并发限制从 {original_limit} 降至 {self.config.max_concurrent_requests}")

        # 5分钟后恢复
        if asyncio:
            asyncio.run_coroutine_threadsafe(
                self._restore_normal_capacity(), self.loop
            )

    async def _restore_normal_capacity(self):
        """恢复正常容量"""
        await asyncio.sleep(300)  # 5分钟
        original_limit = getattr(self, '_original_concurrent_limit',
                                 self.config.max_concurrent_requests * 2)
        self.config.max_concurrent_requests = original_limit
        self.semaphore = Semaphore(self.config.max_concurrent_requests)
        logger.info(f"容量恢复正常: {original_limit}")

        # 发布恢复事件
        self.publish_event(AsyncProcessorEventType.PROCESSOR_RECOVERED, {
            'original_limit': original_limit
        })

    async def _reconfigure_components(self, config_changes):
        """重新配置组件"""
        if 'max_workers' in config_changes:
            # 重新创建线程池
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            self.config.max_workers = config_changes['max_workers']
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)

        if 'max_concurrent_requests' in config_changes:
            self.config.max_concurrent_requests = config_changes['max_concurrent_requests']
            self.semaphore = Semaphore(self.config.max_concurrent_requests)

    def _load_config_from_integration_manager(self) -> Dict[str, Any]:
        """从基础设施集成管理器加载配置"""
        try:
            merged_config = self.config_obj.__dict__.copy()

            # 从基础设施集成管理器获取配置
            infra_config = {
                'max_concurrent_requests': get_data_config('max_concurrent_requests', self.config_obj.max_concurrent_requests),
                'request_timeout': get_data_config('request_timeout', self.config_obj.request_timeout),
                'max_workers': get_data_config('max_workers', self.config_obj.max_workers),
                'enable_process_pool': get_data_config('enable_process_pool', self.config_obj.enable_process_pool),
                'max_processes': get_data_config('max_processes', self.config_obj.max_processes),
                'batch_size': get_data_config('batch_size', self.config_obj.batch_size),
                'retry_count': get_data_config('retry_count', self.config_obj.retry_count),
                'retry_delay': get_data_config('retry_delay', self.config_obj.retry_delay)
            }

            merged_config.update(infra_config)
            return merged_config

        except Exception as e:
            # 如果集成管理器不可用，使用默认配置
            return self.config_obj.__dict__.copy()

    def _create_task_scheduler(self) -> 'AsyncTaskScheduler':
        """创建智能任务调度器"""
        try:
            return AsyncTaskScheduler(
                max_workers=self.config.max_workers,
                queue_size=1000
            )
        except Exception as e:
            log_data_operation("task_scheduler_create_error", "stock",
                               {"error": str(e)}, "warning")
            return None

    def _register_health_checks(self) -> None:
        """注册健康检查"""
        try:
            health_bridge = self.integration_manager.get_health_check_bridge()
            if health_bridge:
                # 注册异步处理器健康检查
                health_bridge.register_data_health_check(
                    "async_processor",
                    self._async_processor_health_check,
                    "stock"
                )

                # 注册线程池健康检查
                health_bridge.register_data_health_check(
                    "thread_pool",
                    self._thread_pool_health_check,
                    "stock"
                )

        except Exception as e:
            log_data_operation("health_check_registration_error", "stock",
                               {"error": str(e)}, "warning")

    def _async_processor_health_check(self) -> Dict[str, Any]:
        """异步处理器健康检查"""
        try:
            health_status = {
                'component': 'AsyncDataProcessor',
                'status': 'healthy',
                'active_threads': len(self.thread_pool._threads) if hasattr(self.thread_pool, '_threads') else 0,
                'semaphore_available': self.config.max_concurrent_requests - len(self.semaphore._waiters) if hasattr(self.semaphore, '_waiters') else self.config.max_concurrent_requests,
                'total_requests': self.stats.total_requests,
                'completed_requests': self.stats.completed_requests,
                'failed_requests': self.stats.failed_requests,
                'avg_response_time': self.stats.avg_response_time,
                'throughput_per_second': self.stats.throughput_per_second,
                'timestamp': datetime.now().isoformat()
            }

            # 检查关键指标
            if self.stats.failed_requests / max(self.stats.total_requests, 1) > 0.1:  # 失败率超过10%
                health_status['status'] = 'warning'
                health_status[
                    'message'] = f'请求失败率过高: {self.stats.failed_requests / max(self.stats.total_requests, 1):.2%}'

            if self.stats.avg_response_time > 10.0:  # 平均响应时间超过10秒
                health_status['status'] = 'warning'
                health_status['message'] = f'平均响应时间过长: {self.stats.avg_response_time:.2f}s'

            return health_status

        except Exception as e:
            return {
                'component': 'AsyncDataProcessor',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _thread_pool_health_check(self) -> Dict[str, Any]:
        """线程池健康检查"""
        try:
            active_threads = len(self.thread_pool._threads) if hasattr(
                self.thread_pool, '_threads') else 0
            total_threads = self.config.max_workers

            health_status = {
                'component': 'ThreadPool',
                'status': 'healthy',
                'active_threads': active_threads,
                'total_threads': total_threads,
                'utilization_rate': active_threads / total_threads if total_threads > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }

            # 检查线程池利用率
            utilization = active_threads / total_threads if total_threads > 0 else 0
            if utilization > 0.9:  # 利用率超过90%
                health_status['status'] = 'warning'
                health_status['message'] = f'线程池利用率过高: {utilization:.2%}'

            return health_status

        except Exception as e:
            return {
                'component': 'ThreadPool',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        try:
            return {
                'total_requests': self.stats.total_requests,
                'completed_requests': self.stats.completed_requests,
                'failed_requests': self.stats.failed_requests,
                'success_rate': self.stats.completed_requests / max(self.stats.total_requests, 1),
                'avg_response_time': self.stats.avg_response_time,
                'max_response_time': self.stats.max_response_time,
                'min_response_time': self.stats.min_response_time,
                'throughput_per_second': self.stats.throughput_per_second,
                'uptime_seconds': (datetime.now() - self.stats.start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            log_data_operation("stats_retrieval_error", "stock",
                               {"error": str(e)}, "error")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def reset_stats(self) -> None:
        """重置统计信息"""
        try:
            self.stats = ProcessingStats()
            log_data_operation("stats_reset", "stock",
                               {"message": "Processing statistics have been reset"}, "info")

        except Exception as e:
            log_data_operation("stats_reset_error", "stock",
                               {"error": str(e)}, "error")

    def start_event_loop(self) -> Any:
        """启动事件循环"""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(
                target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            logger.info("异步事件循环已启动")

    def _run_event_loop(self) -> Any:
        """运行事件循环"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"事件循环运行出错: {e}")
        finally:
            self.loop.close()

    def stop_event_loop(self) -> Any:
        """停止事件循环"""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5.0)
            logger.info("异步事件循环已停止")

    async def process_request_async(self, adapter: IDataAdapter, request: DataRequest) -> DataResponse:
        """异步处理单个请求"""
        async with self.semaphore:
            start_time = time.time()

            try:
                # 在线程池中执行同步操作
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.thread_pool,
                    self._sync_get_data,
                    adapter,
                    request
                )

                response_time = time.time() - start_time
                self.stats.update_stats(response_time, response.success)

                return response

            except Exception as e:
                response_time = time.time() - start_time
                self.stats.update_stats(response_time, False)

                logger.error(f"异步处理请求失败: {e}")
                return DataResponse(
                    request=request,
                    data=None,
                    metadata={"error": str(e)},
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(e)
                )

    def _sync_get_data(self, adapter: IDataAdapter, request: DataRequest) -> DataResponse:
        """同步获取数据"""
        return adapter.get_data(request)

    async def process_batch_async(self, adapter: IDataAdapter, requests: List[DataRequest]) -> List[DataResponse]:
        """异步批量处理请求"""
        if not requests:
            return []

        # 分批处理，避免过多的并发请求
        batch_size = self.config.batch_size
        all_responses = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # 创建批量任务
            tasks = [
                self.process_request_async(adapter, request)
                for request in batch
            ]

            # 并发执行批量任务
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    # 处理异常情况
                    request = batch[j]
                    error_response = DataResponse(
                        request=request,
                        data=None,
                        metadata={"error": str(response)},
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(response)
                    )
                    all_responses.append(error_response)
                else:
                    all_responses.append(response)

        return all_responses

    async def process_with_retry_async(self, adapter: IDataAdapter, request: DataRequest,
                                       max_retries: Optional[int] = None) -> DataResponse:
        """带重试的异步处理"""
        max_retries = max_retries or self.config.retry_count
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.process_request_async(adapter, request)
                if response.success:
                    return response

                # 如果失败且还有重试次数，等待后重试
                if attempt < max_retries:
                    # 指数退避
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    logger.warning(f"请求重试 {attempt + 1}/{max_retries}: {e}")
                else:
                    logger.error(f"请求失败，已达到最大重试次数: {e}")

        # 所有重试都失败
        return DataResponse(
            request=request,
            data=None,
            metadata={"last_exception": str(
                last_exception) if last_exception else "unknown"},
            timestamp=datetime.now(),
            success=False,
            error_message=f"请求失败，已重试 {max_retries} 次"
        )

    def process_request_sync(self, adapter: IDataAdapter, request: DataRequest) -> DataResponse:
        """同步处理单个请求（用于向后兼容）"""
        if self.loop is None:
            self.start_event_loop()

        # 在事件循环中运行异步函数
        future = asyncio.run_coroutine_threadsafe(
            self.process_request_async(adapter, request),
            self.loop
        )

        try:
            return future.result(timeout=self.config.request_timeout)
        except Exception as e:
            logger.error(f"同步处理请求超时或失败: {e}")
            return DataResponse(
                request=request,
                data=None,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )

    def process_batch_sync(self, adapter: IDataAdapter, requests: List[DataRequest]) -> List[DataResponse]:
        """同步批量处理请求（用于向后兼容）"""
        if self.loop is None:
            self.start_event_loop()

        # 在事件循环中运行异步函数
        future = asyncio.run_coroutine_threadsafe(
            self.process_batch_async(adapter, requests),
            self.loop
        )

        try:
            return future.result(timeout=self.config.request_timeout * len(requests))
        except Exception as e:
            logger.error(f"同步批量处理请求超时或失败: {e}")
            return [
                DataResponse(
                    request=request,
                    data=None,
                    metadata={"error": str(e)},
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(e)
                ) for request in requests
            ]

    def __del__(self) -> Any:
        """析构函数，确保资源清理"""
        # 停止事件处理器
        if self.event_processor_task:
            self.event_processor_task.cancel()
            try:
                self.event_processor_task.result(timeout=5)
            except Exception:
                pass

        self.stop_event_loop()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool') and self.process_pool:
            self.process_pool.shutdown(wait=False)


# 工具函数

def async_adapter_wrapper(adapter: IDataAdapter) -> Callable[[DataRequest], Awaitable[DataResponse]]:
    """将同步适配器包装为异步函数"""
    async def async_wrapper(request: DataRequest) -> DataResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, adapter.get_data, request)

    return async_wrapper


def create_async_processor(config: Optional[AsyncConfig] = None) -> AsyncDataProcessor:
    """创建异步数据处理器"""
    return AsyncDataProcessor(config)


# 导出主要类和函数
__all__ = [
    'AsyncConfig',
    'ProcessingStats',
    'AsyncDataProcessor',
    'async_adapter_wrapper',
    'create_async_processor'
]
