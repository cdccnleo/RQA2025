"""
RQA2025异步处理优化系统

实现异步任务队列、协程池管理、并发控制等异步处理优化功能。
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Union
import asyncio
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aio_pika
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """异步任务"""
    id: str
    name: str
    func: Callable[..., Awaitable[Any]]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': str(self.result) if self.result is not None else None,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }


class AsyncTaskQueue:
    """异步任务队列"""

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 max_concurrent: int = 100):
        self.redis_url = redis_url
        self.max_concurrent = max_concurrent
        self.redis: Optional[redis.Redis] = None
        self.rabbitmq_connection: Optional[aio_pika.Connection] = None
        self.task_queues: Dict[TaskPriority, asyncio.Queue] = {}
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown = False

        # 初始化优先级队列
        for priority in TaskPriority:
            self.task_queues[priority] = asyncio.Queue()

    async def initialize(self):
        """初始化队列管理器"""
        # 连接Redis
        self.redis = redis.Redis.from_url(self.redis_url)

        # 连接RabbitMQ
        self.rabbitmq_connection = await aio_pika.connect_robust(self.redis_url)

        # 创建频道
        channel = await self.rabbitmq_connection.channel()

        # 声明队列
        for priority in TaskPriority:
            queue_name = f"async_tasks_{priority.name.lower()}"
            await channel.declare_queue(queue_name, durable=True)

        # 启动任务处理器
        asyncio.create_task(self._process_queues())

        logger.info("异步任务队列初始化完成")

    async def submit_task(self, name: str, func: Callable[..., Awaitable[Any]],
                         *args, priority: TaskPriority = TaskPriority.NORMAL,
                         timeout: Optional[float] = None, **kwargs) -> str:
        """提交异步任务"""
        task_id = f"task_{int(time.time() * 1000000)}"

        task = AsyncTask(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )

        # 保存到Redis
        await self.redis.setex(
            f"task:{task_id}",
            86400,  # 24小时过期
            json.dumps(task.to_dict())
        )

        # 添加到内存队列
        await self.task_queues[priority].put(task)

        # 发送到RabbitMQ
        await self._publish_task(task)

        logger.info(f"任务 {task_id} 已提交，优先级: {priority.name}")
        return task_id

    async def _publish_task(self, task: AsyncTask):
        """发布任务到消息队列"""
        if not self.rabbitmq_connection:
            return

        try:
            channel = await self.rabbitmq_connection.channel()
            queue_name = f"async_tasks_{task.priority.name.lower()}"

            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(task.to_dict()).encode()),
                routing_key=queue_name
            )
        except Exception as e:
            logger.error(f"发布任务失败: {e}")

    async def _process_queues(self):
        """处理任务队列"""
        while not self._shutdown:
            try:
                # 按优先级处理任务
                task = None

                # 先处理高优先级任务
                for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH,
                               TaskPriority.NORMAL, TaskPriority.LOW]:
                    try:
                        task = self.task_queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue

                if task:
                    asyncio.create_task(self._execute_task(task))
                else:
                    await asyncio.sleep(0.1)  # 避免忙等待

            except Exception as e:
                logger.error(f"队列处理错误: {e}")
                await asyncio.sleep(1)

    async def _execute_task(self, task: AsyncTask):
        """执行任务"""
        async with self.semaphore:
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                self.active_tasks[task.id] = task

                # 更新Redis状态
                await self._update_task_status(task)

                # 执行任务
                if task.timeout:
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    result = await task.func(*task.args, **task.kwargs)

                # 任务成功
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = result

                logger.info(f"任务 {task.id} 执行成功")

            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = "Task timeout"
                logger.error(f"任务 {task.id} 超时")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error(f"任务 {task.id} 执行失败: {e}")

                # 重试逻辑
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    await self.task_queues[task.priority].put(task)
                    logger.info(f"任务 {task.id} 将重试 ({task.retry_count}/{task.max_retries})")

            finally:
                if task.status != TaskStatus.PENDING:  # 重试任务不移除
                    task.completed_at = datetime.utcnow()
                    if task.id in self.active_tasks:
                        del self.active_tasks[task.id]

                # 更新最终状态
                await self._update_task_status(task)

    async def _update_task_status(self, task: AsyncTask):
        """更新任务状态到Redis"""
        try:
            await self.redis.setex(
                f"task:{task.id}",
                86400,
                json.dumps(task.to_dict())
            )
        except Exception as e:
            logger.error(f"更新任务状态失败: {e}")

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        try:
            task_data = await self.redis.get(f"task:{task_id}")
            if task_data:
                return json.loads(task_data)
            return None
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            # 从Redis获取任务信息
            task_data = await self.redis.get(f"task:{task_id}")
            if not task_data:
                return False

            task_dict = json.loads(task_data)
            if task_dict['status'] in ['completed', 'failed']:
                return False  # 已完成的任务不能取消

            # 更新状态为取消
            task_dict['status'] = TaskStatus.CANCELLED.value
            task_dict['completed_at'] = datetime.utcnow().isoformat()
            task_dict['error'] = "Task cancelled by user"

            await self.redis.setex(f"task:{task_id}", 86400, json.dumps(task_dict))

            # 从活跃任务中移除
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            logger.info(f"任务 {task_id} 已取消")
            return True

        except Exception as e:
            logger.error(f"取消任务失败: {e}")
            return False

    async def shutdown(self):
        """关闭任务队列"""
        self._shutdown = True

        # 等待活跃任务完成
        if self.active_tasks:
            logger.info(f"等待 {len(self.active_tasks)} 个活跃任务完成...")
            await asyncio.sleep(5)  # 给任务一些时间完成

        # 关闭连接
        if self.redis:
            await self.redis.close()

        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()

        logger.info("异步任务队列已关闭")

    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        stats = {
            'active_tasks': len(self.active_tasks),
            'queue_sizes': {},
            'total_processed': 0,
            'success_rate': 0.0
        }

        for priority, queue in self.task_queues.items():
            stats['queue_sizes'][priority.name] = queue.qsize()

        return stats


class OptimizedCoroutinePool:
    """优化的协程池"""

    def __init__(self, max_concurrent: int = 1000, batch_size: int = 100):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.task_counter = 0
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_execution_time': 0.0,
            'max_concurrent_reached': 0
        }

    async def submit(self, coro: Callable[[], Awaitable[T]],
                    task_name: str = "") -> T:
        """提交协程任务"""
        self.stats['total_submitted'] += 1
        task_id = f"coro_{self.task_counter}"
        self.task_counter += 1

        start_time = time.time()

        async def _execute_with_tracking():
            async with self.semaphore:
                try:
                    # 检查并发限制
                    current_active = len([t for t in self.active_tasks.values()
                                        if not t.done()])
                    if current_active >= self.max_concurrent:
                        self.stats['max_concurrent_reached'] += 1

                    result = await coro()

                    execution_time = time.time() - start_time
                    self.stats['total_completed'] += 1

                    # 记录完成任务
                    self.completed_tasks.append({
                        'id': task_id,
                        'name': task_name,
                        'execution_time': execution_time,
                        'status': 'completed'
                    })

                    # 更新平均执行时间
                    total_completed = self.stats['total_completed']
                    self.stats['average_execution_time'] = (
                        (self.stats['average_execution_time'] * (total_completed - 1)) +
                        execution_time
                    ) / total_completed

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.stats['total_failed'] += 1

                    self.completed_tasks.append({
                        'id': task_id,
                        'name': task_name,
                        'execution_time': execution_time,
                        'status': 'failed',
                        'error': str(e)
                    })

                    raise

        task = asyncio.create_task(_execute_with_tracking())
        self.active_tasks[task_id] = task

        try:
            return await task
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    async def submit_many(self, coros: List[Callable[[], Awaitable[T]]],
                         task_names: Optional[List[str]] = None) -> List[Union[T, Exception]]:
        """批量提交协程任务"""
        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(coros))]

        # 分批执行，避免一次性创建太多任务
        results = []

        for i in range(0, len(coros), self.batch_size):
            batch_coros = coros[i:i + self.batch_size]
            batch_names = task_names[i:i + self.batch_size]

            batch_tasks = [
                self.submit(coro, name)
                for coro, name in zip(batch_coros, batch_names)
            ]

            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批量任务执行失败: {e}")
                # 为失败的批次添加异常结果
                results.extend([e] * len(batch_tasks))

        return results

    async def wait_all(self, timeout: Optional[float] = None):
        """等待所有活跃任务完成"""
        if not self.active_tasks:
            return

        tasks = list(self.active_tasks.values())

        try:
            if timeout:
                await asyncio.wait(tasks, timeout=timeout)
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"等待任务完成时出错: {e}")

        # 清理已完成的任务
        completed_ids = [task_id for task_id, task in self.active_tasks.items()
                        if task.done()]
        for task_id in completed_ids:
            del self.active_tasks[task_id]

    def get_active_count(self) -> int:
        """获取活跃任务数量"""
        return len([t for t in self.active_tasks.values() if not t.done()])

    def get_stats(self) -> Dict[str, Any]:
        """获取协程池统计信息"""
        return {
            **self.stats,
            'active_tasks': self.get_active_count(),
            'recent_completed': self.completed_tasks[-10:] if self.completed_tasks else []
        }


class HybridExecutor:
    """混合执行器 - 结合线程池和协程池"""

    def __init__(self, max_workers: int = 10, max_concurrent: int = 100):
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers // 2)
        self.coroutine_pool = OptimizedCoroutinePool(max_concurrent=max_concurrent)
        self.stats = {
            'thread_tasks': 0,
            'process_tasks': 0,
            'coroutine_tasks': 0
        }

    async def execute_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """在线程池中执行"""
        self.stats['thread_tasks'] += 1
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)

    async def execute_process(self, func: Callable[..., T], *args, **kwargs) -> T:
        """在进程池中执行"""
        self.stats['process_tasks'] += 1
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_executor, func, *args, **kwargs)

    async def execute_coroutine(self, coro: Callable[[], Awaitable[T]]) -> T:
        """在协程池中执行"""
        self.stats['coroutine_tasks'] += 1
        return await self.coroutine_pool.submit(coro)

    def auto_execute(self, func: Callable[..., T], *args,
                    execution_type: str = "auto", **kwargs) -> Awaitable[T]:
        """自动选择执行方式"""
        if execution_type == "thread":
            return self.execute_thread(func, *args, **kwargs)
        elif execution_type == "process":
            return self.execute_process(func, *args, **kwargs)
        elif execution_type == "coroutine":
            async def _wrap():
                return await func(*args, **kwargs)
            return self.execute_coroutine(_wrap)
        else:
            # 自动选择：CPU密集型用进程，IO密集型用线程
            # 这里简化为根据函数名判断，实际应该有更智能的判断
            if 'cpu' in func.__name__.lower() or 'compute' in func.__name__.lower():
                return self.execute_process(func, *args, **kwargs)
            else:
                return self.execute_thread(func, *args, **kwargs)

    async def shutdown(self):
        """关闭执行器"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        await self.coroutine_pool.wait_all()

    def get_stats(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        return {
            **self.stats,
            'coroutine_pool_stats': self.coroutine_pool.get_stats()
        }


# 配置常量
ASYNC_CONFIG = {
    'task_queue': {
        'redis_url': 'redis://localhost:6379',
        'max_concurrent': 100,
        'batch_size': 50
    },
    'coroutine_pool': {
        'max_concurrent': 1000,
        'batch_size': 100
    },
    'hybrid_executor': {
        'max_workers': 20,
        'max_concurrent': 500
    },
    'timeouts': {
        'default_task_timeout': 300,  # 5分钟
        'queue_processing_interval': 0.1,
        'shutdown_timeout': 30
    }
}


class AsyncProcessingManager:
    """异步处理管理器"""

    def __init__(self):
        self.task_queue = AsyncTaskQueue()
        self.coroutine_pool = OptimizedCoroutinePool()
        self.hybrid_executor = HybridExecutor()
        self._initialized = False

    async def initialize(self):
        """初始化异步处理管理器"""
        if self._initialized:
            return

        await self.task_queue.initialize()
        self._initialized = True

        logger.info("异步处理管理器初始化完成")

    async def submit_data_processing_task(self, data: List[Dict[str, Any]],
                                        processor: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
                                        priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交数据处理任务"""
        async def process_batch():
            results = []
            for item in data:
                try:
                    result = await processor(item)
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理数据项失败: {e}")
                    results.append({'error': str(e), 'original': item})
            return results

        return await self.task_queue.submit_task(
            "data_processing",
            process_batch,
            priority=priority,
            timeout=600  # 10分钟超时
        )

    async def submit_market_data_update(self, symbols: List[str]) -> str:
        """提交市场数据更新任务"""
        async def update_market_data():
            results = {}
            for symbol in symbols:
                try:
                    # 这里应该调用实际的市场数据获取逻辑
                    data = await self._fetch_market_data(symbol)
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"获取市场数据失败 {symbol}: {e}")
                    results[symbol] = {'error': str(e)}

            return results

        return await self.task_queue.submit_task(
            "market_data_update",
            update_market_data,
            priority=TaskPriority.HIGH
        )

    async def _fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据（示例实现）"""
        # 这里应该连接实际的数据源
        await asyncio.sleep(0.1)  # 模拟网络延迟

        return {
            'symbol': symbol,
            'price': 150.0 + (hash(symbol) % 100),
            'volume': 1000000,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'task_queue': self.task_queue.get_stats(),
            'coroutine_pool': self.coroutine_pool.get_stats(),
            'hybrid_executor': self.hybrid_executor.get_stats()
        }

    async def shutdown(self):
        """关闭异步处理管理器"""
        await self.task_queue.shutdown()
        await self.hybrid_executor.shutdown()

        logger.info("异步处理管理器已关闭")
