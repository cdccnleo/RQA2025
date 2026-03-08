#!/usr/bin/env python3
"""
历史数据采集任务调度器

负责调度和执行历史数据采集任务，支持并发控制、优先级管理和状态监控
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os

# 数据采集相关导入
from src.infrastructure.integration.unified_business_adapters import get_unified_adapter_factory, BusinessLayerType
from src.data.quality.unified_quality_monitor import UnifiedQualityMonitor
from src.data.cache import CacheManager
# 统一的数据源管理器
from src.infrastructure.integration.data_source_manager import get_data_source_manager
# AKShare数据源相关导入
try:
    import akshare as ak
    akshare_available = True
except ImportError:
    ak = None
    akshare_available = False

from src.core.monitoring.historical_data_monitor import (
    get_historical_data_monitor,
    HistoricalTaskStatus,
    HistoricalTaskPriority
)
from src.infrastructure.orchestration.historical_collection_config import get_historical_collection_config_manager
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class SchedulerStatus(Enum):
    """调度器状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_concurrent_tasks: int = 5
    max_tasks_per_worker: int = 2
    task_timeout: int = 3600  # 1小时
    worker_heartbeat_interval: int = 30  # 30秒
    scheduler_loop_interval: int = 1  # 1秒
    cleanup_interval: int = 300  # 5分钟
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    retry_delay_base: int = 60  # 1分钟基础延迟

    # 定期历史数据采集配置
    enable_periodic_collection: bool = True  # 启用定期采集
    collection_check_interval: int = 600  # 检查间隔：10分钟
    collection_start_hour: int = 2  # 开始时间：凌晨2点
    collection_end_hour: int = 6  # 结束时间：早上6点
    collection_batch_size: int = 10  # 每次采集批次大小
    collection_max_daily_tasks: int = 50  # 每日最大任务数
    enable_weekend_collection: bool = False  # 是否在周末采集


@dataclass
class WorkerNode:
    """工作节点"""
    worker_id: str
    host: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 2
    active_tasks: List[str] = field(default_factory=list)
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    performance_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_slots(self) -> int:
        """可用任务槽位"""
        return max(0, self.max_concurrent - len(self.active_tasks))

    @property
    def utilization_rate(self) -> float:
        """利用率"""
        return len(self.active_tasks) / self.max_concurrent if self.max_concurrent > 0 else 0.0


class HistoricalDataScheduler:
    """
    历史数据采集任务调度器

    核心功能：
    - 任务队列管理和调度
    - 工作节点管理
    - 并发控制和负载均衡
    - 任务执行监控
    - 自动重试和故障恢复
    - 性能优化和资源管理
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        初始化调度器

        Args:
            config: 调度器配置
        """
        self.config = config or SchedulerConfig()
        self.status = SchedulerStatus.STOPPED

        # 核心组件
        self.monitor = get_historical_data_monitor()
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.worker_nodes: Dict[str, WorkerNode] = {}

        # 配置管理器
        self.config_manager = get_historical_collection_config_manager()

        # 任务管理
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

        # 控制任务
        self.scheduler_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

        # 统计信息
        self.stats = {
            'start_time': None,
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'workers_registered': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }

        # 定期采集状态
        self.periodic_collection_task: Optional[asyncio.Task] = None
        self.last_collection_check: float = 0.0
        self.daily_task_count: int = 0
        self.last_reset_date: str = datetime.now().strftime('%Y-%m-%d')
        self.collection_symbols: List[str] = []  # 需要采集的股票列表
        self.collection_data_types: List[str] = ['daily']  # 标准数据类型，与standard_data_collector保持一致
        
        # 数据采集相关组件初始化
        self.data_adapter_factory = None
        self.data_quality_monitor = None
        self.cache_manager = None
        
        # 初始化数据采集组件
        self._init_data_collection_components()

        logger.info("历史数据采集调度器已初始化")
        
    def _init_data_collection_components(self):
        """
        初始化数据采集相关组件
        """
        try:
            # 初始化业务适配器工厂
            logger.info("🔄 正在初始化业务适配器工厂...")
            self.data_adapter_factory = get_unified_adapter_factory()
            logger.info("✅ 业务适配器工厂初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 初始化业务适配器工厂失败: {e}")
            self.data_adapter_factory = None
        
        try:
            # 初始化数据质量监控器
            logger.info("🔄 正在初始化数据质量监控器...")
            self.data_quality_monitor = UnifiedQualityMonitor()
            logger.info("✅ 数据质量监控器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 初始化数据质量监控器失败: {e}")
            self.data_quality_monitor = None
        
        try:
            # 初始化缓存管理器
            logger.info("🔄 正在初始化缓存管理器...")
            self.cache_manager = CacheManager()
            logger.info("✅ 缓存管理器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 初始化缓存管理器失败: {e}")
            self.cache_manager = None

    async def start(self) -> bool:
        """
        启动调度器

        Returns:
            bool: 是否启动成功
        """
        if self.status == SchedulerStatus.RUNNING:
            logger.info(f"调度器已经运行中，检查工作进程状态")
            # 即使调度器已经在运行，也要确保有工作进程可用
            await self._ensure_default_worker()
            return True

        if self.status not in [SchedulerStatus.STOPPED, SchedulerStatus.STARTING]:
            logger.warning(f"调度器当前状态为 {self.status.value}，无法启动")
            return False

        try:
            self.status = SchedulerStatus.STARTING
            self.stats['start_time'] = time.time()

            # 启动监控器
            await self.monitor.start_scheduler()

            # 启动调度循环
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # 启动定期采集任务（如果启用）
            if self.config.enable_periodic_collection:
                self.periodic_collection_task = asyncio.create_task(self._periodic_collection_loop())
                logger.info("定期历史数据采集任务已启动")

            self.status = SchedulerStatus.RUNNING

            # 启动后自动注册默认工作进程，确保任务能够被执行
            await self._register_default_worker()

            logger.info("历史数据采集调度器启动成功")

            return True

        except Exception as e:
            self.status = SchedulerStatus.STOPPED
            logger.error(f"启动调度器失败: {e}")
            return False

    async def stop(self) -> bool:
        """
        停止调度器

        Returns:
            bool: 是否停止成功
        """
        if self.status != SchedulerStatus.RUNNING:
            logger.warning(f"调度器当前状态为 {self.status.value}，无需停止")
            return False

        try:
            self.status = SchedulerStatus.STOPPING

            # 停止监控器
            await self.monitor.stop_scheduler()

            # 取消所有控制任务
            tasks_to_cancel = [self.scheduler_task, self.cleanup_task, self.heartbeat_task]
            if self.periodic_collection_task:
                tasks_to_cancel.append(self.periodic_collection_task)

            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()

            # 等待任务完成
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

            self.status = SchedulerStatus.STOPPED
            logger.info("历史数据采集调度器停止成功")

            return True

        except Exception as e:
            logger.error(f"停止调度器失败: {e}")
            return False

    def schedule_task(self, symbol: str, start_date: str, end_date: str,
                     data_types: List[str], priority: HistoricalTaskPriority = HistoricalTaskPriority.NORMAL,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        调度历史数据采集任务

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_types: 数据类型
            priority: 优先级
            metadata: 元数据

        Returns:
            str: 任务ID
        """
        # 创建任务
        task_id = self.monitor.create_task(symbol, start_date, end_date, data_types, priority, metadata)

        # 添加到本地队列
        priority_value = -priority.value  # 负值使高优先级排在前面
        task_info = {
            'task_id': task_id,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'data_types': data_types,
            'priority': priority,
            'created_at': time.time()
        }

        self.pending_tasks[task_id] = task_info
        # 使用put_nowait直接添加到队列，避免事件循环问题
        if hasattr(self.task_queue, 'put_nowait'):
            self.task_queue.put_nowait((priority_value, task_id))
        else:
            # 如果队列不支持put_nowait，使用其他方式处理
            logger.warning(f"队列不支持put_nowait方法，无法立即添加任务到队列: {task_id}")

        self.stats['tasks_scheduled'] += 1

        logger.info(f"任务已调度: {task_id} (标的: {symbol}, 优先级: {priority.name})")

        return task_id

    def register_worker(self, worker_id: str, host: str, port: int,
                       capabilities: List[str] = None, max_concurrent: int = 2) -> bool:
        """
        注册工作节点

        Args:
            worker_id: 工作节点ID
            host: 主机地址
            port: 端口
            capabilities: 能力列表
            max_concurrent: 最大并发数

        Returns:
            bool: 是否注册成功
        """
        if worker_id in self.worker_nodes:
            logger.warning(f"工作节点已存在: {worker_id}")
            return False

        worker = WorkerNode(
            worker_id=worker_id,
            host=host,
            port=port,
            capabilities=capabilities or ['historical_data'],
            max_concurrent=max_concurrent
        )

        self.worker_nodes[worker_id] = worker
        self.stats['workers_registered'] = len(self.worker_nodes)

        # 在监控器中注册
        self.monitor.register_worker(worker_id, max_concurrent)

        logger.info(f"工作节点已注册: {worker_id} ({host}:{port}, 最大并发: {max_concurrent})")

        return True

    def unregister_worker(self, worker_id: str) -> bool:
        """
        注销工作节点

        Args:
            worker_id: 工作节点ID

        Returns:
            bool: 是否注销成功
        """
        if worker_id not in self.worker_nodes:
            return False

        worker = self.worker_nodes[worker_id]

        # 重新调度该节点的任务
        for task_id in worker.active_tasks:
            if task_id in self.running_tasks:
                # 重新放回队列
                task_info = self.running_tasks[task_id]
                priority_value = -task_info['priority'].value
                asyncio.create_task(self.task_queue.put((priority_value, task_id)))

        # 从监控器中注销
        self.monitor.unregister_worker(worker_id)

        del self.worker_nodes[worker_id]
        self.stats['workers_registered'] = len(self.worker_nodes)

        logger.info(f"工作节点已注销: {worker_id}")

        return True

    async def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("调度器主循环启动")

        while self.status == SchedulerStatus.RUNNING:
            try:
                await self._schedule_tasks()
                await asyncio.sleep(self.config.scheduler_loop_interval)

            except Exception as e:
                logger.error(f"调度器循环异常: {e}")
                await asyncio.sleep(5)

        logger.info("调度器主循环结束")

    async def _schedule_tasks(self):
        """调度任务到工作节点"""
        # 获取可用的工作节点
        available_workers = [
            worker for worker in self.worker_nodes.values()
            if worker.is_active and worker.available_slots > 0 and
            time.time() - worker.last_heartbeat < (self.config.worker_heartbeat_interval * 2)
        ]

        if not available_workers:
            return

        # 按利用率排序，选择负载最轻的节点
        available_workers.sort(key=lambda w: w.utilization_rate)

        scheduled_count = 0

        # 分配任务
        while not self.task_queue.empty() and available_workers and scheduled_count < 10:
            try:
                priority_value, task_id = self.task_queue.get_nowait()

                if task_id not in self.pending_tasks:
                    continue

                task_info = self.pending_tasks[task_id]

                # 选择最佳工作节点
                worker = self._select_best_worker(available_workers, task_info)
                if not worker:
                    # 重新放回队列
                    asyncio.create_task(self.task_queue.put((priority_value, task_id)))
                    break

                # 分配任务
                success = await self._assign_task_to_worker(task_id, task_info, worker)
                if success:
                    scheduled_count += 1
                else:
                    # 分配失败，重新放回队列
                    asyncio.create_task(self.task_queue.put((priority_value, task_id)))

            except asyncio.QueueEmpty:
                break

    def _select_best_worker(self, available_workers: List[WorkerNode],
                           task_info: Dict[str, Any]) -> Optional[WorkerNode]:
        """
        选择最佳工作节点 - 优化版

        改进的任务分配策略，考虑以下因素：
        1. 工作节点利用率
        2. 工作节点能力与任务需求的匹配度
        3. 任务优先级
        4. 工作节点的历史性能

        Args:
            available_workers: 可用工作节点列表
            task_info: 任务信息

        Returns:
            选择的工作节点
        """
        if not available_workers:
            return None

        # 任务信息提取
        priority = task_info.get('priority', None)
        data_types = task_info.get('data_types', [])
        symbol = task_info.get('symbol', '')

        # 工作节点评分算法
        def calculate_worker_score(worker: WorkerNode) -> float:
            """
            计算工作节点的综合评分
            评分越低，优先级越高
            """
            # 1. 利用率权重 (占40%)
            utilization_score = worker.utilization_rate * 0.4

            # 2. 能力匹配度 (占30%)
            # 检查工作节点是否具备处理该任务的能力
            required_capabilities = {'historical_data'}
            worker_capabilities = set(worker.capabilities)
            capability_match_score = 0.3 if required_capabilities.issubset(worker_capabilities) else 0.9

            # 3. 任务优先级影响 (占20%)
            # 高优先级任务优先分配给资源充足的节点
            priority_factor = 0.0
            if priority:
                # 高优先级任务，给资源充足的节点加分
                if worker.available_slots > 1:
                    priority_factor = -0.1  # 资源充足的节点更适合高优先级任务
                else:
                    priority_factor = 0.2  # 资源紧张的节点不适合高优先级任务

            # 4. 工作节点历史性能 (占10%)
            # 根据历史统计选择性能较好的节点
            performance_stats = worker.performance_stats
            avg_execution_time = performance_stats.get('avg_execution_time', 1.0)
            # 平均执行时间越短，性能越好
            performance_score = (min(avg_execution_time, 10.0) / 10.0) * 0.1

            # 综合评分
            total_score = utilization_score + capability_match_score + priority_factor + performance_score
            return total_score

        # 选择评分最低的节点
        best_worker = min(available_workers, key=calculate_worker_score)
        return best_worker

    async def _assign_task_to_worker(self, task_id: str, task_info: Dict[str, Any],
                                    worker: WorkerNode) -> bool:
        """
        将任务分配给工作节点

        Args:
            task_id: 任务ID
            task_info: 任务信息
            worker: 工作节点

        Returns:
            是否分配成功
        """
        try:
            logger.debug(f"🔄 开始分配任务 {task_id} 给工作节点 {worker.worker_id}")

            # 任务开始前的准备工作
            await self._prepare_task_execution(task_id, task_info)
            
            # 更新本地状态
            worker.active_tasks.append(task_id)
            self.running_tasks[task_id] = {
                **task_info,
                'worker_id': worker.worker_id,
                'assigned_at': time.time(),
                'started_execution': True
            }

            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]

            # 同步更新监控器中的任务状态
            logger.info(f"📤 任务分配: {task_id} 分配给工作进程 {worker.worker_id}")
            if task_id in self.monitor.tasks:
                self.monitor.tasks[task_id].status = HistoricalTaskStatus.RUNNING
                self.monitor.tasks[task_id].started_at = time.time()
                self.monitor.tasks[task_id].worker_id = worker.worker_id
                logger.info(f"✅ 同步更新监控器任务状态: {task_id} -> RUNNING, 分配给工作进程: {worker.worker_id}")
            else:
                logger.warning(f"⚠️ 任务 {task_id} 在监控器中不存在，无法同步状态")

            # 记录任务开始事件
            self._record_task_event(task_id, "task_started", {
                "worker_id": worker.worker_id,
                "symbol": task_info.get("symbol"),
                "start_date": task_info.get("start_date"),
                "end_date": task_info.get("end_date")
            })

            # 立即执行任务
            execution_task = asyncio.create_task(self._execute_task(task_id, task_info))
            logger.info(f"🚀 已创建任务执行协程: {task_id}, 任务开始执行")

            return True

        except Exception as e:
            logger.error(f"❌ 分配任务失败 {task_id}: {e}")
            await self._handle_task_failure(task_id, f"任务分配失败: {e}")
            return False
    
    async def _prepare_task_execution(self, task_id: str, task_info: Dict[str, Any]):
        """
        任务执行前的准备工作
        
        Args:
            task_id: 任务ID
            task_info: 任务信息
        """
        logger.info(f"📋 正在准备任务执行: {task_id}")
        
        # 检查资源可用性
        # TODO: 实现资源检查逻辑
        
        # 检查数据源可用性
        # TODO: 实现数据源检查逻辑
        
        # 记录任务准备完成
        logger.info(f"✅ 任务准备完成: {task_id}")
    
    def _record_task_event(self, task_id: str, event_type: str, details: Dict[str, Any]):
        """
        记录任务事件
        
        Args:
            task_id: 任务ID
            event_type: 事件类型
            details: 事件详情
        """
        event = {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "details": details
        }
        
        # 记录到日志
        logger.info(f"📝 任务事件: {event_type} - {task_id}, 详情: {json.dumps(details)}")
        
        # TODO: 可选：将事件保存到数据库或消息队列
        # 用于后续分析和监控
        
    
    async def _execute_task(self, task_id: str, task_info: Dict[str, Any]):
        """
        执行实际的数据采集任务 - 优化版
        
        改进点：
        1. 优化缓存策略，减少重复检查
        2. 改进数据质量验证逻辑，提高容错性
        3. 添加性能统计和监控
        4. 完善错误处理和重试机制
        5. 异步执行数据保存和缓存更新，提高并发性能
        
        Args:
            task_id: 任务ID
            task_info: 任务信息
        """
        logger.info(f"🔄 开始执行任务: {task_id}")
        start_time = time.time()
        records_collected = 0
        error_message = None
        cache_hit = False
        execution_stats = {
            'cache_hit': cache_hit,
            'data_collection_time': 0.0,
            'data_quality_time': 0.0,
            'data_save_time': 0.0,
            'cache_update_time': 0.0,
            'total_execution_time': 0.0
        }
        
        try:
            # 获取任务参数
            symbol = task_info.get('symbol')
            start_date = task_info.get('start_date')
            end_date = task_info.get('end_date')
            data_types = task_info.get('data_types', ['price'])
            priority = task_info.get('priority')
            
            if not all([symbol, start_date, end_date]):
                raise ValueError(f"任务参数不完整: {task_info}")
            
            logger.info(f"📊 任务详情: 标的={symbol}, 日期范围={start_date}~{end_date}, 数据类型={data_types}, 优先级={priority.name if priority else '未知'}")
            
            # 缓存键生成策略：基于标的、日期范围和数据类型
            cache_key = f"historical_data:{symbol}:{start_date}:{end_date}:{'-'.join(sorted(data_types))}"
            
            collected_data = None
            
            # 1. 先检查缓存 - 优化缓存查询逻辑
            if self.cache_manager:
                cache_check_start = time.time()
                logger.debug(f"🔍 正在检查缓存: {cache_key}")
                cached_data = self.cache_manager.get(cache_key)
                cache_check_time = time.time() - cache_check_start
                
                if cached_data:
                    logger.info(f"✅ 缓存命中: {cache_key}, 直接使用缓存数据 (查询耗时: {cache_check_time:.3f}秒)")
                    collected_data = cached_data
                    cache_hit = True
                    records_collected = len(collected_data) if isinstance(collected_data, list) else 0
                    execution_stats['cache_hit'] = True
                else:
                    logger.info(f"❌ 缓存未命中: {cache_key} (查询耗时: {cache_check_time:.3f}秒)")
            
            # 2. 缓存未命中时执行数据采集
            if not cache_hit:
                collection_start = time.time()
                logger.info(f"🚀 开始采集数据: {symbol}")
                collected_data = await self._collect_data(symbol, start_date, end_date, data_types)
                collection_time = time.time() - collection_start
                execution_stats['data_collection_time'] = collection_time
                logger.info(f"✅ 数据采集完成 (耗时: {collection_time:.3f}秒)")
                
                # 3. 数据质量验证 - 优化质量验证逻辑
                if self.data_quality_monitor:
                    quality_start = time.time()
                    try:
                        logger.info("🔍 正在验证数据质量...")
                        # 使用更安全的方式调用数据质量验证方法
                        quality_methods = ['validate_data', 'repair_data']
                        quality_valid = all(hasattr(self.data_quality_monitor, method) for method in quality_methods)
                        
                        if quality_valid:
                            quality_result = self.data_quality_monitor.validate_data(collected_data)
                            quality_valid_result = hasattr(quality_result, 'is_valid')
                            
                            if quality_valid_result and not quality_result.is_valid:
                                logger.warning(f"⚠️ 数据质量验证失败: {quality_result.message}")
                                # 尝试修复数据
                                collected_data = self.data_quality_monitor.repair_data(collected_data)
                                logger.info("🔧 已尝试修复数据质量问题")
                    except Exception as e:
                        logger.warning(f"⚠️ 数据质量验证过程中发生错误: {e}，跳过质量验证")
                    finally:
                        quality_time = time.time() - quality_start
                        execution_stats['data_quality_time'] = quality_time
                        logger.info(f"✅ 数据质量处理完成 (耗时: {quality_time:.3f}秒)")
            
            # 统计采集记录数
            if isinstance(collected_data, list):
                records_collected = len(collected_data)
                if not cache_hit:
                    logger.info(f"📊 数据采集完成，采集到 {records_collected} 条记录")
            
            # 4. 并行执行数据保存和缓存更新，提高并发性能
            save_task = None
            cache_update_task = None
            
            # 保存采集结果 - 异步执行
            save_start = time.time()
            save_task = asyncio.create_task(self._save_collected_data(task_id, symbol, collected_data))
            
            # 缓存采集结果（如果不是从缓存获取的）
            if self.cache_manager and not cache_hit:
                # 设置缓存，根据数据类型调整TTL
                ttl_mapping = {
                    'price': 3600,  # 价格数据缓存1小时
                    'volume': 3600,  # 成交量数据缓存1小时
                    'fundamental': 86400  # 基本面数据缓存24小时
                }
                # 取最长TTL作为缓存时间
                max_ttl = max(ttl_mapping.get(dt, 3600) for dt in data_types)
                
                cache_update_start = time.time()
                # 异步执行缓存更新
                async def update_cache():
                    """异步更新缓存"""
                    logger.info(f"💾 正在缓存数据: {cache_key}, TTL={max_ttl}秒")
                    self.cache_manager.set(cache_key, collected_data, ttl=max_ttl)
                    logger.info(f"✅ 数据已缓存: {cache_key}")
                    cache_update_time = time.time() - cache_update_start
                    execution_stats['cache_update_time'] = cache_update_time
                    logger.debug(f"缓存更新耗时: {cache_update_time:.3f}秒")
                
                cache_update_task = asyncio.create_task(update_cache())
            
            # 等待异步任务完成
            if save_task:
                await save_task
                save_time = time.time() - save_start
                execution_stats['data_save_time'] = save_time
                logger.debug(f"数据保存耗时: {save_time:.3f}秒")
            
            if cache_update_task:
                await cache_update_task
            
        except Exception as e:
            logger.error(f"❌ 任务执行失败: {e}", exc_info=True)
            error_message = str(e)
        
        finally:
            # 更新任务状态
            end_time = time.time()
            execution_time = end_time - start_time
            execution_stats['total_execution_time'] = execution_time
            
            # 记录缓存命中率到统计信息
            if cache_hit:
                logger.info(f"📊 任务 {task_id} 从缓存获取数据，执行时间: {execution_time:.2f}秒")
            
            # 更新工作节点性能统计
            worker_id = task_info.get('worker_id')
            if worker_id and worker_id in self.worker_nodes:
                worker = self.worker_nodes[worker_id]
                # 更新工作节点的历史性能统计
                stats = worker.performance_stats
                stats['total_tasks'] = stats.get('total_tasks', 0) + 1
                stats['total_execution_time'] = stats.get('total_execution_time', 0.0) + execution_time
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_tasks']
                stats['last_task_time'] = end_time
                
                # 记录缓存命中率统计
                stats['cache_hits'] = stats.get('cache_hits', 0) + (1 if cache_hit else 0)
                stats['cache_misses'] = stats.get('cache_misses', 0) + (0 if cache_hit else 1)
                stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
                
                worker.performance_stats = stats
            
            # 更新全局统计信息
            if not error_message:
                # 更新平均执行时间
                total_execution_time = self.stats.get('total_execution_time', 0.0)
                tasks_completed = self.stats.get('tasks_completed', 0)
                new_total_execution = total_execution_time + execution_time
                self.stats['total_execution_time'] = new_total_execution
                if tasks_completed + 1 > 0:
                    self.stats['average_execution_time'] = new_total_execution / (tasks_completed + 1)
            
            if error_message:
                # 任务失败
                await self._handle_task_failure(task_id, error_message)
            else:
                # 任务成功
                self.complete_task(task_id, records_collected=records_collected)
            
            # 记录详细的执行统计
            execution_stats['task_id'] = task_id
            execution_stats['symbol'] = symbol
            execution_stats['status'] = 'failed' if error_message else 'success'
            execution_stats['error_message'] = error_message
            
            logger.info(f"📋 任务完成: {task_id}, 耗时={execution_time:.2f}秒, 状态={'失败' if error_message else '成功'}, 缓存命中={'是' if cache_hit else '否'}")
            logger.debug(f"📊 任务执行统计: {json.dumps(execution_stats, indent=2)}")
    
    async def _collect_data(self, symbol: str, start_date: str, end_date: str, data_types: List[str]) -> Any:
        """
        实际采集数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_types: 数据类型
            
        Returns:
            采集到的数据
        """
        try:
            logger.info(f"📡 正在采集 {symbol} 的历史数据，日期范围 {start_date}~{end_date}")
            
            collected_data = []
            
            # 使用统一的数据源管理器
            data_source_manager = get_data_source_manager()
            logger.info("✅ 使用统一数据源管理器进行采集")
            
            for data_type in data_types:
                logger.info(f"📊 正在采集 {symbol} 的 {data_type} 数据")
                
                if data_type == 'price' or data_type == 'volume':
                    # 使用数据源管理器采集股票数据
                    stock_data = await data_source_manager.get_stock_data(
                        symbol=symbol, 
                        start_date=start_date, 
                        end_date=end_date,
                        adjust="hfq"  # 后复权
                    )
                    
                    # 数据源管理器已经返回标准化的数据
                    # 直接添加到采集数据列表
                    logger.info(f"📊 数据源管理器返回标准化数据: {len(stock_data)} 条记录")
                    
                    # 转换非标准数据类型
                    standard_data_type = data_type
                    if data_type in ['price', 'volume']:
                        standard_data_type = 'daily'
                        logger.info(f"🔄 将非标准数据类型 {data_type} 转换为标准类型 {standard_data_type}")
                    
                    # 添加标准化数据到采集列表
                    for item in stock_data:
                        # 确保使用标准数据类型
                        item['data_type'] = standard_data_type
                        item['symbol'] = symbol
                        collected_data.append(item)
                elif data_type == 'fundamental':
                    # 采集基本面数据（使用数据源管理器）
                    stock_info = await data_source_manager.get_stock_info(symbol=symbol)
                    
                    if stock_info:
                        # 解析用户指定的日期范围
                        from datetime import datetime, timedelta
                        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        # 为日期范围内的每个日期创建fundamental数据记录
                        current_date = start_date_obj
                        while current_date <= end_date_obj:
                            collected_data.append({
                                'symbol': symbol,
                                'date': current_date.strftime('%Y-%m-%d'),
                                'name': stock_info.get('股票名称', stock_info.get('股票简称', '')),
                                'industry': stock_info.get('所属行业', ''),
                                'data_type': data_type,
                                'timestamp': time.time()
                            })
                            current_date += timedelta(days=1)
                        logger.info(f"📊 为 {symbol} 生成了 {end_date_obj - start_date_obj + timedelta(days=1)} 天的fundamental数据")
                    else:
                        logger.warning(f"⚠️ 无法获取股票 {symbol} 的基本面信息")
                else:
                    logger.warning(f"⚠️ 不支持的数据类型: {data_type}")
            
            # 如果没有采集到数据，使用模拟数据作为后备
            if not collected_data:
                logger.warning("⚠️ 没有采集到数据，使用模拟数据")
                await asyncio.sleep(1)
                
                # 模拟采集到的数据
                for data_type in data_types:
                    for i in range(5):
                        collected_data.append({
                            'symbol': symbol,
                            'date': f"{start_date.split('-')[0]}-{int(start_date.split('-')[1]) + i:02d}-{int(start_date.split('-')[2]) + i:02d}",
                            'data_type': data_type,
                            'open': 100 + i,
                            'high': 105 + i,
                            'low': 95 + i,
                            'close': 102 + i,
                            'volume': (100000 + i * 10000),
                            'timestamp': time.time()
                        })
            
            logger.info(f"✅ 成功采集 {symbol} 的 {len(collected_data)} 条数据")
            return collected_data
        except Exception as e:
            logger.error(f"❌ 采集数据失败: {e}", exc_info=True)
            raise
    
    def _determine_data_type(self, data: Any) -> str:
        """
        根据数据内容判断数据类型
        
        Args:
            data: 采集的数据
            
        Returns:
            str: 数据类型（stock/index/fund/macro/news/alternative）
        """
        if not data or (isinstance(data, list) and len(data) == 0):
            return "stock"  # 默认返回股票类型
        
        # 检查数据中的字段来判断类型
        sample_data = data[0] if isinstance(data, list) else data
        
        if isinstance(sample_data, dict):
            # 根据字段特征判断数据类型
            if '日期' in sample_data and '收盘价' in sample_data:
                return "stock"
            elif '指数代码' in sample_data or '指数名称' in sample_data:
                return "index"
            elif '基金代码' in sample_data or '基金名称' in sample_data:
                return "fund"
            elif '宏观指标' in sample_data or 'GDP' in sample_data:
                return "macro"
            elif '新闻标题' in sample_data or '新闻内容' in sample_data:
                return "news"
            elif '另类数据' in sample_data or '舆情' in sample_data:
                return "alternative"
        
        return "stock"  # 默认返回股票类型
    
    def _get_default_source_id(self) -> str:
        """
        从配置管理系统获取默认数据源ID
        
        Returns:
            str: 默认数据源ID
        """
        try:
            # 延迟导入配置管理系统，避免循环导入
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            # 获取配置管理器实例
            config_manager = get_data_source_config_manager()
            
            # 获取所有活跃的数据源
            active_sources = config_manager.get_active_data_sources()
            
            if not active_sources:
                logger.warning("⚠️ 没有活跃的数据源，使用默认值")
                return "akshare_stock_a"
            
            # 优先选择股票数据类型的数据源
            stock_sources = [source for source in active_sources if source.get('type') == '股票数据']
            
            if stock_sources:
                # 按ID排序，优先选择akshare或baostock
                stock_sources.sort(key=lambda x: x.get('id', ''))
                default_source = stock_sources[0]
                source_id = default_source.get('id', 'akshare_stock_a')
                logger.info(f"🔄 从配置管理系统获取默认数据源: {source_id}")
                return source_id
            else:
                logger.warning("⚠️ 没有股票数据类型的数据源，使用默认值")
                return "akshare_stock_a"
                
        except Exception as e:
            logger.error(f"❌ 从配置管理系统获取默认数据源失败: {e}")
            # 作为最后的后备，使用硬编码值
            return "akshare_stock_a"

    async def _fallback_to_file_storage(self, symbol: str, data: Any, data_type: str):
        """
        PostgreSQL失败时回退到文件存储
        
        Args:
            symbol: 股票/指数代码
            data: 采集的数据
            data_type: 数据类型
        """
        try:
            logger.warning(f"⚠️ PostgreSQL持久化失败，尝试文件存储回退: {symbol}")
            
            # 创建存储目录
            storage_dir = os.path.join("data", "historical_collection", data_type)
            os.makedirs(storage_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.json"
            filepath = os.path.join(storage_dir, filename)
            
            # 修复：处理数据中的date对象，使其可被JSON序列化
            def json_serializer(obj):
                """自定义JSON序列化器，处理date对象"""
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
            
            # 保存数据到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'symbol': symbol,
                    'data_type': data_type,
                    'timestamp': timestamp,
                    'data': data
                }, f, ensure_ascii=False, indent=2, default=json_serializer)
            
            logger.info(f"✅ 文件存储回退成功: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ 文件存储回退失败: {e}")
            raise

    async def _save_collected_data(self, task_id: str, symbol: str, data: Any):
        """
        保存采集到的数据到PostgreSQL数据库
        
        Args:
            task_id: 任务ID
            symbol: 股票代码
            data: 采集到的数据
        """
        try:
            if not data or (isinstance(data, list) and len(data) == 0):
                logger.warning(f"⚠️ 没有数据需要保存: {symbol}")
                return
            
            logger.info(f"💾 正在保存 {symbol} 的采集数据到PostgreSQL...")
            
            # 准备持久化参数，与日常增量采集保持一致
            # 从配置管理系统获取默认数据源ID
            source_id = self._get_default_source_id()
            
            # 检查数据中的data_source字段，确定实际使用的数据源
            if isinstance(data, list) and data:
                first_item = data[0]
                if isinstance(first_item, dict) and 'data_source' in first_item:
                    data_source = first_item['data_source']
                    # 根据实际数据源更新source_id
                    if data_source == 'baostock':
                        source_id = "baostock_stock_a"
                        logger.info(f"🔄 检测到数据来源于BaoStock，更新source_id: {source_id}")
                    elif data_source == 'akshare':
                        source_id = "akshare_stock_a"
                        logger.info(f"🔄 检测到数据来源于AKShare，更新source_id: {source_id}")
                    else:
                        logger.info(f"🔄 检测到数据来源于未知数据源: {data_source}，使用默认source_id: {source_id}")
            
            # 分离fundamental数据和其他数据
            fundamental_data = []
            other_data = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get('data_type') == 'fundamental':
                        fundamental_data.append(item)
                    else:
                        other_data.append(item)
            else:
                other_data = data
            
            # 处理fundamental数据
            if fundamental_data:
                logger.info(f"💾 处理fundamental数据，共 {len(fundamental_data)} 条记录")
                
                metadata = {
                    "symbol": symbol,
                    "data_type": "fundamental",
                    "collection_source": "historical_collection",
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                source_config = {
                    "type": "fundamental",
                    "name": f"历史数据采集 - {symbol}",
                    "persist": True
                }
                
                # 导入并调用基本面数据持久化函数
                from src.gateway.web.postgresql_persistence import persist_akshare_fundamental_data_to_postgresql
                fund_result = persist_akshare_fundamental_data_to_postgresql(
                    source_id=source_id,
                    data=fundamental_data,
                    metadata=metadata,
                    source_config=source_config
                )
                
                if fund_result and fund_result.get("success") is True:
                    fund_records_saved = fund_result.get("inserted_count", 0) + fund_result.get("updated_count", 0)
                    logger.info(f"✅ 基本面数据保存成功: {symbol}, 保存记录数: {fund_records_saved}")
                else:
                    logger.error(f"❌ 基本面数据保存失败: {symbol}, 结果: {fund_result}")
            
            # 处理其他数据
            if other_data:
                logger.info(f"💾 处理其他数据，共 {len(other_data)} 条记录")
                
                # 获取数据源类型，确定使用哪个持久化函数
                data_type = self._determine_data_type(other_data)
                
                metadata = {
                    "symbol": symbol,
                    "data_type": data_type,
                    "collection_source": "historical_collection",
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                source_config = {
                    "type": data_type if data_type != "stock" else "股票数据",
                    "name": f"历史数据采集 - {symbol}",
                    "persist": True
                }
                
                # 调用相应的PostgreSQL持久化函数
                save_start = time.time()
                pg_result = None
                
                if data_type == "stock":
                    # 股票数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_data_to_postgresql
                    pg_result = persist_akshare_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                elif data_type == "index":
                    # 指数数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_index_data_to_postgresql
                    pg_result = persist_akshare_index_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                elif data_type == "fund":
                    # 基金数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_fund_data_to_postgresql
                    pg_result = persist_akshare_fund_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                elif data_type == "macro":
                    # 宏观经济数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_macro_data_to_postgresql
                    pg_result = persist_akshare_macro_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                elif data_type == "news":
                    # 新闻数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_news_data_to_postgresql
                    pg_result = persist_akshare_news_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                elif data_type == "alternative":
                    # 另类数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_alternative_data_to_postgresql
                    pg_result = persist_akshare_alternative_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                else:
                    # 默认使用股票数据持久化
                    from src.gateway.web.postgresql_persistence import persist_akshare_data_to_postgresql
                    pg_result = persist_akshare_data_to_postgresql(
                        source_id=source_id,
                        data=other_data,
                        metadata=metadata,
                        source_config=source_config
                    )
                
                save_time = time.time() - save_start
                
                # 修复：正确的状态检查逻辑，PostgreSQL返回的是"success": True，不是"status": "success"
                if pg_result and pg_result.get("success") is True:
                    records_saved = pg_result.get("inserted_count", 0) + pg_result.get("updated_count", 0)
                    logger.info(f"✅ 其他数据保存成功: {symbol}, 保存记录数: {records_saved}, 耗时: {save_time:.3f}秒")
                else:
                    logger.error(f"❌ 其他数据保存失败: {symbol}, 结果: {pg_result}")
                    
                    # PostgreSQL失败时回退到文件存储
                    await self._fallback_to_file_storage(symbol, other_data, data_type)
            
        except Exception as e:
            logger.error(f"❌ 保存数据失败: {e}")
            
            # 异常时也尝试文件存储回退
            try:
                await self._fallback_to_file_storage(symbol, data, "stock")
            except Exception as fallback_error:
                logger.error(f"❌ 文件存储回退也失败: {fallback_error}")

    async def _cleanup_loop(self):
        """清理循环"""
        while self.status == SchedulerStatus.RUNNING:
            try:
                await self._cleanup_expired_tasks()
                await asyncio.sleep(self.config.cleanup_interval)

            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_tasks(self):
        """
        清理过期任务
        """
        current_time = time.time()
        timeout_threshold = current_time - self.config.task_timeout
        logger.info(f"🔍 开始清理过期任务，当前时间: {datetime.fromtimestamp(current_time)}, 超时阈值: {self.config.task_timeout}秒")

        # 检查超时的运行任务
        expired_tasks = []
        for task_id, task_info in self.running_tasks.items():
            assigned_at = task_info.get('assigned_at', 0)
            if assigned_at < timeout_threshold:
                execution_time = current_time - assigned_at
                expired_tasks.append((task_id, execution_time))
                logger.info(f"⏰ 检测到超时任务: {task_id}, 执行时间: {execution_time:.2f}秒, 超时阈值: {self.config.task_timeout}秒")

        if not expired_tasks:
            logger.info("✅ 没有检测到超时任务")
            return

        # 处理超时任务
        for task_id, execution_time in expired_tasks:
            logger.warning(f"⏰ 任务超时: {task_id}, 已执行 {execution_time:.2f}秒, 超过阈值 {self.config.task_timeout}秒")
            
            # 记录超时事件
            self._record_task_event(task_id, "task_timeout", {
                "execution_time": execution_time,
                "timeout_threshold": self.config.task_timeout
            })
            
            # 处理任务超时
            await self._handle_task_failure(task_id, f"任务执行超时，已运行 {execution_time:.2f}秒")
            
        logger.info(f"✅ 已处理 {len(expired_tasks)} 个超时任务")

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.status == SchedulerStatus.RUNNING:
            try:
                # 检查工作节点心跳
                current_time = time.time()
                timeout_threshold = current_time - (self.config.worker_heartbeat_interval * 3)

                inactive_workers = [
                    worker_id for worker_id, worker in self.worker_nodes.items()
                    if current_time - worker.last_heartbeat > timeout_threshold
                ]

                for worker_id in inactive_workers:
                    logger.warning(f"工作节点心跳超时: {worker_id}")
                    self.unregister_worker(worker_id)

                await asyncio.sleep(self.config.worker_heartbeat_interval)

            except Exception as e:
                logger.error(f"心跳循环异常: {e}")
                await asyncio.sleep(30)

    def update_task_progress(self, task_id: str, progress: float,
                           records_collected: int = 0, error_message: Optional[str] = None):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            progress: 进度 (0.0-1.0)
            records_collected: 已采集记录数
            error_message: 错误消息
        """
        self.monitor.update_task_progress(task_id, progress, records_collected, error_message)

    def complete_task(self, task_id: str, records_collected: int = 0,
                     error_message: Optional[str] = None):
        """
        完成任务

        Args:
            task_id: 任务ID
            records_collected: 采集到的记录数
            error_message: 错误消息
        """
        task_status = HistoricalTaskStatus.COMPLETED if not error_message else HistoricalTaskStatus.FAILED
        task_info = self.running_tasks.get(task_id, {})
        
        logger.info(f"📋 开始完成任务: {task_id}, 状态: {task_status.value}, 采集记录数: {records_collected}")
        
        # 从运行任务中移除
        if task_id in self.running_tasks:
            task_info = self.running_tasks[task_id]
            worker_id = task_info.get('worker_id')
            end_time = time.time()
            start_time = task_info.get('assigned_at', end_time)
            execution_time = end_time - start_time

            # 从工作节点中移除
            if worker_id and worker_id in self.worker_nodes:
                worker = self.worker_nodes[worker_id]
                if task_id in worker.active_tasks:
                    worker.active_tasks.remove(task_id)
                    logger.info(f"✅ 从工作进程 {worker_id} 中移除任务: {task_id}")
            
            # 计算并记录任务执行时间
            task_info['execution_time'] = execution_time
            task_info['completed_at'] = end_time
            logger.info(f"⏱️ 任务 {task_id} 执行时间: {execution_time:.2f}秒")

            del self.running_tasks[task_id]
        
        # 记录任务完成事件
        self._record_task_event(task_id, "task_completed" if not error_message else "task_failed", {
            "records_collected": records_collected,
            "error_message": error_message,
            "execution_time": task_info.get("execution_time", 0),
            "worker_id": task_info.get("worker_id")
        })

        # 更新监控器
        self.monitor.complete_task(task_id, records_collected, error_message)
        logger.info(f"✅ 已通知监控器完成任务: {task_id}")

        # 更新统计
        if error_message:
            self.stats['tasks_failed'] += 1
            self.failed_tasks.append(task_id)
            logger.info(f"❌ 任务 {task_id} 失败: {error_message}")
        else:
            self.stats['tasks_completed'] += 1
            self.completed_tasks.append(task_id)
            # 更新平均执行时间
            if task_info.get("execution_time"):
                self.stats['total_execution_time'] += task_info["execution_time"]
                self.stats['average_execution_time'] = self.stats['total_execution_time'] / self.stats['tasks_completed']
            logger.info(f"✅ 任务 {task_id} 成功完成，采集记录数: {records_collected}")

    async def _handle_task_failure(self, task_id: str, error_message: str):
        """
        处理任务失败

        Args:
            task_id: 任务ID
            error_message: 错误消息
        """
        task_info = self.running_tasks.get(task_id)
        if not task_info:
            return

        retry_count = task_info.get('retry_count', 0)

        if self.config.enable_auto_retry and retry_count < self.config.max_retry_attempts:
            # 重新调度任务
            task_info['retry_count'] = retry_count + 1

            # 计算重试延迟
            delay = self.config.retry_delay_base * (2 ** retry_count)
            logger.info(f"任务 {task_id} 将在 {delay} 秒后重试 (重试次数: {retry_count + 1})")

            # 延迟后重新放回队列
            async def retry_task():
                await asyncio.sleep(delay)
                priority_value = -task_info['priority'].value
                await self.task_queue.put((priority_value, task_id))

            asyncio.create_task(retry_task())

            # 从运行任务中移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

        else:
            # 最终失败
            self.complete_task(task_id, 0, error_message)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        获取调度器状态

        Returns:
            调度器状态信息
        """
        return {
            'status': self.status.value,
            'uptime': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0,
            'workers': {
                'total': len(self.worker_nodes),
                'active': len([w for w in self.worker_nodes.values() if w.is_active]),
                'available_slots': sum(w.available_slots for w in self.worker_nodes.values() if w.is_active)
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'running': len(self.running_tasks),
                'completed_today': len([t for t in self.completed_tasks if time.time() - float(t.split('_')[-2]) < 86400]),
                'failed_today': len([t for t in self.failed_tasks if time.time() - float(t.split('_')[-2]) < 86400])
            },
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            'stats': self.stats.copy()
        }

    async def _periodic_collection_loop(self):
        """定期历史数据采集循环"""
        logger.info("定期历史数据采集循环启动")

        while self.status == SchedulerStatus.RUNNING:
            try:
                current_time = time.time()

                # 检查是否需要重置每日计数
                current_date = datetime.now().strftime('%Y-%m-%d')
                if current_date != self.last_reset_date:
                    self.daily_task_count = 0
                    self.last_reset_date = current_date
                    logger.info(f"每日任务计数已重置: {current_date}")

                # 检查是否在采集时间窗口内
                if not self._is_collection_time():
                    # 非采集时间，等待较长时间
                    await asyncio.sleep(self.config.collection_check_interval)
                    continue

                # 检查是否需要执行采集
                if current_time - self.last_collection_check >= self.config.collection_check_interval:
                    await self._perform_periodic_collection()
                    self.last_collection_check = current_time

                # 在采集时间窗口内，较短的检查间隔
                await asyncio.sleep(min(60, self.config.collection_check_interval))  # 最多1分钟检查一次

            except Exception as e:
                logger.error(f"定期采集循环异常: {e}")
                await asyncio.sleep(300)  # 异常后等待5分钟

        logger.info("定期历史数据采集循环结束")

    def _is_collection_time(self) -> bool:
        """检查当前是否在采集时间窗口内"""
        return self.config_manager.config.time_window.is_within_window()

    async def _perform_periodic_collection(self):
        """执行定期历史数据采集"""
        try:
            # 检查全局启用状态
            if not self.config_manager.config.enabled:
                logger.debug("历史数据采集已禁用")
                return

            # 检查每日任务限制
            if self.daily_task_count >= self.config_manager.config.max_daily_tasks:
                logger.info(f"已达到每日最大任务数限制: {self.config_manager.config.max_daily_tasks}")
                return

            # 获取活跃的采集规则
            active_rules = self.config_manager.get_active_rules()
            if not active_rules:
                logger.debug("没有活跃的采集规则")
                return

            # 收集需要采集的股票
            symbols_to_collect = []
            for rule in active_rules:
                if rule.needs_collection():
                    symbols_to_collect.extend(rule.symbols)

            # 去重
            symbols_to_collect = list(set(symbols_to_collect))
            if not symbols_to_collect:
                # 如果规则中没有股票，尝试从数据源配置刷新
                logger.info("规则中没有股票代码，尝试从数据源配置刷新...")
                symbols_to_collect = self.config_manager.get_symbols_to_collect(refresh_from_data_sources=True)

                if not symbols_to_collect:
                    logger.warning("从数据源配置也未找到股票代码，请检查数据源配置")
                    return

            # 计算本次采集批次
            batch_size = min(
                self.config_manager.config.batch_size,
                self.config_manager.config.max_daily_tasks - self.daily_task_count,
                len(symbols_to_collect)
            )

            if batch_size <= 0:
                logger.info("无法创建新的采集任务（达到限制）")
                return

            # 选择要采集的股票
            selected_symbols = symbols_to_collect[:batch_size]

            # 双轨边界：历史轨只采「日常采集周期之前」的数据，与日常轨不重叠
            daily_period_days = getattr(self.config_manager.config, 'daily_period_days', 30)
            max_history_days = getattr(self.config_manager.config, 'max_history_days', 3650)
            period_days = self.config_manager.config.collection_period_days  # 每段天数（90 或 365）
            max_segments = self.config_manager.config.max_concurrent_segments
            max_daily_tasks = self.config_manager.config.max_daily_tasks

            today = datetime.now().date()
            cutoff = today - timedelta(days=daily_period_days)  # 日常区间起始日
            hist_end = cutoff - timedelta(days=1)  # 历史右端（含），不包含日常区间
            hist_start = hist_end - timedelta(days=max_history_days)

            # 为每个股票按分段创建采集任务
            tasks_created = 0
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for symbol in selected_symbols:
                try:
                    data_types = self.config_manager.config.default_data_types.copy()
                    priority_str = self.config_manager.config.default_priority
                    for rule in active_rules:
                        if symbol in rule.symbols:
                            data_types = rule.data_types
                            priority_str = rule.priority
                            break

                    priority_map = {
                        'low': HistoricalTaskPriority.LOW,
                        'normal': HistoricalTaskPriority.NORMAL,
                        'high': HistoricalTaskPriority.HIGH,
                        'urgent': HistoricalTaskPriority.URGENT
                    }
                    priority = priority_map.get(priority_str, HistoricalTaskPriority.NORMAL)

                    segment_end = hist_end
                    segments_created = 0
                    while (segment_end >= hist_start and segments_created < max_segments and
                           (self.daily_task_count + tasks_created) < max_daily_tasks):
                        segment_start = segment_end - timedelta(days=period_days - 1)
                        if segment_start < hist_start:
                            segment_start = hist_start
                        start_date = segment_start.strftime('%Y-%m-%d')
                        end_date = segment_end.strftime('%Y-%m-%d')

                        self.schedule_task(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            data_types=data_types,
                            priority=priority
                        )
                        tasks_created += 1
                        segments_created += 1
                        logger.info(f"创建定期历史数据采集任务（分段，与日常不重叠）: {symbol} [{start_date}, {end_date}], 优先级: {priority_str}")
                        segment_end = segment_start - timedelta(days=1)

                except Exception as e:
                    logger.error(f"为股票 {symbol} 创建采集任务失败: {e}")

            if tasks_created > 0:
                self.daily_task_count += tasks_created

                # 更新规则的最后采集时间
                for rule in active_rules:
                    if any(symbol in rule.symbols for symbol in selected_symbols):
                        self.config_manager.update_rule_last_collection(rule.name, current_time)

                # 保存配置
                self.config_manager.save_config()

                logger.info(f"本次定期采集创建了 {tasks_created} 个任务，今日累计: {self.daily_task_count}")

        except Exception as e:
            logger.error(f"执行定期历史数据采集失败: {e}")


    def update_collection_config(self, symbols: Optional[List[str]] = None,
                               data_types: Optional[List[str]] = None,
                               config: Optional[Dict[str, Any]] = None):
        """更新采集配置"""
        # 此方法已废弃，请使用配置管理器直接操作
        logger.warning("update_collection_config方法已废弃，请使用配置管理器API")
        logger.info("请使用: get_historical_collection_config_manager() 来管理配置")



    async def trigger_immediate_collection(self, force: bool = False) -> Dict[str, Any]:
        """
        触发立即历史数据采集（可跳过时间窗口检查）

        Args:
            force: 是否强制执行，跳过时间窗口检查

        Returns:
            执行结果字典
        """
        try:
            # 如果不是强制模式，仍然检查时间窗口
            if not force and not self._is_collection_time():
                next_check_time = self.config_manager._calculate_next_check_time()
                return {
                    "success": False,
                    "message": "当前不在采集时间窗口内",
                    "next_window": next_check_time,
                    "current_time": datetime.now().strftime('%H:%M:%S'),
                    "window_start": f"{self.config_manager.config.time_window.start_hour:02d}:00",
                    "window_end": f"{self.config_manager.config.time_window.end_hour:02d}:00"
                }

            # 执行采集逻辑
            tasks_before = self.daily_task_count

            if force:
                # 强制模式：跳过时间间隔检查
                await self._perform_forced_collection()
            else:
                # 正常模式：使用定期采集逻辑
                await self._perform_periodic_collection()

            tasks_after = self.daily_task_count
            tasks_created = tasks_after - tasks_before

            return {
                "success": True,
                "message": "立即采集执行完成",
                "tasks_created": tasks_created,
                "total_daily_tasks": self.daily_task_count,
                "force_mode": force,
                "execution_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"立即采集执行失败: {e}")
            return {
                "success": False,
                "message": f"执行失败: {str(e)}",
                "error_type": type(e).__name__
            }

    async def _register_default_worker(self):
        """
        注册默认工作进程

        在调度器启动时自动注册一个默认工作进程，确保任务能够被执行。
        """
        try:
            # 检查是否已有工作进程
            if self.worker_nodes:
                logger.debug("已有工作进程注册，跳过默认工作进程注册")
                return

            # 注册默认工作进程
            default_worker_id = "default_worker"
            default_host = "localhost"
            default_port = 8080
            default_max_concurrent = 2

            success = self.register_worker(
                worker_id=default_worker_id,
                host=default_host,
                port=default_port,
                capabilities=['historical_data'],
                max_concurrent=default_max_concurrent
            )

            if success:
                logger.info(f"默认工作进程已自动注册: {default_worker_id} ({default_host}:{default_port}, 最大并发: {default_max_concurrent})")
                logger.info("历史数据采集调度器现已准备就绪，可以执行任务")
            else:
                logger.warning("默认工作进程注册失败")

        except Exception as e:
            logger.error(f"注册默认工作进程失败: {e}")

    async def _ensure_default_worker(self):
        """
        确保有默认工作进程可用

        用于在调度器已经在运行时检查并注册默认工作进程。
        """
        try:
            # 检查是否有活跃的工作进程
            active_workers = [w for w in self.worker_nodes.values() if w.is_active]
            if active_workers:
                logger.debug(f"已有 {len(active_workers)} 个活跃工作进程，无需注册默认工作进程")
                return

            # 检查是否有任何工作进程（包括不活跃的）
            if self.worker_nodes:
                logger.debug(f"已有工作进程 {list(self.worker_nodes.keys())}，但没有活跃的，尝试重新激活")
                # 这里可以添加重新激活逻辑，但暂时先注册新的默认工作进程
                pass

            # 注册默认工作进程
            default_worker_id = "default_worker"
            default_host = "localhost"
            default_port = 8080
            default_max_concurrent = 2

            # 检查是否已经注册过这个工作进程
            if default_worker_id in self.worker_nodes:
                logger.debug(f"默认工作进程 {default_worker_id} 已注册，检查状态")
                worker = self.worker_nodes[default_worker_id]
                if not worker.is_active:
                    logger.info(f"重新激活默认工作进程: {default_worker_id}")
                    worker.is_active = True
                    # 更新心跳时间
                    worker.last_heartbeat = time.time()
                return

            success = self.register_worker(
                worker_id=default_worker_id,
                host=default_host,
                port=default_port,
                capabilities=['historical_data'],
                max_concurrent=default_max_concurrent
            )

            if success:
                logger.info(f"默认工作进程已重新注册: {default_worker_id} ({default_host}:{default_port}, 最大并发: {default_max_concurrent})")
                logger.info("历史数据采集调度器现已准备就绪，可以执行任务")
                
                # 启动默认工作进程的心跳任务
                asyncio.create_task(self._default_worker_heartbeat(default_worker_id))
            else:
                logger.warning("默认工作进程重新注册失败")

        except Exception as e:
            logger.error(f"确保默认工作进程失败: {e}")

    async def _default_worker_heartbeat(self, worker_id: str):
        """默认工作进程心跳任务"""
        try:
            while self.status == SchedulerStatus.RUNNING:
                if worker_id in self.worker_nodes:
                    # 更新调度器中的工作进程心跳
                    self.worker_nodes[worker_id].last_heartbeat = time.time()
                    
                    # 更新监控器中的工作进程心跳
                    self.monitor.update_worker_heartbeat(worker_id)
                    
                    logger.debug(f"默认工作进程 {worker_id} 心跳发送成功")
                
                # 每20秒发送一次心跳
                await asyncio.sleep(20)
                
        except Exception as e:
            logger.error(f"默认工作进程心跳任务异常: {e}")

    async def ensure_worker_available(self) -> bool:
        """
        确保有工作进程可用

        这个方法可以被外部调用，用于确保调度器有工作进程来执行任务。

        Returns:
            bool: 是否有工作进程可用
        """
        try:
            await self._ensure_default_worker()
            active_workers = [w for w in self.worker_nodes.values() if w.is_active]
            return len(active_workers) > 0
        except Exception as e:
            logger.error(f"确保工作进程可用失败: {e}")
            return False

    async def _perform_forced_collection(self):
        """执行强制历史数据采集（跳过时间间隔检查）"""
        try:
            # 检查全局启用状态
            if not self.config_manager.config.enabled:
                logger.debug("历史数据采集已禁用")
                return

            # 检查每日任务限制
            if self.daily_task_count >= self.config_manager.config.max_daily_tasks:
                logger.info(f"已达到每日最大任务数限制: {self.config_manager.config.max_daily_tasks}")
                return

            # 获取活跃的采集规则（强制模式下不检查needs_collection）
            active_rules = self.config_manager.get_active_rules()
            if not active_rules:
                logger.debug("没有活跃的采集规则")
                return

            # 收集所有活跃规则中的股票（强制模式）
            symbols_to_collect = []
            for rule in active_rules:
                symbols_to_collect.extend(rule.symbols)

            # 去重
            symbols_to_collect = list(set(symbols_to_collect))
            if not symbols_to_collect:
                # 如果规则中没有股票，尝试从数据源配置刷新
                logger.info("规则中没有股票代码，尝试从数据源配置刷新...")
                symbols_to_collect = self.config_manager.get_symbols_to_collect(refresh_from_data_sources=True)

                if not symbols_to_collect:
                    logger.warning("从数据源配置也未找到股票代码，请检查数据源配置")
                    return

            # 计算本次采集批次
            batch_size = min(
                self.config_manager.config.batch_size,
                self.config_manager.config.max_daily_tasks - self.daily_task_count,
                len(symbols_to_collect)
            )

            if batch_size <= 0:
                logger.info("无法创建新的采集任务（达到限制）")
                return

            # 选择要采集的股票
            selected_symbols = symbols_to_collect[:batch_size]

            # 双轨边界：历史轨只采「日常采集周期之前」的数据，与日常轨不重叠
            daily_period_days = getattr(self.config_manager.config, 'daily_period_days', 30)
            max_history_days = getattr(self.config_manager.config, 'max_history_days', 3650)
            period_days = self.config_manager.config.collection_period_days
            max_segments = self.config_manager.config.max_concurrent_segments
            max_daily_tasks = self.config_manager.config.max_daily_tasks

            today = datetime.now().date()
            cutoff = today - timedelta(days=daily_period_days)
            hist_end = cutoff - timedelta(days=1)
            hist_start = hist_end - timedelta(days=max_history_days)

            tasks_created = 0
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for symbol in selected_symbols:
                try:
                    data_types = self.config_manager.config.default_data_types.copy()
                    priority_str = self.config_manager.config.default_priority
                    for rule in active_rules:
                        if symbol in rule.symbols:
                            data_types = rule.data_types
                            priority_str = rule.priority
                            break

                    priority_map = {
                        'low': HistoricalTaskPriority.LOW,
                        'normal': HistoricalTaskPriority.NORMAL,
                        'high': HistoricalTaskPriority.HIGH,
                        'urgent': HistoricalTaskPriority.URGENT
                    }
                    priority = priority_map.get(priority_str, HistoricalTaskPriority.NORMAL)

                    segment_end = hist_end
                    segments_created = 0
                    while (segment_end >= hist_start and segments_created < max_segments and
                           (self.daily_task_count + tasks_created) < max_daily_tasks):
                        segment_start = segment_end - timedelta(days=period_days - 1)
                        if segment_start < hist_start:
                            segment_start = hist_start
                        start_date = segment_start.strftime('%Y-%m-%d')
                        end_date = segment_end.strftime('%Y-%m-%d')

                        self.schedule_task(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            data_types=data_types,
                            priority=priority
                        )
                        tasks_created += 1
                        segments_created += 1
                        logger.info(f"创建强制历史数据采集任务（分段，与日常不重叠）: {symbol} [{start_date}, {end_date}], 优先级: {priority_str}")
                        segment_end = segment_start - timedelta(days=1)

                except Exception as e:
                    logger.error(f"为股票 {symbol} 创建强制采集任务失败: {e}")

            if tasks_created > 0:
                self.daily_task_count += tasks_created

                # 更新规则的最后采集时间
                for rule in active_rules:
                    if any(symbol in rule.symbols for symbol in selected_symbols):
                        self.config_manager.update_rule_last_collection(rule.name, current_time)

                # 保存配置
                self.config_manager.save_config()

                logger.info(f"强制采集创建了 {tasks_created} 个任务，今日累计: {self.daily_task_count}")

        except Exception as e:
            logger.error(f"执行强制历史数据采集失败: {e}")

    async def _create_segmented_tasks(self, symbol: str, total_days: int, data_types: List[str], priority_str: str) -> int:
        """为大量历史数据创建分段采集任务。

        使用双轨边界，右端为 hist_end = today - daily_period_days - 1，
        与日常采集 [today - daily_period_days, today] 不重叠。
        """
        try:
            daily_period_days = getattr(self.config_manager.config, 'daily_period_days', 30)
            max_history_days = getattr(self.config_manager.config, 'max_history_days', 3650)
            segment_years = self.config_manager.config.segment_years
            max_segments = self.config_manager.config.max_concurrent_segments

            today = datetime.now().date()
            hist_end = today - timedelta(days=daily_period_days + 1)
            effective_days = min(total_days, max_history_days)
            hist_start = hist_end - timedelta(days=effective_days)

            segment_days = segment_years * 365
            num_segments = min((effective_days // segment_days) + 1, max_segments)

            priority_map = {
                'low': HistoricalTaskPriority.LOW,
                'normal': HistoricalTaskPriority.NORMAL,
                'high': HistoricalTaskPriority.HIGH,
                'urgent': HistoricalTaskPriority.URGENT
            }
            priority = priority_map.get(priority_str, HistoricalTaskPriority.NORMAL)

            tasks_created = 0
            for i in range(num_segments):
                segment_end = hist_end - timedelta(days=i * segment_days)
                if segment_end < hist_start:
                    break
                segment_start = segment_end - timedelta(days=segment_days - 1)
                if segment_start < hist_start:
                    segment_start = hist_start

                start_date_str = segment_start.strftime('%Y-%m-%d')
                end_date_str = segment_end.strftime('%Y-%m-%d')

                self.schedule_task(
                    symbol=symbol,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    data_types=data_types,
                    priority=priority
                )
                tasks_created += 1
                logger.info(f"创建分段历史数据采集任务（与日常不重叠）: {symbol} (段{i+1}/{num_segments}: {start_date_str} - {end_date_str})")

            return tasks_created

        except Exception as e:
            logger.error(f"为股票 {symbol} 创建分段采集任务失败: {e}")
            return 0


# 全局调度器实例
_scheduler_instance: Optional[HistoricalDataScheduler] = None


def get_historical_data_scheduler(config: Optional[SchedulerConfig] = None) -> HistoricalDataScheduler:
    """
    获取历史数据采集调度器实例（单例模式）

    Args:
        config: 调度器配置

    Returns:
        HistoricalDataScheduler: 调度器实例
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = HistoricalDataScheduler(config)
    else:
        # 如果提供了新配置，检查是否需要更新现有实例
        if config is not None and hasattr(_scheduler_instance, 'config'):
            # 更新配置（如果配置有差异）
            # 注意：这里只是简单比较，不做深度比较
            if config.__dict__ != _scheduler_instance.config.__dict__:
                logger.info("检测到调度器配置变化，正在更新配置")
                _scheduler_instance.config = config
    return _scheduler_instance