#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集调度协调器

功能：
- 管理多个数据采集器
- 协调采集任务调度
- 处理采集失败重试
- 监控采集状态

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time

logger = logging.getLogger(__name__)


class CollectionStatus(Enum):
    """采集状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class CollectionTask:
    """采集任务"""
    task_id: str
    symbol: str
    collector_name: str
    status: CollectionStatus = CollectionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Any = None


@dataclass
class CollectorStatus:
    """采集器状态"""
    name: str
    is_available: bool = False
    last_collection_time: Optional[datetime] = None
    total_collections: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    average_collection_time: float = 0.0
    error_message: Optional[str] = None


class DataCollectionOrchestrator:
    """
    数据采集调度协调器
    
    职责：
    1. 管理多个数据采集器
    2. 协调采集任务调度
    3. 处理采集失败重试
    4. 监控采集状态
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化数据采集协调器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.collectors: Dict[str, Any] = {}
        self.collector_status: Dict[str, CollectorStatus] = {}
        self.tasks: Dict[str, CollectionTask] = {}
        self.max_workers = max_workers
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._scheduled_tasks: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        logger.info(f"数据采集协调器初始化完成，最大工作线程数: {max_workers}")
    
    def register_collector(self, name: str, collector: Any) -> bool:
        """
        注册数据采集器
        
        Args:
            name: 采集器名称
            collector: 采集器实例
            
        Returns:
            是否注册成功
        """
        try:
            with self._lock:
                self.collectors[name] = collector
                self.collector_status[name] = CollectorStatus(name=name)
                
                # 检查采集器可用性
                if hasattr(collector, '_akshare_available'):
                    self.collector_status[name].is_available = collector._akshare_available
                elif hasattr(collector, 'is_available'):
                    self.collector_status[name].is_available = collector.is_available
                else:
                    # 默认假设可用
                    self.collector_status[name].is_available = True
            
            logger.info(f"采集器注册成功: {name}")
            return True
        except Exception as e:
            logger.error(f"采集器注册失败 {name}: {e}")
            return False
    
    def unregister_collector(self, name: str) -> bool:
        """
        注销数据采集器
        
        Args:
            name: 采集器名称
            
        Returns:
            是否注销成功
        """
        try:
            with self._lock:
                if name in self.collectors:
                    del self.collectors[name]
                    del self.collector_status[name]
            
            logger.info(f"采集器注销成功: {name}")
            return True
        except Exception as e:
            logger.error(f"采集器注销失败 {name}: {e}")
            return False
    
    def schedule_collection(
        self,
        symbols: List[str],
        collector_name: Optional[str] = None,
        frequency: str = "daily",
        start_time: Optional[datetime] = None
    ) -> List[str]:
        """
        调度采集任务
        
        Args:
            symbols: 股票代码列表
            collector_name: 采集器名称，默认使用第一个可用的
            frequency: 采集频率 (daily, hourly, realtime)
            start_time: 开始时间
            
        Returns:
            任务ID列表
        """
        task_ids = []
        
        # 如果没有指定采集器，使用第一个可用的
        if collector_name is None:
            for name, status in self.collector_status.items():
                if status.is_available:
                    collector_name = name
                    break
        
        if collector_name is None or collector_name not in self.collectors:
            logger.error("没有可用的采集器")
            return task_ids
        
        for symbol in symbols:
            task_id = f"{collector_name}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            task = CollectionTask(
                task_id=task_id,
                symbol=symbol,
                collector_name=collector_name
            )
            
            with self._lock:
                self.tasks[task_id] = task
                self._scheduled_tasks.append({
                    "task_id": task_id,
                    "symbol": symbol,
                    "collector_name": collector_name,
                    "frequency": frequency,
                    "start_time": start_time or datetime.now(),
                    "next_run_time": start_time or datetime.now()
                })
            
            task_ids.append(task_id)
            logger.info(f"任务调度成功: {task_id}, 股票: {symbol}")
        
        return task_ids
    
    def execute_task(self, task_id: str) -> bool:
        """
        执行单个采集任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否执行成功
        """
        if task_id not in self.tasks:
            logger.error(f"任务不存在: {task_id}")
            return False
        
        task = self.tasks[task_id]
        
        if task.collector_name not in self.collectors:
            logger.error(f"采集器不存在: {task.collector_name}")
            task.status = CollectionStatus.FAILED
            task.error_message = "采集器不存在"
            return False
        
        collector = self.collectors[task.collector_name]
        
        try:
            task.status = CollectionStatus.RUNNING
            task.start_time = datetime.now()
            
            logger.info(f"开始执行任务: {task_id}, 股票: {task.symbol}")
            
            # 执行采集
            if hasattr(collector, 'collect_and_save'):
                result = collector.collect_and_save(task.symbol)
            elif hasattr(collector, 'collect_stock_data'):
                data = collector.collect_stock_data(task.symbol)
                if data and hasattr(collector, 'save_to_database'):
                    result = collector.save_to_database(data, task.symbol)
                else:
                    result = False
            else:
                logger.error(f"采集器没有可用的采集方法: {task.collector_name}")
                result = False
            
            task.end_time = datetime.now()
            
            if result:
                task.status = CollectionStatus.SUCCESS
                task.result = result
                
                # 更新采集器状态
                with self._lock:
                    status = self.collector_status[task.collector_name]
                    status.last_collection_time = datetime.now()
                    status.successful_collections += 1
                    status.total_collections += 1
                
                logger.info(f"任务执行成功: {task_id}")
                return True
            else:
                raise Exception("采集返回失败结果")
                
        except Exception as e:
            task.end_time = datetime.now()
            task.retry_count += 1
            task.error_message = str(e)
            
            if task.retry_count < task.max_retries:
                task.status = CollectionStatus.RETRYING
                logger.warning(f"任务执行失败，准备重试: {task_id}, 错误: {e}, 重试次数: {task.retry_count}")
                return False
            else:
                task.status = CollectionStatus.FAILED
                
                # 更新采集器状态
                with self._lock:
                    status = self.collector_status[task.collector_name]
                    status.failed_collections += 1
                    status.total_collections += 1
                    status.error_message = str(e)
                
                logger.error(f"任务执行失败，已达最大重试次数: {task_id}, 错误: {e}")
                return False
    
    def execute_all_pending_tasks(self) -> Dict[str, bool]:
        """
        执行所有待处理任务
        
        Returns:
            任务ID到执行结果的映射
        """
        results = {}
        
        pending_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [CollectionStatus.PENDING, CollectionStatus.RETRYING]
        ]
        
        logger.info(f"开始执行 {len(pending_tasks)} 个待处理任务")
        
        for task_id in pending_tasks:
            result = self.execute_task(task_id)
            results[task_id] = result
        
        return results
    
    def start_scheduler(self):
        """启动调度器"""
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            logger.warning("调度器已在运行")
            return
        
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
        
        logger.info("调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self._stop_event.set()
        
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while not self._stop_event.is_set():
            try:
                now = datetime.now()
                
                with self._lock:
                    for scheduled_task in self._scheduled_tasks:
                        if scheduled_task["next_run_time"] <= now:
                            task_id = scheduled_task["task_id"]
                            
                            # 执行任务
                            if task_id in self.tasks:
                                task = self.tasks[task_id]
                                if task.status in [CollectionStatus.PENDING, CollectionStatus.RETRYING]:
                                    self.execute_task(task_id)
                            
                            # 更新下次执行时间
                            if scheduled_task["frequency"] == "daily":
                                scheduled_task["next_run_time"] = now + timedelta(days=1)
                            elif scheduled_task["frequency"] == "hourly":
                                scheduled_task["next_run_time"] = now + timedelta(hours=1)
                            elif scheduled_task["frequency"] == "realtime":
                                scheduled_task["next_run_time"] = now + timedelta(minutes=1)
                
                # 每分钟检查一次
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                time.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取采集状态
        
        Returns:
            采集状态字典
        """
        with self._lock:
            return {
                "collectors": {
                    name: {
                        "name": status.name,
                        "is_available": status.is_available,
                        "last_collection_time": status.last_collection_time.isoformat() if status.last_collection_time else None,
                        "total_collections": status.total_collections,
                        "successful_collections": status.successful_collections,
                        "failed_collections": status.failed_collections,
                        "success_rate": (
                            status.successful_collections / status.total_collections * 100
                            if status.total_collections > 0 else 0
                        ),
                        "error_message": status.error_message
                    }
                    for name, status in self.collector_status.items()
                },
                "tasks": {
                    task_id: {
                        "task_id": task.task_id,
                        "symbol": task.symbol,
                        "collector_name": task.collector_name,
                        "status": task.status.value,
                        "start_time": task.start_time.isoformat() if task.start_time else None,
                        "end_time": task.end_time.isoformat() if task.end_time else None,
                        "retry_count": task.retry_count,
                        "error_message": task.error_message
                    }
                    for task_id, task in self.tasks.items()
                },
                "scheduler_running": self._scheduler_thread is not None and self._scheduler_thread.is_alive()
            }
    
    def get_collector(self, name: str) -> Optional[Any]:
        """
        获取采集器
        
        Args:
            name: 采集器名称
            
        Returns:
            采集器实例
        """
        return self.collectors.get(name)


# 单例实例
_orchestrator: Optional[DataCollectionOrchestrator] = None


def get_data_collection_orchestrator() -> DataCollectionOrchestrator:
    """
    获取数据采集协调器单例
    
    Returns:
        DataCollectionOrchestrator实例
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DataCollectionOrchestrator()
    return _orchestrator
