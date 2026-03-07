#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调度器健康监控和自动恢复模块

功能：
1. 定期检查调度器运行状态
2. 调度器异常时自动重启
3. 发送告警通知
4. 记录健康检查历史
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckRecord:
    """健康检查记录"""
    timestamp: float
    status: HealthStatus
    scheduler_running: bool
    worker_count: int
    queue_size: int
    active_tasks: int
    message: str
    recovery_action: Optional[str] = None


class SchedulerHealthMonitor:
    """
    调度器健康监控器
    
    定期检查调度器状态，异常时自动恢复
    """
    
    def __init__(
        self,
        check_interval: int = 60,  # 检查间隔（秒）
        recovery_enabled: bool = True,
        max_recovery_attempts: int = 3,
        alert_threshold: int = 2  # 连续失败多少次后告警
    ):
        """
        初始化健康监控器
        
        Args:
            check_interval: 检查间隔（秒）
            recovery_enabled: 是否启用自动恢复
            max_recovery_attempts: 最大恢复尝试次数
            alert_threshold: 告警阈值（连续失败次数）
        """
        self.check_interval = check_interval
        self.recovery_enabled = recovery_enabled
        self.max_recovery_attempts = max_recovery_attempts
        self.alert_threshold = alert_threshold
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._check_history: List[HealthCheckRecord] = []
        self._max_history_size = 1000
        
        # 连续失败计数
        self._consecutive_failures = 0
        self._recovery_attempts = 0
        
        # 统计信息
        self._stats = {
            "total_checks": 0,
            "healthy_checks": 0,
            "warning_checks": 0,
            "critical_checks": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0
        }
        
        logger.info(f"调度器健康监控器已初始化 (检查间隔: {check_interval}秒)")
    
    async def start(self):
        """启动健康监控"""
        if self._running:
            logger.warning("健康监控已在运行")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("✅ 调度器健康监控已启动")
    
    async def stop(self):
        """停止健康监控"""
        if not self._running:
            return
        
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 调度器健康监控已停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            from src.features.distributed.task_scheduler import get_task_scheduler
            from src.features.distributed.worker_manager import get_worker_manager
            
            scheduler = get_task_scheduler()
            worker_manager = get_worker_manager()
            
            # 获取调度器状态
            scheduler_running = scheduler._running
            workers = worker_manager.get_all_workers()
            worker_count = len(workers)
            
            # 获取任务统计
            try:
                stats = scheduler.get_scheduler_stats()
                queue_size = stats.get("queue_size", 0)
                active_tasks = stats.get("active_tasks", 0)
            except:
                queue_size = 0
                active_tasks = 0
            
            # 判断健康状态
            if scheduler_running and worker_count > 0:
                status = HealthStatus.HEALTHY
                message = f"调度器运行正常，工作节点: {worker_count}"
                self._consecutive_failures = 0
                self._recovery_attempts = 0
                self._stats["healthy_checks"] += 1
            elif scheduler_running and worker_count == 0:
                status = HealthStatus.WARNING
                message = "调度器运行但无工作节点"
                self._consecutive_failures += 1
                self._stats["warning_checks"] += 1
            else:
                status = HealthStatus.CRITICAL
                message = "调度器未运行"
                self._consecutive_failures += 1
                self._stats["critical_checks"] += 1
            
            self._stats["total_checks"] += 1
            
            # 记录检查结果
            record = HealthCheckRecord(
                timestamp=time.time(),
                status=status,
                scheduler_running=scheduler_running,
                worker_count=worker_count,
                queue_size=queue_size,
                active_tasks=active_tasks,
                message=message
            )
            self._add_check_record(record)
            
            # 根据状态采取相应措施
            if status == HealthStatus.CRITICAL:
                logger.error(f"🚨 调度器健康检查失败: {message}")
                
                # 尝试自动恢复
                if self.recovery_enabled and self._recovery_attempts < self.max_recovery_attempts:
                    recovery_success = await self._attempt_recovery()
                    record.recovery_action = f"自动恢复尝试 #{self._recovery_attempts}"
                    
                    if recovery_success:
                        record.status = HealthStatus.HEALTHY
                        record.message = "自动恢复成功"
                        self._stats["successful_recoveries"] += 1
                    else:
                        self._stats["failed_recoveries"] += 1
                
                # 发送告警
                if self._consecutive_failures >= self.alert_threshold:
                    await self._send_alert(record)
            
            elif status == HealthStatus.WARNING:
                logger.warning(f"⚠️ 调度器健康检查警告: {message}")
                
                # 尝试创建工作节点
                if self.recovery_enabled and worker_count == 0:
                    await self._ensure_workers()
            
            else:
                logger.debug(f"✅ 调度器健康检查通过: {message}")
            
        except Exception as e:
            logger.error(f"❌ 健康检查执行失败: {e}")
            error_record = HealthCheckRecord(
                timestamp=time.time(),
                status=HealthStatus.UNKNOWN,
                scheduler_running=False,
                worker_count=0,
                queue_size=0,
                active_tasks=0,
                message=f"健康检查异常: {str(e)}"
            )
            self._add_check_record(error_record)
    
    async def _attempt_recovery(self) -> bool:
        """
        尝试恢复调度器
        
        Returns:
            是否恢复成功
        """
        self._recovery_attempts += 1
        self._stats["recovery_attempts"] += 1
        
        try:
            logger.info(f"🔄 尝试恢复调度器 (第{self._recovery_attempts}次)...")
            
            from src.features.distributed.task_scheduler import get_task_scheduler
            from src.features.distributed.worker_manager import get_worker_manager
            
            scheduler = get_task_scheduler()
            worker_manager = get_worker_manager()
            
            # 1. 尝试启动调度器
            if not scheduler._running:
                try:
                    scheduler.start()
                    logger.info("✅ 调度器已启动")
                except Exception as e:
                    logger.error(f"❌ 启动调度器失败: {e}")
                    return False
            
            # 2. 等待调度器稳定
            await asyncio.sleep(2)
            
            # 3. 检查工作节点
            workers = worker_manager.get_all_workers()
            if len(workers) == 0:
                logger.info("🔄 创建工作节点...")
                try:
                    scheduler.start_with_workers()
                    logger.info("✅ 工作节点已创建")
                except Exception as e:
                    logger.error(f"❌ 创建工作节点失败: {e}")
            
            # 4. 验证恢复结果
            await asyncio.sleep(1)
            if scheduler._running:
                logger.info(f"✅ 调度器恢复成功 (尝试次数: {self._recovery_attempts})")
                self._consecutive_failures = 0
                return True
            else:
                logger.error("❌ 调度器恢复验证失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 恢复过程异常: {e}")
            return False
    
    async def _ensure_workers(self):
        """确保有足够的工作节点"""
        try:
            from src.features.distributed.task_scheduler import get_task_scheduler
            from src.features.distributed.worker_manager import get_worker_manager
            
            scheduler = get_task_scheduler()
            worker_manager = get_worker_manager()
            
            workers = worker_manager.get_all_workers()
            if len(workers) == 0 and scheduler._running:
                logger.info("🔄 自动创建工作节点...")
                scheduler.start_with_workers()
                logger.info("✅ 工作节点已自动创建")
        except Exception as e:
            logger.error(f"❌ 自动创建工作节点失败: {e}")
    
    async def _send_alert(self, record: HealthCheckRecord):
        """
        发送告警通知
        
        Args:
            record: 健康检查记录
        """
        try:
            alert_message = {
                "type": "scheduler_health_alert",
                "level": "critical" if record.status == HealthStatus.CRITICAL else "warning",
                "message": record.message,
                "timestamp": record.timestamp,
                "details": {
                    "scheduler_running": record.scheduler_running,
                    "worker_count": record.worker_count,
                    "queue_size": record.queue_size,
                    "active_tasks": record.active_tasks,
                    "consecutive_failures": self._consecutive_failures,
                    "recovery_attempts": self._recovery_attempts
                }
            }
            
            # 通过WebSocket发送告警
            try:
                from src.gateway.web.websocket_manager import manager
                await manager.broadcast("feature_engineering", alert_message)
            except Exception as e:
                logger.debug(f"WebSocket告警发送失败: {e}")
            
            # 记录到日志
            logger.critical(f"🚨 调度器健康告警: {record.message}")
            
        except Exception as e:
            logger.error(f"❌ 发送告警失败: {e}")
    
    def _add_check_record(self, record: HealthCheckRecord):
        """添加检查记录"""
        self._check_history.append(record)
        
        # 限制历史记录大小
        if len(self._check_history) > self._max_history_size:
            self._check_history = self._check_history[-self._max_history_size:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取当前健康状态
        
        Returns:
            健康状态信息
        """
        if not self._check_history:
            return {
                "status": "unknown",
                "message": "暂无健康检查记录"
            }
        
        latest = self._check_history[-1]
        
        return {
            "status": latest.status.value,
            "timestamp": latest.timestamp,
            "datetime": datetime.fromtimestamp(latest.timestamp).isoformat(),
            "scheduler_running": latest.scheduler_running,
            "worker_count": latest.worker_count,
            "queue_size": latest.queue_size,
            "active_tasks": latest.active_tasks,
            "message": latest.message,
            "consecutive_failures": self._consecutive_failures,
            "recovery_attempts": self._recovery_attempts,
            "stats": self._stats
        }
    
    def get_check_history(
        self,
        limit: int = 100,
        status_filter: Optional[HealthStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        获取检查历史
        
        Args:
            limit: 返回记录数量限制
            status_filter: 状态过滤
            
        Returns:
            检查历史记录列表
        """
        records = self._check_history
        
        if status_filter:
            records = [r for r in records if r.status == status_filter]
        
        records = records[-limit:]
        
        return [
            {
                "timestamp": r.timestamp,
                "datetime": datetime.fromtimestamp(r.timestamp).isoformat(),
                "status": r.status.value,
                "scheduler_running": r.scheduler_running,
                "worker_count": r.worker_count,
                "queue_size": r.queue_size,
                "active_tasks": r.active_tasks,
                "message": r.message,
                "recovery_action": r.recovery_action
            }
            for r in records
        ]


# 全局监控器实例
_health_monitor: Optional[SchedulerHealthMonitor] = None


async def get_scheduler_health_monitor() -> SchedulerHealthMonitor:
    """
    获取全局健康监控器实例
    
    Returns:
        健康监控器实例
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = SchedulerHealthMonitor()
    return _health_monitor


async def start_scheduler_health_monitor():
    """启动调度器健康监控"""
    monitor = await get_scheduler_health_monitor()
    await monitor.start()
    return monitor


async def stop_scheduler_health_monitor():
    """停止调度器健康监控"""
    global _health_monitor
    if _health_monitor:
        await _health_monitor.stop()
        _health_monitor = None
