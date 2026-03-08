#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工作节点管理器

提供分布式特征计算的工作节点管理功能。
支持统一工作节点注册表集成，区分特征工作节点类型。
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# 导入统一工作节点注册表（从分布式协调器层）
from src.infrastructure.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus as UnifiedWorkerStatus
)


logger = logging.getLogger(__name__)


class WorkerStatus(Enum):

    """工作节点状态"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class WorkerInfo:

    """工作节点信息"""
    worker_id: str
    status: WorkerStatus
    capabilities: Dict[str, Any]
    registered_at: datetime
    last_heartbeat: datetime
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    current_load: float = 0.0
    performance_score: float = 1.0


class FeatureWorkerManager:

    """特征工作节点管理器"""

    def __init__(self):
        """初始化工作节点管理器"""
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread = None

        # 统计信息
        self._stats = {
            "total_workers": 0,
            "active_workers": 0,
            "idle_workers": 0,
            "busy_workers": 0,
            "offline_workers": 0
        }

    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]) -> bool:
        """注册工作节点"""
        with self._lock:
            # 同时注册到统一注册表（标记为特征工作节点）
            try:
                registry = get_unified_worker_registry()
                registry.register_worker(
                    worker_id=worker_id,
                    worker_type=WorkerType.FEATURE_WORKER,
                    capabilities=capabilities
                )
            except Exception as e:
                logger.warning(f"注册到统一注册表失败: {e}")
            
            if worker_id in self._workers:
                logger.warning(f"工作节点已存在: {worker_id}")
                # 如果工作节点已存在，更新其信息
                worker_info = self._workers[worker_id]
                worker_info.capabilities = capabilities
                worker_info.last_heartbeat = datetime.now()
                if worker_info.status == WorkerStatus.OFFLINE:
                    worker_info.status = WorkerStatus.IDLE
                    self._stats["offline_workers"] -= 1
                    self._stats["idle_workers"] += 1
                    self._stats["active_workers"] += 1
                logger.info(f"更新已存在的工作节点: {worker_id}")
                return True

            worker_info = WorkerInfo(
                worker_id=worker_id,
                status=WorkerStatus.IDLE,
                capabilities=capabilities,
                registered_at=datetime.now(),
                last_heartbeat=datetime.now()
            )

            self._workers[worker_id] = worker_info
            self._stats["total_workers"] += 1
            self._stats["active_workers"] += 1
            self._stats["idle_workers"] += 1

            logger.info(f"注册工作节点: {worker_id}, 能力: {capabilities}")
            return True

    def unregister_worker(self, worker_id: str) -> bool:
        """注销工作节点"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]

            # 更新统计信息
            self._stats["total_workers"] -= 1
            self._stats["active_workers"] -= 1

            if worker_info.status == WorkerStatus.IDLE:
                self._stats["idle_workers"] -= 1
            elif worker_info.status == WorkerStatus.BUSY:
                self._stats["busy_workers"] -= 1
            elif worker_info.status == WorkerStatus.OFFLINE:
                self._stats["offline_workers"] -= 1

            del self._workers[worker_id]
            logger.info(f"注销工作节点: {worker_id}")
            return True

    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """更新工作节点心跳"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]
            worker_info.last_heartbeat = datetime.now()

            # 如果节点之前离线，现在重新上线
            if worker_info.status == WorkerStatus.OFFLINE:
                worker_info.status = WorkerStatus.IDLE
                self._stats["offline_workers"] -= 1
                self._stats["idle_workers"] += 1
                self._stats["active_workers"] += 1
                logger.info(f"工作节点重新上线: {worker_id}")

            return True

    def update_worker_status(self, worker_id: str, status: WorkerStatus) -> bool:
        """更新工作节点状态"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]
            old_status = worker_info.status

            # 检查状态是否相同，如果相同则不更新
            if old_status == status:
                return True

            # 更新状态
            worker_info.status = status

            # 更新统计信息
            if old_status == WorkerStatus.IDLE:
                self._stats["idle_workers"] -= 1
            elif old_status == WorkerStatus.BUSY:
                self._stats["busy_workers"] -= 1
            elif old_status == WorkerStatus.OFFLINE:
                self._stats["offline_workers"] -= 1

            if status == WorkerStatus.IDLE:
                self._stats["idle_workers"] += 1
                self._stats["active_workers"] += 1
            elif status == WorkerStatus.BUSY:
                self._stats["busy_workers"] += 1
                self._stats["active_workers"] += 1
            elif status == WorkerStatus.OFFLINE:
                self._stats["offline_workers"] += 1
                self._stats["active_workers"] -= 1

            logger.info(f"工作节点 {worker_id} 状态更新: {old_status} -> {status}")
            return True

    def assign_task_to_worker(self, worker_id: str, task_id: str) -> bool:
        """分配任务给工作节点"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]

            # 检查工作节点是否可用
            if worker_info.status != WorkerStatus.IDLE:
                return False

            # 分配任务
            worker_info.current_task = task_id
            worker_info.status = WorkerStatus.BUSY

            # 更新统计信息
            self._stats["idle_workers"] -= 1
            self._stats["busy_workers"] += 1

            logger.info(f"为工作节点 {worker_id} 分配任务: {task_id}")
            return True

    def complete_task(self, worker_id: str, processing_time: float) -> bool:
        """完成任务"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]

            # 完成任务
            worker_info.current_task = None
            worker_info.status = WorkerStatus.IDLE
            worker_info.completed_tasks += 1
            worker_info.total_processing_time += processing_time

            # 更新统计信息
            self._stats["busy_workers"] -= 1
            self._stats["idle_workers"] += 1

            logger.info(f"工作节点 {worker_id} 完成任务，处理时间: {processing_time:.2f}秒")
            return True

    def fail_task(self, worker_id: str) -> bool:
        """任务失败"""
        with self._lock:
            if worker_id not in self._workers:
                return False

            worker_info = self._workers[worker_id]

            # 任务失败
            worker_info.current_task = None
            worker_info.status = WorkerStatus.IDLE
            worker_info.failed_tasks += 1

            # 更新统计信息
            self._stats["busy_workers"] -= 1
            self._stats["idle_workers"] += 1

            logger.warning(f"工作节点 {worker_id} 任务失败")
            return True

    def check_worker_health(self, timeout_minutes: int = 5) -> List[str]:
        """检查工作节点健康状态"""
        current_time = datetime.now()
        timeout_seconds = timeout_minutes * 60
        unhealthy_workers = []

        with self._lock:
            for worker_id, worker_info in self._workers.items():
                # 检查心跳超时
                if (current_time - worker_info.last_heartbeat).seconds > timeout_seconds:
                    if worker_info.status != WorkerStatus.OFFLINE:
                        old_status = worker_info.status
                        worker_info.status = WorkerStatus.OFFLINE

                        # 更新统计信息
                        if old_status == WorkerStatus.IDLE:
                            self._stats["idle_workers"] -= 1
                        elif old_status == WorkerStatus.BUSY:
                            self._stats["busy_workers"] -= 1

                        self._stats["offline_workers"] += 1
                        self._stats["active_workers"] -= 1

                        unhealthy_workers.append(worker_id)
                        logger.warning(f"工作节点健康检查失败: {worker_id}")

        return unhealthy_workers

    def cleanup_offline_workers(self) -> int:
        """清理离线工作节点"""
        cleaned_count = 0

        with self._lock:
            offline_workers = [
                worker_id for worker_id, worker_info in self._workers.items()
                if worker_info.status == WorkerStatus.OFFLINE
            ]

            for worker_id in offline_workers:
                worker_info = self._workers[worker_id]

                # 更新统计信息
                self._stats["total_workers"] -= 1
                self._stats["offline_workers"] -= 1

                # 删除工作节点
                del self._workers[worker_id]
                cleaned_count += 1

                logger.info(f"清理离线工作节点: {worker_id}")

        return cleaned_count

    def get_available_workers(self) -> List[str]:
        """获取可用工作节点"""
        with self._lock:
            current_time = datetime.now()
            available_workers = []

            for worker_id, worker_info in self._workers.items():
                # 检查心跳时间（超过30秒认为离线）
                if (current_time - worker_info.last_heartbeat).seconds < 30:
                    if worker_info.status == WorkerStatus.IDLE:
                        available_workers.append(worker_id)

            return available_workers

    def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """获取工作节点信息"""
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_workers(self) -> List[WorkerInfo]:
        """获取所有工作节点"""
        with self._lock:
            return list(self._workers.values())

    def get_worker_stats(self) -> Dict[str, Any]:
        """获取工作节点统计信息"""
        with self._lock:
            stats = self._stats.copy()

            # 计算平均处理时间
            total_time = 0
            total_tasks = 0
            total_completed = 0
            total_failed = 0
            
            for worker in self._workers.values():
                total_time += worker.total_processing_time
                total_tasks += worker.completed_tasks + worker.failed_tasks
                total_completed += worker.completed_tasks
                total_failed += worker.failed_tasks

            if total_tasks > 0:
                stats["avg_processing_time"] = total_time / total_tasks
                stats["success_rate"] = (total_completed / total_tasks) * 100
            else:
                stats["avg_processing_time"] = 0
                stats["success_rate"] = 0
            
            # 添加更多统计信息
            stats["total_completed_tasks"] = total_completed
            stats["total_failed_tasks"] = total_failed
            stats["total_processing_time"] = total_time
            stats["last_updated"] = datetime.now().isoformat()
            
            # 计算工作节点性能指标
            if self._workers:
                avg_performance = sum(w.performance_score for w in self._workers.values()) / len(self._workers)
                stats["avg_performance_score"] = avg_performance
            else:
                stats["avg_performance_score"] = 0

            return stats

    def get_worker_health_status(self) -> Dict[str, Any]:
        """获取工作节点健康状态"""
        try:
            with self._lock:
                health_status = {
                    "total_workers": len(self._workers),
                    "active_workers": self._stats.get("active_workers", 0),
                    "idle_workers": self._stats.get("idle_workers", 0),
                    "busy_workers": self._stats.get("busy_workers", 0),
                    "offline_workers": self._stats.get("offline_workers", 0),
                    "health_score": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
                
                # 计算健康分数
                if self._workers:
                    healthy_count = self._stats.get("active_workers", 0)
                    health_status["health_score"] = (healthy_count / len(self._workers)) * 100
                else:
                    health_status["health_score"] = 0
                
                # 检查健康状态
                if len(self._workers) == 0:
                    health_status["status"] = "warning"
                    health_status["reason"] = "无工作节点"
                elif health_status["active_workers"] == 0:
                    health_status["status"] = "unhealthy"
                    health_status["reason"] = "无活跃工作节点"
                elif health_status["health_score"] < 50:
                    health_status["status"] = "warning"
                    health_status["reason"] = "工作节点健康状态不佳"
                else:
                    health_status["status"] = "healthy"
                
                logger.debug(f"工作节点健康状态: {health_status['status']}, 健康分数: {health_status['health_score']:.2f}%")
                return health_status
        except Exception as e:
            logger.error(f"获取工作节点健康状态失败: {e}")
            return {
                "status": "unhealthy",
                "reason": str(e),
                "last_updated": datetime.now().isoformat()
            }

    def find_best_worker(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """找到最适合的工作节点"""
        with self._lock:
            available_workers = self.get_available_workers()

            if not available_workers:
                return None

            best_worker = None
            best_score = -1

            for worker_id in available_workers:
                worker_info = self._workers[worker_id]
                score = self._calculate_worker_score(worker_info, task_requirements)

                if score > best_score:
                    best_score = score
                    best_worker = worker_id

            return best_worker

    def _calculate_worker_score(self,


                                worker_info: WorkerInfo,
                                task_requirements: Dict[str, Any]) -> float:
        """计算工作节点评分"""
        score = 0.0

        # 基于完成任务的评分
        total_tasks = worker_info.completed_tasks + worker_info.failed_tasks
        if total_tasks > 0:
            success_rate = worker_info.completed_tasks / total_tasks
            score += success_rate * 10  # 成功率权重

        # 基于处理时间的评分
        if total_tasks > 0:
            avg_time = worker_info.total_processing_time / total_tasks
            score += max(0, 10 - avg_time)  # 处理时间越短评分越高

        # 基于能力的评分
        capabilities = worker_info.capabilities
        if "cpu_cores" in capabilities and "cpu_cores" in task_requirements:
            cpu_score = min(capabilities["cpu_cores"] / task_requirements["cpu_cores"], 1.0)
            score += cpu_score * 5

        if "max_memory" in capabilities and "memory_required" in task_requirements:
            memory_score = min(capabilities["max_memory"] /
                               task_requirements["memory_required"], 1.0)
            score += memory_score * 5

        return score

    def start_monitoring(self) -> None:
        """启动监控"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("工作节点管理器监控已启动")

    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("工作节点管理器监控已停止")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                self._check_worker_status()
                self._update_stats()
                time.sleep(5)  # 每5秒检查一次

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10)

    def _check_worker_status(self) -> None:
        """检查工作节点状态"""
        current_time = datetime.now()
        logger.debug("开始检查工作节点状态")

        with self._lock:
            offline_count = 0
            for worker_id, worker_info in self._workers.items():
                # 计算心跳时间差
                heartbeat_diff = (current_time - worker_info.last_heartbeat).seconds
                logger.debug(f"工作节点 {worker_id} 心跳时间差: {heartbeat_diff}秒")
                
                # 检查心跳超时（增加到90秒，减少误判离线的情况）
                if heartbeat_diff > 90:
                    if worker_info.status != WorkerStatus.OFFLINE:
                        old_status = worker_info.status
                        worker_info.status = WorkerStatus.OFFLINE

                        # 更新统计信息
                        if old_status == WorkerStatus.IDLE:
                            self._stats["idle_workers"] -= 1
                        elif old_status == WorkerStatus.BUSY:
                            self._stats["busy_workers"] -= 1

                        self._stats["offline_workers"] += 1
                        self._stats["active_workers"] -= 1

                        offline_count += 1
                        logger.warning(f"工作节点离线: {worker_id}, 心跳超时: {heartbeat_diff}秒")
                elif heartbeat_diff > 60:
                    # 心跳时间超过60秒但未达到90秒，发出警告但不标记为离线
                    logger.warning(f"工作节点 {worker_id} 心跳延迟: {heartbeat_diff}秒")
                else:
                    # 心跳正常
                    logger.debug(f"工作节点 {worker_id} 心跳正常")
        
        if offline_count > 0:
            logger.info(f"工作节点状态检查完成，发现 {offline_count} 个离线工作节点")
        else:
            logger.debug("工作节点状态检查完成，所有工作节点心跳正常")

    def _update_stats(self) -> None:
        """更新统计信息"""
        with self._lock:
            # 重新计算统计信息
            idle_count = 0
            busy_count = 0
            offline_count = 0
            active_count = 0

            for worker_info in self._workers.values():
                if worker_info.status == WorkerStatus.IDLE:
                    idle_count += 1
                    active_count += 1
                elif worker_info.status == WorkerStatus.BUSY:
                    busy_count += 1
                    active_count += 1
                elif worker_info.status == WorkerStatus.OFFLINE:
                    offline_count += 1

            # 确保统计信息正确更新
            self._stats["total_workers"] = len(self._workers)
            self._stats["idle_workers"] = idle_count
            self._stats["busy_workers"] = busy_count
            self._stats["offline_workers"] = offline_count
            self._stats["active_workers"] = active_count
            
            # 记录统计信息更新
            logger.debug(f"工作节点统计信息更新: 总计={self._stats['total_workers']}, 活跃={self._stats['active_workers']}, 空闲={self._stats['idle_workers']}, 忙碌={self._stats['busy_workers']}, 离线={self._stats['offline_workers']}")


# 全局工作节点管理器实例
_worker_manager = FeatureWorkerManager()


def get_worker_manager() -> FeatureWorkerManager:
    """获取全局工作节点管理器"""
    return _worker_manager
