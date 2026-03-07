"""
unified_worker_registry.py

统一工作节点注册表模块

提供统一的工作节点注册和管理功能，支持：
- 特征工作节点 (Feature Worker) - 特征层
- 训练执行器 (Training Executor) - ML层
- 推理工作节点 (Inference Worker) - 推理层
- 数据采集器 (Data Collector) - 数据层
- 其他类型工作节点

架构位置: 分布式协调器层 - 集群管理器组件
符合架构设计: 
- 分布式协调器架构设计 (docs\architecture\distributed_coordinator_architecture_design.md)
- 核心服务层架构设计 (docs\architecture\core_service_layer_architecture_design.md)

跨层服务说明:
本模块虽物理位于 distributed/registry/ 目录，但逻辑上服务于全系统各层级：
- 特征层 (Feature Layer): 特征工作节点管理
- ML层 (ML Layer): 训练执行器管理
- 推理层 (Inference Layer): 推理工作节点管理
- 数据层 (Data Layer): 数据采集器管理

作者: RQA2025 Team
日期: 2026-02-15
版本: 2.0 (迁移至分布式协调器层)
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set

# 配置日志
logger = logging.getLogger(__name__)


class WorkerType(Enum):
    """工作节点类型枚举"""
    FEATURE_WORKER = "feature_worker"      # 特征工作节点
    TRAINING_EXECUTOR = "training_executor"  # 训练执行器
    INFERENCE_WORKER = "inference_worker"   # 推理工作节点
    DATA_COLLECTOR = "data_collector"       # 数据采集器
    CUSTOM = "custom"                       # 自定义类型


class WorkerStatus(Enum):
    """工作节点状态枚举"""
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    OFFLINE = "offline"     # 离线
    ERROR = "error"         # 错误
    STARTING = "starting"   # 启动中
    STOPPING = "stopping"   # 停止中


@dataclass
class WorkerNode:
    """工作节点数据类"""
    worker_id: str
    worker_type: WorkerType
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.worker_type, str):
            self.worker_type = WorkerType(self.worker_type)
        if isinstance(self.status, str):
            self.status = WorkerStatus(self.status)


class UnifiedWorkerRegistry:
    """
    统一工作节点注册表
    
    管理所有类型的工作节点，提供统一的注册、查询和监控功能。
    
    Attributes:
        _workers: 工作节点字典 {worker_id: WorkerNode}
        _workers_by_type: 按类型分组 {WorkerType: {worker_id}}
        _lock: 线程锁
        
    Example:
        >>> registry = UnifiedWorkerRegistry()
        >>> 
        >>> # 注册特征工作节点
        >>> registry.register_worker(
        ...     worker_id="feature_worker_1",
        ...     worker_type=WorkerType.FEATURE_WORKER,
        ...     capabilities={"cpu": 4, "memory": "8GB"}
        ... )
        >>> 
        >>> # 注册训练执行器
        >>> registry.register_worker(
        ...     worker_id="training_executor_1",
        ...     worker_type=WorkerType.TRAINING_EXECUTOR,
        ...     capabilities={"gpu": "NVIDIA A100", "cuda": "11.8"}
        ... )
        >>> 
        >>> # 获取所有训练执行器
        >>> training_executors = registry.get_workers_by_type(WorkerType.TRAINING_EXECUTOR)
    """
    
    def __init__(self):
        """初始化统一工作节点注册表"""
        self._workers: Dict[str, WorkerNode] = {}
        self._workers_by_type: Dict[WorkerType, Set[str]] = {
            worker_type: set() for worker_type in WorkerType
        }
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            "total_workers": 0,
            "active_workers": 0,
            "offline_workers": 0,
            "by_type": {worker_type.value: 0 for worker_type in WorkerType}
        }
        
        logger.info("UnifiedWorkerRegistry 初始化完成")
    
    def register_worker(
        self,
        worker_id: str,
        worker_type: WorkerType,
        capabilities: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册工作节点
        
        Args:
            worker_id: 工作节点ID
            worker_type: 工作节点类型
            capabilities: 工作能力配置
            metadata: 元数据
            
        Returns:
            是否注册成功
        """
        with self._lock:
            # 如果已存在，更新信息
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.capabilities = capabilities
                worker.last_heartbeat = datetime.now()
                worker.metadata.update(metadata or {})
                
                # 如果之前离线，现在重新上线
                if worker.status == WorkerStatus.OFFLINE:
                    worker.status = WorkerStatus.IDLE
                    self._stats["offline_workers"] -= 1
                    self._stats["active_workers"] += 1
                    logger.info(f"工作节点重新上线: {worker_id} ({worker_type.value})")
                else:
                    logger.info(f"更新工作节点: {worker_id} ({worker_type.value})")
                
                return True
            
            # 创建新节点
            worker = WorkerNode(
                worker_id=worker_id,
                worker_type=worker_type,
                status=WorkerStatus.IDLE,
                capabilities=capabilities,
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                metadata=metadata or {}
            )
            
            self._workers[worker_id] = worker
            self._workers_by_type[worker_type].add(worker_id)
            
            # 更新统计
            self._stats["total_workers"] += 1
            self._stats["active_workers"] += 1
            self._stats["by_type"][worker_type.value] += 1
            
            logger.info(f"注册工作节点: {worker_id} ({worker_type.value}), "
                       f"能力: {capabilities}")
            return True
    
    def unregister_worker(self, worker_id: str) -> bool:
        """
        注销工作节点
        
        Args:
            worker_id: 工作节点ID
            
        Returns:
            是否注销成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker_type = worker.worker_type
            
            # 更新统计
            self._stats["total_workers"] -= 1
            if worker.status != WorkerStatus.OFFLINE:
                self._stats["active_workers"] -= 1
            else:
                self._stats["offline_workers"] -= 1
            self._stats["by_type"][worker_type.value] -= 1
            
            # 从分组中移除
            self._workers_by_type[worker_type].discard(worker_id)
            
            # 删除节点
            del self._workers[worker_id]
            
            logger.info(f"注销工作节点: {worker_id} ({worker_type.value})")
            return True
    
    def update_heartbeat(self, worker_id: str) -> bool:
        """
        更新工作节点心跳
        
        Args:
            worker_id: 工作节点ID
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker.last_heartbeat = datetime.now()
            
            # 如果节点之前离线，现在重新上线
            if worker.status == WorkerStatus.OFFLINE:
                worker.status = WorkerStatus.IDLE
                self._stats["offline_workers"] -= 1
                self._stats["active_workers"] += 1
                logger.info(f"工作节点重新上线: {worker_id}")
            
            return True
    
    def update_status(self, worker_id: str, status: WorkerStatus) -> bool:
        """
        更新工作节点状态
        
        Args:
            worker_id: 工作节点ID
            status: 新状态
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            old_status = worker.status
            
            if old_status == status:
                return True
            
            worker.status = status
            
            # 更新统计
            if old_status == WorkerStatus.OFFLINE and status != WorkerStatus.OFFLINE:
                self._stats["offline_workers"] -= 1
                self._stats["active_workers"] += 1
            elif old_status != WorkerStatus.OFFLINE and status == WorkerStatus.OFFLINE:
                self._stats["active_workers"] -= 1
                self._stats["offline_workers"] += 1
            
            logger.info(f"工作节点 {worker_id} 状态更新: {old_status.value} -> {status.value}")
            return True
    
    def assign_task(self, worker_id: str, task_id: str) -> bool:
        """
        分配任务给工作节点
        
        Args:
            worker_id: 工作节点ID
            task_id: 任务ID
            
        Returns:
            是否分配成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            
            if worker.status != WorkerStatus.IDLE:
                return False
            
            worker.current_task = task_id
            worker.status = WorkerStatus.BUSY
            
            logger.info(f"为工作节点 {worker_id} 分配任务: {task_id}")
            return True
    
    def complete_task(self, worker_id: str, processing_time: float) -> bool:
        """
        完成任务
        
        Args:
            worker_id: 工作节点ID
            processing_time: 处理时间
            
        Returns:
            是否完成成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker.current_task = None
            worker.status = WorkerStatus.IDLE
            worker.completed_tasks += 1
            worker.total_processing_time += processing_time
            
            return True
    
    def fail_task(self, worker_id: str) -> bool:
        """
        任务失败
        
        Args:
            worker_id: 工作节点ID
            
        Returns:
            是否处理成功
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker.current_task = None
            worker.status = WorkerStatus.IDLE
            worker.failed_tasks += 1
            
            return True
    
    def get_worker(self, worker_id: str) -> Optional[WorkerNode]:
        """
        获取工作节点信息
        
        Args:
            worker_id: 工作节点ID
            
        Returns:
            工作节点信息
        """
        with self._lock:
            return self._workers.get(worker_id)
    
    def get_workers_by_type(self, worker_type: WorkerType) -> List[WorkerNode]:
        """
        获取指定类型的工作节点
        
        Args:
            worker_type: 工作节点类型
            
        Returns:
            工作节点列表
        """
        with self._lock:
            worker_ids = self._workers_by_type.get(worker_type, set())
            return [self._workers[wid] for wid in worker_ids if wid in self._workers]
    
    def get_available_workers(self, worker_type: Optional[WorkerType] = None) -> List[str]:
        """
        获取可用的工作节点ID列表
        
        Args:
            worker_type: 可选，指定工作节点类型
            
        Returns:
            可用工作节点ID列表
        """
        with self._lock:
            current_time = datetime.now()
            available = []
            
            workers_to_check = (
                self.get_workers_by_type(worker_type) if worker_type
                else list(self._workers.values())
            )
            
            for worker in workers_to_check:
                # 检查心跳时间（超过30秒认为离线）
                if (current_time - worker.last_heartbeat).seconds < 30:
                    if worker.status == WorkerStatus.IDLE:
                        available.append(worker.worker_id)
            
            return available
    
    def check_health(self, timeout_seconds: int = 30) -> List[str]:
        """
        检查工作节点健康状态
        
        Args:
            timeout_seconds: 超时时间（秒）
            
        Returns:
            不健康的工作节点ID列表
        """
        current_time = datetime.now()
        unhealthy_workers = []
        
        with self._lock:
            for worker_id, worker in self._workers.items():
                if (current_time - worker.last_heartbeat).seconds > timeout_seconds:
                    if worker.status != WorkerStatus.OFFLINE:
                        old_status = worker.status
                        worker.status = WorkerStatus.OFFLINE
                        
                        # 更新统计
                        if old_status != WorkerStatus.OFFLINE:
                            self._stats["active_workers"] -= 1
                            self._stats["offline_workers"] += 1
                        
                        unhealthy_workers.append(worker_id)
                        logger.warning(f"工作节点健康检查失败: {worker_id} "
                                     f"({worker.worker_type.value}), "
                                     f"上次心跳: {worker.last_heartbeat}")
        
        return unhealthy_workers
    
    def cleanup_offline_workers(self, max_offline_minutes: int = 10) -> int:
        """
        清理离线工作节点
        
        Args:
            max_offline_minutes: 最大离线时间（分钟）
            
        Returns:
            清理的工作节点数量
        """
        current_time = datetime.now()
        workers_to_remove = []
        
        with self._lock:
            for worker_id, worker in self._workers.items():
                if worker.status == WorkerStatus.OFFLINE:
                    offline_duration = (current_time - worker.last_heartbeat).seconds / 60
                    if offline_duration > max_offline_minutes:
                        workers_to_remove.append(worker_id)
            
            for worker_id in workers_to_remove:
                self.unregister_worker(worker_id)
        
        if workers_to_remove:
            logger.info(f"清理 {len(workers_to_remove)} 个离线工作节点")
        
        return len(workers_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self._stats.copy()
            
            # 计算各类型统计
            type_stats = {}
            for worker_type in WorkerType:
                workers = self.get_workers_by_type(worker_type)
                type_stats[worker_type.value] = {
                    "total": len(workers),
                    "active": sum(1 for w in workers if w.status != WorkerStatus.OFFLINE),
                    "idle": sum(1 for w in workers if w.status == WorkerStatus.IDLE),
                    "busy": sum(1 for w in workers if w.status == WorkerStatus.BUSY),
                    "offline": sum(1 for w in workers if w.status == WorkerStatus.OFFLINE)
                }
            
            stats["by_type_detailed"] = type_stats
            stats["last_updated"] = datetime.now().isoformat()
            
            return stats
    
    def get_all_workers(self) -> List[WorkerNode]:
        """
        获取所有工作节点
        
        Returns:
            工作节点列表
        """
        with self._lock:
            return list(self._workers.values())


# 全局注册表实例（单例模式）
_global_registry: Optional[UnifiedWorkerRegistry] = None


def get_unified_worker_registry() -> UnifiedWorkerRegistry:
    """
    获取全局统一工作节点注册表实例
    
    Returns:
        统一工作节点注册表实例
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = UnifiedWorkerRegistry()
    
    return _global_registry
