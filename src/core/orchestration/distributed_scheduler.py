#!/usr/bin/env python3
"""
分布式调度器
支持多集群部署和负载均衡的数据采集调度
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import json
import hashlib
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor

from src.core.orchestration.performance_optimizer import ResourceUsage
from src.core.cache.redis_cache import RedisCache


class ClusterRole(Enum):
    """集群角色"""
    LEADER = "leader"      # 主节点
    WORKER = "worker"      # 工作节点
    CANDIDATE = "candidate"  # 候选节点


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ClusterNode:
    """集群节点"""
    node_id: str
    host: str
    port: int
    role: ClusterRole = ClusterRole.WORKER
    last_heartbeat: datetime = field(default_factory=datetime.now)
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    active_tasks: int = 0
    max_concurrent_tasks: int = 5
    capabilities: Set[str] = field(default_factory=lambda: {"stock", "index"})
    is_alive: bool = True


@dataclass
class DistributedTask:
    """分布式任务"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class TaskDistributionStrategy:
    """任务分配策略"""
    strategy_name: str
    weight_cpu: float = 0.3
    weight_memory: float = 0.3
    weight_network: float = 0.2
    weight_load: float = 0.2


class ServiceDiscovery:
    """服务发现"""

    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = RedisCache(redis_config)
        self.node_key_prefix = "distributed_scheduler:nodes:"
        self.task_key_prefix = "distributed_scheduler:tasks:"
        self.leader_key = "distributed_scheduler:leader"

    async def register_node(self, node: ClusterNode):
        """注册节点"""
        key = f"{self.node_key_prefix}{node.node_id}"
        data = {
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "role": node.role.value,
            "last_heartbeat": node.last_heartbeat.isoformat(),
            "resource_usage": {
                "memory_percent": node.resource_usage.memory_percent,
                "cpu_percent": node.resource_usage.cpu_percent,
                "active_threads": node.resource_usage.active_threads,
                "active_coroutines": node.resource_usage.active_coroutines
            },
            "active_tasks": node.active_tasks,
            "max_concurrent_tasks": node.max_concurrent_tasks,
            "capabilities": list(node.capabilities),
            "is_alive": node.is_alive
        }

        await self.redis.set_json(key, data, expire_seconds=300)  # 5分钟过期

    async def unregister_node(self, node_id: str):
        """注销节点"""
        key = f"{self.node_key_prefix}{node_id}"
        await self.redis.delete(key)

    async def discover_nodes(self) -> List[ClusterNode]:
        """发现所有节点"""
        pattern = f"{self.node_key_prefix}*"
        keys = await self.redis.scan_keys(pattern)

        nodes = []
        for key in keys:
            data = await self.redis.get_json(key)
            if data:
                # 解析资源使用情况
                resource_usage = ResourceUsage(**data["resource_usage"])

                node = ClusterNode(
                    node_id=data["node_id"],
                    host=data["host"],
                    port=data["port"],
                    role=ClusterRole(data["role"]),
                    last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
                    resource_usage=resource_usage,
                    active_tasks=data["active_tasks"],
                    max_concurrent_tasks=data["max_concurrent_tasks"],
                    capabilities=set(data["capabilities"]),
                    is_alive=data["is_alive"]
                )
                nodes.append(node)

        return nodes

    async def elect_leader(self, candidate_id: str) -> bool:
        """选举主节点"""
        # 使用Redis的原子操作实现领导者选举
        result = await self.redis.set_nx(self.leader_key, candidate_id, expire_seconds=60)
        if result:
            return True

        # 检查当前领导者是否还活着
        current_leader = await self.redis.get(self.leader_key)
        if current_leader == candidate_id:
            # 续期领导者任期
            await self.redis.expire(self.leader_key, 60)
            return True

        return False

    async def get_leader(self) -> Optional[str]:
        """获取当前主节点"""
        return await self.redis.get(self.leader_key)

    async def submit_task(self, task: DistributedTask):
        """提交任务"""
        key = f"{self.task_key_prefix}{task.task_id}"
        data = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "payload": task.payload,
            "priority": task.priority.value,
            "status": task.status.value,
            "assigned_node": task.assigned_node,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "dependencies": task.dependencies,
            "error_message": task.error_message
        }

        await self.redis.set_json(key, data)

    async def update_task_status(self, task_id: str, status: TaskStatus,
                               assigned_node: Optional[str] = None,
                               error_message: Optional[str] = None):
        """更新任务状态"""
        key = f"{self.task_key_prefix}{task_id}"
        data = await self.redis.get_json(key)

        if data:
            data["status"] = status.value
            if assigned_node:
                data["assigned_node"] = assigned_node
            if error_message:
                data["error_message"] = error_message

            if status == TaskStatus.RUNNING:
                data["started_at"] = datetime.now().isoformat()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                data["completed_at"] = datetime.now().isoformat()

            await self.redis.set_json(key, data)

    async def get_pending_tasks(self) -> List[DistributedTask]:
        """获取待处理任务"""
        pattern = f"{self.task_key_prefix}*"
        keys = await self.redis.scan_keys(pattern)

        pending_tasks = []
        for key in keys:
            data = await self.redis.get_json(key)
            if data and data["status"] == TaskStatus.PENDING.value:
                task = DistributedTask(
                    task_id=data["task_id"],
                    task_type=data["task_type"],
                    payload=data["payload"],
                    priority=TaskPriority(data["priority"]),
                    status=TaskStatus(data["status"]),
                    assigned_node=data["assigned_node"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=datetime.fromisoformat(data["started_at"]) if data["started_at"] else None,
                    completed_at=datetime.fromisoformat(data["completed_at"]) if data["completed_at"] else None,
                    retry_count=data["retry_count"],
                    max_retries=data["max_retries"],
                    dependencies=data["dependencies"],
                    error_message=data["error_message"]
                )
                pending_tasks.append(task)

        return pending_tasks

    async def get_node_tasks(self, node_id: str) -> List[DistributedTask]:
        """获取节点的任务"""
        pattern = f"{self.task_key_prefix}*"
        keys = await self.redis.scan_keys(pattern)

        node_tasks = []
        for key in keys:
            data = await self.redis.get_json(key)
            if data and data["assigned_node"] == node_id:
                task = DistributedTask(
                    task_id=data["task_id"],
                    task_type=data["task_type"],
                    payload=data["payload"],
                    priority=TaskPriority(data["priority"]),
                    status=TaskStatus(data["status"]),
                    assigned_node=data["assigned_node"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    retry_count=data["retry_count"],
                    max_retries=data["max_retries"],
                    dependencies=data["dependencies"]
                )
                node_tasks.append(task)

        return node_tasks


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, strategy: TaskDistributionStrategy):
        self.strategy = strategy

    def select_best_node(self, task: DistributedTask, available_nodes: List[ClusterNode]) -> Optional[ClusterNode]:
        """
        选择最佳节点执行任务

        基于多维度指标进行智能选择：
        1. CPU使用率
        2. 内存使用率
        3. 网络负载
        4. 当前任务负载
        5. 节点能力匹配
        """
        if not available_nodes:
            return None

        # 过滤有能力的节点
        capable_nodes = [
            node for node in available_nodes
            if self._node_has_capability(node, task) and self._node_has_capacity(node)
        ]

        if not capable_nodes:
            return None

        # 计算每个节点的评分
        node_scores = []
        for node in capable_nodes:
            score = self._calculate_node_score(node)
            node_scores.append((node, score))

        # 选择评分最高的节点
        best_node = max(node_scores, key=lambda x: x[1])[0]
        return best_node

    def _node_has_capability(self, node: ClusterNode, task: DistributedTask) -> bool:
        """检查节点是否有执行任务的能力"""
        # 检查数据类型能力
        data_type = task.payload.get("data_type", "stock")
        return data_type in node.capabilities

    def _node_has_capacity(self, node: ClusterNode) -> bool:
        """检查节点是否有容量执行新任务"""
        return node.active_tasks < node.max_concurrent_tasks and node.is_alive

    def _calculate_node_score(self, node: ClusterNode) -> float:
        """计算节点评分（越高越好）"""
        # CPU评分（CPU使用率越低越好）
        cpu_score = max(0, 100 - node.resource_usage.cpu_percent) / 100

        # 内存评分（内存使用率越低越好）
        memory_score = max(0, 100 - node.resource_usage.memory_percent) / 100

        # 负载评分（活跃任务越少越好）
        load_score = max(0, 1 - (node.active_tasks / node.max_concurrent_tasks))

        # 网络评分（暂时使用固定值，实际可基于网络监控数据）
        network_score = 0.8

        # 综合评分
        total_score = (
            self.strategy.weight_cpu * cpu_score +
            self.strategy.weight_memory * memory_score +
            self.strategy.weight_load * load_score +
            self.strategy.weight_network * network_score
        )

        return total_score


class FailureHandler:
    """故障处理器"""

    def __init__(self, max_retry_attempts: int = 3):
        self.max_retry_attempts = max_retry_attempts

    async def handle_task_failure(self, task: DistributedTask, error: Exception) -> Tuple[bool, Optional[datetime]]:
        """
        处理任务失败

        Returns:
            (should_retry, next_retry_time)
        """
        task.retry_count += 1
        task.error_message = str(error)

        if task.retry_count < task.max_retries:
            # 计算下次重试时间（指数退避）
            delay_seconds = 2 ** task.retry_count
            next_retry_time = datetime.now() + timedelta(seconds=delay_seconds)
            return True, next_retry_time

        return False, None

    async def handle_node_failure(self, node: ClusterNode) -> List[DistributedTask]:
        """
        处理节点故障

        Returns:
            需要重新分配的任务列表
        """
        # 这里应该从服务发现中获取节点的任务
        # 暂时返回空列表，实际实现需要查询任务状态
        return []


class DistributedScheduler:
    """分布式调度器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.service_discovery = ServiceDiscovery(config.get('redis_config', {}))
        self.load_balancer = LoadBalancer(
            TaskDistributionStrategy(
                strategy_name="balanced",
                weight_cpu=0.3,
                weight_memory=0.3,
                weight_network=0.2,
                weight_load=0.2
            )
        )
        self.failure_handler = FailureHandler(config.get('max_retry_attempts', 3))

        # 节点信息
        self.node_id = config.get('node_id', str(uuid.uuid4()))
        self.host = config.get('host', socket.gethostname())
        self.port = config.get('port', 8080)
        self.role = ClusterRole.WORKER
        self.capabilities = set(config.get('capabilities', ["stock", "index"]))
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 5)

        # 任务队列
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, DistributedTask] = {}

        # 控制标志
        self.running = False
        self.leader_election_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.task_dispatcher_task: Optional[asyncio.Task] = None
        self.task_monitor_task: Optional[asyncio.Task] = None

        # 线程池用于CPU密集型任务
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)

    async def start(self):
        """启动分布式调度器"""
        self.running = True
        self.logger.info(f"启动分布式调度器: {self.node_id} ({self.host}:{self.port})")

        # 注册节点
        await self._register_node()

        # 启动后台任务
        self.leader_election_task = asyncio.create_task(self._leader_election_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.task_dispatcher_task = asyncio.create_task(self._task_dispatcher_loop())
        self.task_monitor_task = asyncio.create_task(self._task_monitor_loop())

        self.logger.info("分布式调度器启动完成")

    async def stop(self):
        """停止分布式调度器"""
        self.running = False
        self.logger.info("正在停止分布式调度器...")

        # 取消所有任务
        tasks_to_cancel = [
            self.leader_election_task,
            self.heartbeat_task,
            self.task_dispatcher_task,
            self.task_monitor_task
        ]

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()

        # 等待任务完成
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # 注销节点
        await self.service_discovery.unregister_node(self.node_id)

        # 关闭线程池
        self.executor.shutdown(wait=True)

        self.logger.info("分布式调度器已停止")

    async def submit_task(self, task_type: str, payload: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         dependencies: List[str] = None) -> str:
        """提交任务"""
        task_id = f"task_{int(datetime.now().timestamp() * 1000000)}_{hashlib.md5(str(payload).encode()).hexdigest()[:8]}"

        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            dependencies=dependencies or []
        )

        # 保存到服务发现
        await self.service_discovery.submit_task(task)

        # 如果是主节点，直接加入队列；否则等待其他节点处理
        if self.role == ClusterRole.LEADER:
            await self.task_queue.put(task)

        self.logger.info(f"任务已提交: {task_id} ({task_type})")
        return task_id

    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """获取任务状态"""
        # 从服务发现中查询任务
        # 这里简化实现，实际应该从Redis查询
        return self.active_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            # 更新服务发现
            await self.service_discovery.update_task_status(task_id, TaskStatus.CANCELLED)

            del self.active_tasks[task_id]
            self.logger.info(f"任务已取消: {task_id}")
            return True

        return False

    async def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        nodes = await self.service_discovery.discover_nodes()
        pending_tasks = await self.service_discovery.get_pending_tasks()
        leader = await self.service_discovery.get_leader()

        return {
            "leader": leader,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "role": node.role.value,
                    "host": node.host,
                    "port": node.port,
                    "is_alive": node.is_alive,
                    "active_tasks": node.active_tasks,
                    "resource_usage": {
                        "cpu_percent": node.resource_usage.cpu_percent,
                        "memory_percent": node.resource_usage.memory_percent
                    }
                } for node in nodes
            ],
            "pending_tasks": len(pending_tasks),
            "total_active_tasks": sum(node.active_tasks for node in nodes),
            "timestamp": datetime.now().isoformat()
        }

    async def _register_node(self):
        """注册节点"""
        self.node = ClusterNode(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            role=self.role,
            capabilities=self.capabilities,
            max_concurrent_tasks=self.max_concurrent_tasks
        )

        await self.service_discovery.register_node(self.node)
        self.logger.info(f"节点已注册: {self.node_id}")

    async def _leader_election_loop(self):
        """领导者选举循环"""
        while self.running:
            try:
                if self.role != ClusterRole.LEADER:
                    # 尝试成为领导者
                    if await self.service_discovery.elect_leader(self.node_id):
                        self.role = ClusterRole.LEADER
                        self.node.role = ClusterRole.LEADER
                        await self.service_discovery.register_node(self.node)
                        self.logger.info(f"成为集群领导者: {self.node_id}")

                await asyncio.sleep(30)  # 每30秒检查一次

            except Exception as e:
                self.logger.error(f"领导者选举异常: {e}")
                await asyncio.sleep(30)

    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                # 更新节点状态
                await self._update_node_status()

                # 重新注册节点（续期）
                await self.service_discovery.register_node(self.node)

                await asyncio.sleep(10)  # 每10秒发送心跳

            except Exception as e:
                self.logger.error(f"心跳异常: {e}")
                await asyncio.sleep(10)

    async def _update_node_status(self):
        """更新节点状态"""
        # 这里应该获取实际的资源使用情况
        # 暂时使用模拟数据
        import psutil

        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()

        self.node.resource_usage.memory_percent = memory.percent
        self.node.resource_usage.cpu_percent = cpu
        self.node.resource_usage.active_threads = threading.active_count()
        self.node.resource_usage.active_coroutines = len(asyncio.all_tasks())
        self.node.active_tasks = len(self.active_tasks)
        self.node.last_heartbeat = datetime.now()

    async def _task_dispatcher_loop(self):
        """任务分发循环"""
        while self.running:
            try:
                if self.role == ClusterRole.LEADER:
                    # 获取待处理任务
                    pending_tasks = await self.service_discovery.get_pending_tasks()

                    for task in pending_tasks:
                        # 检查依赖是否满足
                        if await self._check_dependencies(task):
                            # 选择最佳节点
                            available_nodes = await self.service_discovery.discover_nodes()
                            best_node = self.load_balancer.select_best_node(task, available_nodes)

                            if best_node:
                                # 分配任务
                                await self._assign_task_to_node(task, best_node)
                            else:
                                self.logger.warning(f"没有可用的节点执行任务: {task.task_id}")

                await asyncio.sleep(5)  # 每5秒分发一次

            except Exception as e:
                self.logger.error(f"任务分发异常: {e}")
                await asyncio.sleep(5)

    async def _check_dependencies(self, task: DistributedTask) -> bool:
        """检查任务依赖"""
        if not task.dependencies:
            return True

        # 检查依赖任务是否已完成
        # 这里简化实现，实际应该查询依赖任务状态
        return True

    async def _assign_task_to_node(self, task: DistributedTask, node: ClusterNode):
        """分配任务到节点"""
        task.status = TaskStatus.ASSIGNED
        task.assigned_node = node.node_id

        # 更新服务发现
        await self.service_discovery.update_task_status(
            task.task_id, TaskStatus.ASSIGNED, node.node_id
        )

        self.logger.info(f"任务已分配: {task.task_id} -> {node.node_id}")

    async def _task_monitor_loop(self):
        """任务监控循环"""
        while self.running:
            try:
                # 检查超时任务
                await self._check_timeout_tasks()

                # 检查失败任务的重试
                await self._check_failed_tasks()

                # 检查节点故障
                await self._check_node_failures()

                await asyncio.sleep(30)  # 每30秒检查一次

            except Exception as e:
                self.logger.error(f"任务监控异常: {e}")
                await asyncio.sleep(30)

    async def _check_timeout_tasks(self):
        """检查超时任务"""
        timeout_threshold = timedelta(minutes=30)  # 30分钟超时

        for task_id, task in self.active_tasks.items():
            if task.started_at and (datetime.now() - task.started_at) > timeout_threshold:
                self.logger.warning(f"任务超时: {task_id}")

                # 标记为失败并重试
                task.status = TaskStatus.FAILED
                task.error_message = "Task timeout"

                should_retry, next_retry = await self.failure_handler.handle_task_failure(
                    task, Exception("Timeout")
                )

                if should_retry and next_retry:
                    # 重新提交任务
                    await self.service_discovery.update_task_status(task_id, TaskStatus.PENDING)
                    await self.task_queue.put(task)
                else:
                    await self.service_discovery.update_task_status(task_id, TaskStatus.FAILED)

    async def _check_failed_tasks(self):
        """检查失败任务"""
        # 这里应该从服务发现中获取失败任务
        # 暂时跳过，实际实现需要查询失败任务列表
        pass

    async def _check_node_failures(self):
        """检查节点故障"""
        nodes = await self.service_discovery.discover_nodes()

        for node in nodes:
            # 检查心跳超时
            if (datetime.now() - node.last_heartbeat) > timedelta(seconds=60):
                self.logger.warning(f"节点故障检测: {node.node_id}")

                # 处理节点故障
                failed_tasks = await self.failure_handler.handle_node_failure(node)

                # 重新分配失败任务
                for task in failed_tasks:
                    task.status = TaskStatus.PENDING
                    task.assigned_node = None
                    await self.service_discovery.update_task_status(task.task_id, TaskStatus.PENDING)
                    await self.task_queue.put(task)

    async def execute_task(self, task: DistributedTask):
        """执行任务"""
        self.active_tasks[task.task_id] = task
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        await self.service_discovery.update_task_status(task.task_id, TaskStatus.RUNNING)

        try:
            self.logger.info(f"开始执行任务: {task.task_id}")

            # 根据任务类型执行相应逻辑
            if task.task_type == "data_collection":
                result = await self._execute_data_collection_task(task)
            elif task.task_type == "data_quality_check":
                result = await self._execute_quality_check_task(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            await self.service_discovery.update_task_status(task.task_id, TaskStatus.COMPLETED)

            self.logger.info(f"任务执行完成: {task.task_id}")

        except Exception as e:
            self.logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")

            task.status = TaskStatus.FAILED
            task.error_message = str(e)

            # 处理失败
            should_retry, next_retry = await self.failure_handler.handle_task_failure(task, e)

            if should_retry and next_retry:
                task.status = TaskStatus.PENDING
                task.retry_count += 1
                await self.service_discovery.update_task_status(task.task_id, TaskStatus.PENDING)
                # 重新加入队列的逻辑可以在这里添加
            else:
                await self.service_discovery.update_task_status(task.task_id, TaskStatus.FAILED)

        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def _execute_data_collection_task(self, task: DistributedTask) -> Any:
        """执行数据采集任务"""
        # 这里应该调用实际的数据采集服务
        # 暂时返回模拟结果
        payload = task.payload

        # 模拟采集延迟
        await asyncio.sleep(2)

        return {
            "symbol": payload.get("symbol"),
            "data_type": payload.get("data_type"),
            "records_collected": 1000,
            "quality_score": 0.92
        }

    async def _execute_quality_check_task(self, task: DistributedTask) -> Any:
        """执行质量检查任务"""
        # 这里应该调用质量检查服务
        # 暂时返回模拟结果
        payload = task.payload

        await asyncio.sleep(1)

        return {
            "data_id": payload.get("data_id"),
            "quality_score": 0.88,
            "issues_found": 2,
            "recommendations": ["数据完整性良好", "建议进一步验证"]
        }