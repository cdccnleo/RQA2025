#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分布式策略管理器
Distributed Strategy Manager

支持多节点部署的分布式策略执行和管理。
"""

import time
import asyncio
import threading
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import requests

from ..interfaces.strategy_interfaces import StrategyConfig
from ..core.strategy_service import UnifiedStrategyService
from ..persistence.strategy_persistence import StrategyPersistence

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:

    """节点信息"""
    node_id: str
    host: str
    port: int
    status: str = "active"  # active, inactive, maintenance
    capabilities: List[str] = field(default_factory=list)
    load_factor: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    strategy_count: int = 0


@dataclass
class DistributedTask:

    """分布式任务"""
    task_id: str
    strategy_id: str
    node_id: str
    task_type: str  # execute, backtest, optimize
    status: str = "pending"  # pending, running, completed, failed
    priority: int = 1
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class LoadBalancingMetrics:

    """负载均衡指标"""
    node_id: str
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    queue_length: int
    response_time: float
    last_updated: datetime = field(default_factory=datetime.now)


class DistributedStrategyManager:

    """分布式策略管理器"""

    def __init__(self, node_id: str = None, host: str = "localhost", port: int = 8080):

        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port

        # 节点管理
        self.nodes: Dict[str, NodeInfo] = {}
        self.current_node = NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            capabilities=["strategy_execution", "backtest", "optimization"]
        )

        # 任务管理
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue = asyncio.Queue()
        self.task_executor = ThreadPoolExecutor(max_workers=10)

        # 负载均衡
        self.load_metrics: Dict[str, LoadBalancingMetrics] = {}
        self.load_balancer = LoadBalancer(self)

        # 服务组件
        self.strategy_service = UnifiedStrategyService()
        self.persistence = StrategyPersistence()

        # 网络通信
        self.http_client = requests.Session()
        self.http_client.timeout = 30

        # 心跳和监控
        self.heartbeat_interval = 30  # 秒
        self.heartbeat_thread = None
        self.monitoring_thread = None

        # 故障转移
        self.failover_manager = FailoverManager(self)

        logger.info(f"DistributedStrategyManager initialized: {self.node_id}")

    async def start(self):
        """启动分布式管理器"""
        logger.info("Starting DistributedStrategyManager...")

        # 注册当前节点
        self.nodes[self.node_id] = self.current_node

        # 启动心跳线程
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # 启动故障转移管理器
        await self.failover_manager.start()

        # 启动任务处理循环
        asyncio.create_task(self._task_processing_loop())

        logger.info("DistributedStrategyManager started successfully")

    async def stop(self):
        """停止分布式管理器"""
        logger.info("Stopping DistributedStrategyManager...")

        # 停止心跳和监控
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        # 停止故障转移管理器
        await self.failover_manager.stop()

        # 关闭线程池
        self.task_executor.shutdown(wait=True)

        logger.info("DistributedStrategyManager stopped")

    def register_node(self, node_info: NodeInfo):
        """注册新节点"""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"Node registered: {node_info.node_id} at {node_info.host}:{node_info.port}")

    def unregister_node(self, node_id: str):
        """注销节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node unregistered: {node_id}")

    async def execute_strategy_distributed(self, config: StrategyConfig,
                                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """分布式执行策略"""
        # 选择最佳节点
        target_node = self.load_balancer.select_best_node(config)

        if target_node.node_id == self.node_id:
            # 本地执行
            return await self._execute_locally(config, market_data)
        else:
            # 远程执行
            return await self._execute_remotely(target_node, config, market_data)

    async def submit_distributed_task(self, strategy_id: str, task_type: str,
                                      data: Dict[str, Any], priority: int = 1) -> str:
        """提交分布式任务"""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            strategy_id=strategy_id,
            node_id="",  # 将由负载均衡器分配
            task_type=task_type,
            priority=priority,
            data=data
        )

        # 分配节点
        target_node = self.load_balancer.select_best_node_for_task(task)
        task.node_id = target_node.node_id

        # 保存任务
        self.tasks[task.task_id] = task

        # 提交到队列
        await self.task_queue.put(task)

        logger.info(f"Distributed task submitted: {task.task_id} to node {task.node_id}")
        return task.task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            'task_id': task.task_id,
            'status': task.status,
            'node_id': task.node_id,
            'progress': self._calculate_task_progress(task),
            'result': task.result,
            'error_message': task.error_message,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        }

    async def _execute_locally(self, config: StrategyConfig,
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """本地执行策略"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.task_executor,
                self.strategy_service.execute_strategy,
                config.strategy_id,
                market_data
            )
            return result
        except Exception as e:
            logger.error(f"Local strategy execution failed: {e}")
            raise

    async def _execute_remotely(self, target_node: NodeInfo, config: StrategyConfig,
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """远程执行策略"""
        try:
            url = f"http://{target_node.host}:{target_node.port}/api / strategy / execute"

            payload = {
                'config': {
                    'strategy_id': config.strategy_id,
                    'strategy_name': config.strategy_name,
                    'strategy_type': config.strategy_type.value,
                    'parameters': config.parameters,
                    'risk_limits': config.risk_limits,
                    'market_data_sources': config.market_data_sources
                },
                'market_data': market_data
            }

            response = await asyncio.get_event_loop().run_in_executor(
                self.task_executor,
                self.http_client.post,
                url,
                json=payload
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Remote strategy execution failed: {e}")
            # 尝试故障转移
            await self.failover_manager.handle_node_failure(target_node.node_id)
            raise

    async def _task_processing_loop(self):
        """任务处理循环"""
        while True:
            try:
                # 获取任务
                task = await self.task_queue.get()

                # 更新任务状态
                task.status = "running"
                task.started_at = datetime.now()

                # 执行任务
                try:
                    if task.node_id == self.node_id:
                        # 本地执行
                        result = await self._execute_task_locally(task)
                    else:
                        # 转发到远程节点
                        result = await self._execute_task_remotely(task)

                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.now()

                except Exception as e:
                    task.status = "failed"
                    task.error_message = str(e)
                    task.completed_at = datetime.now()
                    logger.error(f"Task execution failed: {task.task_id}, error: {e}")

                # 通知任务完成
                self._notify_task_completion(task)

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"Task processing loop error: {e}")
                await asyncio.sleep(1)

    async def _execute_task_locally(self, task: DistributedTask) -> Dict[str, Any]:
        """本地执行任务"""
        if task.task_type == "execute":
            config = StrategyConfig(**task.data['config'])
            market_data = task.data['market_data']
            return await self._execute_locally(config, market_data)
        elif task.task_type == "backtest":
            # 实现回测任务
            return await self._execute_backtest_task(task)
        elif task.task_type == "optimize":
            # 实现优化任务
            return await self._execute_optimization_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    async def _execute_task_remotely(self, task: DistributedTask) -> Dict[str, Any]:
        """远程执行任务"""
        target_node = self.nodes.get(task.node_id)
        if not target_node:
            raise ValueError(f"Target node not found: {task.node_id}")

        try:
            url = f"http://{target_node.host}:{target_node.port}/api / task / execute"
            response = await asyncio.get_event_loop().run_in_executor(
                self.task_executor,
                self.http_client.post,
                url,
                json={
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'data': task.data
                }
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Remote task execution failed: {e}")
            raise

    async def _execute_backtest_task(self, task: DistributedTask) -> Dict[str, Any]:
        """执行回测任务"""
        # 这里应该调用实际的回测引擎
        # 暂时返回模拟结果
        await asyncio.sleep(1)  # 模拟处理时间
        return {
            'task_type': 'backtest',
            'status': 'completed',
            'result': {
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'total_return': 0.35
            }
        }

    async def _execute_optimization_task(self, task: DistributedTask) -> Dict[str, Any]:
        """执行优化任务"""
        # 这里应该调用实际的优化引擎
        # 暂时返回模拟结果
        await asyncio.sleep(2)  # 模拟处理时间
        return {
            'task_type': 'optimize',
            'status': 'completed',
            'result': {
                'best_params': {'lookback_period': 20, 'threshold': 0.05},
                'best_score': 1.5,
                'iterations': 25
            }
        }

    def _calculate_task_progress(self, task: DistributedTask) -> float:
        """计算任务进度"""
        if task.status == "completed":
            return 1.0
        elif task.status == "running":
            # 基于任务类型估算进度
            if task.task_type == "backtest":
                return 0.6
            elif task.task_type == "optimize":
                return 0.4
            else:
                return 0.5
        else:
            return 0.0

    def _notify_task_completion(self, task: DistributedTask):
        """通知任务完成"""
        # 这里可以实现任务完成通知机制
        logger.info(f"Task completed: {task.task_id}, status: {task.status}")

    def _heartbeat_loop(self):
        """心跳循环"""
        while True:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5)

    def _send_heartbeat(self):
        """发送心跳"""
        # 更新当前节点状态
        self.current_node.last_heartbeat = datetime.now()
        self.current_node.load_factor = self._calculate_load_factor()
        self.current_node.strategy_count = len([t for t in self.tasks.values()
                                                if t.node_id == self.node_id and t.status == "running"])

        # 向其他节点发送心跳
        for node in self.nodes.values():
            if node.node_id != self.node_id:
                try:
                    url = f"http://{node.host}:{node.port}/api / heartbeat"
                    self.http_client.post(url, json={
                        'node_id': self.node_id,
                        'status': 'alive',
                        'timestamp': datetime.now().isoformat()
                    }, timeout=5)
                except Exception as e:
                    logger.debug(f"Failed to send heartbeat to {node.node_id}: {e}")

    def _calculate_load_factor(self) -> float:
        """计算负载因子"""
        # 简单的负载计算
        active_tasks = len([t for t in self.tasks.values()
                            if t.node_id == self.node_id and t.status == "running"])
        queue_length = self.task_queue.qsize()

        load_factor = min((active_tasks + queue_length * 0.5) / 10, 1.0)
        return load_factor

    def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                self._update_load_metrics()
                time.sleep(10)  # 每10秒更新一次监控指标
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _update_load_metrics(self):
        """更新负载指标"""
        # 这里应该收集实际的系统指标
        # 暂时使用模拟数据
        metrics = LoadBalancingMetrics(
            node_id=self.node_id,
            cpu_usage=0.3,  # 30%
            memory_usage=0.5,  # 50%
            active_tasks=len([t for t in self.tasks.values()
                              if t.node_id == self.node_id and t.status == "running"]),
            queue_length=self.task_queue.qsize(),
            response_time=0.01  # 10ms
        )

        self.load_metrics[self.node_id] = metrics


class LoadBalancer:

    """负载均衡器"""

    def __init__(self, manager: DistributedStrategyManager):

        self.manager = manager
        self.balancing_strategy = "least_loaded"  # least_loaded, round_robin, weighted

    def select_best_node(self, config: StrategyConfig) -> NodeInfo:
        """选择最佳节点"""
        if self.balancing_strategy == "least_loaded":
            return self._select_least_loaded_node()
        elif self.balancing_strategy == "weighted":
            return self._select_weighted_node(config)
        else:
            return self._select_round_robin_node()

    def select_best_node_for_task(self, task: DistributedTask) -> NodeInfo:
        """为任务选择最佳节点"""
        available_nodes = [node for node in self.manager.nodes.values()
                           if node.status == "active"
                           and self._node_supports_task(node, task)]

        if not available_nodes:
            raise ValueError("No available nodes for task execution")

        # 选择负载最小的节点
        return min(available_nodes, key=lambda n: n.load_factor)

    def _select_least_loaded_node(self) -> NodeInfo:
        """选择负载最小的节点"""
        active_nodes = [node for node in self.manager.nodes.values()
                        if node.status == "active"]

        if not active_nodes:
            raise ValueError("No active nodes available")

        return min(active_nodes, key=lambda n: n.load_factor)

    def _select_weighted_node(self, config: StrategyConfig) -> NodeInfo:
        """基于权重选择节点"""
        active_nodes = [node for node in self.manager.nodes.values()
                        if node.status == "active"]

        if not active_nodes:
            raise ValueError("No active nodes available")

        # 计算权重（基于节点能力和当前负载）
        node_weights = {}
        for node in active_nodes:
            base_weight = len(node.capabilities)  # 能力数量作为基础权重
            load_penalty = node.load_factor  # 负载惩罚
            node_weights[node.node_id] = base_weight * (1 - load_penalty)

        # 选择权重最高的节点
        best_node_id = max(node_weights, key=node_weights.get)
        return self.manager.nodes[best_node_id]

    def _select_round_robin_node(self) -> NodeInfo:
        """轮询选择节点"""
        active_nodes = [node for node in self.manager.nodes.values()
                        if node.status == "active"]

        if not active_nodes:
            raise ValueError("No active nodes available")

        # 简单的轮询实现
        current_time = int(time.time())
        node_index = current_time % len(active_nodes)
        return active_nodes[node_index]

    def _node_supports_task(self, node: NodeInfo, task: DistributedTask) -> bool:
        """检查节点是否支持任务"""
        # 基于任务类型检查节点能力
        if task.task_type == "execute":
            return "strategy_execution" in node.capabilities
        elif task.task_type == "backtest":
            return "backtest" in node.capabilities
        elif task.task_type == "optimize":
            return "optimization" in node.capabilities
        else:
            return True  # 默认支持


class FailoverManager:

    """故障转移管理器"""

    def __init__(self, manager: DistributedStrategyManager):

        self.manager = manager
        self.failed_nodes: Dict[str, datetime] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.recovery_interval = 300  # 5分钟

    async def start(self):
        """启动故障转移管理器"""
        logger.info("FailoverManager started")

    async def stop(self):
        """停止故障转移管理器"""
        logger.info("FailoverManager stopped")

    async def handle_node_failure(self, node_id: str):
        """处理节点故障"""
        logger.warning(f"Node failure detected: {node_id}")

        # 记录故障时间
        self.failed_nodes[node_id] = datetime.now()

        # 增加恢复尝试次数
        self.recovery_attempts[node_id] = self.recovery_attempts.get(node_id, 0) + 1

        # 标记节点为非活跃状态
        if node_id in self.manager.nodes:
            self.manager.nodes[node_id].status = "inactive"

        # 重新分配任务
        await self._redistribute_tasks(node_id)

        # 尝试恢复节点
        if self.recovery_attempts[node_id] <= self.max_recovery_attempts:
            asyncio.create_task(self._attempt_node_recovery(node_id))
        else:
            logger.error(f"Max recovery attempts reached for node {node_id}")

    async def _redistribute_tasks(self, failed_node_id: str):
        """重新分配任务"""
        failed_tasks = [task for task in self.manager.tasks.values()
                        if task.node_id == failed_node_id and task.status in ["pending", "running"]]

        for task in failed_tasks:
            try:
                # 选择新的节点
                new_node = self.manager.load_balancer.select_best_node_for_task(task)
                task.node_id = new_node.node_id

                # 重新提交任务
                await self.manager.task_queue.put(task)

                logger.info(f"Task {task.task_id} redistributed to node {new_node.node_id}")

            except Exception as e:
                logger.error(f"Failed to redistribute task {task.task_id}: {e}")

    async def _attempt_node_recovery(self, node_id: str):
        """尝试恢复节点"""
        await asyncio.sleep(self.recovery_interval)

        node = self.manager.nodes.get(node_id)
        if not node:
            return

        try:
            # 尝试连接节点
            url = f"http://{node.host}:{node.port}/api / health"
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.manager.http_client.get, url, timeout=10
            )

            if response.status_code == 200:
                # 节点恢复成功
                node.status = "active"
                node.last_heartbeat = datetime.now()
                self.recovery_attempts[node_id] = 0

                logger.info(f"Node {node_id} recovered successfully")
            else:
                # 节点仍不可用
                logger.warning(f"Node {node_id} still unavailable")

        except Exception as e:
            logger.warning(f"Node recovery attempt failed for {node_id}: {e}")


# 全局实例
_distributed_manager = None


def get_distributed_strategy_manager() -> DistributedStrategyManager:
    """获取分布式策略管理器实例"""
    global _distributed_manager
    if _distributed_manager is None:
        _distributed_manager = DistributedStrategyManager()
    return _distributed_manager
