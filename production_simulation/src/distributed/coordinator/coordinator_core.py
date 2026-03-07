"""
分布式协调器核心

主协调器类实现，负责整体协调管理。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import threading
import time
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from .models import (
    NodeInfo, DistributedTask, ClusterStats,
    NodeStatus, TaskStatus, TaskPriority
)
from .cluster_manager import ClusterManager
from .task_manager import TaskManager

# 获取统一基础设施 - 延迟导入避免循环依赖
try:
    from ...core.integration import get_event_bus, get_service_container
    event_bus = get_event_bus()
    service_container = get_service_container()
except ImportError:
    # 如果函数不存在，使用默认值
    event_bus = None
    service_container = None

logger = logging.getLogger(__name__)


class DistributedCoordinator:
    """
    分布式协调器

    主要功能：
    1. 节点管理和监控
    2. 任务调度和分配
    3. 负载均衡
    4. 故障恢复
    5. 资源调度
    6. 性能监控
    """

    def __init__(self):
        # 管理器组件
        self.cluster_manager = ClusterManager()
        self.task_manager = TaskManager()

        # 锁
        self._lock = threading.RLock()

        # 监控线程
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._cluster_monitor, daemon=True)
        self._monitor_thread.start()

        # 故障恢复线程
        self._recovery_thread = threading.Thread(target=self._failure_recovery, daemon=True)
        self._recovery_thread.start()

        logger.info("分布式协调器初始化完成")

    def register_node(self, node_info: NodeInfo) -> bool:
        """注册节点"""
        result = self.cluster_manager.register_node(node_info)
        
        if result:
            # 发布节点注册事件
            if event_bus:
                event_bus.publish_sync({
                    'event_type': 'node_registered',
                    'node_id': node_info.node_id,
                    'node_info': {
                        'hostname': node_info.hostname,
                        'ip_address': node_info.ip_address,
                        'cpu_cores': node_info.cpu_cores,
                        'memory_gb': node_info.memory_gb,
                        'gpu_devices': node_info.gpu_devices
                    }
                })
        
        return result

    def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        # 重新分配该节点上的任务
        self._reassign_node_tasks(node_id)
        
        result = self.cluster_manager.unregister_node(node_id)
        
        if result and event_bus:
            event_bus.publish_sync({
                'event_type': 'node_unregistered',
                'node_id': node_id
            })
        
        return result

    def submit_task(self, task_type: str, data: Dict[str, Any],
                    priority: TaskPriority = TaskPriority.NORMAL,
                    timeout_seconds: int = 3600) -> str:
        """提交任务"""
        task_id = self.task_manager.submit_task(task_type, data, priority, timeout_seconds)
        
        if task_id:
            # 尝试立即调度任务
            self._schedule_pending_tasks()
        
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.task_manager.cancel_task(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.task_manager.get_task_status(task_id)

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        cluster_status = self.cluster_manager.get_cluster_status()
        task_stats = self.task_manager.get_task_stats()
        
        return {
            **cluster_status,
            **task_stats,
                'stats': {
                    'total_nodes': self.stats.total_nodes,
                    'online_nodes': self.stats.online_nodes,
                    'total_tasks': self.stats.total_tasks,
                    'running_tasks': self.stats.running_tasks,
                    'completed_tasks': self.stats.completed_tasks,
                    'failed_tasks': self.stats.failed_tasks,
                    'avg_load_factor': self.stats.avg_load_factor,
                    'total_cpu_cores': self.stats.total_cpu_cores,
                    'total_memory_gb': self.stats.total_memory_gb,
                    'total_gpu_devices': self.stats.total_gpu_devices
                },
                'nodes': {
                    node_id: {
                        'hostname': node.hostname,
                        'status': node.status.value,
                        'cpu_cores': node.cpu_cores,
                        'memory_gb': node.memory_gb,
                        'gpu_devices': len(node.gpu_devices),
                        'load_factor': node.load_factor,
                        'active_tasks': len(node.active_tasks)
                    }
                    for node_id, node in self.nodes.items()
                },
                'tasks': {
                    task_id: self.get_task_status(task_id)
                    for task_id in list(self.tasks.keys())[:10]  # 只返回最近10个任务
                }
            }

    def _schedule_pending_tasks(self):
        """调度待处理任务"""
        try:
            # 获取可用的在线节点
            available_nodes = {
                node_id: node for node_id, node in self.nodes.items()
                if node.status == NodeStatus.ONLINE
            }

            if not available_nodes:
                logger.warning("没有可用的在线节点")
                return

            # 应用优先级老化
            self.priority_engine.check_aging_tasks(list(self.tasks.values()))

            # 处理待调度任务
            scheduled_count = 0
            for task_id in self.task_queue[:]:
                if scheduled_count >= 10:  # 每次最多调度10个任务
                    break

                if task_id in self.tasks:
                    task = self.tasks[task_id]

                    if task.status == TaskStatus.PENDING:
                        # 使用调度引擎选择执行节点
                        assigned_node = self.scheduling_engine.schedule_task(
                            task, available_nodes, self.task_queue)

                        if assigned_node:
                            # 检查是否需要抢占
                            preemption_needed = self._check_preemption_needed(task, assigned_node)

                            if preemption_needed:
                                # 执行抢占
                                if not self._execute_preemption(task, assigned_node):
                                    logger.warning(f"任务 {task_id} 抢占失败，跳过调度")
                                    continue

                            # 分配任务到节点
                            if self._assign_task_to_node(task, assigned_node):
                                # 从队列移除
                                if task_id in self.task_queue:
                                    self.task_queue.remove(task_id)
                                scheduled_count += 1
                                logger.info(f"任务 {task_id} 已分配到节点 {assigned_node}")
                            else:
                                logger.warning(f"任务 {task_id} 分配到节点 {assigned_node} 失败")
                        else:
                            logger.debug(f"任务 {task_id} 未能找到合适的执行节点")

        except Exception as e:
            logger.error(f"任务调度异常: {e}")

    def _check_preemption_needed(self, task: DistributedTask, node_id: str) -> bool:
        """检查是否需要抢占节点上的任务"""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        # 如果节点没有达到最大并发限制，不需要抢占
        if len(node.active_tasks) < node.cpu_cores * 2:  # 假设每个CPU核心可运行2个任务
            return False

        # 检查是否有可被抢占的任务
        running_tasks = {
            task_id: self.tasks[task_id]
            for task_id in node.active_tasks
            if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING
        }

        preemption_candidates = self.priority_engine.get_preemption_candidates(task, running_tasks)
        return len(preemption_candidates) > 0

    def _execute_preemption(self, task: DistributedTask, node_id: str) -> bool:
        """执行任务抢占"""
        try:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]

            # 获取运行中的任务
            running_tasks = {
                task_id: self.tasks[task_id]
                for task_id in node.active_tasks
                if task_id in self.tasks and self.tasks[task_id].status == TaskStatus.RUNNING
            }

            # 获取可抢占的任务
            preemption_candidates = self.priority_engine.get_preemption_candidates(
                task, running_tasks)

            if not preemption_candidates:
                return False

            # 抢占优先级最低的任务
            task_to_preempt = min(
                preemption_candidates,
                key=lambda tid: running_tasks[tid].priority.value
            )

            # 取消被抢占的任务
            success = self._notify_node_cancel_task(node_id, task_to_preempt)
            if success:
                # 从节点活跃任务中移除
                if task_to_preempt in node.active_tasks:
                    node.active_tasks.remove(task_to_preempt)

                # 更新被抢占任务状态
                if task_to_preempt in self.tasks:
                    preempted_task = self.tasks[task_to_preempt]
                    preempted_task.status = TaskStatus.CANCELLED
                    preempted_task.error_message = f"任务被更高优先级任务 {task.task_id} 抢占"

                    # 重新加入队列等待调度
                    self.queue_engine.enqueue_task(preempted_task)

                logger.info(f"任务 {task_to_preempt} 被任务 {task.task_id} 抢占")
                return True
            else:
                logger.warning(f"任务 {task_to_preempt} 取消失败")
                return False

        except Exception as e:
            logger.error(f"执行抢占失败: {e}")
            return False

    def _assign_task_to_node(self, task: DistributedTask, node_id: str) -> bool:
        """分配任务到节点"""
        try:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]

            # 更新任务状态
            task.assigned_node = node_id
            task.status = TaskStatus.RUNNING
            task.started_time = datetime.now()

            # 更新节点状态
            node.active_tasks.add(task.task_id)
            self._update_node_load_factor(node)

            # 通知节点执行任务
            self._notify_node_execute_task(node_id, task)

            return True

        except Exception as e:
            logger.error(f"任务分配失败: {e}")
            return False

    def _notify_node_execute_task(self, node_id: str, task: DistributedTask):
        """通知节点执行任务"""
        try:
            # 这里应该实现实际的节点通信逻辑
            # 例如：通过网络发送任务到节点

            # 模拟异步任务执行
            self.executor.submit(self._simulate_task_execution, node_id, task)

        except Exception as e:
            logger.error(f"任务执行通知失败: {e}")

    def _simulate_task_execution(self, node_id: str, task: DistributedTask):
        """模拟任务执行（用于演示）"""
        try:
            # 模拟任务执行时间
            execution_time = 5  # 5秒
            time.sleep(execution_time)

            # 模拟任务完成
            task.status = TaskStatus.COMPLETED
            task.completed_time = datetime.now()
            task.result = {"success": True, "execution_time": execution_time}

            # 更新节点状态
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.active_tasks.discard(task.task_id)
                self._update_node_load_factor(node)

            logger.info(f"任务 {task.task_id} 在节点 {node_id} 上执行完成")

        except Exception as e:
            logger.error(f"任务执行异常: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)

    def _notify_node_cancel_task(self, node_id: str, task_id: str) -> bool:
        """通知节点取消任务"""
        try:
            # 这里应该实现实际的节点通信逻辑
            logger.info(f"通知节点 {node_id} 取消任务 {task_id}")
            return True

        except Exception as e:
            logger.error(f"任务取消通知失败: {e}")
            return False

    def _reassign_node_tasks(self, node_id: str):
        """重新分配节点的任务"""
        try:
            if node_id not in self.nodes:
                return

            node = self.nodes[node_id]
            tasks_to_reassign = list(node.active_tasks)

            for task_id in tasks_to_reassign:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    task.status = TaskStatus.PENDING
                    task.assigned_node = None
                    if task_id not in self.task_queue:
                        self.task_queue.append(task_id)

            # 重新排序任务队列
            self._sort_task_queue()

            logger.info(f"节点 {node_id} 的 {len(tasks_to_reassign)} 个任务已重新分配")

        except Exception as e:
            logger.error(f"任务重新分配失败: {e}")

    def _sort_task_queue(self):
        """排序任务队列（按优先级）"""
        def get_priority(task_id):
            if task_id in self.tasks:
                return self.tasks[task_id].priority.value
            return 0

        self.task_queue.sort(key=get_priority, reverse=True)

    def _update_node_load_factor(self, node: NodeInfo):
        """更新节点负载因子"""
        try:
            # 基于活跃任务数量和CPU核心数计算负载
            task_load = len(node.active_tasks) / max(node.cpu_cores, 1)

            # 限制在0-1范围内
            node.load_factor = min(max(task_load, 0.0), 1.0)

        except Exception as e:
            logger.error(f"节点负载因子更新失败: {e}")

    def _update_cluster_stats(self):
        """更新集群统计信息"""
        try:
            self.stats.total_nodes = len(self.nodes)
            self.stats.online_nodes = sum(1 for node in self.nodes.values()
                                          if node.status == NodeStatus.ONLINE)
            self.stats.total_tasks = len(self.tasks)
            self.stats.running_tasks = sum(
                1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)
            self.stats.completed_tasks = sum(
                1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
            self.stats.failed_tasks = sum(1 for task in self.tasks.values()
                                          if task.status == TaskStatus.FAILED)

            if self.nodes:
                self.stats.avg_load_factor = sum(
                    node.load_factor for node in self.nodes.values()) / len(self.nodes)
                self.stats.total_cpu_cores = sum(node.cpu_cores for node in self.nodes.values())
                self.stats.total_memory_gb = sum(node.memory_gb for node in self.nodes.values())
                self.stats.total_gpu_devices = sum(len(node.gpu_devices)
                                                   for node in self.nodes.values())

        except Exception as e:
            logger.error(f"集群统计更新失败: {e}")

    def _calculate_task_progress(self, task: DistributedTask) -> float:
        """计算任务进度"""
        try:
            if task.status == TaskStatus.COMPLETED:
                return 1.0
            elif task.status == TaskStatus.PENDING:
                return 0.0
            elif task.status == TaskStatus.RUNNING:
                # 基于运行时间估算进度
                if task.started_time:
                    elapsed = (datetime.now() - task.started_time).total_seconds()
                    estimated_total = task.timeout_seconds
                    return min(elapsed / estimated_total, 0.99)
                return 0.0
            else:
                return 0.0

        except Exception as e:
            return 0.0

    def _cluster_monitor(self):
        """集群监控线程"""
        while self._monitoring:
            try:
                # 检查节点心跳
                self._check_node_heartbeats()

                # 检查超时任务
                self._check_timeout_tasks()

                # 调度待处理任务
                self._schedule_pending_tasks()

                # 更新统计信息
                self._update_cluster_stats()

                time.sleep(30)  # 每30秒检查一次

            except Exception as e:
                logger.error(f"集群监控异常: {e}")
                time.sleep(60)

    def _check_node_heartbeats(self):
        """检查节点心跳"""
        try:
            current_time = datetime.now()
            timeout_threshold = timedelta(minutes=5)

            for node_id, node in list(self.nodes.items()):
                if current_time - node.last_heartbeat > timeout_threshold:
                    if node.status == NodeStatus.ONLINE:
                        logger.warning(f"节点 {node_id} 心跳超时，标记为离线")
                        node.status = NodeStatus.OFFLINE

                        # 重新分配该节点的任务
                        self._reassign_node_tasks(node_id)

        except Exception as e:
            logger.error(f"心跳检查异常: {e}")

    def _check_timeout_tasks(self):
        """检查超时任务"""
        try:
            current_time = datetime.now()

            for task_id, task in list(self.tasks.items()):
                if task.status == TaskStatus.RUNNING and task.started_time:
                    elapsed = (current_time - task.started_time).total_seconds()

                    if elapsed > task.timeout_seconds:
                        logger.warning(
                            f"任务 {task_id} 执行超时 ({elapsed:.1f}s > {task.timeout_seconds}s)")

                        # 标记任务为超时
                        task.status = TaskStatus.TIMEOUT
                        task.completed_time = current_time
                        task.error_message = f"任务执行超时: {elapsed:.1f}秒"

                        # 如果有分配的节点，更新节点状态
                        if task.assigned_node and task.assigned_node in self.nodes:
                            node = self.nodes[task.assigned_node]
                            node.active_tasks.discard(task_id)
                            self._update_node_load_factor(node)

        except Exception as e:
            logger.error(f"超时检查异常: {e}")

    def _failure_recovery(self):
        """故障恢复线程"""
        while self._monitoring:
            try:
                # 检查失败的任务
                self._recover_failed_tasks()

                # 检查失败的节点
                self._recover_failed_nodes()

                time.sleep(300)  # 每5分钟检查一次

            except Exception as e:
                logger.error(f"故障恢复异常: {e}")
                time.sleep(60)

    def _recover_failed_tasks(self):
        """恢复失败的任务"""
        try:
            for task_id, task in list(self.tasks.items()):
                if task.status == TaskStatus.FAILED and task.retry_count < task.max_retries:
                    # 重新提交任务
                    task.status = TaskStatus.PENDING
                    task.retry_count += 1
                    task.assigned_node = None
                    task.started_time = None

                    if task_id not in self.task_queue:
                        self.task_queue.append(task_id)

                    logger.info(f"重新提交失败的任务 {task_id} (重试次数: {task.retry_count})")

        except Exception as e:
            logger.error(f"任务恢复异常: {e}")

    def _recover_failed_nodes(self):
        """恢复失败的节点"""
        try:
            # 这里应该实现节点故障恢复逻辑
            # 例如：尝试重启节点、迁移到备用节点等
            pass

        except Exception as e:
            logger.error(f"节点恢复异常: {e}")

    def sync_cluster_config(self, config: Dict[str, Any]) -> bool:
        """同步集群配置到所有节点"""
        try:
            with self._lock:
                # 更新本地配置
                self.cluster_config = config

                # 同步配置到所有在线节点
                for node_id, node in self.nodes.items():
                    if node.status == NodeStatus.ONLINE:
                        self._sync_config_to_node(node_id, config)

                logger.info(
                    f"集群配置已同步到 {len([n for n in self.nodes.values() if n.status == NodeStatus.ONLINE])} 个节点")

                # 发布配置同步事件
                if event_bus:
                    event_bus.publish_sync({
                        'event_type': 'cluster_config_synced',
                        'config': config
                    })

                return True

        except Exception as e:
            logger.error(f"集群配置同步失败: {e}")
            return False

    def authenticate_node(self, node_id: str, credentials: Dict[str, Any]) -> bool:
        """节点身份认证"""
        try:
            # 简单的认证逻辑（实际应使用更安全的认证机制）
            required_fields = ['token', 'hostname', 'ip_address']

            for field in required_fields:
                if field not in credentials:
                    logger.warning(f"节点 {node_id} 认证失败: 缺少必要字段 {field}")
                    return False

            # 验证token（这里使用简单的字符串比较，实际应使用加密验证）
            expected_token = self._generate_node_token(node_id, credentials['hostname'])
            if credentials['token'] != expected_token:
                logger.warning(f"节点 {node_id} 认证失败: token无效")
                return False

            logger.info(f"节点 {node_id} 认证成功")
            return True

        except Exception as e:
            logger.error(f"节点认证失败: {e}")
            return False

    def update_node_status(self, node_id: str, status: NodeStatus, additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """更新节点状态"""
        with self._lock:
            try:
                if node_id not in self.nodes:
                    logger.warning(f"节点 {node_id} 不存在，无法更新状态")
                    return False

                old_status = self.nodes[node_id].status
                self.nodes[node_id].status = status

                # 更新最后心跳时间
                if status == NodeStatus.ONLINE:
                    self.nodes[node_id].last_heartbeat = datetime.now()

                # 处理状态变更
                if old_status != status:
                    self._handle_node_status_change(node_id, old_status, status, additional_info)

                self._update_cluster_stats()
                self._sync_cluster_state()

                logger.info(f"节点 {node_id} 状态已更新: {old_status.value} -> {status.value}")
                return True

            except Exception as e:
                logger.error(f"更新节点状态失败: {e}")
                return False

    def get_cluster_config(self) -> Dict[str, Any]:
        """获取集群配置"""
        with self._lock:
            return getattr(self, 'cluster_config', {
                'coordinator_id': 'main-coordinator',
                'heartbeat_interval': 30,
                'task_timeout': 3600,
                'max_concurrent_tasks': 100,
                'load_balance_strategy': 'round_robin'
            })

    def _sync_cluster_state(self) -> None:
        """同步集群状态到所有节点"""
        try:
            cluster_state = {
                'nodes': {
                    node_id: {
                        'hostname': node.hostname,
                        'ip_address': node.ip_address,
                        'status': node.status.value,
                        'cpu_cores': node.cpu_cores,
                        'memory_gb': node.memory_gb,
                        'gpu_devices': node.gpu_devices,
                        'load_factor': node.load_factor,
                        'active_tasks': list(node.active_tasks),
                        'capabilities': list(node.capabilities)
                    }
                    for node_id, node in self.nodes.items()
                },
                'stats': {
                    'total_nodes': self.stats.total_nodes,
                    'online_nodes': self.stats.online_nodes,
                    'avg_load_factor': self.stats.avg_load_factor
                },
                'timestamp': datetime.now().isoformat()
            }

            # 广播状态到所有在线节点
            for node_id, node in self.nodes.items():
                if node.status == NodeStatus.ONLINE:
                    self._broadcast_state_to_node(node_id, cluster_state)

        except Exception as e:
            logger.error(f"集群状态同步失败: {e}")

    def _sync_config_to_node(self, node_id: str, config: Dict[str, Any]) -> None:
        """同步配置到指定节点"""
        try:
            # 这里应该实现实际的网络通信
            # 目前使用事件总线模拟
            if event_bus:
                event_bus.publish_sync({
                    'event_type': 'node_config_update',
                    'node_id': node_id,
                    'config': config
                })
        except Exception as e:
            logger.error(f"配置同步到节点 {node_id} 失败: {e}")

    def _broadcast_state_to_node(self, node_id: str, state: Dict[str, Any]) -> None:
        """广播状态到指定节点"""
        try:
            # 这里应该实现实际的网络通信
            # 目前使用事件总线模拟
            if event_bus:
                event_bus.publish_sync({
                    'event_type': 'cluster_state_update',
                    'node_id': node_id,
                    'state': state
                })
        except Exception as e:
            logger.error(f"状态广播到节点 {node_id} 失败: {e}")

    def _handle_node_status_change(self, node_id: str, old_status: NodeStatus,
                                   new_status: NodeStatus, additional_info: Optional[Dict[str, Any]]) -> None:
        """处理节点状态变更"""
        try:
            if new_status == NodeStatus.OFFLINE and old_status == NodeStatus.ONLINE:
                # 节点离线，重新分配任务
                self._reassign_node_tasks(node_id)

            elif new_status == NodeStatus.ONLINE and old_status != NodeStatus.ONLINE:
                # 节点上线，同步最新状态
                self._sync_cluster_state()

            # 发布状态变更事件
            if event_bus:
                event_bus.publish_sync({
                    'event_type': 'node_status_changed',
                    'node_id': node_id,
                    'old_status': old_status.value,
                    'new_status': new_status.value,
                    'additional_info': additional_info or {}
                })

        except Exception as e:
            logger.error(f"处理节点状态变更失败: {e}")

    def _generate_node_token(self, node_id: str, hostname: str) -> str:
        """生成节点认证token（简化实现）"""
        # 实际应该使用更安全的token生成机制
        import hashlib
        token_string = f"{node_id}:{hostname}:{getattr(self, 'cluster_secret', 'rqa2025-secret')}"
        return hashlib.sha256(token_string.encode()).hexdigest()[:32]

    def shutdown(self):
        """关闭分布式协调器"""
        logger.info("正在关闭分布式协调器...")

        self._monitoring = False

        # 等待监控线程结束
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)

        # 等待恢复线程结束
        if self._recovery_thread.is_alive():
            self._recovery_thread.join(timeout=10)

        # 关闭执行器
        self.executor.shutdown(wait=True)

        logger.info("分布式协调器已关闭")


# 全局分布式协调器实例
_distributed_coordinator = None


def get_distributed_coordinator() -> DistributedCoordinator:
    """获取分布式协调器实例"""
    global _distributed_coordinator

    if _distributed_coordinator is None:
        _distributed_coordinator = DistributedCoordinator()

    return _distributed_coordinator


# 便捷函数

def submit_distributed_task(task_type: str, data: Dict[str, Any],
                            priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """提交分布式任务的便捷函数"""
    coordinator = get_distributed_coordinator()
    return coordinator.submit_task(task_type, data, priority)


def get_cluster_status() -> Dict[str, Any]:
    """获取集群状态的便捷函数"""
    coordinator = get_distributed_coordinator()
    return coordinator.get_cluster_status()


__all__ = [
    'DistributedCoordinator',
    'get_distributed_coordinator',
    'submit_distributed_task',
    'get_cluster_status'
]

