import logging
"""
分布式交易节点管理器

实现多节点交易执行的核心功能，包括：
- 节点注册与发现
- 负载均衡
- 故障转移
- 任务分发
"""

import time
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from src.infrastructure.logging.distributed_lock import DistributedLockManager
from src.infrastructure.config.config_center import ConfigCenterManager
from src.infrastructure.logging.distributed_monitoring import DistributedMonitoringManager

logger = logging.getLogger(__name__)


@dataclass
class TradingNodeInfo:

    """交易节点信息"""
    node_id: str
    host: str
    port: int
    status: str  # 'active', 'inactive', 'failed'
    capabilities: List[str]  # ['equity', 'futures', 'options', 'forex']
    load: float  # 当前负载 (0 - 1)
    last_heartbeat: datetime
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'status': self.status,
            'capabilities': self.capabilities,
            'load': self.load,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class TradingTask:

    """交易任务"""
    task_id: str
    task_type: str  # 'order_execution', 'risk_check', 'position_update'
    priority: int  # 1 - 10, 10为最高优先级
    data: Dict[str, Any]
    created_at: datetime
    assigned_node: Optional[str] = None
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'assigned_node': self.assigned_node,
            'status': self.status
        }


class DistributedTradingNode:

    """
    分布式交易节点管理器

    功能:
    - 节点注册与发现
    - 负载均衡
    - 故障转移
    - 任务分发
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化分布式交易节点管理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.node_id = config.get('node_id', str(uuid.uuid4()))
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8080)

        # 初始化分布式组件
        self._init_distributed_components()

        # 节点管理
        self.nodes: Dict[str, TradingNodeInfo] = {}
        self.tasks: Dict[str, TradingTask] = {}
        self._lock = threading.Lock()

        # 心跳线程
        self._heartbeat_thread = None
        self._stop_heartbeat = False

        # 任务处理回调
        self._task_handlers: Dict[str, Callable] = {}

        # 启动心跳
        self._start_heartbeat()

        logger.info(f"分布式交易节点管理器初始化完成: {self.node_id}")

    def _init_distributed_components(self):
        """初始化分布式组件"""
        try:
            # 分布式锁管理器
            lock_config = self.config.get('distributed_lock', {})
            self.lock_manager = DistributedLockManager(lock_config)

            # 配置中心管理器
            config_center_config = self.config.get('config_center', {})
            self.config_manager = ConfigCenterManager(config_center_config)

            # 分布式监控管理器
            monitoring_config = self.config.get('distributed_monitoring', {})
            self.monitoring_manager = DistributedMonitoringManager(monitoring_config)

            logger.info("分布式组件初始化成功")
        except Exception as e:
            logger.error(f"分布式组件初始化失败: {e}")
            raise

    def register_node(self, capabilities: List[str] = None) -> bool:
        """
        注册当前节点

        Args:
            capabilities: 节点能力列表

        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                node_info = TradingNodeInfo(
                    node_id=self.node_id,
                    host=self.host,
                    port=self.port,
                    status='active',
                    capabilities=capabilities or ['equity'],
                    load=0.0,
                    last_heartbeat=datetime.now(),
                    created_at=datetime.now()
                )

                # 使用分布式锁确保注册的原子性
                lock_key = f"node_registration_{self.node_id}"
                with self.lock_manager.acquire_lock(lock_key, timeout=5):
                    self.nodes[self.node_id] = node_info

                    # 将节点信息存储到配置中心
                    self.config_manager.set_config(
                        f"trading_nodes/{self.node_id}",
                        node_info.to_dict()
                    )

                    logger.info(f"节点注册成功: {self.node_id}")
                    return True

        except Exception as e:
            logger.error(f"节点注册失败: {e}")
            return False

    def discover_nodes(self) -> List[TradingNodeInfo]:
        """
        发现其他节点

        Returns:
            List[TradingNodeInfo]: 发现的节点列表
        """
        try:
            nodes = []

            # 从配置中心获取所有节点信息
            node_configs = self.config_manager.get_config("trading_nodes")
            if node_configs:
                for node_id, node_data in node_configs.items():
                    if node_id != self.node_id:
                        node_info = TradingNodeInfo(
                            node_id=node_data['node_id'],
                            host=node_data['host'],
                            port=node_data['port'],
                            status=node_data['status'],
                            capabilities=node_data['capabilities'],
                            load=node_data['load'],
                            last_heartbeat=datetime.fromisoformat(node_data['last_heartbeat']),
                            created_at=datetime.fromisoformat(node_data['created_at'])
                        )
                        nodes.append(node_info)

            # 更新本地节点缓存
            with self._lock:
                for node in nodes:
                    self.nodes[node.node_id] = node

            logger.info(f"发现 {len(nodes)} 个节点")
            return nodes

        except Exception as e:
            logger.error(f"节点发现失败: {e}")
            return []

    def submit_task(self, task_type: str, data: Dict[str, Any],


                    priority: int = 5) -> str:
        """
        提交交易任务

        Args:
            task_type: 任务类型
            data: 任务数据
            priority: 任务优先级

        Returns:
            str: 任务ID
        """
        try:
            task_id = str(uuid.uuid4())
            task = TradingTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                data=data,
                created_at=datetime.now()
            )

            with self._lock:
                self.tasks[task_id] = task

                # 存储任务到配置中心
                self.config_manager.set_config(
                    f"trading_tasks/{task_id}",
                    task.to_dict()
                )

            # 尝试分配任务
            self._assign_task(task)

            logger.info(f"任务提交成功: {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"任务提交失败: {e}")
            raise

    def _assign_task(self, task: TradingTask):
        """
        分配任务到合适的节点

        Args:
            task: 交易任务
        """
        try:
            # 获取可用节点
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == 'active' and node.load < 0.8
            ]

            if not available_nodes:
                logger.warning("没有可用的节点来执行任务")
                return

            # 根据负载均衡策略选择节点
            selected_node = self._select_node_for_task(task, available_nodes)

            if selected_node:
                task.assigned_node = selected_node.node_id
                task.status = 'running'

                # 更新任务状态
                with self._lock:
                    self.tasks[task.task_id] = task

                # 更新配置中心
                self.config_manager.set_config(
                    f"trading_tasks/{task.task_id}",
                    task.to_dict()
                )

                logger.info(f"任务 {task.task_id} 分配给节点 {selected_node.node_id}")

        except Exception as e:
            logger.error(f"任务分配失败: {e}")

    def _select_node_for_task(self, task: TradingTask,


                              available_nodes: List[TradingNodeInfo]) -> Optional[TradingNodeInfo]:
        """
        为任务选择合适的节点

        Args:
            task: 交易任务
            available_nodes: 可用节点列表

        Returns:
            Optional[TradingNodeInfo]: 选中的节点
        """
        try:
            # 简单的负载均衡策略：选择负载最低的节点
            if not available_nodes:
                return None

            # 按负载排序，选择负载最低的节点
            selected_node = min(available_nodes, key=lambda x: x.load)

            # 更新节点负载
            selected_node.load += 0.1  # 简单增加负载

            return selected_node

        except Exception as e:
            logger.error(f"节点选择失败: {e}")
            return None

    def register_task_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器

        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self._task_handlers[task_type] = handler
        logger.info(f"注册任务处理器: {task_type}")

    def process_task(self, task_id: str) -> Dict[str, Any]:
        """
        处理任务

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            with self._lock:
                task = self.tasks.get(task_id)
                if not task:
                    raise ValueError(f"任务不存在: {task_id}")

                if task.status != 'running':
                    raise ValueError(f"任务状态不正确: {task.status}")

            # 获取任务处理器
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"未找到任务处理器: {task.task_type}")

            # 执行任务
            result = handler(task.data)

            # 更新任务状态
            with self._lock:
                task.status = 'completed'
                self.tasks[task_id] = task

            # 更新配置中心
            self.config_manager.set_config(
                f"trading_tasks/{task_id}",
                task.to_dict()
            )

            logger.info(f"任务处理完成: {task_id}")
            return result

        except Exception as e:
            logger.error(f"任务处理失败: {e}")

            # 更新任务状态为失败
            with self._lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = 'failed'

            raise

    def _start_heartbeat(self):
        """启动心跳线程"""

        def heartbeat_worker():

            while not self._stop_heartbeat:
                try:
                    self._send_heartbeat()
                    time.sleep(30)  # 30秒发送一次心跳
                except Exception as e:
                    logger.error(f"心跳发送失败: {e}")

        self._heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
        logger.info("心跳线程启动")

    def _send_heartbeat(self):
        """发送心跳"""
        try:
            with self._lock:
                if self.node_id in self.nodes:
                    node = self.nodes[self.node_id]
                    node.last_heartbeat = datetime.now()

                    # 更新配置中心
                    self.config_manager.set_config(
                        f"trading_nodes/{self.node_id}",
                        node.to_dict()
                    )

            logger.debug(f"心跳发送成功: {self.node_id}")

        except Exception as e:
            logger.error(f"心跳发送失败: {e}")

    def get_node_status(self) -> Dict[str, Any]:
        """
        获取节点状态

        Returns:
            Dict[str, Any]: 节点状态信息
        """
        try:
            with self._lock:
                node = self.nodes.get(self.node_id)
                if node:
                    return {
                        'node_id': node.node_id,
                        'status': node.status,
                        'load': node.load,
                        'capabilities': node.capabilities,
                        'last_heartbeat': node.last_heartbeat.isoformat(),
                        'total_nodes': len(self.nodes),
                        'total_tasks': len(self.tasks)
                    }
                else:
                    return {'error': 'Node not found'}

        except Exception as e:
            logger.error(f"获取节点状态失败: {e}")
            return {'error': str(e)}

    def shutdown(self):
        """关闭节点管理器"""
        try:
            self._stop_heartbeat = True

            # 注销节点
            with self._lock:
                if self.node_id in self.nodes:
                    self.nodes[self.node_id].status = 'inactive'

                    # 更新配置中心
                    self.config_manager.set_config(
                        f"trading_nodes/{self.node_id}",
                        self.nodes[self.node_id].to_dict()
                    )

            logger.info(f"节点管理器已关闭: {self.node_id}")

        except Exception as e:
            logger.error(f"节点管理器关闭失败: {e}")


def create_distributed_trading_node(config: Dict[str, Any]) -> DistributedTradingNode:
    """
    创建分布式交易节点管理器

    Args:
        config: 配置字典

    Returns:
        DistributedTradingNode: 分布式交易节点管理器
    """
    return DistributedTradingNode(config)
