"""
流程实例池组件

职责:
- 流程实例的创建和复用
- 实例池管理
- 资源优化
"""

import time
import logging
from typing import Dict, List, Optional
from collections import deque

from ..models.process_models import ProcessConfig, ProcessInstance, create_process_instance, BusinessProcessState

logger = logging.getLogger(__name__)


class ProcessInstancePool:
    """
    流程实例池组件

    提供实例复用机制，优化资源使用
    """

    def __init__(self, config: 'PoolConfig'):
        """
        初始化实例池

        Args:
            config: 实例池配置
        """
        self.config = config
        self._available_instances: deque = deque()
        self._in_use_instances: Dict[str, ProcessInstance] = {}
        self._instance_counter = 0

        logger.info(f"流程实例池初始化完成 (最大: {config.max_size})")

    def get_instance(self, process_config: ProcessConfig) -> ProcessInstance:
        """
        获取流程实例

        Args:
            process_config: 流程配置

        Returns:
            ProcessInstance: 流程实例
        """
        # 检查池大小限制
        if len(self._in_use_instances) >= self.config.max_size:
            raise RuntimeError(f"实例池已满 (最大: {self.config.max_size})")

        # 尝试复用实例
        if self.config.enable_reuse and self._available_instances:
            instance = self._available_instances.popleft()
            # 重置实例
            instance.process_config = process_config
            instance.status = BusinessProcessState.IDLE
            instance.start_time = time.time()
            instance.end_time = None
            instance.progress = 0.0
            instance.error_message = ""
            instance.context = {}
            logger.debug(f"复用实例: {instance.instance_id}")
        else:
            # 创建新实例
            self._instance_counter += 1
            instance = create_process_instance(
                instance_id=f"inst_{self._instance_counter:06d}",
                process_config=process_config
            )
            logger.debug(f"创建新实例: {instance.instance_id}")

        # 标记为使用中
        self._in_use_instances[instance.instance_id] = instance

        return instance

    def return_instance(self, instance: ProcessInstance):
        """
        归还实例到池

        Args:
            instance: 流程实例
        """
        if instance.instance_id in self._in_use_instances:
            del self._in_use_instances[instance.instance_id]

            # 如果启用复用且实例状态正常，放回池中
            if self.config.enable_reuse and instance.status == BusinessProcessState.COMPLETED:
                if len(self._available_instances) < self.config.max_size:
                    self._available_instances.append(instance)
                    logger.debug(f"实例已归还到池: {instance.instance_id}")
                else:
                    logger.debug(f"实例池已满，释放实例: {instance.instance_id}")
            else:
                logger.debug(f"实例已释放: {instance.instance_id}")

    def get_pool_stats(self) -> Dict:
        """获取池统计信息"""
        return {
            'max_size': self.config.max_size,
            'available': len(self._available_instances),
            'in_use': len(self._in_use_instances),
            'total_created': self._instance_counter,
            'enable_reuse': self.config.enable_reuse
        }

    def clear_pool(self):
        """清空池"""
        self._available_instances.clear()
        logger.info("实例池已清空")

    def get_status(self) -> Dict:
        """获取池状态"""
        return self.get_pool_stats()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..configs.orchestrator_configs import PoolConfig
