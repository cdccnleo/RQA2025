"""
流程监控组件

职责:
- 流程实例的监控
- 性能指标收集
- 自动清理过期流程
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
# defaultdict 已移除（未使用）

from ..models.process_models import ProcessInstance, BusinessProcessState

logger = logging.getLogger(__name__)


class ProcessMonitor:
    """
    流程监控组件

    监控所有流程实例的状态和性能
    """

    def __init__(self, config: 'MonitorConfig'):
        """
        初始化流程监控器

        Args:
            config: 监控器配置
        """
        self.config = config
        self.processes: Dict[str, ProcessInstance] = {}
        self.metrics = {
            'total_processes': 0,
            'running_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'total_memory_usage': 0.0
        }
        self._lock = threading.RLock()
        self._cleanup_timer = None

        if config.enable_cleanup:
            self._start_cleanup_timer()

        logger.info("流程监控器初始化完成")

    def register_process(self, instance: ProcessInstance):
        """注册流程"""
        with self._lock:
            self.processes[instance.instance_id] = instance
            self.metrics['total_processes'] += 1
            self._update_metrics()
            logger.debug(f"流程已注册: {instance.instance_id}")

    def update_process(self, instance_id: str, status: BusinessProcessState, **kwargs):
        """更新流程状态"""
        with self._lock:
            if instance_id in self.processes:
                instance = self.processes[instance_id]
                instance.update_status(status)

                # 更新其他属性
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)

                self._update_metrics()
                logger.debug(f"流程已更新: {instance_id} -> {status.value}")

    def get_process(self, instance_id: str) -> Optional[ProcessInstance]:
        """获取流程实例"""
        return self.processes.get(instance_id)

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        with self._lock:
            return self.metrics.copy()

    def get_running_processes(self) -> List[ProcessInstance]:
        """获取运行中的流程"""
        with self._lock:
            return [
                p for p in self.processes.values()
                if p.status not in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]
            ]

    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        return {
            'total_processes': len(self.processes),
            'running': len(self.get_running_processes()),
            'metrics': self.get_metrics()
        }

    def cleanup_old_processes(self):
        """清理过期流程"""
        with self._lock:
            current_time = time.time()
            to_remove = []

            for instance_id, instance in self.processes.items():
                # 清理完成或错误状态且超过TTL的流程
                if instance.status in [BusinessProcessState.COMPLETED, BusinessProcessState.ERROR]:
                    if instance.end_time and (current_time - instance.end_time) > self.config.process_ttl:
                        to_remove.append(instance_id)

            for instance_id in to_remove:
                del self.processes[instance_id]

            if to_remove:
                logger.info(f"清理了{len(to_remove)}个过期流程")

    def _update_metrics(self):
        """更新指标"""
        running = 0
        completed = 0
        failed = 0
        total_memory = 0.0

        for instance in self.processes.values():
            if instance.status in [BusinessProcessState.COMPLETED]:
                completed += 1
            elif instance.status == BusinessProcessState.ERROR:
                failed += 1
            else:
                running += 1
            total_memory += instance.memory_usage

        self.metrics['running_processes'] = running
        self.metrics['completed_processes'] = completed
        self.metrics['failed_processes'] = failed
        self.metrics['total_memory_usage'] = total_memory

    def _start_cleanup_timer(self):
        """启动清理定时器"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self.cleanup_old_processes()
                except Exception as e:
                    logger.error(f"清理进程失败: {e}")

        self._cleanup_timer = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_timer.start()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..configs.orchestrator_configs import MonitorConfig
