import logging
from typing import Dict, Set, Optional
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

class QuotaManager:
    """策略资源配额管理器"""

    def __init__(self):
        """
        初始化配额管理器
        """
        self.quota_map: Dict[str, Dict] = {}  # 策略配额配置
        self.strategy_resources: Dict[str, Dict] = {}  # 策略当前资源使用
        self.lock = Lock()  # 线程安全锁

    def set_quota(
        self,
        strategy: str,
        cpu_percent: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        max_workers: Optional[int] = None
    ) -> None:
        """
        设置策略资源配额

        Args:
            strategy: 策略名称
            cpu_percent: 最大CPU使用率(0-100)
            gpu_memory_mb: 最大GPU显存(MB)
            max_workers: 最大工作线程数
        """
        with self.lock:
            if strategy not in self.quota_map:
                self.quota_map[strategy] = {}

            if cpu_percent is not None:
                self.quota_map[strategy]['cpu_percent'] = cpu_percent

            if gpu_memory_mb is not None:
                self.quota_map[strategy]['gpu_memory_mb'] = gpu_memory_mb

            if max_workers is not None:
                self.quota_map[strategy]['max_workers'] = max_workers

            logger.info(f"Set quota for strategy {strategy}: {self.quota_map[strategy]}")

    def check_quota(
        self,
        strategy: str,
        current_cpu: Optional[float] = None,
        current_gpu_mem: Optional[float] = None
    ) -> bool:
        """
        检查策略资源配额

        Args:
            strategy: 策略名称
            current_cpu: 当前CPU使用率
            current_gpu_mem: 当前GPU显存使用(MB)

        Returns:
            bool: 是否满足资源配额
        """
        with self.lock:
            if strategy not in self.quota_map:
                return True  # 无配额限制

            quota = self.quota_map[strategy]
            violations = []

            # 检查CPU配额
            if 'cpu_percent' in quota and current_cpu is not None:
                if current_cpu > quota['cpu_percent']:
                    violations.append(f"CPU usage {current_cpu}% > {quota['cpu_percent']}%")

            # 检查GPU显存配额
            if 'gpu_memory_mb' in quota and current_gpu_mem is not None:
                if current_gpu_mem > quota['gpu_memory_mb']:
                    violations.append(f"GPU memory {current_gpu_mem}MB > {quota['gpu_memory_mb']}MB")

            # 检查工作线程配额
            if 'max_workers' in quota:
                current_workers = len(self.strategy_resources.get(strategy, {}).get('workers', set()))
                if current_workers >= quota['max_workers']:
                    violations.append(f"Workers {current_workers} >= {quota['max_workers']}")

            if violations:
                logger.warning(f"Quota violation for {strategy}: {'; '.join(violations)}")
                return False

            return True

    def register_worker(
        self,
        strategy: str,
        worker_id: str
    ) -> bool:
        """
        注册工作线程

        Args:
            strategy: 策略名称
            worker_id: 工作线程ID

        Returns:
            bool: 是否注册成功(配额允许)
        """
        with self.lock:
            # 检查工作线程配额
            if 'max_workers' in self.quota_map.get(strategy, {}):
                current_workers = len(self.strategy_resources.get(strategy, {}).get('workers', set()))
                if current_workers >= self.quota_map[strategy]['max_workers']:
                    logger.warning(f"Cannot register worker for {strategy}: max workers reached")
                    return False

            # 注册工作线程
            if strategy not in self.strategy_resources:
                self.strategy_resources[strategy] = {'workers': set()}

            self.strategy_resources[strategy]['workers'].add(worker_id)
            logger.debug(f"Registered worker {worker_id} for strategy {strategy}")
            return True

    def unregister_worker(
        self,
        strategy: str,
        worker_id: str
    ) -> None:
        """
        注销工作线程

        Args:
            strategy: 策略名称
            worker_id: 工作线程ID
        """
        with self.lock:
            if strategy in self.strategy_resources:
                self.strategy_resources[strategy]['workers'].discard(worker_id)
                logger.debug(f"Unregistered worker {worker_id} from strategy {strategy}")

    def get_quota(self, strategy: str) -> Dict:
        """
        获取策略配额配置

        Args:
            strategy: 策略名称

        Returns:
            Dict: 配额配置
        """
        return self.quota_map.get(strategy, {}).copy()

    def get_resource_usage(self, strategy: str) -> Dict:
        """
        获取策略资源使用情况

        Args:
            strategy: 策略名称

        Returns:
            Dict: 资源使用情况
        """
        with self.lock:
            return {
                'workers': len(self.strategy_resources.get(strategy, {}).get('workers', set())),
                'timestamp': datetime.now().isoformat()
            }
