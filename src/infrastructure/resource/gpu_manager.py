import logging
from typing import List, Dict, Optional
from datetime import datetime
import subprocess
import re

logger = logging.getLogger(__name__)

class GPUManager:
    """GPU资源管理器"""

    def __init__(self):
        """初始化GPU监控器"""
        self._gpu_count = self._get_gpu_count()
        logger.info(f"Initialized GPU monitor with {self._gpu_count} GPUs")

    def _get_gpu_count(self) -> int:
        """获取GPU数量"""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            logger.warning("PyTorch not available, GPU monitoring disabled")
            return 0
        except Exception as e:
            logger.error(f"Failed to get GPU count: {e}")
            return 0

    def get_gpu_stats(self) -> Optional[List[Dict]]:
        """
        获取所有GPU的统计信息

        Returns:
            List[Dict]: GPU统计信息列表，每个GPU一个字典
        """
        if self._gpu_count == 0:
            return None

        stats = []
        for i in range(self._gpu_count):
            try:
                gpu_stat = self._get_single_gpu_stats(i)
                if gpu_stat:
                    stats.append(gpu_stat)
            except Exception as e:
                logger.error(f"Failed to get stats for GPU {i}: {e}")

        return stats if stats else None

    def _get_single_gpu_stats(self, gpu_index: int) -> Dict:
        """
        获取单个GPU的统计信息

        Args:
            gpu_index: GPU索引

        Returns:
            Dict: GPU统计信息
        """
        import torch

        # 获取GPU设备属性
        device_props = torch.cuda.get_device_properties(gpu_index)

        # 获取显存使用情况
        memory_allocated = torch.cuda.memory_allocated(gpu_index)
        memory_reserved = torch.cuda.memory_reserved(gpu_index)
        memory_total = device_props.total_memory

        # 获取GPU利用率
        utilization = self._get_gpu_utilization(gpu_index)

        # 获取GPU温度
        temperature = self._get_gpu_temperature(gpu_index)

        return {
            'index': gpu_index,
            'name': device_props.name,
            'memory': {
                'total': memory_total,
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'percent': memory_allocated / memory_total * 100 if memory_total > 0 else 0
            },
            'utilization': utilization,
            'temperature': temperature,
            'timestamp': datetime.now().isoformat()
        }

    def _get_gpu_utilization(self, gpu_index: int) -> float:
        """
        获取GPU利用率(百分比)

        Args:
            gpu_index: GPU索引

        Returns:
            float: GPU利用率(0-100)
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits', f'--id={gpu_index}'],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_index} utilization: {e}")
            return 0.0

    def _get_gpu_temperature(self, gpu_index: int) -> Optional[float]:
        """
        获取GPU温度(摄氏度)

        Args:
            gpu_index: GPU索引

        Returns:
            Optional[float]: GPU温度，获取失败返回None
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu=temperature.gpu',
                 '--format=csv,noheader,nounits', f'--id={gpu_index}'],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get GPU {gpu_index} temperature: {e}")
            return None

    def get_gpu_count(self) -> int:
        """
        获取GPU数量

        Returns:
            int: GPU数量
        """
        return self._gpu_count
