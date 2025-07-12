# RQA2025 基础设施层功能增强分析报告（续）

## 2. 功能分析（续）

### 2.3 资源管理增强（续）

#### 2.3.1 计算资源管理（续）

**实现建议**（续）：

```python
    def get_resource_usage(self) -> Dict:
        """
        获取资源使用情况
        
        Returns:
            Dict: 资源使用统计
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        }
    
    def _check_thresholds(self, stats: Dict) -> None:
        """
        检查资源使用是否超过阈值
        
        Args:
            stats: 资源使用统计
        """
        # 检查CPU使用率
        if stats['cpu']['percent'] > self.cpu_threshold:
            logger.warning(f"CPU usage exceeds threshold: {stats['cpu']['percent']}% > {self.cpu_threshold}%")
        
        # 检查内存使用率
        if stats['memory']['percent'] > self.memory_threshold:
            logger.warning(f"Memory usage exceeds threshold: {stats['memory']['percent']}% > {self.memory_threshold}%")
        
        # 检查磁盘使用率
        if stats['disk']['percent'] > self.disk_threshold:
            logger.warning(f"Disk usage exceeds threshold: {stats['disk']['percent']}% > {self.disk_threshold}%")
    
    def get_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取资源使用统计
        
        Args:
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: 资源使用统计列表
        """
        if not start_time and not end_time:
            return self.stats
        
        filtered_stats = self.stats
        
        if start_time:
            filtered_stats = [s for s in filtered_stats if s['timestamp'] >= start_time]
        
        if end_time:
            filtered_stats = [s for s in filtered_stats if s['timestamp'] <= end_time]
        
        return filtered_stats
    
    def get_summary(self) -> Dict:
        """
        获取资源使用摘要
        
        Returns:
            Dict: 资源使用摘要
        """
        if not self.stats:
            return {
                'cpu': {'avg': 0, 'max': 0},
                'memory': {'avg': 0, 'max': 0},
                'disk': {'avg': 0, 'max': 0}
            }
        
        cpu_avg = sum(s['cpu']['percent'] for s in self.stats) / len(self.stats)
        cpu_max = max(s['cpu']['percent'] for s in self.stats)
        
        memory_avg = sum(s['memory']['percent'] for s in self.stats) / len(self.stats)
        memory_max = max(s['memory']['percent'] for s in self.stats)
        
        disk_avg = sum(s['disk']['percent'] for s in self.stats) / len(self.stats)
        disk_max = max(s['disk']['percent'] for s in self.stats)
        
        return {
            'cpu': {
                'avg': cpu_avg,
                'max': cpu_max
            },
            'memory': {
                'avg': memory_avg,
                'max': memory_max
            },
            'disk': {
                'avg': disk_avg,
                'max': disk_max
            }
        }
```

#### 2.3.2 GPU资源管理

**现状分析**：
缺乏对GPU资源的管理和监控，无法充分利用GPU加速能力。

**实现建议**：
实现一个 `GPUManager` 类，提供GPU资源管理功能：

```python
import os
import time
import threading
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GPUManager:
    """GPU资源管理器"""
    
    def __init__(
        self,
        memory_threshold: float = 80.0,
        utilization_threshold: float = 80.0,
        check_interval: float = 5.0
    ):
        """
        初始化GPU资源管理器
        
        Args:
            memory_threshold: GPU内存使用率阈值（百分比）
            utilization_threshold: GPU利用率阈值（百分比）
            check_interval: 检查间隔（秒）
        """
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self.check_interval = check_interval
        
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # GPU使用统计
        self.stats: List[Dict] = []
        
        # 检查是否有可用的GPU
        self.has_gpu = self._check_gpu_available()
    
    def _check_gpu_available(self) -> bool:
        """
        检查是否有可用的GPU
        
        Returns:
            bool: 是否有可用的GPU
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, GPU monitoring disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to check GPU availability: {e}")
            return False
    
    def start_monitoring(self) -> None:
        """启动GPU监控"""
        if self.monitoring or not self.has_gpu:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止GPU监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval + 1)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """GPU监控循环"""
        while self.monitoring:
            try:
                stats = self.get_gpu_usage()
                self.stats.append(stats)
                
                # 检查GPU使用是否超过阈值
                self._check_thresholds(stats)
                
                # 限制统计数据数量
                if len(self.stats) > 1000:
                    self.stats = self.stats[-1000:]
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
            
            time.sleep(self.check_interval)
    
    def get_gpu_usage(self) -> Dict:
        """
        获取GPU使用情况
        
        Returns:
            Dict: GPU使用统计
        """
        if not self.has_gpu:
            return {
                'timestamp': datetime.now().isoformat(),
                'gpus': []
            }
        
        import torch
        
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            # 获取GPU内存使用情况
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            
            # 计算内存使用率
            memory_percent = memory_allocated / memory_total * 100
            
            # 获取GPU利用率（需要使用nvidia-smi）
            utilization = self._get_gpu_utilization(i)
            
            gpu_stats.append({
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory': {
                    'total': memory_total,
                    'allocated': memory_allocated,
                    'reserved': memory_reserved,
                    'percent': memory_percent
                },
                'utilization': utilization
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'gpus': gpu_stats
        }
    
    def _get_gpu_utilization(self, gpu_index: int) -> float:
        """
        获取GPU利用率
        
        Args:
            gpu_index: GPU索引
            
        Returns:
            float: GPU利用率（百分比）
        """
        try:
            import subprocess
            result = subprocess.check_output(
                [
                    'nvidia-smi',
                    f'--query-gpu=utilization.gpu',
                    '--format=csv,noheader,nounits',
                    f'--id={gpu_index}'
                ],
                encoding='utf-8'
            )
            return float(result.strip())
        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
            return 0.0
    
    def _check_thresholds(self, stats: Dict) -> None:
        """
        检查GPU使用是否超过阈值
        
        Args:
            stats: GPU使用统计
        """
        for gpu in stats['gpus']:
            # 检查GPU内存使用率
            if gpu['memory']['percent'] > self.memory_threshold:
                logger.warning(f"GPU {gpu['index']} memory usage exceeds threshold: {gpu['memory']['percent']:.2f}% > {self.memory_threshold}%")
            
            # 检查GPU利用率
            if gpu['utilization'] > self.utilization_threshold:
                logger.warning(f"GPU {gpu['index']} utilization exceeds threshold: {gpu['utilization']:.2f}% > {self.utilization_threshold}%")
    
    def get_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict]:
        """
        获取GPU使用统计
        
        Args:
            start_time: 开始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            
        Returns:
            List[Dict]: GPU使用统计列表
        """
        if not start_time and not end_time:
            return self.stats
        
        filtered_stats = self.stats
        
        if start_time:
            filtered_stats = [s for s in filtered_stats if s['timestamp'] >= start_time]
        
        if end_time:
            filtered_stats = [s for s in filtered_stats if s['timestamp'] <= end_time]
        
        return filtered_stats
    
    def get_summary(self) -> Dict:
        """
        获取GPU使用摘要
        
        Returns:
            Dict: GPU使用摘要
        """
        if not self.stats or not self.stats[0]['gpus']:
            return {'gpus': []}
        
        gpu_count = len(self.stats[0]['gpus'])
        summary = {'gpus': []}
        
        for i in range(gpu_count):
            memory_avg = sum(s['gpus'][i]['memory']['percent'] for s in self.stats) / len(self.stats)
            memory_max = max(s['gpus'][i]['memory']['percent'] for s in self.stats)
            
            utilization_avg = sum(s['gpus'][i]['utilization'] for s in self.stats) / len(self.stats)
            utilization_max = max(s['gpus'][i]['utilization'] for s in self.stats)
            
            summary['gpus'].append({
                'index': i,
                'name': self.stats[0]['gpus'][i]['name'],
                'memory': {
                    'avg': memory_avg,
                    'max': memory_max
                },
                'utilization': {
                    'avg': utilization_avg,
                    'max': utilization_max
                }
            })
        
        return summary
```

### 2.4 监控系统增强

#### 2.4.1 系统监控

**现状分析**：
缺乏全面的系统监控能力，无法及时发现和解决系统问题。

**实现建议**：
实现一个 `SystemMonitor` 类，提供系统监控功能：

```python
import os
import time
import threading
import socket
import platform
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemMonitor:
    """系统监控器"""
    
    def __init__(
        self,
        check_interval: float = 60.0,
        alert_callbacks: Optional[List[Callable[[str, Dict], None]]] = None
    ):
        """
        初始化系统监控器
        
        Args:
            check_interval: 检查间隔（秒）
            alert_callbacks: 告警回调函数列表
        """
        self.check_interval = check_interval
        self.alert_callbacks = alert_callbacks or []
        
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 系统信息
        self.system_info = self._get_system_info()
        
        # 监控数据
        self.monitoring_data: List[Dict] = []
    
    def _get_system_info(self) -> Dict:
        """
        获取系统信息
        
        Returns:
            Dict: 系统信息
        """
        return {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'start_time': datetime.now().isoformat()
        }
    
    def start_monitoring(self) -> None:
        """启动系统监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止系统监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_