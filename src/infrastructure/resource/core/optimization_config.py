
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
"""
优化配置参数对象

使用参数对象模式封装复杂的配置参数，解决长参数列表问题
"""


@dataclass
class MemoryOptimizationConfig:
    """内存优化配置参数对象"""
    enabled: bool = False
    gc_threshold: float = 80.0
    enable_pooling: bool = False
    monitor_large_objects: bool = False
    max_memory_mb: Optional[int] = None
    cleanup_interval_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "gc_threshold": self.gc_threshold,
            "enable_pooling": self.enable_pooling,
            "monitor_large_objects": self.monitor_large_objects,
            "max_memory_mb": self.max_memory_mb,
            "cleanup_interval_seconds": self.cleanup_interval_seconds
        }


@dataclass
class CpuOptimizationConfig:
    """CPU优化配置参数对象"""
    enabled: bool = False
    cpu_affinity: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    priority_threshold: float = 90.0
    power_saving: bool = False
    max_cpu_percent: Optional[float] = None
    load_balancing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "cpu_affinity": self.cpu_affinity,
            "priority_threshold": self.priority_threshold,
            "power_saving": self.power_saving,
            "max_cpu_percent": self.max_cpu_percent,
            "load_balancing": self.load_balancing
        }


@dataclass
class DiskOptimizationConfig:
    """磁盘优化配置参数对象"""
    enabled: bool = False
    io_scheduler: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    caching: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    readahead: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    max_disk_usage_percent: float = 95.0
    cleanup_threshold_percent: float = 85.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "io_scheduler": self.io_scheduler,
            "caching": self.caching,
            "readahead": self.readahead,
            "max_disk_usage_percent": self.max_disk_usage_percent,
            "cleanup_threshold_percent": self.cleanup_threshold_percent
        }


@dataclass
class ParallelizationConfig:
    """并行化配置参数对象"""
    enabled: bool = False
    thread_pool_size: int = 4
    process_pool_size: Optional[int] = None
    async_enabled: bool = False
    max_concurrent_tasks: int = 10
    queue_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "thread_pool_size": self.thread_pool_size,
            "process_pool_size": self.process_pool_size,
            "async_enabled": self.async_enabled,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "queue_size": self.queue_size
        }


@dataclass
class CheckpointingConfig:
    """检查点配置参数对象"""
    enabled: bool = False
    interval_seconds: int = 300
    storage_path: Optional[str] = None
    compression: Dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    max_checkpoints: int = 10
    retention_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "storage_path": self.storage_path,
            "compression": self.compression,
            "max_checkpoints": self.max_checkpoints,
            "retention_hours": self.retention_hours
        }


@dataclass
class ResourceOptimizationConfig:
    """资源优化配置参数对象

    使用参数对象模式封装所有优化相关的配置参数，
    解决长参数列表和复杂配置管理问题
    """
    memory_optimization: MemoryOptimizationConfig = field(default_factory=MemoryOptimizationConfig)
    cpu_optimization: CpuOptimizationConfig = field(default_factory=CpuOptimizationConfig)
    disk_optimization: DiskOptimizationConfig = field(default_factory=DiskOptimizationConfig)
    parallelization: ParallelizationConfig = field(default_factory=ParallelizationConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    # 全局配置
    enabled_optimizations: List[str] = field(default_factory=lambda: ["memory", "cpu", "disk"])
    optimization_priority: List[str] = field(
        default_factory=lambda: ["memory", "cpu", "disk", "parallelization", "checkpointing"])
    auto_optimization: bool = False
    monitoring_interval_seconds: int = 60

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResourceOptimizationConfig':
        """从字典创建配置对象"""
        return cls(
            memory_optimization=MemoryOptimizationConfig(
                **config_dict.get("memory_optimization", {})),
            cpu_optimization=CpuOptimizationConfig(**config_dict.get("cpu_optimization", {})),
            disk_optimization=DiskOptimizationConfig(**config_dict.get("disk_optimization", {})),
            parallelization=ParallelizationConfig(**config_dict.get("parallelization", {})),
            checkpointing=CheckpointingConfig(**config_dict.get("checkpointing", {})),
            enabled_optimizations=config_dict.get(
                "enabled_optimizations", ["memory", "cpu", "disk"]),
            optimization_priority=config_dict.get(
                "optimization_priority", ["memory", "cpu", "disk", "parallelization", "checkpointing"]),
            auto_optimization=config_dict.get("auto_optimization", False),
            monitoring_interval_seconds=config_dict.get("monitoring_interval_seconds", 60)
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "memory_optimization": self.memory_optimization.to_dict(),
            "cpu_optimization": self.cpu_optimization.to_dict(),
            "disk_optimization": self.disk_optimization.to_dict(),
            "parallelization": self.parallelization.to_dict(),
            "checkpointing": self.checkpointing.to_dict(),
            "enabled_optimizations": self.enabled_optimizations,
            "optimization_priority": self.optimization_priority,
            "auto_optimization": self.auto_optimization,
            "monitoring_interval_seconds": self.monitoring_interval_seconds
        }

    def get_enabled_configs(self) -> List[tuple]:
        """获取启用的配置项"""
        enabled_configs = []
        if "memory" in self.enabled_optimizations:
            enabled_configs.append(("memory", self.memory_optimization))
        if "cpu" in self.enabled_optimizations:
            enabled_configs.append(("cpu", self.cpu_optimization))
        if "disk" in self.enabled_optimizations:
            enabled_configs.append(("disk", self.disk_optimization))
        if "parallelization" in self.enabled_optimizations:
            enabled_configs.append(("parallelization", self.parallelization))
        if "checkpointing" in self.enabled_optimizations:
            enabled_configs.append(("checkpointing", self.checkpointing))

        return enabled_configs

    def validate(self) -> List[str]:
        """验证配置的有效性"""
        issues = []

        # 验证线程池大小
        if self.parallelization.thread_pool_size <= 0:
            issues.append("thread_pool_size必须大于0")

        # 验证检查点间隔
        if self.checkpointing.interval_seconds <= 0:
            issues.append("checkpointing.interval_seconds必须大于0")

        # 验证监控间隔
        if self.monitoring_interval_seconds <= 0:
            issues.append("monitoring_interval_seconds必须大于0")

        return issues
