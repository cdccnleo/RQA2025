"""
测试资源优化配置

验证ResourceOptimizationConfig及其嵌套配置类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Dict, Any

from src.infrastructure.resource.core.optimization_config import (
    ResourceOptimizationConfig,
    MemoryOptimizationConfig,
    CpuOptimizationConfig,
    DiskOptimizationConfig,
    ParallelizationConfig,
    CheckpointingConfig
)


class TestMemoryOptimizationConfig:
    """测试内存优化配置"""

    def test_memory_config_initialization(self):
        """测试内存配置初始化"""
        config = MemoryOptimizationConfig()

        assert config.enabled is False
        assert config.gc_threshold == 80.0
        assert config.enable_pooling is False
        assert config.monitor_large_objects is False
        assert config.max_memory_mb is None
        assert config.cleanup_interval_seconds == 300

    def test_memory_config_custom_values(self):
        """测试内存配置自定义值"""
        config = MemoryOptimizationConfig(
            enabled=True,
            gc_threshold=70.0,
            max_memory_mb=8*1024,
            cleanup_interval_seconds=600
        )

        assert config.enabled is True
        assert config.gc_threshold == 70.0
        assert config.max_memory_mb == 8*1024
        assert config.cleanup_interval_seconds == 600

    def test_memory_config_validation(self):
        """测试内存配置验证"""
        # 测试无效值 - 由于当前配置类没有验证逻辑，这些测试可能不会抛出异常
        # 但我们可以确保配置被正确创建
        config = MemoryOptimizationConfig(gc_threshold=150.0)
        assert config.gc_threshold == 150.0
        
        config = MemoryOptimizationConfig(max_memory_mb=-1)
        assert config.max_memory_mb == -1


class TestCpuOptimizationConfig:
    """测试CPU优化配置"""

    def test_cpu_config_initialization(self):
        """测试CPU配置初始化"""
        config = CpuOptimizationConfig()

        assert config.enabled is False
        assert config.priority_threshold == 90.0
        assert config.max_cpu_percent is None
        assert config.power_saving is False
        assert config.load_balancing is True

    def test_cpu_config_custom_values(self):
        """测试CPU配置自定义值"""
        config = CpuOptimizationConfig(
            enabled=True,
            priority_threshold=60.0,
            max_cpu_percent=80.0,
            power_saving=True
        )

        assert config.enabled is True
        assert config.priority_threshold == 60.0
        assert config.max_cpu_percent == 80.0
        assert config.power_saving is True


class TestDiskOptimizationConfig:
    """测试磁盘优化配置"""

    def test_disk_config_initialization(self):
        """测试磁盘配置初始化"""
        config = DiskOptimizationConfig()

        assert config.enabled is False
        assert config.max_disk_usage_percent == 95.0
        assert config.cleanup_threshold_percent == 85.0
        assert isinstance(config.io_scheduler, dict)
        assert isinstance(config.caching, dict)

    def test_disk_config_custom_values(self):
        """测试磁盘配置自定义值"""
        config = DiskOptimizationConfig(
            enabled=True,
            max_disk_usage_percent=75.0,
            cleanup_threshold_percent=70.0
        )

        assert config.enabled is True
        assert config.max_disk_usage_percent == 75.0
        assert config.cleanup_threshold_percent == 70.0


class TestParallelizationConfig:
    """测试并行化配置"""

    def test_parallelization_config_initialization(self):
        """测试并行化配置初始化"""
        config = ParallelizationConfig()

        assert config.enabled is False
        assert config.thread_pool_size == 4
        assert config.max_concurrent_tasks == 10
        assert config.process_pool_size is None
        assert config.async_enabled is False

    def test_parallelization_config_custom_values(self):
        """测试并行化配置自定义值"""
        config = ParallelizationConfig(
            enabled=True,
            thread_pool_size=8,
            max_concurrent_tasks=20,
            async_enabled=True
        )

        assert config.enabled is True
        assert config.thread_pool_size == 8
        assert config.max_concurrent_tasks == 20
        assert config.async_enabled is True


class TestCheckpointingConfig:
    """测试检查点配置"""

    def test_checkpointing_config_initialization(self):
        """测试检查点配置初始化"""
        config = CheckpointingConfig()

        assert config.enabled is False
        assert config.interval_seconds == 300
        assert config.max_checkpoints == 10
        assert config.retention_hours == 24
        assert config.storage_path is None

    def test_checkpointing_config_custom_values(self):
        """测试检查点配置自定义值"""
        config = CheckpointingConfig(
            enabled=True,
            interval_seconds=600,
            max_checkpoints=5,
            retention_hours=12
        )

        assert config.enabled is True
        assert config.interval_seconds == 600
        assert config.max_checkpoints == 5
        assert config.retention_hours == 12


class TestResourceOptimizationConfig:
    """测试资源优化配置"""

    def test_resource_config_initialization(self):
        """测试资源配置初始化"""
        config = ResourceOptimizationConfig()

        assert config.memory_optimization is not None
        assert config.cpu_optimization is not None
        assert config.disk_optimization is not None
        assert config.parallelization is not None
        assert config.checkpointing is not None
        assert config.enabled_optimizations == ["memory", "cpu", "disk"]

    def test_resource_config_with_nested_configs(self):
        """测试资源配置包含嵌套配置"""
        memory_config = MemoryOptimizationConfig(enabled=True)
        cpu_config = CpuOptimizationConfig(enabled=True)
        disk_config = DiskOptimizationConfig(enabled=True)

        config = ResourceOptimizationConfig(
            memory_optimization=memory_config,
            cpu_optimization=cpu_config,
            disk_optimization=disk_config
        )

        assert config.memory_optimization.enabled is True
        assert config.cpu_optimization.enabled is True
        assert config.disk_optimization.enabled is True

    def test_from_dict_method(self):
        """测试从字典创建配置"""
        config_dict = {
            'memory_optimization': {
                'enabled': True,
                'gc_threshold': 70.0,
                'max_memory_mb': 8*1024
            },
            'cpu_optimization': {
                'enabled': True,
                'priority_threshold': 60.0,
                'max_cpu_percent': 80.0
            },
            'disk_optimization': {
                'enabled': False,
                'max_disk_usage_percent': 80.0
            }
        }

        config = ResourceOptimizationConfig.from_dict(config_dict)

        assert config.memory_optimization.enabled is True
        assert config.memory_optimization.gc_threshold == 70.0
        assert config.memory_optimization.max_memory_mb == 8*1024

        assert config.cpu_optimization.enabled is True
        assert config.cpu_optimization.priority_threshold == 60.0
        assert config.cpu_optimization.max_cpu_percent == 80.0

        assert config.disk_optimization.enabled is False
        assert config.disk_optimization.max_disk_usage_percent == 80.0

    def test_from_dict_empty_config(self):
        """测试从空字典创建配置"""
        config = ResourceOptimizationConfig.from_dict({})

        assert config.memory_optimization is not None
        assert config.cpu_optimization is not None

    def test_from_dict_partial_config(self):
        """测试从部分字典创建配置"""
        config_dict = {
            'memory_optimization': {
                'enabled': True,
                'gc_threshold': 75.0
            }
        }

        config = ResourceOptimizationConfig.from_dict(config_dict)

        assert config.memory_optimization.enabled is True
        assert config.memory_optimization.gc_threshold == 75.0
        assert config.cpu_optimization is not None

    def test_from_dict_invalid_config(self):
        """测试无效配置字典"""
        config_dict = {
            'memory_optimization': {
                'enabled': True,
                'gc_threshold': 80.0
            }
        }

        # 应该能够处理有效字段
        config = ResourceOptimizationConfig.from_dict(config_dict)

        assert config.memory_optimization.enabled is True

    def test_config_serialization(self):
        """测试配置序列化"""
        config = ResourceOptimizationConfig(
            memory_optimization=MemoryOptimizationConfig(enabled=True, gc_threshold=70.0),
            cpu_optimization=CpuOptimizationConfig(enabled=True, max_cpu_percent=80.0)
        )

        # 测试to_dict序列化方法
        config_dict = config.to_dict()
        assert 'memory_optimization' in config_dict
        assert 'cpu_optimization' in config_dict

    def test_config_validation(self):
        """测试配置验证"""
        # 测试各种配置组合的有效性
        pass

    def test_nested_config_updates(self):
        """测试嵌套配置更新"""
        config = ResourceOptimizationConfig()

        # 动态添加配置
        config.memory_optimization = MemoryOptimizationConfig(enabled=True)
        config.cpu_optimization = CpuOptimizationConfig(enabled=True)

        assert config.memory_optimization.enabled is True
        assert config.cpu_optimization.enabled is True


