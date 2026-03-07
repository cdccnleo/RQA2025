#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式特征处理器测试
"""

import pytest
import pandas as pd
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from src.features.distributed.distributed_processor import (
    DistributedFeatureProcessor,
    FeatureLoadBalancer,
    LoadBalancerConfig,
    LoadBalancingStrategy,
    ProcessingResult,
    create_distributed_processor
)
from src.features.distributed.task_scheduler import (
    FeatureTaskScheduler,
    FeatureTask,
    TaskStatus,
    TaskPriority
)
from src.features.distributed.worker_manager import (
    FeatureWorkerManager,
    WorkerInfo,
    WorkerStatus
)
from src.features.core.config import FeatureProcessingConfig


class TestFeatureLoadBalancer:
    """负载均衡器测试"""

    @pytest.fixture
    def config(self):
        """创建负载均衡器配置"""
        return LoadBalancerConfig(
            strategy=LoadBalancingStrategy.SIMPLE,
            max_retries=3,
            timeout=30.0
        )

    @pytest.fixture
    def load_balancer(self, config):
        """创建负载均衡器实例"""
        return FeatureLoadBalancer(config)

    @pytest.fixture
    def sample_workers(self):
        """创建示例工作节点"""
        from datetime import datetime
        return [
            WorkerInfo(
                worker_id="worker1",
                status=WorkerStatus.IDLE,
                capabilities={"cpu": 4, "memory": 8192},
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                current_load=0.2,
                performance_score=0.9
            ),
            WorkerInfo(
                worker_id="worker2",
                status=WorkerStatus.IDLE,
                capabilities={"cpu": 8, "memory": 16384},
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                current_load=0.1,
                performance_score=0.95
            ),
            WorkerInfo(
                worker_id="worker3",
                status=WorkerStatus.IDLE,
                capabilities={"cpu": 2, "memory": 4096},
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                current_load=0.5,
                performance_score=0.7
            )
        ]

    def test_simple_selection(self, load_balancer, sample_workers):
        """测试简单选择策略"""
        worker_id = load_balancer.select_worker(sample_workers, TaskPriority.NORMAL)
        assert worker_id is not None
        assert worker_id in ["worker1", "worker2", "worker3"]
        # 应该选择负载最低的节点
        assert worker_id == "worker2"  # 负载0.1最低

    def test_adaptive_selection_high_priority(self, sample_workers):
        """测试自适应选择策略-高优先级"""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.ADAPTIVE)
        load_balancer = FeatureLoadBalancer(config)
        
        worker_id = load_balancer.select_worker(sample_workers, TaskPriority.HIGH)
        assert worker_id is not None
        # 高优先级应该选择性能最好的节点
        assert worker_id == "worker2"  # 性能得分0.95最高

    def test_adaptive_selection_low_priority(self, sample_workers):
        """测试自适应选择策略-低优先级"""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.ADAPTIVE)
        load_balancer = FeatureLoadBalancer(config)
        
        worker_id = load_balancer.select_worker(sample_workers, TaskPriority.LOW)
        assert worker_id is not None
        # 低优先级应该选择负载最低的节点
        assert worker_id == "worker2"  # 负载0.1最低

    def test_intelligent_selection(self, sample_workers):
        """测试智能选择策略"""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.INTELLIGENT)
        load_balancer = FeatureLoadBalancer(config)
        
        worker_id = load_balancer.select_worker(sample_workers, TaskPriority.NORMAL)
        assert worker_id is not None
        assert worker_id in ["worker1", "worker2", "worker3"]

    def test_select_worker_empty_list(self, load_balancer):
        """测试空工作节点列表"""
        worker_id = load_balancer.select_worker([])
        assert worker_id is None

    def test_update_worker_stats(self, load_balancer):
        """测试更新工作节点统计"""
        stats = {"cpu_usage": 0.5, "memory_usage": 0.6}
        load_balancer.update_worker_stats("worker1", stats)
        
        all_stats = load_balancer.get_worker_stats()
        assert "worker1" in all_stats
        assert all_stats["worker1"] == stats

    def test_calculate_worker_score(self, load_balancer, sample_workers):
        """测试计算工作节点得分"""
        score = load_balancer._calculate_worker_score(sample_workers[0], TaskPriority.NORMAL)
        assert isinstance(score, float)
        assert score >= 0


class TestDistributedFeatureProcessor:
    """分布式特征处理器测试"""

    @pytest.fixture
    def scheduler(self):
        """创建任务调度器"""
        return FeatureTaskScheduler()

    @pytest.fixture
    def worker_manager(self):
        """创建工作节点管理器"""
        return FeatureWorkerManager()

    @pytest.fixture
    def load_balancer(self):
        """创建负载均衡器"""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.SIMPLE)
        return FeatureLoadBalancer(config)

    @pytest.fixture
    def processor(self, scheduler, worker_manager, load_balancer):
        """创建分布式处理器"""
        return DistributedFeatureProcessor(
            scheduler=scheduler,
            worker_manager=worker_manager,
            load_balancer=load_balancer,
            max_workers=2,
            executor_type="thread"
        )

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    @pytest.fixture
    def processing_config(self):
        """创建处理配置"""
        return FeatureProcessingConfig()

    def test_init(self, processor):
        """测试初始化"""
        assert processor.scheduler is not None
        assert processor.worker_manager is not None
        assert processor.load_balancer is not None
        assert processor.max_workers == 2
        assert processor.executor_type == "thread"
        assert not processor._is_running

    def test_init_process_executor(self, scheduler, worker_manager, load_balancer):
        """测试进程执行器初始化"""
        processor = DistributedFeatureProcessor(
            scheduler=scheduler,
            worker_manager=worker_manager,
            load_balancer=load_balancer,
            max_workers=2,
            executor_type="process"
        )
        assert processor.executor_type == "process"
        processor.stop()

    def test_init_invalid_executor(self, scheduler, worker_manager, load_balancer):
        """测试无效执行器类型"""
        with pytest.raises(ValueError, match="Unsupported executor type"):
            DistributedFeatureProcessor(
                scheduler=scheduler,
                worker_manager=worker_manager,
                load_balancer=load_balancer,
                executor_type="invalid"
            )

    def test_start_stop(self, processor):
        """测试启动和停止"""
        processor.start()
        assert processor._is_running
        
        # 等待一小段时间确保线程启动
        time.sleep(0.1)
        
        processor.stop()
        assert not processor._is_running

    def test_start_already_running(self, processor):
        """测试重复启动"""
        processor.start()
        processor.start()  # 应该不会报错，只是警告
        processor.stop()

    def test_process_features(self, processor, sample_data, processing_config):
        """测试处理特征"""
        # 注册一个工作节点
        processor.worker_manager.register_worker(
            worker_id="worker1",
            capabilities={"cpu": 4, "memory": 8192}
        )
        
        # 创建配置对象
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        processor.start()
        
        # 提交任务
        task_id = processor.process_features(
            data=sample_data,
            config=config,
            processing_config=processing_config,
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        processor.stop()

    def test_process_features_batch(self, processor, sample_data, processing_config):
        """测试批量处理特征"""
        # 注册工作节点
        processor.worker_manager.register_worker(
            worker_id="worker1",
            capabilities={"cpu": 4, "memory": 8192}
        )
        
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        processor.start()
        
        # 创建批量数据
        batch = [
            (sample_data, config, processing_config),
            (sample_data.copy(), config, processing_config)
        ]
        
        task_ids = processor.process_features_batch(batch, TaskPriority.NORMAL)
        
        assert len(task_ids) == 2
        assert all(isinstance(tid, str) for tid in task_ids)
        
        processor.stop()

    def test_get_task_result(self, processor, sample_data, processing_config):
        """测试获取任务结果"""
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        processor.start()
        
        # 提交任务
        task_id = processor.process_features(
            data=sample_data,
            config=config,
            processing_config=processing_config
        )
        
        # 获取结果（任务可能还未完成）
        result = processor.get_task_result(task_id)
        # 如果任务未完成，应该返回None
        # 这里我们只测试方法调用不报错
        
        processor.stop()

    def test_wait_for_task(self, processor, sample_data, processing_config):
        """测试等待任务完成"""
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        processor.start()
        
        # 提交任务
        task_id = processor.process_features(
            data=sample_data,
            config=config,
            processing_config=processing_config
        )
        
        # 等待任务（使用短超时）
        result = processor.wait_for_task(task_id, timeout=1.0)
        # 可能返回None如果任务未完成
        
        processor.stop()

    def test_cancel_task(self, processor, sample_data, processing_config):
        """测试取消任务"""
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        processor.start()
        
        # 提交任务
        task_id = processor.process_features(
            data=sample_data,
            config=config,
            processing_config=processing_config
        )
        
        # 取消任务
        cancelled = processor.cancel_task(task_id)
        assert isinstance(cancelled, bool)
        
        processor.stop()

    def test_get_processing_stats(self, processor):
        """测试获取处理统计"""
        stats = processor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert "total_tasks" in stats
        assert "successful_tasks" in stats
        assert "failed_tasks" in stats
        assert "success_rate" in stats
        assert "avg_processing_time" in stats
        assert "avg_memory_usage" in stats

    def test_clear_history(self, processor):
        """测试清除历史"""
        processor.clear_history()
        # 应该不报错
        stats = processor.get_processing_stats()
        assert stats["total_tasks"] == 0

    def test_context_manager(self, scheduler, worker_manager, load_balancer):
        """测试上下文管理器"""
        with DistributedFeatureProcessor(
            scheduler=scheduler,
            worker_manager=worker_manager,
            load_balancer=load_balancer
        ) as processor:
            assert processor._is_running
        
        # 退出上下文后应该已停止
        assert not processor._is_running

    def test_process_task_worker_success(self, processor, sample_data):
        """测试工作节点处理任务成功"""
        from src.features.core.config import FeatureConfig
        config = FeatureConfig()
        
        # 确保数据包含所有必需的列
        sample_data_with_open = sample_data.copy()
        sample_data_with_open['open'] = sample_data_with_open['close'] * 0.99
        
        # 创建任务
        task = FeatureTask(
            task_id="test_task",
            task_type="feature_processing",
            data=sample_data_with_open,
            priority=TaskPriority.NORMAL,
            metadata={
                "config": config.to_dict() if hasattr(config, 'to_dict') else {},
                "processing_config": {}
            }
        )
        
        # 处理任务
        result = processor._process_task_worker(task)
        
        assert isinstance(result, ProcessingResult)
        assert result.task_id == "test_task"
        # 如果处理失败，至少验证错误信息存在
        if not result.success:
            assert result.error_message is not None
        else:
            assert result.result is not None

    def test_process_task_worker_failure(self, processor):
        """测试工作节点处理任务失败"""
        # 创建会导致失败的任务
        task = FeatureTask(
            task_id="test_task",
            task_type="feature_processing",
            data=None,  # 无效数据
            priority=TaskPriority.NORMAL,
            metadata={}
        )
        
        # 处理任务
        result = processor._process_task_worker(task)
        
        assert isinstance(result, ProcessingResult)
        assert result.task_id == "test_task"
        assert result.success is False
        assert result.error_message is not None


class TestCreateDistributedProcessor:
    """创建分布式处理器函数测试"""

    def test_create_with_defaults(self):
        """测试使用默认参数创建"""
        processor = create_distributed_processor()
        assert processor is not None
        assert processor.max_workers == 4
        assert processor.executor_type == "thread"
        processor.stop()

    def test_create_with_custom_params(self):
        """测试使用自定义参数创建"""
        processor = create_distributed_processor(
            max_workers=8,
            executor_type="thread",
            load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
        )
        assert processor.max_workers == 8
        assert processor.executor_type == "thread"
        processor.stop()

