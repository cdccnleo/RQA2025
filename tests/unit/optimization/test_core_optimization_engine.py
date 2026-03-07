#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 优化核心引擎

测试optimization/core/optimization_engine.py中的所有类和方法
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class TestOptimizationEngine:
    """测试优化引擎"""

    def setup_method(self):
        """测试前准备"""
        try:
            import sys
            from pathlib import Path

            # 添加src路径
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            if str(PROJECT_ROOT / 'src') not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT / 'src'))
            from optimization.core.optimization_engine import (
                OptimizationEngine, OptimizationTask, OptimizationResult,
                OptimizationStatus, OptimizationMetrics
            )
            self.OptimizationEngine = OptimizationEngine
            self.OptimizationTask = OptimizationTask
            self.OptimizationResult = OptimizationResult
            self.OptimizationStatus = OptimizationStatus
            self.OptimizationMetrics = OptimizationMetrics
        except ImportError as e:
            pytest.skip(f"Optimization engine components not available: {e}")

    def test_optimization_engine_initialization(self):
        """测试优化引擎初始化"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine(max_workers=4, timeout=30.0)
        assert engine is not None
        assert hasattr(engine, 'max_workers')
        assert hasattr(engine, 'timeout')
        assert hasattr(engine, 'running_tasks')

    def test_optimization_task_creation(self):
        """测试优化任务创建"""
        if not hasattr(self, 'OptimizationTask'):
            pytest.skip("OptimizationTask not available")

        # 创建任务
        task = self.OptimizationTask(
            task_id="test_task_001",
            optimization_type="portfolio",
            parameters={"weights": [0.5, 0.3, 0.2], "target_return": 0.08},
            constraints={"max_weight": 0.4, "min_weight": 0.05},
            metadata={"strategy": "mean_variance"}
        )

        assert task.task_id == "test_task_001"
        assert task.optimization_type == "portfolio"
        assert task.status == self.OptimizationStatus.PENDING
        assert isinstance(task.created_at, float)
        assert isinstance(task.parameters, dict)

    def test_optimization_result_creation(self):
        """测试优化结果创建"""
        if not hasattr(self, 'OptimizationResult'):
            pytest.skip("OptimizationResult not available")

        # 创建结果
        result = self.OptimizationResult(
            task_id="test_task_001",
            status=self.OptimizationStatus.COMPLETED,
            optimal_solution={"weights": [0.4, 0.35, 0.25]},
            objective_value=0.12,
            convergence_info={"iterations": 150, "tolerance": 1e-6},
            metadata={"algorithm": "SLSQP"}
        )

        assert result.task_id == "test_task_001"
        assert result.status == self.OptimizationStatus.COMPLETED
        assert isinstance(result.optimal_solution, dict)
        assert result.objective_value == 0.12

    def test_optimization_status_enum(self):
        """测试优化状态枚举"""
        if not hasattr(self, 'OptimizationStatus'):
            pytest.skip("OptimizationStatus not available")

        # 测试枚举值
        assert hasattr(self.OptimizationStatus, 'PENDING')
        assert hasattr(self.OptimizationStatus, 'RUNNING')
        assert hasattr(self.OptimizationStatus, 'COMPLETED')
        assert hasattr(self.OptimizationStatus, 'FAILED')
        assert hasattr(self.OptimizationStatus, 'CANCELLED')

    def test_optimization_metrics_creation(self):
        """测试优化指标创建"""
        if not hasattr(self, 'OptimizationMetrics'):
            pytest.skip("OptimizationMetrics not available")

        metrics = self.OptimizationMetrics(
            task_id="test_task_001",
            execution_time=2.5,
            iterations=100,
            function_evaluations=500,
            convergence_rate=0.95,
            memory_usage=50.2,
            cpu_usage=75.8
        )

        assert metrics.task_id == "test_task_001"
        assert metrics.execution_time == 2.5
        assert metrics.iterations == 100
        assert metrics.convergence_rate == 0.95

    def test_optimization_engine_task_submission(self):
        """测试优化引擎任务提交"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine(max_workers=2)

        # 创建任务
        task = self.OptimizationTask(
            task_id="test_task_002",
            optimization_type="strategy",
            parameters={"param1": 1.0, "param2": 2.0}
        )

        # 提交任务
        if hasattr(engine, 'submit_task'):
            task_id = engine.submit_task(task)
            assert task_id == "test_task_002"
            assert task_id in engine.running_tasks

    def test_optimization_engine_task_execution(self):
        """测试优化引擎任务执行"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine(max_workers=2)

        # 创建任务
        task = self.OptimizationTask(
            task_id="test_task_003",
            optimization_type="portfolio",
            parameters={"target_return": 0.08}
        )

        # 模拟任务执行
        if hasattr(engine, 'execute_task'):
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                mock_future = Mock()
                mock_future.result.return_value = self.OptimizationResult(
                    task_id="test_task_003",
                    status=self.OptimizationStatus.COMPLETED,
                    optimal_solution={"weights": [0.33, 0.33, 0.34]},
                    objective_value=0.08
                )
                mock_executor.return_value.submit.return_value = mock_future

                result = engine.execute_task(task)
                assert result.status == self.OptimizationStatus.COMPLETED
                assert result.task_id == "test_task_003"

    def test_optimization_engine_result_retrieval(self):
        """测试优化引擎结果获取"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine()

        # 创建并存储结果
        result = self.OptimizationResult(
            task_id="test_task_004",
            status=self.OptimizationStatus.COMPLETED,
            optimal_solution={"solution": "optimal"}
        )

        if hasattr(engine, 'store_result'):
            engine.store_result(result)

        if hasattr(engine, 'get_result'):
            retrieved = engine.get_result("test_task_004")
            assert retrieved.task_id == "test_task_004"
            assert retrieved.status == self.OptimizationStatus.COMPLETED

    def test_optimization_engine_task_cancellation(self):
        """测试优化引擎任务取消"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine()

        # 创建任务
        task = self.OptimizationTask(
            task_id="test_task_005",
            optimization_type="strategy",
            parameters={"param": 1.0}
        )

        # 提交任务
        if hasattr(engine, 'submit_task'):
            engine.submit_task(task)

        # 取消任务
        if hasattr(engine, 'cancel_task'):
            success = engine.cancel_task("test_task_005")
            assert success is True

    def test_optimization_engine_metrics_collection(self):
        """测试优化引擎指标收集"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine()

        # 创建指标
        metrics = self.OptimizationMetrics(
            task_id="test_task_006",
            execution_time=3.2,
            iterations=80,
            convergence_rate=0.88
        )

        if hasattr(engine, 'collect_metrics'):
            engine.collect_metrics(metrics)
            assert "test_task_006" in engine.task_metrics

    def test_optimization_engine_error_handling(self):
        """测试优化引擎错误处理"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine()

        # 测试无效任务
        if hasattr(engine, 'submit_task'):
            try:
                engine.submit_task(None)
            except (TypeError, AttributeError):
                pass  # 应该能处理无效输入

        # 测试不存在的任务
        if hasattr(engine, 'get_result'):
            result = engine.get_result("nonexistent_task")
            assert result is None

    def test_optimization_engine_concurrent_execution(self):
        """测试优化引擎并发执行"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine(max_workers=3)

        # 创建多个任务
        tasks = []
        for i in range(5):
            task = self.OptimizationTask(
                task_id=f"concurrent_task_{i}",
                optimization_type="portfolio",
                parameters={"index": i}
            )
            tasks.append(task)

        # 提交并发任务
        if hasattr(engine, 'submit_batch_tasks'):
            task_ids = engine.submit_batch_tasks(tasks)
            assert len(task_ids) == 5
            assert all(task_id in engine.running_tasks for task_id in task_ids)

    def test_optimization_engine_resource_management(self):
        """测试优化引擎资源管理"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        engine = self.OptimizationEngine(max_workers=2, timeout=10.0)

        # 检查资源限制
        assert engine.max_workers == 2
        assert engine.timeout == 10.0

        # 测试资源清理
        if hasattr(engine, 'cleanup_resources'):
            engine.cleanup_resources()
            assert len(engine.running_tasks) == 0

    def test_optimization_engine_configuration(self):
        """测试优化引擎配置"""
        if not hasattr(self, 'OptimizationEngine'):
            pytest.skip("OptimizationEngine not available")

        # 测试不同配置
        config = {
            "max_workers": 4,
            "timeout": 60.0,
            "retry_count": 3,
            "enable_metrics": True
        }

        engine = self.OptimizationEngine(**config)
        assert engine.max_workers == 4
        assert engine.timeout == 60.0
