#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云自动伸缩管理器深度测试
测试 AutoScalingManager 的完整功能覆盖和边界条件
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.infrastructure.config.environment.cloud_auto_scaling import AutoScalingManager
from src.infrastructure.config.environment.cloud_native_configs import AutoScalingConfig, ScalingPolicy


class TestAutoScalingManagerComprehensive(unittest.TestCase):
    """自动伸缩管理器深度测试"""

    def setUp(self):
        """测试前准备"""
        self.config = AutoScalingConfig(
            enabled=True,
            min_replicas=1,
            max_replicas=10,
            target_cpu_utilization=70,
            target_memory_utilization=80,
            scale_up_threshold=80,
            scale_down_threshold=30,
            stabilization_window_seconds=300,
            scaling_policy=ScalingPolicy.CPU_UTILIZATION,
            custom_metrics=["response_time", "error_rate"],
            cooldown_period_seconds=60
        )
        self.manager = AutoScalingManager(self.config)

    def tearDown(self):
        """测试后清理"""
        if hasattr(self, 'manager'):
            pass

    # ==================== 初始化测试 ====================

    def test_initialization_with_valid_config(self):
        """测试使用有效配置初始化"""
        manager = AutoScalingManager(self.config)
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager._current_replicas, self.config.min_replicas)
        self.assertEqual(manager._scale_up_count, 0)
        self.assertEqual(manager._scale_down_count, 0)
        self.assertIsInstance(manager._scaling_history, list)
        self.assertIsInstance(manager._metrics_cache, dict)

    def test_initialization_with_disabled_config(self):
        """测试使用禁用配置初始化"""
        disabled_config = AutoScalingConfig(enabled=False)
        manager = AutoScalingManager(disabled_config)
        self.assertFalse(manager.config.enabled)
        self.assertEqual(manager._current_replicas, disabled_config.min_replicas)

    # ==================== 扩容判断测试 ====================

    def test_should_scale_up_disabled(self):
        """测试禁用状态下不扩容"""
        disabled_config = AutoScalingConfig(enabled=False)
        manager = AutoScalingManager(disabled_config)

        metrics = {"cpu": 95.0}
        self.assertFalse(manager.should_scale_up(metrics))

    def test_should_scale_up_in_cooldown(self):
        """测试冷却期内不扩容"""
        # 先触发一次扩容进入冷却期
        self.manager._enter_cooldown()
        metrics = {"cpu": 95.0}
        self.assertFalse(self.manager.should_scale_up(metrics))

    def test_should_scale_up_at_max_replicas(self):
        """测试达到最大副本数时不扩容"""
        self.manager._current_replicas = self.config.max_replicas
        metrics = {"cpu": 95.0}
        self.assertFalse(self.manager.should_scale_up(metrics))

    def test_should_scale_up_cpu_based(self):
        """测试基于CPU的扩容判断"""
        metrics = {"cpu_percent": 85.0}  # 超过阈值
        self.assertTrue(self.manager.should_scale_up(metrics))

    def test_should_scale_up_cpu_below_threshold(self):
        """测试CPU低于阈值时不扩容"""
        metrics = {"cpu_percent": 60.0}  # 低于阈值
        self.assertFalse(self.manager.should_scale_up(metrics))

    def test_should_scale_up_memory_based(self):
        """测试基于内存的扩容判断"""
        memory_config = AutoScalingConfig(
            scaling_policy=ScalingPolicy.MEMORY_UTILIZATION,
            target_memory_utilization=75
        )
        manager = AutoScalingManager(memory_config)

        metrics = {"memory_percent": 85.0}  # 超过阈值
        self.assertTrue(manager.should_scale_up(metrics))

    def test_should_scale_up_custom_metrics(self):
        """测试基于自定义指标的扩容判断"""
        custom_config = AutoScalingConfig(
            scaling_policy=ScalingPolicy.CUSTOM_METRIC,
            custom_metrics=["response_time"],
            scale_up_threshold=1000  # 响应时间超过1秒
        )
        manager = AutoScalingManager(custom_config)

        metrics = {"response_time": 1200}  # 响应时间过长
        self.assertTrue(manager.should_scale_up(metrics))

    def test_should_scale_up_request_rate(self):
        """测试基于请求率的扩容判断"""
        rate_config = AutoScalingConfig(scaling_policy=ScalingPolicy.REQUEST_RATE)
        manager = AutoScalingManager(rate_config)

        metrics = {"requests_per_second": 1500}  # 请求率过高
        self.assertTrue(manager.should_scale_up(metrics))

    # ==================== 缩容判断测试 ====================

    def test_should_scale_down_disabled(self):
        """测试禁用状态下不缩容"""
        disabled_config = AutoScalingConfig(enabled=False)
        manager = AutoScalingManager(disabled_config)

        metrics = {"cpu": 20.0}
        self.assertFalse(manager.should_scale_down(metrics))

    def test_should_scale_down_in_cooldown(self):
        """测试冷却期内不缩容"""
        self.manager._enter_cooldown()
        metrics = {"cpu": 20.0}
        self.assertFalse(self.manager.should_scale_down(metrics))

    def test_should_scale_down_at_min_replicas(self):
        """测试达到最小副本数时不缩容"""
        self.manager._current_replicas = self.config.min_replicas
        metrics = {"cpu": 20.0}
        self.assertFalse(self.manager.should_scale_down(metrics))

    def test_should_scale_down_cpu_based(self):
        """测试基于CPU的缩容判断"""
        # 先设置当前副本数高于最小值
        self.manager._current_replicas = 5
        metrics = {"cpu": 20.0}  # 低于阈值
        self.assertTrue(self.manager.should_scale_down(metrics))

    def test_should_scale_down_memory_based(self):
        """测试基于内存的缩容判断"""
        memory_config = AutoScalingConfig(scaling_policy=ScalingPolicy.MEMORY_UTILIZATION)
        manager = AutoScalingManager(memory_config)
        manager._current_replicas = 5

        metrics = {"memory": 25.0}  # 内存使用率低
        self.assertTrue(manager.should_scale_down(metrics))

    # ==================== 扩容操作测试 ====================

    @patch('src.infrastructure.config.environment.cloud_auto_scaling.datetime')
    @patch.object(AutoScalingManager, 'should_scale_up', return_value=True)
    @patch.object(AutoScalingManager, '_perform_scale_operation', return_value=True)
    def test_scale_up_success(self, mock_perform, mock_should_scale, mock_datetime):
        """测试扩容成功"""
        mock_datetime.now.return_value.isoformat.return_value = "2025-01-13T10:00:00"
        mock_datetime.now.return_value.timestamp.return_value = time.time()

        result = self.manager.scale_up("测试扩容")
        self.assertTrue(result)
        self.assertEqual(self.manager._current_replicas, 2)  # 从1扩容到2
        self.assertEqual(self.manager._scale_up_count, 1)
        self.assertEqual(len(self.manager._scaling_history), 1)

    def test_scale_up_at_max_capacity(self):
        """测试达到最大容量时扩容失败"""
        self.manager._current_replicas = self.config.max_replicas

        result = self.manager.scale_up("测试扩容")
        self.assertFalse(result)
        self.assertEqual(self.manager._current_replicas, self.config.max_replicas)

    def test_scale_up_in_cooldown(self):
        """测试冷却期内扩容失败"""
        self.manager._enter_cooldown()

        result = self.manager.scale_up("测试扩容")
        self.assertFalse(result)

    # ==================== 缩容操作测试 ====================

    def test_scale_down_success(self):
        """测试缩容成功"""
        # 先设置当前副本数
        self.manager._current_replicas = 5

        result = self.manager.scale_down("测试缩容")
        self.assertTrue(result)
        self.assertEqual(self.manager._current_replicas, 4)  # 从5缩容到4
        self.assertEqual(self.manager._scale_down_count, 1)

    def test_scale_down_at_min_capacity(self):
        """测试达到最小容量时缩容失败"""
        self.manager._current_replicas = self.config.min_replicas

        result = self.manager.scale_down("测试缩容")
        self.assertFalse(result)
        self.assertEqual(self.manager._current_replicas, self.config.min_replicas)

    # ==================== 手动伸缩测试 ====================

    def test_manual_scale_valid_range(self):
        """测试手动伸缩到有效范围"""
        result = self.manager.manual_scale(5, "手动伸缩测试")
        self.assertTrue(result)
        self.assertEqual(self.manager._current_replicas, 5)
        self.assertEqual(len(self.manager._scaling_history), 1)

    def test_manual_scale_above_max(self):
        """测试手动伸缩超过最大值"""
        result = self.manager.manual_scale(15, "超过最大值")
        self.assertFalse(result)
        self.assertEqual(self.manager._current_replicas, self.config.min_replicas)  # 保持不变

    def test_manual_scale_below_min(self):
        """测试手动伸缩低于最小值"""
        result = self.manager.manual_scale(0, "低于最小值")
        self.assertFalse(result)
        self.assertEqual(self.manager._current_replicas, self.config.min_replicas)  # 保持不变

    # ==================== 状态查询测试 ====================

    def test_get_current_replicas(self):
        """测试获取当前副本数"""
        self.assertEqual(self.manager.get_current_replicas(), self.config.min_replicas)

    def test_get_scaling_status(self):
        """测试获取伸缩状态"""
        status = self.manager.get_scaling_status()
        self.assertIsInstance(status, dict)
        self.assertIn("current_replicas", status)
        self.assertIn("enabled", status)
        self.assertIn("scale_up_count", status)
        self.assertIn("scale_down_count", status)
        self.assertIn("in_cooldown", status)

    def test_get_scaling_history_empty(self):
        """测试获取空的伸缩历史"""
        history = self.manager.get_scaling_history()
        self.assertEqual(history, [])

    def test_get_scaling_history_with_events(self):
        """测试获取有事件的伸缩历史"""
        self.manager.manual_scale(3, "测试历史记录")

        history = self.manager.get_scaling_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["new_replicas"], 3)
        self.assertEqual(history[0]["reason"], "测试历史记录")

    def test_get_scaling_history_with_limit(self):
        """测试获取限制数量的伸缩历史"""
        # 添加多个历史记录
        for i in range(5):
            self.manager.manual_scale(2 + i, f"测试记录{i}")

        history = self.manager.get_scaling_history(limit=3)
        self.assertEqual(len(history), 3)

    # ==================== 指标处理测试 ====================

    def test_update_metrics(self):
        """测试更新指标"""
        metrics = {"cpu": 75.0, "memory": 60.0, "response_time": 500}
        self.manager.update_metrics(metrics)

        self.assertIn("cpu", self.manager._metrics_cache)
        self.assertIn("memory", self.manager._metrics_cache)
        self.assertIn("response_time", self.manager._metrics_cache)

    def test_get_average_metric(self):
        """测试获取平均指标"""
        # 添加一些指标数据
        for i in range(5):
            self.manager.update_metrics({"cpu": 70.0 + i})

        avg_cpu = self.manager.get_average_metric("cpu", window_size=3)
        self.assertIsInstance(avg_cpu, float)
        self.assertGreater(avg_cpu, 70.0)

    def test_get_average_metric_no_data(self):
        """测试获取不存在指标的平均值"""
        avg = self.manager.get_average_metric("nonexistent_metric")
        self.assertIsNone(avg)

    # ==================== 配置验证测试 ====================

    def test_validate_scaling_config_valid(self):
        """测试验证有效配置"""
        result = self.manager.validate_scaling_config()
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("valid", False))

    def test_validate_scaling_config_invalid_min_max(self):
        """测试验证无效的min/max配置"""
        invalid_config = AutoScalingConfig(min_replicas=10, max_replicas=5)
        manager = AutoScalingManager(invalid_config)

        result = manager.validate_scaling_config()
        self.assertFalse(result.get("valid", True))
        self.assertIn("errors", result)

    def test_validate_scaling_config_invalid_thresholds(self):
        """测试验证无效阈值配置"""
        invalid_config = AutoScalingConfig(
            scale_up_threshold=30,
            scale_down_threshold=80  # 缩容阈值高于扩容阈值（无效）
        )
        manager = AutoScalingManager(invalid_config)

        result = manager.validate_scaling_config()
        self.assertFalse(result.get("valid", True))

    # ==================== 冷却期测试 ====================

    def test_cooldown_period(self):
        """测试冷却期功能"""
        # 初始状态不应在冷却期
        self.assertFalse(self.manager._is_in_cooldown())

        # 进入冷却期
        self.manager._enter_cooldown()
        self.assertTrue(self.manager._is_in_cooldown())

    # ==================== 统计重置测试 ====================

    def test_reset_statistics(self):
        """测试重置统计信息"""
        # 先进行一些操作
        self.manager.manual_scale(3, "测试")
        self.manager.manual_scale(5, "测试2")

        # 重置统计
        self.manager.reset_statistics()

        self.assertEqual(self.manager._scale_up_count, 0)
        self.assertEqual(self.manager._scale_down_count, 0)
        # 注意：历史记录可能保留或清空，取决于实现

    # ==================== 边界条件测试 ====================

    def test_edge_case_min_equals_max_replicas(self):
        """测试最小副本数等于最大副本数的边界情况"""
        edge_config = AutoScalingConfig(min_replicas=5, max_replicas=5)
        manager = AutoScalingManager(edge_config)

        # 不应该能扩容或缩容
        metrics = {"cpu": 95.0}
        self.assertFalse(manager.should_scale_up(metrics))

        manager._current_replicas = 5
        metrics = {"cpu": 10.0}
        self.assertFalse(manager.should_scale_down(metrics))

    def test_edge_case_zero_cooldown(self):
        """测试零冷却期的情况"""
        zero_cooldown_config = AutoScalingConfig(cooldown_period_seconds=0)
        manager = AutoScalingManager(zero_cooldown_config)

        # 即使进入冷却期，也应该立即可用
        manager._enter_cooldown()
        # 零冷却期应该立即过期
        self.assertFalse(manager._is_in_cooldown())

    def test_edge_case_large_metrics_window(self):
        """测试大指标窗口的情况"""
        # 添加很多指标数据
        for i in range(100):
            self.manager.update_metrics({"cpu": 70.0 + (i % 20)})

        # 获取大窗口的平均值
        avg = self.manager.get_average_metric("cpu", window_size=50)
        self.assertIsInstance(avg, float)

    # ==================== 并发安全测试 ====================

    def test_concurrent_scaling_operations(self):
        """测试并发伸缩操作的安全性"""
        import concurrent.futures

        results = []
        errors = []

        def scaling_worker(worker_id):
            try:
                # 每个线程尝试不同的操作
                if worker_id % 3 == 0:
                    result = self.manager.manual_scale(min(10, self.manager._current_replicas + 1), f"Worker {worker_id}")
                elif worker_id % 3 == 1:
                    metrics = {"cpu": 80.0 + worker_id}
                    result = self.manager.should_scale_up(metrics)
                else:
                    self.manager.update_metrics({"cpu": 60.0 + worker_id})
                    result = True

                results.append(result)
                return result
            except Exception as e:
                errors.append(str(e))
                results.append(False)
                return False

        # 启动多个线程并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(scaling_worker, i) for i in range(20)]
            concurrent.futures.wait(futures)

        # 验证没有出现异常
        self.assertEqual(len(errors), 0, f"Concurrent operations failed with errors: {errors}")
        self.assertEqual(len(results), 20)

        # 验证最终状态合理
        final_replicas = self.manager.get_current_replicas()
        self.assertGreaterEqual(final_replicas, self.config.min_replicas)
        self.assertLessEqual(final_replicas, self.config.max_replicas)

    # ==================== 异常处理测试 ====================

    def test_invalid_metrics_handling(self):
        """测试无效指标的处理"""
        # 测试None指标
        self.assertFalse(self.manager.should_scale_up(None))

        # 测试空字典
        self.assertFalse(self.manager.should_scale_up({}))

        # 测试无效的数值类型
        invalid_metrics = {"cpu": "invalid_value"}
        # 应该不会抛出异常，而是返回False
        result = self.manager.should_scale_up(invalid_metrics)
        self.assertIsInstance(result, bool)

    def test_corrupted_config_handling(self):
        """测试损坏配置的处理"""
        # 创建一个有问题的配置
        corrupted_config = AutoScalingConfig(
            min_replicas=-1,  # 无效的负数
            max_replicas=0    # 无效的零值
        )

        # 应该能够创建管理器，但验证会失败
        manager = AutoScalingManager(corrupted_config)
        validation = manager.validate_scaling_config()

        # 验证应该检测到问题
        self.assertFalse(validation.get("valid", True))

    def test_extreme_values_handling(self):
        """测试极端值的处理"""
        # 测试非常大的指标值
        extreme_metrics = {"cpu": 999999.0, "memory": -50.0}
        result = self.manager.should_scale_up(extreme_metrics)
        self.assertIsInstance(result, bool)

        # 测试非常小的副本数目标
        result = self.manager.manual_scale(-100, "Extreme test")
        self.assertFalse(result)

        # 测试非常大的副本数目标
        result = self.manager.manual_scale(10000, "Extreme test")
        self.assertFalse(result)

    # ==================== 性能测试 ====================

    @pytest.mark.skip(reason="Performance test with large history - resource intensive in CI")
    def test_performance_large_history(self):
        """测试大量历史记录的性能"""
        start_time = time.time()

        # 添加大量历史记录
        for i in range(1000):
            self.manager.manual_scale(
                min(self.config.max_replicas, self.config.min_replicas + (i % 5)),
                f"Performance test {i}"
            )

        # 测试获取历史记录的性能
        history = self.manager.get_scaling_history(limit=100)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能在合理范围内
        self.assertLess(duration, 2.0, f"Large history operations took too long: {duration}s")
        self.assertEqual(len(history), 100)

    def test_memory_efficiency_metrics_cache(self):
        """测试指标缓存的内存效率"""
        # 添加大量指标数据
        for i in range(1000):
            metrics = {f"metric_{j}": float(i + j) for j in range(10)}
            self.manager.update_metrics(metrics)

        # 验证缓存不会无限增长（假设有清理机制）
        total_cached_values = sum(len(values) for values in self.manager._metrics_cache.values())

        # 即使有清理机制，也应该有合理的限制
        self.assertLess(total_cached_values, 10000, "Metrics cache grew too large")

    # ==================== 集成场景测试 ====================

    def test_complete_auto_scaling_workflow(self):
        """测试完整的自动伸缩工作流程"""
        # 1. 初始状态检查
        self.assertEqual(self.manager.get_current_replicas(), 1)
        self.assertFalse(self.manager._is_in_cooldown())

        # 2. 模拟高负载场景
        high_load_metrics = {
            "cpu_percent": 90.0,
            "memory": 85.0,
            "request_rate": 200,
            "response_time": 1500
        }

        # 3. 应该触发扩容
        self.assertTrue(self.manager.should_scale_up(high_load_metrics))

        # 4. 执行扩容
        result = self.manager.scale_up("高负载扩容")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_replicas(), 2)

        # 5. 检查历史记录
        history = self.manager.get_scaling_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["reason"], "高负载扩容")

        # 6. 模拟负载降低
        low_load_metrics = {
            "cpu_percent": 20.0,
            "memory_percent": 25.0,
            "request_rate": 50,
            "response_time": 200
        }

        # 重置冷却状态以便测试缩容
        self.manager._cooldown_active = False
        self.manager._last_scale_time = 0

        # 7. 设置更多的副本数以便缩容
        self.manager._current_replicas = 2  # 设置为2，这样缩容一次可以到1
        # 应该触发缩容
        self.assertTrue(self.manager.should_scale_down(low_load_metrics))

        # 8. 执行缩容
        result = self.manager.scale_down("负载降低缩容")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_replicas(), 1)

        # 9. 最终状态检查
        final_status = self.manager.get_scaling_status()
        self.assertEqual(final_status["current_replicas"], 1)
        self.assertEqual(final_status["scale_up_count"], 1)
        self.assertEqual(final_status["scale_down_count"], 1)

    def test_policy_switching_scenario(self):
        """测试策略切换场景"""
        # 初始为CPU策略
        self.assertEqual(self.manager.config.scaling_policy, ScalingPolicy.CPU_UTILIZATION)

        # 创建基于内存的策略管理器
        memory_config = AutoScalingConfig(scaling_policy=ScalingPolicy.MEMORY_UTILIZATION)
        memory_manager = AutoScalingManager(memory_config)

        # 测试不同策略的扩容判断
        cpu_metrics = {"cpu_percent": 85.0}
        memory_metrics = {"memory_percent": 90.0}

        # CPU策略对CPU指标响应
        self.assertTrue(self.manager.should_scale_up(cpu_metrics))
        self.assertFalse(self.manager.should_scale_up(memory_metrics))

        # 内存策略对内存指标响应
        self.assertTrue(memory_manager.should_scale_up(memory_metrics))
        self.assertFalse(memory_manager.should_scale_up(cpu_metrics))


if __name__ == '__main__':
    unittest.main()
