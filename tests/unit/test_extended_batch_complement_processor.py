#!/usr/bin/env python3
"""
扩展批次补全处理器单元测试
测试年度批次处理和STRATEGY_BACKTEST模式支持
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.batch_complement_processor import (
    BatchComplementProcessor,
    ComplementBatch
)
from src.core.orchestration.data_complement_scheduler import (
    ComplementTask,
    ComplementMode,
    ComplementPriority
)


class TestExtendedBatchComplementProcessor:
    """扩展批次补全处理器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.processor = BatchComplementProcessor()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_strategy_backtest_batch_size_calculation(self):
        """测试策略回测模式的批次大小计算"""
        # 创建策略回测模式的补全任务
        task = ComplementTask(
            task_id="strategy_backtest_test",
            source_id="strategy_backtest_data",
            data_type="stock",
            mode=ComplementMode.STRATEGY_BACKTEST,
            priority=ComplementPriority.HIGH,
            start_date=datetime.now() - timedelta(days=3650),  # 10年
            end_date=datetime.now(),
            estimated_records=100000
        )

        # 计算批次大小
        batch_size = self.processor._calculate_optimal_batch_size(task)

        # 验证策略回测模式使用年度批次（365天）
        assert batch_size == 365, f"策略回测模式应该使用365天批次，实际是{batch_size}天"

    def test_strategy_backtest_batch_creation(self):
        """测试策略回测模式的批次创建"""
        # 创建10年的策略回测任务
        start_date = datetime(2014, 1, 1)  # 10年前
        end_date = datetime(2024, 1, 1)    # 现在

        task = ComplementTask(
            task_id="strategy_backtest_batch_test",
            source_id="strategy_backtest_data",
            data_type="stock",
            mode=ComplementMode.STRATEGY_BACKTEST,
            priority=ComplementPriority.HIGH,
            start_date=start_date,
            end_date=end_date,
            estimated_records=100000
        )

        # 创建批次
        batches = self.processor.create_complement_batches(task)

        # 验证批次数量（10年数据，按年分批）
        assert len(batches) == 10, f"应该创建10个批次，实际创建了{len(batches)}个"

        # 验证每个批次的时间范围
        for i, batch in enumerate(batches):
            expected_start = start_date + timedelta(days=i * 365)
            expected_end = min(expected_start + timedelta(days=365), end_date)

            assert batch.start_date.date() == expected_start.date(), \
                f"批次{i+1}开始日期错误：期望{expected_start.date()}，实际{batch.start_date.date()}"
            assert batch.end_date.date() == expected_end.date(), \
                f"批次{i+1}结束日期错误：期望{expected_end.date()}，实际{batch.end_date.date()}"

            # 验证批次属性
            assert batch.batch_index == i + 1
            assert batch.total_batches == 10
            assert batch.task_id == task.task_id
            assert batch.source_id == task.source_id

    def test_different_priority_batch_sizes(self):
        """测试不同优先级的批次大小"""
        base_batch_size = self.processor.config['default_batch_size_days']

        test_cases = [
            (ComplementPriority.CRITICAL, base_batch_size * 0.5),
            (ComplementPriority.HIGH, base_batch_size * 0.7),
            (ComplementPriority.MEDIUM, base_batch_size * 1.0),
            (ComplementPriority.LOW, base_batch_size * 1.5),
        ]

        for priority, expected_multiplier in test_cases:
            task = ComplementTask(
                task_id=f"priority_test_{priority.value}",
                source_id="test_source",
                data_type="stock",
                mode=ComplementMode.MONTHLY,
                priority=priority,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                estimated_records=1000
            )

            batch_size = self.processor._calculate_optimal_batch_size(task)
            expected_size = int(base_batch_size * expected_multiplier)

            # 验证批次大小在合理范围内
            assert abs(batch_size - expected_size) <= 1, \
                f"优先级{priority.value}的批次大小错误：期望{expected_size}，实际{batch_size}"

    def test_data_type_batch_adjustments(self):
        """测试数据类型对批次大小的影响"""
        test_cases = [
            ("stock", 1.0),
            ("index", 0.8),
            ("macro", 2.0),
            ("news", 0.5),
        ]

        for data_type, expected_multiplier in test_cases:
            task = ComplementTask(
                task_id=f"datatype_test_{data_type}",
                source_id="test_source",
                data_type=data_type,
                mode=ComplementMode.MONTHLY,
                priority=ComplementPriority.MEDIUM,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                estimated_records=1000
            )

            batch_size = self.processor._calculate_optimal_batch_size(task)
            base_batch_size = self.processor.config['default_batch_size_days']
            expected_size = int(base_batch_size * expected_multiplier)

            # 验证批次大小调整正确
            assert abs(batch_size - expected_size) <= 1, \
                f"数据类型{data_type}的批次大小错误：期望{expected_size}，实际{batch_size}"

    def test_batch_size_limits(self):
        """测试批次大小限制"""
        config = self.processor.config
        min_size = config['min_batch_size_days']
        max_size = config['max_batch_size_days']

        # 创建一个大时间范围的任务，应该触发最大批次限制
        task = ComplementTask(
            task_id="large_range_test",
            source_id="test_source",
            data_type="macro",  # 宏观数据通常批次更大
            mode=ComplementMode.SEMI_ANNUAL,
            priority=ComplementPriority.LOW,  # 低优先级批次更大
            start_date=datetime.now() - timedelta(days=1000),
            end_date=datetime.now(),
            estimated_records=50000
        )

        batch_size = self.processor._calculate_optimal_batch_size(task)

        # 验证批次大小在限制范围内
        assert batch_size >= min_size, f"批次大小{batch_size}小于最小限制{min_size}"
        assert batch_size <= max_size, f"批次大小{batch_size}大于最大限制{max_size}"

    def test_batch_record_estimation(self):
        """测试批次记录数估算"""
        # 创建测试任务
        task = ComplementTask(
            task_id="estimation_test",
            source_id="test_source",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM,
            start_date=datetime.now() - timedelta(days=100),
            end_date=datetime.now(),
            estimated_records=10000  # 100天内预计1万条记录
        )

        # 测试30天批次的记录数估算
        batch_days = 30
        estimated_records = self.processor._estimate_batch_records(task, batch_days)

        # 验证估算逻辑：总记录数/总天数*批次天数
        total_days = 100
        expected_records = int(10000 / total_days * batch_days)

        assert estimated_records == expected_records, \
            f"记录数估算错误：期望{expected_records}，实际{estimated_records}"

    def test_empty_task_handling(self):
        """测试空任务处理"""
        # 创建时间范围为0的任务
        task = ComplementTask(
            task_id="empty_test",
            source_id="test_source",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM,
            start_date=datetime.now(),
            end_date=datetime.now(),
            estimated_records=0
        )

        batches = self.processor.create_complement_batches(task)

        # 验证至少创建一个批次
        assert len(batches) >= 1, "即使时间范围为0，也应该创建至少一个批次"

        # 验证批次时间范围正确
        batch = batches[0]
        assert batch.start_date == task.start_date
        assert batch.end_date == task.end_date

    def test_large_time_range_batch_creation(self):
        """测试大时间范围的批次创建"""
        # 创建20年的历史数据任务
        start_date = datetime(2004, 1, 1)
        end_date = datetime(2024, 1, 1)

        task = ComplementTask(
            task_id="large_range_test",
            source_id="historical_data",
            data_type="stock",
            mode=ComplementMode.FULL_HISTORY,
            priority=ComplementPriority.MEDIUM,
            start_date=start_date,
            end_date=end_date,
            estimated_records=500000
        )

        batches = self.processor.create_complement_batches(task)

        # 验证批次数量合理
        total_days = (end_date - start_date).days
        batch_size = self.processor._calculate_optimal_batch_size(task)
        expected_batches = max(1, (total_days + batch_size - 1) // batch_size)

        assert len(batches) == expected_batches, \
            f"批次数量错误：期望{expected_batches}，实际{len(batches)}"

        # 验证所有批次连续且不重叠
        for i in range(len(batches) - 1):
            current_batch = batches[i]
            next_batch = batches[i + 1]

            # 当前批次的结束时间应该等于下一批次的开始时间
            assert current_batch.end_date == next_batch.start_date, \
                f"批次{i+1}和{i+2}之间不连续"

        # 验证第一个批次从任务开始时间开始
        assert batches[0].start_date == task.start_date

        # 验证最后一个批次到任务结束时间结束
        assert batches[-1].end_date == task.end_date

    def test_batch_processing_concurrency(self):
        """测试批次处理并发性"""
        import threading
        import time

        results = []
        errors = []

        def process_batch_worker(batch_id, batch_data):
            """批次处理工作线程"""
            try:
                time.sleep(0.01)  # 模拟处理时间
                results.append((batch_id, len(batch_data)))
            except Exception as e:
                errors.append((batch_id, str(e)))

        # 创建多个批次
        batches = []
        for i in range(5):
            batch = ComplementBatch(
                batch_id=f"concurrent_batch_{i}",
                task_id="concurrent_test",
                source_id="test_source",
                batch_index=i + 1,
                total_batches=5,
                start_date=datetime.now() + timedelta(days=i * 10),
                end_date=datetime.now() + timedelta(days=(i + 1) * 10),
                estimated_records=100 * (i + 1)
            )
            batches.append(batch)

        # 并发处理批次
        threads = []
        for batch in batches:
            t = threading.Thread(
                target=process_batch_worker,
                args=(batch.batch_id, f"data_for_{batch.batch_id}")
            )
            threads.append(t)

        # 启动所有线程
        for t in threads:
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证所有批次都被处理
        assert len(results) == 5, f"应该处理5个批次，实际处理了{len(results)}个"
        assert len(errors) == 0, f"不应该有处理错误，实际有{len(errors)}个"

        # 验证结果正确性
        for batch_id, data_length in results:
            expected_length = len(f"data_for_{batch_id}")
            assert data_length == expected_length, \
                f"批次{batch_id}处理结果错误：期望{expected_length}，实际{data_length}"

    def test_batch_error_handling(self):
        """测试批次处理错误处理"""
        # 创建一个无效的任务（开始时间晚于结束时间）
        task = ComplementTask(
            task_id="error_test",
            source_id="test_source",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM,
            start_date=datetime.now(),
            end_date=datetime.now() - timedelta(days=1),  # 开始时间晚于结束时间
            estimated_records=1000
        )

        # 尝试创建批次
        batches = self.processor.create_complement_batches(task)

        # 验证错误处理：应该返回空列表或处理错误情况
        # （具体行为取决于实现，这里验证不会崩溃）
        assert isinstance(batches, list), "应该返回列表类型"

    def test_config_validation(self):
        """测试配置验证"""
        config = self.processor.config

        # 验证必要配置项存在
        required_keys = [
            'default_batch_size_days',
            'min_batch_size_days',
            'max_batch_size_days',
            'max_concurrent_batches'
        ]

        for key in required_keys:
            assert key in config, f"配置缺少必要项：{key}"
            assert isinstance(config[key], int), f"配置项{key}应该是整数"
            assert config[key] > 0, f"配置项{key}应该是正数"

    def test_performance_monitoring(self):
        """测试性能监控"""
        # 创建测试批次
        batch = ComplementBatch(
            batch_id="perf_test_batch",
            task_id="perf_test_task",
            source_id="test_source",
            batch_index=1,
            total_batches=1,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            estimated_records=1000
        )

        # 记录批次开始处理
        self.processor._record_batch_start(batch)

        # 模拟处理时间
        import time
        time.sleep(0.1)

        # 记录批次完成
        self.processor._record_batch_completion(batch, 950, success=True)

        # 验证性能统计
        stats = self.processor.get_performance_stats()

        # 验证统计信息
        assert 'total_batches_processed' in stats
        assert 'successful_batches' in stats
        assert 'failed_batches' in stats
        assert 'average_processing_time' in stats

        assert stats['total_batches_processed'] >= 1
        assert stats['successful_batches'] >= 1

    def test_resource_limits(self):
        """测试资源限制"""
        # 创建大量批次来测试并发限制
        batches = []
        for i in range(20):  # 超过默认最大并发数
            batch = ComplementBatch(
                batch_id=f"limit_test_{i}",
                task_id="limit_test_task",
                source_id="test_source",
                batch_index=i + 1,
                total_batches=20,
                start_date=datetime.now() + timedelta(days=i * 5),
                end_date=datetime.now() + timedelta(days=(i + 1) * 5),
                estimated_records=500
            )
            batches.append(batch)

        # 验证批次创建没有问题
        assert len(batches) == 20

        # 验证处理器可以处理这些批次（不测试实际并发限制）
        # 实际的并发控制应该在调度层实现

    def test_cleanup_functionality(self):
        """测试清理功能"""
        # 添加一些模拟的批次统计
        # （这里主要测试接口，不测试实际清理逻辑）
        stats = self.processor.get_performance_stats()
        assert isinstance(stats, dict), "应该返回字典类型的统计信息"

        # 测试重置统计（如果有的话）
        if hasattr(self.processor, 'reset_performance_stats'):
            self.processor.reset_performance_stats()
            new_stats = self.processor.get_performance_stats()

            # 验证统计已重置（某些字段应该为0）
            assert new_stats['total_batches_processed'] == 0


if __name__ == '__main__':
    pytest.main([__file__])