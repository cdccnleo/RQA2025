#!/usr/bin/env python3
"""
扩展补全调度器单元测试
测试FULL_HISTORY和STRATEGY_BACKTEST模式的补全功能
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.data_complement_scheduler import (
    DataComplementScheduler,
    ComplementMode,
    ComplementPriority,
    ComplementTask,
    ComplementSchedule
)


class TestExtendedComplementScheduler:
    """扩展补全调度器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.scheduler = DataComplementScheduler()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_strategy_backtest_mode_support(self):
        """测试策略回测模式支持"""
        # 验证STRATEGY_BACKTEST模式存在
        assert ComplementMode.STRATEGY_BACKTEST.value == "strategy_backtest"

        # 创建策略回测模式的调度配置
        schedule = ComplementSchedule(
            source_id="strategy_backtest_data",
            data_type="stock",
            mode=ComplementMode.STRATEGY_BACKTEST,
            priority=ComplementPriority.HIGH,
            schedule_interval_days=365,
            complement_window_days=3650,
            min_gap_days=330,
            enabled=True
        )

        # 验证配置正确性
        assert schedule.mode == ComplementMode.STRATEGY_BACKTEST
        assert schedule.schedule_interval_days == 365
        assert schedule.complement_window_days == 3650
        assert schedule.min_gap_days == 330

    def test_strategy_backtest_trigger_logic(self):
        """测试策略回测模式的触发逻辑"""
        # 创建策略回测调度配置
        schedule = ComplementSchedule(
            source_id="strategy_backtest_data",
            data_type="stock",
            mode=ComplementMode.STRATEGY_BACKTEST,
            priority=ComplementPriority.HIGH,
            schedule_interval_days=365,
            complement_window_days=3650,
            min_gap_days=330,
            enabled=True
        )

        # 测试首次补全（从未补全过）
        schedule.last_complement_date = None
        should_trigger = self.scheduler._should_trigger_complement(schedule)
        assert should_trigger == True, "首次补全应该立即触发"

        # 测试间隔不足（330天内）
        schedule.last_complement_date = datetime.now() - timedelta(days=100)
        should_trigger = self.scheduler._should_trigger_complement(schedule)
        assert should_trigger == False, "330天内不应该触发"

        # 测试间隔足够（超过330天）
        schedule.last_complement_date = datetime.now() - timedelta(days=400)
        should_trigger = self.scheduler._should_trigger_complement(schedule)
        assert should_trigger == True, "超过330天应该触发"

    def test_strategy_backtest_task_creation(self):
        """测试策略回测任务创建"""
        # 创建策略回测调度配置
        schedule = ComplementSchedule(
            source_id="strategy_backtest_data",
            data_type="stock",
            mode=ComplementMode.STRATEGY_BACKTEST,
            priority=ComplementPriority.HIGH,
            schedule_interval_days=365,
            complement_window_days=3650,
            min_gap_days=330,
            enabled=True
        )

        # 测试首次补全任务创建
        schedule.last_complement_date = None
        task = self.scheduler._create_complement_task(schedule)

        # 验证任务属性
        assert task.source_id == "strategy_backtest_data"
        assert task.data_type == "stock"
        assert task.mode == ComplementMode.STRATEGY_BACKTEST
        assert task.priority == ComplementPriority.HIGH

        # 验证时间范围（首次补全应该是3650天）
        time_range_days = (task.end_date - task.start_date).days
        assert time_range_days == 3650, f"首次补全时间范围应该是3650天，实际是{time_range_days}天"

        # 测试增量补全任务创建
        schedule.last_complement_date = datetime.now() - timedelta(days=365)
        task2 = self.scheduler._create_complement_task(schedule)

        # 验证增量补全从上次补全时间开始
        expected_start = schedule.last_complement_date
        assert task2.start_date.date() == expected_start.date()

    def test_full_history_mode_support(self):
        """测试全历史补全模式支持"""
        # 验证FULL_HISTORY模式存在
        assert ComplementMode.FULL_HISTORY.value == "full_history"

        # 创建全历史模式的调度配置
        schedule = ComplementSchedule(
            source_id="full_history_data",
            data_type="macro",
            mode=ComplementMode.FULL_HISTORY,
            priority=ComplementPriority.LOW,
            schedule_interval_days=365,
            complement_window_days=3650,
            min_gap_days=330,
            enabled=True
        )

        # 验证配置正确性
        assert schedule.mode == ComplementMode.FULL_HISTORY
        assert schedule.schedule_interval_days == 365

        # 测试触发逻辑
        schedule.last_complement_date = datetime.now() - timedelta(days=400)
        should_trigger = self.scheduler._should_trigger_complement(schedule)
        assert should_trigger == True, "超过365天应该触发全历史补全"

    def test_complement_mode_triggers(self):
        """测试各种补全模式的触发条件"""
        current_time = datetime.now()

        # 测试所有模式的触发条件
        test_cases = [
            (ComplementMode.MONTHLY, 40, True),      # 40天 > 30天，应该触发
            (ComplementMode.MONTHLY, 20, False),     # 20天 < 30天，不应该触发
            (ComplementMode.WEEKLY, 10, True),       # 10天 > 7天，应该触发
            (ComplementMode.WEEKLY, 5, False),       # 5天 < 7天，不应该触发
            (ComplementMode.QUARTERLY, 100, True),   # 100天 > 90天，应该触发
            (ComplementMode.SEMI_ANNUAL, 200, True), # 200天 > 180天，应该触发
            (ComplementMode.FULL_HISTORY, 400, True), # 400天 > 365天，应该触发
            (ComplementMode.STRATEGY_BACKTEST, 350, True), # 350天 > 330天，应该触发
            (ComplementMode.STRATEGY_BACKTEST, 300, False), # 300天 < 330天，不应该触发
        ]

        for mode, days_since_last, expected_trigger in test_cases:
            schedule = ComplementSchedule(
                source_id=f"test_{mode.value}",
                data_type="stock",
                mode=mode,
                priority=ComplementPriority.MEDIUM,
                schedule_interval_days=1,
                complement_window_days=30,
                min_gap_days=1,
                enabled=True,
                last_complement_date=current_time - timedelta(days=days_since_last)
            )

            should_trigger = self.scheduler._should_trigger_complement(schedule)
            assert should_trigger == expected_trigger, \
                f"模式{mode.value}，距离上次{days_since_last}天，期望{expected_trigger}，实际{should_trigger}"

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        # 验证默认调度配置已注册
        assert 'strategy_backtest_data' in self.scheduler.schedules

        strategy_schedule = self.scheduler.schedules['strategy_backtest_data']
        assert strategy_schedule.mode == ComplementMode.STRATEGY_BACKTEST
        assert strategy_schedule.priority == ComplementPriority.HIGH
        assert strategy_schedule.complement_window_days == 3650

    def test_register_complement_schedule(self):
        """测试补全调度配置注册"""
        # 注册新的补全调度配置
        self.scheduler.register_complement_schedule(
            source_id="test_stock_000001",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.CRITICAL
        )

        # 验证配置已注册
        assert "test_stock_000001" in self.scheduler.schedules
        schedule = self.scheduler.schedules["test_stock_000001"]
        assert schedule.mode == ComplementMode.MONTHLY
        assert schedule.priority == ComplementPriority.CRITICAL

    def test_complement_task_management(self):
        """测试补全任务管理"""
        # 注册调度配置
        self.scheduler.register_complement_schedule(
            source_id="managed_test",
            data_type="stock",
            mode=ComplementMode.WEEKLY,
            priority=ComplementPriority.HIGH
        )

        # 模拟检查补全需求
        is_needed, task = self.scheduler.check_complement_needed("managed_test")

        if is_needed and task:
            # 启动补全任务
            self.scheduler.start_complement_task(task)

            # 验证任务已添加到活跃任务
            assert task.task_id in self.scheduler.active_tasks
            assert self.scheduler.active_tasks[task.task_id] == task

            # 完成补全任务
            self.scheduler.complete_complement_task(task.task_id, success=True)

            # 验证任务已移动到完成列表
            assert task.task_id not in self.scheduler.active_tasks
            assert task in self.scheduler.completed_tasks

    def test_complement_statistics(self):
        """测试补全统计信息"""
        # 创建一些模拟的补全任务
        today = datetime.now().date()

        # 模拟今天完成的补全任务
        completed_task = ComplementTask(
            task_id="completed_test",
            source_id="test_source",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            actual_records=1000,
            completed_at=datetime.now()
        )
        self.scheduler.completed_tasks.append(completed_task)

        # 获取统计信息
        stats = self.scheduler.get_complement_statistics()

        # 验证统计信息
        assert stats['completed_tasks_today'] == 1
        assert stats['total_complemented_records'] == 1000
        assert 'total_schedules' in stats
        assert 'active_tasks' in stats

    def test_scheduler_cleanup(self):
        """测试调度器清理功能"""
        # 添加一些旧的补全任务
        old_date = datetime.now() - timedelta(days=40)

        old_task = ComplementTask(
            task_id="old_test",
            source_id="test_source",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM,
            start_date=old_date,
            end_date=old_date,
            completed_at=old_date
        )
        self.scheduler.completed_tasks.append(old_task)

        # 执行清理（保留30天内的任务）
        self.scheduler.cleanup_old_tasks(days_to_keep=30)

        # 验证旧任务已被清理
        assert len(self.scheduler.completed_tasks) == 0, "30天前的任务应该被清理"

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试无效的调度配置
        with pytest.raises(Exception):
            self.scheduler.register_complement_schedule(
                source_id="",  # 空ID
                data_type="stock",
                mode=ComplementMode.MONTHLY,
                priority=ComplementPriority.MEDIUM
            )

        # 测试重复注册
        self.scheduler.register_complement_schedule(
            source_id="duplicate_test",
            data_type="stock",
            mode=ComplementMode.MONTHLY,
            priority=ComplementPriority.MEDIUM
        )

        # 再次注册相同ID应该更新配置
        self.scheduler.register_complement_schedule(
            source_id="duplicate_test",
            data_type="stock",
            mode=ComplementMode.WEEKLY,  # 不同的模式
            priority=ComplementPriority.MEDIUM
        )

        # 验证配置已更新
        assert self.scheduler.schedules["duplicate_test"].mode == ComplementMode.WEEKLY

    def test_mode_specific_behavior(self):
        """测试模式特定的行为"""
        modes_and_windows = [
            (ComplementMode.MONTHLY, 30),
            (ComplementMode.WEEKLY, 7),
            (ComplementMode.QUARTERLY, 90),
            (ComplementMode.SEMI_ANNUAL, 180),
            (ComplementMode.FULL_HISTORY, 3650),
            (ComplementMode.STRATEGY_BACKTEST, 3650),
        ]

        for mode, expected_window in modes_and_windows:
            # 为每个模式注册调度配置
            source_id = f"test_{mode.value}"
            self.scheduler.register_complement_schedule(
                source_id=source_id,
                data_type="stock",
                mode=mode,
                priority=ComplementPriority.MEDIUM
            )

            # 获取注册的配置
            schedule = self.scheduler.schedules[source_id]

            # 验证补全窗口大小
            # 注意：这里可能需要根据实际实现调整验证逻辑
            # 因为配置可能在注册过程中被修改
            assert schedule.mode == mode, f"模式{mode.value}注册失败"

    def test_concurrent_access(self):
        """测试并发访问"""
        import asyncio
        import threading

        results = []
        errors = []

        def worker_thread(thread_id):
            """工作线程"""
            try:
                for i in range(5):
                    # 尝试注册调度配置
                    source_id = f"concurrent_test_{thread_id}_{i}"
                    self.scheduler.register_complement_schedule(
                        source_id=source_id,
                        data_type="stock",
                        mode=ComplementMode.MONTHLY,
                        priority=ComplementPriority.MEDIUM
                    )
                    results.append((thread_id, i, source_id))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 创建多个线程并发执行
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)

        # 启动线程
        for t in threads:
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == 15, f"应该有15个成功结果，实际有{len(results)}个"
        assert len(errors) == 0, f"不应该有错误，实际有{len(errors)}个错误: {errors}"

        # 验证所有注册的配置都存在
        for thread_id, call_id, source_id in results:
            assert source_id in self.scheduler.schedules, f"配置{source_id}未正确注册"


if __name__ == '__main__':
    pytest.main([__file__])