#!/usr/bin/env python3
"""
增量采集策略单元测试
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.incremental_collection_strategy import (
    IncrementalCollectionStrategy,
    CollectionMode,
    CollectionPriority
)


class TestIncrementalCollectionStrategy:
    """增量采集策略测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'max_incremental_days': 10,
            'complement_period_days': 90,
            'trading_days_only': True,
            'weekends_enabled': False,
            'holidays_enabled': False
        }
        self.strategy = IncrementalCollectionStrategy(self.config)

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_initialization(self):
        """测试初始化"""
        assert self.strategy.max_incremental_days == 10
        assert self.strategy.complement_period_days == 90
        assert self.strategy.trading_days_only is True

    def test_collection_mode_determination(self):
        """测试采集模式判断"""
        # 测试增量模式
        mode = self.strategy._determine_collection_mode('stock', '000001')
        assert mode == CollectionMode.INCREMENTAL

        # 测试补全模式（通过配置测试）
        self.strategy.last_collection_dates = {'000001': datetime.now() - timedelta(days=15)}
        mode = self.strategy._determine_collection_mode('stock', '000001')
        assert mode == CollectionMode.COMPLEMENT

        # 测试全量模式
        self.strategy.last_collection_dates = {}
        mode = self.strategy._determine_collection_mode('stock', '000001')
        assert mode == CollectionMode.FULL

    def test_time_window_calculation_incremental(self):
        """测试增量模式时间窗口计算"""
        base_date = datetime(2023, 1, 10)  # 假设今天是2023-01-10

        with patch('src.core.orchestration.incremental_collection_strategy.datetime') as mock_datetime:
            mock_datetime.now.return_value = base_date
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            window = self.strategy._calculate_incremental_window('000001', 'stock')

            # 增量模式应该采集最近10天的交易日
            assert window.mode == CollectionMode.INCREMENTAL
            assert window.start_date <= base_date
            assert (base_date - window.start_date).days <= 10

    def test_time_window_calculation_complement(self):
        """测试补全模式时间窗口计算"""
        base_date = datetime(2023, 1, 10)

        with patch('src.core.orchestration.incremental_collection_strategy.datetime') as mock_datetime:
            mock_datetime.now.return_value = base_date
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # 设置最后采集日期为20天前
            self.strategy.last_collection_dates['000001'] = base_date - timedelta(days=20)

            window = self.strategy._calculate_complement_window('000001', 'stock')

            assert window.mode == CollectionMode.COMPLEMENT
            # 补全窗口应该从最后采集日期开始，到现在为止
            assert window.start_date >= self.strategy.last_collection_dates['000001']
            assert window.end_date == base_date

    def test_priority_calculation(self):
        """测试优先级计算"""
        # 核心股票应该有最高优先级
        priority = self.strategy._calculate_priority('000001', 'stock')
        assert priority == CollectionPriority.CRITICAL

        # 主要指数应该有高优先级
        priority = self.strategy._calculate_priority('sh000001', 'index')
        assert priority == CollectionPriority.HIGH

        # 普通股票应该有中等优先级
        priority = self.strategy._calculate_priority('normal_stock', 'stock')
        assert priority == CollectionPriority.MEDIUM

        # 宏观数据应该有低优先级
        priority = self.strategy._calculate_priority('macro_gdp', 'macro')
        assert priority == CollectionPriority.LOW

    def test_trading_days_filter(self):
        """测试交易日过滤"""
        # 创建包含周末的日期序列
        start_date = datetime(2023, 1, 1)  # 2023-01-01是星期日
        end_date = datetime(2023, 1, 7)   # 2023-01-07是星期六

        trading_days = self.strategy._filter_trading_days(start_date, end_date)

        # 应该只包含周一到周五
        expected_days = 5  # 1/2, 1/3, 1/4, 1/5, 1/6
        assert len(trading_days) == expected_days

        # 验证都是工作日
        for day in trading_days:
            assert day.weekday() < 5  # 0-4表示周一到周五

    def test_missing_data_detection(self):
        """测试缺失数据检测"""
        # 设置已有的数据日期
        existing_dates = {
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 5),  # 1月4日缺失
            datetime(2023, 1, 6)
        }

        # 模拟已有数据
        self.strategy.collected_dates_cache['000001'] = existing_dates

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 6)

        missing_dates = self.strategy._detect_missing_dates('000001', start_date, end_date)

        # 应该检测到1月4日缺失
        assert len(missing_dates) == 1
        assert datetime(2023, 1, 4) in missing_dates

    def test_collection_window_creation(self):
        """测试采集窗口创建"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 5)
        source_id = '000001'
        data_type = 'stock'

        window = self.strategy._create_collection_window(
            start_date, end_date, source_id, data_type, CollectionMode.INCREMENTAL
        )

        assert window.start_date == start_date
        assert window.end_date == end_date
        assert window.source_id == source_id
        assert window.data_type == data_type
        assert window.mode == CollectionMode.INCREMENTAL
        assert window.priority == CollectionPriority.CRITICAL  # 000001是核心股票

    def test_determine_collection_strategy_with_missing_data(self):
        """测试包含缺失数据的采集策略"""
        source_id = '000001'
        data_type = 'stock'

        # 设置最后采集日期
        self.strategy.last_collection_dates[source_id] = datetime(2023, 1, 1)

        # 设置已有数据（模拟缺失数据）
        existing_dates = {datetime(2023, 1, 1), datetime(2023, 1, 3)}  # 1月2日缺失
        self.strategy.collected_dates_cache[source_id] = existing_dates

        window = self.strategy.determine_collection_strategy(source_id, data_type)

        # 由于有缺失数据，应该选择补全模式
        assert window.mode == CollectionMode.COMPLEMENT

    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = {
            'max_incremental_days': 10,
            'complement_period_days': 90
        }

        strategy = IncrementalCollectionStrategy(valid_config)
        assert strategy.max_incremental_days == 10

        # 无效配置应该使用默认值
        invalid_config = {}
        strategy = IncrementalCollectionStrategy(invalid_config)
        assert strategy.max_incremental_days == 10  # 默认值

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空数据源ID
        with pytest.raises(ValueError):
            self.strategy.determine_collection_strategy('', 'stock')

        # 测试无效数据类型
        with pytest.raises(ValueError):
            self.strategy.determine_collection_strategy('000001', 'invalid_type')

        # 测试未来日期
        future_date = datetime.now() + timedelta(days=30)
        self.strategy.last_collection_dates['000001'] = future_date

        window = self.strategy.determine_collection_strategy('000001', 'stock')
        # 应该正常处理，不抛出异常
        assert window is not None

    def test_cache_management(self):
        """测试缓存管理"""
        source_id = '000001'
        test_dates = {datetime(2023, 1, 1), datetime(2023, 1, 2)}

        # 设置缓存
        self.strategy.collected_dates_cache[source_id] = test_dates
        assert self.strategy._get_cached_dates(source_id) == test_dates

        # 更新缓存
        new_dates = {datetime(2023, 1, 3)}
        self.strategy._update_date_cache(source_id, new_dates)
        assert datetime(2023, 1, 3) in self.strategy.collected_dates_cache[source_id]

        # 清理缓存
        self.strategy._clear_date_cache(source_id)
        assert source_id not in self.strategy.collected_dates_cache

    def test_concurrent_access(self):
        """测试并发访问"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(10):
                    window = self.strategy.determine_collection_strategy('000001', 'stock')
                    results.append((worker_id, i, window.mode))
                    time.sleep(0.01)  # 模拟处理时间
            except Exception as e:
                errors.append((worker_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        # 启动线程
        for t in threads:
            t.start()

        # 等待完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == 30  # 3线程 * 10次调用
        assert len(errors) == 0    # 不应该有错误

        # 验证所有结果都有效
        for worker_id, call_id, mode in results:
            assert mode in [CollectionMode.INCREMENTAL, CollectionMode.COMPLEMENT, CollectionMode.FULL]


if __name__ == '__main__':
    pytest.main([__file__])