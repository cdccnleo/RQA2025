import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime, timedelta

from src.data.sync.multi_market_sync import (
    GlobalMarketDataManager,
    MarketConfig,
    MarketData,
    DataType,
    MultiCurrencyProcessor,
    CrossTimezoneSynchronizer,
    MultiMarketSyncManager,
    SyncType,
    SyncStatus,
    SyncTask,
)


def test_global_market_data_manager_add_data_unregistered_market():
    """测试添加数据到未注册的市场"""
    mgr = GlobalMarketDataManager()
    data = MarketData(
        market_id="UNREGISTERED",
        symbol="TEST",
        price=10.0,
        volume=100,
        timestamp=datetime.now(),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    result = mgr.add_market_data("UNREGISTERED", data)
    assert result is False


def test_global_market_data_manager_data_limit():
    """测试数据量限制"""
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    mgr.register_market(cfg)
    
    # 先添加10000条数据
    for i in range(10000):
        data = MarketData(
            market_id="TEST",
            symbol="TEST",
            price=10.0 + i,
            volume=100,
            timestamp=datetime.now(),
            timezone="Asia/Shanghai",
            currency="CNY",
            data_type=DataType.OHLC,
            source="test"
        )
        mgr.add_market_data("TEST", data)
    
    # 此时应该有10000条
    assert len(mgr.market_data["TEST"]) == 10000
    
    # 再添加1条，应该触发限制，截断到5000（保留最后5000条）
    data = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=10010.0,
        volume=100,
        timestamp=datetime.now(),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    mgr.add_market_data("TEST", data)
    
    # 数据应该被限制在5000条（截断保留最后5000条，不包括刚添加的那条）
    assert len(mgr.market_data["TEST"]) == 5000


def test_global_market_data_manager_get_data_nonexistent_market():
    """测试获取不存在的市场数据"""
    mgr = GlobalMarketDataManager()
    result = mgr.get_market_data("NONEXISTENT")
    assert result == []


def test_global_market_data_manager_get_data_time_filter():
    """测试获取市场数据（时间过滤）"""
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    mgr.register_market(cfg)
    
    now = datetime.now()
    data1 = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=10.0,
        volume=100,
        timestamp=now - timedelta(hours=2),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    data2 = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=11.0,
        volume=200,
        timestamp=now - timedelta(hours=1),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    data3 = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=12.0,
        volume=300,
        timestamp=now,
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    mgr.add_market_data("TEST", data1)
    mgr.add_market_data("TEST", data2)
    mgr.add_market_data("TEST", data3)
    
    # 只获取最近1小时的数据（start_time >= now - 1小时）
    result = mgr.get_market_data("TEST", start_time=now - timedelta(hours=1))
    assert len(result) == 2
    
    # 只获取1小时前的数据（end_time <= now - 1小时）
    # 注意：end_time 使用 <= 比较，所以包含等于该时间的数据
    cutoff_time = now - timedelta(hours=1)
    result2 = mgr.get_market_data("TEST", end_time=cutoff_time)
    # data2 的时间是 now - 1小时，应该被包含（因为 <=）
    assert len(result2) >= 1


def test_global_market_data_manager_get_data_type_filter():
    """测试获取市场数据（类型过滤）"""
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    mgr.register_market(cfg)
    
    now = datetime.now()
    data1 = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=10.0,
        volume=100,
        timestamp=now,
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    data2 = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=11.0,
        volume=200,
        timestamp=now,
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.QUOTE,
        source="test"
    )
    mgr.add_market_data("TEST", data1)
    mgr.add_market_data("TEST", data2)
    
    # 只获取 OHLC 类型
    result = mgr.get_market_data("TEST", data_type=DataType.OHLC)
    assert len(result) == 1
    assert result[0].data_type == DataType.OHLC


def test_global_market_data_manager_get_statistics_nonexistent_market():
    """测试获取不存在的市场统计信息"""
    mgr = GlobalMarketDataManager()
    result = mgr.get_market_statistics("NONEXISTENT")
    assert result == {}


def test_global_market_data_manager_get_statistics_empty_data():
    """测试获取市场统计信息（空数据）"""
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    mgr.register_market(cfg)
    result = mgr.get_market_statistics("TEST")
    assert result['market_id'] == "TEST"
    assert result['data_count'] == 0


def test_global_market_data_manager_get_statistics_single_data():
    """测试获取市场统计信息（单条数据）"""
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    mgr.register_market(cfg)
    
    data = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=10.0,
        volume=100,
        timestamp=datetime.now(),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    mgr.add_market_data("TEST", data)
    result = mgr.get_market_statistics("TEST")
    assert result['data_count'] == 1
    assert result['price_stats']['min'] == 10.0
    assert result['price_stats']['max'] == 10.0
    assert result['price_stats']['avg'] == 10.0


def test_cross_timezone_synchronizer_invalid_timezone():
    """测试跨时区同步器（无效时区）"""
    syncer = CrossTimezoneSynchronizer()
    t = datetime.now()
    
    # 无效时区应该抛出异常
    with pytest.raises(Exception):
        syncer.convert_timezone(t, "Invalid/Timezone", "UTC")


def test_cross_timezone_synchronizer_timezone_with_tzinfo():
    """测试跨时区同步器（已有时区信息）"""
    syncer = CrossTimezoneSynchronizer()
    import pytz
    t = datetime.now(pytz.UTC)
    result = syncer.convert_timezone(t, "UTC", "Asia/Shanghai")
    assert result.tzinfo is not None


def test_cross_timezone_synchronizer_schedule_sync_zero_frequency():
    """测试安排同步（零频率）"""
    syncer = CrossTimezoneSynchronizer()
    schedule_id = syncer.schedule_sync("M1", "UTC", 0)
    assert schedule_id is not None
    schedules = syncer.get_sync_schedules()
    assert any(s["schedule_id"] == schedule_id for s in schedules)


def test_cross_timezone_synchronizer_get_sync_schedules_empty():
    """测试获取同步计划（空计划）"""
    syncer = CrossTimezoneSynchronizer()
    schedules = syncer.get_sync_schedules()
    assert schedules == []


def test_cross_timezone_synchronizer_get_sync_schedules_with_none():
    """测试获取同步计划（包含 None 值）"""
    syncer = CrossTimezoneSynchronizer()
    schedule_id = syncer.schedule_sync("M1", "UTC", 60)
    schedules = syncer.get_sync_schedules()
    # 验证不会因为 None 值而失败
    assert len(schedules) > 0


def test_multi_currency_processor_convert_same_currency():
    """测试多货币处理器（同币种转换）"""
    proc = MultiCurrencyProcessor()
    result = proc.convert_currency(100, "USD", "USD")
    assert result == 100.0


def test_multi_currency_processor_convert_nonexistent_from_currency():
    """测试多货币处理器（不存在的源货币）"""
    proc = MultiCurrencyProcessor()
    result = proc.convert_currency(100, "UNKNOWN", "USD")
    assert result is None


def test_multi_currency_processor_convert_nonexistent_to_currency():
    """测试多货币处理器（不存在的目标货币）"""
    proc = MultiCurrencyProcessor()
    proc.set_exchange_rate("USD", "EUR", 0.85, datetime.now())
    result = proc.convert_currency(100, "USD", "UNKNOWN")
    assert result is None


def test_multi_currency_processor_get_exchange_rate_same_currency():
    """测试获取汇率（同币种）"""
    proc = MultiCurrencyProcessor()
    result = proc.get_exchange_rate("USD", "USD")
    assert result == 1.0


def test_multi_currency_processor_get_exchange_rate_nonexistent():
    """测试获取汇率（不存在）"""
    proc = MultiCurrencyProcessor()
    result = proc.get_exchange_rate("UNKNOWN", "USD")
    assert result is None


def test_multi_currency_processor_get_rate_history_nonexistent():
    """测试获取汇率历史（不存在）"""
    proc = MultiCurrencyProcessor()
    result = proc.get_rate_history("UNKNOWN", "USD")
    assert result == []


def test_multi_currency_processor_get_rate_history_zero_days():
    """测试获取汇率历史（0天）"""
    proc = MultiCurrencyProcessor()
    now = datetime.now()
    proc.set_exchange_rate("CNY", "USD", 0.14, now)
    result = proc.get_rate_history("CNY", "USD", days=0)
    # days=0 时，只返回时间戳 >= now 的记录（通常为空）
    assert isinstance(result, list)


def test_multi_currency_processor_get_rate_history_negative_days():
    """测试获取汇率历史（负数天）"""
    proc = MultiCurrencyProcessor()
    now = datetime.now()
    proc.set_exchange_rate("CNY", "USD", 0.14, now)
    result = proc.get_rate_history("CNY", "USD", days=-1)
    # 负数天应该被处理（cutoff_date 在未来）
    assert isinstance(result, list)


def test_multi_currency_processor_get_rate_history_large_days():
    """测试获取汇率历史（大天数）"""
    proc = MultiCurrencyProcessor()
    now = datetime.now()
    proc.set_exchange_rate("CNY", "USD", 0.14, now)
    result = proc.get_rate_history("CNY", "USD", days=3650)  # 10年
    assert isinstance(result, list)


def test_multi_currency_processor_get_rate_history_filter_by_currency():
    """测试获取汇率历史（按货币过滤）"""
    proc = MultiCurrencyProcessor()
    now = datetime.now()
    proc.set_exchange_rate("CNY", "USD", 0.14, now)
    proc.set_exchange_rate("CNY", "EUR", 0.13, now)
    result = proc.get_rate_history("CNY", "USD", days=30)
    # 应该只返回 USD 的历史
    assert all(r['to_currency'] == 'USD' for r in result)


def test_multi_market_sync_manager_complete_nonexistent_task():
    """测试完成不存在的任务"""
    mgr = MultiMarketSyncManager()
    result = mgr.complete_sync_task("nonexistent_task", records_synced=100)
    assert result is False


def test_multi_market_sync_manager_complete_task_with_errors():
    """测试完成任务（有错误）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    result = mgr.complete_sync_task(task_id, records_synced=100, error_count=5)
    assert result is True
    assert mgr.sync_tasks[task_id].status == SyncStatus.FAILED


def test_multi_market_sync_manager_complete_task_zero_records():
    """测试完成任务（零记录）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    result = mgr.complete_sync_task(task_id, records_synced=0, error_count=0)
    assert result is True
    # 零记录时，success_rate 应该为 0（因为 records_synced > 0 检查）
    metrics = mgr.sync_metrics[task_id]
    assert metrics['success_rate'] == 0


def test_multi_market_sync_manager_get_sync_report_empty():
    """测试获取同步报告（空任务）"""
    mgr = MultiMarketSyncManager()
    report = mgr.get_sync_report()
    assert report['active_tasks_count'] == 0
    assert report['completed_tasks_count'] == 0
    assert report['failed_tasks_count'] == 0
    assert report['total_records_synced'] == 0
    assert report['total_errors'] == 0
    assert report['overall_success_rate'] == 0


def test_multi_market_sync_manager_get_sync_report_all_failed():
    """测试获取同步报告（全部失败）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    mgr.complete_sync_task(task_id, records_synced=100, error_count=100)
    report = mgr.get_sync_report()
    assert report['failed_tasks_count'] == 1
    assert report['total_errors'] == 100
    assert report['overall_success_rate'] == 0


def test_multi_market_sync_manager_get_sync_report_all_success():
    """测试获取同步报告（全部成功）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    mgr.complete_sync_task(task_id, records_synced=100, error_count=0)
    report = mgr.get_sync_report()
    assert report['completed_tasks_count'] == 1
    assert report['total_errors'] == 0
    assert report['overall_success_rate'] == 1.0


def test_multi_market_sync_manager_get_sync_report_active_tasks():
    """测试获取同步报告（活跃任务）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    # 不完成任务，保持为 RUNNING 状态
    report = mgr.get_sync_report()
    assert report['active_tasks_count'] == 1


def test_multi_market_sync_manager_sync_metrics_success_rate():
    """测试同步指标（成功率计算）"""
    mgr = MultiMarketSyncManager()
    mgr.initialize_markets()
    task_id = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    mgr.complete_sync_task(task_id, records_synced=100, error_count=10)
    metrics = mgr.sync_metrics[task_id]
    assert metrics['success_rate'] == pytest.approx(0.9)


def test_market_data_to_dict():
    """测试 MarketData.to_dict 方法"""
    data = MarketData(
        market_id="TEST",
        symbol="TEST",
        price=10.0,
        volume=100,
        timestamp=datetime.now(),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.OHLC,
        source="test"
    )
    result = data.to_dict()
    assert result['market_id'] == "TEST"
    assert result['data_type'] == DataType.OHLC.value
    assert 'timestamp' in result


def test_market_config_to_dict():
    """测试 MarketConfig.to_dict 方法"""
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    result = cfg.to_dict()
    assert result['market_id'] == "TEST"
    assert result['market_name'] == "测试市场"


def test_sync_task_to_dict():
    """测试 SyncTask.to_dict 方法"""
    task = SyncTask(
        task_id="test_task",
        market_id="TEST",
        sync_type=SyncType.REAL_TIME,
        status=SyncStatus.RUNNING,
        start_time=datetime.now(),
        end_time=None,
        records_synced=0,
        error_count=0
    )
    result = task.to_dict()
    assert result['task_id'] == "test_task"
    assert result['sync_type'] == SyncType.REAL_TIME.value
    assert result['status'] == SyncStatus.RUNNING.value
    assert 'end_time' in result


def test_sync_task_to_dict_with_end_time():
    """测试 SyncTask.to_dict 方法（有结束时间）"""
    task = SyncTask(
        task_id="test_task",
        market_id="TEST",
        sync_type=SyncType.REAL_TIME,
        status=SyncStatus.COMPLETED,
        start_time=datetime.now(),
        end_time=datetime.now(),
        records_synced=100,
        error_count=0
    )
    result = task.to_dict()
    assert result['end_time'] is not None


def test_multi_market_sync_manager_initialize_markets():
    """测试初始化市场"""
    mgr = MultiMarketSyncManager()
    result = mgr.initialize_markets()
    assert result['markets_registered'] >= 1
    assert result['timezone_mappings'] >= 1
    assert result['exchange_rates_set'] >= 1


def test_multi_market_sync_manager_start_sync_task_nonexistent_market():
    """测试启动同步任务（不存在的市场）"""
    mgr = MultiMarketSyncManager()
    # 即使市场不存在，也应该能创建任务
    task_id = mgr.start_sync_task("NONEXISTENT", SyncType.REAL_TIME)
    assert task_id is not None
    assert task_id in mgr.sync_tasks

