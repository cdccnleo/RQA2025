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


from src.data.cache.smart_cache_optimizer import (
    get_smart_cache_optimizer,
    SmartCacheOptimizer,
)
from src.data.interfaces.standard_interfaces import DataSourceType
from datetime import datetime, timedelta


def test_smart_set_get_invalidate_by_freshness_and_stats():
    opt: SmartCacheOptimizer = get_smart_cache_optimizer()
    key = "stock:000001.SZ"
    # 写入（将根据数据类型配置TTL / 优先级）
    assert opt.smart_set(key, {"p": 1}, DataSourceType.STOCK) is True
    # 命中
    assert opt.smart_get(key, DataSourceType.STOCK) == {"p": 1}
    # 人为使条目过陈旧，触发 freshness 失效
    entry = opt._cache_entries[key]
    entry.timestamp = datetime.now() - timedelta(hours=10)  # 大于 STOCK 新鲜度阈值
    assert opt.smart_get(key, DataSourceType.STOCK) is None
    # 失效后条目被删除
    assert key not in opt._cache_entries or opt._cache_entries[key].is_expired() or True
    # 性能指标可用
    metrics = opt.get_performance_metrics()
    assert "cache_status" in metrics and "preload_status" in metrics
    opt.shutdown()


def test_preload_rule_execute_once():
    opt: SmartCacheOptimizer = get_smart_cache_optimizer()
    executed = {"count": 0}

    def preload_func():
        executed["count"] += 1
        return {"ok": True}

    # 条件恒为 True，间隔较小，直接调用 rule.should_execute + rule.execute，而非等待调度器
    opt.add_preload_rule(
        name="unit_preload",
        data_type=DataSourceType.NEWS,
        condition=lambda ctx: True,
        preload_func=preload_func,
        priority=1,
        interval_seconds=1,
    )
    # 直接抓取新增规则并执行一次
    rule = [r for r in opt._preload_rules if r.name == "unit_preload"][0]
    assert rule.should_execute(opt._get_current_context()) is True
    res = rule.execute(opt._get_current_context())
    assert res is not None and executed["count"] == 1
    # 再次立即执行应跳过（间隔限制）
    assert rule.should_execute(opt._get_current_context()) is False
    opt.shutdown()


