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


from datetime import datetime, timedelta
from src.data.sync.multi_market_sync import MultiCurrencyProcessor, CrossTimezoneSynchronizer


def test_currency_processor_unknown_and_identity():
    proc = MultiCurrencyProcessor()
    # 未设置汇率返回 None
    assert proc.convert_currency(100, "EUR", "USD") is None
    # 同币种恒等
    assert proc.convert_currency(123.45, "USD", "USD") == 123.45
    # 历史窗口=0 返回空或仅当时间戳>=now 的记录
    proc.set_exchange_rate("CNY", "USD", 0.14, datetime.now())
    hist = proc.get_rate_history("CNY", "USD", days=0)
    assert isinstance(hist, list)


def test_timezone_synchronizer_convert_timezone():
    syncer = CrossTimezoneSynchronizer()
    # 基本转换：若无 tzinfo，将按 from 时区本地化
    t = datetime(2024, 1, 1, 12, 0, 0)
    converted = syncer.convert_timezone(t, "Asia/Shanghai", "America/New_York")
    assert converted.tzinfo is not None


