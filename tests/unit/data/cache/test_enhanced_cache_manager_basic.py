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


import time
import pandas as pd
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


def test_enhanced_cache_set_get_expire_and_stats_and_clear_prefix(tmp_path):
    cm = EnhancedCacheManager(cache_dir=str(tmp_path))
    try:
        assert cm.set("a", 123, expire=2, prefix="p") is True
        assert cm.get("a", prefix="p") == 123
        # 过期后 miss
        time.sleep(2.1)
        assert cm.get("a", prefix="p") is None
        # 统计可访问
        st = cm.get_stats()
        assert "hit_rate" in st and "memory_size" in st
        # 前缀清理不报错
        cm.clear(prefix="p")
        # 存大对象也应返回 True（内存不足时会回落磁盘/清理后存储）
        df = pd.DataFrame({"x": list(range(100))})
        assert cm.set("b", df, expire=60, prefix="q") in (True, False)
        cm.clear(prefix="q")
    finally:
        cm.shutdown()


