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


import os
import json
from datetime import datetime, timedelta
import pandas as pd
from src.data.monitoring.quality_monitor import DataQualityMonitor, DataModel


def test_evaluate_quality_boundaries_and_report(tmp_path):
    # 构造含缺失的 DataFrame
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    dm = DataModel(df)
    # 设置 metadata.created_at 为 2 天前，timeliness 应递减但大于 0
    dm.metadata = {"source": "s1", "created_at": (datetime.now() - timedelta(days=2)).isoformat()}
    qm = DataQualityMonitor(report_dir=str(tmp_path))
    metrics = qm.evaluate_quality(dm)
    # 指标范围断言
    assert 0.0 <= metrics.completeness <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.consistency <= 1.0
    assert 0.0 <= metrics.timeliness <= 1.0
    assert 0.0 <= metrics.validity <= 1.0
    # 副作用：history 文件存在且包含 s1
    hist = tmp_path / "quality_history.json"
    assert hist.exists()
    data = json.loads(hist.read_text(encoding="utf-8"))
    assert "s1" in data and isinstance(data["s1"], list) and len(data["s1"]) >= 1
    # 生成报告包含 sources
    report = qm.generate_quality_report(dm)
    assert "sources" in report and "s1" in report["sources"]


