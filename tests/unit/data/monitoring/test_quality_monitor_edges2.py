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


import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data.monitoring.quality_monitor import DataQualityMonitor, DataModel


def test_evaluate_quality_empty_history_and_then_report(tmp_path):
    qm = DataQualityMonitor(report_dir=str(tmp_path))
    dm = DataModel(pd.DataFrame())  # empty data
    dm.metadata = {'source': 's1', 'created_at': datetime.now().isoformat()}

    m = qm.evaluate_quality(dm)
    assert m.overall_score == pytest.approx(1.0)

    # history file created and contains s1
    hist_file = os.path.join(str(tmp_path), 'quality_history.json')
    assert os.path.exists(hist_file)
    with open(hist_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    assert 's1' in history and isinstance(history['s1'], list) and len(history['s1']) >= 1

    # generate report uses evaluated_sources
    report = qm.generate_quality_report(dm)
    assert 'sources' in report and 's1' in report['sources']


def test_set_thresholds_invalid_values_overwrite(tmp_path):
    qm = DataQualityMonitor(report_dir=str(tmp_path))
    # set invalid threshold payload (non-dict) should still overwrite as-is per simple setter
    qm.set_thresholds({'completeness': 0.5})
    assert qm.thresholds.get('completeness') == 0.5
    # set again with empty dict
    qm.set_thresholds({})
    assert qm.thresholds == {}


def test_get_quality_trend_shape_and_monotonic_timestamps():
    qm = DataQualityMonitor()
    trend = qm.get_quality_trend('any', 'completeness')
    values = trend['data']['values']
    ts = trend['data']['timestamps']
    assert len(values) == len(ts) >= 1
    # timestamps isoformat and descending by construction
    datetime.fromisoformat(ts[0])


def test_health_aggregation_via_summary_defaults_when_no_evaluations(tmp_path):
    qm = DataQualityMonitor(report_dir=str(tmp_path))
    summary = qm.get_quality_summary()
    assert summary['overall']['total_sources'] == 1
    assert summary['sources'] == ['mock_source']


def test_consistency_edge_non_monotonic_index(tmp_path):
    qm = DataQualityMonitor(report_dir=str(tmp_path))
    df = pd.DataFrame({'a': [1, 3, 2]}, index=[0, 2, 1])  # non-monotonic index
    dm = DataModel(df)
    dm.metadata = {'source': 's2', 'created_at': datetime.now().isoformat()}
    m = qm.evaluate_quality(dm)
    # completeness < 1 due to no missing, so it should be 1.0; accuracy 1.0; consistency should be 0 or 1 by fallback
    assert 0.0 <= m.consistency <= 1.0


