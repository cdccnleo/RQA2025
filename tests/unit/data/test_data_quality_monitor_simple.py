"""简化的数据质量监控测试"""

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
import pandas as pd
import numpy as np
from src.data.quality.data_quality_monitor import (
    DataQualityMonitor,
    CompletenessRule
)


class TestDataQualityMonitorSimple:
    """简化的数据质量监控测试类"""

    @pytest.fixture
    def quality_monitor(self):
        """DataQualityMonitor实例"""
        return DataQualityMonitor()

    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'price': np.random.normal(100, 5, 100),
            'volume': np.random.randint(1000, 10000, 100),
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min')
        })

    def test_quality_monitor_initialization(self, quality_monitor):
        """测试质量监控器初始化"""
        assert quality_monitor is not None
        assert hasattr(quality_monitor, 'config')
        assert hasattr(quality_monitor, 'rules')
        assert isinstance(quality_monitor.rules, list)
        assert len(quality_monitor.rules) > 0  # 默认规则已初始化

    def test_quality_check(self, quality_monitor, sample_data):
        """测试质量检查"""
        # 执行质量检查
        report = quality_monitor.check_quality(sample_data)

        assert report is not None
        assert hasattr(report, 'overall_score')
        assert hasattr(report, 'metrics')
        assert isinstance(report.metrics, dict)

        # 验证分数范围
        assert 0.0 <= report.overall_score <= 100.0

    def test_add_rule(self, quality_monitor):
        """测试添加规则"""
        initial_count = len(quality_monitor.rules)
        rule = CompletenessRule()
        quality_monitor.add_rule(rule)

        assert len(quality_monitor.rules) == initial_count + 1
        assert rule in quality_monitor.rules
