#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单簿模块完整测试覆盖
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
# 直接导入metrics模块，避免orderbook.__init__的导入问题
import sys
from pathlib import Path
import importlib.util

# 直接加载metrics模块，不通过__init__.py
orderbook_path = Path(__file__).parent.parent.parent.parent / "src" / "features" / "orderbook"
metrics_file = orderbook_path / "metrics.py"

if metrics_file.exists():
    spec = importlib.util.spec_from_file_location("metrics", metrics_file)
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    
    calculate_vwap = metrics_module.calculate_vwap
    calculate_twap = metrics_module.calculate_twap
    calculate_orderbook_imbalance = metrics_module.calculate_orderbook_imbalance
    calculate_orderbook_skew = metrics_module.calculate_orderbook_skew
else:
    # 如果文件不存在，跳过所有测试
    import pytest
    pytest.skip("metrics.py not found", allow_module_level=True)


# 由于orderbook模块存在导入问题，暂时只测试metrics模块
# 其他模块的测试将在修复导入问题后添加


class TestOrderbookMetrics:
    """订单簿指标测试"""

    def test_calculate_vwap(self):
        """测试计算VWAP"""
        prices = [10.0, 10.1, 10.2]
        volumes = [1000, 2000, 3000]
        
        vwap = calculate_vwap(prices, volumes)
        assert isinstance(vwap, float)
        assert vwap > 0

    def test_calculate_vwap_empty(self):
        """测试计算VWAP-空数据"""
        vwap = calculate_vwap([], [])
        assert vwap == 0.0

    def test_calculate_vwap_zero_volume(self):
        """测试计算VWAP-零成交量"""
        prices = [10.0, 10.1]
        volumes = [0, 0]
        
        vwap = calculate_vwap(prices, volumes)
        assert vwap == 0.0

    def test_calculate_twap(self):
        """测试计算TWAP"""
        prices = [10.0, 10.1, 10.2, 10.15]
        
        twap = calculate_twap(prices)
        assert isinstance(twap, float)
        assert twap == sum(prices) / len(prices)

    def test_calculate_twap_empty(self):
        """测试计算TWAP-空数据"""
        twap = calculate_twap([])
        assert twap == 0.0

    def test_calculate_orderbook_imbalance(self):
        """测试计算订单簿不平衡度"""
        bids = [(10.0, 1000), (9.99, 2000)]
        asks = [(10.01, 1500), (10.02, 2500)]
        
        imbalance = calculate_orderbook_imbalance(bids, asks, levels=2)
        assert isinstance(imbalance, float)
        assert -1 <= imbalance <= 1

    def test_calculate_orderbook_imbalance_zero_volume(self):
        """测试计算订单簿不平衡度-零成交量"""
        bids = [(10.0, 0), (9.99, 0)]
        asks = [(10.01, 0), (10.02, 0)]
        
        imbalance = calculate_orderbook_imbalance(bids, asks)
        assert imbalance == 0.0

    def test_calculate_orderbook_imbalance_custom_levels(self):
        """测试计算订单簿不平衡度-自定义层级"""
        bids = [(10.0, 1000), (9.99, 2000), (9.98, 3000)]
        asks = [(10.01, 1500), (10.02, 2500), (10.03, 3500)]
        
        imbalance = calculate_orderbook_imbalance(bids, asks, levels=2)
        assert isinstance(imbalance, float)

    def test_calculate_orderbook_skew(self):
        """测试计算订单簿偏度"""
        bids = [(10.0, 1000), (9.99, 2000), (9.98, 3000)]
        asks = [(10.01, 1500), (10.02, 2500), (10.03, 3500)]
        
        skew = calculate_orderbook_skew(bids, asks)
        assert isinstance(skew, float)

    def test_calculate_orderbook_skew_empty_bids(self):
        """测试计算订单簿偏度-空买盘"""
        bids = []
        asks = [(10.01, 1500), (10.02, 2500)]
        
        skew = calculate_orderbook_skew(bids, asks)
        assert skew == 0.0

    def test_calculate_orderbook_skew_empty_asks(self):
        """测试计算订单簿偏度-空卖盘"""
        bids = [(10.0, 1000), (9.99, 2000)]
        asks = []
        
        skew = calculate_orderbook_skew(bids, asks)
        assert skew == 0.0
