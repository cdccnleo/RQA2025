#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.features.orderbook.analyzer import OrderbookAnalyzer
from src.features.feature_manager import FeatureManager

class TestOrderbookAnalyzer:
    """订单簿分析器测试"""

    def setUp(self, tmp_model_path, mock_model_manager):
        """设置测试环境"""
        # Mock SentimentAnalyzer的模型加载，避免HuggingFace下载
        with patch('src.features.sentiment.sentiment_analyzer.SentimentAnalyzer._load_pretrained_model', 
                  return_value=(MagicMock(), MagicMock())):
            self.manager = FeatureManager(
                model_path=str(tmp_model_path),
                stock_code="000001",
                model_manager=mock_model_manager
            )
            self.analyzer = OrderbookAnalyzer(self.manager)

    def test_calculate_metrics(self, tmp_model_path, mock_model_manager):
        """测试指标计算"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        # 模拟订单簿数据
        symbol = "000001"
        bids = [(100.0, 1000.0), (99.0, 2000.0), (98.0, 3000.0)]
        asks = [(101.0, 1000.0), (102.0, 2000.0), (103.0, 3000.0)]
        
        # 更新订单簿
        self.analyzer.update_orderbook(symbol, bids, asks)
        
        # 计算指标
        result = self.analyzer.calculate_metrics(symbol)
        assert 'spread' in result
        assert 'imbalance' in result
        assert 'depth' in result

    def test_register_features(self, tmp_model_path, mock_model_manager):
        """测试特征注册"""
        self.setUp(tmp_model_path, mock_model_manager)

        # 测试特征注册
        self.analyzer.register_features()
        # 验证特征已注册到FeatureEngineer
        cache = self.manager.feature_engineer.cache_metadata
        assert 'orderbook_imbalance' in cache
        assert 'orderbook_spread' in cache

    def test_update_orderbook(self, tmp_model_path, mock_model_manager):
        """测试订单簿更新"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        # 模拟新订单簿数据
        symbol = "000001"
        new_bids = [(100.5, 1500.0), (100.0, 2500.0), (99.5, 3500.0)]
        new_asks = [(101.5, 1500.0), (102.5, 2500.0), (103.5, 3500.0)]
        
        # 更新订单簿
        self.analyzer.update_orderbook(symbol, new_bids, new_asks)
        
        # 验证更新
        assert symbol in self.analyzer.orderbook_cache
        assert len(self.analyzer.orderbook_cache[symbol]['bids']) == 3
        assert len(self.analyzer.orderbook_cache[symbol]['asks']) == 3

    def test_calculate_metrics_empty_orderbook(self, tmp_model_path, mock_model_manager):
        """测试空订单簿指标计算"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        # 测试不存在的订单簿
        result = self.analyzer.calculate_metrics("nonexistent")
        assert result == {}

    def test_calculate_imbalance(self, tmp_model_path, mock_model_manager):
        """测试不平衡度计算"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        symbol = "000001"
        bids = [(100.0, 1000.0), (99.0, 2000.0), (98.0, 3000.0)]
        asks = [(101.0, 500.0), (102.0, 1000.0), (103.0, 1500.0)]
        
        self.analyzer.update_orderbook(symbol, bids, asks)
        result = self.analyzer.calculate_metrics(symbol)
        
        assert 'imbalance' in result
        assert -1 <= result['imbalance'] <= 1

    def test_calculate_depth(self, tmp_model_path, mock_model_manager):
        """测试深度计算"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        symbol = "000001"
        bids = [(100.0, 1000.0), (99.0, 2000.0), (98.0, 3000.0)]
        asks = [(101.0, 1000.0), (102.0, 2000.0), (103.0, 3000.0)]
        
        self.analyzer.update_orderbook(symbol, bids, asks)
        result = self.analyzer.calculate_metrics(symbol)
        
        assert 'depth' in result
        assert isinstance(result['depth'], dict)
        assert 'depth_1pct' in result['depth']
        assert 'depth_2pct' in result['depth']
        assert 'depth_5pct' in result['depth']

    def test_orderbook_sorting(self, tmp_model_path, mock_model_manager):
        """测试订单簿排序"""
        self.setUp(tmp_model_path, mock_model_manager)
        
        symbol = "000001"
        # 无序的订单簿数据
        bids = [(99.0, 1000.0), (100.0, 2000.0), (98.0, 3000.0)]
        asks = [(102.0, 1000.0), (101.0, 2000.0), (103.0, 3000.0)]
        
        self.analyzer.update_orderbook(symbol, bids, asks)
        
        # 验证排序
        cached_bids = self.analyzer.orderbook_cache[symbol]['bids']
        cached_asks = self.analyzer.orderbook_cache[symbol]['asks']
        
        # 买盘应该按价格降序
        assert cached_bids[0][0] > cached_bids[1][0] > cached_bids[2][0]
        # 卖盘应该按价格升序
        assert cached_asks[0][0] < cached_asks[1][0] < cached_asks[2][0]
