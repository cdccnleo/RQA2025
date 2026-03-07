#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层优化组件测试

测试目标：提升utils/components/optimized_components.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.optimized_components模块
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock


class TestIOptimizedComponent:
    """测试优化组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.components.optimized_components import IOptimizedComponent
        
        with pytest.raises(TypeError):
            IOptimizedComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.components.optimized_components import IOptimizedComponent
        
        assert hasattr(IOptimizedComponent, 'get_info')
        assert hasattr(IOptimizedComponent, 'process')
        assert hasattr(IOptimizedComponent, 'get_status')
        assert hasattr(IOptimizedComponent, 'get_component_id')


class TestOptimizedComponent:
    """测试优化组件实现"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=1)
        assert component.component_id == 1
        assert component.component_type == "Optimized"
        assert component.component_name == "Optimized_Component_1"
        assert isinstance(component.creation_time, datetime)
    
    def test_init_with_custom_type(self):
        """测试使用自定义类型初始化"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=2, component_type="Custom")
        assert component.component_type == "Custom"
        assert component.component_name == "Custom_Component_2"
    
    def test_get_component_id(self):
        """测试获取组件ID"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=3)
        assert component.get_component_id() == 3
    
    def test_get_info(self):
        """测试获取组件信息"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=4)
        info = component.get_info()
        
        assert info["component_id"] == 4
        assert info["component_name"] == "Optimized_Component_4"
        assert info["component_type"] == "Optimized"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
    
    def test_process(self):
        """测试处理数据"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=5)
        data = {"key": "value"}
        
        result = component.process(data)
        
        assert result["component_id"] == 5
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
    
    def test_get_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.components.optimized_components import OptimizedComponent
        
        component = OptimizedComponent(component_id=6)
        status = component.get_status()
        
        assert status["component_id"] == 6
        assert status["component_name"] == "Optimized_Component_6"
        assert status["status"] == "active"
        assert "creation_time" in status


class TestMarketDataDeduplicator:
    """测试行情数据去重处理器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        assert deduplicator.window_size == 3
        assert len(deduplicator.last_hashes) == 0
    
    def test_init_with_window_size(self):
        """测试使用窗口大小初始化"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator(window_size=5)
        assert deduplicator.window_size == 5
    
    def test_generate_hash(self):
        """测试生成哈希"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data = {
            "symbol": "600519",
            "price": 100.0,
            "volume": 1000,
            "bid": [99.9, 99.8],
            "ask": [100.1, 100.2]
        }
        
        hash1 = deduplicator._generate_hash(tick_data)
        hash2 = deduplicator._generate_hash(tick_data)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0
    
    def test_is_duplicate_false(self):
        """测试判断非重复数据"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data = {
            "symbol": "600519",
            "price": 100.0,
            "volume": 1000,
            "bid": [99.9],
            "ask": [100.1]
        }
        
        result = deduplicator.is_duplicate(tick_data)
        assert result is False
    
    def test_is_duplicate_true(self):
        """测试判断重复数据"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        import time
        
        deduplicator = MarketDataDeduplicator(window_size=10)
        tick_data = {
            "symbol": "600519",
            "price": 100.0,
            "volume": 1000,
            "bid": [99.9],
            "ask": [100.1]
        }
        
        # 第一次处理
        result1 = deduplicator.is_duplicate(tick_data)
        assert result1 is False
        
        # 立即再次处理（应该检测为重复）
        result2 = deduplicator.is_duplicate(tick_data)
        assert result2 is True
    
    def test_deduplicator_cleanup(self):
        """测试去重器清理过期数据"""
        from src.infrastructure.utils.components.optimized_components import MarketDataDeduplicator
        import time
        
        deduplicator = MarketDataDeduplicator(window_size=1)
        tick_data = {
            "symbol": "600519",
            "price": 100.0,
            "volume": 1000,
            "bid": [99.9],
            "ask": [100.1]
        }
        
        # 第一次处理
        result1 = deduplicator.is_duplicate(tick_data)
        assert result1 is False
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后应该不再认为是重复
        result2 = deduplicator.is_duplicate(tick_data)
        assert result2 is False

