"""
策略执行服务单元测试
测试策略执行服务的核心功能，包括状态查询、指标获取和策略管理
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, Any

# 导入被测试的模块
from src.gateway.web.strategy_execution_service import (
    get_strategy_execution_status,
    get_execution_metrics,
    get_realtime_metrics,
    start_strategy,
    pause_strategy,
    get_realtime_engine
)


class TestStrategyExecutionService(unittest.TestCase):
    """
    策略执行服务单元测试类
    """
    
    def setUp(self):
        """
        设置测试环境
        """
        # 重置全局变量
        from src.gateway.web.strategy_execution_service import (
            _realtime_engine, _event_bus, _adapter_factory,
            _strategy_adapter, _trading_adapter
        )
        _realtime_engine = None
        _event_bus = None
        _adapter_factory = None
        _strategy_adapter = None
        _trading_adapter = None
    
    async def test_get_strategy_execution_status_with_cache(self):
        """
        测试从缓存获取策略执行状态
        """
        # 模拟缓存返回数据
        mock_cached_data = {
            "strategies": [],
            "running_count": 0,
            "paused_count": 0,
            "stopped_count": 0,
            "total_count": 0
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_status_cache') as mock_get_cache:
            mock_get_cache.return_value = mock_cached_data
            
            result = await get_strategy_execution_status()
            self.assertEqual(result, mock_cached_data)
    
    async def test_get_strategy_execution_status_with_persistence(self):
        """
        测试从持久化存储获取策略执行状态
        """
        # 模拟持久化存储返回数据
        mock_persisted_states = [
            {
                "strategy_id": "test_strategy_1",
                "name": "Test Strategy 1",
                "type": "trend_following",
                "status": "running",
                "latency": 10,
                "throughput": 100,
                "signals_count": 50,
                "positions_count": 5
            }
        ]
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_status_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            with patch('src.gateway.web.strategy_execution_service.list_execution_states') as mock_list_states:
                mock_list_states.return_value = mock_persisted_states
                
                with patch('src.gateway.web.strategy_execution_service.set_execution_status_cache') as mock_set_cache:
                    result = await get_strategy_execution_status()
                    
                    # 验证结果
                    self.assertEqual(len(result["strategies"]), 1)
                    self.assertEqual(result["running_count"], 1)
                    self.assertEqual(result["total_count"], 1)
                    mock_set_cache.assert_called_once()
    
    async def test_get_strategy_execution_status_with_engine(self):
        """
        测试从实时引擎获取策略执行状态
        """
        # 模拟实时引擎
        mock_strategy = Mock()
        mock_strategy.name = "Test Strategy"
        mock_strategy.strategy_type = "mean_reversion"
        mock_strategy.is_active = True
        mock_strategy.get_performance_metrics.return_value = {
            "latency": 5,
            "throughput": 200
        }
        mock_strategy.signals = [1, 2, 3]
        mock_strategy.position = 3
        
        mock_engine = Mock()
        mock_engine.strategies = {"test_strategy": mock_strategy}
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_status_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            with patch('src.gateway.web.strategy_execution_service.list_execution_states') as mock_list_states:
                mock_list_states.return_value = []
                
                with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
                    mock_get_engine.return_value = mock_engine
                    
                    with patch('src.gateway.web.strategy_execution_service.set_execution_status_cache') as mock_set_cache:
                        with patch('src.gateway.web.strategy_execution_service.save_execution_state') as mock_save_state:
                            result = await get_strategy_execution_status()
                            
                            # 验证结果
                            self.assertEqual(len(result["strategies"]), 1)
                            self.assertEqual(result["running_count"], 1)
                            mock_save_state.assert_called_once()
                            mock_set_cache.assert_called_once()
    
    async def test_get_execution_metrics_with_cache(self):
        """
        测试从缓存获取执行指标
        """
        # 模拟缓存返回数据
        mock_cached_data = {
            "avg_latency": 5,
            "today_signals": 100,
            "total_trades": 50,
            "latency_history": [],
            "throughput_history": []
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_metrics_cache') as mock_get_cache:
            mock_get_cache.return_value = mock_cached_data
            
            result = await get_execution_metrics()
            self.assertEqual(result, mock_cached_data)
    
    async def test_get_execution_metrics_with_engine(self):
        """
        测试从实时引擎获取执行指标
        """
        # 模拟实时引擎
        mock_engine = Mock()
        mock_engine.get_performance_metrics.return_value = {
            "stream_metrics": {
                "processing_latency": 8
            },
            "strategy_metrics": {
                "total_signals": 150,
                "total_trades": 75
            }
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_execution_metrics_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
                mock_get_engine.return_value = mock_engine
                
                with patch('src.gateway.web.strategy_execution_service.set_execution_metrics_cache') as mock_set_cache:
                    result = await get_execution_metrics()
                    
                    # 验证结果
                    self.assertEqual(result["avg_latency"], 8)
                    self.assertEqual(result["today_signals"], 150)
                    self.assertEqual(result["total_trades"], 75)
                    mock_set_cache.assert_called_once()
    
    async def test_get_realtime_metrics_with_cache(self):
        """
        测试从缓存获取实时指标
        """
        # 模拟缓存返回数据
        mock_cached_data = {
            "metrics": {},
            "stream_metrics": {},
            "strategies": [],
            "history": {"latency": [], "throughput": []}
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_signals_cache') as mock_get_cache:
            mock_get_cache.return_value = mock_cached_data
            
            result = await get_realtime_metrics()
            self.assertEqual(result, mock_cached_data)
    
    async def test_get_realtime_metrics_with_engine(self):
        """
        测试从实时引擎获取实时指标
        """
        # 模拟实时引擎
        mock_strategy = Mock()
        mock_strategy.name = "Test Strategy"
        mock_strategy.strategy_type = "momentum"
        mock_strategy.get_performance_metrics.return_value = {
            "latency": 3
        }
        mock_strategy.signals = [1, 2, 3, 4, 5]
        mock_strategy.position = 2
        mock_strategy.trades = [1, 2]
        
        mock_engine = Mock()
        mock_engine.strategies = {"test_strategy": mock_strategy}
        mock_engine.get_performance_metrics.return_value = {
            "stream_metrics": {
                "processing_latency": 2
            }
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_signals_cache') as mock_get_cache:
            mock_get_cache.return_value = None
            
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
                mock_get_engine.return_value = mock_engine
                
                with patch('src.gateway.web.strategy_execution_service.set_realtime_signals_cache') as mock_set_cache:
                    result = await get_realtime_metrics()
                    
                    # 验证结果
                    self.assertEqual(len(result["strategies"]), 1)
                    self.assertEqual(result["strategies"][0]["name"], "Test Strategy")
                    mock_set_cache.assert_called_once()
    
    async def test_start_strategy_success(self):
        """
        测试成功启动策略
        """
        # 模拟策略配置
        mock_strategy_config = {
            "id": "test_strategy",
            "name": "Test Strategy",
            "type": "trend_following",
            "parameters": {}
        }
        
        mock_engine = Mock()
        
        with patch('src.gateway.web.strategy_execution_service.load_strategy_conceptions') as mock_load_strategies:
            mock_load_strategies.return_value = [mock_strategy_config]
            
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
                mock_get_engine.return_value = mock_engine
                
                result = await start_strategy("test_strategy")
                self.assertTrue(result)
                mock_engine.register_strategy.assert_called_once()
    
    async def test_start_strategy_not_found(self):
        """
        测试启动不存在的策略
        """
        with patch('src.gateway.web.strategy_execution_service.load_strategy_conceptions') as mock_load_strategies:
            mock_load_strategies.return_value = []
            
            result = await start_strategy("non_existent_strategy")
            self.assertFalse(result)
    
    async def test_pause_strategy_success(self):
        """
        测试成功暂停策略
        """
        # 模拟策略
        mock_strategy = Mock()
        
        mock_engine = Mock()
        mock_engine.strategies = {"test_strategy": mock_strategy}
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            
            result = await pause_strategy("test_strategy")
            self.assertTrue(result)
            self.assertFalse(mock_strategy.is_active)
    
    async def test_pause_strategy_not_found(self):
        """
        测试暂停不存在的策略
        """
        mock_engine = Mock()
        mock_engine.strategies = {}
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            
            result = await pause_strategy("non_existent_strategy")
            self.assertFalse(result)
    
    def test_sync_entry_points(self):
        """
        测试同步入口点，确保异步函数可以正常调用
        """
        # 测试get_strategy_execution_status
        result = asyncio.run(get_strategy_execution_status())
        self.assertIsInstance(result, dict)
        
        # 测试get_execution_metrics
        result = asyncio.run(get_execution_metrics())
        self.assertIsInstance(result, dict)
        
        # 测试get_realtime_metrics
        result = asyncio.run(get_realtime_metrics())
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
