"""
策略暂停功能修复测试

测试 pause_strategy 函数的修复，确保：
1. 可以从持久化存储暂停策略
2. 可以暂停仅存在于实时引擎中的策略
3. 当策略不存在时返回失败
"""

import pytest
from unittest.mock import Mock, patch
import time


class TestPauseStrategyFix:
    """测试策略暂停修复"""
    
    @pytest.mark.asyncio
    async def test_pause_strategy_from_persistence(self):
        """测试从持久化存储暂停策略"""
        mock_state = {
            "strategy_id": "test_strategy_001",
            "name": "测试策略",
            "status": "running"
        }
        
        with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
            mock_load.return_value = mock_state
            
            with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                mock_save.return_value = True
                
                with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
                    mock_engine.return_value = None  # 引擎不可用
                    
                    from src.gateway.web.strategy_execution_service import pause_strategy
                    result = await pause_strategy("test_strategy_001")
                    
                    assert result is True
                    mock_save.assert_called_once()
                    # 验证状态被更新为 paused
                    saved_state = mock_save.call_args[0][1]
                    assert saved_state['status'] == 'paused'
                    assert 'paused_at' in saved_state
    
    @pytest.mark.asyncio
    async def test_pause_strategy_from_engine_only(self):
        """测试暂停仅存在于实时引擎中的策略（修复的场景）"""
        mock_strategy = Mock()
        mock_strategy.name = "model_strategy_1771503574"
        mock_strategy.is_active = True
        
        mock_engine = Mock()
        mock_engine.strategies = {"model_strategy_1771503574": mock_strategy}
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = None  # 持久化存储中不存在
                
                with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                    mock_save.return_value = True
                    
                    from src.gateway.web.strategy_execution_service import pause_strategy
                    result = await pause_strategy("model_strategy_1771503574")
                    
                    assert result is True
                    # 验证策略被暂停
                    assert mock_strategy.is_active is False
                    # 验证在持久化存储中创建了状态
                    mock_save.assert_called_once()
                    saved_state = mock_save.call_args[0][1]
                    assert saved_state['status'] == 'paused'
                    assert saved_state['strategy_id'] == "model_strategy_1771503574"
    
    @pytest.mark.asyncio
    async def test_pause_strategy_not_found(self):
        """测试暂停不存在的策略"""
        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
            mock_engine.return_value = None  # 引擎不可用
            
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = None  # 持久化存储中也不存在
                
                from src.gateway.web.strategy_execution_service import pause_strategy
                result = await pause_strategy("non_existent_strategy")
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_pause_strategy_both_engine_and_persistence(self):
        """测试同时存在于引擎和持久化存储的策略暂停"""
        mock_strategy = Mock()
        mock_strategy.name = "test_strategy"
        mock_strategy.is_active = True
        
        mock_engine = Mock()
        mock_engine.strategies = {"test_strategy": mock_strategy}
        
        mock_state = {
            "strategy_id": "test_strategy",
            "name": "测试策略",
            "status": "running"
        }
        
        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = mock_state
                
                with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                    mock_save.return_value = True
                    
                    from src.gateway.web.strategy_execution_service import pause_strategy
                    result = await pause_strategy("test_strategy")
                    
                    assert result is True
                    # 验证引擎中的策略被暂停
                    assert mock_strategy.is_active is False
                    # 验证持久化状态被更新
                    mock_save.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
