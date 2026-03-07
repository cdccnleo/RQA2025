"""
策略执行服务修复测试

测试策略启动功能的修复，确保：
1. 可以从策略构思目录加载策略
2. 可以从执行状态加载策略（当策略构思不存在时）
3. 可以使用默认配置启动策略（当两者都不存在时）
"""

import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# 测试 start_strategy 函数的修复
class TestStartStrategyFix:
    """测试策略启动修复"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """创建临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_start_strategy_from_conceptions(self):
        """测试从策略构思目录启动策略"""
        # 模拟策略构思数据
        mock_strategy = {
            "id": "test_strategy_001",
            "name": "测试策略",
            "type": "quantitative",
            "parameters": {"param1": 100}
        }
        
        with patch('src.gateway.web.strategy_routes.load_strategy_conceptions') as mock_load:
            mock_load.return_value = [mock_strategy]
            
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
                mock_engine_instance = Mock()
                mock_engine_instance.register_strategy = Mock()
                mock_engine.return_value = mock_engine_instance
                
                with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                    mock_save.return_value = True
                    
                    from src.gateway.web.strategy_execution_service import start_strategy
                    result = await start_strategy("test_strategy_001")
                    
                    assert result is True
                    mock_engine_instance.register_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_strategy_from_execution_state(self):
        """测试从执行状态启动策略（当策略构思不存在时）"""
        # 模拟执行状态数据
        mock_execution_state = {
            "strategy_id": "model_strategy_1771503574",
            "name": "模型策略",
            "type": "ml_model",
            "status": "stopped",
            "parameters": {"model_type": "lstm"}
        }
        
        with patch('src.gateway.web.strategy_routes.load_strategy_conceptions') as mock_load:
            mock_load.return_value = []  # 策略构思不存在
            
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load_state:
                mock_load_state.return_value = mock_execution_state
                
                with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
                    mock_engine_instance = Mock()
                    mock_engine_instance.register_strategy = Mock()
                    mock_engine.return_value = mock_engine_instance
                    
                    with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                        mock_save.return_value = True
                        
                        from src.gateway.web.strategy_execution_service import start_strategy
                        result = await start_strategy("model_strategy_1771503574")
                        
                        assert result is True
                        mock_load_state.assert_called_once_with("model_strategy_1771503574")
                        mock_engine_instance.register_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_strategy_with_default_config(self):
        """测试使用默认配置启动策略（当策略构思和执行状态都不存在时）"""
        with patch('src.gateway.web.strategy_routes.load_strategy_conceptions') as mock_load:
            mock_load.return_value = []  # 策略构思不存在
            
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load_state:
                mock_load_state.return_value = None  # 执行状态也不存在
                
                with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
                    mock_engine_instance = Mock()
                    mock_engine_instance.register_strategy = Mock()
                    mock_engine.return_value = mock_engine_instance
                    
                    with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                        mock_save.return_value = True
                        
                        from src.gateway.web.strategy_execution_service import start_strategy
                        result = await start_strategy("unknown_strategy_001")
                        
                        assert result is True
                        mock_engine_instance.register_strategy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_strategy_engine_not_available(self):
        """测试当实时引擎不可用时启动失败"""
        with patch('src.gateway.web.strategy_routes.load_strategy_conceptions') as mock_load:
            mock_load.return_value = [{"id": "test_strategy", "name": "测试"}]
            
            with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
                mock_engine.return_value = None  # 引擎不可用
                
                from src.gateway.web.strategy_execution_service import start_strategy
                result = await start_strategy("test_strategy")
                
                assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
