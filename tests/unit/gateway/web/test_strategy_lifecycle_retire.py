"""
策略生命周期退市功能测试

测试 transition_strategy_status 函数的修复，确保：
1. 现有策略生命周期可以正常退市
2. 不存在生命周期的运行中策略可以退市（自动创建生命周期）
3. 不存在生命周期的非运行策略可以退市（从DRAFT转换）
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil
import os


class TestStrategyLifecycleRetire:
    """测试策略生命周期退市功能"""

    @pytest.fixture
    def temp_lifecycle_dir(self):
        """创建临时生命周期目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_retire_existing_lifecycle(self, temp_lifecycle_dir):
        """测试退市已存在生命周期的策略"""
        with patch('src.gateway.web.strategy_lifecycle.lifecycle_manager') as mock_manager:
            # 模拟已存在的生命周期
            mock_lifecycle = Mock()
            mock_lifecycle.current_status = Mock()
            mock_lifecycle.current_status.value = 'live_trading'
            mock_manager.get_lifecycle.return_value = mock_lifecycle
            mock_manager.transition_status.return_value = True

            from src.gateway.web.strategy_lifecycle import transition_strategy_status
            result = transition_strategy_status('strategy_001', 'archived', 'user', '测试退市')

            assert result is True
            mock_manager.transition_status.assert_called_once()

    def test_retire_strategy_without_lifecycle_running(self, temp_lifecycle_dir):
        """测试退市没有生命周期的运行中策略（model_strategy_1771503574场景）"""
        with patch('src.gateway.web.strategy_lifecycle.lifecycle_manager') as mock_manager:
            # 模拟生命周期不存在
            mock_manager.get_lifecycle.return_value = None

            # 模拟执行状态（策略正在运行）
            mock_exec_state = {
                'name': '模型策略',
                'status': 'running'
            }

            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = mock_exec_state

                # 模拟创建的生命周期
                mock_new_lifecycle = Mock()
                mock_new_lifecycle.current_status = Mock()
                mock_new_lifecycle.current_status.value = 'live_trading'
                mock_manager.create_lifecycle.return_value = mock_new_lifecycle
                mock_manager.transition_status.return_value = True

                from src.gateway.web.strategy_lifecycle import transition_strategy_status, LifecycleStatus
                result = transition_strategy_status('model_strategy_1771503574', 'archived', 'user', '测试退市')

                # 验证创建生命周期时使用了 LIVE_TRADING 初始状态
                mock_manager.create_lifecycle.assert_called_once()
                call_args = mock_manager.create_lifecycle.call_args
                assert call_args[0][0] == 'model_strategy_1771503574'
                assert call_args[0][1] == '模型策略'
                # initial_status 是第3个位置参数
                assert call_args[0][2] == LifecycleStatus.LIVE_TRADING

                assert result is True

    def test_retire_strategy_without_lifecycle_not_running(self, temp_lifecycle_dir):
        """测试退市没有生命周期的非运行策略"""
        with patch('src.gateway.web.strategy_lifecycle.lifecycle_manager') as mock_manager:
            # 模拟生命周期不存在
            mock_manager.get_lifecycle.return_value = None

            # 模拟执行状态（策略未运行）
            mock_exec_state = {
                'name': '测试策略',
                'status': 'stopped'
            }

            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = mock_exec_state

                # 模拟创建的生命周期
                mock_new_lifecycle = Mock()
                mock_new_lifecycle.current_status = Mock()
                mock_new_lifecycle.current_status.value = 'draft'
                mock_manager.create_lifecycle.return_value = mock_new_lifecycle
                mock_manager.transition_status.return_value = True

                from src.gateway.web.strategy_lifecycle import transition_strategy_status, LifecycleStatus
                result = transition_strategy_status('strategy_002', 'archived', 'user', '测试退市')

                # 验证创建生命周期时使用了 DRAFT 初始状态
                call_args = mock_manager.create_lifecycle.call_args
                assert call_args[0][2] == LifecycleStatus.DRAFT

                assert result is True

    def test_retire_strategy_without_any_state(self, temp_lifecycle_dir):
        """测试退市没有任何状态的策略"""
        with patch('src.gateway.web.strategy_lifecycle.lifecycle_manager') as mock_manager:
            # 模拟生命周期不存在
            mock_manager.get_lifecycle.return_value = None

            # 模拟执行状态也不存在
            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = None

                # 模拟创建的生命周期
                mock_new_lifecycle = Mock()
                mock_new_lifecycle.current_status = Mock()
                mock_new_lifecycle.current_status.value = 'draft'
                mock_manager.create_lifecycle.return_value = mock_new_lifecycle
                mock_manager.transition_status.return_value = True

                from src.gateway.web.strategy_lifecycle import transition_strategy_status, LifecycleStatus
                result = transition_strategy_status('strategy_003', 'archived', 'user', '测试退市')

                # 验证使用默认策略ID作为名称，DRAFT作为初始状态
                call_args = mock_manager.create_lifecycle.call_args
                assert call_args[0][0] == 'strategy_003'
                assert call_args[0][1] == 'strategy_003'  # 使用ID作为名称
                assert call_args[0][2] == LifecycleStatus.DRAFT

                assert result is True

    def test_invalid_status(self):
        """测试无效的生命周期状态"""
        from src.gateway.web.strategy_lifecycle import transition_strategy_status
        result = transition_strategy_status('strategy_001', 'invalid_status', 'user', '测试')

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
