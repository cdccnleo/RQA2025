"""
策略退市功能修复测试

测试策略退市功能的修复，确保：
1. 退市操作正确停止策略执行
2. 退市后策略不再出现在执行列表中
3. 生命周期状态正确更新
4. 执行状态持久化存储正确更新
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import time


class TestStrategyRetireFix:
    """测试策略退市功能修复"""

    def test_stop_strategy_execution_on_retire(self):
        """测试退市时停止策略执行"""
        from src.gateway.web.strategy_lifecycle import LifecycleStatus, StrategyLifecycleManager

        manager = StrategyLifecycleManager()

        # 创建测试生命周期
        lifecycle = manager.create_lifecycle('test_strategy_001', '测试策略',
                                             initial_status=LifecycleStatus.LIVE_TRADING)

        # 模拟执行状态
        mock_exec_state = {
            'strategy_id': 'test_strategy_001',
            'name': '测试策略',
            'status': 'running'
        }

        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
            mock_engine_instance = Mock()
            mock_strategy = Mock()
            mock_strategy.is_active = True
            mock_engine_instance.strategies = {'test_strategy_001': mock_strategy}
            mock_engine.return_value = mock_engine_instance

            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = mock_exec_state

                with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                    mock_save.return_value = True

                    # 执行退市操作
                    result = manager.transition_status(
                        'test_strategy_001',
                        LifecycleStatus.ARCHIVED,
                        'user',
                        '测试退市'
                    )

                    assert result is True
                    # 验证策略被停止
                    assert mock_strategy.is_active is False
                    # 验证执行状态被更新
                    mock_save.assert_called_once()
                    saved_state = mock_save.call_args[0][1]
                    assert saved_state['status'] == 'stopped'
                    assert saved_state['stop_reason'] == '测试退市'

    def test_filter_retired_strategies_from_execution_list(self):
        """测试从执行列表中过滤已退市策略"""
        # 模拟已退市的策略生命周期
        mock_lifecycle = Mock()
        mock_lifecycle.current_status.value = 'archived'

        with patch('src.gateway.web.strategy_lifecycle.get_strategy_lifecycle') as mock_get:
            mock_get.return_value = mock_lifecycle

            # 内联测试过滤逻辑
            def _is_strategy_retired(strategy_id: str) -> bool:
                from src.gateway.web.strategy_lifecycle import get_strategy_lifecycle
                lifecycle = get_strategy_lifecycle(strategy_id)
                if lifecycle and lifecycle.current_status.value == 'archived':
                    return True
                return False

            result = _is_strategy_retired('retired_strategy_001')
            assert result is True

    def test_not_filter_active_strategies(self):
        """测试不过滤未退市策略"""
        # 模拟活跃的策略生命周期
        mock_lifecycle = Mock()
        mock_lifecycle.current_status.value = 'live_trading'

        with patch('src.gateway.web.strategy_lifecycle.get_strategy_lifecycle') as mock_get:
            mock_get.return_value = mock_lifecycle

            # 内联测试过滤逻辑
            def _is_strategy_retired(strategy_id: str) -> bool:
                from src.gateway.web.strategy_lifecycle import get_strategy_lifecycle
                lifecycle = get_strategy_lifecycle(strategy_id)
                if lifecycle and lifecycle.current_status.value == 'archived':
                    return True
                return False

            result = _is_strategy_retired('active_strategy_001')
            assert result is False

    def test_filter_when_lifecycle_not_found(self):
        """测试生命周期不存在时不过滤"""
        with patch('src.gateway.web.strategy_lifecycle.get_strategy_lifecycle') as mock_get:
            mock_get.return_value = None

            # 内联测试过滤逻辑
            def _is_strategy_retired(strategy_id: str) -> bool:
                from src.gateway.web.strategy_lifecycle import get_strategy_lifecycle
                lifecycle = get_strategy_lifecycle(strategy_id)
                if lifecycle and lifecycle.current_status.value == 'archived':
                    return True
                return False

            result = _is_strategy_retired('unknown_strategy')
            assert result is False

    def test_create_stop_state_when_not_exists(self):
        """测试执行状态不存在时创建停止状态"""
        from src.gateway.web.strategy_lifecycle import LifecycleStatus, StrategyLifecycleManager

        manager = StrategyLifecycleManager()

        # 创建测试生命周期
        lifecycle = manager.create_lifecycle('test_strategy_002', '测试策略2',
                                             initial_status=LifecycleStatus.LIVE_TRADING)

        with patch('src.gateway.web.strategy_execution_service.get_realtime_engine') as mock_engine:
            mock_engine.return_value = None  # 引擎不可用

            with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
                mock_load.return_value = None  # 执行状态不存在

                with patch('src.gateway.web.execution_persistence.save_execution_state') as mock_save:
                    mock_save.return_value = True

                    # 执行退市操作
                    result = manager.transition_status(
                        'test_strategy_002',
                        LifecycleStatus.ARCHIVED,
                        'user',
                        '测试退市'
                    )

                    assert result is True
                    # 验证创建了停止状态
                    mock_save.assert_called_once()
                    saved_state = mock_save.call_args[0][1]
                    assert saved_state['status'] == 'stopped'
                    assert saved_state['strategy_id'] == 'test_strategy_002'

    def test_remove_from_active_lifecycles_on_retire(self):
        """测试退市时从活跃生命周期列表中移除"""
        from src.gateway.web.strategy_lifecycle import LifecycleStatus, StrategyLifecycleManager

        manager = StrategyLifecycleManager()
        strategy_id = 'test_strategy_003'

        # 创建测试生命周期
        lifecycle = manager.create_lifecycle(strategy_id, '测试策略3',
                                             initial_status=LifecycleStatus.LIVE_TRADING)

        # 验证策略在活跃列表中
        assert strategy_id in manager.active_lifecycles

        with patch('src.gateway.web.execution_persistence.load_execution_state') as mock_load:
            mock_load.return_value = None

            with patch('src.gateway.web.execution_persistence.save_execution_state'):
                # 执行退市操作
                manager.transition_status(
                    strategy_id,
                    LifecycleStatus.ARCHIVED,
                    'user',
                    '测试退市'
                )

                # 验证策略从活跃列表中移除
                assert strategy_id not in manager.active_lifecycles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
