"""
BusinessProcessStateMachine组件单元测试

测试状态机的核心功能
"""

import pytest
import time

try:
    from src.core.orchestration.components.state_machine import BusinessProcessStateMachine
    from src.core.orchestration.models.process_models import BusinessProcessState
    from src.core.orchestration.configs import StateMachineConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestStateMachine:
    """StateMachine测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return StateMachineConfig(
            enable_timeout_check=True,
            default_state_timeout=300,
            enable_state_logging=True
        )
    
    @pytest.fixture
    def state_machine(self, config):
        """状态机实例"""
        return BusinessProcessStateMachine(config)
    
    def test_init(self, config):
        """测试初始化"""
        sm = BusinessProcessStateMachine(config)
        
        assert sm is not None
        assert sm.current_state == BusinessProcessState.IDLE
        assert len(sm.state_history) == 0
    
    def test_valid_transition(self, state_machine):
        """测试有效状态转换"""
        result = state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        assert result is True
        assert state_machine.current_state == BusinessProcessState.DATA_COLLECTING
    
    def test_invalid_transition(self, state_machine):
        """测试无效状态转换"""
        # IDLE只能转到DATA_COLLECTING或ERROR
        result = state_machine.transition_to(BusinessProcessState.COMPLETED)
        
        assert result is False
        assert state_machine.current_state == BusinessProcessState.IDLE
    
    def test_transition_chain(self, state_machine):
        """测试状态转换链"""
        # IDLE -> DATA_COLLECTING -> DATA_QUALITY_CHECKING
        assert state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        assert state_machine.transition_to(BusinessProcessState.DATA_QUALITY_CHECKING)
        
        assert state_machine.current_state == BusinessProcessState.DATA_QUALITY_CHECKING
    
    def test_get_state_history(self, state_machine):
        """测试获取状态历史"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        state_machine.transition_to(BusinessProcessState.DATA_QUALITY_CHECKING)
        
        history = state_machine.get_state_history()
        
        assert len(history) == 2
        assert history[0]['from_state'] == BusinessProcessState.IDLE
        assert history[0]['to_state'] == BusinessProcessState.DATA_COLLECTING
    
    def test_state_listener(self, state_machine):
        """测试状态监听器"""
        listener_called = []
        
        def test_listener(state, context):
            listener_called.append(state)
        
        state_machine.add_state_listener(BusinessProcessState.DATA_COLLECTING, test_listener)
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        assert len(listener_called) == 1
        assert listener_called[0] == BusinessProcessState.DATA_COLLECTING
    
    def test_transition_hook(self, state_machine):
        """测试转换钩子"""
        hook_called = []
        
        def test_hook(from_state, to_state, context):
            hook_called.append((from_state, to_state))
        
        state_machine.add_transition_hook(
            BusinessProcessState.IDLE,
            BusinessProcessState.DATA_COLLECTING,
            test_hook
        )
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        assert len(hook_called) == 1
        assert hook_called[0] == (BusinessProcessState.IDLE, BusinessProcessState.DATA_COLLECTING)
    
    def test_remove_state_listener(self, state_machine):
        """测试移除监听器"""
        def test_listener(state, context):
            pass
        
        state_machine.add_state_listener(BusinessProcessState.IDLE, test_listener)
        state_machine.remove_state_listener(BusinessProcessState.IDLE, test_listener)
        
        assert len(state_machine.state_listeners[BusinessProcessState.IDLE]) == 0
    
    def test_get_state_duration(self, state_machine):
        """测试获取状态持续时间"""
        time.sleep(0.1)
        duration = state_machine.get_state_duration()
        
        assert duration >= 0.1
    
    def test_reset(self, state_machine):
        """测试重置"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        state_machine.reset()
        
        assert state_machine.current_state == BusinessProcessState.IDLE
        assert len(state_machine.state_history) == 0
    
    def test_get_status(self, state_machine):
        """测试获取状态"""
        status = state_machine.get_status()
        
        assert isinstance(status, dict)
        assert 'current_state' in status
        assert 'state_duration' in status
        assert 'history_size' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

