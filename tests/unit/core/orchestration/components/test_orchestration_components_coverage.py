"""
Orchestration Components组件测试覆盖率补充

补充state_machine、config_manager的测试覆盖
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import os
import json
import time

import pytest

try:
    from src.core.orchestration.components.state_machine import BusinessProcessStateMachine
    from src.core.orchestration.components.config_manager import ProcessConfigManager
    from src.core.orchestration.models.process_models import BusinessProcessState, ProcessConfig
    from src.core.orchestration.configs.orchestrator_configs import (
        StateMachineConfig,
        ConfigManagerConfig
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.fixture
def state_machine_config():
    """创建状态机配置"""
    return StateMachineConfig(
        enable_timeout_check=True,
        default_state_timeout=300,
        enable_state_logging=True,
        enable_hooks=True,
        enable_listeners=True
    )


@pytest.fixture
def state_machine(state_machine_config):
    """创建状态机实例"""
    return BusinessProcessStateMachine(state_machine_config)


@pytest.fixture
def config_manager_config():
    """创建配置管理器配置"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ConfigManagerConfig(
            config_dir=tmpdir,
            auto_save=True,
            enable_validation=True,
            backup_configs=True
        )


@pytest.fixture
def config_manager(config_manager_config):
    """创建配置管理器实例"""
    return ProcessConfigManager(config_manager_config)


@pytest.fixture
def sample_process_config():
    """创建示例流程配置"""
    return ProcessConfig(
        process_id="test_process_1",
        name="Test Process",
        description="Test Description",
        version="1.0.0",
        enabled=True,
        max_retries=3,
        timeout=3600,
        auto_rollback=True,
        parallel_execution=False,
        steps=[],
        parameters={},
        memory_limit=1024
    )


class TestBusinessProcessStateMachine:
    """测试BusinessProcessStateMachine组件"""

    def test_state_machine_initialization(self, state_machine):
        """测试状态机初始化"""
        assert state_machine.current_state == BusinessProcessState.IDLE
        assert len(state_machine.state_history) == 0
        assert state_machine.config == state_machine.config

    def test_transition_to_valid_state(self, state_machine):
        """测试有效状态转换"""
        result = state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        assert result is True
        assert state_machine.current_state == BusinessProcessState.DATA_COLLECTING
        assert len(state_machine.state_history) == 1

    def test_transition_to_invalid_state(self, state_machine):
        """测试无效状态转换"""
        # IDLE不能直接转换到COMPLETED
        result = state_machine.transition_to(BusinessProcessState.COMPLETED)
        
        assert result is False
        assert state_machine.current_state == BusinessProcessState.IDLE

    def test_transition_with_context(self, state_machine):
        """测试带上下文的状态转换"""
        context = {"user_id": 123, "action": "start"}
        result = state_machine.transition_to(
            BusinessProcessState.DATA_COLLECTING,
            context=context
        )
        
        assert result is True
        history = state_machine.get_state_history()
        assert history[0]["context"] == context

    def test_get_current_state(self, state_machine):
        """测试获取当前状态"""
        assert state_machine.get_current_state() == BusinessProcessState.IDLE
        
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        assert state_machine.get_current_state() == BusinessProcessState.DATA_COLLECTING

    def test_get_state_history(self, state_machine):
        """测试获取状态历史"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        state_machine.transition_to(BusinessProcessState.DATA_QUALITY_CHECKING)
        
        history = state_machine.get_state_history()
        assert len(history) == 2
        assert history[0]["from_state"] == BusinessProcessState.IDLE
        assert history[0]["to_state"] == BusinessProcessState.DATA_COLLECTING
        assert history[1]["to_state"] == BusinessProcessState.DATA_QUALITY_CHECKING

    def test_get_state_history_summary(self, state_machine):
        """测试获取状态历史摘要"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        summary = state_machine.get_state_history_summary()
        assert len(summary) == 1
        assert "transition" in summary[0]
        assert "timestamp" in summary[0]

    def test_add_state_listener(self, state_machine):
        """测试添加状态监听器"""
        listener = Mock()
        state_machine.add_state_listener(BusinessProcessState.DATA_COLLECTING, listener)
        
        assert listener in state_machine.state_listeners[BusinessProcessState.DATA_COLLECTING]

    def test_remove_state_listener(self, state_machine):
        """测试移除状态监听器"""
        listener = Mock()
        state_machine.add_state_listener(BusinessProcessState.DATA_COLLECTING, listener)
        state_machine.remove_state_listener(BusinessProcessState.DATA_COLLECTING, listener)
        
        assert listener not in state_machine.state_listeners[BusinessProcessState.DATA_COLLECTING]

    def test_add_transition_hook(self, state_machine):
        """测试添加转换钩子"""
        hook = Mock()
        state_machine.add_transition_hook(
            BusinessProcessState.IDLE,
            BusinessProcessState.DATA_COLLECTING,
            hook
        )
        
        key = (BusinessProcessState.IDLE, BusinessProcessState.DATA_COLLECTING)
        assert hook in state_machine.transition_hooks[key]

    def test_state_listener_notification(self, state_machine):
        """测试状态监听器通知"""
        listener = Mock()
        state_machine.add_state_listener(BusinessProcessState.DATA_COLLECTING, listener)
        
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        listener.assert_called_once()

    def test_transition_hook_execution(self, state_machine):
        """测试转换钩子执行"""
        hook = Mock()
        state_machine.add_transition_hook(
            BusinessProcessState.IDLE,
            BusinessProcessState.DATA_COLLECTING,
            hook
        )
        
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        hook.assert_called_once()

    def test_check_state_timeout_not_enabled(self, state_machine_config):
        """测试超时检查未启用"""
        state_machine_config.enable_timeout_check = False
        sm = BusinessProcessStateMachine(state_machine_config)
        
        result = sm.check_state_timeout()
        assert result is None

    def test_check_state_timeout_not_timed_out(self, state_machine):
        """测试状态未超时"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        
        result = state_machine.check_state_timeout()
        assert result is None

    def test_get_state_duration(self, state_machine):
        """测试获取状态持续时间"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        time.sleep(0.1)
        
        duration = state_machine.get_state_duration()
        assert duration >= 0.1

    def test_reset(self, state_machine):
        """测试重置状态机"""
        state_machine.transition_to(BusinessProcessState.DATA_COLLECTING)
        state_machine.transition_to(BusinessProcessState.DATA_QUALITY_CHECKING)
        
        state_machine.reset()
        
        assert state_machine.current_state == BusinessProcessState.IDLE
        assert len(state_machine.state_history) == 0

    def test_get_status(self, state_machine):
        """测试获取状态机状态"""
        listener = Mock()
        state_machine.add_state_listener(BusinessProcessState.DATA_COLLECTING, listener)
        
        status = state_machine.get_status()
        
        assert "current_state" in status
        assert "state_duration" in status
        assert "history_size" in status
        assert "listeners_count" in status
        assert "hooks_count" in status

    def test_state_transition_chain(self, state_machine):
        """测试状态转换链"""
        # 测试完整的状态转换链
        assert state_machine.transition_to(BusinessProcessState.DATA_COLLECTING) is True
        assert state_machine.transition_to(BusinessProcessState.DATA_QUALITY_CHECKING) is True
        assert state_machine.transition_to(BusinessProcessState.FEATURE_EXTRACTING) is True
        assert state_machine.transition_to(BusinessProcessState.MODEL_PREDICTING) is True
        
        assert state_machine.current_state == BusinessProcessState.MODEL_PREDICTING
        assert len(state_machine.state_history) == 4


class TestProcessConfigManager:
    """测试ProcessConfigManager组件"""

    def test_config_manager_initialization(self, config_manager):
        """测试配置管理器初始化"""
        assert len(config_manager.configs) == 0
        assert config_manager.config is not None

    def test_save_config_success(self, config_manager, sample_process_config):
        """测试成功保存配置"""
        result = config_manager.save_config(sample_process_config)
        
        assert result is True
        assert sample_process_config.process_id in config_manager.configs

    def test_save_config_validation_failure(self, config_manager):
        """测试配置验证失败"""
        invalid_config = ProcessConfig(
            process_id="",  # 无效：空字符串
            name="Test",
            max_retries=3,
            timeout=3600,
            memory_limit=1024
        )
        
        result = config_manager.save_config(invalid_config)
        
        assert result is False
        assert invalid_config.process_id not in config_manager.configs

    def test_get_config_existing(self, config_manager, sample_process_config):
        """测试获取存在的配置"""
        config_manager.save_config(sample_process_config)
        
        retrieved = config_manager.get_config(sample_process_config.process_id)
        
        assert retrieved is not None
        assert retrieved.process_id == sample_process_config.process_id

    def test_get_config_nonexistent(self, config_manager):
        """测试获取不存在的配置"""
        result = config_manager.get_config("nonexistent")
        
        assert result is None

    def test_delete_config_existing(self, config_manager, sample_process_config):
        """测试删除存在的配置"""
        config_manager.save_config(sample_process_config)
        
        result = config_manager.delete_config(sample_process_config.process_id)
        
        assert result is True
        assert sample_process_config.process_id not in config_manager.configs

    def test_delete_config_nonexistent(self, config_manager):
        """测试删除不存在的配置"""
        result = config_manager.delete_config("nonexistent")
        
        assert result is False

    def test_list_configs(self, config_manager, sample_process_config):
        """测试列出所有配置"""
        config_manager.save_config(sample_process_config)
        
        config2 = ProcessConfig(
            process_id="test_process_2",
            name="Test Process 2",
            max_retries=3,
            timeout=3600,
            memory_limit=1024
        )
        config_manager.save_config(config2)
        
        configs = config_manager.list_configs()
        
        assert len(configs) == 2
        assert any(c.process_id == "test_process_1" for c in configs)
        assert any(c.process_id == "test_process_2" for c in configs)

    def test_validate_config_valid(self, config_manager, sample_process_config):
        """测试验证有效配置"""
        errors = config_manager.validate_config(sample_process_config)
        
        assert len(errors) == 0

    def test_validate_config_invalid(self, config_manager):
        """测试验证无效配置"""
        invalid_config = ProcessConfig(
            process_id="",  # 无效
            name="",  # 无效
            max_retries=-1,  # 无效
            timeout=0,  # 无效
            memory_limit=1024
        )
        
        errors = config_manager.validate_config(invalid_config)
        
        assert len(errors) > 0
        assert any("process_id" in error.lower() for error in errors)
        assert any("name" in error.lower() for error in errors)
        assert any("max_retries" in error.lower() for error in errors)
        assert any("timeout" in error.lower() for error in errors)

    def test_get_status(self, config_manager, sample_process_config):
        """测试获取管理器状态"""
        config_manager.save_config(sample_process_config)
        
        status = config_manager.get_status()
        
        assert status["total_configs"] == 1
        assert "config_dir" in status
        assert "auto_save" in status

    def test_load_configs_from_directory(self, config_manager_config):
        """测试从目录加载配置"""
        # 创建临时配置文件
        config_data = {
            "process_id": "loaded_process",
            "name": "Loaded Process",
            "description": "Loaded Description",
            "version": "1.0.0",
            "enabled": True,
            "max_retries": 3,
            "timeout": 3600,
            "auto_rollback": True,
            "parallel_execution": False,
            "steps": [],
            "parameters": {},
            "memory_limit": 1024
        }
        
        config_path = os.path.join(config_manager_config.config_dir, "loaded_process.json")
        os.makedirs(config_manager_config.config_dir, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 创建新的配置管理器，应该加载配置文件
        manager = ProcessConfigManager(config_manager_config)
        
        loaded_config = manager.get_config("loaded_process")
        assert loaded_config is not None
        assert loaded_config.process_id == "loaded_process"
        assert loaded_config.name == "Loaded Process"

    def test_save_config_persists_to_file(self, config_manager, sample_process_config):
        """测试保存配置持久化到文件"""
        config_manager.save_config(sample_process_config)
        
        config_path = os.path.join(config_manager.config.config_dir, f"{sample_process_config.process_id}.json")
        assert os.path.exists(config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data["process_id"] == sample_process_config.process_id
        assert saved_data["name"] == sample_process_config.name

    def test_delete_config_removes_file(self, config_manager, sample_process_config):
        """测试删除配置同时删除文件"""
        config_manager.save_config(sample_process_config)
        config_path = os.path.join(config_manager.config.config_dir, f"{sample_process_config.process_id}.json")
        
        assert os.path.exists(config_path)
        
        config_manager.delete_config(sample_process_config.process_id)
        
        assert not os.path.exists(config_path)

