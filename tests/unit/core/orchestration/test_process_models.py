"""
流程数据模型测试

验证process_models.py的独立性和功能
"""

import pytest
import time

try:
    from src.core.orchestration.models.process_models import (
        BusinessProcessState,
        ProcessConfig,
        ProcessInstance,
        create_process_config,
        create_process_instance
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestBusinessProcessState:
    """BusinessProcessState枚举测试"""
    
    def test_all_states_defined(self):
        """测试所有状态已定义"""
        expected_states = [
            'IDLE', 'DATA_COLLECTING', 'DATA_QUALITY_CHECKING',
            'FEATURE_EXTRACTING', 'GPU_ACCELERATING',
            'MODEL_PREDICTING', 'MODEL_ENSEMBLING',
            'STRATEGY_DECIDING', 'SIGNAL_GENERATING',
            'RISK_CHECKING', 'COMPLIANCE_VERIFYING',
            'ORDER_GENERATING', 'ORDER_EXECUTING',
            'MONITORING_FEEDBACK', 'COMPLETED', 'ERROR',
            'PAUSED', 'RESUMED'
        ]
        
        for state_name in expected_states:
            assert hasattr(BusinessProcessState, state_name)
    
    def test_state_values(self):
        """测试状态值"""
        assert BusinessProcessState.IDLE.value == "idle"
        assert BusinessProcessState.COMPLETED.value == "completed"
        assert BusinessProcessState.ERROR.value == "error"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestProcessConfig:
    """ProcessConfig数据类测试"""
    
    def test_create_minimal_config(self):
        """测试最小配置创建"""
        config = ProcessConfig(
            process_id="test_001",
            name="测试流程"
        )
        
        assert config.process_id == "test_001"
        assert config.name == "测试流程"
        assert config.enabled is True
        assert config.max_retries == 3
    
    def test_create_full_config(self):
        """测试完整配置创建"""
        config = ProcessConfig(
            process_id="test_002",
            name="完整流程",
            description="测试描述",
            version="2.0.0",
            max_retries=5,
            timeout=7200,
            memory_limit=200
        )
        
        assert config.version == "2.0.0"
        assert config.max_retries == 5
        assert config.timeout == 7200
        assert config.memory_limit == 200
    
    def test_validation_max_retries(self):
        """测试max_retries验证"""
        with pytest.raises(ValueError, match="max_retries不能为负数"):
            config = ProcessConfig(
                process_id="test",
                name="test",
                max_retries=-1
            )
            config.__post_init__()
    
    def test_validation_timeout(self):
        """测试timeout验证"""
        with pytest.raises(ValueError, match="timeout必须大于0"):
            config = ProcessConfig(
                process_id="test",
                name="test",
                timeout=0
            )
            config.__post_init__()
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = ProcessConfig(
            process_id="test_003",
            name="测试"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['process_id'] == "test_003"
        assert config_dict['name'] == "测试"
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'process_id': 'test_004',
            'name': '字典创建',
            'version': '3.0.0'
        }
        
        config = ProcessConfig.from_dict(data)
        
        assert config.process_id == 'test_004'
        assert config.name == '字典创建'
        assert config.version == '3.0.0'


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestProcessInstance:
    """ProcessInstance数据类测试"""
    
    @pytest.fixture
    def sample_config(self):
        """样本配置"""
        return ProcessConfig(
            process_id="test_001",
            name="测试流程"
        )
    
    def test_create_instance(self, sample_config):
        """测试创建实例"""
        instance = ProcessInstance(
            instance_id="inst_001",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        assert instance.instance_id == "inst_001"
        assert instance.status == BusinessProcessState.IDLE
        assert instance.progress == 0.0
    
    def test_update_progress(self, sample_config):
        """测试更新进度"""
        instance = ProcessInstance(
            instance_id="inst_002",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        instance.update_progress(0.5)
        
        assert instance.progress == 0.5
        assert instance.last_updated is not None
    
    def test_update_status(self, sample_config):
        """测试更新状态"""
        instance = ProcessInstance(
            instance_id="inst_003",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        instance.update_status(BusinessProcessState.DATA_COLLECTING)
        
        assert instance.status == BusinessProcessState.DATA_COLLECTING
    
    def test_mark_error(self, sample_config):
        """测试标记错误"""
        instance = ProcessInstance(
            instance_id="inst_004",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        instance.mark_error("测试错误")
        
        assert instance.status == BusinessProcessState.ERROR
        assert instance.error_message == "测试错误"
        assert instance.end_time is not None
    
    def test_mark_completed(self, sample_config):
        """测试标记完成"""
        instance = ProcessInstance(
            instance_id="inst_005",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        instance.mark_completed()
        
        assert instance.status == BusinessProcessState.COMPLETED
        assert instance.progress == 1.0
        assert instance.end_time is not None
    
    def test_get_duration(self, sample_config):
        """测试获取时长"""
        start = time.time()
        instance = ProcessInstance(
            instance_id="inst_006",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=start
        )
        
        time.sleep(0.1)  # 睡眠0.1秒
        duration = instance.get_duration()
        
        assert duration >= 0.1
    
    def test_to_dict(self, sample_config):
        """测试转换为字典"""
        instance = ProcessInstance(
            instance_id="inst_007",
            process_config=sample_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        instance_dict = instance.to_dict()
        
        assert isinstance(instance_dict, dict)
        assert instance_dict['instance_id'] == "inst_007"
        assert instance_dict['status'] == "idle"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_process_config(self):
        """测试create_process_config"""
        config = create_process_config(
            process_id="test_001",
            name="测试",
            max_retries=5
        )
        
        assert isinstance(config, ProcessConfig)
        assert config.process_id == "test_001"
        assert config.max_retries == 5
    
    def test_create_process_instance(self):
        """测试create_process_instance"""
        config = create_process_config("test", "测试")
        instance = create_process_instance(
            instance_id="inst_001",
            process_config=config
        )
        
        assert isinstance(instance, ProcessInstance)
        assert instance.instance_id == "inst_001"
        assert instance.status == BusinessProcessState.IDLE


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

