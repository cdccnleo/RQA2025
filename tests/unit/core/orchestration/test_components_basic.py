"""
Orchestration组件基础测试

快速验证ConfigManager, ProcessMonitor, InstancePool的基本功能
"""

import pytest
import time

try:
    from src.core.orchestration.components.config_manager import ProcessConfigManager
    from src.core.orchestration.components.process_monitor import ProcessMonitor
    from src.core.orchestration.components.instance_pool import ProcessInstancePool
    from src.core.orchestration.models import ProcessConfig, ProcessInstance, BusinessProcessState
    from src.core.orchestration.configs import ConfigManagerConfig, MonitorConfig, PoolConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestConfigManager:
    """ConfigManager基础测试"""
    
    def test_init(self):
        """测试初始化"""
        config = ConfigManagerConfig(config_dir="test_configs")
        manager = ProcessConfigManager(config)
        assert manager is not None
    
    def test_save_and_get_config(self):
        """测试保存和获取配置"""
        config_mgr_config = ConfigManagerConfig(config_dir="test_configs", auto_save=False)
        manager = ProcessConfigManager(config_mgr_config)
        
        process_config = ProcessConfig(process_id="test_001", name="测试")
        manager.save_config(process_config)
        
        retrieved = manager.get_config("test_001")
        assert retrieved is not None
        assert retrieved.process_id == "test_001"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestProcessMonitor:
    """ProcessMonitor基础测试"""
    
    def test_init(self):
        """测试初始化"""
        config = MonitorConfig(enable_cleanup=False)
        monitor = ProcessMonitor(config)
        assert monitor is not None
    
    def test_register_process(self):
        """测试注册流程"""
        config = MonitorConfig(enable_cleanup=False)
        monitor = ProcessMonitor(config)
        
        process_config = ProcessConfig(process_id="test", name="测试")
        instance = ProcessInstance(
            instance_id="inst_001",
            process_config=process_config,
            status=BusinessProcessState.IDLE,
            start_time=time.time()
        )
        
        monitor.register_process(instance)
        assert monitor.get_process("inst_001") is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestInstancePool:
    """InstancePool基础测试"""
    
    def test_init(self):
        """测试初始化"""
        config = PoolConfig(max_size=10)
        pool = ProcessInstancePool(config)
        assert pool is not None
    
    def test_get_instance(self):
        """测试获取实例"""
        config = PoolConfig(max_size=10, enable_reuse=False)
        pool = ProcessInstancePool(config)
        
        process_config = ProcessConfig(process_id="test", name="测试")
        instance = pool.get_instance(process_config)
        
        assert instance is not None
        assert instance.process_config == process_config
    
    def test_return_instance(self):
        """测试归还实例"""
        config = PoolConfig(max_size=10, enable_reuse=True)
        pool = ProcessInstancePool(config)
        
        process_config = ProcessConfig(process_id="test", name="测试")
        instance = pool.get_instance(process_config)
        instance.mark_completed()
        
        pool.return_instance(instance)
        
        stats = pool.get_pool_stats()
        assert stats['available'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

