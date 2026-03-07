"""
ProcessExecutor组件单元测试

测试流程执行器的核心功能
"""

import pytest
import asyncio

try:
    from src.core.business_process.optimizer.components.process_executor import (
        ProcessExecutor,
        ExecutionResult,
        ProcessStatus
    )
    from src.core.business_process.optimizer.configs import ExecutionConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestProcessExecutor:
    """ProcessExecutor测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return ExecutionConfig(
            max_concurrent_processes=5,
            execution_timeout=10,
            enable_retry=True,
            max_retries=2
        )
    
    @pytest.fixture
    def executor(self, config):
        """执行器实例"""
        return ProcessExecutor(config)
    
    @pytest.fixture
    def mock_context(self):
        """模拟上下文"""
        class MockContext:
            process_id = "test_process_001"
        return MockContext()
    
    @pytest.fixture
    def mock_decision_engine(self):
        """模拟决策引擎"""
        class MockDecisionEngine:
            async def make_market_decision(self, *args, **kwargs):
                class MockDecision:
                    decision_type = "buy"
                    confidence = 0.8
                return MockDecision()
        return MockDecisionEngine()
    
    def test_init(self, config):
        """测试初始化"""
        executor = ProcessExecutor(config)
        
        assert executor is not None
        assert executor.config == config
        assert len(executor._active_processes) == 0
        assert executor._completed_count == 0
        assert executor._failed_count == 0
    
    @pytest.mark.asyncio
    async def test_execute_process(self, executor, mock_context, mock_decision_engine):
        """测试流程执行"""
        result = await executor.execute_process(mock_context, mock_decision_engine)
        
        assert isinstance(result, ExecutionResult)
        assert result.process_id == "test_process_001"
        assert isinstance(result.status, ProcessStatus)
    
    @pytest.mark.asyncio
    async def test_execute_stage(self, executor, mock_context, mock_decision_engine):
        """测试阶段执行"""
        result = await executor.execute_stage(mock_context, "stage1", mock_decision_engine)
        
        assert isinstance(result, dict)
        assert 'stage' in result
        assert 'status' in result
    
    def test_get_active_processes(self, executor):
        """测试获取活跃流程"""
        active = executor.get_active_processes()
        
        assert isinstance(active, dict)
        assert 'count' in active
        assert 'processes' in active
        assert 'max_concurrent' in active
        assert active['count'] == 0
    
    @pytest.mark.asyncio
    async def test_cancel_process(self, executor):
        """测试取消流程"""
        # 添加一个活跃流程
        executor._active_processes['test_id'] = {'status': 'running'}
        
        result = await executor.cancel_process('test_id')
        
        assert result is True
        assert 'test_id' not in executor._active_processes
    
    def test_get_status(self, executor):
        """测试获取状态"""
        status = executor.get_status()
        
        assert isinstance(status, dict)
        assert 'active_processes' in status
        assert 'completed_count' in status
        assert 'failed_count' in status
        assert 'circuit_breaker_active' in status


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")  
class TestExecutionConfig:
    """ExecutionConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ExecutionConfig()
        
        assert config.max_concurrent_processes == 10
        assert config.execution_timeout == 300
        assert config.enable_retry is True
        assert config.max_retries == 3
    
    def test_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            config = ExecutionConfig(max_concurrent_processes=0)
            config.__post_init__()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

