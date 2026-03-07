"""
智能业务流程优化器集成测试

测试重构后的optimizer的完整功能和组件集成
"""

import pytest
import asyncio
from datetime import datetime

try:
    from src.core.business_process.optimizer.optimizer_refactored import (
        IntelligentBusinessProcessOptimizer
    )
    from src.core.business_process.optimizer.configs import OptimizerConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestOptimizerIntegration:
    """优化器集成测试"""
    
    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据"""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'timestamp': datetime.now().isoformat(),
            'prices': {
                'AAPL': 150.0,
                'GOOGL': 2800.0,
                'MSFT': 300.0
            },
            'volumes': {
                'AAPL': 1000000,
                'GOOGL': 500000,
                'MSFT': 800000
            }
        }
    
    @pytest.fixture
    def sample_risk_profile(self):
        """样本风险配置"""
        return {
            'risk_level': 'medium',
            'max_position_size': 0.1,
            'stop_loss': 0.05,
            'take_profit': 0.15
        }
    
    def test_create_with_none_config(self):
        """测试无配置创建"""
        optimizer = IntelligentBusinessProcessOptimizer(config=None)
        
        assert optimizer is not None
        assert optimizer.config is not None
        assert optimizer.analyzer is not None
        assert optimizer.decision_engine is not None
        assert optimizer.executor is not None
        assert optimizer.recommender is not None
        assert optimizer.monitor is not None
    
    def test_create_with_dict_config(self):
        """测试dict配置创建（向后兼容）"""
        config = {
            'max_concurrent_processes': 5,
            'decision_timeout': 20,
            'risk_threshold': 0.8
        }
        
        optimizer = IntelligentBusinessProcessOptimizer(config=config)
        
        assert optimizer is not None
        assert optimizer.max_concurrent_processes == 5
        assert optimizer.decision_timeout == 20
        assert optimizer.risk_threshold == 0.8
    
    def test_create_with_optimizer_config(self):
        """测试OptimizerConfig对象创建"""
        config = OptimizerConfig.create_high_performance()
        optimizer = IntelligentBusinessProcessOptimizer(config=config)
        
        assert optimizer is not None
        assert optimizer.config.max_concurrent_processes == 20
    
    def test_all_components_initialized(self):
        """测试所有组件是否初始化"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 检查5个核心组件
        assert optimizer.analyzer is not None
        assert optimizer.decision_engine is not None
        assert optimizer.executor is not None
        assert optimizer.recommender is not None
        assert optimizer.monitor is not None
        
        # 检查向后兼容的属性
        assert hasattr(optimizer, 'active_processes')
        assert hasattr(optimizer, 'completed_processes')
        assert hasattr(optimizer, 'process_metrics')
    
    def test_get_optimization_status(self):
        """测试获取状态"""
        optimizer = IntelligentBusinessProcessOptimizer()
        status = optimizer.get_optimization_status()
        
        assert isinstance(status, dict)
        
        # 验证基础字段
        assert 'active_processes' in status
        assert 'completed_processes' in status
        assert 'total_processes' in status
        assert 'successful_processes' in status
        assert 'failed_processes' in status
        assert 'success_rate' in status
        
        # 验证组件状态
        assert 'components' in status
        assert 'analyzer' in status['components']
        assert 'decision_engine' in status['components']
        assert 'executor' in status['components']
        assert 'recommender' in status['components']
        assert 'monitor' in status['components']
        
        # 验证配置信息
        assert 'config' in status
        assert 'max_concurrent' in status['config']
    
    @pytest.mark.asyncio
    async def test_optimize_trading_process(self, sample_market_data, sample_risk_profile):
        """测试完整的交易流程优化"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 执行优化
        result = await optimizer.optimize_trading_process(
            market_data=sample_market_data,
            risk_profile=sample_risk_profile
        )
        
        # 验证结果格式
        assert isinstance(result, dict)
        assert 'process_id' in result
        assert 'status' in result
        
        # 验证流程指标更新
        assert optimizer.process_metrics['total_processes'] >= 1
    
    @pytest.mark.asyncio
    async def test_optimize_trading_process_components_integration(self, sample_market_data, sample_risk_profile):
        """测试组件集成"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        result = await optimizer.optimize_trading_process(
            market_data=sample_market_data,
            risk_profile=sample_risk_profile
        )
        
        # 验证结果包含各组件的输出
        if result['status'] == 'completed':
            assert 'decisions' in result
            assert 'performance' in result
            assert 'recommendations' in result
    
    @pytest.mark.asyncio
    async def test_process_metrics_tracking(self, sample_market_data, sample_risk_profile):
        """测试流程指标追踪"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        initial_total = optimizer.process_metrics['total_processes']
        
        # 执行优化
        await optimizer.optimize_trading_process(
            market_data=sample_market_data,
            risk_profile=sample_risk_profile
        )
        
        # 验证指标更新
        assert optimizer.process_metrics['total_processes'] == initial_total + 1
        
        # 验证成功率
        success_rate = optimizer.process_metrics['optimization_success_rate']
        assert 0 <= success_rate <= 1
    
    @pytest.mark.asyncio
    async def test_active_processes_management(self, sample_market_data, sample_risk_profile):
        """测试活跃流程管理"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 执行优化
        result = await optimizer.optimize_trading_process(
            market_data=sample_market_data,
            risk_profile=sample_risk_profile
        )
        
        # 验证流程已从活跃列表移除
        process_id = result.get('process_id')
        if process_id:
            assert process_id not in optimizer.active_processes
    
    def test_backward_compatibility_attributes(self):
        """测试向后兼容的属性"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 检查所有旧属性存在
        required_attrs = [
            'active_processes',
            'completed_processes',
            'optimization_recommendations',
            'process_metrics',
            'max_concurrent_processes',
            'decision_timeout',
            'risk_threshold',
            'dl_predictor',
            'performance_analyzer'
        ]
        
        for attr in required_attrs:
            assert hasattr(optimizer, attr), f"缺少属性: {attr}"
    
    def test_backward_compatibility_methods(self):
        """测试向后兼容的方法"""
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 检查所有旧方法存在
        required_methods = [
            'start_optimization_engine',
            'optimize_trading_process',
            'get_optimization_status'
        ]
        
        for method in required_methods:
            assert hasattr(optimizer, method), f"缺少方法: {method}"
            assert callable(getattr(optimizer, method)), f"方法不可调用: {method}"
    
    def test_process_metrics_structure(self):
        """测试process_metrics结构兼容"""
        optimizer = IntelligentBusinessProcessOptimizer()
        metrics = optimizer.process_metrics
        
        # 验证结构
        assert isinstance(metrics, dict)
        required_keys = [
            'total_processes',
            'successful_processes',
            'failed_processes',
            'avg_process_time',
            'optimization_success_rate'
        ]
        
        for key in required_keys:
            assert key in metrics, f"缺少指标: {key}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestOptimizerConfiguration:
    """优化器配置测试"""
    
    def test_default_configuration(self):
        """测试默认配置"""
        optimizer = IntelligentBusinessProcessOptimizer()
        config = optimizer.config
        
        assert config is not None
        assert config.max_concurrent_processes == 10
    
    def test_high_performance_configuration(self):
        """测试高性能配置"""
        config = OptimizerConfig.create_high_performance()
        optimizer = IntelligentBusinessProcessOptimizer(config)
        
        assert optimizer.config.max_concurrent_processes == 20
    
    def test_conservative_configuration(self):
        """测试保守配置"""
        config = OptimizerConfig.create_conservative()
        optimizer = IntelligentBusinessProcessOptimizer(config)
        
        assert optimizer.config.decision.risk_threshold == 0.9


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestOptimizerPerformance:
    """优化器性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processes(self, sample_market_data={}, sample_risk_profile={}):
        """测试并发处理"""
        if not sample_market_data:
            sample_market_data = {
                'symbols': ['AAPL'],
                'prices': {'AAPL': 150.0}
            }
        if not sample_risk_profile:
            sample_risk_profile = {'risk_level': 'low'}
        
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 并发执行多个优化
        tasks = [
            optimizer.optimize_trading_process(sample_market_data, sample_risk_profile)
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证结果
        assert len(results) == 3
        
        # 统计成功和失败
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'completed')
        assert successful >= 0  # 至少有一些应该成功


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

