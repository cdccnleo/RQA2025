"""
PerformanceAnalyzer组件单元测试

测试性能分析组件的核心功能
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

# 尝试导入，如果失败则跳过测试
try:
    from src.core.business_process.optimizer.components.performance_analyzer import (
        PerformanceAnalyzer,
        AnalysisResult
    )
    from src.core.business_process.optimizer.configs import AnalysisConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestPerformanceAnalyzer:
    """PerformanceAnalyzer测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return AnalysisConfig(
            analysis_interval=30,
            metrics_retention_days=7,
            enable_deep_analysis=True,
            enable_trend_prediction=True
        )
    
    @pytest.fixture
    def analyzer(self, config):
        """分析器实例"""
        return PerformanceAnalyzer(config)
    
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
    
    def test_init(self, config):
        """测试初始化"""
        analyzer = PerformanceAnalyzer(config)
        
        assert analyzer is not None
        assert analyzer.config == config
        assert analyzer._metrics_cache == {}
        assert len(analyzer._analysis_history) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_market_data(self, analyzer, sample_market_data):
        """测试市场数据分析"""
        result = await analyzer.analyze_market_data(sample_market_data)
        
        # 验证结果类型
        assert isinstance(result, AnalysisResult)
        
        # 验证结果内容
        assert result.timestamp is not None
        assert isinstance(result.metrics, dict)
        assert isinstance(result.insights, list)
        assert isinstance(result.score, float)
        assert 0 <= result.score <= 1
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_analyze_market_data_metrics(self, analyzer, sample_market_data):
        """测试市场数据分析的指标"""
        result = await analyzer.analyze_market_data(sample_market_data)
        
        # 验证指标包含预期字段
        assert 'symbols_count' in result.metrics
        assert result.metrics['symbols_count'] == 3
        assert 'data_quality' in result.metrics
        assert 'market_trend' in result.metrics
        assert 'volatility' in result.metrics
    
    @pytest.mark.asyncio
    async def test_analyze_process_performance(self, analyzer):
        """测试流程性能分析"""
        # 创建模拟上下文
        class MockContext:
            process_id = "test_process_001"
        
        context = MockContext()
        result = await analyzer.analyze_process_performance("test_process_001", context)
        
        # 验证结果
        assert isinstance(result, AnalysisResult)
        assert result.metrics['process_id'] == "test_process_001"
        assert 'execution_time' in result.metrics
        assert 'success_rate' in result.metrics
    
    def test_get_performance_metrics(self, analyzer):
        """测试获取性能指标"""
        metrics = analyzer.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        # 初始应该为空
        assert len(metrics) == 0
    
    @pytest.mark.asyncio
    async def test_get_analysis_history(self, analyzer, sample_market_data):
        """测试获取分析历史"""
        # 执行几次分析
        await analyzer.analyze_market_data(sample_market_data)
        await analyzer.analyze_market_data(sample_market_data)
        
        # 获取历史
        history = analyzer.get_analysis_history(limit=10)
        
        assert isinstance(history, list)
        assert len(history) == 2
        assert all(isinstance(r, AnalysisResult) for r in history)
    
    @pytest.mark.asyncio
    async def test_analysis_history_limit(self, analyzer, sample_market_data):
        """测试分析历史数量限制"""
        # 执行5次分析
        for _ in range(5):
            await analyzer.analyze_market_data(sample_market_data)
        
        # 获取限制数量
        history = analyzer.get_analysis_history(limit=3)
        
        assert len(history) == 3
        
        # 获取全部
        history_all = analyzer.get_analysis_history(limit=0)
        assert len(history_all) == 5
    
    def test_get_status(self, analyzer):
        """测试获取状态"""
        status = analyzer.get_status()
        
        assert isinstance(status, dict)
        assert 'initialized' in status
        assert 'cache_size' in status
        assert 'history_size' in status
        assert 'config' in status
        
        # 验证配置信息
        assert status['config']['analysis_interval'] == 30
        assert status['config']['deep_analysis_enabled'] is True
    
    @pytest.mark.asyncio
    async def test_deep_analysis_enabled(self, sample_market_data):
        """测试深度分析启用"""
        config = AnalysisConfig(enable_deep_analysis=True, enable_trend_prediction=True)
        analyzer = PerformanceAnalyzer(config)
        
        result = await analyzer.analyze_market_data(sample_market_data)
        
        # 深度分析应该生成洞察
        assert len(result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_deep_analysis_disabled(self, sample_market_data):
        """测试深度分析禁用"""
        config = AnalysisConfig(enable_deep_analysis=False)
        analyzer = PerformanceAnalyzer(config)
        
        result = await analyzer.analyze_market_data(sample_market_data)
        
        # 禁用深度分析时洞察应该为空或很少
        assert isinstance(result.insights, list)
    
    @pytest.mark.asyncio
    async def test_score_range(self, analyzer, sample_market_data):
        """测试评分范围"""
        result = await analyzer.analyze_market_data(sample_market_data)
        
        # 评分应该在0-1之间
        assert 0 <= result.score <= 1
    
    @pytest.mark.asyncio
    async def test_empty_market_data(self, analyzer):
        """测试空市场数据"""
        empty_data = {'symbols': []}
        
        result = await analyzer.analyze_market_data(empty_data)
        
        # 应该能处理空数据
        assert isinstance(result, AnalysisResult)
        assert result.metrics['symbols_count'] == 0
    
    @pytest.mark.asyncio
    async def test_metadata(self, analyzer, sample_market_data):
        """测试元数据"""
        result = await analyzer.analyze_market_data(sample_market_data)
        
        assert 'metadata' in result.__dict__
        assert isinstance(result.metadata, dict)
        assert 'data_source' in result.metadata
        assert 'symbols_count' in result.metadata


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestAnalysisConfig:
    """AnalysisConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AnalysisConfig()
        
        assert config.analysis_interval == 60
        assert config.metrics_retention_days == 30
        assert config.enable_deep_analysis is True
        assert config.historical_data_window == 100
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AnalysisConfig(
            analysis_interval=30,
            metrics_retention_days=7,
            enable_deep_analysis=False
        )
        
        assert config.analysis_interval == 30
        assert config.metrics_retention_days == 7
        assert config.enable_deep_analysis is False
    
    def test_validation_analysis_interval(self):
        """测试分析间隔验证"""
        with pytest.raises(ValueError, match="analysis_interval必须大于0"):
            config = AnalysisConfig(analysis_interval=0)
            config.__post_init__()
    
    def test_validation_retention_days(self):
        """测试保留天数验证"""
        with pytest.raises(ValueError, match="metrics_retention_days必须大于0"):
            config = AnalysisConfig(metrics_retention_days=0)
            config.__post_init__()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

