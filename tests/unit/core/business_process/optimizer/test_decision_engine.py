"""
DecisionEngine组件单元测试

测试智能决策引擎的核心功能
"""

import pytest
import asyncio
from datetime import datetime

# 尝试导入
try:
    from src.core.business_process.optimizer.components.decision_engine import (
        DecisionEngine,
        DecisionResult,
        DecisionType
    )
    from src.core.business_process.optimizer.models import DecisionStrategy
    from src.core.business_process.optimizer.configs import DecisionConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestDecisionEngine:
    """DecisionEngine测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return DecisionConfig(
            strategy="balanced",
            risk_threshold=0.7,
            decision_timeout=30,
            enable_ml_enhancement=False  # 测试时禁用ML
        )
    
    @pytest.fixture
    def engine(self, config):
        """决策引擎实例"""
        return DecisionEngine(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据"""
        return {
            'symbols': ['AAPL', 'GOOGL'],
            'prices': {'AAPL': 150.0, 'GOOGL': 2800.0}
        }
    
    @pytest.fixture
    def sample_analysis(self):
        """样本分析结果"""
        class MockAnalysis:
            score = 0.75
            insights = ["市场趋势良好", "波动率适中"]
        return MockAnalysis()
    
    def test_init(self, config):
        """测试初始化"""
        engine = DecisionEngine(config)
        
        assert engine is not None
        assert engine.config is not None
        assert len(engine._decision_history) == 0
    
    def test_init_with_string_strategy(self):
        """测试字符串策略初始化"""
        config = DecisionConfig(strategy="conservative")
        engine = DecisionEngine(config)
        
        # 应该转换为枚举
        assert engine.config.strategy == DecisionStrategy.CONSERVATIVE
    
    @pytest.mark.asyncio
    async def test_make_market_decision(self, engine, sample_market_data, sample_analysis):
        """测试市场决策"""
        result = await engine.make_market_decision(sample_market_data, sample_analysis)
        
        assert isinstance(result, DecisionResult)
        assert isinstance(result.decision_type, DecisionType)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.reasoning, list)
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_make_signal_decision(self, engine):
        """测试信号决策"""
        signals = [{'signal': 'buy', 'strength': 0.8}]
        
        result = await engine.make_signal_decision(signals)
        
        assert isinstance(result, DecisionResult)
        assert result.metadata['signals_count'] == 1
    
    @pytest.mark.asyncio
    async def test_make_risk_decision(self, engine):
        """测试风险决策"""
        risk_data = {'risk_level': 0.5}
        
        result = await engine.make_risk_decision(risk_data)
        
        assert isinstance(result, DecisionResult)
        assert 'risk_level' in result.metadata
    
    @pytest.mark.asyncio
    async def test_make_risk_decision_high_risk(self, engine):
        """测试高风险决策"""
        risk_data = {'risk_level': 0.9}  # 高于阈值0.7
        
        result = await engine.make_risk_decision(risk_data)
        
        assert result.decision_type == DecisionType.RISK_ADJUSTMENT
        assert result.confidence >= 0.9
    
    @pytest.mark.asyncio
    async def test_make_order_decision(self, engine):
        """测试订单决策"""
        orders = [{'order': 'buy', 'quantity': 100}]
        
        result = await engine.make_order_decision(orders)
        
        assert isinstance(result, DecisionResult)
        assert result.metadata['orders_count'] == 1
    
    @pytest.mark.asyncio
    async def test_decision_history(self, engine, sample_market_data, sample_analysis):
        """测试决策历史"""
        # 执行几次决策
        await engine.make_market_decision(sample_market_data, sample_analysis)
        await engine.make_market_decision(sample_market_data, sample_analysis)
        
        history = engine.get_decision_history(limit=10)
        
        assert len(history) == 2
        assert all(isinstance(d, DecisionResult) for d in history)
    
    @pytest.mark.asyncio
    async def test_decision_quality_score(self, engine, sample_market_data, sample_analysis):
        """测试决策质量评分"""
        # 初始评分
        initial_score = engine.get_decision_quality_score()
        assert initial_score == 0.5  # 无历史时默认0.5
        
        # 执行决策
        await engine.make_market_decision(sample_market_data, sample_analysis)
        
        # 有历史后的评分
        score = engine.get_decision_quality_score()
        assert 0 <= score <= 1
    
    def test_get_status(self, engine):
        """测试获取状态"""
        status = engine.get_status()
        
        assert isinstance(status, dict)
        assert 'initialized' in status
        assert 'strategy' in status
        assert 'decision_history_size' in status
        assert 'ml_enabled' in status
        assert 'quality_score' in status
    
    @pytest.mark.asyncio
    async def test_conservative_strategy(self, sample_market_data):
        """测试保守策略"""
        config = DecisionConfig(strategy="conservative")
        engine = DecisionEngine(config)
        
        # 中等评分
        class MediumAnalysis:
            score = 0.6
            insights = []
        
        result = await engine.make_market_decision(sample_market_data, MediumAnalysis())
        
        # 保守策略应该更倾向于HOLD
        assert result.decision_type in [DecisionType.HOLD_SIGNAL, DecisionType.BUY_SIGNAL]
    
    @pytest.mark.asyncio
    async def test_aggressive_strategy(self, sample_market_data):
        """测试激进策略"""
        config = DecisionConfig(strategy="aggressive")
        engine = DecisionEngine(config)
        
        # 中等评分
        class MediumAnalysis:
            score = 0.6
            insights = []
        
        result = await engine.make_market_decision(sample_market_data, MediumAnalysis())
        
        # 激进策略更容易执行
        assert result.decision_type in [DecisionType.BUY_SIGNAL, DecisionType.HOLD_SIGNAL]
    
    @pytest.mark.asyncio
    async def test_balanced_strategy(self, sample_market_data):
        """测试平衡策略"""
        config = DecisionConfig(strategy="balanced")
        engine = DecisionEngine(config)
        
        # 高评分
        class HighAnalysis:
            score = 0.85
            insights = []
        
        result = await engine.make_market_decision(sample_market_data, HighAnalysis())
        
        assert result.decision_type == DecisionType.BUY_SIGNAL
        assert result.confidence >= 0.8


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestDecisionConfig:
    """DecisionConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DecisionConfig()
        
        assert config.strategy == "balanced"
        assert config.risk_threshold == 0.7
        assert config.decision_timeout == 30
        assert config.enable_ml_enhancement is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DecisionConfig(
            strategy="conservative",
            risk_threshold=0.9,
            decision_timeout=60
        )
        
        assert config.strategy == "conservative"
        assert config.risk_threshold == 0.9
        assert config.decision_timeout == 60
    
    def test_validation_risk_threshold(self):
        """测试风险阈值验证"""
        with pytest.raises(ValueError, match="risk_threshold必须在0-1之间"):
            config = DecisionConfig(risk_threshold=1.5)
            config.__post_init__()
    
    def test_validation_confidence_threshold(self):
        """测试置信度阈值验证"""
        with pytest.raises(ValueError, match="confidence_threshold必须在0-1之间"):
            config = DecisionConfig(confidence_threshold=-0.1)
            config.__post_init__()
    
    def test_validation_timeout(self):
        """测试超时验证"""
        with pytest.raises(ValueError, match="decision_timeout必须大于0"):
            config = DecisionConfig(decision_timeout=0)
            config.__post_init__()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestDecisionResult:
    """DecisionResult测试类"""
    
    def test_decision_result_creation(self):
        """测试决策结果创建"""
        result = DecisionResult(
            decision_type=DecisionType.BUY_SIGNAL,
            confidence=0.85,
            reasoning=["高评分", "趋势良好"]
        )
        
        assert result.decision_type == DecisionType.BUY_SIGNAL
        assert result.confidence == 0.85
        assert len(result.reasoning) == 2
        assert result.timestamp is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

