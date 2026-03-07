"""
RecommendationGenerator组件单元测试

测试建议生成器的核心功能
"""

import pytest
import asyncio

try:
    from src.core.business_process.optimizer.components.recommendation_generator import (
        RecommendationGenerator,
        Recommendation,
        RecommendationCategory,
        PriorityLevel
    )
    from src.core.business_process.optimizer.configs import RecommendationConfig
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    SKIP_REASON = f"导入失败: {e}"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestRecommendationGenerator:
    """RecommendationGenerator测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return RecommendationConfig(
            min_confidence=0.5,
            max_recommendations=5,
            enable_ai_insights=False
        )
    
    @pytest.fixture
    def generator(self, config):
        """生成器实例"""
        return RecommendationGenerator(config)
    
    @pytest.fixture
    def mock_context(self):
        """模拟上下文"""
        class MockContext:
            process_id = "test_001"
        return MockContext()
    
    @pytest.fixture
    def mock_analysis(self):
        """模拟分析结果"""
        class MockAnalysis:
            recommendations = ["优化性能", "降低风险"]
        return MockAnalysis()
    
    @pytest.fixture
    def mock_execution(self):
        """模拟执行结果"""
        class MockExecution:
            execution_time = 5.0
            metrics = {}
        return MockExecution()
    
    def test_init(self, config):
        """测试初始化"""
        generator = RecommendationGenerator(config)
        
        assert generator is not None
        assert generator.config == config
        assert len(generator._recommendations_cache) == 0
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, generator, mock_context, 
                                           mock_analysis, mock_execution):
        """测试生成建议"""
        recommendations = await generator.generate_recommendations(
            mock_context, mock_analysis, mock_execution
        )
        
        assert isinstance(recommendations, list)
        assert all(isinstance(r, Recommendation) for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_stage_recommendation(self, generator):
        """测试生成阶段建议"""
        stage_result = {
            'status': 'failed',
            'error': 'test error'
        }
        
        recommendation = await generator.generate_stage_recommendation(
            "stage1", stage_result
        )
        
        if recommendation:
            assert isinstance(recommendation, Recommendation)
            assert recommendation.priority == PriorityLevel.HIGH
    
    def test_prioritize_recommendations(self, generator):
        """测试建议优先级排序"""
        recs = [
            Recommendation(
                recommendation_id="r1",
                title="低优先级",
                description="test",
                category=RecommendationCategory.PERFORMANCE,
                priority=PriorityLevel.LOW,
                confidence=0.6
            ),
            Recommendation(
                recommendation_id="r2",
                title="高优先级",
                description="test",
                category=RecommendationCategory.RISK,
                priority=PriorityLevel.HIGH,
                confidence=0.9
            )
        ]
        
        sorted_recs = generator.prioritize_recommendations(recs)
        
        assert len(sorted_recs) == 2
        assert sorted_recs[0].priority == PriorityLevel.HIGH
    
    def test_track_implementation(self, generator):
        """测试追踪实施"""
        result = generator.track_implementation(
            "rec_001", 
            "in_progress",
            progress=0.5,
            notes="进行中"
        )
        
        assert result is True
        
        # 获取状态
        status = generator.get_implementation_status("rec_001")
        assert status is not None
        assert status.status == "in_progress"
        assert status.progress == 0.5
    
    def test_get_active_recommendations(self, generator):
        """测试获取活跃建议"""
        recs = generator.get_active_recommendations()
        
        assert isinstance(recs, list)
    
    def test_get_status(self, generator):
        """测试获取状态"""
        status = generator.get_status()
        
        assert isinstance(status, dict)
        assert 'cached_recommendations' in status
        assert 'tracked_implementations' in status
        assert 'config' in status


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=SKIP_REASON if not IMPORTS_AVAILABLE else "")
class TestRecommendationConfig:
    """RecommendationConfig测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RecommendationConfig()
        
        assert config.min_confidence == 0.6
        assert config.max_recommendations == 10
        assert config.enable_ai_insights is True
    
    def test_validation(self):
        """测试配置验证"""
        with pytest.raises(ValueError):
            config = RecommendationConfig(min_confidence=1.5)
            config.__post_init__()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

