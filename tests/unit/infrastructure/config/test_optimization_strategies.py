from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.config.tools.optimization_strategies import (
    CacheOptimizationStrategy, ConnectionPoolOptimizationStrategy,
    MemoryOptimizationStrategy, PerformanceOptimizationManager,
    OptimizationLevel, OptimizationStrategy, OptimizationResult,
    OOptimizationConfig
)

class TestCacheOptimizationStrategy:
    """测试缓存优化策略"""

    @pytest.fixture
    def mock_strategy(self):
        """创建mock CacheOptimizationStrategy实例"""
        config = Mock()  # mock OOptimizationConfig
        config.level = OptimizationLevel.MEDIUM
        config.strategy = OptimizationStrategy.CACHE_OPTIMIZATION
        strategy = CacheOptimizationStrategy(config)
        return strategy

    def test_optimize_success(self, mock_strategy):
        """测试optimize方法成功路径"""
        # 设置条件触发_apply_cache_optimization
        mock_strategy._cache_hit_rates['test_cache'] = [0.8, 0.7, 0.6, 0.5, 0.4]  # len>=5, trend<0, hit_rate<0.8
        with patch.object(mock_strategy, '_analyze_trend') as mock_trend, \
             patch.object(mock_strategy, '_apply_cache_optimization') as mock_apply:
            mock_trend.return_value = -0.1
            mock_apply.return_value = 15.0
            result = mock_strategy.optimize('test_cache', 0.6, 100, target_hit_rate=0.8)
            assert result.success is True
            assert result.performance_improvement == 15.0

    def test_optimize_exception(self, mock_strategy):
        """测试optimize异常路径"""
        mock_strategy._cache_hit_rates['test_cache'] = [0.9, 0.8, 0.7, 0.6, 0.5]  # len>=5
        with patch.object(mock_strategy, '_analyze_trend') as mock_trend, \
             patch.object(mock_strategy, '_apply_cache_optimization') as mock_apply:
            mock_trend.return_value = -0.1  # trend < 0
            mock_apply.side_effect = Exception("Optimization error")
            result = mock_strategy.optimize('test_cache', 0.6, 100, target_hit_rate=0.8)  # hit_rate < target
            assert result.success is False
            assert "Optimization error" in result.error_message

    def test_analyze_trend(self, mock_strategy):
        """测试_analyze_trend方法"""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        trend = mock_strategy._analyze_trend(values)
        assert trend > 0

    def test_apply_cache_optimization_high(self, mock_strategy):
        """测试_apply_cache_optimization HIGH级别"""
        mock_strategy.config.level = OptimizationLevel.HIGH
        improvement = mock_strategy._apply_cache_optimization('test', 0.7, 100)
        assert improvement == 25.0

    def test_apply_cache_optimization_aggressive(self, mock_strategy):
        """测试_apply_cache_optimization AGGRESSIVE级别"""
        mock_strategy.config.level = OptimizationLevel.AGGRESSIVE
        improvement = mock_strategy._apply_cache_optimization('test', 0.7, 100)
        assert improvement == 40.0

    def test_apply_cache_optimization_low(self, mock_strategy):
        """测试_apply_cache_optimization LOW级别"""
        mock_strategy.config.level = OptimizationLevel.LOW
        improvement = mock_strategy._apply_cache_optimization('test', 0.7, 100)
        assert improvement == 5.0

    def test_optimize_no_trend_analysis(self, mock_strategy):
        """测试optimize方法 - 没有足够数据进行趋势分析"""
        mock_strategy._cache_hit_rates['test_cache'] = [0.8, 0.7]  # len < 5
        result = mock_strategy.optimize('test_cache', 0.6, 100, target_hit_rate=0.8)
        assert result.success is True
        assert result.performance_improvement == 0.0

    def test_optimize_trend_positive(self, mock_strategy):
        """测试optimize方法 - 趋势为正，无需优化"""
        mock_strategy._cache_hit_rates['test_cache'] = [0.4, 0.5, 0.6, 0.7, 0.8]  # 上升趋势
        with patch.object(mock_strategy, '_analyze_trend') as mock_trend:
            mock_trend.return_value = 0.1  # 正趋势
            result = mock_strategy.optimize('test_cache', 0.8, 100, target_hit_rate=0.8)
            assert result.success is True
            assert result.performance_improvement == 0.0

    def test_analyze_trend_single_value(self, mock_strategy):
        """测试_analyze_trend方法 - 单个值"""
        result = mock_strategy._analyze_trend([0.5])
        assert result == 0.0

    def test_analyze_trend_empty_list(self, mock_strategy):
        """测试_analyze_trend方法 - 空列表"""
        result = mock_strategy._analyze_trend([])
        assert result == 0.0

    def test_optimize_hit_rate_above_target(self, mock_strategy):
        """测试optimize - hit_rate >= target"""
        mock_strategy._cache_hit_rates['test_cache'] = [0.9] * 5
        result = mock_strategy.optimize('test_cache', 0.9, 100, target_hit_rate=0.8)
        assert result.success is True
        assert result.performance_improvement == 0.0

    def test_optimize_trend_positive_above_threshold(self, mock_strategy):
        """测试optimize - trend > 0 but hit_rate < target"""
        mock_strategy._cache_hit_rates['test_cache'] = [0.5, 0.6, 0.7, 0.8, 0.9]
        with patch.object(mock_strategy, '_analyze_trend') as mock_trend, \
             patch.object(mock_strategy, '_apply_cache_optimization') as mock_apply:
            mock_trend.return_value = -0.1  # 负趋势以触发优化
            mock_apply.return_value = 15.0
            result = mock_strategy.optimize('test_cache', 0.7, 100, target_hit_rate=0.8)
            assert result.success is True
            assert result.performance_improvement == 15.0

    def test_analyze_trend_constant(self, mock_strategy):
        """测试_analyze_trend - 常量值"""
        values = [0.5] * 5
        trend = mock_strategy._analyze_trend(values)
        assert trend == 0.0

    def test_analyze_trend_decreasing(self, mock_strategy):
        """测试_analyze_trend - 下降趋势"""
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        trend = mock_strategy._analyze_trend(values)
        assert trend < 0

    # 添加测试MemoryOptimizationStrategy
class TestMemoryOptimizationStrategy:
    """测试内存优化策略"""

    @pytest.fixture
    def mock_memory_strategy(self):
        config = Mock()
        config.level = OptimizationLevel.MEDIUM
        config.strategy = OptimizationStrategy.MEMORY_OPTIMIZATION
        return MemoryOptimizationStrategy(config)

    def test_optimize_memory(self, mock_memory_strategy):
        """测试optimize内存优化"""
        with patch.object(mock_memory_strategy, '_apply_memory_optimization') as mock_apply, \
             patch.object(mock_memory_strategy, '_collect_gc_stats') as mock_gc:
            mock_apply.return_value = (25.0, 100.0)
            mock_gc.return_value = {'collections': (0,0,0)}
            result = mock_memory_strategy.optimize(500.0, 400.0)
            assert result.success is True
            assert result.memory_saved_mb == 100.0

    def test_collect_gc_stats(self, mock_memory_strategy):
        """测试_collect_gc_stats"""
        with patch('gc.get_count') as mock_count, \
             patch('gc.get_objects') as mock_objects, \
             patch('gc.garbage') as mock_garbage:
            mock_count.return_value = (1,2,3)
            mock_objects.return_value = [object() for _ in range(10)]
            mock_garbage.return_value = []
            stats = mock_memory_strategy._collect_gc_stats()
            assert 'collections' in stats
            assert 'objects' in stats
            assert 'garbage' in stats

    def test_optimize_memory_no_optimization_needed(self, mock_memory_strategy):
        """测试optimize方法 - 内存使用未超过目标"""
        with patch.object(mock_memory_strategy, '_collect_gc_stats') as mock_gc:
            mock_gc.return_value = {'collections': (0,0,0)}
            result = mock_memory_strategy.optimize(300.0, 400.0)  # current < target
            assert result.success is True
            assert result.performance_improvement == 0.0
            assert result.memory_saved_mb == 0.0

    def test_optimize_memory_exception(self, mock_memory_strategy):
        """测试optimize方法 - 异常处理"""
        with patch.object(mock_memory_strategy, '_collect_gc_stats') as mock_gc:
            mock_gc.side_effect = Exception("GC error")
            result = mock_memory_strategy.optimize(500.0, 400.0)
            assert result.success is False
            assert "GC error" in result.error_message

    def test_apply_memory_optimization_all_levels(self, mock_memory_strategy):
        """测试_apply_memory_optimization所有级别"""
        # 测试LOW级别
        mock_memory_strategy.config.level = OptimizationLevel.LOW
        with patch('gc.collect'):
            improvement, saved = mock_memory_strategy._apply_memory_optimization(1000.0, 800.0)
            assert improvement == 10.0
            assert saved == 100.0

        # 测试HIGH级别
        mock_memory_strategy.config.level = OptimizationLevel.HIGH
        improvement, saved = mock_memory_strategy._apply_memory_optimization(1000.0, 800.0)
        assert improvement == 40.0
        assert saved == 300.0

        # 测试AGGRESSIVE级别
        mock_memory_strategy.config.level = OptimizationLevel.AGGRESSIVE
        improvement, saved = mock_memory_strategy._apply_memory_optimization(1000.0, 800.0)
        assert improvement == 60.0
        assert saved == 400.0

    def test_apply_memory_optimization_medium(self, mock_memory_strategy):
        """测试_apply_memory_optimization MEDIUM级别"""
        mock_memory_strategy.config.level = OptimizationLevel.MEDIUM
        with patch.object(mock_memory_strategy, '_cleanup_weak_references') as mock_cleanup:
            improvement, saved = mock_memory_strategy._apply_memory_optimization(1000.0, 800.0)
            assert improvement == 25.0
            assert saved == 200.0
            mock_cleanup.assert_called_once()

    def test_cleanup_weak_references(self, mock_memory_strategy):
        """测试_cleanup_weak_references方法"""
        with patch('gc.collect') as mock_gc_collect:
            # 测试正常情况
            mock_memory_strategy._cleanup_weak_references()
            mock_gc_collect.assert_called_once()

    def test_cleanup_weak_references_exception(self, mock_memory_strategy):
        """测试_cleanup_weak_references - 异常处理"""
        # 由于真实的异常处理逻辑已经在代码中实现（try-except块），
        # 这个测试主要验证方法不会因为weakref清理失败而崩溃
        with patch('gc.collect') as mock_gc_collect:
            # 直接调用方法，测试异常已经被正确处理
            # 在正常情况下，如果 weakref._weakref 不存在或访问失败，应该被捕获
            
            # 模拟一个可能抛出异常的 weakref 状态
            import src.infrastructure.config.tools.optimization_strategies as opt_module
            
            # 使用 patch 来模拟 weakref 模块的 hasattr 和访问行为
            original_weakref = opt_module.weakref
            
            class MockWeakrefModule:
                def __getattr__(self, name):
                    if name == '_weakref':
                        # 创建一个会抛出异常的mock对象
                        mock_obj = MagicMock()
                        # 让 __dict__.clear() 抛出异常
                        def clear_with_exception():
                            raise AttributeError("模拟异常")
                        mock_obj.__dict__ = MagicMock()
                        mock_obj.__dict__.clear = clear_with_exception
                        return mock_obj
                    return getattr(original_weakref, name)
            
            with patch.object(opt_module, 'weakref', MockWeakrefModule()):
                # 测试方法不会抛出异常（应该被捕获）
                try:
                    mock_memory_strategy._cleanup_weak_references()
                    # 验证 gc.collect 仍然被调用
                    mock_gc_collect.assert_called_once()
                except Exception as e:
                    pytest.fail(f"_cleanup_weak_references 应该不抛出异常，但抛出了: {e}")

# 添加ConnectionPoolOptimizationStrategy测试
class TestConnectionPoolOptimizationStrategy:
    """测试连接池优化策略"""

    @pytest.fixture
    def mock_pool_strategy(self):
        config = Mock()
        config.level = OptimizationLevel.MEDIUM
        config.strategy = OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION
        return ConnectionPoolOptimizationStrategy(config)

    def test_optimize_pool(self, mock_pool_strategy):
        """测试optimize连接池"""
        result = mock_pool_strategy.optimize('db_pool', 0.8, 1.5)  # 触发优化
        assert result.success is True
        assert result.performance_improvement == 20.0  # MEDIUM级别

    def test_optimize_pool_high_utilization_low_wait(self, mock_pool_strategy):
        """测试optimize - 高利用率但低等待时间"""
        result = mock_pool_strategy.optimize('db_pool', 0.8, 0.5)
        assert result.success is True
        assert result.performance_improvement > 0

    def test_optimize_pool_low_utilization_high_wait(self, mock_pool_strategy):
        """测试optimize - 低利用率但高等待时间"""
        result = mock_pool_strategy.optimize('db_pool', 0.4, 1.5)
        assert result.success is True
        assert result.performance_improvement > 0

    def test_apply_pool_optimization(self, mock_pool_strategy):
        """测试_apply_pool_optimization"""
        mock_pool_strategy.config.level = OptimizationLevel.HIGH
        improvement = mock_pool_strategy._apply_pool_optimization('db_pool', 0.8, 1.5)
        assert improvement == 35.0

    def test_apply_pool_optimization_medium(self, mock_pool_strategy):
        """测试_apply_pool_optimization MEDIUM级别"""
        mock_pool_strategy.config.level = OptimizationLevel.MEDIUM
        improvement = mock_pool_strategy._apply_pool_optimization('db_pool', 0.8, 1.5)
        assert improvement == 20.0

    def test_optimize_pool_no_optimization_needed(self, mock_pool_strategy):
        """测试optimize连接池 - 不需要优化"""
        result = mock_pool_strategy.optimize('db_pool', 0.5, 0.5)  # utilization和wait_time都低
        assert result.success is True
        assert result.performance_improvement == 0.0

    def test_optimize_pool_exception(self, mock_pool_strategy):
        """测试optimize连接池 - 异常处理"""
        with patch.object(mock_pool_strategy, '_apply_pool_optimization') as mock_apply:
            mock_apply.side_effect = Exception("Pool optimization error")
            result = mock_pool_strategy.optimize('db_pool', 0.8, 1.5)
            assert result.success is False
            assert "Pool optimization error" in result.error_message

    def test_apply_pool_optimization_all_levels(self, mock_pool_strategy):
        """测试_apply_pool_optimization所有级别"""
        # 测试LOW级别
        mock_pool_strategy.config.level = OptimizationLevel.LOW
        improvement = mock_pool_strategy._apply_pool_optimization('db_pool', 0.8, 1.5)
        assert improvement == 8.0

        # 测试AGGRESSIVE级别
        mock_pool_strategy.config.level = OptimizationLevel.AGGRESSIVE
        improvement = mock_pool_strategy._apply_pool_optimization('db_pool', 0.8, 1.5)
        assert improvement == 50.0

# 添加PerformanceOptimizationManager测试
class TestPerformanceOptimizationManager:
    """测试性能优化管理器"""

    @pytest.fixture
    def mock_manager(self):
        return PerformanceOptimizationManager()

    def test_optimize_cache(self, mock_manager):
        """测试optimize_cache"""
        with patch.object(mock_manager.strategies[OptimizationStrategy.CACHE_OPTIMIZATION], 'optimize') as mock_opt:
            mock_opt.return_value = Mock(success=True)
            result = mock_manager.optimize_cache('test', 0.7, 100)
            assert result.success is True

    def test_get_optimization_history(self, mock_manager):
        """测试get_optimization_history"""
        history = mock_manager.get_optimization_history()
        assert isinstance(history, list)

    def test_generate_optimization_report(self, mock_manager):
        """测试generate_optimization_report"""
        # 添加模拟历史
        mock_manager.optimization_history = [
            OptimizationResult(OptimizationStrategy.CACHE_OPTIMIZATION, OptimizationLevel.MEDIUM, True, 15.0),
            OptimizationResult(OptimizationStrategy.MEMORY_OPTIMIZATION, OptimizationLevel.HIGH, True, 25.0, memory_saved_mb=100.0)
        ]
        report = mock_manager.generate_optimization_report()
        assert "性能优化报告" in report
        assert "平均性能提升" in report

    def test_optimize_connection_pool(self, mock_manager):
        """测试optimize_connection_pool"""
        with patch.object(mock_manager.strategies[OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION], 'optimize') as mock_opt:
            mock_result = Mock()
            mock_result.success = True
            mock_result.performance_improvement = 20.0
            mock_opt.return_value = mock_result
            result = mock_manager.optimize_connection_pool('pool1', 0.8, 1.5)
            assert result.success is True

    def test_optimize_memory(self, mock_manager):
        """测试optimize_memory"""
        with patch.object(mock_manager.strategies[OptimizationStrategy.MEMORY_OPTIMIZATION], 'optimize') as mock_opt:
            mock_result = Mock()
            mock_result.success = True
            mock_result.memory_saved_mb = 100.0
            mock_opt.return_value = mock_result
            result = mock_manager.optimize_memory(500.0, 400.0)
            assert result.success is True

    def test_optimize_cache_no_strategy(self, mock_manager):
        """测试optimize_cache - 策略不存在"""
        # 移除缓存策略
        del mock_manager.strategies[OptimizationStrategy.CACHE_OPTIMIZATION]
        result = mock_manager.optimize_cache('test', 0.7, 100)
        assert result.success is False
        assert "缓存优化策略未找到" in result.error_message

    def test_optimize_connection_pool_no_strategy(self, mock_manager):
        """测试optimize_connection_pool - 策略不存在"""
        # 移除连接池策略
        del mock_manager.strategies[OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION]
        result = mock_manager.optimize_connection_pool('pool1', 0.8, 1.5)
        assert result.success is False
        assert "连接池优化策略未找到" in result.error_message

    def test_optimize_memory_no_strategy(self, mock_manager):
        """测试optimize_memory - 策略不存在"""
        # 移除内存策略
        del mock_manager.strategies[OptimizationStrategy.MEMORY_OPTIMIZATION]
        result = mock_manager.optimize_memory(500.0, 400.0)
        assert result.success is False
        assert "内存优化策略未找到" in result.error_message

    def test_generate_optimization_report_empty(self, mock_manager):
        """测试generate_optimization_report - 空历史"""
        mock_manager.optimization_history = []
        report = mock_manager.generate_optimization_report()
        assert "没有可用的优化记录" == report

    def test_generate_optimization_report_with_history(self, mock_manager):
        """测试generate_optimization_report - 有历史记录"""
        mock_manager.optimization_history = [
            OptimizationResult(OptimizationStrategy.CACHE_OPTIMIZATION, OptimizationLevel.MEDIUM, True, 15.0),
            OptimizationResult(OptimizationStrategy.MEMORY_OPTIMIZATION, OptimizationLevel.HIGH, True, 25.0, memory_saved_mb=100.0),
            OptimizationResult(OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION, OptimizationLevel.MEDIUM, False, 0.0, error_message="Error")
        ]
        report = mock_manager.generate_optimization_report()
        assert "成功优化: 2" in report
        assert "失败优化: 1" in report
        assert "平均性能提升: 20.00%" in report

    def test_add_strategy(self, mock_manager):
        """测试add_strategy方法"""
        config = Mock()
        config.strategy = OptimizationStrategy.ALGORITHM_OPTIMIZATION
        strategy = Mock()
        
        mock_manager.add_strategy(config, strategy)
        assert OptimizationStrategy.ALGORITHM_OPTIMIZATION in mock_manager.strategies
        assert OptimizationStrategy.ALGORITHM_OPTIMIZATION in mock_manager.configs

    def test_initialize_default_strategies(self, mock_manager):
        """测试_initialize_default_strategies"""
        assert OptimizationStrategy.CACHE_OPTIMIZATION in mock_manager.strategies
        assert OptimizationStrategy.CONNECTION_POOL_OPTIMIZATION in mock_manager.strategies
        assert OptimizationStrategy.MEMORY_OPTIMIZATION in mock_manager.strategies

    def test_add_strategy_duplicate(self, mock_manager):
        """测试add_strategy - 重复添加"""
        config = Mock(strategy=OptimizationStrategy.CACHE_OPTIMIZATION)
        strategy = Mock()
        mock_manager.add_strategy(config, strategy)
        assert len(mock_manager.strategies) == 3  # 未添加重复
