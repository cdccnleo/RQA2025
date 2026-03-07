"""
性能和质量管理工具模块的边界测试
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import logging

from src.data.integration.enhanced_data_integration_modules.performance_utils import (
    check_data_quality,
    update_avg_response_time,
    monitor_performance,
    get_integration_stats,
    shutdown,
)


class TestCheckDataQuality:
    """测试 check_data_quality 函数"""

    def test_check_data_quality_none_data(self):
        """测试 None 数据"""
        result = check_data_quality(None, "AAPL")
        assert result is None

    def test_check_data_quality_empty_dataframe(self):
        """测试空 DataFrame"""
        data = pd.DataFrame()
        result = check_data_quality(data, "AAPL")
        assert result is None

    def test_check_data_quality_with_quality_monitor(self):
        """测试带质量监控器"""
        data = pd.DataFrame({"price": [100, 101, 102]})
        quality_monitor = Mock()
        quality_monitor.check_quality = Mock(return_value={"score": 0.95})
        
        result = check_data_quality(data, "AAPL", quality_monitor)
        assert result == {"score": 0.95}
        quality_monitor.check_quality.assert_called_once_with(data, "AAPL")

    def test_check_data_quality_without_quality_monitor(self):
        """测试不带质量监控器"""
        data = pd.DataFrame({"price": [100, 101, 102]})
        result = check_data_quality(data, "AAPL", None)
        assert result is None

    def test_check_data_quality_quality_monitor_exception(self):
        """测试质量监控器抛出异常"""
        data = pd.DataFrame({"price": [100, 101, 102]})
        quality_monitor = Mock()
        quality_monitor.check_quality = Mock(side_effect=Exception("Quality check error"))
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            result = check_data_quality(data, "AAPL", quality_monitor)
            assert result is None
            mock_logger.warning.assert_called_once()

    def test_check_data_quality_empty_identifier(self):
        """测试空标识符"""
        data = pd.DataFrame({"price": [100, 101, 102]})
        quality_monitor = Mock()
        quality_monitor.check_quality = Mock(return_value={"score": 0.95})
        
        result = check_data_quality(data, "", quality_monitor)
        assert result == {"score": 0.95}
        quality_monitor.check_quality.assert_called_once_with(data, "")

    def test_check_data_quality_large_dataframe(self):
        """测试大型 DataFrame"""
        data = pd.DataFrame({"price": range(10000)})
        quality_monitor = Mock()
        quality_monitor.check_quality = Mock(return_value={"score": 0.90})
        
        result = check_data_quality(data, "AAPL", quality_monitor)
        assert result == {"score": 0.90}


class TestUpdateAvgResponseTime:
    """测试 update_avg_response_time 函数"""

    def test_update_avg_response_time_first_call(self):
        """测试第一次调用"""
        metrics = {}
        update_avg_response_time(metrics, 100.0)
        assert metrics["total_requests"] == 1
        assert metrics["avg_response_time"] == 10.0  # 0.1 * 100 + 0.9 * 0

    def test_update_avg_response_time_multiple_calls(self):
        """测试多次调用"""
        metrics = {}
        update_avg_response_time(metrics, 100.0)
        update_avg_response_time(metrics, 200.0)
        assert metrics["total_requests"] == 2
        # 第二次：0.1 * 200 + 0.9 * 10 = 20 + 9 = 29
        assert metrics["avg_response_time"] == pytest.approx(29.0, abs=0.1)

    def test_update_avg_response_time_zero_response_time(self):
        """测试零响应时间"""
        metrics = {}
        update_avg_response_time(metrics, 0.0)
        assert metrics["total_requests"] == 1
        assert metrics["avg_response_time"] == 0.0

    def test_update_avg_response_time_negative_response_time(self):
        """测试负响应时间"""
        metrics = {}
        update_avg_response_time(metrics, -10.0)
        assert metrics["total_requests"] == 1
        assert metrics["avg_response_time"] == pytest.approx(-1.0, abs=0.1)

    def test_update_avg_response_time_very_large_response_time(self):
        """测试非常大的响应时间"""
        metrics = {}
        update_avg_response_time(metrics, 1000000.0)
        assert metrics["total_requests"] == 1
        assert metrics["avg_response_time"] == pytest.approx(100000.0, abs=0.1)

    def test_update_avg_response_time_existing_metrics(self):
        """测试已有指标"""
        metrics = {"total_requests": 5, "avg_response_time": 50.0}
        update_avg_response_time(metrics, 100.0)
        assert metrics["total_requests"] == 6
        # 0.1 * 100 + 0.9 * 50 = 10 + 45 = 55
        assert metrics["avg_response_time"] == pytest.approx(55.0, abs=0.1)

    def test_update_avg_response_time_float_precision(self):
        """测试浮点精度"""
        metrics = {}
        update_avg_response_time(metrics, 33.333)
        assert metrics["total_requests"] == 1
        assert isinstance(metrics["avg_response_time"], float)


class TestMonitorPerformance:
    """测试 monitor_performance 函数"""

    def test_monitor_performance_calls_logger(self):
        """测试调用日志记录器"""
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            monitor_performance()
            mock_logger.warning.assert_called_once_with("性能监控功能需要集成上下文对象")

    def test_monitor_performance_no_exception(self):
        """测试不抛出异常"""
        monitor_performance()  # 应该不抛出异常


class TestGetIntegrationStats:
    """测试 get_integration_stats 函数"""

    def test_get_integration_stats_basic(self):
        """测试基本统计信息"""
        metrics = {
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "avg_response_time": 50.0,
            "memory_usage": 0.75,
            "quality_score": 0.95
        }
        result = get_integration_stats(metrics)
        assert result["total_requests"] == 100
        assert result["successful_requests"] == 95
        assert result["failed_requests"] == 5
        assert result["avg_response_time"] == 50.0
        assert result["memory_usage"] == 0.75
        assert result["quality_score"] == 0.95

    def test_get_integration_stats_empty_metrics(self):
        """测试空指标"""
        metrics = {}
        result = get_integration_stats(metrics)
        assert result["total_requests"] == 0
        assert result["successful_requests"] == 0
        assert result["failed_requests"] == 0
        assert result["avg_response_time"] == 0.0
        assert result["memory_usage"] == 0.0
        assert result["quality_score"] == 0.0

    def test_get_integration_stats_with_cache_strategy(self):
        """测试带缓存策略"""
        metrics = {}
        cache_strategy = Mock()
        cache_strategy.get_stats = Mock(return_value={"hit_rate": 0.85, "misses": 10})
        
        result = get_integration_stats(metrics, cache_strategy=cache_strategy)
        assert result["cache_hit_rate"] == 0.85
        assert result["cache_stats"] == {"hit_rate": 0.85, "misses": 10}

    def test_get_integration_stats_without_cache_strategy(self):
        """测试不带缓存策略"""
        metrics = {}
        result = get_integration_stats(metrics, cache_strategy=None)
        assert result["cache_hit_rate"] == 0.0
        assert result["cache_stats"] == {}

    def test_get_integration_stats_cache_strategy_no_get_stats(self):
        """测试缓存策略没有 get_stats 方法"""
        metrics = {}
        cache_strategy = Mock()
        del cache_strategy.get_stats
        
        result = get_integration_stats(metrics, cache_strategy=cache_strategy)
        assert result["cache_stats"] == {}

    def test_get_integration_stats_with_parallel_manager(self):
        """测试带并行管理器"""
        metrics = {}
        parallel_manager = Mock()
        parallel_manager.get_stats = Mock(return_value={"active_tasks": 5, "completed": 100})
        
        result = get_integration_stats(metrics, parallel_manager=parallel_manager)
        assert result["parallel_stats"] == {"active_tasks": 5, "completed": 100}

    def test_get_integration_stats_without_parallel_manager(self):
        """测试不带并行管理器"""
        metrics = {}
        result = get_integration_stats(metrics, parallel_manager=None)
        assert result["parallel_stats"] == {}

    def test_get_integration_stats_parallel_manager_no_get_stats(self):
        """测试并行管理器没有 get_stats 方法"""
        metrics = {}
        parallel_manager = Mock()
        del parallel_manager.get_stats
        
        result = get_integration_stats(metrics, parallel_manager=parallel_manager)
        assert result["parallel_stats"] == {}

    def test_get_integration_stats_all_components(self):
        """测试所有组件"""
        metrics = {"total_requests": 100}
        cache_strategy = Mock()
        cache_strategy.get_stats = Mock(return_value={"hit_rate": 0.8})
        parallel_manager = Mock()
        parallel_manager.get_stats = Mock(return_value={"tasks": 10})
        
        result = get_integration_stats(metrics, cache_strategy, parallel_manager)
        assert result["total_requests"] == 100
        assert result["cache_hit_rate"] == 0.8
        assert result["cache_stats"] == {"hit_rate": 0.8}
        assert result["parallel_stats"] == {"tasks": 10}

    def test_get_integration_stats_performance_metrics_copy(self):
        """测试性能指标副本"""
        metrics = {"total_requests": 100, "custom": "value"}
        result = get_integration_stats(metrics)
        # 修改原始字典不应该影响结果
        metrics["total_requests"] = 200
        assert result["total_requests"] == 100
        assert result["performance_metrics"]["total_requests"] == 100


class TestShutdown:
    """测试 shutdown 函数"""

    def test_shutdown_all_none(self):
        """测试所有参数为 None"""
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown()
            # 应该记录开始和完成日志
            assert mock_logger.info.call_count >= 2

    def test_shutdown_parallel_manager_success(self):
        """测试成功关闭并行管理器"""
        parallel_manager = Mock()
        parallel_manager.shutdown = Mock()
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(parallel_manager=parallel_manager)
            parallel_manager.shutdown.assert_called_once()
            mock_logger.debug.assert_any_call("并行管理器已关闭")

    def test_shutdown_parallel_manager_no_shutdown_method(self):
        """测试并行管理器没有 shutdown 方法"""
        parallel_manager = Mock()
        del parallel_manager.shutdown
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(parallel_manager=parallel_manager)
            mock_logger.warning.assert_any_call("并行管理器没有shutdown方法，跳过关闭")

    def test_shutdown_parallel_manager_exception(self):
        """测试并行管理器关闭时抛出异常"""
        parallel_manager = Mock()
        parallel_manager.shutdown = Mock(side_effect=Exception("Shutdown error"))
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(parallel_manager=parallel_manager)
            mock_logger.error.assert_any_call("关闭并行管理器失败: Shutdown error")

    def test_shutdown_cache_strategy_success(self):
        """测试成功清理缓存策略"""
        cache_strategy = Mock()
        cache_strategy.cleanup = Mock()
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(cache_strategy=cache_strategy)
            cache_strategy.cleanup.assert_called_once()
            mock_logger.debug.assert_any_call("缓存策略已清理")

    def test_shutdown_cache_strategy_no_cleanup_method(self):
        """测试缓存策略没有 cleanup 方法"""
        cache_strategy = Mock()
        del cache_strategy.cleanup
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(cache_strategy=cache_strategy)
            mock_logger.warning.assert_any_call("缓存策略没有cleanup方法，跳过清理")

    def test_shutdown_cache_strategy_exception(self):
        """测试缓存策略清理时抛出异常"""
        cache_strategy = Mock()
        cache_strategy.cleanup = Mock(side_effect=Exception("Cleanup error"))
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(cache_strategy=cache_strategy)
            mock_logger.error.assert_any_call("清理缓存失败: Cleanup error")

    def test_shutdown_quality_monitor(self):
        """测试质量监控器"""
        quality_monitor = Mock()
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(quality_monitor=quality_monitor)
            mock_logger.debug.assert_any_call("质量监控器无需特殊清理")

    def test_shutdown_all_components(self):
        """测试关闭所有组件"""
        parallel_manager = Mock()
        parallel_manager.shutdown = Mock()
        cache_strategy = Mock()
        cache_strategy.cleanup = Mock()
        quality_monitor = Mock()
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(parallel_manager, cache_strategy, quality_monitor)
            parallel_manager.shutdown.assert_called_once()
            cache_strategy.cleanup.assert_called_once()
            mock_logger.debug.assert_any_call("质量监控器无需特殊清理")

    def test_shutdown_multiple_exceptions(self):
        """测试多个组件都抛出异常"""
        parallel_manager = Mock()
        parallel_manager.shutdown = Mock(side_effect=Exception("Parallel error"))
        cache_strategy = Mock()
        cache_strategy.cleanup = Mock(side_effect=Exception("Cache error"))
        
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger') as mock_logger:
            shutdown(parallel_manager, cache_strategy)
            # 两个错误都应该被记录
            error_calls = [call for call in mock_logger.error.call_args_list]
            assert len(error_calls) == 2


class TestEdgeCases:
    """测试边界情况"""

    def test_check_data_quality_none_identifier(self):
        """测试 None 标识符"""
        data = pd.DataFrame({"price": [100]})
        quality_monitor = Mock()
        quality_monitor.check_quality = Mock(return_value={"score": 0.95})
        
        result = check_data_quality(data, None, quality_monitor)
        assert result == {"score": 0.95}
        quality_monitor.check_quality.assert_called_once_with(data, None)

    def test_update_avg_response_time_very_small_value(self):
        """测试非常小的响应时间"""
        metrics = {}
        update_avg_response_time(metrics, 0.0001)
        assert metrics["total_requests"] == 1
        assert metrics["avg_response_time"] == pytest.approx(0.00001, abs=0.000001)

    def test_get_integration_stats_nested_dict(self):
        """测试嵌套字典"""
        metrics = {
            "nested": {
                "value": 100
            }
        }
        result = get_integration_stats(metrics)
        assert "nested" in result["performance_metrics"]

    def test_shutdown_components_with_special_methods(self):
        """测试具有特殊方法的组件"""
        # 创建一个有特殊属性的对象
        class SpecialComponent:
            def shutdown(self):
                return "shutdown"
        
        component = SpecialComponent()
        with patch('src.data.integration.enhanced_data_integration_modules.performance_utils.logger'):
            shutdown(parallel_manager=component)
            # 应该成功调用 shutdown

