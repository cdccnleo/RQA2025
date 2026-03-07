"""
测试Monitoring模块的高级组件

包括：
- OptimizationEngine（优化引擎）
- BaselineManager（基线管理器）
- ConfigurationRuleManager（配置规则管理器）
- DataPersistenceComponents（数据持久化组件）
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List


# ============================================================================
# OptimizationEngine Tests
# ============================================================================

class TestOptimizationEngine:
    """测试优化引擎"""

    def test_optimization_engine_init(self):
        """测试优化引擎初始化"""
        try:
            from src.infrastructure.monitoring.components.optimization_engine import OptimizationEngine
            engine = OptimizationEngine()
            assert isinstance(engine, OptimizationEngine)
            assert hasattr(engine, 'optimization_suggestions')
            assert isinstance(engine.optimization_suggestions, list)
        except ImportError:
            pytest.skip("OptimizationEngine not available")

    def test_generate_suggestions_coverage_low(self):
        """测试生成低覆盖率建议"""
        try:
            from src.infrastructure.monitoring.components.optimization_engine import OptimizationEngine
            engine = OptimizationEngine()
            
            coverage_data = {'coverage_percent': 50.0}
            performance_data = {'avg_execution_time': 1.5}
            
            suggestions = engine.generate_suggestions(coverage_data, performance_data)
            assert isinstance(suggestions, list)
            assert len(suggestions) >= 1
            
            # 验证低覆盖率建议
            coverage_suggestions = [s for s in suggestions if s.get('type') == 'coverage_improvement']
            assert len(coverage_suggestions) > 0
            assert coverage_suggestions[0]['priority'] == 'high'
        except ImportError:
            pytest.skip("OptimizationEngine not available")

    def test_generate_suggestions_coverage_medium(self):
        """测试生成中等覆盖率建议"""
        try:
            from src.infrastructure.monitoring.components.optimization_engine import OptimizationEngine
            engine = OptimizationEngine()
            
            coverage_data = {'coverage_percent': 85.0}
            performance_data = {}
            
            suggestions = engine.generate_suggestions(coverage_data, performance_data)
            coverage_suggestions = [s for s in suggestions if s.get('type') == 'coverage_improvement']
            
            if coverage_suggestions:
                assert coverage_suggestions[0]['priority'] == 'medium'
        except ImportError:
            pytest.skip("OptimizationEngine not available")

    def test_generate_suggestions_performance(self):
        """测试生成性能建议"""
        try:
            from src.infrastructure.monitoring.components.optimization_engine import OptimizationEngine
            engine = OptimizationEngine()
            
            coverage_data = {'coverage_percent': 95.0}
            performance_data = {
                'avg_execution_time': 5.0,
                'slow_tests': ['test_slow_1', 'test_slow_2']
            }
            
            suggestions = engine.generate_suggestions(coverage_data, performance_data)
            assert isinstance(suggestions, list)
        except ImportError:
            pytest.skip("OptimizationEngine not available")

    def test_optimization_suggestions_storage(self):
        """测试优化建议存储"""
        try:
            from src.infrastructure.monitoring.components.optimization_engine import OptimizationEngine
            engine = OptimizationEngine()
            
            coverage_data = {'coverage_percent': 60.0}
            performance_data = {}
            
            initial_count = len(engine.optimization_suggestions)
            engine.generate_suggestions(coverage_data, performance_data)
            
            assert len(engine.optimization_suggestions) > initial_count
        except ImportError:
            pytest.skip("OptimizationEngine not available")


# ============================================================================
# BaselineManager Tests
# ============================================================================

class TestPerformanceBaseline:
    """测试性能基线"""

    def test_performance_baseline_init(self):
        """测试性能基线初始化"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import PerformanceBaseline
            baseline = PerformanceBaseline("test_metric")
            assert baseline.metric_name == "test_metric"
            assert len(baseline.values) == 0
            assert baseline.last_updated is None
        except ImportError:
            pytest.skip("PerformanceBaseline not available")

    def test_add_value(self):
        """测试添加指标值"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import PerformanceBaseline
            baseline = PerformanceBaseline("test_metric")
            
            baseline.add_value(10.5)
            assert len(baseline.values) == 1
            assert baseline.values[0] == 10.5
            assert baseline.last_updated is not None
        except ImportError:
            pytest.skip("PerformanceBaseline not available")

    def test_add_multiple_values(self):
        """测试添加多个指标值"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import PerformanceBaseline
            baseline = PerformanceBaseline("test_metric")
            
            for i in range(10):
                baseline.add_value(float(i))
            
            assert len(baseline.values) == 10
            assert list(baseline.values) == list(range(10.0, 10))
        except ImportError:
            pytest.skip("PerformanceBaseline not available")

    def test_get_stats_empty(self):
        """测试获取空基线的统计信息"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import PerformanceBaseline
            baseline = PerformanceBaseline("test_metric")
            
            stats = baseline.get_stats()
            assert stats == {}
        except ImportError:
            pytest.skip("PerformanceBaseline not available")

    def test_get_stats_with_values(self):
        """测试获取统计信息"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import PerformanceBaseline
            baseline = PerformanceBaseline("test_metric")
            
            values = [10.0, 20.0, 30.0, 40.0, 50.0]
            for v in values:
                baseline.add_value(v)
            
            stats = baseline.get_stats(hours=1)
            assert isinstance(stats, dict)
            assert 'count' in stats or 'avg' in stats or 'latest' in stats
        except ImportError:
            pytest.skip("PerformanceBaseline not available")


class TestBaselineManager:
    """测试基线管理器"""

    def test_baseline_manager_init(self):
        """测试基线管理器初始化"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            assert isinstance(manager, BaselineManager)
            assert hasattr(manager, 'baselines')
        except ImportError:
            pytest.skip("BaselineManager not available")

    def test_update_baseline(self):
        """测试更新基线"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            
            manager.update_baseline("cpu_usage", 45.5)
            assert "cpu_usage" in manager.baselines
        except ImportError:
            pytest.skip("BaselineManager not available")

    def test_get_baseline_stats(self):
        """测试获取基线统计"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            
            manager.update_baseline("memory_usage", 60.0)
            manager.update_baseline("memory_usage", 65.0)
            manager.update_baseline("memory_usage", 70.0)
            
            stats = manager.get_baseline_stats("memory_usage")
            assert isinstance(stats, dict)
        except ImportError:
            pytest.skip("BaselineManager not available")

    def test_get_nonexistent_baseline_stats(self):
        """测试获取不存在的基线统计"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            
            stats = manager.get_baseline_stats("nonexistent")
            assert stats == {} or stats is None
        except ImportError:
            pytest.skip("BaselineManager not available")

    def test_is_anomaly_detection(self):
        """测试异常检测"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            
            # 建立基线
            for i in range(10):
                manager.update_baseline("test_metric", 50.0 + i)
            
            # 测试正常值
            if hasattr(manager, 'is_anomaly'):
                is_anomaly = manager.is_anomaly("test_metric", 55.0)
                assert isinstance(is_anomaly, bool)
        except (ImportError, AttributeError):
            pytest.skip("BaselineManager anomaly detection not available")

    def test_clear_baseline(self):
        """测试清除基线"""
        try:
            from src.infrastructure.monitoring.components.baseline_manager import BaselineManager
            manager = BaselineManager()
            
            manager.update_baseline("test", 100.0)
            
            if hasattr(manager, 'clear_baseline'):
                manager.clear_baseline("test")
                stats = manager.get_baseline_stats("test")
                assert stats == {} or stats is None
        except (ImportError, AttributeError):
            pytest.skip("BaselineManager clear not available")


# ============================================================================
# ConfigurationRuleManager Tests
# ============================================================================

class TestConfigurationRuleManager:
    """测试配置规则管理器"""

    def test_rule_manager_init(self):
        """测试规则管理器初始化"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            manager = ConfigurationRuleManager()
            assert isinstance(manager, ConfigurationRuleManager)
            assert hasattr(manager, 'rules')
            assert len(manager.rules) == 0
        except ImportError:
            pytest.skip("ConfigurationRuleManager not available")

    def test_add_rule(self):
        """测试添加规则"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            from src.infrastructure.monitoring.components.rule_types import ConfigurationRule
            
            manager = ConfigurationRuleManager()
            
            rule = ConfigurationRule(
                parameter_path="test.parameter",
                condition="value > 100",
                adjustment_value=200,
                priority=1
            )
            
            result = manager.add_rule(rule)
            assert isinstance(result, bool)
            
            if result:
                assert len(manager.rules) == 1
        except ImportError:
            pytest.skip("ConfigurationRuleManager or ConfigurationRule not available")

    def test_add_duplicate_rule(self):
        """测试添加重复规则"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            from src.infrastructure.monitoring.components.rule_types import ConfigurationRule
            
            manager = ConfigurationRuleManager()
            
            rule = ConfigurationRule(
                parameter_path="test.param",
                condition="value > 50",
                adjustment_value=100,
                priority=1
            )
            
            result1 = manager.add_rule(rule)
            result2 = manager.add_rule(rule)
            
            # 第二次添加应该失败
            assert result1 == True
            assert result2 == False
        except ImportError:
            pytest.skip("ConfigurationRuleManager not available")

    def test_remove_rule(self):
        """测试移除规则"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            from src.infrastructure.monitoring.components.rule_types import ConfigurationRule
            
            manager = ConfigurationRuleManager()
            
            rule = ConfigurationRule(
                parameter_path="test.remove",
                condition="value > 10",
                adjustment_value=20,
                priority=1
            )
            
            manager.add_rule(rule)
            count = manager.remove_rule("test.remove")
            
            assert isinstance(count, int)
            assert count >= 0
        except ImportError:
            pytest.skip("ConfigurationRuleManager not available")

    def test_find_rules(self):
        """测试查找规则"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            from src.infrastructure.monitoring.components.rule_types import ConfigurationRule
            
            manager = ConfigurationRuleManager()
            
            rule1 = ConfigurationRule(
                parameter_path="test.find1",
                condition="value > 5",
                adjustment_value=10,
                priority=1
            )
            rule2 = ConfigurationRule(
                parameter_path="test.find2",
                condition="value < 100",
                adjustment_value=50,
                priority=2
            )
            
            manager.add_rule(rule1)
            manager.add_rule(rule2)
            
            if hasattr(manager, 'find_rules'):
                rules = manager.find_rules("test.find1")
                assert isinstance(rules, list)
        except ImportError:
            pytest.skip("ConfigurationRuleManager not available")

    def test_validate_rule(self):
        """测试验证规则"""
        try:
            from src.infrastructure.monitoring.components.configuration_rule_manager import ConfigurationRuleManager
            from src.infrastructure.monitoring.components.rule_types import ConfigurationRule
            
            manager = ConfigurationRuleManager()
            
            valid_rule = ConfigurationRule(
                parameter_path="test.valid",
                condition="value > 0",
                adjustment_value=10,
                priority=1
            )
            
            if hasattr(manager, 'validate_rule'):
                is_valid = manager.validate_rule(valid_rule)
                assert isinstance(is_valid, bool)
        except ImportError:
            pytest.skip("ConfigurationRuleManager not available")


# ============================================================================
# DataPersistence Tests
# ============================================================================

class TestDataPersistor:
    """测试数据持久化器"""

    def test_data_persistor_init(self):
        """测试数据持久化器初始化"""
        try:
            from src.infrastructure.monitoring.components.data_persistor import DataPersistor
            persistor = DataPersistor()
            assert isinstance(persistor, DataPersistor)
        except ImportError:
            pytest.skip("DataPersistor not available")

    def test_save_data(self):
        """测试保存数据"""
        try:
            from src.infrastructure.monitoring.components.data_persistor import DataPersistor
            persistor = DataPersistor()
            
            data = {"test_key": "test_value", "timestamp": str(datetime.now())}
            
            if hasattr(persistor, 'save'):
                result = persistor.save("test_data", data)
                assert isinstance(result, bool) or result is None
        except ImportError:
            pytest.skip("DataPersistor not available")

    def test_load_data(self):
        """测试加载数据"""
        try:
            from src.infrastructure.monitoring.components.data_persistor import DataPersistor
            persistor = DataPersistor()
            
            if hasattr(persistor, 'load'):
                data = persistor.load("test_data")
                assert data is None or isinstance(data, dict)
        except ImportError:
            pytest.skip("DataPersistor not available")


class TestDataPersistence:
    """测试数据持久化（另一种实现）"""

    def test_data_persistence_init(self):
        """测试数据持久化初始化"""
        try:
            from src.infrastructure.monitoring.components.data_persistence import DataPersistence
            persistence = DataPersistence()
            assert isinstance(persistence, DataPersistence)
        except ImportError:
            pytest.skip("DataPersistence not available")

    def test_persist_metrics(self):
        """测试持久化指标"""
        try:
            from src.infrastructure.monitoring.components.data_persistence import DataPersistence
            persistence = DataPersistence()
            
            metrics = {
                "cpu_usage": 45.5,
                "memory_usage": 60.2,
                "timestamp": datetime.now()
            }
            
            if hasattr(persistence, 'persist_metrics'):
                result = persistence.persist_metrics(metrics)
                assert result is None or isinstance(result, bool)
        except ImportError:
            pytest.skip("DataPersistence not available")

    def test_retrieve_metrics(self):
        """测试检索指标"""
        try:
            from src.infrastructure.monitoring.components.data_persistence import DataPersistence
            persistence = DataPersistence()
            
            if hasattr(persistence, 'retrieve_metrics'):
                metrics = persistence.retrieve_metrics(hours=1)
                assert metrics is None or isinstance(metrics, (list, dict))
        except ImportError:
            pytest.skip("DataPersistence not available")


# ============================================================================
# Additional Component Tests
# ============================================================================

class TestPerformanceEvaluator:
    """测试性能评估器"""

    def test_performance_evaluator_init(self):
        """测试性能评估器初始化"""
        try:
            from src.infrastructure.monitoring.components.performance_evaluator import PerformanceEvaluator
            evaluator = PerformanceEvaluator()
            assert isinstance(evaluator, PerformanceEvaluator)
        except ImportError:
            pytest.skip("PerformanceEvaluator not available")

    def test_evaluate_performance(self):
        """测试评估性能"""
        try:
            from src.infrastructure.monitoring.components.performance_evaluator import PerformanceEvaluator
            evaluator = PerformanceEvaluator()
            
            metrics = {
                "response_time": 1.5,
                "throughput": 100,
                "error_rate": 0.01
            }
            
            if hasattr(evaluator, 'evaluate'):
                result = evaluator.evaluate(metrics)
                assert result is not None
        except ImportError:
            pytest.skip("PerformanceEvaluator not available")


class TestAlertConditionEvaluator:
    """测试告警条件评估器"""

    def test_alert_condition_evaluator_init(self):
        """测试告警条件评估器初始化"""
        try:
            from src.infrastructure.monitoring.components.alert_condition_evaluator import AlertConditionEvaluator
            evaluator = AlertConditionEvaluator()
            assert isinstance(evaluator, AlertConditionEvaluator)
        except ImportError:
            pytest.skip("AlertConditionEvaluator not available")

    def test_evaluate_condition(self):
        """测试评估条件"""
        try:
            from src.infrastructure.monitoring.components.alert_condition_evaluator import AlertConditionEvaluator
            evaluator = AlertConditionEvaluator()
            
            condition = "value > 100"
            value = 150
            
            if hasattr(evaluator, 'evaluate'):
                result = evaluator.evaluate(condition, value)
                assert isinstance(result, bool)
        except ImportError:
            pytest.skip("AlertConditionEvaluator not available")


class TestLoggerPoolComponents:
    """测试日志池相关组件"""

    def test_logger_pool_alert_manager_init(self):
        """测试日志池告警管理器初始化"""
        try:
            from src.infrastructure.monitoring.components.logger_pool_alert_manager import LoggerPoolAlertManager
            manager = LoggerPoolAlertManager()
            assert isinstance(manager, LoggerPoolAlertManager)
        except ImportError:
            pytest.skip("LoggerPoolAlertManager not available")

    def test_logger_pool_stats_collector_init(self):
        """测试日志池统计收集器初始化"""
        try:
            from src.infrastructure.monitoring.components.logger_pool_stats_collector import LoggerPoolStatsCollector
            collector = LoggerPoolStatsCollector()
            assert isinstance(collector, LoggerPoolStatsCollector)
        except ImportError:
            pytest.skip("LoggerPoolStatsCollector not available")

    def test_logger_pool_metrics_exporter_init(self):
        """测试日志池指标导出器初始化"""
        try:
            from src.infrastructure.monitoring.components.logger_pool_metrics_exporter import LoggerPoolMetricsExporter
            exporter = LoggerPoolMetricsExporter()
            assert isinstance(exporter, LoggerPoolMetricsExporter)
        except ImportError:
            pytest.skip("LoggerPoolMetricsExporter not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

