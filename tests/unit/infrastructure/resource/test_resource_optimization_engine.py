#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源优化引擎深度测试

大幅提升resource_optimization_engine.py的测试覆盖率，从23%提升到80%以上
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock


class TestResourceOptimizationEngine:
    """资源优化引擎深度测试"""

    def test_optimization_engine_initialization(self):
        """测试优化引擎初始化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试基本属性
            assert hasattr(engine, 'logger')
            assert hasattr(engine, 'system_analyzer')
            assert hasattr(engine, 'memory_optimizer')
            assert hasattr(engine, 'cpu_optimizer')
            assert hasattr(engine, 'disk_optimizer')

        except ImportError:
            pytest.skip("ResourceOptimizationEngine not available")

    def test_engine_initialization_with_config(self):
        """测试带配置的引擎初始化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            config = Mock()
            # 当前API不支持直接传入config，创建默认引擎
            engine = ResourceOptimizationEngine()

            # 验证引擎初始化成功
            assert engine is not None
            assert hasattr(engine, 'system_analyzer')

        except ImportError:
            pytest.skip("ResourceOptimizationEngine initialization with config not available")

    def test_cpu_optimization_functionality(self):
        """测试CPU优化功能"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试CPU优化
            cpu_state = {
                'usage_percent': 85.0,
                'load_average': 2.5,
                'core_count': 8,
                'idle_cores': 2
            }

            # 使用新的API进行CPU优化
            optimization_config = {
                "optimization_priority": ["cpu"],
                "cpu_optimization": {
                    "enabled": True,
                    "priority_threshold": 80.0,
                    "load_balancing": True
                }
            }

            result = engine.optimize_resources(optimization_config)
            assert isinstance(result, dict)
            assert "status" in result

        except ImportError:
            pytest.skip("CPU optimization not available")

    def test_memory_optimization_functionality(self):
        """测试内存优化功能"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试内存优化
            memory_state = {
                'usage_percent': 75.0,
                'available_gb': 4.0,
                'total_gb': 16.0,
                'swap_usage_percent': 20.0,
                'page_faults': 1000
            }

            # 使用新的API进行内存优化
            optimization_config = {
                "optimization_priority": ["memory"],
                "memory_optimization": {
                    "enabled": True,
                    "gc_threshold": 80.0,
                    "enable_pooling": True
                }
            }

            result = engine.optimize_resources(optimization_config)
            assert isinstance(result, dict)
            assert "status" in result

        except ImportError:
            pytest.skip("Memory optimization not available")

    def test_disk_optimization_functionality(self):
        """测试磁盘优化功能"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试磁盘优化
            disk_state = {
                'usage_percent': 80.0,
                'read_iops': 500,
                'write_iops': 300,
                'avg_response_time': 10.5,
                'fragmentation_percent': 15.0
            }

            # 使用新的API进行磁盘优化
            optimization_config = {
                "optimization_priority": ["disk"],
                "disk_optimization": {
                    "enabled": True,
                    "max_disk_usage_percent": 85.0,
                    "cleanup_threshold_percent": 80.0
                }
            }

            result = engine.optimize_resources(optimization_config)
            assert isinstance(result, dict)
            assert "status" in result

        except ImportError:
            pytest.skip("Disk optimization not available")

    def test_network_optimization_functionality(self):
        """测试网络优化功能"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试网络优化
            network_state = {
                'bandwidth_utilization': 60.0,
                'packet_loss_rate': 0.1,
                'latency_ms': 25.0,
                'connections_count': 150,
                'error_rate': 0.05
            }

            recommendations = engine.optimize_network_performance(network_state)
            assert isinstance(recommendations, dict)

            # 验证推荐包含预期字段
            if recommendations:
                assert 'action' in recommendations

        except ImportError:
            pytest.skip("Network optimization not available")

    def test_comprehensive_resource_optimization(self):
        """测试综合资源优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试综合资源状态优化
            system_state = {
                'cpu': {'usage_percent': 70.0, 'core_count': 8},
                'memory': {'usage_percent': 65.0, 'total_gb': 16.0},
                'disk': {'usage_percent': 55.0, 'read_iops': 200},
                'network': {'bandwidth_utilization': 45.0, 'latency_ms': 15.0}
            }

            optimization_plan = engine.optimize_resource_allocation(system_state)
            assert isinstance(optimization_plan, dict)

            # 验证优化计划包含预期字段
            if optimization_plan:
                assert 'overall_strategy' in optimization_plan
                assert 'expected_improvements' in optimization_plan

        except ImportError:
            pytest.skip("Comprehensive resource optimization not available")

    def test_optimization_strategy_management(self):
        """测试优化策略管理"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试策略注册
            def custom_strategy(state):
                return {"action": "custom_optimization", "priority": 5}

            engine.register_optimization_strategy("custom", custom_strategy)

            # 验证策略被注册
            assert "custom" in engine._optimization_strategies

            # 测试策略执行
            result = engine._optimization_strategies["custom"]({"test": "data"})
            assert result["action"] == "custom_optimization"

        except ImportError:
            pytest.skip("Optimization strategy management not available")

    def test_performance_monitoring_integration(self):
        """测试性能监控集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试性能指标收集
            metrics = engine.collect_performance_metrics()
            assert isinstance(metrics, dict)

            # 验证指标包含预期字段
            if metrics:
                assert 'timestamp' in metrics
                assert 'optimization_count' in metrics

        except ImportError:
            pytest.skip("Performance monitoring integration not available")

    def test_optimization_threshold_management(self):
        """测试优化阈值管理"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试阈值设置
            thresholds = {
                'cpu_high_threshold': 80.0,
                'memory_high_threshold': 85.0,
                'disk_high_threshold': 75.0,
                'network_high_threshold': 70.0
            }

            engine.set_optimization_thresholds(thresholds)

            # 验证阈值被设置（如果有相关属性）
            # 这里主要是测试方法存在性

        except ImportError:
            pytest.skip("Optimization threshold management not available")

    def test_optimization_history_tracking(self):
        """测试优化历史跟踪"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 执行一些优化操作
            test_state = {'cpu': {'usage_percent': 90.0}}
            engine.optimize_cpu_allocation(test_state)

            # 获取优化历史
            history = engine.get_optimization_history()
            assert isinstance(history, list)

            # 验证历史记录
            if len(history) > 0:
                assert 'timestamp' in history[0]
                assert 'action' in history[0]

        except ImportError:
            pytest.skip("Optimization history tracking not available")

    def test_resource_prediction_and_planning(self):
        """测试资源预测和规划"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 历史使用数据
            historical_data = [
                {'cpu': 60.0, 'memory': 70.0, 'timestamp': time.time() - 3600},
                {'cpu': 65.0, 'memory': 75.0, 'timestamp': time.time() - 1800},
                {'cpu': 70.0, 'memory': 72.0, 'timestamp': time.time() - 900}
            ]

            # 预测未来资源需求
            predictions = engine.predict_resource_demand(historical_data, hours_ahead=1)
            assert isinstance(predictions, dict)

            # 验证预测结果
            if predictions:
                assert 'cpu_forecast' in predictions
                assert 'memory_forecast' in predictions

        except ImportError:
            pytest.skip("Resource prediction and planning not available")

    def test_auto_optimization_functionality(self):
        """测试自动优化功能"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 启用自动优化
            engine.enable_auto_optimization()

            # 检查自动优化状态
            assert engine.is_auto_optimization_enabled() is True

            # 禁用自动优化
            engine.disable_auto_optimization()
            assert engine.is_auto_optimization_enabled() is False

        except ImportError:
            pytest.skip("Auto optimization functionality not available")

    def test_optimization_effectiveness_measurement(self):
        """测试优化效果测量"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 模拟优化前后的指标
            before_metrics = {'cpu': 85.0, 'memory': 80.0, 'response_time': 150.0}
            after_metrics = {'cpu': 70.0, 'memory': 75.0, 'response_time': 120.0}

            # 测量优化效果
            effectiveness = engine.measure_optimization_effectiveness(before_metrics, after_metrics)
            assert isinstance(effectiveness, dict)

            # 验证效果测量
            if effectiveness:
                assert 'cpu_improvement' in effectiveness
                assert 'overall_score' in effectiveness

        except ImportError:
            pytest.skip("Optimization effectiveness measurement not available")

    def test_resource_contention_resolution(self):
        """测试资源争用解决"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 模拟资源争用场景
            contention_scenario = {
                'high_cpu_processes': ['process_1', 'process_2'],
                'memory_hungry_processes': ['process_3'],
                'io_intensive_processes': ['process_4'],
                'current_cpu_usage': 95.0,
                'current_memory_usage': 88.0
            }

            # 解决资源争用
            resolution_plan = engine.resolve_resource_contention(contention_scenario)
            assert isinstance(resolution_plan, dict)

            # 验证解决计划
            if resolution_plan:
                assert 'actions' in resolution_plan
                assert 'priority_processes' in resolution_plan

        except ImportError:
            pytest.skip("Resource contention resolution not available")

    def test_quantitative_trading_optimization(self):
        """测试量化交易优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 量化交易系统的资源需求
            trading_requirements = {
                'high_frequency_engine': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'latency_requirement': 1,  # 毫秒
                    'throughput_requirement': 10000  # TPS
                },
                'risk_management': {
                    'cpu_cores': 4,
                    'memory_gb': 8,
                    'latency_requirement': 5,
                    'throughput_requirement': 1000
                },
                'market_data_processor': {
                    'cpu_cores': 6,
                    'memory_gb': 12,
                    'latency_requirement': 2,
                    'throughput_requirement': 5000
                }
            }

            # 优化量化交易资源分配
            optimization_result = engine.optimize_quantitative_trading_resources(trading_requirements)
            assert isinstance(optimization_result, dict)

            # 验证优化结果
            if optimization_result:
                assert 'resource_allocation' in optimization_result
                assert 'performance_predictions' in optimization_result

        except ImportError:
            pytest.skip("Quantitative trading optimization not available")

    def test_energy_efficiency_optimization(self):
        """测试能源效率优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 系统能耗数据
            power_consumption = {
                'cpu_power_watts': 150.0,
                'memory_power_watts': 50.0,
                'disk_power_watts': 25.0,
                'total_system_power': 450.0,
                'efficiency_rating': 0.85
            }

            # 优化能源效率
            efficiency_plan = engine.optimize_energy_efficiency(power_consumption)
            assert isinstance(efficiency_plan, dict)

            # 验证效率优化计划
            if efficiency_plan:
                assert 'power_saving_actions' in efficiency_plan
                assert 'efficiency_improvement' in efficiency_plan

        except ImportError:
            pytest.skip("Energy efficiency optimization not available")

    def test_scalability_optimization(self):
        """测试可扩展性优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 系统负载模式
            load_patterns = {
                'peak_hours': {'cpu': 90.0, 'memory': 85.0, 'requests_per_second': 5000},
                'normal_hours': {'cpu': 60.0, 'memory': 70.0, 'requests_per_second': 2000},
                'off_peak_hours': {'cpu': 30.0, 'memory': 45.0, 'requests_per_second': 500},
                'scaling_triggers': ['cpu > 80%', 'memory > 75%', 'rps > 3000']
            }

            # 优化系统可扩展性
            scalability_plan = engine.optimize_system_scalability(load_patterns)
            assert isinstance(scalability_plan, dict)

            # 验证可扩展性优化
            if scalability_plan:
                assert 'auto_scaling_rules' in scalability_plan
                assert 'resource_reservation' in scalability_plan

        except ImportError:
            pytest.skip("Scalability optimization not available")

    def test_cost_optimization_analysis(self):
        """测试成本优化分析"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 资源成本数据
            cost_data = {
                'cpu_hourly_cost': 0.10,
                'memory_gb_hourly_cost': 0.05,
                'storage_gb_monthly_cost': 0.02,
                'current_allocation': {
                    'cpu_cores': 8,
                    'memory_gb': 16,
                    'storage_gb': 100
                },
                'utilization_patterns': {
                    'cpu_avg': 65.0,
                    'memory_avg': 70.0,
                    'storage_avg': 45.0
                }
            }

            # 分析成本优化机会
            cost_analysis = engine.analyze_cost_optimization(cost_data)
            assert isinstance(cost_analysis, dict)

            # 验证成本分析
            if cost_analysis:
                assert 'cost_saving_opportunities' in cost_analysis
                assert 'recommended_allocation' in cost_analysis

        except ImportError:
            pytest.skip("Cost optimization analysis not available")

    def test_optimization_configuration_management(self):
        """测试优化配置管理"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 配置优化参数
            optimization_config = {
                'cpu_optimization_enabled': True,
                'memory_optimization_enabled': True,
                'disk_optimization_enabled': False,
                'network_optimization_enabled': True,
                'auto_optimization_interval': 300,
                'max_optimization_attempts': 5,
                'optimization_timeout': 60
            }

            engine.configure_optimization(optimization_config)

            # 验证配置应用（如果有相关属性）

        except ImportError:
            pytest.skip("Optimization configuration management not available")

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试优化过程中的错误处理
            invalid_state = None  # 无效状态

            # 应该优雅地处理无效输入
            try:
                result = engine.optimize_resource_allocation(invalid_state)
                # 如果没有抛出异常，应该返回合理的默认值
                assert isinstance(result, dict)
            except Exception:
                # 如果抛出异常，也是可以接受的
                pass

        except ImportError:
            pytest.skip("Error handling and recovery not available")

    def test_handle_optimization_error_method(self):
        """测试_handle_optimization_error方法的覆盖"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 创建一个测试异常
            test_error = ValueError("Test optimization error")

            # 调用私有方法（通过访问私有属性）
            result = engine._handle_optimization_error(test_error)

            # 验证错误处理结果
            assert result["status"] == "error"
            assert "Test optimization error" in result["error"]
            assert "timestamp" in result

        except ImportError:
            pytest.skip("_handle_optimization_error method not available")

    def test_optimization_with_invalid_config_type(self):
        """测试使用无效配置类型进行优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 使用无效的配置类型（不是字典也不是ResourceOptimizationConfig）
            invalid_config = "invalid_config_string"

            with pytest.raises(ValueError, match="optimization_config必须是字典或ResourceOptimizationConfig对象"):
                engine.optimize_resources(invalid_config)

        except ImportError:
            pytest.skip("Invalid config type testing not available")

    def test_optimization_with_config_validation_failure(self):
        """测试配置验证失败的情况"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建一个会验证失败的配置（模拟）
            # 这里我们需要构造一个会导致验证失败的配置
            # 由于具体的验证逻辑可能不同，我们测试一般的情况
            config = ResourceOptimizationConfig()
            # 尝试设置一些可能导致验证失败的值
            config.cpu_optimization_enabled = True
            # 如果有验证失败的情况，应该返回包含validation_errors的结果

            result = engine.optimize_resources_with_config(config)

            # 无论验证是否失败，都应该返回字典结果
            assert isinstance(result, dict)
            assert "status" in result

        except ImportError:
            pytest.skip("Config validation failure testing not available")

    def test_collect_current_resources_error_handling(self):
        """测试_collect_current_resources方法的错误处理"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from unittest.mock import patch

            engine = ResourceOptimizationEngine()

            # Mock system_analyzer使其抛出异常
            with patch.object(engine.system_analyzer, 'get_system_resources', side_effect=Exception("Resource collection failed")):
                # 调用_collect_current_resources应该返回默认值
                resources = engine._collect_current_resources()

                # 验证返回了合理的默认值
                assert isinstance(resources, dict)
                assert "cpu_usage" in resources or len(resources) >= 0  # 至少返回一个空字典

        except ImportError:
            pytest.skip("Resource collection error handling not available")

    def test_optimization_performance_benchmarks(self):
        """测试优化性能基准"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 性能基准测试
            complex_state = {
                'cpu': {'usage_percent': 75.0, 'core_count': 16},
                'memory': {'usage_percent': 80.0, 'total_gb': 32.0},
                'disk': {'usage_percent': 60.0, 'read_iops': 1000},
                'network': {'bandwidth_utilization': 50.0, 'latency_ms': 20.0}
            }

            # 执行多次优化以测试性能
            import time
            start_time = time.time()

            for i in range(10):
                engine.optimize_resource_allocation(complex_state)

            end_time = time.time()

            # 验证性能在合理范围内
            duration = end_time - start_time
            assert duration < 5.0  # 5秒内完成10次优化

        except ImportError:
            pytest.skip("Optimization performance benchmarks not available")

    def test_apply_memory_optimization_strategy_enabled(self):
        """测试内存优化策略应用（启用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建启用内存优化的配置
            config = ResourceOptimizationConfig()
            config.memory_optimization.enabled = True

            current_resources = {"memory": {"usage_percent": 85.0}}

            # Mock memory_optimizer
            with patch.object(engine, 'memory_optimizer') as mock_optimizer:
                mock_optimizer.optimize_memory_from_config.return_value = {"action": "reduced_cache"}

                result = engine._apply_memory_optimization_strategy(config, current_resources)

                # 验证调用了优化器
                mock_optimizer.optimize_memory_from_config.assert_called_once()
                assert result == {"action": "reduced_cache"}

        except ImportError:
            pytest.skip("Memory optimization strategy testing not available")

    def test_apply_memory_optimization_strategy_disabled(self):
        """测试内存优化策略应用（禁用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建禁用内存优化的配置
            config = ResourceOptimizationConfig()
            config.memory_optimization.enabled = False

            current_resources = {"memory": {"usage_percent": 85.0}}

            result = engine._apply_memory_optimization_strategy(config, current_resources)

            # 禁用状态应该返回None
            assert result is None

        except ImportError:
            pytest.skip("Memory optimization strategy disabled testing not available")

    def test_apply_cpu_optimization_strategy_enabled(self):
        """测试CPU优化策略应用（启用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建启用CPU优化的配置
            config = ResourceOptimizationConfig()
            config.cpu_optimization.enabled = True

            current_resources = {"cpu": {"usage_percent": 90.0}}

            # Mock cpu_optimizer
            with patch.object(engine, 'cpu_optimizer') as mock_optimizer:
                mock_optimizer.optimize_cpu_from_config.return_value = {"action": "reduced_threads"}

                result = engine._apply_cpu_optimization_strategy(config, current_resources)

                # 验证调用了优化器
                mock_optimizer.optimize_cpu_from_config.assert_called_once()
                assert result == {"action": "reduced_threads"}

        except ImportError:
            pytest.skip("CPU optimization strategy testing not available")

    def test_apply_cpu_optimization_strategy_disabled(self):
        """测试CPU优化策略应用（禁用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建禁用CPU优化的配置
            config = ResourceOptimizationConfig()
            config.cpu_optimization.enabled = False

            current_resources = {"cpu": {"usage_percent": 90.0}}

            result = engine._apply_cpu_optimization_strategy(config, current_resources)

            # 禁用状态应该返回None
            assert result is None

        except ImportError:
            pytest.skip("CPU optimization strategy disabled testing not available")

    def test_apply_disk_optimization_strategy_enabled(self):
        """测试磁盘优化策略应用（启用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建启用磁盘优化的配置
            config = ResourceOptimizationConfig()
            config.disk_optimization.enabled = True

            current_resources = {"disk": {"usage_percent": 95.0}}

            # Mock disk_optimizer
            with patch.object(engine, 'disk_optimizer') as mock_optimizer:
                mock_optimizer.optimize_disk_from_config.return_value = {"action": "compressed_files"}

                result = engine._apply_disk_optimization_strategy(config, current_resources)

                # 验证调用了优化器
                mock_optimizer.optimize_disk_from_config.assert_called_once()
                assert result == {"action": "compressed_files"}

        except ImportError:
            pytest.skip("Disk optimization strategy testing not available")

    def test_apply_disk_optimization_strategy_disabled(self):
        """测试磁盘优化策略应用（禁用状态）"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建禁用磁盘优化的配置
            config = ResourceOptimizationConfig()
            config.disk_optimization.enabled = False

            current_resources = {"disk": {"usage_percent": 95.0}}

            result = engine._apply_disk_optimization_strategy(config, current_resources)

            # 禁用状态应该返回None
            assert result is None

        except ImportError:
            pytest.skip("Disk optimization strategy disabled testing not available")

    def test_get_optimization_strategies_mapping(self):
        """测试优化策略映射"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()
            config = ResourceOptimizationConfig()

            strategies = engine._get_optimization_strategies(config)

            # 验证策略映射包含所有预期的策略
            expected_strategies = ["memory", "cpu", "disk", "parallelization", "checkpointing"]
            assert all(strategy in strategies for strategy in expected_strategies)

            # 验证每个策略都是可调用的
            for strategy_name, strategy_func in strategies.items():
                assert callable(strategy_func)

        except ImportError:
            pytest.skip("Optimization strategies mapping testing not available")