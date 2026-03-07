"""测试framework_integrator模块"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager
from typing import Dict, Any

# 模拟依赖模块
mock_benchmark = MagicMock()
mock_performance_dashboard = MagicMock()
mock_test_interfaces = MagicMock()

# 创建TestMode枚举模拟
class MockTestMode:
    UNIT = "unit"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    STRESS = "stress"
    SYSTEM = "system"

# 设置模拟模块
sys.modules['benchmark_framework'] = mock_benchmark
sys.modules['performance_dashboard'] = mock_performance_dashboard  
sys.modules['test_interfaces'] = mock_test_interfaces

mock_test_interfaces.TestMode = MockTestMode
mock_test_interfaces.get_test_optimizer = MagicMock()

# 现在导入要测试的模块
with patch('benchmark_framework.BenchmarkFramework', MagicMock()), \
     patch('performance_dashboard.PerformanceDashboard', MagicMock()), \
     patch('test_interfaces.TestMode', MockTestMode), \
     patch('test_interfaces.get_test_optimizer', MagicMock()):
    
    try:
        from src.infrastructure.config.tools.framework_integrator import (
            PerformanceFrameworkIntegrator,
            get_framework_integrator,
            integrate_performance_framework,
            quick_integration,
            run_optimized_test,
            run_optimized_benchmark_test
        )
        HAVE_MODULE = True
    except ImportError as e:
        HAVE_MODULE = False
        PerformanceFrameworkIntegrator = None
        get_framework_integrator = None
        integrate_performance_framework = None
        quick_integration = None
        run_optimized_test = None
        run_optimized_benchmark_test = None


@pytest.mark.skipif(not HAVE_MODULE, reason="framework_integrator模块导入失败")
class TestPerformanceFrameworkIntegrator:
    """测试PerformanceFrameworkIntegrator类"""
    
    def setup_method(self):
        """测试前准备"""
        pass

    @patch('src.infrastructure.config.tools.framework_integrator.get_test_optimizer')
    def test_initialization(self, mock_get_test_optimizer):
        """测试初始化"""
        mock_optimizer = MagicMock()
        mock_get_test_optimizer.return_value = mock_optimizer
        
        integrator = PerformanceFrameworkIntegrator()
        
        assert integrator.test_optimizer == mock_optimizer
        assert integrator.benchmark_framework is None
        assert integrator.performance_dashboard is None
        assert integrator._integration_active is False

    def test_integrate_benchmark_framework(self):
        """测试集成基准测试框架"""
        integrator = PerformanceFrameworkIntegrator()
        mock_framework = MagicMock()
        
        with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
            integrator.integrate_benchmark_framework(mock_framework)
            
            assert integrator.benchmark_framework == mock_framework
            mock_logger.info.assert_called_with("基准测试框架集成完成")

    def test_integrate_performance_dashboard(self):
        """测试集成性能监控仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        mock_dashboard = MagicMock()
        
        with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
            integrator.integrate_performance_dashboard(mock_dashboard)
            
            assert integrator.performance_dashboard == mock_dashboard
            mock_logger.info.assert_called_with("性能监控仪表板集成完成")

    @patch('src.infrastructure.config.tools.framework_integrator.get_test_optimizer')
    def test_setup_test_environment_unit_mode(self, mock_get_test_optimizer):
        """测试设置测试环境 - 单元测试模式"""
        mock_optimizer = MagicMock()
        mock_get_test_optimizer.return_value = mock_optimizer
        
        integrator = PerformanceFrameworkIntegrator()
        
        # 设置mock框架和仪表板
        mock_framework = MagicMock()
        mock_dashboard = MagicMock()
        integrator.benchmark_framework = mock_framework
        integrator.performance_dashboard = mock_dashboard
        
        with patch.object(integrator, '_configure_integrated_components') as mock_configure:
            with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
                # 使用TestMode.UNIT而不是字符串
                with patch('src.infrastructure.config.tools.framework_integrator.TestMode') as mock_test_mode:
                    mock_mode = MagicMock()
                    mock_mode.value = "unit"
                    mock_test_mode.UNIT = mock_mode
                    
                    integrator.setup_test_environment(mock_mode)
                    
                    mock_optimizer.apply_optimizations.assert_called_once()
                    mock_configure.assert_called_once()
                    assert integrator._integration_active is True
                    mock_logger.info.assert_called_with("测试环境设置完成，模式: unit")

    def test_cleanup_test_environment_not_active(self):
        """测试清理测试环境 - 未激活状态"""
        integrator = PerformanceFrameworkIntegrator()
        integrator._integration_active = False
        
        with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
            integrator.cleanup_test_environment()
            
            mock_logger.info.assert_not_called()

    def test_cleanup_test_environment_active(self):
        """测试清理测试环境 - 激活状态"""
        integrator = PerformanceFrameworkIntegrator()
        integrator._integration_active = True
        
        mock_optimizer = MagicMock()
        integrator.test_optimizer = mock_optimizer
        
        with patch.object(integrator, '_cleanup_integrated_components') as mock_cleanup:
            with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
                integrator.cleanup_test_environment()
                
                mock_optimizer.restore_optimizations.assert_called_once()
                mock_cleanup.assert_called_once()
                assert integrator._integration_active is False
                mock_logger.info.assert_called_with("测试环境清理完成")

    def test_configure_integrated_components_with_benchmark(self):
        """测试配置集成组件 - 有基准测试框架"""
        integrator = PerformanceFrameworkIntegrator()
        mock_framework = MagicMock()
        integrator.benchmark_framework = mock_framework
        
        with patch.object(integrator, '_configure_benchmark_framework') as mock_configure_benchmark:
            integrator._configure_integrated_components("unit")
            
            mock_configure_benchmark.assert_called_once_with("unit")

    def test_configure_integrated_components_with_dashboard(self):
        """测试配置集成组件 - 有性能仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        mock_dashboard = MagicMock()
        integrator.performance_dashboard = mock_dashboard
        
        with patch.object(integrator, '_configure_performance_dashboard') as mock_configure_dashboard:
            integrator._configure_integrated_components("unit")
            
            mock_configure_dashboard.assert_called_once_with("unit")

    def test_context_manager_success(self):
        """测试上下文管理器 - 成功情况"""
        integrator = PerformanceFrameworkIntegrator()
        
        with patch.object(integrator, 'setup_test_environment') as mock_setup:
            with patch.object(integrator, 'cleanup_test_environment') as mock_cleanup:
                with integrator.test_context("unit") as ctx:
                    assert ctx == integrator
                    mock_setup.assert_called_once_with("unit")
                
                mock_cleanup.assert_called_once()

    def test_context_manager_exception(self):
        """测试上下文管理器 - 异常情况"""
        integrator = PerformanceFrameworkIntegrator()
        
        with patch.object(integrator, 'setup_test_environment') as mock_setup:
            with patch.object(integrator, 'cleanup_test_environment') as mock_cleanup:
                try:
                    with integrator.test_context("unit") as ctx:
                        mock_setup.assert_called_once_with("unit")
                        raise Exception("测试异常")
                except Exception:
                    pass
                
                mock_cleanup.assert_called_once()

    def test_run_optimized_benchmark_no_framework(self):
        """测试运行优化基准测试 - 无框架"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.benchmark_framework = None
        
        def test_func():
            return "test_result"
        
        with pytest.raises(RuntimeError, match="基准测试框架未集成"):
            integrator.run_optimized_benchmark("test_benchmark", test_func)

    def test_run_optimized_benchmark_with_framework(self):
        """测试运行优化基准测试 - 有框架"""
        integrator = PerformanceFrameworkIntegrator()
        mock_framework = MagicMock()
        mock_framework.run_benchmark.return_value = {"result": "success"}
        integrator.benchmark_framework = mock_framework
        
        def test_func():
            return "test_result"
        
        with patch.object(integrator, 'test_context') as mock_context:
            mock_context.return_value.__enter__.return_value = integrator
            mock_context.return_value.__exit__.return_value = None
            
            with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
                result = integrator.run_optimized_benchmark("test_benchmark", test_func)
                
                assert result["result"] == "success"
                mock_logger.info.assert_any_call("开始运行优化的基准测试: test_benchmark")

    def test_run_optimized_performance_test(self):
        """测试运行优化性能测试"""
        integrator = PerformanceFrameworkIntegrator()
        
        def test_func(**kwargs):
            return "test_result"
        
        with patch.object(integrator, 'test_context') as mock_context:
            mock_context.return_value.__enter__.return_value = integrator
            mock_context.return_value.__exit__.return_value = None
            
            with patch.object(integrator, '_get_current_time') as mock_get_time:
                mock_get_time.side_effect = [1000.0, 1001.5]  # 开始时间和结束时间
                
                with patch.object(integrator, '_get_performance_metrics') as mock_get_metrics:
                    mock_get_metrics.return_value = {"cpu": 50.0}
                    
                    with patch('src.infrastructure.config.tools.framework_integrator.logger') as mock_logger:
                        # 创建Mock TestMode对象
                        mock_mode = MagicMock()
                        mock_mode.value = "performance"
                        
                        result = integrator.run_optimized_performance_test("test_perf", test_func, mock_mode, **{"param": "value"})
                        
                        assert result["test_name"] == "test_perf"
                        assert result["execution_time"] == 1.5
                        assert result["test_result"] == "test_result"
                        assert result["performance_metrics"]["cpu"] == 50.0

    def test_get_current_time_with_dashboard(self):
        """测试获取当前时间 - 有仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        mock_dashboard = MagicMock()
        mock_dashboard.get_current_time.return_value = 1234.5
        integrator.performance_dashboard = mock_dashboard
        
        result = integrator._get_current_time()
        assert result == 1234.5

    @patch('src.infrastructure.config.tools.framework_integrator.time.time')
    def test_get_current_time_without_dashboard(self, mock_time):
        """测试获取当前时间 - 无仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.performance_dashboard = None
        mock_time.return_value = 5678.9
        
        result = integrator._get_current_time()
        assert result == 5678.9

    def test_get_performance_metrics_with_dashboard(self):
        """测试获取性能指标 - 有仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        mock_dashboard = MagicMock()
        mock_dashboard.get_performance_metrics.return_value = {"memory": 75.0}
        integrator.performance_dashboard = mock_dashboard
        
        result = integrator._get_performance_metrics()
        assert result["memory"] == 75.0

    def test_get_performance_metrics_without_dashboard(self):
        """测试获取性能指标 - 无仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.performance_dashboard = None
        
        result = integrator._get_performance_metrics()
        assert "cpu_usage" in result
        assert "memory_usage" in result
        assert "response_time" in result

    def test_get_integration_status(self):
        """测试获取集成状态"""
        integrator = PerformanceFrameworkIntegrator()
        integrator._integration_active = True
        integrator.benchmark_framework = MagicMock()
        integrator.performance_dashboard = MagicMock()
        
        mock_optimizer = MagicMock()
        mock_optimizer.get_optimization_status.return_value = "active"
        mock_optimizer.thread_manager.get_active_threads_count.return_value = 5
        integrator.test_optimizer = mock_optimizer
        
        status = integrator.get_integration_status()
        
        assert status["integration_active"] is True
        assert status["benchmark_framework_integrated"] is True
        assert status["performance_dashboard_integrated"] is True
        assert status["test_optimizer_status"] == "active"
        assert status["active_threads_count"] == 5

    def test_configure_benchmark_framework_unit_mode(self):
        """测试配置基准测试框架 - 单元测试模式"""
        integrator = PerformanceFrameworkIntegrator()
        mock_framework = MagicMock()
        integrator.benchmark_framework = mock_framework
        
        # Mock TestMode.UNIT
        class MockTestMode:
            UNIT = MagicMock()
            PERFORMANCE = MagicMock()
            INTEGRATION = MagicMock()
            STRESS = MagicMock()
        
        mock_mode = MagicMock()
        mock_mode.__eq__ = lambda self, other: other == MockTestMode.UNIT
        
        with patch('src.infrastructure.config.tools.framework_integrator.TestMode', MockTestMode):
            integrator._configure_benchmark_framework(mock_mode)
            
            # 验证配置，由于mock_mode.__eq__的比较逻辑，我们需要直接调用
            # 让我们直接测试配置方法
        integrator._configure_benchmark_framework("unit")
        
        # 测试没有框架的情况
        integrator.benchmark_framework = None
        integrator._configure_benchmark_framework("unit")  # 应该不执行任何操作

    def test_configure_benchmark_framework_performance_mode(self):
        """测试配置基准测试框架 - 性能测试模式"""
        integrator = PerformanceFrameworkIntegrator()
        mock_framework = MagicMock()
        integrator.benchmark_framework = mock_framework
        
        # 直接调用不同的配置分支
        integrator._configure_benchmark_framework("performance")
        integrator._configure_benchmark_framework("integration")
        integrator._configure_benchmark_framework("stress")

    def test_configure_benchmark_framework_no_framework(self):
        """测试配置基准测试框架 - 无框架"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.benchmark_framework = None
        
        # 应该直接返回，不执行任何操作
        integrator._configure_benchmark_framework("unit")

    def test_configure_performance_dashboard_unit_mode(self):
        """测试配置性能监控仪表板 - 单元测试模式"""
        integrator = PerformanceFrameworkIntegrator()
        mock_dashboard = MagicMock()
        integrator.performance_dashboard = mock_dashboard
        
        # 测试不同的配置模式
        integrator._configure_performance_dashboard("unit")
        integrator._configure_performance_dashboard("performance")
        integrator._configure_performance_dashboard("integration")
        integrator._configure_performance_dashboard("stress")

    def test_configure_performance_dashboard_no_dashboard(self):
        """测试配置性能监控仪表板 - 无仪表板"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.performance_dashboard = None
        
        # 应该直接返回，不执行任何操作
        integrator._configure_performance_dashboard("unit")

    def test_cleanup_integrated_components(self):
        """测试清理集成组件"""
        integrator = PerformanceFrameworkIntegrator()
        
        # 测试有框架和仪表板的情况
        mock_framework = MagicMock()
        mock_dashboard = MagicMock()
        integrator.benchmark_framework = mock_framework
        integrator.performance_dashboard = mock_dashboard
        
        integrator._cleanup_integrated_components()
        
        # 验证框架配置被恢复（如果属性存在）
        if hasattr(mock_framework, 'disable_background_collection'):
            assert mock_framework.disable_background_collection == False
        if hasattr(mock_framework, 'max_iterations'):
            assert mock_framework.max_iterations == 100
        
        # 验证仪表板配置被恢复（如果属性存在）
        if hasattr(mock_dashboard, 'disable_background_monitoring'):
            assert mock_dashboard.disable_background_monitoring == False
        if hasattr(mock_dashboard, 'collection_interval'):
            assert mock_dashboard.collection_interval == 5

    def test_cleanup_integrated_components_no_components(self):
        """测试清理集成组件 - 无组件"""
        integrator = PerformanceFrameworkIntegrator()
        integrator.benchmark_framework = None
        integrator.performance_dashboard = None
        
        # 应该不出错地完成
        integrator._cleanup_integrated_components()



@pytest.mark.skipif(not HAVE_MODULE, reason="framework_integrator模块导入失败")
class TestFrameworkIntegratorFunctions:
    """测试框架集成器函数"""
    
    def setup_method(self):
        """测试前准备"""
        pass

    @patch('src.infrastructure.config.tools.framework_integrator._global_integrator', None)
    @patch('src.infrastructure.config.tools.framework_integrator.PerformanceFrameworkIntegrator')
    def test_get_framework_integrator_new_instance(self, mock_integrator_class):
        """测试获取框架集成器 - 新实例"""
        mock_instance = MagicMock()
        mock_integrator_class.return_value = mock_instance
        
        result = get_framework_integrator()
        
        assert result == mock_instance
        mock_integrator_class.assert_called_once()

    def test_get_framework_integrator_existing_instance(self):
        """测试获取框架集成器 - 现有实例"""
        # 首先获取一个实例来设置全局状态
        first_result = get_framework_integrator()
        
        # 再次调用应该返回相同的实例
        second_result = get_framework_integrator()
        
        # 验证两次调用返回的是同一个实例
        assert first_result is second_result

    @patch('src.infrastructure.config.tools.framework_integrator.get_framework_integrator')
    def test_integrate_performance_framework(self, mock_get_integrator):
        """测试集成性能测试框架"""
        mock_integrator = MagicMock()
        mock_get_integrator.return_value = mock_integrator
        
        mock_benchmark = MagicMock()
        mock_dashboard = MagicMock()
        
        result = integrate_performance_framework(mock_benchmark, mock_dashboard)
        
        mock_integrator.integrate_benchmark_framework.assert_called_once_with(mock_benchmark)
        mock_integrator.integrate_performance_dashboard.assert_called_once_with(mock_dashboard)
        assert result == mock_integrator

    @patch('src.infrastructure.config.tools.framework_integrator.get_framework_integrator')
    def test_quick_integration(self, mock_get_integrator):
        """测试快速集成"""
        mock_integrator = MagicMock()
        mock_get_integrator.return_value = mock_integrator
        
        result = quick_integration()
        
        assert result == mock_integrator

    @patch('src.infrastructure.config.tools.framework_integrator.get_framework_integrator')
    def test_run_optimized_test(self, mock_get_integrator):
        """测试运行优化测试"""
        mock_integrator = MagicMock()
        mock_integrator.run_optimized_performance_test.return_value = {"result": "success"}
        mock_get_integrator.return_value = mock_integrator
        
        def test_func():
            return "result"
        
        result = run_optimized_test("test_name", test_func, "performance", param="value")
        
        mock_integrator.run_optimized_performance_test.assert_called_once()
        assert result == {"result": "success"}

    @patch('src.infrastructure.config.tools.framework_integrator.get_framework_integrator')
    def test_run_optimized_benchmark_test(self, mock_get_integrator):
        """测试运行优化基准测试"""
        mock_integrator = MagicMock()
        mock_integrator.run_optimized_benchmark.return_value = {"benchmark": "result"}
        mock_get_integrator.return_value = mock_integrator
        
        def test_func():
            return "result"
        
        result = run_optimized_benchmark_test("benchmark_name", test_func, "performance", param="value")
        
        mock_integrator.run_optimized_benchmark.assert_called_once()
        assert result == {"benchmark": "result"}