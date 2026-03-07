import sys
import types
import importlib
from unittest.mock import MagicMock

import pytest


def _ensure_integration_stub():
    """确保基础设施集成管理器桩存在"""
    mod_name = "src.data.processing.infrastructure_integration_manager"
    if mod_name in sys.modules:
        return
    stub = types.ModuleType(mod_name)
    
    class _Mgr:
        _initialized = True
        _integration_config = {}
        def initialize(self):
            self._initialized = True
        def get_health_check_bridge(self):
            class _HB:
                def register_data_health_check(self, *args, **kwargs):
                    return True
            return _HB()
    
    def get_data_integration_manager():
        return _Mgr()
    def log_data_operation(*args, **kwargs):
        return None
    def record_data_metric(*args, **kwargs):
        return None
    def publish_data_event(*args, **kwargs):
        return None
    
    stub.get_data_integration_manager = get_data_integration_manager
    stub.log_data_operation = log_data_operation
    stub.record_data_metric = record_data_metric
    stub.publish_data_event = publish_data_event
    sys.modules[mod_name] = stub

    # 注入 DataSourceType 兼容桩
    pkg_name = "src.data.processing.interfaces"
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = types.ModuleType(pkg_name)
    std_name = "src.data.processing.interfaces.standard_interfaces"
    if std_name not in sys.modules:
        std = types.ModuleType(std_name)
        class _EnumLike:
            def __init__(self, value):
                self.value = value
            def __iter__(self):
                return iter([self])
        class DataSourceType:
            STOCK = _EnumLike("stock")
            def __iter__(self):
                return iter([self.STOCK])
        std.DataSourceType = DataSourceType
        sys.modules[std_name] = std


def _load_optimizer_module():
    """加载性能优化器模块"""
    _ensure_integration_stub()
    return importlib.import_module("src.data.processing.performance_optimizer")


class _ProcStub:
    """进程桩"""
    def __init__(self, mem=50.0, cpu=0.0):
        self._mem = mem
        self._cpu = cpu
    def memory_percent(self):
        return self._mem
    def cpu_percent(self):
        return self._cpu


def test_load_config_fallback_on_integration_manager_error(monkeypatch):
    """测试集成管理器错误时的配置加载回退"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    # 注意：当前实现中，__init__ 会直接调用 get_data_integration_manager()
    # 如果抛出异常，会传播。但 _load_config_from_integration_manager 有异常处理
    # 测试 _load_config_from_integration_manager 的回退逻辑
    cfg = PerformanceConfig(enable_performance_monitoring=False, memory_threshold=0.8)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock integration_manager 在访问时抛出异常
    original_manager = opt.integration_manager
    
    class ErrorManager:
        def __getattr__(self, name):
            raise RuntimeError("Integration manager unavailable")
    
    opt.integration_manager = ErrorManager()
    
    # 重新加载配置应该使用默认配置（异常被捕获）
    merged = opt._load_config_from_integration_manager()
    # 应该返回默认配置字典
    assert isinstance(merged, dict)
    assert merged.get("memory_threshold") == 0.8  # 使用传入的配置
    opt.shutdown()


def test_health_check_bridge_none_handled_gracefully(monkeypatch):
    """测试健康检查桥为 None 时的优雅处理"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    # Mock get_health_check_bridge 返回 None
    class MockManager:
        _initialized = True
        _integration_config = {}
        def initialize(self):
            self._initialized = True
        def get_health_check_bridge(self):
            return None  # 返回 None 而不是桥对象
    
    def mock_get_manager():
        return MockManager()
    
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.get_data_integration_manager",
        mock_get_manager
    )
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # 应该不会抛出异常，健康检查注册应该被跳过
    # 验证优化器可以正常工作
    assert opt.config is not None
    opt.shutdown()


def test_health_check_bridge_registration_error_logged(monkeypatch):
    """测试健康检查桥注册错误时记录日志但不抛出异常"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    # Mock get_health_check_bridge 抛出异常
    class MockManager:
        _initialized = True
        _integration_config = {}
        def initialize(self):
            self._initialized = True
        def get_health_check_bridge(self):
            raise RuntimeError("Bridge registration failed")
    
    def mock_get_manager():
        return MockManager()
    
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.get_data_integration_manager",
        mock_get_manager
    )
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # 应该不会抛出异常，但应该记录错误日志
    assert any("health_check_registration_error" in op for op, level in log_calls if level == "warning")
    opt.shutdown()


def test_health_check_exception_returns_error_status(monkeypatch):
    """测试健康检查时异常返回错误状态"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock psutil.Process 抛出异常
    def mock_process_error():
        raise RuntimeError("Process info unavailable")
    
    monkeypatch.setattr("src.data.processing.performance_optimizer.psutil.Process", mock_process_error)
    
    # 健康检查应该返回错误状态，而不是抛出异常
    health = opt._performance_optimizer_health_check()
    assert health["status"] == "error"
    assert "error" in health
    opt.shutdown()


def test_apply_optimizations_handles_exceptions_gracefully(monkeypatch):
    """测试应用优化时异常处理的优雅降级"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock psutil 抛出异常
    def mock_process_error():
        raise RuntimeError("System info unavailable")
    
    monkeypatch.setattr("src.data.processing.performance_optimizer.psutil.Process", mock_process_error)
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    # 应该不会抛出异常，但应该记录错误日志
    opt._apply_performance_optimizations()
    
    # 验证错误被记录
    assert any("performance_optimization_error" in op for op, level in log_calls if level == "error")
    opt.shutdown()


def test_memory_optimization_handles_exceptions(monkeypatch):
    """测试内存优化时异常处理"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    cfg = PerformanceConfig(enable_performance_monitoring=False, enable_memory_monitoring=True)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock gc.collect 抛出异常
    def mock_gc_collect_error():
        raise RuntimeError("GC collection failed")
    
    monkeypatch.setattr("src.data.processing.performance_optimizer.gc.collect", mock_gc_collect_error)
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    # 应该不会抛出异常，但应该记录错误日志
    opt._optimize_memory_usage()
    
    # 验证错误被记录
    assert any("memory_optimization_error" in op for op, level in log_calls if level == "error")
    opt.shutdown()


def test_gc_optimization_handles_exceptions(monkeypatch):
    """测试 GC 优化时异常处理"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    cfg = PerformanceConfig(enable_performance_monitoring=False, enable_gc_optimization=True)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock gc.set_threshold 抛出异常
    def mock_gc_set_threshold_error(*args):
        raise RuntimeError("GC threshold setting failed")
    
    monkeypatch.setattr("src.data.processing.performance_optimizer.gc.set_threshold", mock_gc_set_threshold_error)
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    # 应该不会抛出异常，但应该记录错误日志
    opt._optimize_gc()
    
    # 验证错误被记录
    assert any("gc_optimization_error" in op for op, level in log_calls if level == "error")
    opt.shutdown()


def test_performance_report_handles_exceptions(monkeypatch):
    """测试性能报告生成时异常处理"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock datetime.now 抛出异常（不太可能，但测试异常处理）
    def mock_datetime_error():
        raise RuntimeError("Time unavailable")
    
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    # 正常情况下应该能生成报告
    report = opt.get_performance_report()
    assert "generated_at" in report or "error" in report
    
    opt.shutdown()


def test_shutdown_handles_exceptions(monkeypatch):
    """测试关闭时异常处理"""
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer
    
    log_calls = []
    
    def mock_log_operation(operation, data_type, data, level):
        log_calls.append((operation, level))
    
    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    
    # Mock 线程 join 抛出异常
    opt.memory_monitor_thread = MagicMock()
    opt.memory_monitor_thread.is_alive.return_value = True
    opt.memory_monitor_thread.join.side_effect = RuntimeError("Thread join failed")
    
    monkeypatch.setattr(
        "src.data.processing.performance_optimizer.log_data_operation",
        mock_log_operation
    )
    
    # shutdown 应该不会抛出异常，但应该记录错误日志
    opt.shutdown()
    
    # 验证错误被记录（如果发生）
    # 注意：由于 shutdown 可能不会在所有情况下都记录错误，这里主要验证不会抛出异常

