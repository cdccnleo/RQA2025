import sys
import types
import importlib
import pytest


def _ensure_integration_stub():
    # 注入 src.data.processing.infrastructure_integration_manager 兼容桩
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

    # 注入 src.data.processing.interfaces.standard_interfaces 兼容桩
    pkg_name = "src.data.processing.interfaces"
    if pkg_name not in sys.modules:
        sys.modules[pkg_name] = types.ModuleType(pkg_name)
    std_name = "src.data.processing.interfaces.standard_interfaces"
    if std_name not in sys.modules:
        std = types.ModuleType(std_name)
        # 最低限度枚举桩
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
    _ensure_integration_stub()
    return importlib.import_module("src.data.processing.performance_optimizer")


class _ProcStub:
    def __init__(self, mem=50.0, cpu=0.0):
        self._mem = mem
        self._cpu = cpu
    def memory_percent(self):
        return self._mem
    def cpu_percent(self):
        return self._cpu


def test_apply_optimizations_triggers_on_memory_threshold(monkeypatch):
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer

    cfg = PerformanceConfig(enable_performance_monitoring=False, memory_threshold=0.01)
    opt = DataPerformanceOptimizer(cfg)

    monkeypatch.setattr("src.data.processing.performance_optimizer.psutil.Process", lambda: _ProcStub(mem=99.0))
    before = dict(opt.stats)
    opt._apply_performance_optimizations()
    assert opt.stats["optimizations_applied"] == before["optimizations_applied"] + 1
    opt.shutdown()


def test_get_performance_report_empty_history(monkeypatch):
    perf_mod = _load_optimizer_module()
    PerformanceConfig = perf_mod.PerformanceConfig
    DataPerformanceOptimizer = perf_mod.DataPerformanceOptimizer

    cfg = PerformanceConfig(enable_performance_monitoring=False)
    opt = DataPerformanceOptimizer(cfg)
    monkeypatch.setattr("src.data.processing.performance_optimizer.psutil.Process", lambda: _ProcStub(mem=12.0))
    monkeypatch.setattr("src.data.processing.performance_optimizer.psutil.cpu_percent", lambda: 34.0)
    report = opt.get_performance_report()
    assert "generated_at" in report
    assert "stats" in report
    assert isinstance(report["stats"], dict)
    opt.shutdown()


