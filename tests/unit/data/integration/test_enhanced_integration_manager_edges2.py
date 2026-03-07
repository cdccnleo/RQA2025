import time
import importlib
import sys
import types
import pytest


def _ensure_pkg(name: str):
    """确保包存在"""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        setattr(mod, "__path__", [])
        sys.modules[name] = mod
    else:
        mod = sys.modules[name]
        if not hasattr(mod, "__path__"):
            setattr(mod, "__path__", [])
    return sys.modules[name]


def _import_with_stubs():
    """导入模块并注入必要的 stubs"""
    try:
        return importlib.import_module("src.data.integration.enhanced_integration_manager")
    except ModuleNotFoundError as exc:
        msg = str(exc)
        if "quality.monitor" in msg or "integration.quality" in msg:
            _ensure_pkg("src")
            _ensure_pkg("src.data")
            _ensure_pkg("src.data.integration")
            _ensure_pkg("src.data.integration.quality")
            
            pkg_path = "src.data.quality.monitor"
            if pkg_path not in sys.modules:
                pkg = types.ModuleType(pkg_path)
                class DataQualityMonitor:
                    def generate_report(self, *args, **kwargs):
                        return {"ok": True}
                setattr(pkg, "DataQualityMonitor", DataQualityMonitor)
                sys.modules[pkg_path] = pkg
        
        if "cache.cache_manager" in msg or "integration.cache" in msg:
            _ensure_pkg("src")
            _ensure_pkg("src.data")
            _ensure_pkg("src.data.integration")
            _ensure_pkg("src.data.integration.cache")
            cache_mod_path = "src.data.cache.cache_manager"
            if cache_mod_path not in sys.modules:
                cache_mod = types.ModuleType(cache_mod_path)
                class CacheConfig:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                class CacheManager:
                    def __init__(self, *args, **kwargs):
                        self._stats = {"cache": {"size": 0, "hit_rate": 0.0, "total_entries": 0}}
                    def get_stats(self):
                        return self._stats
                    def close(self): ...
                setattr(cache_mod, "CacheConfig", CacheConfig)
                setattr(cache_mod, "CacheManager", CacheManager)
                sys.modules[cache_mod_path] = cache_mod
        
        if "data_manager" in msg:
            _ensure_pkg("src")
            _ensure_pkg("src.data")
            _ensure_pkg("src.data.integration")
            dm_mod_path = "src.data.core.data_manager"
            if dm_mod_path not in sys.modules:
                dm_mod = types.ModuleType(dm_mod_path)
                class DataManagerSingleton:
                    _inst = None
                    @classmethod
                    def get_instance(cls, *args, **kwargs):
                        if cls._inst is None:
                            cls._inst = object()
                        return cls._inst
                setattr(dm_mod, "DataManagerSingleton", DataManagerSingleton)
                sys.modules[dm_mod_path] = dm_mod
        
        try:
            return importlib.import_module("src.data.integration.enhanced_integration_manager")
        except ModuleNotFoundError:
            pytest.skip(f"enhanced_integration_manager import skipped: {msg}")


def test_performance_monitor_get_metric_stats_empty():
    """测试获取空指标的统计"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    # 获取不存在的指标
    stats = pm.get_metric_stats("nonexistent_metric")
    assert stats == {}
    
    # 获取空列表的指标
    pm.metrics["empty_metric"] = []
    stats = pm.get_metric_stats("empty_metric")
    assert stats == {}


def test_performance_monitor_get_metric_stats_single_value():
    """测试获取单值指标的统计"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("single_metric", 42.0)
    
    stats = pm.get_metric_stats("single_metric")
    assert stats["count"] == 1
    assert stats["min"] == 42.0
    assert stats["max"] == 42.0
    assert stats["avg"] == 42.0
    assert stats["latest"] == 42.0


def test_performance_monitor_get_metric_stats_multiple_values():
    """测试获取多值指标的统计"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for v in values:
        pm.record_metric("multi_metric", v)
    
    stats = pm.get_metric_stats("multi_metric")
    assert stats["count"] == 5
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["avg"] == 30.0
    assert stats["latest"] == 50.0


def test_performance_monitor_metrics_truncation():
    """测试指标截断（保留最近1000个）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    # 记录超过1000个值
    for i in range(1500):
        pm.record_metric("truncated_metric", float(i))
    
    # 应该只保留最近1000个
    assert len(pm.metrics["truncated_metric"]) == 1000
    assert pm.metrics["truncated_metric"][0] == 500.0  # 第一个应该是第500个值
    assert pm.metrics["truncated_metric"][-1] == 1499.0  # 最后一个应该是第1499个值


def test_performance_monitor_get_all_metrics_empty():
    """测试获取所有指标（空指标）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    all_metrics = pm.get_all_metrics()
    assert all_metrics == {}


def test_performance_monitor_get_all_metrics_multiple():
    """测试获取所有指标（多个指标）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("metric1", 10.0)
    pm.record_metric("metric2", 20.0)
    pm.record_metric("metric3", 30.0)
    
    all_metrics = pm.get_all_metrics()
    assert len(all_metrics) == 3
    assert "metric1" in all_metrics
    assert "metric2" in all_metrics
    assert "metric3" in all_metrics


def test_performance_monitor_clear_metrics():
    """测试清空所有指标"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("metric1", 10.0)
    pm.record_metric("metric2", 20.0)
    
    assert len(pm.metrics) == 2
    
    pm.clear_metrics()
    
    assert len(pm.metrics) == 0


def test_performance_monitor_get_cache_efficiency_no_data():
    """测试获取缓存效率（无数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    efficiency = pm.get_cache_efficiency()
    assert efficiency == 0.0


def test_performance_monitor_get_cache_efficiency_with_data():
    """测试获取缓存效率（有数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_cache_hit_rate(0.85)
    pm.record_cache_hit_rate(0.90)
    pm.record_cache_hit_rate(0.95)
    
    efficiency = pm.get_cache_efficiency()
    assert efficiency == 0.9  # 平均值


def test_performance_monitor_get_average_load_time_no_data():
    """测试获取平均加载时间（无数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    avg_time = pm.get_average_load_time()
    assert avg_time == 0.0


def test_performance_monitor_get_average_load_time_with_data():
    """测试获取平均加载时间（有数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_data_load_time(1.0)
    pm.record_data_load_time(2.0)
    pm.record_data_load_time(3.0)
    
    avg_time = pm.get_average_load_time()
    assert avg_time == 2.0  # 平均值


def test_performance_monitor_get_current_metric_no_data():
    """测试获取当前指标值（无数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    metric_value = pm.get_current_metric("nonexistent_metric")
    assert metric_value.value == 0.0


def test_performance_monitor_get_current_metric_with_data():
    """测试获取当前指标值（有数据）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("test_metric", 42.0)
    pm.record_metric("test_metric", 43.0)
    
    metric_value = pm.get_current_metric("test_metric")
    assert metric_value.value == 43.0  # 最新值


def test_performance_monitor_export_metrics_json():
    """测试导出指标（JSON格式）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("metric1", 10.0)
    pm.record_metric("metric2", 20.0)
    
    exported = pm.export_metrics("json")
    assert isinstance(exported, str)
    assert "json" in exported.lower()
    assert "metric1" in exported
    assert "metric2" in exported


def test_performance_monitor_export_metrics_other_format():
    """测试导出指标（其他格式）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("metric1", 10.0)
    
    exported = pm.export_metrics("csv")
    assert isinstance(exported, str)
    assert "csv" in exported.lower()


def test_performance_monitor_get_recent_alerts_low_rate():
    """测试获取最近告警（低命中率）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_cache_hit_rate(0.5)  # 低于0.6
    
    alerts = pm.get_recent_alerts()
    assert len(alerts) > 0
    assert any("too low" in alert.message.lower() for alert in alerts)


def test_performance_monitor_get_recent_alerts_warning_rate():
    """测试获取最近告警（警告命中率）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_cache_hit_rate(0.75)  # 低于0.8但高于0.6
    
    alerts = pm.get_recent_alerts()
    assert len(alerts) > 0
    assert any("warning" in alert.message.lower() for alert in alerts)


def test_performance_monitor_get_recent_alerts_normal_rate():
    """测试获取最近告警（正常命中率）"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_cache_hit_rate(0.9)  # 高于0.8
    
    alerts = pm.get_recent_alerts()
    assert len(alerts) > 0
    assert any("normal" in alert.message.lower() for alert in alerts)


def test_alert_manager_trigger_alert_not_found():
    """测试触发不存在的告警配置"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    # 触发不存在的告警配置应该不抛出异常
    am.trigger_alert("nonexistent_alert", {"value": 100})
    
    # 验证告警历史为空
    assert len(am.alert_history) == 0


def test_alert_manager_trigger_alert_below_threshold():
    """测试触发告警（低于阈值）"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    config = mod.AlertConfig(
        level="warning",
        threshold=50.0,
        channels=["email"],
        message_template="Value: {value}"
    )
    am.add_alert_config("test_alert", config)
    
    # 触发低于阈值的告警应该不发送
    am.trigger_alert("test_alert", {"value": 30.0})
    
    assert len(am.alert_history) == 0


def test_alert_manager_trigger_alert_cooldown():
    """测试触发告警（冷却时间内）"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    config = mod.AlertConfig(
        level="warning",
        threshold=50.0,
        channels=["email"],
        message_template="Value: {value}",
        cooldown=300  # 5分钟
    )
    am.add_alert_config("test_alert", config)
    
    # 第一次触发
    am.trigger_alert("test_alert", {"value": 100.0})
    assert len(am.alert_history) == 1
    
    # 立即再次触发（在冷却时间内）
    am.trigger_alert("test_alert", {"value": 100.0})
    assert len(am.alert_history) == 1  # 应该不增加


def test_alert_manager_trigger_alert_after_cooldown():
    """测试触发告警（冷却时间后）"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    config = mod.AlertConfig(
        level="warning",
        threshold=50.0,
        channels=["email"],
        message_template="Value: {value}",
        cooldown=0  # 无冷却时间
    )
    am.add_alert_config("test_alert", config)
    
    # 第一次触发
    am.trigger_alert("test_alert", {"value": 100.0})
    assert len(am.alert_history) == 1
    
    # 立即再次触发（无冷却时间）
    am.trigger_alert("test_alert", {"value": 100.0})
    assert len(am.alert_history) == 2  # 应该增加


def test_alert_manager_send_alert_channel_error():
    """测试发送告警（通道错误）"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    # Mock _send_email_alert 抛出异常
    original_send = am._send_email_alert
    def _bad_send(message, level):
        raise RuntimeError("email send failed")
    
    am._send_email_alert = _bad_send
    
    config = mod.AlertConfig(
        level="warning",
        threshold=50.0,
        channels=["email"],
        message_template="Value: {value}",
        cooldown=0
    )
    am.add_alert_config("test_alert", config)
    
    # 触发告警（应该捕获异常，不抛出）
    am.trigger_alert("test_alert", {"value": 100.0})
    
    # 验证告警历史已记录（即使发送失败）
    assert len(am.alert_history) == 1


def test_alert_manager_clear_history():
    """测试清空告警历史"""
    mod = _import_with_stubs()
    am = mod.AlertManager()
    
    config = mod.AlertConfig(
        level="warning",
        threshold=50.0,
        channels=["email"],
        message_template="Value: {value}",
        cooldown=0
    )
    am.add_alert_config("test_alert", config)
    
    # 触发多个告警
    am.trigger_alert("test_alert", {"value": 100.0})
    am.trigger_alert("test_alert", {"value": 100.0})
    
    assert len(am.alert_history) == 2
    assert len(am._last_alert_time) > 0
    
    # 清空历史
    am.clear_history()
    
    assert len(am.alert_history) == 0
    assert len(am._last_alert_time) == 0


def test_performance_monitor_start_stop_monitoring():
    """测试开始和停止监控"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    # 开始监控
    result = pm.start_monitoring()
    assert result is True
    assert pm.start_time > 0
    
    # 停止监控
    result = pm.stop_monitoring()
    assert result is True


def test_performance_monitor_get_performance_report():
    """测试获取性能报告"""
    mod = _import_with_stubs()
    pm = mod.PerformanceMonitor()
    
    pm.record_metric("metric1", 10.0)
    pm.record_cache_hit_rate(0.85)
    pm.record_data_load_time(1.5)
    
    report = pm.get_performance_report()
    
    assert "metrics" in report
    assert "metrics_summary" in report
    assert "monitoring_status" in report
    assert report["monitoring_status"] == "active"
    assert "total_metrics" in report["metrics_summary"]
    assert "uptime" in report["metrics_summary"]
    assert "cache_efficiency" in report["metrics_summary"]
    assert "average_load_time" in report["metrics_summary"]

