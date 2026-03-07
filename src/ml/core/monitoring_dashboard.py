#!/usr/bin/env python3
"""简化版监控面板，实现测试所需的核心功能。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .performance_monitor import MLPerformanceMonitor, get_ml_performance_monitor
from .process_orchestrator import get_ml_process_orchestrator


class MLMonitoringDashboard:
    """收集监控数据并提供导出接口。"""

    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.performance_monitor: MLPerformanceMonitor = get_ml_performance_monitor()
        self.process_orchestrator = get_ml_process_orchestrator()

        self.running = False
        self.current_stats: Dict[str, Any] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.process_history: Dict[str, List[Dict[str, Any]]] = {"active_count": []}
        self.display_config: Dict[str, Any] = {
            "show_inference_metrics": True,
            "show_resource_metrics": True,
            "show_process_metrics": True,
            "show_alerts": True,
            "max_alert_history": 100,
            "max_process_history": 50,
        }

    def start_dashboard(self):
        if self.running:
            return
        self.running = True
        self.performance_monitor.add_alert_callback(self._handle_alert)

    def stop_dashboard(self):
        if not self.running:
            return
        self.running = False

    def _update_stats(self):
        self.current_stats = self.performance_monitor.get_current_stats()

    def _update_process_status(self):
        orchestrator_stats = self.process_orchestrator.get_statistics()
        active = orchestrator_stats.get("active_processes", 0)
        self.process_history["active_count"].append(
            {"timestamp": datetime.now(), "count": active}
        )
        if len(self.process_history["active_count"]) > self.display_config["max_process_history"]:
            self.process_history["active_count"] = self.process_history["active_count"][
                -self.display_config["max_process_history"] :
            ]

    def _handle_alert(self, alert: Dict[str, Any]):
        self.alert_history.append({"timestamp": datetime.now().isoformat(), "alert": alert})
        if len(self.alert_history) > self.display_config["max_alert_history"]:
            self.alert_history = self.alert_history[-self.display_config["max_alert_history"] :]

    # ------------------------------------------------------------------ #
    # 公有接口
    # ------------------------------------------------------------------ #
    def get_dashboard_data(self) -> Dict[str, Any]:
        self._update_stats()
        self._update_process_status()
        return {
            "timestamp": datetime.now().isoformat(),
            "current_stats": self.current_stats,
            "alert_history": self.alert_history[-10:],
            "process_history": {
                key: [{"timestamp": item["timestamp"].isoformat(), "count": item["count"]} for item in values]
                for key, values in self.process_history.items()
            },
            "display_config": self.display_config.copy(),
        }

    def configure_display(self, config: Dict[str, bool]):
        for key, value in config.items():
            if key in self.display_config:
                self.display_config[key] = value

    def export_dashboard_data(self, format: str = "json") -> str:
        data = self.get_dashboard_data()
        if format.lower() != "json":
            raise ValueError("不支持的导出格式: {}".format(format))
        return json.dumps(data, indent=2)

    def get_health_score(self) -> float:
        stats = self.get_dashboard_data()
        inference = stats["current_stats"].get("inference", {})
        resources = stats["current_stats"].get("resources", {})

        score = 100.0
        error_rate = inference.get("error_rate", 0.0)
        score -= min(error_rate * 100, 30)

        latency = inference.get("avg_latency_ms", 0.0)
        if latency > 2000:
            score -= 20

        cpu_max = resources.get("cpu_max_percent", 0.0)
        if cpu_max > 90:
            score -= 15

        memory_max = resources.get("memory_max_percent", 0.0)
        if memory_max > 90:
            score -= 15

        recent_alerts = []
        for alert in self.alert_history:
            ts = alert["timestamp"]
            if isinstance(ts, str):
                ts_value = datetime.fromisoformat(ts)
            else:
                ts_value = ts
            if (datetime.now() - ts_value).total_seconds() < 300:
                recent_alerts.append(alert)
        score -= min(len(recent_alerts) * 5, 30)
        return max(0.0, min(100.0, score))


_GLOBAL_DASHBOARD = MLMonitoringDashboard()


def get_ml_monitoring_dashboard() -> MLMonitoringDashboard:
    return _GLOBAL_DASHBOARD


def start_ml_dashboard():
    _GLOBAL_DASHBOARD.start_dashboard()


def stop_ml_dashboard():
    _GLOBAL_DASHBOARD.stop_dashboard()


def get_ml_dashboard_data() -> Dict[str, Any]:
    return _GLOBAL_DASHBOARD.get_dashboard_data()


def get_ml_health_score() -> float:
    return _GLOBAL_DASHBOARD.get_health_score()


__all__ = [
    "MLMonitoringDashboard",
    "get_ml_monitoring_dashboard",
    "start_ml_dashboard",
    "stop_ml_dashboard",
    "get_ml_dashboard_data",
    "get_ml_health_score",
]

