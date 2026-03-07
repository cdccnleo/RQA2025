#!/usr/bin/env python3
"""
RQA2025 基础设施层监控协调器

负责监控系统的启动、停止和协调各个组件的工作。
"""

import threading
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..core.parameter_objects import MonitoringConfig
from .performance_monitor import global_performance_monitor, monitor_performance
from ..core.component_bus import global_component_bus, publish_event, Message, MessageType


class MonitoringCoordinator:
    """
    监控协调器

    负责协调监控系统的各个组件，提供统一的监控生命周期管理。
    """

    def __init__(
        self,
        pool_name: str = "default_pool",
        config: Optional[MonitoringConfig] = None,
    ):
        """
        初始化监控协调器

        Args:
            pool_name: 监控对象名称
            config: 监控配置
        """
        self.pool_name = pool_name
        self.config = config or MonitoringConfig()

        # 监控状态
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time: Optional[datetime] = None

        # 组件引用
        self.stats_collector = None
        self.alert_manager = None
        self.metrics_exporter = None

        # 线程同步
        self._lock = threading.RLock()

        # 事件订阅
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self):
        """设置事件订阅"""
        # 订阅监控控制事件
        global_component_bus.subscribe(
            "MonitoringCoordinator",
            "monitoring.control.*",
            self._handle_control_event
        )

        # 订阅组件状态事件
        global_component_bus.subscribe(
            "MonitoringCoordinator",
            "component.status.*",
            self._handle_component_status_event
        )

    def _handle_control_event(self, message: Message):
        """处理控制事件"""
        if message.topic == "monitoring.control.start":
            self.start_monitoring()
        elif message.topic == "monitoring.control.stop":
            self.stop_monitoring()
        elif message.topic == "monitoring.control.restart":
            self.stop_monitoring()
            time.sleep(1)  # 短暂等待
            self.start_monitoring()

    def _handle_component_status_event(self, message: Message):
        """处理组件状态事件"""
        component_name = message.payload.get('component')
        status = message.payload.get('status')
        error = message.payload.get('error')

        if status == 'error' and error:
            print(f"组件 {component_name} 报告错误: {error}")
            # 可以在这里实现错误恢复逻辑

    def set_components(self, stats_collector, alert_manager, metrics_exporter):
        """
        设置监控组件

        Args:
            stats_collector: 统计收集器
            alert_manager: 告警管理器
            metrics_exporter: 指标导出器
        """
        self.stats_collector = stats_collector
        self.alert_manager = alert_manager
        self.metrics_exporter = metrics_exporter

    @monitor_performance("MonitoringCoordinator", "start_monitoring")
    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            bool: 是否成功启动
        """
        with self._lock:
            if self.monitoring_active:
                return True

            try:
                self.monitoring_active = True
                self.start_time = datetime.now()

                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name=f"MonitoringCoordinator-{self.pool_name}",
                    daemon=True
                )
                self.monitoring_thread.start()

                print(f"✅ 监控协调器已启动: {self.pool_name}")
                return True

            except Exception as e:
                self.monitoring_active = False
                print(f"❌ 启动监控协调器失败: {e}")
                return False

    @monitor_performance("MonitoringCoordinator", "stop_monitoring")
    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            bool: 是否成功停止
        """
        with self._lock:
            if not self.monitoring_active:
                return True

            try:
                self.monitoring_active = False

                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=5.0)

                print(f"✅ 监控协调器已停止: {self.pool_name}")
                return True

            except Exception as e:
                print(f"❌ 停止监控协调器失败: {e}")
                return False

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态

        Returns:
            Dict[str, Any]: 监控状态信息
        """
        with self._lock:
            return {
                'active': self.monitoring_active,
                'pool_name': self.pool_name,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'collection_interval': self.config.collection_interval,
            }

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 执行一次监控周期
                self._execute_monitoring_cycle()

                # 等待下一个周期
                time.sleep(self.config.collection_interval)

            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(self.config.collection_interval)

    @monitor_performance("MonitoringCoordinator", "execute_monitoring_cycle")
    def _execute_monitoring_cycle(self):
        """执行监控周期"""
        try:
            self._publish_cycle_started_event()

            # 执行监控步骤
            monitoring_result = self._perform_monitoring_steps()

            self._publish_cycle_completed_event(monitoring_result)

        except Exception as e:
            self._publish_cycle_error_event(e)
            print(f"执行监控周期失败: {e}")

    def _publish_cycle_started_event(self):
        """发布监控周期开始事件"""
        publish_event(
            "monitoring.cycle.started",
            {"pool_name": self.pool_name, "timestamp": datetime.now().isoformat()},
            "MonitoringCoordinator"
        )

    def _perform_monitoring_steps(self) -> Dict[str, Any]:
        """
        执行监控步骤

        Returns:
            Dict[str, Any]: 监控结果
        """
        stats = self._collect_statistics()
        alerts = self._check_alerts(stats)
        export_success = self._export_metrics(stats)

        return {
            'stats': stats,
            'alerts': alerts,
            'export_success': export_success
        }

    def _collect_statistics(self) -> Optional[Dict[str, Any]]:
        """
        收集统计信息

        Returns:
            Optional[Dict[str, Any]]: 统计信息
        """
        if not self.stats_collector:
            return None

        stats = self.stats_collector.collect_stats()

        if stats:
            self._publish_stats_collected_event(stats)

        return stats

    def _check_alerts(self, stats: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检查告警

        Args:
            stats: 统计信息

        Returns:
            List[Dict[str, Any]]: 告警列表
        """
        if not self.alert_manager or not stats:
            return []

        alerts = self.alert_manager.check_alerts(stats)

        if alerts:
            self._publish_alerts_detected_event(alerts)

        return alerts

    def _export_metrics(self, stats: Optional[Dict[str, Any]]) -> bool:
        """
        导出指标

        Args:
            stats: 统计信息

        Returns:
            bool: 导出是否成功
        """
        if not self.metrics_exporter or not stats:
            return False

        export_success = self.metrics_exporter.export_metrics(stats)
        self._publish_metrics_exported_event(export_success, stats)

        return export_success

    def _publish_stats_collected_event(self, stats: Dict[str, Any]):
        """发布统计收集事件"""
        publish_event(
            "monitoring.stats.collected",
            {"pool_name": self.pool_name, "stats": stats},
            "MonitoringCoordinator"
        )

    def _publish_alerts_detected_event(self, alerts: List[Dict[str, Any]]):
        """发布告警检测事件"""
        publish_event(
            "monitoring.alerts.detected",
            {"pool_name": self.pool_name, "alerts": alerts},
            "MonitoringCoordinator"
        )

    def _publish_metrics_exported_event(self, export_success: bool, stats: Dict[str, Any]):
        """发布指标导出事件"""
        publish_event(
            "monitoring.metrics.exported",
            {
                "pool_name": self.pool_name,
                "export_success": export_success,
                "stats_count": len(stats) if stats else 0
            },
            "MonitoringCoordinator"
        )

    def _publish_cycle_completed_event(self, result: Dict[str, Any]):
        """发布监控周期完成事件"""
        publish_event(
            "monitoring.cycle.completed",
            {
                "pool_name": self.pool_name,
                "stats_collected": result['stats'] is not None,
                "alerts_count": len(result['alerts']),
                "export_success": result['export_success']
            },
            "MonitoringCoordinator"
        )

    def _publish_cycle_error_event(self, error: Exception):
        """发布监控周期错误事件"""
        publish_event(
            "monitoring.cycle.error",
            {"pool_name": self.pool_name, "error": str(error)},
            "MonitoringCoordinator"
        )

    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
