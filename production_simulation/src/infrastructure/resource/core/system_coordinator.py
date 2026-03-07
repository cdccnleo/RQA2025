"""
system_coordinator 模块

提供 system_coordinator 相关功能和接口。
"""


import threading
import time

from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from typing import Dict, Optional, Any
"""
系统协调器

负责监控告警系统的启动、停止和协调各个组件的工作
"""


class SystemCoordinator:
    """系统协调器 - 负责系统的启动、停止和组件协调"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        # 使用共享的日志记录器和错误处理器
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        self.config = config or {}
        self.running = False
        self.alert_check_thread: Optional[threading.Thread] = None

        # 组件引用（将在facade中设置）
        self.performance_monitor = None
        self.alert_manager = None
        self.notification_manager = None
        self.test_monitor = None
        self.alert_rule_manager = None

    def set_components(self, performance_monitor, alert_manager,
                       notification_manager, test_monitor, alert_rule_manager):
        """设置组件引用"""
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.notification_manager = notification_manager
        self.test_monitor = test_monitor
        self.alert_rule_manager = alert_rule_manager

    def start(self):
        """启动监控告警系统"""
        if self.running:
            return

        try:
            # 启动性能监控
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()

            # 启动测试监控
            if self.test_monitor:
                self.test_monitor.start_monitoring()

            # 启动告警检查循环
            self._start_alert_check_loop()

            self.running = True
            self.logger.log_info("监控告警系统启动成功")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "监控告警系统启动失败"})
            raise

    def stop(self):
        """停止监控告警系统"""
        if not self.running:
            return

        try:
            self.running = False

            # 停止告警检查
            if self.alert_check_thread:
                self.alert_check_thread.join(timeout=5.0)

            # 停止各个组件
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            if self.test_monitor:
                self.test_monitor.stop_monitoring()

            self.logger.log_info("监控告警系统已停止")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "监控告警系统停止失败"})

    def _start_alert_check_loop(self):
        """启动告警检查循环"""
        self.alert_check_thread = threading.Thread(target=self._alert_check_loop, daemon=True)
        self.alert_check_thread.start()

    def _alert_check_loop(self):
        """告警检查循环"""
        check_interval = self.config.get('alert_check_interval', 30)

        while self.running:
            try:
                self._check_alerts()
                time.sleep(check_interval)
            except Exception as e:
                self.error_handler.handle_error(e, {"context": "告警检查循环异常"})
                time.sleep(check_interval)

    def _check_alerts(self):
        """检查告警条件"""
        if not self.alert_rule_manager:
            return

        try:
            # 获取当前性能指标
            if self.performance_monitor:
                current_metrics = self.performance_monitor.get_current_metrics()
                if current_metrics:
                    # 检查告警规则
                    alerts = self.alert_rule_manager.check_alerts(current_metrics)
                    for alert in alerts:
                        # 处理告警
                        if self.alert_manager:
                            self.alert_manager.add_alert(alert)
                        # 发送通知
                        if self.notification_manager:
                            self.notification_manager.send_notification(alert)

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "告警检查异常"})

    def is_running(self) -> bool:
        """检查系统是否正在运行"""
        return self.running

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self.running,
            "components": {
                "performance_monitor": self.performance_monitor is not None,
                "alert_manager": self.alert_manager is not None,
                "notification_manager": self.notification_manager is not None,
                "test_monitor": self.test_monitor is not None,
                "alert_rule_manager": self.alert_rule_manager is not None
            }
        }
