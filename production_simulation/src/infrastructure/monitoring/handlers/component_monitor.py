"""
component_monitor 模块

提供 component_monitor 相关功能和接口。
"""

import json
import logging

import threading
import time

from infrastructure.utils.common.core.base_components import ComponentFactory
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
#!/usr/bin/env python3
"""
ComponentFactory使用情况监控系统

实时监控ComponentFactory的使用情况，包括：
1. 创建频率统计
2. 性能指标监控
3. 异常检测和告警
4. 使用模式分析

作者: RQA2025 Team
版本: 1.0.0
更新: 2025年9月21日
"""


@dataclass
class ComponentUsageMetrics:
    """组件使用指标"""
    component_type: str
    total_creations: int
    successful_creations: int
    failed_creations: int
    average_creation_time: float
    peak_concurrent_usage: int
    memory_usage_mb: float
    last_used: datetime
    error_rate: float


@dataclass
class ComponentAlert:
    """组件告警"""
    alert_id: str
    component_type: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool


class ComponentFactoryMonitor:
    """ComponentFactory监控器"""

    def __init__(self, max_history_size: int = 1000):
        self.usage_metrics: Dict[str, ComponentUsageMetrics] = {}
        self.creation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.active_instances: Dict[str, int] = defaultdict(int)
        self.alerts: List[ComponentAlert] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 配置告警阈值
        self.alert_thresholds = {
            'error_rate': 0.05,      # 5%错误率
            'creation_time': 1000,   # 1秒创建时间
            'memory_usage': 500,     # 500MB内存使用
            'concurrent_usage': 1000  # 1000个并发实例
        }

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("ComponentFactory监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("ComponentFactory监控已停止")

    def record_component_creation(self, component_type: str, creation_time: float, success: bool = True):
        """记录组件创建"""
        # 更新使用指标
        if component_type not in self.usage_metrics:
            self.usage_metrics[component_type] = ComponentUsageMetrics(
                component_type=component_type,
                total_creations=0,
                successful_creations=0,
                failed_creations=0,
                average_creation_time=0.0,
                peak_concurrent_usage=0,
                memory_usage_mb=0.0,
                last_used=datetime.now(),
                error_rate=0.0
            )

        metrics = self.usage_metrics[component_type]
        metrics.total_creations += 1
        metrics.last_used = datetime.now()

        if success:
            metrics.successful_creations += 1
        else:
            metrics.failed_creations += 1

        # 更新错误率
        if metrics.total_creations > 0:
            metrics.error_rate = metrics.failed_creations / metrics.total_creations

        # 记录创建时间
        self.creation_times[component_type].append(creation_time)

        # 更新平均创建时间
        if self.creation_times[component_type]:
            metrics.average_creation_time = sum(
                self.creation_times[component_type]) / len(self.creation_times[component_type])

        # 更新并发使用
        self.active_instances[component_type] += 1
        metrics.peak_concurrent_usage = max(
            metrics.peak_concurrent_usage, self.active_instances[component_type])

        # 检查告警条件
        self._check_alerts(component_type, metrics)

    def record_component_destruction(self, component_type: str):
        """记录组件销毁"""
        if component_type in self.active_instances:
            self.active_instances[component_type] = max(
                0, self.active_instances[component_type] - 1)

    def get_usage_report(self) -> Dict[str, Any]:
        """获取使用报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_components': len(self.usage_metrics),
                'total_creations': sum(m.total_creations for m in self.usage_metrics.values()),
                'active_instances': dict(self.active_instances),
                'alerts_count': len([a for a in self.alerts if not a.resolved])
            },
            'component_metrics': {},
            'recent_alerts': []
        }

        # 组件指标
        for component_type, metrics in self.usage_metrics.items():
            report['component_metrics'][component_type] = {
                'total_creations': metrics.total_creations,
                'successful_creations': metrics.successful_creations,
                'failed_creations': metrics.failed_creations,
                'error_rate': metrics.error_rate,
                'average_creation_time': metrics.average_creation_time,
                'peak_concurrent_usage': metrics.peak_concurrent_usage,
                'active_instances': self.active_instances[component_type],
                'last_used': metrics.last_used.isoformat()
            }

        # 最近告警
        recent_alerts = [a for a in self.alerts if not a.resolved][-10:]  # 最近10个未解决告警
        report['recent_alerts'] = [
            {
                'alert_id': a.alert_id,
                'component_type': a.component_type,
                'alert_type': a.alert_type,
                'severity': a.severity,
                'message': a.message,
                'timestamp': a.timestamp.isoformat()
            } for a in recent_alerts
        ]

        return report

    def _check_alerts(self, component_type: str, metrics: ComponentUsageMetrics):
        """检查告警条件"""

        # 错误率告警
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            self._create_alert(
                component_type,
                'high_error_rate',
                'HIGH',
                f"组件 {component_type} 错误率过高: {metrics.error_rate:.1%} (阈值: {self.alert_thresholds['error_rate']:.1%})"
            )

        # 创建时间告警
        if metrics.average_creation_time > self.alert_thresholds['creation_time']:
            self._create_alert(
                component_type,
                'slow_creation',
                'MEDIUM',
                f"组件 {component_type} 创建时间过慢: {metrics.average_creation_time:.0f}ms (阈值: {self.alert_thresholds['creation_time']}ms)"
            )

        # 并发使用告警
        if self.active_instances[component_type] > self.alert_thresholds['concurrent_usage']:
            self._create_alert(
                component_type,
                'high_concurrency',
                'MEDIUM',
                f"组件 {component_type} 并发使用过高: {self.active_instances[component_type]} (阈值: {self.alert_thresholds['concurrent_usage']})"
            )

    def _create_alert(self, component_type: str, alert_type: str, severity: str, message: str):
        """创建告警"""
        alert_id = f"{component_type}_{alert_type}_{int(time.time())}"

        alert = ComponentAlert(
            alert_id=alert_id,
            component_type=component_type,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            resolved=False
        )

        self.alerts.append(alert)

        self.logger.warning(f"告警创建: {alert_id} - {message}")

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.logger.info(f"告警解决: {alert_id}")
                break

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 定期检查和报告
                time.sleep(60)  # 每分钟检查一次

                # 检查长期未使用的组件
                now = datetime.now()
                for component_type, metrics in self.usage_metrics.items():
                    if (now - metrics.last_used) > timedelta(hours=1):
                        self.logger.info(f"组件 {component_type} 已1小时未使用")

                # 清理旧告警 (保留7天)
                cutoff_date = now - timedelta(days=7)
                self.alerts = [a for a in self.alerts if a.timestamp >
                               cutoff_date or not a.resolved]

            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")

    def export_metrics(self, format_type: str = 'json') -> str:
        """导出指标数据"""
        report = self.get_usage_report()

        if format_type == 'json':
            return json.dumps(report, indent=2, ensure_ascii=False)
        elif format_type == 'csv':
            # 转换为CSV格式
            lines = ["Component,Total_Creations,Successful,Failed,Error_Rate,Avg_Time,Peak_Usage,Active"]

            for component, metrics in report['component_metrics'].items():
                line = ",".join([
                    component,
                    str(metrics['total_creations']),
                    str(metrics['successful_creations']),
                    str(metrics['failed_creations']),
                    ".4f",
                    ".2f",
                    str(metrics['peak_concurrent_usage']),
                    str(metrics['active_instances'])
                ])
                lines.append(line)

            return "\n".join(lines)

        return str(report)


# 全局监控实例
_component_monitor = None


def get_component_monitor() -> ComponentFactoryMonitor:
    """获取全局ComponentFactory监控器"""
    global _component_monitor
    if _component_monitor is None:
        _component_monitor = ComponentFactoryMonitor()
    return _component_monitor


def monitor_component_creation(component_type: str, creation_time: float, success: bool = True):
    """监控组件创建的便捷函数"""
    monitor = get_component_monitor()
    monitor.record_component_creation(component_type, creation_time, success)


def monitor_component_destruction(component_type: str):
    """监控组件销毁的便捷函数"""
    monitor = get_component_monitor()
    monitor.record_component_destruction(component_type)

# 示例：如何在ComponentFactory中使用监控


class MonitoredComponentFactory:
    """带监控功能的ComponentFactory示例"""

    def __init__(self):
        self.factory = ComponentFactory()
        self.monitor = get_component_monitor()

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """创建组件（带监控）"""
        start_time = time.time()

        try:
            component = self.factory.create_component(component_type, config or {})
            creation_time = (time.time() - start_time) * 1000  # 毫秒

            success = component is not None
            self.monitor.record_component_creation(component_type, creation_time, success)

            return component

        except Exception as e:
            creation_time = (time.time() - start_time) * 1000
            self.monitor.record_component_creation(component_type, creation_time, False)
            raise


def main():
    """主函数 - 演示监控功能"""
    print("📊 ComponentFactory监控系统演示")
    print("=" * 40)

    # 获取监控器
    monitor = get_component_monitor()
    monitor.start_monitoring()

    try:
        # 执行完整的演示流程
        _run_component_simulation(monitor)
        _display_usage_report(monitor)
        _export_metrics_demo(monitor)

    finally:
        monitor.stop_monitoring()

    print("\\n✅ 监控演示完成!")


def _run_component_simulation(monitor) -> None:
    """运行组件创建和销毁模拟"""
    print("\\n🔧 模拟组件创建...")

    # 模拟成功创建
    for i in range(10):
        monitor.record_component_creation("test_component", 50.0 + i, True)
        time.sleep(0.1)

    # 模拟失败创建
    for i in range(2):
        monitor.record_component_creation("test_component", 100.0, False)
        time.sleep(0.1)

    # 模拟销毁
    for i in range(5):
        monitor.record_component_destruction("test_component")


def _display_usage_report(monitor) -> None:
    """显示使用报告"""
    print("\\n📋 生成使用报告...")
    report = monitor.get_usage_report()

    print("\\n📊 使用统计:")
    print(f"  总组件类型: {report['summary']['total_components']}")
    print(f"  总创建次数: {report['summary']['total_creations']}")
    print(f"  活跃实例: {report['summary']['active_instances']}")

    for component, metrics in report['component_metrics'].items():
        print(f"\\n🔧 组件: {component}")
        print(f"  创建次数: {metrics['total_creations']}")
        print(f"  错误率: {metrics['error_rate']:.1%}")
        print(f"  平均创建时间: {metrics['average_creation_time']:.1f}ms")
        print(f"  峰值并发: {metrics['peak_concurrent_usage']}")


def _export_metrics_demo(monitor) -> None:
    """演示指标导出功能"""
    print("\\n💾 导出指标数据...")
    
    # 导出JSON格式
    json_data = monitor.export_metrics('json')
    print(f"📄 导出了 {len(json_data)} 字符的JSON数据")

    # 导出CSV格式
    csv_data = monitor.export_metrics('csv')
    csv_lines = csv_data.split('\n')
    print(f"📊 导出了 {len(csv_lines)} 行CSV数据")


if __name__ == "__main__":
    main()
