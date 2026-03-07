#!/usr/bin/env python3
"""
RQA2025 增强日志系统
支持结构化日志、监控指标和性能追踪
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import psutil
import platform

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()

    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """增加计数器"""
        key = self._make_key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表盘"""
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

    def record_timing(self, name: str, duration: float, labels: Dict[str, str] = None):
        """记录时间指标"""
        self.record_histogram(f"{name}_duration", duration, labels)

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成指标键"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}[{label_str}]"
        return name

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {},
            'uptime': time.time() - self.start_time
        }

        for name, values in self.histograms.items():
            if values:
                summary['histograms'][name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99)
                }

        return summary

    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def __init__(self):
        super().__init__()

    def format(self, record):
        """格式化日志记录"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceTracker:
    """性能追踪器"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_timers = {}

    def start_timer(self, name: str, labels: Dict[str, str] = None):
        """开始计时"""
        timer_id = f"{name}_{threading.get_ident()}"
        self.active_timers[timer_id] = {
            'name': name,
            'start_time': time.time(),
            'labels': labels or {}
        }

    def end_timer(self, name: str, labels: Dict[str, str] = None):
        """结束计时"""
        timer_id = f"{name}_{threading.get_ident()}"
        if timer_id in self.active_timers:
            timer = self.active_timers.pop(timer_id)
            duration = time.time() - timer['start_time']
            self.metrics.record_timing(name, duration, timer['labels'])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SystemMonitor:
    """系统监控器"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: int = 30):
        """开始系统监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止系统监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logging.error(f"系统监控异常: {e}")

    def _collect_system_metrics(self):
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge('system_cpu_percent', cpu_percent)

        # 内存使用率
        memory = psutil.virtual_memory()
        self.metrics.set_gauge('system_memory_percent', memory.percent)
        self.metrics.set_gauge('system_memory_available_gb', memory.available / (1024**3))

        # 磁盘使用率
        disk = psutil.disk_usage('.')
        self.metrics.set_gauge('system_disk_percent', disk.percent)
        self.metrics.set_gauge('system_disk_free_gb', disk.free / (1024**3))

        # 网络IO
        net_io = psutil.net_io_counters()
        self.metrics.set_gauge('system_network_bytes_sent', net_io.bytes_sent)
        self.metrics.set_gauge('system_network_bytes_recv', net_io.bytes_recv)

        # 进程信息
        process = psutil.Process()
        self.metrics.set_gauge('process_cpu_percent', process.cpu_percent())
        self.metrics.set_gauge('process_memory_mb', process.memory_info().rss / (1024**2))


class EnhancedLoggingSystem:
    """增强日志系统"""

    def __init__(self, log_dir: str = "logs"):
        logger = logging.getLogger(__name__)
        logger.info("【日志测试】EnhancedLoggingSystem.__init__ 开始")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("【日志测试】日志目录创建完成")
        # 初始化组件
        self.metrics = MetricsCollector()
        logger.info("【日志测试】MetricsCollector初始化完成")
        self.performance_tracker = PerformanceTracker(self.metrics)
        logger.info("【日志测试】PerformanceTracker初始化完成")
        self.system_monitor = SystemMonitor(self.metrics)
        logger.info("【日志测试】SystemMonitor初始化完成")
        # 配置日志
        logger.info("【日志测试】准备调用self._setup_logging()")
        self._setup_logging()
        logger.info("【日志测试】_setup_logging完成")
        # 启动系统监控
        logger.info("【日志测试】准备调用system_monitor.start_monitoring()")
        self.system_monitor.start_monitoring()
        logger.info("【日志测试】system_monitor.start_monitoring完成")
        logger.info("【日志测试】EnhancedLoggingSystem.__init__ 结束")

    def _setup_logging(self):
        """设置日志配置"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        try:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

            # 文件处理器 - 结构化日志
            structured_handler = logging.FileHandler(
                self.log_dir / "structured.log",
                encoding='utf-8'
            )
            structured_handler.setLevel(logging.INFO)
            structured_formatter = StructuredFormatter()
            structured_handler.setFormatter(structured_formatter)
            root_logger.addHandler(structured_handler)

            # 错误日志处理器
            error_handler = logging.FileHandler(
                self.log_dir / "errors.log",
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
                'Exception: %(exc_info)s\n'
                'Location: %(pathname)s:%(lineno)d\n'
            )
            error_handler.setFormatter(error_formatter)
            root_logger.addHandler(error_handler)

            # 性能日志处理器
            performance_handler = logging.FileHandler(
                self.log_dir / "performance.log",
                encoding='utf-8'
            )
            performance_handler.setLevel(logging.INFO)
            performance_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            performance_handler.setFormatter(performance_formatter)
            root_logger.addHandler(performance_handler)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"【日志测试】_setup_logging异常: {e}")

    def log_with_metrics(self, level: str, message: str,
                         extra_fields: Dict[str, Any] = None,
                         metrics_labels: Dict[str, str] = None):
        """记录带指标的日志"""
        logger = logging.getLogger()

        # 创建日志记录
        record = logger.makeRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            fn='',
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # 添加额外字段
        if extra_fields:
            record.extra_fields = extra_fields

        # 记录日志
        logger.handle(record)

        # 记录指标
        self.metrics.increment_counter(f"log_{level.lower()}", 1, metrics_labels)

    def log_performance(self, operation: str, duration: float,
                        success: bool = True, extra_data: Dict[str, Any] = None):
        """记录性能日志"""
        labels = {
            'operation': operation,
            'success': str(success)
        }

        self.metrics.record_timing(f"{operation}_duration", duration, labels)
        self.metrics.increment_counter(f"{operation}_count", 1, labels)

        if not success:
            self.metrics.increment_counter(f"{operation}_errors", 1, labels)

        # 记录性能日志
        log_data = {
            'operation': operation,
            'duration': duration,
            'success': success
        }
        if extra_data:
            log_data.update(extra_data)

        self.log_with_metrics('INFO', f"Performance: {operation}", log_data, labels)

    def log_security_event(self, event_type: str, severity: str,
                           details: Dict[str, Any] = None):
        """记录安全事件"""
        labels = {
            'event_type': event_type,
            'severity': severity
        }

        self.metrics.increment_counter('security_events', 1, labels)

        log_data = {
            'event_type': event_type,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        if details:
            log_data.update(details)

        self.log_with_metrics('WARNING', f"Security event: {event_type}", log_data, labels)

    def log_business_event(self, event_type: str, user_id: str = None,
                           data: Dict[str, Any] = None):
        """记录业务事件"""
        labels = {
            'event_type': event_type,
            'user_id': user_id or 'anonymous'
        }

        self.metrics.increment_counter('business_events', 1, labels)

        log_data = {
            'event_type': event_type,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        if data:
            log_data.update(data)

        self.log_with_metrics('INFO', f"Business event: {event_type}", log_data, labels)

    def get_metrics_report(self) -> Dict[str, Any]:
        """获取指标报告"""
        metrics_summary = self.metrics.get_metrics_summary()

        # 添加系统信息
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }

        return {
            'metrics': metrics_summary,
            'system_info': system_info,
            'report_time': datetime.now().isoformat()
        }

    def generate_metrics_report(self) -> str:
        """生成指标报告文件"""
        report_file = "reports/testing/metrics_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        report_data = self.get_metrics_report()

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_file

    def cleanup_old_logs(self, days: int = 30):
        """清理旧日志文件"""
        cutoff_time = datetime.now() - timedelta(days=days)

        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    log_file.unlink()
                    logging.info(f"已删除旧日志文件: {log_file}")
                except Exception as e:
                    logging.error(f"删除日志文件失败 {log_file}: {e}")


def main():
    """主函数"""
    # 初始化增强日志系统
    logging_system = EnhancedLoggingSystem()

    # 测试各种日志功能
    logging_system.log_with_metrics('INFO', '测试信息日志', {'test': True})
    logging_system.log_with_metrics('WARNING', '测试警告日志', {'warning_type': 'test'})

    # 测试性能日志
    with logging_system.performance_tracker:
        logging_system.performance_tracker.start_timer('test_operation')
        time.sleep(0.1)  # 模拟操作
        logging_system.performance_tracker.end_timer('test_operation')

    # 测试安全事件日志
    logging_system.log_security_event('code_review', 'medium', {'file': 'test.py'})

    # 测试业务事件日志
    logging_system.log_business_event('test_completed', 'user123', {'result': 'success'})

    # 生成报告
    report_file = logging_system.generate_metrics_report()
    print(f"指标报告已生成: {report_file}")

    # 清理旧日志
    logging_system.cleanup_old_logs(7)


if __name__ == "__main__":
    main()
