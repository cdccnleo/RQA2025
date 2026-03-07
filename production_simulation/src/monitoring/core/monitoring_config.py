#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 监控体系配置

实现全链路追踪和性能指标监控
"""

import time
import logging
import json
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringSystem:

    """监控系统"""

    def __init__(self):

        self.metrics = {}
        self.traces = []
        self.alerts = []

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录指标"""
        metric = {
            'name': name,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'tags': tags or {}
        }

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append(metric)

        # 保留最近1000个指标
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]

        logger.info(f"记录指标: {name} = {value}")

    def start_trace(self, trace_id: str, operation: str) -> str:
        """开始链路追踪"""
        span_id = f"{trace_id}-{len(self.traces)}"

        trace = {
            'trace_id': trace_id,
            'span_id': span_id,
            'operation': operation,
            'start_time': time.time(),
            'end_time': None,
            'duration': None,
            'tags': {},
            'events': []
        }

        self.traces.append(trace)
        return span_id

    def end_trace(self, span_id: str, tags: Dict[str, str] = None):
        """结束链路追踪"""
        for trace in self.traces:
            if trace['span_id'] == span_id and trace['end_time'] is None:
                trace['end_time'] = time.time()
                trace['duration'] = trace['end_time'] - trace['start_time']
                trace['tags'].update(tags or {})

                logger.info(f"结束追踪: {span_id}, 耗时: {trace['duration']:.3f}s")
                break

    def add_trace_event(self, span_id: str, event: str, data: Dict[str, Any] = None):
        """添加追踪事件"""
        for trace in self.traces:
            if trace['span_id'] == span_id:
                trace['events'].append({
                    'event': event,
                    'timestamp': time.time(),
                    'data': data or {}
                })
                break

    def check_alerts(self) -> list:
        """检查告警条件"""
        alerts = []

        # CPU使用率告警
        cpu_metric = self.metrics.get('cpu_usage', [])
        if cpu_metric and cpu_metric[-1]['value'] > 80:
            alerts.append({
                'type': 'cpu_high',
                'message': f"CPU使用率过高: {cpu_metric[-1]['value']:.1f}%",
                'severity': 'critical',
                'timestamp': datetime.now().isoformat()
            })

        # 内存使用率告警
        memory_metric = self.metrics.get('memory_usage', [])
        if memory_metric and memory_metric[-1]['value'] > 70:
            alerts.append({
                'type': 'memory_high',
                'message': f"内存使用率过高: {memory_metric[-1]['value']:.1f}%",
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })

        # API响应时间告警
        api_metric = self.metrics.get('api_response_time', [])
        if api_metric and api_metric[-1]['value'] > 1000:  # 1秒
            alerts.append({
                'type': 'api_slow',
                'message': f"API响应时间过慢: {api_metric[-1]['value']:.0f}ms",
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })

        self.alerts.extend(alerts)
        return alerts

    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics_count': sum(len(metrics) for metrics in self.metrics.values()),
            'traces_count': len(self.traces),
            'alerts_count': len(self.alerts),
            'latest_metrics': {},
            'performance_summary': {}
        }

        # 最新指标
        for name, metrics in self.metrics.items():
            if metrics:
                report['latest_metrics'][name] = metrics[-1]

        # 性能摘要
        if self.traces:
            durations = [t['duration'] for t in self.traces if t['duration']]
            if durations:
                report['performance_summary'] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'total_traces': len(durations)
                }

        return report


# 全局监控实例
monitoring = MonitoringSystem()


def collect_system_metrics():
    """收集系统指标"""
    import psutil

    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    monitoring.record_metric('cpu_usage', cpu_percent, {'unit': 'percent'})

    # 内存使用率
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    monitoring.record_metric('memory_usage', memory_percent, {'unit': 'percent'})

    # 磁盘使用率
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    monitoring.record_metric('disk_usage', disk_percent, {'unit': 'percent'})

    # 网络流量
    network = psutil.net_io_counters()
    if network:
        monitoring.record_metric('network_bytes_sent', network.bytes_sent, {'unit': 'bytes'})
        monitoring.record_metric('network_bytes_recv', network.bytes_recv, {'unit': 'bytes'})

    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'disk_percent': disk_percent
    }


def simulate_api_performance_test():
    """模拟API性能测试"""
    import time
    import random

    print("🏃 模拟API性能测试...")

    # 模拟不同响应时间
    response_times = []

    for i in range(100):
        # 模拟API调用
        trace_id = f"api_test_{i}"
        span_id = monitoring.start_trace(trace_id, "api_call")

        # 模拟处理时间 (正常: 50 - 200ms, 偶尔慢响应)
        if random.random() < 0.9:  # 90 % 正常响应
            processing_time = random.uniform(0.05, 0.2)
        else:  # 10 % 慢响应
            processing_time = random.uniform(0.5, 2.0)

        time.sleep(processing_time)
        response_time = processing_time * 1000  # 转换为毫秒

        monitoring.end_trace(span_id, {'response_time_ms': response_time})
        monitoring.record_metric('api_response_time', response_time, {'endpoint': '/api/test'})

        response_times.append(response_time)

    avg_response_time = sum(response_times) / len(response_times)
    p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]

    print(f"   平均响应时间: {avg_response_time:.2f}ms")
    print(f"   P95响应时间: {p95_response_time:.2f}ms")
    return {
        'avg_response_time': avg_response_time,
        'p95_response_time': p95_response_time,
        'total_requests': len(response_times)
    }


def test_concurrency_performance():
    """测试并发性能"""
    import threading
    import time
    import random

    print("🔄 测试并发性能...")

    results = []
    lock = threading.Lock()

    def worker(worker_id):

        # 模拟并发请求
        trace_id = f"concurrency_test_{worker_id}"
        span_id = monitoring.start_trace(trace_id, "concurrent_request")

        # 模拟处理时间
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)

        response_time = processing_time * 1000

        monitoring.end_trace(span_id, {
            'worker_id': worker_id,
            'response_time_ms': response_time
        })

        with lock:
            results.append(response_time)

    # 启动多个并发线程
    threads = []
    num_threads = 50  # 模拟50个并发请求

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    avg_response_time = sum(results) / len(results) if results else 0
    max_response_time = max(results) if results else 0

    print(f"   并发请求数: {len(results)}")
    print(f"   平均响应时间: {avg_response_time:.2f}ms")
    print(f"   最大响应时间: {max_response_time:.2f}ms")
    return {
        'concurrent_requests': len(results),
        'avg_response_time': avg_response_time,
        'max_response_time': max_response_time
    }


if __name__ == "__main__":
    print("测试监控系统...")

    # 收集系统指标
    print("收集系统指标...")
    metrics = collect_system_metrics()
    print(f"   CPU: {metrics['cpu_percent']:.1f}%")
    print(f"   内存: {metrics['memory_percent']:.1f}%")
    print(f"   磁盘: {metrics['disk_percent']:.1f}%")

    print()

    # 模拟API性能测试
    api_results = simulate_api_performance_test()

    print()

    # 测试并发性能
    concurrency_results = test_concurrency_performance()

    print()

    # 检查告警
    alerts = monitoring.check_alerts()
    if alerts:
        print("⚠️  检测到告警:")
        for alert in alerts:
            print(f"   {alert['type']}: {alert['message']}")
    else:
        print("✅ 无告警")

    print()

    # 生成报告
    report = monitoring.generate_report()
    print("📊 监控报告:")
    print(f"   指标数量: {report['metrics_count']}")
    print(f"   追踪数量: {report['traces_count']}")
    print(f"   告警数量: {report['alerts_count']}")

    if report['performance_summary']:
        perf = report['performance_summary']
        print(f"   平均耗时: {perf['avg_duration']:.3f}s")
        print(f"   最大耗时: {perf['max_duration']:.3f}s")
        print(f"   追踪总数: {perf['total_traces']}")

    print("\n✅ 监控系统测试完成")

    # 保存测试结果
    test_results = {
        'system_metrics': metrics,
        'api_performance': api_results,
        'concurrency_performance': concurrency_results,
        'alerts': alerts,
        'monitoring_report': report,
        'timestamp': datetime.now().isoformat()
    }

    with open('monitoring_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)

    print("📁 测试结果已保存: monitoring_test_results.json")
