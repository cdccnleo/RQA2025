#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E测试执行监控器
"""

import time
import psutil
import threading
from datetime import datetime


class TestExecutionMonitor:
    """测试执行监控器"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "test_progress": []
        }
        self.monitoring = False

    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.monitoring = True

        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        monitor_thread.start()

        print("📊 开始监控E2E测试执行...")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.end_time = time.time()

        print("📊 测试执行监控完成")

    def _monitor_system(self):
        """监控系统资源"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append({
                    "timestamp": datetime.now().isoformat(),
                    "value": cpu_percent
                })

                # 内存使用率
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append({
                    "timestamp": datetime.now().isoformat(),
                    "value": memory.percent,
                    "used": memory.used,
                    "available": memory.available
                })

                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics["disk_io"].append({
                        "timestamp": datetime.now().isoformat(),
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes
                    })

            except Exception as e:
                print(f"监控出错: {e}")

            time.sleep(5)  # 每5秒收集一次数据

    def record_test_progress(self, test_name, status, duration=None):
        """记录测试进度"""
        self.metrics["test_progress"].append({
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "status": status,
            "duration": duration
        })

    def generate_report(self):
        """生成监控报告"""
        if not self.start_time or not self.end_time:
            return None

        total_duration = self.end_time - self.start_time

        report = {
            "monitoring_period": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration_seconds": total_duration,
                "total_duration_minutes": total_duration / 60
            },
            "system_metrics": {
                "avg_cpu_usage": sum(m["value"] for m in self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
                "max_cpu_usage": max(m["value"] for m in self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
                "avg_memory_usage": sum(m["value"] for m in self.metrics["memory_usage"]) / len(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
                "max_memory_usage": max(m["value"] for m in self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
            },
            "test_metrics": {
                "total_tests": len(self.metrics["test_progress"]),
                "passed_tests": len([t for t in self.metrics["test_progress"] if t["status"] == "passed"]),
                "failed_tests": len([t for t in self.metrics["test_progress"] if t["status"] == "failed"]),
                "avg_test_duration": sum(t["duration"] or 0 for t in self.metrics["test_progress"]) / len(self.metrics["test_progress"]) if self.metrics["test_progress"] else 0
            },
            "performance_analysis": {
                "efficiency_rating": "good" if total_duration < 120 else "needs_improvement",
                "resource_usage": "optimal" if self.metrics["cpu_usage"] and max(m["value"] for m in self.metrics["cpu_usage"]) < 80 else "high",
                "bottleneck_identified": self._identify_bottlenecks()
            }
        }

        return report

    def _identify_bottlenecks(self):
        """识别性能瓶颈"""
        bottlenecks = []

        # 检查CPU瓶颈
        if self.metrics["cpu_usage"]:
            max_cpu = max(m["value"] for m in self.metrics["cpu_usage"])
            if max_cpu > 85:
                bottlenecks.append(f"CPU使用率过高: {max_cpu}%")

        # 检查内存瓶颈
        if self.metrics["memory_usage"]:
            max_memory = max(m["value"] for m in self.metrics["memory_usage"])
            if max_memory > 90:
                bottlenecks.append(f"内存使用率过高: {max_memory}%")

        return bottlenecks if bottlenecks else ["无明显瓶颈"]

    def save_report(self, file_path):
        """保存监控报告"""
        report = self.generate_report()
        if report:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"📊 监控报告已保存: {file_path}")
            return True
        return False


# 全局监控实例
test_monitor = TestExecutionMonitor()


def start_test_monitoring():
    """开始测试监控"""
    test_monitor.start_monitoring()


def stop_test_monitoring():
    """停止测试监控"""
    test_monitor.stop_monitoring()
    return test_monitor.generate_report()


if __name__ == "__main__":
    # 测试监控功能
    print("测试监控器功能...")

    monitor = TestExecutionMonitor()
    monitor.start_monitoring()

    # 模拟测试执行
    time.sleep(10)  # 运行10秒

    monitor.record_test_progress("test_user_login", "passed", 2.5)
    monitor.record_test_progress("test_portfolio_creation", "passed", 3.1)
    monitor.record_test_progress("test_strategy_execution", "failed", 5.2)

    monitor.stop_monitoring()

    # 生成报告
    report = monitor.generate_report()
    if report:
        print(f"监控报告: {report['monitoring_period']}")
        print(
            f"测试通过率: {report['test_metrics']['passed_tests']}/{report['test_metrics']['total_tests']}")
        print(f"平均测试时长: {report['test_metrics']['avg_test_duration']:.1f}秒")

    print("✅ 监控器功能测试完成")
