#!/usr/bin/env python3
"""
性能监控体系

建立基础设施层的性能监控体系
"""

import re
import json
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import functools


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"

        # 监控配置
        self.config = {
            "enable_cpu_monitoring": True,
            "enable_memory_monitoring": True,
            "enable_io_monitoring": True,
            "enable_network_monitoring": True,
            "enable_code_metrics": True,
            "sampling_interval": 60,  # 每60秒采样一次
            "retention_days": 7,      # 数据保留7天
            "alert_thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_io_percent": 90.0,
                "response_time": 2.0  # 秒
            }
        }

        # 监控数据存储
        self.monitoring_data = {
            "system_metrics": [],
            "code_metrics": [],
            "performance_alerts": [],
            "response_times": defaultdict(list),
            "error_rates": defaultdict(list)
        }

        # 监控状态
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_callbacks = []

    def start_monitoring(self) -> Dict[str, Any]:
        """启动性能监控"""
        print("📊 启动性能监控...")

        if self.is_monitoring:
            return {"success": False, "message": "监控已在运行中"}

        self.is_monitoring = True

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        print("✅ 性能监控已启动")
        return {
            "success": True,
            "message": "性能监控已启动",
            "config": self.config
        }

    def stop_monitoring(self) -> Dict[str, Any]:
        """停止性能监控"""
        print("🛑 停止性能监控...")

        if not self.is_monitoring:
            return {"success": False, "message": "监控未在运行"}

        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        print("✅ 性能监控已停止")
        return {"success": True, "message": "性能监控已停止"}

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                if self.config["enable_cpu_monitoring"] or self.config["enable_memory_monitoring"]:
                    self._collect_system_metrics()

                # 收集代码指标
                if self.config["enable_code_metrics"]:
                    self._collect_code_metrics()

                # 检查告警阈值
                self._check_alert_thresholds()

                # 清理过期数据
                self._cleanup_expired_data()

                # 等待下一个采样间隔
                time.sleep(self.config["sampling_interval"])

            except Exception as e:
                print(f"❌ 监控循环错误: {e}")
                time.sleep(5)  # 出错后等待5秒再试

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            metrics = {
                "timestamp": datetime.now(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                    "used": psutil.virtual_memory().used
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "network": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv,
                    "packets_sent": psutil.net_io_counters().packets_sent,
                    "packets_recv": psutil.net_io_counters().packets_recv
                }
            }

            self.monitoring_data["system_metrics"].append(metrics)

            # 保留最近7天的数据
            cutoff_time = datetime.now() - timedelta(days=self.config["retention_days"])
            self.monitoring_data["system_metrics"] = [
                m for m in self.monitoring_data["system_metrics"]
                if m["timestamp"] > cutoff_time
            ]

        except Exception as e:
            print(f"❌ 收集系统指标失败: {e}")

    def _collect_code_metrics(self):
        """收集代码指标"""
        try:
            code_metrics = {
                "timestamp": datetime.now(),
                "file_count": 0,
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0,
                "complexity": {},
                "interface_count": 0,
                "class_count": 0,
                "function_count": 0
            }

            # 分析Python文件
            for py_file in self.infrastructure_dir.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                code_metrics["file_count"] += 1
                file_metrics = self._analyze_file_metrics(py_file)

                code_metrics["total_lines"] += file_metrics["total_lines"]
                code_metrics["code_lines"] += file_metrics["code_lines"]
                code_metrics["comment_lines"] += file_metrics["comment_lines"]
                code_metrics["blank_lines"] += file_metrics["blank_lines"]
                code_metrics["interface_count"] += file_metrics["interface_count"]
                code_metrics["class_count"] += file_metrics["class_count"]
                code_metrics["function_count"] += file_metrics["function_count"]

                # 合并复杂度数据
                for key, value in file_metrics["complexity"].items():
                    if key not in code_metrics["complexity"]:
                        code_metrics["complexity"][key] = []
                    code_metrics["complexity"][key].append(value)

            self.monitoring_data["code_metrics"].append(code_metrics)

        except Exception as e:
            print(f"❌ 收集代码指标失败: {e}")

    def _analyze_file_metrics(self, file_path: Path) -> Dict[str, Any]:
        """分析单个文件的指标"""
        metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "interface_count": 0,
            "class_count": 0,
            "function_count": 0,
            "complexity": {
                "cyclomatic_complexity": 0,
                "halstead_complexity": 0
            }
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            metrics["total_lines"] = len(lines)

            for line in lines:
                line = line.strip()
                if not line:
                    metrics["blank_lines"] += 1
                elif line.startswith('#'):
                    metrics["comment_lines"] += 1
                else:
                    metrics["code_lines"] += 1

            # 重新读取内容进行分析
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 统计接口、类、函数数量
            metrics["interface_count"] = len(re.findall(r'class I[A-Z]\w*Component', content))
            metrics["class_count"] = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
            metrics["function_count"] = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))

            # 计算复杂度（简化版）
            metrics["complexity"]["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(
                content)
            metrics["complexity"]["halstead_complexity"] = self._calculate_halstead_complexity(
                content)

        except Exception as e:
            print(f"❌ 分析文件 {file_path} 失败: {e}")

        return metrics

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """计算循环复杂度（简化版）"""
        complexity = 1  # 基础复杂度

        # 统计控制流关键字
        control_flow_keywords = ['if ', 'elif ', 'else:',
                                 'for ', 'while ', 'try:', 'except ', 'with ']
        for keyword in control_flow_keywords:
            complexity += content.count(keyword)

        return complexity

    def _calculate_halstead_complexity(self, content: str) -> float:
        """计算Halstead复杂度（简化版）"""
        # 提取操作符和操作数
        operators = re.findall(r'[+\-*/=<>!&|%^~]', content)
        operands = re.findall(r'\b[a-zA-Z_]\w*\b', content)

        if not operators or not operands:
            return 0.0

        n1 = len(set(operators))  # 唯一操作符数
        n2 = len(set(operands))   # 唯一操作数
        N1 = len(operators)       # 总操作符数
        N2 = len(operands)        # 总操作数

        # Halstead复杂度公式
        try:
            return (n1 + n2) * (N1 + N2) / (2 * n2)
        except:
            return 0.0

    def _check_alert_thresholds(self):
        """检查告警阈值"""
        if not self.monitoring_data["system_metrics"]:
            return

        latest_metrics = self.monitoring_data["system_metrics"][-1]

        # 检查CPU使用率
        if latest_metrics["cpu_percent"] > self.config["alert_thresholds"]["cpu_percent"]:
            self._trigger_alert("high_cpu_usage", {
                "current": latest_metrics["cpu_percent"],
                "threshold": self.config["alert_thresholds"]["cpu_percent"]
            })

        # 检查内存使用率
        if latest_metrics["memory"]["percent"] > self.config["alert_thresholds"]["memory_percent"]:
            self._trigger_alert("high_memory_usage", {
                "current": latest_metrics["memory"]["percent"],
                "threshold": self.config["alert_thresholds"]["memory_percent"]
            })

    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """触发告警"""
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "data": data,
            "severity": "warning"
        }

        self.monitoring_data["performance_alerts"].append(alert)

        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"❌ 告警回调失败: {e}")

        print(f"⚠️ 性能告警: {alert_type} - {data}")

    def _cleanup_expired_data(self):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=self.config["retention_days"])

        for data_type in ["system_metrics", "code_metrics", "performance_alerts"]:
            self.monitoring_data[data_type] = [
                item for item in self.monitoring_data[data_type]
                if item["timestamp"] > cutoff_time
            ]

    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": datetime.now(),
            "is_monitoring": self.is_monitoring,
            "config": self.config,
            "current_metrics": {},
            "historical_data": {},
            "alerts": []
        }

        # 当前指标
        if self.monitoring_data["system_metrics"]:
            report["current_metrics"] = self.monitoring_data["system_metrics"][-1]

        if self.monitoring_data["code_metrics"]:
            report["current_metrics"]["code"] = self.monitoring_data["code_metrics"][-1]

        # 历史数据统计
        if self.monitoring_data["system_metrics"]:
            cpu_usage = [m["cpu_percent"] for m in self.monitoring_data["system_metrics"]]
            memory_usage = [m["memory"]["percent"] for m in self.monitoring_data["system_metrics"]]

            report["historical_data"] = {
                "cpu_usage_avg": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "memory_usage_avg": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "monitoring_points": len(self.monitoring_data["system_metrics"])
            }

        # 最近告警
        recent_alerts = self.monitoring_data["performance_alerts"][-10:]
        report["alerts"] = recent_alerts

        return report

    def performance_decorator(func: Callable) -> Callable:
        """性能监控装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                response_time = end_time - start_time

                # 记录响应时间
                func_name = func.__name__
                self.monitoring_data["response_times"][func_name].append({
                    "timestamp": datetime.now(),
                    "response_time": response_time
                })

                # 检查响应时间阈值
                if response_time > self.config["alert_thresholds"]["response_time"]:
                    self._trigger_alert("slow_response", {
                        "function": func_name,
                        "response_time": response_time,
                        "threshold": self.config["alert_thresholds"]["response_time"]
                    })

        return wrapper

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        report_data = self.get_performance_report()

        # 保存报告
        report_path = self.project_root / "reports" / \
            f"performance_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "report_path": str(report_path),
            "data": report_data
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='性能监控体系')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--start', action='store_true', help='启动性能监控')
    parser.add_argument('--stop', action='store_true', help='停止性能监控')
    parser.add_argument('--status', action='store_true', help='查看监控状态')
    parser.add_argument('--report', action='store_true', help='生成性能报告')
    parser.add_argument('--alert-callback', help='设置告警回调函数')

    args = parser.parse_args()

    monitor = PerformanceMonitor(args.project)

    if args.start:
        result = monitor.start_monitoring()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.stop:
        result = monitor.stop_monitoring()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.status:
        result = monitor.get_performance_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.report:
        result = monitor.generate_monitoring_report()
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    elif args.alert_callback:
        # 这里可以实现自定义告警回调
        print("📢 告警回调功能待实现")

    else:
        # 默认启动监控
        result = monitor.start_monitoring()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
