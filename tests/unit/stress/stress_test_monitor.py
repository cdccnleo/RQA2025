"""压力测试监控和分析工具"""
import time
import logging
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

class StressTestMonitor:
    """压力测试实时监控系统"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "start_time": None,
            "scenarios": {},
            "system_metrics": []
        }

    def start_monitoring(self):
        """开始监控"""
        self.metrics["start_time"] = datetime.now()
        self.logger.info("压力测试监控系统启动")

    def record_scenario_start(self, scenario_name: str):
        """记录场景开始"""
        self.metrics["scenarios"][scenario_name] = {
            "start_time": datetime.now(),
            "status": "running",
            "metrics": []
        }
        self.logger.info(f"开始监控场景: {scenario_name}")

    def record_scenario_end(self, scenario_name: str, status: str):
        """记录场景结束"""
        if scenario_name in self.metrics["scenarios"]:
            self.metrics["scenarios"][scenario_name].update({
                "end_time": datetime.now(),
                "status": status
            })
            self.logger.info(f"场景 {scenario_name} 完成, 状态: {status}")

    def record_system_metrics(self, metrics: Dict):
        """记录系统指标"""
        timestamp = datetime.now()
        self.metrics["system_metrics"].append({
            "timestamp": timestamp,
            **metrics
        })

        # 实时打印关键指标
        self.logger.info(
            f"系统指标 - 延迟: {metrics.get('latency', 0):.2f}ms | "
            f"吞吐量: {metrics.get('throughput', 0):.2f}/s | "
            f"CPU: {metrics.get('cpu_usage', 0):.1f}% | "
            f"内存: {metrics.get('memory_usage', 0):.1f}%"
        )

    def generate_realtime_report(self) -> Dict:
        """生成实时报告"""
        current_time = datetime.now()
        elapsed = (current_time - self.metrics["start_time"]).total_seconds()

        running_scenarios = [
            name for name, data in self.metrics["scenarios"].items()
            if data.get("status") == "running"
        ]

        completed_scenarios = [
            name for name, data in self.metrics["scenarios"].items()
            if data.get("status") in ["completed", "failed"]
        ]

        return {
            "elapsed_time": f"{elapsed:.2f}秒",
            "running_scenarios": running_scenarios,
            "completed_scenarios": completed_scenarios,
            "latest_metrics": self.metrics["system_metrics"][-1] if self.metrics["system_metrics"] else {}
        }

    def generate_final_report(self) -> Dict:
        """生成最终测试报告"""
        # 计算总体统计信息
        total_scenarios = len(self.metrics["scenarios"])
        completed = sum(
            1 for data in self.metrics["scenarios"].values()
            if data.get("status") == "completed"
        )
        failed = total_scenarios - completed

        # 准备详细场景数据
        scenario_details = []
        for name, data in self.metrics["scenarios"].items():
            duration = (data["end_time"] - data["start_time"]).total_seconds() if "end_time" in data else 0
            scenario_details.append({
                "scenario": name,
                "status": data.get("status", "unknown"),
                "duration": f"{duration:.2f}秒",
                "metrics": data.get("metrics", [])
            })

        # 准备系统指标数据
        system_metrics_df = pd.DataFrame(self.metrics["system_metrics"])

        return {
            "summary": {
                "start_time": self.metrics["start_time"].isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": f"{(datetime.now() - self.metrics['start_time']).total_seconds():.2f}秒",
                "total_scenarios": total_scenarios,
                "completed": completed,
                "failed": failed,
                "success_rate": f"{(completed / total_scenarios * 100):.1f}%" if total_scenarios > 0 else "0%"
            },
            "scenario_details": scenario_details,
            "system_metrics": system_metrics_df.describe().to_dict()
        }

    def visualize_metrics(self):
        """可视化系统指标"""
        if not self.metrics["system_metrics"]:
            self.logger.warning("没有可用的指标数据")
            return

        # 准备数据
        df = pd.DataFrame(self.metrics["system_metrics"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 延迟图表
        plt.subplot(2, 2, 1)
        df["latency"].plot(title="系统延迟(ms)", color='blue')
        plt.grid(True)

        # 吞吐量图表
        plt.subplot(2, 2, 2)
        df["throughput"].plot(title="系统吞吐量(ops/s)", color='green')
        plt.grid(True)

        # CPU使用率图表
        plt.subplot(2, 2, 3)
        df["cpu_usage"].plot(title="CPU使用率(%)", color='red')
        plt.grid(True)

        # 内存使用率图表
        plt.subplot(2, 2, 4)
        df["memory_usage"].plot(title="内存使用率(%)", color='purple')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("stress_test_metrics.png")
        self.logger.info("系统指标图表已保存为 stress_test_metrics.png")

class StressTestAnalyzer:
    """压力测试结果分析器"""

    @staticmethod
    def analyze_throughput(results: Dict) -> Dict:
        """分析吞吐量性能"""
        df = pd.DataFrame(results["system_metrics"])
        throughput_stats = {
            "max": df["throughput"].max(),
            "min": df["throughput"].min(),
            "mean": df["throughput"].mean(),
            "p95": df["throughput"].quantile(0.95)
        }
        return throughput_stats

    @staticmethod
    def analyze_latency(results: Dict) -> Dict:
        """分析延迟性能"""
        df = pd.DataFrame(results["system_metrics"])
        latency_stats = {
            "max": df["latency"].max(),
            "min": df["latency"].min(),
            "mean": df["latency"].mean(),
            "p95": df["latency"].quantile(0.95)
        }
        return latency_stats

    @staticmethod
    def analyze_resource_usage(results: Dict) -> Dict:
        """分析资源使用情况"""
        df = pd.DataFrame(results["system_metrics"])
        return {
            "max_cpu": df["cpu_usage"].max(),
            "avg_cpu": df["cpu_usage"].mean(),
            "max_memory": df["memory_usage"].max(),
            "avg_memory": df["memory_usage"].mean()
        }

    @staticmethod
    def generate_comparison_report(baseline: Dict, current: Dict) -> Dict:
        """生成与基准线的对比报告"""
        return {
            "throughput": {
                "current": StressTestAnalyzer.analyze_throughput(current),
                "baseline": StressTestAnalyzer.analyze_throughput(baseline),
                "improvement": f"{((current['system_metrics']['throughput'].mean() - baseline['system_metrics']['throughput'].mean()) / baseline['system_metrics']['throughput'].mean() * 100):.1f}%"
            },
            "latency": {
                "current": StressTestAnalyzer.analyze_latency(current),
                "baseline": StressTestAnalyzer.analyze_latency(baseline),
                "improvement": f"{((baseline['system_metrics']['latency'].mean() - current['system_metrics']['latency'].mean()) / baseline['system_metrics']['latency'].mean() * 100):.1f}%"
            }
        }

if __name__ == "__main__":
    # 示例用法
    import random
    logging.basicConfig(level=logging.INFO)

    monitor = StressTestMonitor()
    monitor.start_monitoring()

    # 模拟测试场景
    scenarios = ["2015股灾重现", "千股跌停", "Level2数据风暴"]
    for scenario in scenarios:
        monitor.record_scenario_start(scenario)

        # 模拟记录指标
        for _ in range(5):
            metrics = {
                "latency": random.uniform(10, 100),
                "throughput": random.uniform(500, 5000),
                "cpu_usage": random.uniform(20, 80),
                "memory_usage": random.uniform(30, 90)
            }
            monitor.record_system_metrics(metrics)
            time.sleep(1)

        monitor.record_scenario_end(scenario, "completed")

    # 生成报告和图表
    report = monitor.generate_final_report()
    print("测试报告摘要:")
    print(f"总耗时: {report['summary']['total_duration']}")
    print(f"完成场景: {report['summary']['completed']}/{report['summary']['total_scenarios']}")

    monitor.visualize_metrics()
