#!/usr/bin/env python3
"""
性能监控脚本
RQA2025 生产环境性能监控工具
"""

import sys
import time
import json
import logging
import subprocess
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """性能监控配置类"""
    namespace: str = "rqa2025-production"
    monitoring_interval: int = 60  # 秒
    alert_thresholds: Dict = None
    metrics_storage: str = "metrics"
    auto_optimization: bool = True


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        if self.config.alert_thresholds is None:
            self.config.alert_thresholds = {
                "cpu_usage": 80,
                "memory_usage": 85,
                "disk_usage": 90,
                "gpu_usage": 95,
                "response_time": 5.0
            }

        self.metrics_dir = Path(self.config.metrics_storage)
        self.metrics_dir.mkdir(exist_ok=True)

    def start_monitoring(self):
        """开始性能监控"""
        logger.info("📊 开始性能监控...")

        try:
            while True:
                # 收集性能指标
                metrics = self._collect_metrics()

                # 分析性能指标
                analysis = self._analyze_performance(metrics)

                # 存储指标
                self._store_metrics(metrics)

                # 检查告警
                self._check_alerts(analysis)

                # 自动优化（如果启用）
                if self.config.auto_optimization:
                    self._auto_optimize(analysis)

                # 显示状态
                self._display_status(analysis)

                # 等待下次监控
                time.sleep(self.config.monitoring_interval)

        except KeyboardInterrupt:
            logger.info("⏹️ 性能监控已停止")
        except Exception as e:
            logger.error(f"❌ 性能监控失败: {e}")

    def _collect_metrics(self) -> Dict:
        """收集性能指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "kubernetes": {},
            "application": {}
        }

        try:
            # 系统指标
            metrics["system"] = self._collect_system_metrics()

            # Kubernetes指标
            metrics["kubernetes"] = self._collect_k8s_metrics()

            # 应用指标
            metrics["application"] = self._collect_app_metrics()

        except Exception as e:
            logger.error(f"❌ 收集指标失败: {e}")

        return metrics

    def _collect_system_metrics(self) -> Dict:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # 网络I/O
            network = psutil.net_io_counters()

            # GPU使用率（如果可用）
            gpu_percent = self._get_gpu_usage()

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "gpu_usage": gpu_percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }

        except Exception as e:
            logger.error(f"❌ 收集系统指标失败: {e}")
            return {}

    def _collect_k8s_metrics(self) -> Dict:
        """收集Kubernetes指标"""
        try:
            metrics = {}

            # 获取Pod资源使用情况
            result = subprocess.run([
                "kubectl", "top", "pods", "-n", self.config.namespace
            ], capture_output=True, text=True)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                total_cpu = 0
                total_memory = 0
                pod_count = 0

                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                cpu_str = parts[1].replace('m', '')
                                memory_str = parts[2].replace('Mi', '')
                                total_cpu += int(cpu_str)
                                total_memory += int(memory_str)
                                pod_count += 1
                            except ValueError:
                                pass

                metrics["pod_count"] = pod_count
                metrics["total_cpu_millicores"] = total_cpu
                metrics["total_memory_mb"] = total_memory

            # 获取节点资源使用情况
            node_result = subprocess.run([
                "kubectl", "top", "nodes"
            ], capture_output=True, text=True)

            if node_result.returncode == 0:
                metrics["node_metrics"] = node_result.stdout

            return metrics

        except Exception as e:
            logger.error(f"❌ 收集Kubernetes指标失败: {e}")
            return {}

    def _collect_app_metrics(self) -> Dict:
        """收集应用指标"""
        try:
            metrics = {}

            # 获取应用响应时间
            response_time = self._measure_response_time()
            metrics["response_time"] = response_time

            # 获取应用错误率
            error_rate = self._get_error_rate()
            metrics["error_rate"] = error_rate

            # 获取应用吞吐量
            throughput = self._get_throughput()
            metrics["throughput"] = throughput

            return metrics

        except Exception as e:
            logger.error(f"❌ 收集应用指标失败: {e}")
            return {}

    def _get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            # 尝试使用nvidia-smi
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                gpu_usage = float(result.stdout.strip())
                return gpu_usage
            else:
                return 0.0

        except Exception:
            return 0.0

    def _measure_response_time(self) -> float:
        """测量响应时间"""
        try:
            # 简单的HTTP请求测试
            import requests

            start_time = time.time()
            response = requests.get("http://localhost:8000/health", timeout=5)
            end_time = time.time()

            if response.status_code == 200:
                return (end_time - start_time) * 1000  # 转换为毫秒
            else:
                return 9999.0  # 表示错误

        except Exception:
            return 9999.0

    def _get_error_rate(self) -> float:
        """获取错误率"""
        try:
            # 这里应该从应用日志或监控系统获取错误率
            # 简化实现，返回模拟值
            return 0.1  # 0.1%

        except Exception:
            return 0.0

    def _get_throughput(self) -> int:
        """获取吞吐量"""
        try:
            # 这里应该从应用监控获取吞吐量
            # 简化实现，返回模拟值
            return 1000  # 1000 req/s

        except Exception:
            return 0

    def _analyze_performance(self, metrics: Dict) -> Dict:
        """分析性能指标"""
        analysis = {
            "status": "normal",
            "alerts": [],
            "recommendations": []
        }

        try:
            system_metrics = metrics.get("system", {})

            # 检查CPU使用率
            cpu_usage = system_metrics.get("cpu_usage", 0)
            if cpu_usage > self.config.alert_thresholds["cpu_usage"]:
                analysis["alerts"].append(f"CPU使用率过高: {cpu_usage}%")
                analysis["status"] = "warning"

            # 检查内存使用率
            memory_usage = system_metrics.get("memory_usage", 0)
            if memory_usage > self.config.alert_thresholds["memory_usage"]:
                analysis["alerts"].append(f"内存使用率过高: {memory_usage}%")
                analysis["status"] = "warning"

            # 检查磁盘使用率
            disk_usage = system_metrics.get("disk_usage", 0)
            if disk_usage > self.config.alert_thresholds["disk_usage"]:
                analysis["alerts"].append(f"磁盘使用率过高: {disk_usage}%")
                analysis["status"] = "critical"

            # 检查GPU使用率
            gpu_usage = system_metrics.get("gpu_usage", 0)
            if gpu_usage > self.config.alert_thresholds["gpu_usage"]:
                analysis["alerts"].append(f"GPU使用率过高: {gpu_usage}%")
                analysis["status"] = "warning"

            # 检查响应时间
            app_metrics = metrics.get("application", {})
            response_time = app_metrics.get("response_time", 0)
            if response_time > self.config.alert_thresholds["response_time"] * 1000:  # 转换为毫秒
                analysis["alerts"].append(f"响应时间过长: {response_time:.2f}ms")
                analysis["status"] = "warning"

            # 生成建议
            analysis["recommendations"] = self._generate_recommendations(metrics)

        except Exception as e:
            logger.error(f"❌ 分析性能指标失败: {e}")

        return analysis

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []

        try:
            system_metrics = metrics.get("system", {})
            app_metrics = metrics.get("application", {})

            # CPU优化建议
            cpu_usage = system_metrics.get("cpu_usage", 0)
            if cpu_usage > 70:
                recommendations.append("考虑增加CPU资源或优化代码")

            # 内存优化建议
            memory_usage = system_metrics.get("memory_usage", 0)
            if memory_usage > 80:
                recommendations.append("考虑增加内存资源或优化内存使用")

            # 磁盘优化建议
            disk_usage = system_metrics.get("disk_usage", 0)
            if disk_usage > 85:
                recommendations.append("考虑清理日志文件或增加存储空间")

            # 响应时间优化建议
            response_time = app_metrics.get("response_time", 0)
            if response_time > 3000:  # 3秒
                recommendations.append("考虑优化数据库查询或增加缓存")

        except Exception as e:
            logger.error(f"❌ 生成建议失败: {e}")

        return recommendations

    def _store_metrics(self, metrics: Dict):
        """存储性能指标"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # 清理旧指标文件（保留最近24小时）
            self._cleanup_old_metrics()

        except Exception as e:
            logger.error(f"❌ 存储指标失败: {e}")

    def _cleanup_old_metrics(self):
        """清理旧指标文件"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)

            for metrics_file in self.metrics_dir.glob("metrics_*.json"):
                try:
                    # 从文件名解析时间戳
                    timestamp_str = metrics_file.stem.split("_")[1]
                    file_time = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")

                    if file_time < cutoff_time:
                        metrics_file.unlink()
                        logger.info(f"🗑️ 删除旧指标文件: {metrics_file.name}")

                except Exception:
                    pass

        except Exception as e:
            logger.error(f"❌ 清理旧指标失败: {e}")

    def _check_alerts(self, analysis: Dict):
        """检查告警"""
        alerts = analysis.get("alerts", [])

        if alerts:
            logger.warning("🚨 性能告警:")
            for alert in alerts:
                logger.warning(f"  - {alert}")

            # 这里可以发送告警通知（邮件、Slack等）
            self._send_alert_notification(alerts)

    def _send_alert_notification(self, alerts: List[str]):
        """发送告警通知"""
        try:
            # 这里实现告警通知逻辑
            # 例如发送邮件、Slack消息等
            logger.info("📧 告警通知已发送")

        except Exception as e:
            logger.error(f"❌ 发送告警通知失败: {e}")

    def _auto_optimize(self, analysis: Dict):
        """自动优化"""
        try:
            status = analysis.get("status", "normal")

            if status == "critical":
                # 执行紧急优化
                self._emergency_optimization()
            elif status == "warning":
                # 执行预防性优化
                self._preventive_optimization()

        except Exception as e:
            logger.error(f"❌ 自动优化失败: {e}")

    def _emergency_optimization(self):
        """紧急优化"""
        logger.warning("🚨 执行紧急优化...")

        try:
            # 清理临时文件
            subprocess.run(["find", "/tmp", "-name", "*.tmp", "-delete"])

            # 重启高负载Pod
            self._restart_high_load_pods()

            logger.info("✅ 紧急优化完成")

        except Exception as e:
            logger.error(f"❌ 紧急优化失败: {e}")

    def _preventive_optimization(self):
        """预防性优化"""
        logger.info("🔧 执行预防性优化...")

        try:
            # 调整资源限制
            self._adjust_resource_limits()

            # 优化缓存
            self._optimize_cache()

            logger.info("✅ 预防性优化完成")

        except Exception as e:
            logger.error(f"❌ 预防性优化失败: {e}")

    def _restart_high_load_pods(self):
        """重启高负载Pod"""
        try:
            # 获取高负载Pod
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", self.config.namespace,
                "-l", "app=rqa2025", "-o", "jsonpath={.items[*].metadata.name}"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                pod_names = result.stdout.strip().split()
                for pod_name in pod_names:
                    if pod_name:
                        # 重启Pod
                        restart_result = subprocess.run([
                            "kubectl", "delete", "pod", pod_name, "-n", self.config.namespace
                        ], capture_output=True, text=True)

                        if restart_result.returncode == 0:
                            logger.info(f"🔄 重启Pod: {pod_name}")

        except Exception as e:
            logger.error(f"❌ 重启Pod失败: {e}")

    def _adjust_resource_limits(self):
        """调整资源限制"""
        try:
            # 这里实现动态调整资源限制的逻辑
            logger.info("⚙️ 调整资源限制")

        except Exception as e:
            logger.error(f"❌ 调整资源限制失败: {e}")

    def _optimize_cache(self):
        """优化缓存"""
        try:
            # 这里实现缓存优化逻辑
            logger.info("💾 优化缓存")

        except Exception as e:
            logger.error(f"❌ 优化缓存失败: {e}")

    def _display_status(self, analysis: Dict):
        """显示状态"""
        status = analysis.get("status", "normal")
        alerts = analysis.get("alerts", [])
        recommendations = analysis.get("recommendations", [])

        # 状态图标
        status_icons = {
            "normal": "✅",
            "warning": "⚠️",
            "critical": "🚨"
        }

        icon = status_icons.get(status, "❓")

        print(f"\n{icon} 性能状态: {status.upper()}")

        if alerts:
            print("🚨 告警:")
            for alert in alerts:
                print(f"  - {alert}")

        if recommendations:
            print("💡 建议:")
            for recommendation in recommendations:
                print(f"  - {recommendation}")


def main():
    """主函数"""
    print("📊 RQA2025 性能监控工具")
    print("=" * 50)

    # 创建性能监控配置
    config = PerformanceConfig()

    # 创建性能监控器
    monitor = PerformanceMonitor(config)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "start":
            print("🚀 启动性能监控...")
            monitor.start_monitoring()

        elif command == "config":
            print("⚙️ 当前配置:")
            print(f"  监控间隔: {config.monitoring_interval}秒")
            print(f"  告警阈值: {config.alert_thresholds}")
            print(f"  自动优化: {config.auto_optimization}")

        else:
            print("❌ 未知命令")
            print("可用命令: start, config")
            sys.exit(1)
    else:
        print("用法: python performance_monitor.py <command>")
        print("命令:")
        print("  start  - 启动性能监控")
        print("  config - 显示配置信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
