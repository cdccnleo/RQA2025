#!/usr/bin/env python3
"""
RQA2025 灰度发布监控脚本

实时监控灰度发布状态：
- 容器健康状态
- 应用性能指标
- 错误率统计
- 资源使用情况
- 自动告警

使用方法：
python canary_monitor.py --version v1.2.3 --duration 300
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CanaryMonitor:
    """灰度发布监控器"""

    def __init__(self, config_file: str = "canary_config.json"):
        self.config_file = Path(config_file)
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config()
        self.monitoring_data = []

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")

    def monitor_deployment(self, version: str, duration: int = 300) -> bool:
        """监控部署状态"""
        logger.info(f"🔍 开始监控灰度发布: {version}，持续时间: {duration}秒")

        start_time = time.time()
        alert_count = 0

        print("
📊 实时监控面板"        print("=" * 80)
        print("<15" print("-" * 80)

        while time.time() - start_time < duration:
            try:
                # 收集监控数据
                metrics=self._collect_all_metrics()

                # 显示实时状态
                self._display_status(metrics, version)

                # 检查告警条件
                alerts=self._check_alerts(metrics)
                if alerts:
                    alert_count += len(alerts)
                    self._handle_alerts(alerts, version)

                # 保存监控数据
                self.monitoring_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics,
                    "alerts": alerts
                })

                time.sleep(10)  # 10秒更新一次

            except KeyboardInterrupt:
                logger.info("监控被用户中断")
                break
            except Exception as e:
                logger.error(f"监控异常: {e}")
                continue

        # 生成监控报告
        self._generate_report(version, alert_count)
        logger.info("✅ 监控完成")
        return alert_count == 0

    def _collect_all_metrics(self) -> Dict[str, Any]:
        """收集所有监控指标"""
        metrics={
            "timestamp": datetime.now().isoformat(),
            "containers": self._get_container_status(),
            "application": self._get_application_metrics(),
            "system": self._get_system_metrics(),
            "traffic": self._get_traffic_distribution()
        }

        return metrics

    def _get_container_status(self) -> Dict[str, Any]:
        """获取容器状态"""
        status={}

        try:
            # 获取所有容器状态
            result=subprocess.run(
                "docker ps --format json",
                shell=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                containers=[json.loads(line) for line in result.stdout.strip().split('\n') if line]

                for container in containers:
                    name=container.get('Names', '').replace('rqa2025-', '')
                    status[name]={
                        "status": container.get('Status', ''),
                        "ports": container.get('Ports', ''),
                        "running": 'Up' in container.get('Status', '')
                    }

        except Exception as e:
            logger.warning(f"获取容器状态失败: {e}")

        return status

    def _get_application_metrics(self) -> Dict[str, Any]:
        """获取应用指标"""
        metrics={
            "health": "unknown",
            "version": "unknown",
            "response_time": 0,
            "error_rate": 0
        }

        try:
            # 健康检查
            response=requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                health_data=response.json()
                metrics["health"]=health_data.get("status", "unknown")
                metrics["version"]=health_data.get("version", "unknown")

            # 响应时间
            start_time=time.time()
            requests.get("http://localhost:8000/", timeout=5)
            metrics["response_time"]=int((time.time() - start_time) * 1000)

        except Exception as e:
            logger.warning(f"获取应用指标失败: {e}")
            metrics["health"]="error"

        return metrics

    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        metrics={
            "cpu_percent": 0,
            "memory_percent": 0,
            "disk_usage": 0,
            "network_io": 0
        }

        try:
            # 从Prometheus获取系统指标
            prometheus_url=self.config["monitoring"]["prometheus_url"]

            # CPU使用率
            response=requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"})
            if response.status_code == 200:
                data=response.json()
                if data["data"]["result"]:
                    metrics["cpu_percent"]=float(data["data"]["result"][0]["value"][1])

            # 内存使用率
            response=requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100"})
            if response.status_code == 200:
                data=response.json()
                if data["data"]["result"]:
                    metrics["memory_percent"]=float(data["data"]["result"][0]["value"][1])

        except Exception as e:
            logger.warning(f"获取系统指标失败: {e}")

        return metrics

    def _get_traffic_distribution(self) -> Dict[str, Any]:
        """获取流量分布"""
        traffic={
            "stable_traffic": 0,
            "canary_traffic": 0,
            "total_traffic": 0
        }

        try:
            # 从Prometheus获取流量指标
            prometheus_url=self.config["monitoring"]["prometheus_url"]

            # 稳定版本流量
            response=requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "rate(http_requests_total{version=\"stable\"}[5m])"})
            if response.status_code == 200:
                data=response.json()
                if data["data"]["result"]:
                    traffic["stable_traffic"]=float(data["data"]["result"][0]["value"][1])

            # 金丝雀版本流量
            response=requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": "rate(http_requests_total{version=\"canary\"}[5m])"})
            if response.status_code == 200:
                data=response.json()
                if data["data"]["result"]:
                    traffic["canary_traffic"]=float(data["data"]["result"][0]["value"][1])

            traffic["total_traffic"]=traffic["stable_traffic"] + traffic["canary_traffic"]

        except Exception as e:
            logger.warning(f"获取流量分布失败: {e}")

        return traffic

    def _display_status(self, metrics: Dict[str, Any], version: str):
        """显示实时状态"""
        # 清除屏幕
        print("\033[2J\033[H", end="")

        print(f"🔍 RQA2025 灰度发布监控 - 版本: {version}")
        print("=" * 80)

        # 容器状态
        print("🏗️  容器状态:"        containers=metrics.get("containers", {})
        for name, info in containers.items():
            status_icon="✅" if info.get("running", False) else "❌"
            print(f"   {status_icon} {name}: {info.get('status', 'unknown')}")
        print()

        # 应用状态
        print("🚀 应用状态:"        app=metrics.get("application", {})
        health_icon="✅" if app.get("health") == "healthy" else "❌"
        print(f"   {health_icon} 健康状态: {app.get('health', 'unknown')}")
        print(f"   📦 版本: {app.get('version', 'unknown')}")
        print(f"   ⚡ 响应时间: {app.get('response_time', 0)}ms")
        print()

        # 系统资源
        print("💻 系统资源:"        system=metrics.get("system", {})
        print(f"   🖥️  CPU使用率: {system.get('cpu_percent', 0):.1f}%")
        print(f"   🧠 内存使用率: {system.get('memory_percent', 0):.1f}%")
        print()

        # 流量分布
        print("🌐 流量分布:"        traffic=metrics.get("traffic", {})
        total=traffic.get("total_traffic", 1)
        stable_pct=(traffic.get("stable_traffic", 0) / total * 100) if total > 0 else 0
        canary_pct=(traffic.get("canary_traffic", 0) / total * 100) if total > 0 else 0

        print(f"   📊 稳定版本: {stable_pct:.1f}% ({traffic.get('stable_traffic', 0):.1f} req/s)")
        print(f"   🐦 金丝雀版本: {canary_pct:.1f}% ({traffic.get('canary_traffic', 0):.1f} req/s)")
        print(f"   📈 总流量: {traffic.get('total_traffic', 0):.1f} req/s")
        print()

        # 时间戳
        print(f"🕐 最后更新: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 80)

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts=[]
        thresholds=self.config["metrics"]

        # 应用健康检查
        app=metrics.get("application", {})
        if app.get("health") != "healthy":
            alerts.append({
                "level": "CRITICAL",
                "message": f"应用健康状态异常: {app.get('health')}",
                "metric": "application_health"
            })

        # 响应时间检查
        if app.get("response_time", 0) > thresholds["response_time_threshold"]:
            alerts.append({
                "level": "WARNING",
                "message".1f"                "metric": "response_time"
            })

        # 系统资源检查
        system=metrics.get("system", {})
        if system.get("cpu_percent", 0) > thresholds["cpu_usage_threshold"]:
            alerts.append({
                "level": "WARNING",
                "message": ".1f"                "metric": "cpu_usage"
            })

        if system.get("memory_percent", 0) > thresholds["memory_usage_threshold"]:
            alerts.append({
                "level": "WARNING",
                "message": ".1f"                "metric": "memory_usage"
            })

        # 错误率检查
        if app.get("error_rate", 0) > thresholds["error_rate_threshold"]:
            alerts.append({
                "level": "CRITICAL",
                "message": ".2f"                "metric": "error_rate"
            })

        return alerts

    def _handle_alerts(self, alerts: List[Dict[str, Any]], version: str):
        """处理告警"""
        for alert in alerts:
            level=alert["level"]
            message=alert["message"]

            if level == "CRITICAL":
                logger.error(f"🚨 CRITICAL: {message}")
                print(f"\033[91m🚨 CRITICAL: {message}\033[0m")  # 红色
            elif level == "WARNING":
                logger.warning(f"⚠️  WARNING: {message}")
                print(f"\033[93m⚠️  WARNING: {message}\033[0m")  # 黄色

    def _generate_report(self, version: str, alert_count: int):
        """生成监控报告"""
        report_file=self.project_root / f"canary_monitor_report_{version}_{int(time.time())}.json"

        report={
            "version": version,
            "monitoring_period": {
                "start": self.monitoring_data[0]["timestamp"] if self.monitoring_data else None,
                "end": self.monitoring_data[-1]["timestamp"] if self.monitoring_data else None,
                "duration_seconds": len(self.monitoring_data) * 10
            },
            "summary": {
                "total_alerts": alert_count,
                "data_points": len(self.monitoring_data),
                "status": "PASSED" if alert_count == 0 else "FAILED"
            },
            "monitoring_data": self.monitoring_data
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📋 监控报告已保存: {report_file}")

        # 控制台报告摘要
        print("
📋 监控报告摘要"        print("=" * 40)
        print(f"版本: {version}")
        print(f"监控时长: {len(self.monitoring_data) * 10} 秒")
        print(f"数据点数: {len(self.monitoring_data)}")
        print(f"告警次数: {alert_count}")
        print(f"状态: {'✅ 通过' if alert_count == 0 else '❌ 失败'}")
        print(f"报告文件: {report_file}")


def main():
    """主函数"""
    parser=argparse.ArgumentParser(description="RQA2025 灰度发布监控")
    parser.add_argument("--version", required=True, help="监控的版本号")
    parser.add_argument("--duration", type=int, default=300, help="监控持续时间(秒)")
    parser.add_argument("--config", default="canary_config.json", help="配置文件")

    args=parser.parse_args()

    try:
        monitor=CanaryMonitor(args.config)
        success=monitor.monitor_deployment(args.version, args.duration)

        if success:
            logger.info("✅ 监控完成，无严重问题")
            sys.exit(0)
        else:
            logger.error("❌ 监控完成，发现问题")
            sys.exit(1)

    except Exception as e:
        logger.error(f"监控失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
