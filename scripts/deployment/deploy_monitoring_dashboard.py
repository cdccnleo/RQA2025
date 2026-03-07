#!/usr/bin/env python3
"""
监控仪表板部署脚本
用于RQA2025项目的监控仪表板自动部署
"""

import argparse
import json
import logging
import subprocess
import time
import requests
from datetime import datetime
from typing import Dict, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringDashboardDeployer:
    """监控仪表板部署器"""

    def __init__(self, namespace: str = "monitoring"):
        self.namespace = namespace
        self.grafana_url = "http://grafana-service:3000"
        self.prometheus_url = "http://prometheus-service:9090"

    def deploy_prometheus(self) -> bool:
        """部署Prometheus"""
        try:
            logger.info("🚀 开始部署Prometheus...")

            # 创建命名空间
            subprocess.run([
                "kubectl", "create", "namespace", self.namespace, "--dry-run=client", "-o", "yaml"
            ], check=True)
            subprocess.run([
                "kubectl", "apply", "-f", "-"
            ], input=subprocess.run([
                "kubectl", "create", "namespace", self.namespace, "--dry-run=client", "-o", "yaml"
            ], capture_output=True, text=True, check=True).stdout, text=True, check=True)

            # 部署Prometheus配置
            subprocess.run([
                "kubectl", "apply", "-f", "deploy/monitoring/prometheus.yml",
                "-n", self.namespace
            ], check=True)

            # 等待Prometheus启动
            logger.info("⏳ 等待Prometheus启动...")
            time.sleep(30)

            # 检查Prometheus状态
            if self._check_prometheus_health():
                logger.info("✅ Prometheus部署成功")
                return True
            else:
                logger.error("❌ Prometheus部署失败")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Prometheus部署失败: {e}")
            return False

    def deploy_grafana(self) -> bool:
        """部署Grafana"""
        try:
            logger.info("🚀 开始部署Grafana...")

            # 部署Grafana数据源配置
            subprocess.run([
                "kubectl", "apply", "-f", "deploy/monitoring/grafana-datasources.yml",
                "-n", self.namespace
            ], check=True)

            # 部署Grafana
            subprocess.run([
                "kubectl", "apply", "-f", "deploy/monitoring/grafana.yml",
                "-n", self.namespace
            ], check=True)

            # 等待Grafana启动
            logger.info("⏳ 等待Grafana启动...")
            time.sleep(60)

            # 检查Grafana状态
            if self._check_grafana_health():
                logger.info("✅ Grafana部署成功")
                return True
            else:
                logger.error("❌ Grafana部署失败")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Grafana部署失败: {e}")
            return False

    def deploy_dashboards(self) -> bool:
        """部署仪表板"""
        try:
            logger.info("🚀 开始部署监控仪表板...")

            # 获取Grafana API Token
            token = self._get_grafana_token()
            if not token:
                logger.error("❌ 无法获取Grafana API Token")
                return False

            # 部署综合监控仪表板
            dashboard_path = "dashboards/comprehensive_monitoring_dashboard.json"
            if self._deploy_dashboard(dashboard_path, token):
                logger.info("✅ 综合监控仪表板部署成功")
            else:
                logger.error("❌ 综合监控仪表板部署失败")
                return False

            # 部署自动化仪表板
            automation_dashboard_path = "dashboards/automation_dashboard.json"
            if self._deploy_dashboard(automation_dashboard_path, token):
                logger.info("✅ 自动化仪表板部署成功")
            else:
                logger.error("❌ 自动化仪表板部署失败")
                return False

            logger.info("✅ 所有仪表板部署完成")
            return True

        except Exception as e:
            logger.error(f"❌ 仪表板部署失败: {e}")
            return False

    def deploy_alert_rules(self) -> bool:
        """部署告警规则"""
        try:
            logger.info("🚀 开始部署告警规则...")

            # 部署Prometheus告警规则
            subprocess.run([
                "kubectl", "apply", "-f", "deploy/monitoring/alert_rules.yml",
                "-n", self.namespace
            ], check=True)

            # 重启Prometheus以加载新规则
            subprocess.run([
                "kubectl", "rollout", "restart", "deployment/prometheus",
                "-n", self.namespace
            ], check=True)

            logger.info("✅ 告警规则部署成功")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 告警规则部署失败: {e}")
            return False

    def _check_prometheus_health(self) -> bool:
        """检查Prometheus健康状态"""
        try:
            # 检查Pod状态
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.namespace,
                "-l", "app=prometheus",
                "-o", "jsonpath={.items[*].status.phase}"
            ], capture_output=True, text=True, check=True)

            pod_statuses = result.stdout.strip().split()
            if not pod_statuses:
                return False

            # 检查所有Pod是否都是Running状态
            running_pods = [status for status in pod_statuses if status == "Running"]
            return len(running_pods) == len(pod_statuses)

        except subprocess.CalledProcessError:
            return False

    def _check_grafana_health(self) -> bool:
        """检查Grafana健康状态"""
        try:
            # 检查Pod状态
            result = subprocess.run([
                "kubectl", "get", "pods",
                "-n", self.namespace,
                "-l", "app=grafana",
                "-o", "jsonpath={.items[*].status.phase}"
            ], capture_output=True, text=True, check=True)

            pod_statuses = result.stdout.strip().split()
            if not pod_statuses:
                return False

            # 检查所有Pod是否都是Running状态
            running_pods = [status for status in pod_statuses if status == "Running"]
            return len(running_pods) == len(pod_statuses)

        except subprocess.CalledProcessError:
            return False

    def _get_grafana_token(self) -> Optional[str]:
        """获取Grafana API Token"""
        try:
            # 这里应该从Kubernetes Secret或环境变量获取
            # 简化实现，实际应该从安全存储获取
            return "admin:admin"  # 默认用户名密码

        except Exception as e:
            logger.error(f"获取Grafana Token失败: {e}")
            return None

    def _deploy_dashboard(self, dashboard_path: str, token: str) -> bool:
        """部署单个仪表板"""
        try:
            # 读取仪表板配置
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                dashboard_config = json.load(f)

            # 准备API请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Basic {token}"
            }

            # 发送到Grafana API
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers=headers,
                json=dashboard_config,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"✅ 仪表板 {dashboard_path} 部署成功")
                return True
            else:
                logger.error(f"❌ 仪表板 {dashboard_path} 部署失败: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ 仪表板 {dashboard_path} 部署失败: {e}")
            return False

    def get_monitoring_status(self) -> Dict:
        """获取监控状态"""
        try:
            status = {
                "prometheus": {
                    "health": self._check_prometheus_health(),
                    "url": self.prometheus_url
                },
                "grafana": {
                    "health": self._check_grafana_health(),
                    "url": self.grafana_url
                },
                "dashboards": [
                    "comprehensive_monitoring_dashboard.json",
                    "automation_dashboard.json"
                ],
                "alerts": "alert_rules.yml",
                "timestamp": datetime.now().isoformat()
            }
            return status
        except Exception as e:
            logger.error(f"获取监控状态失败: {e}")
            return {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 监控仪表板部署器")
    parser.add_argument("--namespace", default="monitoring", help="Kubernetes命名空间")
    parser.add_argument("--action", required=True,
                        choices=["deploy-all", "deploy-prometheus", "deploy-grafana",
                                 "deploy-dashboards", "deploy-alerts", "status"],
                        help="执行操作")

    args = parser.parse_args()

    # 初始化部署器
    deployer = MonitoringDashboardDeployer(args.namespace)

    try:
        if args.action == "deploy-all":
            # 部署所有组件
            success = True
            success &= deployer.deploy_prometheus()
            success &= deployer.deploy_grafana()
            success &= deployer.deploy_dashboards()
            success &= deployer.deploy_alert_rules()

            if success:
                logger.info("🎉 所有监控组件部署完成")
                return 0
            else:
                logger.error("❌ 部分监控组件部署失败")
                return 1

        elif args.action == "deploy-prometheus":
            success = deployer.deploy_prometheus()
            return 0 if success else 1

        elif args.action == "deploy-grafana":
            success = deployer.deploy_grafana()
            return 0 if success else 1

        elif args.action == "deploy-dashboards":
            success = deployer.deploy_dashboards()
            return 0 if success else 1

        elif args.action == "deploy-alerts":
            success = deployer.deploy_alert_rules()
            return 0 if success else 1

        elif args.action == "status":
            status = deployer.get_monitoring_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0

    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
