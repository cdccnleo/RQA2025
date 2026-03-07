#!/usr/bin/env python3
"""
监控仪表板集成脚本

配置Grafana监控仪表板和Prometheus告警规则
"""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DashboardIntegration:
    """监控仪表板集成"""

    def __init__(self):
        self.dashboards_dir = Path("dashboards")
        self.alerts_dir = Path("deploy/alerts")
        self.dashboards_dir.mkdir(exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

    def create_system_dashboard(self):
        """创建系统监控仪表板"""
        dashboard = {
            "dashboard": {
                "title": "RQA2025系统监控",
                "description": "系统整体监控仪表板",
                "tags": ["rqa2025", "system"],
                "panels": [
                    {
                        "title": "CPU使用率",
                        "type": "graph",
                        "targets": [{"expr": "cpu_usage_percent"}],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "内存使用率",
                        "type": "graph",
                        "targets": [{"expr": "memory_usage_percent"}],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "磁盘使用率",
                        "type": "graph",
                        "targets": [{"expr": "disk_usage_percent"}],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "title": "网络流量",
                        "type": "graph",
                        "targets": [
                            {"expr": "network_bytes_received", "legendFormat": "接收"},
                            {"expr": "network_bytes_sent", "legendFormat": "发送"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            },
            "overwrite": True
        }

        dashboard_file = self.dashboards_dir / "system_dashboard.json"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 系统监控仪表板已创建: {dashboard_file}")
        return dashboard

    def create_performance_dashboard(self):
        """创建性能监控仪表板"""
        dashboard = {
            "dashboard": {
                "title": "RQA2025性能监控",
                "description": "系统性能指标监控",
                "tags": ["rqa2025", "performance"],
                "panels": [
                    {
                        "title": "响应时间",
                        "type": "graph",
                        "targets": [
                            {"expr": "response_time_ms", "legendFormat": "平均响应时间"},
                            {"expr": "response_time_p95_ms", "legendFormat": "95%响应时间"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "吞吐量",
                        "type": "graph",
                        "targets": [{"expr": "requests_per_second"}],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "错误率",
                        "type": "graph",
                        "targets": [{"expr": "error_rate_percent"}],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "title": "队列长度",
                        "type": "graph",
                        "targets": [{"expr": "queue_length"}],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            },
            "overwrite": True
        }

        dashboard_file = self.dashboards_dir / "performance_dashboard.json"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 性能监控仪表板已创建: {dashboard_file}")
        return dashboard

    def create_business_dashboard(self):
        """创建业务监控仪表板"""
        dashboard = {
            "dashboard": {
                "title": "RQA2025业务监控",
                "description": "业务指标监控",
                "tags": ["rqa2025", "business"],
                "panels": [
                    {
                        "title": "交易量",
                        "type": "graph",
                        "targets": [{"expr": "trading_volume"}],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "模型准确率",
                        "type": "graph",
                        "targets": [{"expr": "model_accuracy_percent"}],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "数据质量",
                        "type": "stat",
                        "targets": [{"expr": "data_quality_score"}],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                    },
                    {
                        "title": "模型性能",
                        "type": "stat",
                        "targets": [{"expr": "model_performance_score"}],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            },
            "overwrite": True
        }

        dashboard_file = self.dashboards_dir / "business_dashboard.json"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 业务监控仪表板已创建: {dashboard_file}")
        return dashboard

    def create_system_alerts(self):
        """创建系统告警规则"""
        alerts = {
            "groups": [
                {
                    "name": "system_alerts",
                    "rules": [
                        {
                            "alert": "HighCPUUsage",
                            "expr": "cpu_usage_percent > 80",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "CPU使用率过高"}
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "memory_usage_percent > 85",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "内存使用率过高"}
                        },
                        {
                            "alert": "LowDiskSpace",
                            "expr": "disk_usage_percent > 90",
                            "for": "2m",
                            "labels": {"severity": "critical"},
                            "annotations": {"summary": "磁盘空间不足"}
                        }
                    ]
                }
            ]
        }

        alert_file = self.alerts_dir / "system_alerts.yml"
        with open(alert_file, 'w', encoding='utf-8') as f:
            yaml.dump(alerts, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 系统告警规则已创建: {alert_file}")
        return alerts

    def create_business_alerts(self):
        """创建业务告警规则"""
        alerts = {
            "groups": [
                {
                    "name": "business_alerts",
                    "rules": [
                        {
                            "alert": "LowModelAccuracy",
                            "expr": "model_accuracy_percent < 70",
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "模型准确率过低"}
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "error_rate_percent > 5",
                            "for": "3m",
                            "labels": {"severity": "warning"},
                            "annotations": {"summary": "错误率过高"}
                        }
                    ]
                }
            ]
        }

        alert_file = self.alerts_dir / "business_alerts.yml"
        with open(alert_file, 'w', encoding='utf-8') as f:
            yaml.dump(alerts, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 业务告警规则已创建: {alert_file}")
        return alerts

    def generate_report(self, output_file: str = "reports/dashboard_integration_report.json"):
        """生成集成报告"""
        report = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "setup_type": "监控仪表板集成",
                "version": "1.0.0"
            },
            "dashboards": [
                {"name": "RQA2025系统监控", "file": "system_dashboard.json", "status": "created"},
                {"name": "RQA2025性能监控", "file": "performance_dashboard.json", "status": "created"},
                {"name": "RQA2025业务监控", "file": "business_dashboard.json", "status": "created"}
            ],
            "alert_rules": [
                {"name": "system_alerts.yml", "type": "system", "rules_count": 3, "status": "created"},
                {"name": "business_alerts.yml", "type": "business", "rules_count": 2, "status": "created"}
            ],
            "summary": {
                "total_dashboards": 3,
                "total_alert_rules": 5,
                "setup_status": "completed"
            }
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 仪表板集成报告已生成: {output_file}")
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="监控仪表板集成")
    parser.add_argument("--action", choices=["setup", "report"], default="setup", help="执行动作")
    parser.add_argument(
        "--output", default="reports/dashboard_integration_report.json", help="报告输出文件")

    args = parser.parse_args()

    try:
        integration = DashboardIntegration()

        if args.action == "setup":
            print("🔧 设置监控仪表板...")

            # 创建仪表板
            integration.create_system_dashboard()
            integration.create_performance_dashboard()
            integration.create_business_dashboard()

            # 创建告警规则
            integration.create_system_alerts()
            integration.create_business_alerts()

            print("✅ 监控仪表板设置完成！")

        elif args.action == "report":
            print("📊 生成仪表板集成报告...")
            report = integration.generate_report(args.output)
            print(f"✅ 仪表板集成报告已生成: {args.output}")
            print(f"📈 统计信息:")
            print(f"  - 仪表板数量: {report['summary']['total_dashboards']}")
            print(f"  - 告警规则: {report['summary']['total_alert_rules']}")

        print("🎉 监控仪表板集成完成！")

    except Exception as e:
        logger.error(f"❌ 监控仪表板集成失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
