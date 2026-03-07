#!/usr/bin/env python3
"""
RQA2025 监控集成配置脚本

用于集成Prometheus等监控系统，包括：
- Prometheus配置
- Grafana仪表板
- 应用指标收集
- 告警规则配置
"""

import json
import logging
from pathlib import Path
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringIntegrationSetup:
    """监控集成配置器"""

    def __init__(self, config_dir: str = "config/monitoring"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def setup_prometheus_config(self) -> bool:
        """配置Prometheus"""
        logger.info("🔧 配置Prometheus...")

        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "rules/*.yml"
            ],
            "scrape_configs": [
                {
                    "job_name": "rqa2025-data-layer",
                    "static_configs": [
                        {
                            "targets": ["localhost:8080"],
                            "labels": {
                                "service": "data-layer",
                                "environment": "production"
                            }
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_interval": "10s"
                },
                {
                    "job_name": "rqa2025-cache",
                    "static_configs": [
                        {
                            "targets": ["localhost:6379"],
                            "labels": {
                                "service": "redis-cache",
                                "environment": "production"
                            }
                        }
                    ],
                    "scrape_interval": "15s"
                }
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {
                                "targets": ["localhost:9093"]
                            }
                        ]
                    }
                ]
            }
        }

        config_path = self.config_dir / "prometheus.yml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)

        logger.info(f"✅ Prometheus配置已生成: {config_path}")
        return True

    def setup_grafana_dashboards(self) -> bool:
        """配置Grafana仪表板"""
        logger.info("🔧 配置Grafana仪表板...")

        # 数据层性能仪表板
        data_layer_dashboard = {
            "dashboard": {
                "id": None,
                "title": "RQA2025 数据层性能监控",
                "tags": ["rqa2025", "data-layer"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "数据加载性能",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "data_load_time_seconds",
                                "legendFormat": "加载时间"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "缓存命中率",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "cache_hit_rate",
                                "legendFormat": "命中率"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "错误率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "error_rate",
                                "legendFormat": "错误率"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "资源使用",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "cpu_usage_percent",
                                "legendFormat": "CPU使用率"
                            },
                            {
                                "expr": "memory_usage_percent",
                                "legendFormat": "内存使用率"
                            }
                        ]
                    }
                ]
            }
        }

        dashboard_path = self.config_dir / "grafana_dashboards" / "data_layer_performance.json"
        dashboard_path.parent.mkdir(exist_ok=True)

        with open(dashboard_path, 'w', encoding='utf-8') as f:
            json.dump(data_layer_dashboard, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Grafana仪表板已生成: {dashboard_path}")
        return True

    def setup_alert_rules(self) -> bool:
        """配置告警规则"""
        logger.info("🔧 配置告警规则...")

        alert_rules = {
            "groups": [
                {
                    "name": "data_layer_alerts",
                    "rules": [
                        {
                            "alert": "DataLoadTimeHigh",
                            "expr": "data_load_time_seconds > 5",
                            "for": "2m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "数据加载时间过长",
                                "description": "数据加载时间超过5秒"
                            }
                        },
                        {
                            "alert": "CacheHitRateLow",
                            "expr": "cache_hit_rate < 0.8",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "缓存命中率过低",
                                "description": "缓存命中率低于80%"
                            }
                        },
                        {
                            "alert": "ErrorRateHigh",
                            "expr": "error_rate > 0.01",
                            "for": "2m",
                            "labels": {
                                "severity": "critical"
                            },
                            "annotations": {
                                "summary": "错误率过高",
                                "description": "错误率超过1%"
                            }
                        },
                        {
                            "alert": "CPUUsageHigh",
                            "expr": "cpu_usage_percent > 80",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "CPU使用率过高",
                                "description": "CPU使用率超过80%"
                            }
                        },
                        {
                            "alert": "MemoryUsageHigh",
                            "expr": "memory_usage_percent > 85",
                            "for": "5m",
                            "labels": {
                                "severity": "warning"
                            },
                            "annotations": {
                                "summary": "内存使用率过高",
                                "description": "内存使用率超过85%"
                            }
                        }
                    ]
                }
            ]
        }

        rules_path = self.config_dir / "rules" / "data_layer_alerts.yml"
        rules_path.parent.mkdir(exist_ok=True)

        with open(rules_path, 'w', encoding='utf-8') as f:
            yaml.dump(alert_rules, f, default_flow_style=False)

        logger.info(f"✅ 告警规则已生成: {rules_path}")
        return True

    def setup_application_metrics(self) -> bool:
        """配置应用指标收集"""
        logger.info("🔧 配置应用指标收集...")

        # 创建指标收集器配置
        metrics_config = {
            "metrics": {
                "enabled": True,
                "port": 8080,
                "path": "/metrics",
                "collectors": [
                    "data_load_time",
                    "cache_hit_rate",
                    "error_rate",
                    "cpu_usage",
                    "memory_usage",
                    "disk_usage"
                ]
            },
            "exporters": {
                "prometheus": {
                    "enabled": True,
                    "endpoint": "localhost:9090"
                }
            }
        }

        metrics_path = self.config_dir / "application_metrics.yml"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            yaml.dump(metrics_config, f, default_flow_style=False)

        logger.info(f"✅ 应用指标配置已生成: {metrics_path}")
        return True

    def create_docker_compose_monitoring(self) -> bool:
        """创建监控服务的Docker Compose配置"""
        logger.info("🔧 创建监控服务Docker Compose配置...")

        docker_compose = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": "rqa2025-prometheus",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./config/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "./config/monitoring/rules:/etc/prometheus/rules",
                        "prometheus_data:/prometheus"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--storage.tsdb.retention.time=200h",
                        "--web.enable-lifecycle"
                    ]
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": "rqa2025-grafana",
                    "ports": ["3000:3000"],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "admin123"
                    },
                    "volumes": [
                        "./config/monitoring/grafana_dashboards:/etc/grafana/provisioning/dashboards",
                        "grafana_data:/var/lib/grafana"
                    ]
                },
                "alertmanager": {
                    "image": "prom/alertmanager:latest",
                    "container_name": "rqa2025-alertmanager",
                    "ports": ["9093:9093"],
                    "volumes": [
                        "./config/monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml"
                    ],
                    "command": [
                        "--config.file=/etc/alertmanager/alertmanager.yml",
                        "--storage.path=/alertmanager"
                    ]
                }
            },
            "volumes": {
                "prometheus_data": {},
                "grafana_data": {}
            }
        }

        compose_path = Path("docker-compose.monitoring.yml")
        with open(compose_path, 'w', encoding='utf-8') as f:
            yaml.dump(docker_compose, f, default_flow_style=False)

        logger.info(f"✅ Docker Compose监控配置已生成: {compose_path}")
        return True

    def create_alertmanager_config(self) -> bool:
        """创建AlertManager配置"""
        logger.info("🔧 创建AlertManager配置...")

        alertmanager_config = {
            "global": {
                "smtp_smarthost": "localhost:587",
                "smtp_from": "alertmanager@rqa2025.com"
            },
            "route": {
                "group_by": ["alertname"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "web.hook"
            },
            "receivers": [
                {
                    "name": "web.hook",
                    "webhook_configs": [
                        {
                            "url": "http://localhost:5001/"
                        }
                    ]
                }
            ]
        }

        alertmanager_path = self.config_dir / "alertmanager.yml"
        with open(alertmanager_path, 'w', encoding='utf-8') as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False)

        logger.info(f"✅ AlertManager配置已生成: {alertmanager_path}")
        return True

    def setup_monitoring_scripts(self) -> bool:
        """创建监控管理脚本"""
        logger.info("🔧 创建监控管理脚本...")

        # 启动监控脚本
        start_script = """#!/bin/bash
# 启动监控服务
echo "启动RQA2025监控服务..."

# 启动Prometheus
docker-compose -f docker-compose.monitoring.yml up -d prometheus
echo "Prometheus已启动: http://localhost:9090"

# 启动Grafana
docker-compose -f docker-compose.monitoring.yml up -d grafana
echo "Grafana已启动: http://localhost:3000 (admin/admin123)"

# 启动AlertManager
docker-compose -f docker-compose.monitoring.yml up -d alertmanager
echo "AlertManager已启动: http://localhost:9093"

echo "监控服务启动完成！"
"""

        start_path = Path("scripts/monitoring/start_monitoring.sh")
        start_path.parent.mkdir(exist_ok=True)

        with open(start_path, 'w', encoding='utf-8') as f:
            f.write(start_script)

        # 停止监控脚本
        stop_script = """#!/bin/bash
# 停止监控服务
echo "停止RQA2025监控服务..."

docker-compose -f docker-compose.monitoring.yml down

echo "监控服务已停止！"
"""

        stop_path = Path("scripts/monitoring/stop_monitoring.sh")
        with open(stop_path, 'w', encoding='utf-8') as f:
            f.write(stop_script)

        logger.info("✅ 监控管理脚本已生成")
        return True

    def run_setup(self) -> bool:
        """运行完整设置"""
        logger.info("🚀 开始监控集成设置...")

        try:
            # 执行所有配置步骤
            steps = [
                ("Prometheus配置", self.setup_prometheus_config),
                ("Grafana仪表板", self.setup_grafana_dashboards),
                ("告警规则", self.setup_alert_rules),
                ("应用指标", self.setup_application_metrics),
                ("Docker Compose", self.create_docker_compose_monitoring),
                ("AlertManager", self.create_alertmanager_config),
                ("管理脚本", self.setup_monitoring_scripts),
            ]

            for step_name, step_func in steps:
                logger.info(f"📋 执行步骤: {step_name}")
                if not step_func():
                    logger.error(f"❌ 步骤失败: {step_name}")
                    return False
                logger.info(f"✅ 步骤完成: {step_name}")

            logger.info("🎉 监控集成设置完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 监控集成设置失败: {e}")
            return False


def main():
    """主函数"""
    setup = MonitoringIntegrationSetup()

    if setup.run_setup():
        print("\n" + "="*60)
        print("✅ 监控集成配置完成！")
        print("="*60)
        print("\n📋 配置项目:")
        print("  - Prometheus配置: config/monitoring/prometheus.yml")
        print("  - Grafana仪表板: config/monitoring/grafana_dashboards/")
        print("  - 告警规则: config/monitoring/rules/")
        print("  - 应用指标: config/monitoring/application_metrics.yml")
        print("  - Docker Compose: docker-compose.monitoring.yml")
        print("  - AlertManager: config/monitoring/alertmanager.yml")
        print("  - 管理脚本: scripts/monitoring/")

        print("\n🚀 启动监控服务:")
        print("  ./scripts/monitoring/start_monitoring.sh")

        print("\n🛑 停止监控服务:")
        print("  ./scripts/monitoring/stop_monitoring.sh")

        print("\n📊 访问地址:")
        print("  - Prometheus: http://localhost:9090")
        print("  - Grafana: http://localhost:3000 (admin/admin123)")
        print("  - AlertManager: http://localhost:9093")

        print("\n" + "="*60)
    else:
        print("❌ 监控集成配置失败！")
        exit(1)


if __name__ == "__main__":
    main()
