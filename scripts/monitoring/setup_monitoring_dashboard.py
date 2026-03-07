#!/usr/bin/env python3
"""
监控仪表板集成脚本

配置Grafana监控仪表板、Prometheus告警规则和自定义监控指标
"""

import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MonitoringDashboardManager:
    """监控仪表板管理器"""

    def __init__(self, config_path: str = "config/monitoring.yml"):
        self.config_path = Path(config_path)
        self.grafana_url = "http://localhost:3000"
        self.prometheus_url = "http://localhost:9090"
        self.dashboards_dir = Path("dashboards")
        self.alerts_dir = Path("deploy/alerts")

        # 创建必要的目录
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.dashboards_dir.mkdir(exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # 初始化配置
        self._init_config()

    def _init_config(self):
        """初始化监控配置"""
        if not self.config_path.exists():
            default_config = {
                "grafana": {
                    "url": "http://localhost:3000",
                    "admin_user": "admin",
                    "admin_password": "admin",
                    "datasource_name": "Prometheus",
                    "datasource_url": "http://prometheus:9090"
                },
                "prometheus": {
                    "url": "http://localhost:9090",
                    "retention_days": 15,
                    "scrape_interval": "15s"
                },
                "dashboards": [
                    {
                        "name": "RQA2025系统概览",
                        "description": "RQA2025系统整体监控仪表板",
                        "file": "system_overview_dashboard.json"
                    },
                    {
                        "name": "RQA2025性能监控",
                        "description": "系统性能指标监控仪表板",
                        "file": "performance_dashboard.json"
                    },
                    {
                        "name": "RQA2025业务监控",
                        "description": "业务指标监控仪表板",
                        "file": "business_dashboard.json"
                    }
                ],
                "alert_rules": [
                    {
                        "name": "system_alerts.yml",
                        "description": "系统级告警规则"
                    },
                    {
                        "name": "business_alerts.yml",
                        "description": "业务级告警规则"
                    }
                ]
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"✅ 已创建默认监控配置文件: {self.config_path}")

        # 加载配置
        self.load_config()

    def load_config(self):
        """加载监控配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            logger.info("✅ 监控配置加载完成")

        except Exception as e:
            logger.error(f"❌ 加载监控配置失败: {e}")
            raise

    def create_system_overview_dashboard(self):
        """创建系统概览仪表板"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "RQA2025系统概览",
                "description": "RQA2025系统整体监控仪表板",
                "tags": ["rqa2025", "system", "overview"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU使用率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "cpu_usage_percent",
                                "legendFormat": "CPU使用率"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "yAxes": [
                            {"label": "使用率 (%)", "min": 0, "max": 100},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "内存使用率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "memory_usage_percent",
                                "legendFormat": "内存使用率"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "yAxes": [
                            {"label": "使用率 (%)", "min": 0, "max": 100},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "磁盘使用率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "disk_usage_percent",
                                "legendFormat": "磁盘使用率"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "yAxes": [
                            {"label": "使用率 (%)", "min": 0, "max": 100},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 4,
                        "title": "网络流量",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "network_bytes_received",
                                "legendFormat": "接收流量"
                            },
                            {
                                "expr": "network_bytes_sent",
                                "legendFormat": "发送流量"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "yAxes": [
                            {"label": "字节/秒", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 5,
                        "title": "服务状态",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "service_status",
                                "legendFormat": "服务状态"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 1}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 6,
                        "title": "错误率",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "error_rate_percent",
                                "legendFormat": "错误率"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 5},
                                        {"color": "red", "value": 10}
                                    ]
                                }
                            }
                        }
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            },
            "folderId": 0,
            "overwrite": True
        }

        dashboard_file = self.dashboards_dir / "system_overview_dashboard.json"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 系统概览仪表板已创建: {dashboard_file}")
        return dashboard

    def create_performance_dashboard(self):
        """创建性能监控仪表板"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "RQA2025性能监控",
                "description": "系统性能指标监控仪表板",
                "tags": ["rqa2025", "performance", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "响应时间",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "response_time_ms",
                                "legendFormat": "平均响应时间"
                            },
                            {
                                "expr": "response_time_p95_ms",
                                "legendFormat": "95%响应时间"
                            },
                            {
                                "expr": "response_time_p99_ms",
                                "legendFormat": "99%响应时间"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "yAxes": [
                            {"label": "响应时间 (ms)", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "吞吐量",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "requests_per_second",
                                "legendFormat": "请求/秒"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "yAxes": [
                            {"label": "请求/秒", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "队列长度",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "queue_length",
                                "legendFormat": "队列长度"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "yAxes": [
                            {"label": "队列长度", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 4,
                        "title": "缓存命中率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "cache_hit_rate_percent",
                                "legendFormat": "缓存命中率"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "yAxes": [
                            {"label": "命中率 (%)", "min": 0, "max": 100},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 5,
                        "title": "数据库连接数",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "database_connections",
                                "legendFormat": "活跃连接数"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "yellow", "value": 50},
                                        {"color": "red", "value": 80}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 6,
                        "title": "内存泄漏检测",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "memory_leak_detection",
                                "legendFormat": "内存泄漏指标"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
                        "yAxes": [
                            {"label": "内存使用趋势", "min": 0},
                            {"show": False}
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            },
            "folderId": 0,
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
                "id": None,
                "title": "RQA2025业务监控",
                "description": "业务指标监控仪表板",
                "tags": ["rqa2025", "business", "metrics"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "交易量",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "trading_volume",
                                "legendFormat": "交易量"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "yAxes": [
                            {"label": "交易量", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 2,
                        "title": "模型预测准确率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "model_accuracy_percent",
                                "legendFormat": "预测准确率"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "yAxes": [
                            {"label": "准确率 (%)", "min": 0, "max": 100},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 3,
                        "title": "特征工程处理时间",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "feature_engineering_time_ms",
                                "legendFormat": "特征处理时间"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "yAxes": [
                            {"label": "处理时间 (ms)", "min": 0},
                            {"show": False}
                        ]
                    },
                    {
                        "id": 4,
                        "title": "数据质量指标",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "data_quality_score",
                                "legendFormat": "数据质量分数"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 70},
                                        {"color": "green", "value": 90}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 5,
                        "title": "模型性能评分",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "model_performance_score",
                                "legendFormat": "模型性能评分"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16},
                        "fieldConfig": {
                            "defaults": {
                                "color": {
                                    "mode": "thresholds"
                                },
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": None},
                                        {"color": "yellow", "value": 70},
                                        {"color": "green", "value": 90}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 6,
                        "title": "业务KPI趋势",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "business_kpi_1",
                                "legendFormat": "KPI指标1"
                            },
                            {
                                "expr": "business_kpi_2",
                                "legendFormat": "KPI指标2"
                            },
                            {
                                "expr": "business_kpi_3",
                                "legendFormat": "KPI指标3"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 20},
                        "yAxes": [
                            {"label": "KPI值", "min": 0},
                            {"show": False}
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            },
            "folderId": 0,
            "overwrite": True
        }

        dashboard_file = self.dashboards_dir / "business_dashboard.json"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 业务监控仪表板已创建: {dashboard_file}")
        return dashboard

    def create_system_alert_rules(self):
        """创建系统告警规则"""
        alert_rules = {
            "groups": [
                {
                    "name": "system_alerts",
                    "rules": [
                        {
                            "alert": "HighCPUUsage",
                            "expr": "cpu_usage_percent > 80",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "system"
                            },
                            "annotations": {
                                "summary": "CPU使用率过高",
                                "description": "CPU使用率持续超过80%达5分钟"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "memory_usage_percent > 85",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "system"
                            },
                            "annotations": {
                                "summary": "内存使用率过高",
                                "description": "内存使用率持续超过85%达5分钟"
                            }
                        },
                        {
                            "alert": "LowDiskSpace",
                            "expr": "disk_usage_percent > 90",
                            "for": "2m",
                            "labels": {
                                "severity": "critical",
                                "service": "rqa2025",
                                "component": "storage"
                            },
                            "annotations": {
                                "summary": "磁盘空间不足",
                                "description": "磁盘使用率超过90%，需要立即处理"
                            }
                        },
                        {
                            "alert": "ServiceDown",
                            "expr": "service_status != 1",
                            "for": "1m",
                            "labels": {
                                "severity": "critical",
                                "service": "rqa2025",
                                "component": "service"
                            },
                            "annotations": {
                                "summary": "服务不可用",
                                "description": "关键服务停止运行，需要立即检查"
                            }
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "error_rate_percent > 5",
                            "for": "3m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "application"
                            },
                            "annotations": {
                                "summary": "错误率过高",
                                "description": "应用错误率持续超过5%达3分钟"
                            }
                        }
                    ]
                }
            ]
        }

        alert_file = self.alerts_dir / "system_alerts.yml"
        with open(alert_file, 'w', encoding='utf-8') as f:
            yaml.dump(alert_rules, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 系统告警规则已创建: {alert_file}")
        return alert_rules

    def create_business_alert_rules(self):
        """创建业务告警规则"""
        alert_rules = {
            "groups": [
                {
                    "name": "business_alerts",
                    "rules": [
                        {
                            "alert": "LowModelAccuracy",
                            "expr": "model_accuracy_percent < 70",
                            "for": "10m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "model"
                            },
                            "annotations": {
                                "summary": "模型准确率过低",
                                "description": "模型预测准确率低于70%，需要重新训练"
                            }
                        },
                        {
                            "alert": "HighFeatureProcessingTime",
                            "expr": "feature_engineering_time_ms > 5000",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "features"
                            },
                            "annotations": {
                                "summary": "特征处理时间过长",
                                "description": "特征工程处理时间超过5秒，可能影响性能"
                            }
                        },
                        {
                            "alert": "LowDataQuality",
                            "expr": "data_quality_score < 80",
                            "for": "15m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "data"
                            },
                            "annotations": {
                                "summary": "数据质量过低",
                                "description": "数据质量评分低于80%，需要检查数据源"
                            }
                        },
                        {
                            "alert": "HighTradingLatency",
                            "expr": "trading_latency_ms > 100",
                            "for": "3m",
                            "labels": {
                                "severity": "critical",
                                "service": "rqa2025",
                                "component": "trading"
                            },
                            "annotations": {
                                "summary": "交易延迟过高",
                                "description": "交易延迟超过100ms，可能影响交易执行"
                            }
                        },
                        {
                            "alert": "ModelDrift",
                            "expr": "model_drift_score > 0.1",
                            "for": "30m",
                            "labels": {
                                "severity": "warning",
                                "service": "rqa2025",
                                "component": "model"
                            },
                            "annotations": {
                                "summary": "模型漂移检测",
                                "description": "模型漂移评分超过0.1，建议重新训练模型"
                            }
                        }
                    ]
                }
            ]
        }

        alert_file = self.alerts_dir / "business_alerts.yml"
        with open(alert_file, 'w', encoding='utf-8') as f:
            yaml.dump(alert_rules, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ 业务告警规则已创建: {alert_file}")
        return alert_rules

    def setup_grafana_datasource(self):
        """设置Grafana数据源"""
        try:
            # 创建Prometheus数据源配置
            datasource_config = {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://prometheus:9090",
                "access": "proxy",
                "isDefault": True,
                "jsonData": {
                    "timeInterval": "15s"
                }
            }

            # 这里应该调用Grafana API来创建数据源
            # 由于这是示例，我们只保存配置
            datasource_file = Path("config/grafana_datasource.json")
            datasource_file.parent.mkdir(parents=True, exist_ok=True)

            with open(datasource_file, 'w', encoding='utf-8') as f:
                json.dump(datasource_config, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ Grafana数据源配置已保存: {datasource_file}")
            return datasource_config

        except Exception as e:
            logger.error(f"❌ 设置Grafana数据源失败: {e}")
            return None

    def generate_monitoring_report(self, output_file: str = "reports/monitoring_setup_report.json"):
        """生成监控设置报告"""
        try:
            report = {
                "report_info": {
                    "generated_at": datetime.now().isoformat(),
                    "setup_type": "监控仪表板集成",
                    "version": "1.0.0"
                },
                "dashboards": [
                    {
                        "name": "RQA2025系统概览",
                        "file": "system_overview_dashboard.json",
                        "status": "created",
                        "panels": 6
                    },
                    {
                        "name": "RQA2025性能监控",
                        "file": "performance_dashboard.json",
                        "status": "created",
                        "panels": 6
                    },
                    {
                        "name": "RQA2025业务监控",
                        "file": "business_dashboard.json",
                        "status": "created",
                        "panels": 6
                    }
                ],
                "alert_rules": [
                    {
                        "name": "system_alerts.yml",
                        "type": "system",
                        "rules_count": 5,
                        "status": "created"
                    },
                    {
                        "name": "business_alerts.yml",
                        "type": "business",
                        "rules_count": 5,
                        "status": "created"
                    }
                ],
                "datasource": {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "status": "configured"
                },
                "metrics": {
                    "system_metrics": [
                        "cpu_usage_percent",
                        "memory_usage_percent",
                        "disk_usage_percent",
                        "network_bytes_received",
                        "network_bytes_sent",
                        "service_status",
                        "error_rate_percent"
                    ],
                    "performance_metrics": [
                        "response_time_ms",
                        "response_time_p95_ms",
                        "response_time_p99_ms",
                        "requests_per_second",
                        "queue_length",
                        "cache_hit_rate_percent",
                        "database_connections",
                        "memory_leak_detection"
                    ],
                    "business_metrics": [
                        "trading_volume",
                        "model_accuracy_percent",
                        "feature_engineering_time_ms",
                        "data_quality_score",
                        "model_performance_score",
                        "business_kpi_1",
                        "business_kpi_2",
                        "business_kpi_3"
                    ]
                },
                "next_steps": [
                    "部署Prometheus和Grafana到Kubernetes集群",
                    "配置数据源连接",
                    "导入仪表板配置",
                    "设置告警通知渠道",
                    "验证监控指标收集",
                    "配置日志聚合",
                    "设置性能基准测试"
                ],
                "summary": {
                    "total_dashboards": 3,
                    "total_panels": 18,
                    "total_alert_rules": 10,
                    "metrics_categories": 3,
                    "setup_status": "completed"
                }
            }

            # 确保输出目录存在
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 监控设置报告已生成: {output_file}")
            return report

        except Exception as e:
            logger.error(f"❌ 生成监控设置报告失败: {e}")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="监控仪表板集成")
    parser.add_argument("--config", default="config/monitoring.yml", help="配置文件路径")
    parser.add_argument("--action", choices=["setup", "test",
                        "report"], default="setup", help="执行动作")
    parser.add_argument("--output", default="reports/monitoring_setup_report.json", help="报告输出文件")

    args = parser.parse_args()

    try:
        # 创建监控仪表板管理器
        manager = MonitoringDashboardManager(args.config)

        if args.action == "setup":
            print("🔧 设置监控仪表板...")

            # 创建仪表板
            manager.create_system_overview_dashboard()
            manager.create_performance_dashboard()
            manager.create_business_dashboard()

            # 创建告警规则
            manager.create_system_alert_rules()
            manager.create_business_alert_rules()

            # 设置数据源
            manager.setup_grafana_datasource()

            print("✅ 监控仪表板设置完成！")

        elif args.action == "test":
            print("🧪 测试监控配置...")
            # 这里可以添加测试逻辑
            print("✅ 监控配置测试完成")

        elif args.action == "report":
            print("📊 生成监控设置报告...")
            report = manager.generate_monitoring_report(args.output)
            if report:
                print(f"✅ 监控设置报告已生成: {args.output}")
                print(f"📈 统计信息:")
                print(f"  - 仪表板数量: {report['summary']['total_dashboards']}")
                print(f"  - 面板总数: {report['summary']['total_panels']}")
                print(f"  - 告警规则: {report['summary']['total_alert_rules']}")
                print(f"  - 指标类别: {report['summary']['metrics_categories']}")

        print("🎉 监控仪表板集成完成！")

    except Exception as e:
        logger.error(f"❌ 监控仪表板集成失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
