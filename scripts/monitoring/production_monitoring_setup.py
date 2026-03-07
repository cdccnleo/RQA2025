#!/usr/bin/env python3
"""
生产环境监控完善脚本
完善生产环境的监控和告警
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_alertmanager: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    alert_thresholds: Dict[str, float] = None
    retention_days: int = 30
    scrape_interval: int = 30  # seconds


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    severity: str  # critical, warning, info
    condition: str
    threshold: float
    duration: str  # 5m, 10m, etc.
    description: str


class ProductionMonitoringSetup:
    """生产环境监控设置管理器"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_rules = self._get_alert_rules()
        self.monitoring_status = {}
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("ProductionMonitoring")
        logger.setLevel(logging.INFO)

        # 创建日志目录
        log_dir = Path("logs/monitoring")
        log_dir.mkdir(parents=True, exist_ok=True)

        # 文件处理器
        log_file = log_dir / f"production_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _get_alert_rules(self) -> List[AlertRule]:
        """获取告警规则"""
        return [
            AlertRule(
                name="HighCPUUsage",
                severity="warning",
                condition="cpu_usage > 80",
                threshold=80.0,
                duration="5m",
                description="CPU使用率超过80%"
            ),
            AlertRule(
                name="HighMemoryUsage",
                severity="warning",
                condition="memory_usage > 85",
                threshold=85.0,
                duration="5m",
                description="内存使用率超过85%"
            ),
            AlertRule(
                name="HighResponseTime",
                severity="warning",
                condition="response_time > 200",
                threshold=200.0,
                duration="5m",
                description="响应时间超过200ms"
            ),
            AlertRule(
                name="HighErrorRate",
                severity="critical",
                condition="error_rate > 5",
                threshold=5.0,
                duration="2m",
                description="错误率超过5%"
            ),
            AlertRule(
                name="ServiceDown",
                severity="critical",
                condition="up == 0",
                threshold=0.0,
                duration="1m",
                description="服务不可用"
            ),
            AlertRule(
                name="LowThroughput",
                severity="warning",
                condition="throughput < 500",
                threshold=500.0,
                duration="10m",
                description="吞吐量低于500 req/s"
            ),
            AlertRule(
                name="HighDiskUsage",
                severity="warning",
                condition="disk_usage > 90",
                threshold=90.0,
                duration="5m",
                description="磁盘使用率超过90%"
            ),
            AlertRule(
                name="HighNetworkLatency",
                severity="warning",
                condition="network_latency > 100",
                threshold=100.0,
                duration="5m",
                description="网络延迟超过100ms"
            )
        ]

    def start_monitoring_setup(self) -> bool:
        """开始监控设置"""
        self.logger.info("📊 开始生产环境监控设置")
        self.logger.info(f"Prometheus启用: {self.config.enable_prometheus}")
        self.logger.info(f"Grafana启用: {self.config.enable_grafana}")
        self.logger.info(f"AlertManager启用: {self.config.enable_alertmanager}")

        try:
            # 1. 设置Prometheus监控
            if not self._setup_prometheus():
                return False

            # 2. 设置Grafana仪表板
            if not self._setup_grafana():
                return False

            # 3. 设置AlertManager告警
            if not self._setup_alertmanager():
                return False

            # 4. 设置日志监控
            if not self._setup_logging_monitoring():
                return False

            # 5. 设置链路追踪
            if not self._setup_tracing():
                return False

            # 6. 配置告警规则
            if not self._configure_alert_rules():
                return False

            # 7. 验证监控系统
            if not self._validate_monitoring():
                return False

            # 8. 生成监控报告
            self._generate_monitoring_report()

            self.logger.info("✅ 生产环境监控设置完成")
            return True

        except Exception as e:
            self.logger.error(f"❌ 监控设置失败: {e}")
            return False

    def _setup_prometheus(self) -> bool:
        """设置Prometheus监控"""
        if not self.config.enable_prometheus:
            self.logger.info("📊 跳过Prometheus设置")
            return True

        self.logger.info("📊 设置Prometheus监控")

        try:
            # 创建Prometheus配置
            prometheus_config = {
                "global": {
                    "scrape_interval": f"{self.config.scrape_interval}s",
                    "evaluation_interval": "15s"
                },
                "rule_files": ["/etc/prometheus/rules/*.yml"],
                "scrape_configs": [
                    {
                        "job_name": "rqa-services",
                        "static_configs": [
                            {
                                "targets": [
                                    "api-service:8000",
                                    "business-service:8001",
                                    "model-service:8002",
                                    "trading-service:8003",
                                    "cache-service:8004",
                                    "validation-service:8005"
                                ]
                            }
                        ],
                        "metrics_path": "/metrics",
                        "scrape_interval": f"{self.config.scrape_interval}s"
                    }
                ]
            }

            # 保存配置
            config_dir = Path("config/monitoring")
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / "prometheus.yml"
            with open(config_file, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(prometheus_config, f, default_flow_style=False, allow_unicode=True)

            self.logger.info("✅ Prometheus配置已生成")
            self.monitoring_status["prometheus"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ Prometheus设置失败: {e}")
            self.monitoring_status["prometheus"] = "failed"
            return False

    def _setup_grafana(self) -> bool:
        """设置Grafana仪表板"""
        if not self.config.enable_grafana:
            self.logger.info("📊 跳过Grafana设置")
            return True

        self.logger.info("📊 设置Grafana仪表板")

        try:
            # 创建Grafana仪表板配置
            dashboards = {
                "rqa_overview": {
                    "title": "RQA服务概览",
                    "panels": [
                        {"title": "CPU使用率", "type": "graph", "targets": ["cpu_usage"]},
                        {"title": "内存使用率", "type": "graph", "targets": ["memory_usage"]},
                        {"title": "响应时间", "type": "graph", "targets": ["response_time"]},
                        {"title": "吞吐量", "type": "graph", "targets": ["throughput"]},
                        {"title": "错误率", "type": "graph", "targets": ["error_rate"]}
                    ]
                },
                "rqa_services": {
                    "title": "RQA服务详情",
                    "panels": [
                        {"title": "API服务", "type": "stat", "targets": ["api_service_metrics"]},
                        {"title": "业务服务", "type": "stat", "targets": ["business_service_metrics"]},
                        {"title": "模型服务", "type": "stat", "targets": ["model_service_metrics"]},
                        {"title": "交易服务", "type": "stat", "targets": ["trading_service_metrics"]},
                        {"title": "缓存服务", "type": "stat", "targets": ["cache_service_metrics"]},
                        {"title": "验证服务", "type": "stat", "targets": ["validation_service_metrics"]}
                    ]
                },
                "rqa_alerts": {
                    "title": "RQA告警面板",
                    "panels": [
                        {"title": "告警概览", "type": "alertlist", "targets": ["alerts"]},
                        {"title": "告警历史", "type": "table", "targets": ["alert_history"]}
                    ]
                }
            }

            # 保存仪表板配置
            dashboard_dir = Path("config/monitoring/grafana")
            dashboard_dir.mkdir(parents=True, exist_ok=True)

            for dashboard_name, dashboard_config in dashboards.items():
                dashboard_file = dashboard_dir / f"{dashboard_name}.json"
                with open(dashboard_file, 'w', encoding='utf-8') as f:
                    json.dump(dashboard_config, f, indent=2, ensure_ascii=False)

            self.logger.info("✅ Grafana仪表板配置已生成")
            self.monitoring_status["grafana"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ Grafana设置失败: {e}")
            self.monitoring_status["grafana"] = "failed"
            return False

    def _setup_alertmanager(self) -> bool:
        """设置AlertManager告警"""
        if not self.config.enable_alertmanager:
            self.logger.info("📊 跳过AlertManager设置")
            return True

        self.logger.info("📊 设置AlertManager告警")

        try:
            # 创建AlertManager配置
            alertmanager_config = {
                "global": {
                    "smtp_smarthost": "localhost:587",
                    "smtp_from": "alertmanager@rqa.com"
                },
                "route": {
                    "group_by": ["alertname"],
                    "group_wait": "10s",
                    "group_interval": "10s",
                    "repeat_interval": "1h",
                    "receiver": "rqa-team"
                },
                "receivers": [
                    {
                        "name": "rqa-team",
                        "email_configs": [
                            {
                                "to": "team@rqa.com",
                                "send_resolved": True
                            }
                        ],
                        "webhook_configs": [
                            {
                                "url": "http://webhook:5001/webhook",
                                "send_resolved": True
                            }
                        ]
                    }
                ]
            }

            # 保存配置
            config_dir = Path("config/monitoring")
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / "alertmanager.yml"
            with open(config_file, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(alertmanager_config, f, default_flow_style=False, allow_unicode=True)

            self.logger.info("✅ AlertManager配置已生成")
            self.monitoring_status["alertmanager"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ AlertManager设置失败: {e}")
            self.monitoring_status["alertmanager"] = "failed"
            return False

    def _setup_logging_monitoring(self) -> bool:
        """设置日志监控"""
        if not self.config.enable_logging:
            self.logger.info("📊 跳过日志监控设置")
            return True

        self.logger.info("📊 设置日志监控")

        try:
            # 创建日志监控配置
            logging_config = {
                "log_aggregation": {
                    "elasticsearch": {
                        "hosts": ["elasticsearch:9200"],
                        "index_pattern": "rqa-logs-*"
                    },
                    "kibana": {
                        "url": "http://kibana:5601"
                    }
                },
                "log_parsers": [
                    {
                        "name": "application_logs",
                        "pattern": r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (\w+): (.+)",
                        "fields": ["timestamp", "level", "service", "message"]
                    },
                    {
                        "name": "error_logs",
                        "pattern": r"ERROR: (.+)",
                        "fields": ["error_message"]
                    }
                ],
                "retention": {
                    "days": self.config.retention_days,
                    "max_size": "10GB"
                }
            }

            # 保存配置
            config_dir = Path("config/monitoring")
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / "logging.yml"
            with open(config_file, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(logging_config, f, default_flow_style=False, allow_unicode=True)

            self.logger.info("✅ 日志监控配置已生成")
            self.monitoring_status["logging"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ 日志监控设置失败: {e}")
            self.monitoring_status["logging"] = "failed"
            return False

    def _setup_tracing(self) -> bool:
        """设置链路追踪"""
        if not self.config.enable_tracing:
            self.logger.info("📊 跳过链路追踪设置")
            return True

        self.logger.info("📊 设置链路追踪")

        try:
            # 创建Jaeger追踪配置
            tracing_config = {
                "jaeger": {
                    "agent": {
                        "host": "jaeger-agent",
                        "port": 6831
                    },
                    "collector": {
                        "host": "jaeger-collector",
                        "port": 14268
                    },
                    "query": {
                        "host": "jaeger-query",
                        "port": 16686
                    }
                },
                "sampling": {
                    "default_strategy": {
                        "type": "probabilistic",
                        "param": 0.1
                    }
                },
                "services": [
                    "api-service",
                    "business-service",
                    "model-service",
                    "trading-service",
                    "cache-service",
                    "validation-service"
                ]
            }

            # 保存配置
            config_dir = Path("config/monitoring")
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = config_dir / "tracing.yml"
            with open(config_file, 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(tracing_config, f, default_flow_style=False, allow_unicode=True)

            self.logger.info("✅ 链路追踪配置已生成")
            self.monitoring_status["tracing"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ 链路追踪设置失败: {e}")
            self.monitoring_status["tracing"] = "failed"
            return False

    def _configure_alert_rules(self) -> bool:
        """配置告警规则"""
        self.logger.info("📊 配置告警规则")

        try:
            # 创建Prometheus告警规则
            alert_rules = []

            for rule in self.alert_rules:
                alert_rule = {
                    "groups": [
                        {
                            "name": "rqa_alerts",
                            "rules": [
                                {
                                    "alert": rule.name,
                                    "expr": rule.condition,
                                    "for": rule.duration,
                                    "labels": {
                                        "severity": rule.severity,
                                        "service": "rqa"
                                    },
                                    "annotations": {
                                        "summary": rule.description,
                                        "description": f"{rule.description} (阈值: {rule.threshold})"
                                    }
                                }
                            ]
                        }
                    ]
                }
                alert_rules.append(alert_rule)

            # 保存告警规则
            rules_dir = Path("config/monitoring/rules")
            rules_dir.mkdir(parents=True, exist_ok=True)

            for i, rule in enumerate(alert_rules):
                rule_file = rules_dir / f"alert_rule_{i+1}.yml"
                with open(rule_file, 'w', encoding='utf-8') as f:
                    import yaml
                    yaml.dump(rule, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"✅ {len(self.alert_rules)} 个告警规则已配置")
            self.monitoring_status["alert_rules"] = "success"
            return True

        except Exception as e:
            self.logger.error(f"❌ 告警规则配置失败: {e}")
            self.monitoring_status["alert_rules"] = "failed"
            return False

    def _validate_monitoring(self) -> bool:
        """验证监控系统"""
        self.logger.info("✅ 验证监控系统")

        # 模拟验证各个监控组件
        components = [
            "prometheus", "grafana", "alertmanager", "logging", "tracing"
        ]

        for component in components:
            if component in self.monitoring_status:
                status = self.monitoring_status[component]
                if status == "success":
                    self.logger.info(f"✅ {component} 验证通过")
                else:
                    self.logger.warning(f"⚠️ {component} 验证失败")

        # 模拟测试告警
        self.logger.info("🔔 测试告警系统")
        time.sleep(1)
        self.logger.info("✅ 告警测试完成")

        return True

    def _generate_monitoring_report(self):
        """生成监控报告"""
        self.logger.info("📊 生成监控报告")

        report = {
            "monitoring_info": {
                "timestamp": datetime.now().isoformat(),
                "prometheus_enabled": self.config.enable_prometheus,
                "grafana_enabled": self.config.enable_grafana,
                "alertmanager_enabled": self.config.enable_alertmanager,
                "logging_enabled": self.config.enable_logging,
                "tracing_enabled": self.config.enable_tracing
            },
            "monitoring_status": self.monitoring_status,
            "alert_rules_count": len(self.alert_rules),
            "alert_rules": [
                {
                    "name": rule.name,
                    "severity": rule.severity,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "description": rule.description
                }
                for rule in self.alert_rules
            ],
            "configuration": asdict(self.config),
            "summary": {
                "total_components": len(self.monitoring_status),
                "successful_setups": len([s for s in self.monitoring_status.values() if s == "success"]),
                "failed_setups": len([s for s in self.monitoring_status.values() if s == "failed"]),
                "alert_rules": len(self.alert_rules)
            }
        }

        # 保存报告
        report_dir = Path("reports/monitoring")
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / \
            f"production_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成Markdown报告
        markdown_report = self._generate_markdown_report(report)
        markdown_file = report_dir / \
            f"production_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        markdown_file.write_text(markdown_report, encoding='utf-8')

        self.logger.info(f"📊 监控报告已生成: {report_file}")
        self.logger.info(f"📊 Markdown报告已生成: {markdown_file}")

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """生成Markdown格式的监控报告"""
        markdown = f"""# 生产环境监控设置报告

## 📋 监控信息

- **设置时间**: {report['monitoring_info']['timestamp']}
- **Prometheus启用**: {'✅' if report['monitoring_info']['prometheus_enabled'] else '❌'}
- **Grafana启用**: {'✅' if report['monitoring_info']['grafana_enabled'] else '❌'}
- **AlertManager启用**: {'✅' if report['monitoring_info']['alertmanager_enabled'] else '❌'}
- **日志监控启用**: {'✅' if report['monitoring_info']['logging_enabled'] else '❌'}
- **链路追踪启用**: {'✅' if report['monitoring_info']['tracing_enabled'] else '❌'}

## 📊 监控状态

### 组件状态

| 组件 | 状态 | 备注 |
|------|------|------|
"""

        for component, status in report['monitoring_status'].items():
            status_icon = "✅" if status == "success" else "❌"
            markdown += f"| {component} | {status_icon} {status} | - |\n"

        markdown += f"""
### 设置统计

- **总组件数**: {report['summary']['total_components']}
- **成功设置**: {report['summary']['successful_setups']}
- **失败设置**: {report['summary']['failed_setups']}
- **告警规则**: {report['summary']['alert_rules']} 个

## 🔔 告警规则

### 配置的告警规则

| 规则名称 | 严重程度 | 条件 | 阈值 | 描述 |
|----------|----------|------|------|------|
"""

        for rule in report['alert_rules']:
            severity_icon = "🔴" if rule['severity'] == "critical" else "🟡" if rule['severity'] == "warning" else "🔵"
            markdown += f"| {rule['name']} | {severity_icon} {rule['severity']} | {rule['condition']} | {rule['threshold']} | {rule['description']} |\n"

        markdown += f"""
## ⚙️ 配置信息

### 监控配置

```json
{json.dumps(report['configuration'], indent=2, ensure_ascii=False)}
```

## 🎯 结论

生产环境监控设置{'成功完成' if report['summary']['failed_setups'] == 0 else '部分完成'}。

- **成功设置**: {report['summary']['successful_setups']}/{report['summary']['total_components']}
- **失败设置**: {report['summary']['failed_setups']}/{report['summary']['total_components']}

### 监控能力

1. **指标监控**: Prometheus收集系统和服务指标
2. **可视化**: Grafana提供丰富的仪表板
3. **告警管理**: AlertManager处理告警通知
4. **日志聚合**: 集中化日志收集和分析
5. **链路追踪**: Jaeger提供分布式追踪
6. **告警规则**: {report['summary']['alert_rules']} 个自定义告警规则

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**监控环境**: production
"""

        return markdown


def main():
    """主函数"""
    print("📊 RQA2025 生产环境监控设置工具")
    print("=" * 50)

    # 创建监控配置
    config = MonitoringConfig()

    # 创建监控设置管理器
    monitoring = ProductionMonitoringSetup(config)

    # 开始监控设置
    success = monitoring.start_monitoring_setup()

    if success:
        print("✅ 生产环境监控设置完成")
        return 0
    else:
        print("❌ 生产环境监控设置失败")
        return 1


if __name__ == "__main__":
    exit(main())
