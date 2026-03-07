#!/usr/bin/env python3
"""
监控告警系统设置脚本 - 生产环境监控配置
用于配置Prometheus + Grafana监控栈、设置告警规则、部署健康检查

配置内容:
✅ Prometheus配置生成
✅ Grafana仪表板配置
✅ 告警规则设置
✅ 健康检查端点配置
✅ 监控指标收集器

使用方法:
python scripts/setup_monitoring_alerts.py --setup prometheus
python scripts/setup_monitoring_alerts.py --setup grafana
python scripts/setup_monitoring_alerts.py --setup alerts
python scripts/setup_monitoring_alerts.py --deploy all
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    alertmanager_port: int = 9093
    node_exporter_port: int = 9100
    redis_exporter_port: int = 9121
    postgres_exporter_port: int = 9187

    # 监控目标
    targets: List[str] = None

    # 告警配置
    alert_rules: Dict[str, Any] = None

    # 仪表板配置
    dashboards: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.targets is None:
            self.targets = [
                "localhost:8000",  # RQA2025 API
                "localhost:8001",  # Simple App
                "localhost:6379",  # Redis
                "localhost:5432",  # PostgreSQL
            ]

        if self.alert_rules is None:
            self.alert_rules = self._get_default_alert_rules()

        if self.dashboards is None:
            self.dashboards = self._get_default_dashboards()

    def _get_default_alert_rules(self) -> Dict[str, Any]:
        """获取默认告警规则"""
        return {
            'groups': [
                {
                    'name': 'rqa2025_application',
                    'rules': [
                        {
                            'alert': 'HighResponseTime',
                            'expr': 'http_request_duration_seconds{quantile="0.95"} > 2.0',
                            'for': '5m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': '高响应时间',
                                'description': 'API响应时间超过2秒 (当前值: {{ $value }}s)'
                            }
                        },
                        {
                            'alert': 'HighErrorRate',
                            'expr': 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05',
                            'for': '5m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': '高错误率',
                                'description': 'HTTP 5xx错误率超过5% (当前值: {{ $value }})'
                            }
                        },
                        {
                            'alert': 'LowCacheHitRate',
                            'expr': 'cache_hit_ratio < 0.7',
                            'for': '10m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': '缓存命中率低',
                                'description': '缓存命中率低于70% (当前值: {{ $value }})'
                            }
                        }
                    ]
                },
                {
                    'name': 'rqa2025_system',
                    'rules': [
                        {
                            'alert': 'HighMemoryUsage',
                            'expr': '100 - (avg_over_time(node_memory_MemAvailable_bytes[5m]) / avg_over_time(node_memory_MemTotal_bytes[5m]) * 100) > 85',
                            'for': '5m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': '内存使用率高',
                                'description': '系统内存使用率超过85% (当前值: {{ $value }}%)'
                            }
                        },
                        {
                            'alert': 'HighCPUUsage',
                            'expr': '100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80',
                            'for': '5m',
                            'labels': {'severity': 'warning'},
                            'annotations': {
                                'summary': 'CPU使用率高',
                                'description': '系统CPU使用率超过80% (当前值: {{ $value }}%)'
                            }
                        },
                        {
                            'alert': 'RedisDown',
                            'expr': 'up{job="redis"} == 0',
                            'for': '1m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': 'Redis服务宕机',
                                'description': 'Redis服务不可用'
                            }
                        },
                        {
                            'alert': 'PostgreSQLDown',
                            'expr': 'up{job="postgres"} == 0',
                            'for': '1m',
                            'labels': {'severity': 'critical'},
                            'annotations': {
                                'summary': 'PostgreSQL服务宕机',
                                'description': 'PostgreSQL服务不可用'
                            }
                        }
                    ]
                }
            ]
        }

    def _get_default_dashboards(self) -> List[Dict[str, Any]]:
        """获取默认仪表板配置"""
        return [
            {
                'name': 'RQA2025_Overview',
                'description': 'RQA2025系统总览仪表板',
                'panels': [
                    {
                        'title': 'API响应时间',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                                'legendFormat': '95th percentile'
                            }
                        ]
                    },
                    {
                        'title': '错误率',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100',
                                'legendFormat': '5xx error rate'
                            }
                        ]
                    },
                    {
                        'title': '缓存性能',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'cache_hit_ratio * 100',
                                'legendFormat': 'Hit rate %'
                            },
                            {
                                'expr': 'rate(cache_misses_total[5m])',
                                'legendFormat': 'Miss rate'
                            }
                        ]
                    }
                ]
            }
        ]


class MonitoringSetup:
    """监控设置工具"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)

    def setup_prometheus_config(self):
        """生成Prometheus配置文件"""
        logger.info("生成Prometheus配置...")

        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'alert_rules.yml'
            ],
            'alerting': {
                'alertmanagers': [
                    {
                        'static_configs': [
                            {
                                'targets': [f'localhost:{self.config.alertmanager_port}']
                            }
                        ]
                    }
                ]
            },
            'scrape_configs': [
                {
                    'job_name': 'prometheus',
                    'static_configs': [
                        {
                            'targets': ['localhost:9090']
                        }
                    ]
                },
                {
                    'job_name': 'rqa2025_api',
                    'static_configs': [
                        {
                            'targets': ['localhost:8000']
                        }
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '10s'
                },
                {
                    'job_name': 'node',
                    'static_configs': [
                        {
                            'targets': [f'localhost:{self.config.node_exporter_port}']
                        }
                    ]
                },
                {
                    'job_name': 'redis',
                    'static_configs': [
                        {
                            'targets': [f'localhost:{self.config.redis_exporter_port}']
                        }
                    ]
                },
                {
                    'job_name': 'postgres',
                    'static_configs': [
                        {
                            'targets': [f'localhost:{self.config.postgres_exporter_port}']
                        }
                    ]
                }
            ]
        }

        config_file = self.monitoring_dir / "prometheus.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Prometheus配置已保存到: {config_file}")

    def setup_alert_rules(self):
        """生成告警规则文件"""
        logger.info("生成告警规则配置...")

        rules_file = self.monitoring_dir / "alert_rules.yml"
        with open(rules_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config.alert_rules, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"告警规则已保存到: {rules_file}")

    def setup_grafana_dashboards(self):
        """生成Grafana仪表板配置"""
        logger.info("生成Grafana仪表板配置...")

        dashboards_dir = self.monitoring_dir / "grafana" / "dashboards"
        dashboards_dir.mkdir(parents=True, exist_ok=True)

        for dashboard in self.config.dashboards:
            dashboard_file = dashboards_dir / f"{dashboard['name']}.json"

            # 简化的仪表板JSON结构
            dashboard_json = {
                'dashboard': {
                    'title': dashboard['name'],
                    'description': dashboard['description'],
                    'tags': ['rqa2025', 'trading', 'monitoring'],
                    'timezone': 'browser',
                    'panels': dashboard['panels'],
                    'time': {
                        'from': 'now-1h',
                        'to': 'now'
                    },
                    'timepicker': {},
                    'templating': {
                        'list': []
                    },
                    'annotations': {
                        'list': []
                    },
                    'refresh': '5s',
                    'schemaVersion': 27,
                    'version': 0,
                    'links': []
                }
            }

            with open(dashboard_file, 'w', encoding='utf-8') as f:
                json.dump(dashboard_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Grafana仪表板已保存到: {dashboards_dir}")

    def setup_health_check_endpoints(self):
        """配置健康检查端点"""
        logger.info("配置健康检查端点...")

        health_config = {
            'endpoints': [
                {
                    'name': 'database_health',
                    'url': 'http://localhost:8000/health/db',
                    'interval': '30s',
                    'timeout': '10s'
                },
                {
                    'name': 'cache_health',
                    'url': 'http://localhost:8000/health/cache',
                    'interval': '30s',
                    'timeout': '10s'
                },
                {
                    'name': 'trading_engine_health',
                    'url': 'http://localhost:8000/health/trading',
                    'interval': '30s',
                    'timeout': '10s'
                }
            ],
            'alerts': {
                'database_down': {
                    'condition': 'response_code != 200',
                    'severity': 'critical',
                    'description': '数据库健康检查失败'
                },
                'cache_down': {
                    'condition': 'response_time > 5.0',
                    'severity': 'warning',
                    'description': '缓存响应时间过长'
                }
            }
        }

        health_file = self.monitoring_dir / "health_checks.json"
        with open(health_file, 'w', encoding='utf-8') as f:
            json.dump(health_config, f, indent=2, ensure_ascii=False)

        logger.info(f"健康检查配置已保存到: {health_file}")

    def generate_docker_compose(self):
        """生成Docker Compose配置文件"""
        logger.info("生成Docker Compose配置...")

        docker_compose = {
            'version': '3.8',
            'services': {
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': [f'{self.config.prometheus_port}:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        './monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml',
                        'prometheus_data:/prometheus'
                    ],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--storage.tsdb.retention.time=200h',
                        '--web.enable-lifecycle'
                    ],
                    'restart': 'unless-stopped'
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': [f'{self.config.grafana_port}:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin',
                        'GF_USERS_ALLOW_SIGN_UP': 'false'
                    },
                    'volumes': [
                        './monitoring/grafana/provisioning:/etc/grafana/provisioning',
                        './monitoring/grafana/dashboards:/var/lib/grafana/dashboards',
                        'grafana_data:/var/lib/grafana'
                    ],
                    'restart': 'unless-stopped'
                },
                'alertmanager': {
                    'image': 'prom/alertmanager:latest',
                    'ports': [f'{self.config.alertmanager_port}:9093'],
                    'volumes': [
                        './monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml'
                    ],
                    'command': [
                        '--config.file=/etc/alertmanager/alertmanager.yml',
                        '--storage.path=/alertmanager'
                    ],
                    'restart': 'unless-stopped'
                },
                'node-exporter': {
                    'image': 'prom/node-exporter:latest',
                    'ports': [f'{self.config.node_exporter_port}:9100'],
                    'volumes': [
                        '/proc:/host/proc:ro',
                        '/sys:/host/sys:ro',
                        '/:/rootfs:ro'
                    ],
                    'command': [
                        '--path.procfs=/host/proc',
                        '--path.rootfs=/rootfs',
                        '--path.sysfs=/host/sys',
                        '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
                    ],
                    'restart': 'unless-stopped'
                },
                'redis-exporter': {
                    'image': 'oliver006/redis_exporter:latest',
                    'ports': [f'{self.config.redis_exporter_port}:9121'],
                    'environment': {
                        'REDIS_ADDR': 'redis:6379',
                        'REDIS_PASSWORD': ''
                    },
                    'restart': 'unless-stopped'
                },
                'postgres-exporter': {
                    'image': 'wrouesnel/postgres_exporter:latest',
                    'ports': [f'{self.config.postgres_exporter_port}:9187'],
                    'environment': {
                        'DATA_SOURCE_NAME': 'postgresql://rqa_user:password@postgres:5432/rqa2025?sslmode=disable'
                    },
                    'restart': 'unless-stopped'
                }
            },
            'volumes': {
                'prometheus_data': {},
                'grafana_data': {}
            }
        }

        compose_file = self.monitoring_dir / "docker-compose.monitoring.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            yaml.dump(docker_compose, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Docker Compose配置已保存到: {compose_file}")

    def setup_all(self):
        """设置所有监控组件"""
        logger.info("开始设置完整的监控系统...")

        self.setup_prometheus_config()
        self.setup_alert_rules()
        self.setup_grafana_dashboards()
        self.setup_health_check_endpoints()
        self.generate_docker_compose()

        # 生成部署脚本
        self._generate_deployment_script()

        logger.info("监控系统设置完成！")
        logger.info("下一步:")
        logger.info("1. 运行: docker-compose -f monitoring/docker-compose.monitoring.yml up -d")
        logger.info("2. 访问Grafana: http://localhost:3000 (admin/admin)")
        logger.info("3. 访问Prometheus: http://localhost:9090")

    def _generate_deployment_script(self):
        """生成部署脚本"""
        script_content = '''#!/bin/bash
# RQA2025监控系统部署脚本

set -e

echo "🚀 部署RQA2025监控系统..."

# 检查Docker和docker-compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 创建必要的目录
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

# 生成Grafana数据源配置
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# 生成Grafana仪表板配置
cat > monitoring/grafana/provisioning/dashboards/rqa2025.yml << EOF
apiVersion: 1
providers:
  - name: 'RQA2025'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

echo "📊 启动监控服务..."
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

echo "⏳ 等待服务启动..."
sleep 30

echo "✅ 监控系统部署完成！"
echo ""
echo "📈 访问地址:"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo "  • Prometheus: http://localhost:9090"
echo "  • AlertManager: http://localhost:9093"
echo ""
echo "🔍 查看服务状态:"
echo "  docker-compose -f monitoring/docker-compose.monitoring.yml ps"
echo ""
echo "🛑 停止服务:"
echo "  docker-compose -f monitoring/docker-compose.monitoring.yml down"
'''

        script_file = self.monitoring_dir / "deploy_monitoring.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # 设置执行权限
        os.chmod(script_file, 0o755)

        logger.info(f"部署脚本已保存到: {script_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='监控告警系统设置工具')
    parser.add_argument('--setup', choices=['prometheus', 'grafana', 'alerts', 'health', 'docker', 'all'],
                        default='all', help='设置的组件')
    parser.add_argument('--config', help='配置文件路径')

    args = parser.parse_args()

    # 初始化监控配置
    config = MonitoringConfig()

    # 初始化设置工具
    setup = MonitoringSetup(config)

    try:
        if args.setup == 'prometheus' or args.setup == 'all':
            setup.setup_prometheus_config()

        if args.setup == 'grafana' or args.setup == 'all':
            setup.setup_grafana_dashboards()

        if args.setup == 'alerts' or args.setup == 'all':
            setup.setup_alert_rules()

        if args.setup == 'health' or args.setup == 'all':
            setup.setup_health_check_endpoints()

        if args.setup == 'docker' or args.setup == 'all':
            setup.generate_docker_compose()

        if args.setup == 'all':
            setup._generate_deployment_script()

        logger.info("监控系统配置完成")

    except Exception as e:
        logger.error(f"监控系统设置失败: {e}")
        raise


if __name__ == "__main__":
    main()
