#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025生产环境部署配置脚本
生成生产环境所需的配置文件和部署清单

Author: RQA2025 Development Team
Date: 2025-12-02
"""

import os
import sys
import json
import yaml
import logging
import argparse
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import secrets
import string

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_deployment_config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionConfigGenerator:
    """生产环境配置生成器"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config_dir = Path("config/production")
        self.deploy_dir = Path("deploy/production")
        self.templates_dir = Path("templates/production")

        # 创建必要的目录
        for dir_path in [self.config_dir, self.deploy_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_secret_key(self, length: int = 64) -> str:
        """生成安全的密钥"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def generate_database_config(self) -> Dict[str, Any]:
        """生成数据库配置"""
        return {
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "rqa2025_prod"),
                "user": os.getenv("DB_USER", "rqa2025_user"),
                "password": os.getenv("DB_PASSWORD", self.generate_secret_key(32)),
                "ssl_mode": "require",
                "connection_pool": {
                    "min_size": 5,
                    "max_size": 20,
                    "max_idle_time": 300
                }
            }
        }

    def generate_redis_config(self) -> Dict[str, Any]:
        """生成Redis配置"""
        return {
            "redis": {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "password": os.getenv("REDIS_PASSWORD", self.generate_secret_key(32)),
                "db": 0,
                "ssl": True,
                "cluster": {
                    "enabled": True,
                    "nodes": [
                        {"host": "redis-node-1", "port": 6379},
                        {"host": "redis-node-2", "port": 6379},
                        {"host": "redis-node-3", "port": 6379}
                    ]
                },
                "connection_pool": {
                    "max_connections": 50,
                    "retry_on_timeout": True
                }
            }
        }

    def generate_api_config(self) -> Dict[str, Any]:
        """生成API配置"""
        return {
            "api": {
                "host": "0.0.0.0",
                "port": int(os.getenv("API_PORT", "8080")),
                "workers": int(os.getenv("API_WORKERS", "4")),
                "ssl": {
                    "enabled": True,
                    "cert_file": "/etc/ssl/certs/rqa2025.crt",
                    "key_file": "/etc/ssl/private/rqa2025.key"
                },
                "cors": {
                    "allowed_origins": ["https://trading.rqa2025.com"],
                    "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
                    "allowed_headers": ["*"],
                    "max_age": 86400
                },
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 1000,
                    "burst_limit": 100
                },
                "authentication": {
                    "jwt_secret": self.generate_secret_key(64),
                    "jwt_algorithm": "HS256",
                    "token_expiry": 3600
                }
            }
        }

    def generate_monitoring_config(self) -> Dict[str, Any]:
        """生成监控配置"""
        return {
            "monitoring": {
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "metrics_path": "/metrics"
                },
                "grafana": {
                    "enabled": True,
                    "port": 3000,
                    "admin_user": "admin",
                    "admin_password": self.generate_secret_key(16)
                },
                "alertmanager": {
                    "enabled": True,
                    "port": 9093,
                    "smtp": {
                        "host": "smtp.gmail.com",
                        "port": 587,
                        "user": os.getenv("ALERT_EMAIL_USER", "alerts@rqa2025.com"),
                        "password": os.getenv("ALERT_EMAIL_PASSWORD", self.generate_secret_key(16))
                    }
                },
                "health_checks": {
                    "enabled": True,
                    "interval": 30,
                    "timeout": 10,
                    "endpoints": [
                        "/health",
                        "/api/v1/health",
                        "/metrics"
                    ]
                }
            }
        }

    def generate_logging_config(self) -> Dict[str, Any]:
        """生成日志配置"""
        return {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO"
                    },
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": "/var/log/rqa2025/application.log",
                        "level": "INFO",
                        "maxBytes": 10485760,  # 10MB
                        "backupCount": 5
                    },
                    "syslog": {
                        "class": "logging.handlers.SysLogHandler",
                        "address": "/dev/log",
                        "level": "ERROR"
                    }
                },
                "loggers": {
                    "src": {
                        "level": "DEBUG",
                        "handlers": ["console", "file"],
                        "propagate": False
                    }
                }
            }
        }

    def generate_security_config(self) -> Dict[str, Any]:
        """生成安全配置"""
        return {
            "security": {
                "encryption": {
                    "algorithm": "AES-256-GCM",
                    "key_rotation_days": 90
                },
                "firewall": {
                    "allowed_ports": [80, 443, 8080, 9090, 3000],
                    "rate_limiting": {
                        "max_requests_per_minute": 1000,
                        "block_duration_minutes": 15
                    }
                },
                "ssl": {
                    "min_version": "TLSv1.2",
                    "cipher_suites": [
                        "ECDHE-RSA-AES256-GCM-SHA384",
                        "ECDHE-RSA-AES128-GCM-SHA256"
                    ]
                },
                "audit": {
                    "enabled": True,
                    "log_security_events": True,
                    "log_file": "/var/log/rqa2025/security.log"
                }
            }
        }

    def generate_docker_compose_config(self) -> Dict[str, Any]:
        """生成Docker Compose配置"""
        return {
            "version": "3.8",
            "services": {
                "rqa2025-api": {
                    "build": {
                        "context": ".",
                        "dockerfile": "Dockerfile.production"
                    },
                    "ports": ["8080:8080"],
                    "environment": [
                        "ENVIRONMENT=production",
                        "LOG_LEVEL=INFO"
                    ],
                    "volumes": [
                        "./config:/app/config:ro",
                        "./logs:/app/logs"
                    ],
                    "depends_on": ["postgres", "redis"],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "postgres": {
                    "image": "postgres:15",
                    "environment": {
                        "POSTGRES_DB": "rqa2025_prod",
                        "POSTGRES_USER": "rqa2025_user",
                        "POSTGRES_PASSWORD": "${DB_PASSWORD}"
                    },
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data",
                        "./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql"
                    ],
                    "ports": ["5432:5432"],
                    "restart": "unless-stopped"
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "command": "redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "restart": "unless-stopped"
                },
                "prometheus": {
                    "image": "prom/prometheus",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                        "prometheus_data:/prometheus"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles"
                    ],
                    "restart": "unless-stopped"
                },
                "grafana": {
                    "image": "grafana/grafana",
                    "ports": ["3000:3000"],
                    "environment": {
                        "GF_SECURITY_ADMIN_USER": "admin",
                        "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD}"
                    },
                    "volumes": [
                        "grafana_data:/var/lib/grafana",
                        "./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro"
                    ],
                    "restart": "unless-stopped"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {},
                "prometheus_data": {},
                "grafana_data": {}
            }
        }

    def generate_kubernetes_manifests(self) -> Dict[str, Any]:
        """生成Kubernetes清单"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "rqa2025-api",
                "namespace": "rqa2025",
                "labels": {
                    "app": "rqa2025-api",
                    "version": "v1.0.0"
                }
            },
            "spec": {
                "replicas": 3,
                "selector": {
                    "matchLabels": {
                        "app": "rqa2025-api"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "rqa2025-api"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "api",
                            "image": "rqa2025/rqa2025-api:v1.0.0",
                            "ports": [{"containerPort": 8080}],
                            "env": [
                                {"name": "ENVIRONMENT", "value": "production"},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "1Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }

    def generate_nginx_config(self) -> str:
        """生成Nginx配置"""
        return """
# RQA2025 Production Nginx Configuration
upstream rqa2025_backend {
    server api-1:8080;
    server api-2:8080;
    server api-3:8080;
}

server {
    listen 80;
    server_name trading.rqa2025.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name trading.rqa2025.com;

    ssl_certificate /etc/ssl/certs/rqa2025.crt;
    ssl_certificate_key /etc/ssl/private/rqa2025.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req zone=api burst=100 nodelay;

    location / {
        proxy_pass http://rqa2025_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location /metrics {
        proxy_pass http://prometheus:9090;
        allow 10.0.0.0/8;
        deny all;
    }

    location /grafana {
        proxy_pass http://grafana:3000;
        rewrite ^/grafana/(.*) /$1 break;
    }
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
"""

    def generate_monitoring_dashboard_config(self) -> Dict[str, Any]:
        """生成监控仪表板配置"""
        return {
            "dashboard": {
                "title": "RQA2025 Production Monitoring",
                "panels": [
                    {
                        "title": "API Response Time",
                        "type": "graph",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
                            "legendFormat": "Error rate %"
                        }]
                    },
                    {
                        "title": "Active Connections",
                        "type": "graph",
                        "targets": [{
                            "expr": "sum(active_connections)",
                            "legendFormat": "Active connections"
                        }]
                    },
                    {
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": "process_resident_memory_bytes / 1024 / 1024",
                            "legendFormat": "Memory usage (MB)"
                        }]
                    },
                    {
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(process_cpu_user_seconds_total[5m]) * 100",
                            "legendFormat": "CPU usage %"
                        }]
                    }
                ]
            }
        }

    def generate_backup_config(self) -> Dict[str, Any]:
        """生成备份配置"""
        return {
            "backup": {
                "database": {
                    "schedule": "0 2 * * *",  # 每天凌晨2点
                    "retention_days": 30,
                    "compression": True,
                    "encryption": True
                },
                "config": {
                    "schedule": "0 3 * * *",  # 每天凌晨3点
                    "retention_days": 90,
                    "include_secrets": False
                },
                "logs": {
                    "schedule": "0 */4 * * *",  # 每4小时
                    "retention_days": 7,
                    "compression": True
                },
                "storage": {
                    "type": "s3",
                    "bucket": "rqa2025-backups",
                    "region": "us-east-1",
                    "access_key": os.getenv("BACKUP_ACCESS_KEY"),
                    "secret_key": os.getenv("BACKUP_SECRET_KEY")
                }
            }
        }

    def save_config(self, config_name: str, config_data: Any, format_type: str = "json"):
        """保存配置文件"""
        if format_type == "json":
            file_path = self.config_dir / f"{config_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        elif format_type == "yaml":
            file_path = self.config_dir / f"{config_name}.yml"
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        else:  # plain text
            file_path = self.config_dir / config_name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(config_data)

        logger.info(f"Generated config: {file_path}")

    def generate_all_configs(self):
        """生成所有生产环境配置"""
        logger.info("开始生成生产环境配置...")

        # 生成各种配置文件
        configs = [
            ("database", self.generate_database_config(), "json"),
            ("redis", self.generate_redis_config(), "json"),
            ("api", self.generate_api_config(), "json"),
            ("monitoring", self.generate_monitoring_config(), "json"),
            ("logging", self.generate_logging_config(), "json"),
            ("security", self.generate_security_config(), "json"),
            ("docker-compose", self.generate_docker_compose_config(), "yaml"),
            ("kubernetes", self.generate_kubernetes_manifests(), "yaml"),
            ("grafana-dashboard", self.generate_monitoring_dashboard_config(), "json"),
            ("backup", self.generate_backup_config(), "json")
        ]

        for config_name, config_data, format_type in configs:
            self.save_config(config_name, config_data, format_type)

        # 生成Nginx配置
        nginx_config = self.generate_nginx_config()
        self.save_config("nginx.conf", nginx_config, "text")

        logger.info("生产环境配置生成完成！")

    def validate_configs(self) -> bool:
        """验证生成的配置"""
        logger.info("验证配置文件...")

        required_configs = [
            "database.json", "redis.json", "api.json",
            "monitoring.json", "logging.json", "security.json",
            "nginx.conf"
        ]

        for config_file in required_configs:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                logger.error(f"配置文件不存在: {config_path}")
                return False

            # 尝试加载JSON配置文件进行基本验证
            if config_file.endswith('.json'):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON配置文件格式错误 {config_file}: {e}")
                    return False

        logger.info("配置文件验证通过！")
        return True

    def generate_deployment_checklist(self) -> Dict[str, Any]:
        """生成部署检查清单"""
        return {
            "pre_deployment_checks": [
                "✅ 生产环境服务器准备就绪",
                "✅ 网络安全组配置完成",
                "✅ SSL证书已部署",
                "✅ 域名DNS解析配置",
                "✅ 数据库实例创建",
                "✅ Redis集群部署",
                "✅ 监控系统搭建"
            ],
            "deployment_steps": [
                "1. 备份当前生产数据",
                "2. 部署Docker镜像到测试环境",
                "3. 执行自动化测试套件",
                "4. 进行灰度发布",
                "5. 监控系统指标",
                "6. 逐步增加流量",
                "7. 验证业务功能",
                "8. 完成全量发布"
            ],
            "post_deployment_verification": [
                "✅ API响应时间 < 200ms",
                "✅ 错误率 < 0.1%",
                "✅ 系统可用性 > 99.9%",
                "✅ 数据一致性验证",
                "✅ 监控告警配置",
                "✅ 日志收集正常"
            ],
            "rollback_plan": [
                "🔄 保留上一版本镜像",
                "🔄 准备数据库备份",
                "🔄 配置快速回滚脚本",
                "🔄 制定应急响应流程"
            ]
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025生产环境配置生成器")
    parser.add_argument("--environment", default="production",
                       help="部署环境 (production/staging)")
    parser.add_argument("--validate-only", action="store_true",
                       help="仅验证现有配置")
    parser.add_argument("--generate-checklist", action="store_true",
                       help="生成部署检查清单")

    args = parser.parse_args()

    config_generator = ProductionConfigGenerator(args.environment)

    if args.validate_only:
        success = config_generator.validate_configs()
        sys.exit(0 if success else 1)

    if args.generate_checklist:
        checklist = config_generator.generate_deployment_checklist()
        checklist_path = config_generator.config_dir / "deployment_checklist.json"
        with open(checklist_path, 'w', encoding='utf-8') as f:
            json.dump(checklist, f, indent=2, ensure_ascii=False)
        logger.info(f"部署检查清单已生成: {checklist_path}")
        return

    # 生成所有配置
    config_generator.generate_all_configs()

    # 验证配置
    if config_generator.validate_configs():
        logger.info("🎉 生产环境配置生成和验证全部完成！")
        logger.info("📁 配置文件位置: config/production/")
        logger.info("🚀 可以开始生产环境部署")
    else:
        logger.error("❌ 配置验证失败，请检查配置文件")
        sys.exit(1)


if __name__ == "__main__":
    main()


