#!/usr/bin/env python3
"""
RQA2025 上线部署准备脚本
生成部署配置文件、生产环境检查清单和监控配置
"""

import os
import json
import yaml
from datetime import datetime


def create_deployment_config():
    """创建部署配置文件"""
    print("📋 创建部署配置文件...")

    config = {
        "deployment": {
            "version": "1.0.0",
            "environment": "production",
            "deployment_date": datetime.now().isoformat(),
            "components": {
                "data_layer": {
                    "enabled": True,
                    "cache_enabled": True,
                    "max_cache_size": "10GB",
                    "data_sources": ["tushare", "wind", "custom"]
                },
                "feature_layer": {
                    "enabled": True,
                    "feature_cache": True,
                    "parallel_processing": True,
                    "max_workers": 8
                },
                "model_layer": {
                    "enabled": True,
                    "model_cache": True,
                    "ensemble_enabled": True,
                    "prediction_batch_size": 1000
                },
                "strategy_layer": {
                    "enabled": True,
                    "strategies": ["limit_up", "dragon_tiger", "margin", "st"],
                    "signal_cache": True
                },
                "trading_layer": {
                    "enabled": True,
                    "risk_control": True,
                    "position_management": True,
                    "execution_engine": True
                }
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics_collection": True,
            "alerting": True,
            "log_level": "INFO",
            "performance_monitoring": True
        },
        "security": {
            "enabled": True,
            "data_encryption": True,
            "access_control": True,
            "audit_logging": True
        }
    }

    # 保存配置文件
    config_path = "config/deployment_config.json"
    os.makedirs("config", exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 部署配置文件已创建: {config_path}")
    return config


def create_production_checklist():
    """创建生产环境检查清单"""
    print("📋 创建生产环境检查清单...")

    checklist = {
        "pre_deployment": {
            "code_quality": [
                "所有单元测试通过",
                "代码覆盖率 > 90%",
                "静态代码分析通过",
                "安全漏洞扫描通过"
            ],
            "performance": [
                "性能基准测试通过",
                "内存使用优化",
                "CPU使用率检查",
                "网络延迟测试"
            ],
            "data": [
                "数据源连接测试",
                "数据质量验证",
                "缓存机制测试",
                "数据备份策略"
            ]
        },
        "deployment": {
            "infrastructure": [
                "服务器资源充足",
                "网络连接稳定",
                "数据库连接正常",
                "监控系统就绪"
            ],
            "security": [
                "防火墙配置正确",
                "SSL证书有效",
                "访问权限设置",
                "数据加密启用"
            ],
            "backup": [
                "数据备份就绪",
                "灾难恢复计划",
                "回滚方案准备",
                "监控告警配置"
            ]
        },
        "post_deployment": {
            "verification": [
                "功能测试通过",
                "性能监控正常",
                "日志记录完整",
                "错误处理验证"
            ],
            "monitoring": [
                "系统监控就绪",
                "业务指标监控",
                "告警机制测试",
                "日志分析配置"
            ]
        }
    }

    # 保存检查清单
    checklist_path = "config/production_checklist.json"
    with open(checklist_path, 'w', encoding='utf-8') as f:
        json.dump(checklist, f, indent=2, ensure_ascii=False)

    print(f"✅ 生产环境检查清单已创建: {checklist_path}")
    return checklist


def create_monitoring_config():
    """创建监控配置"""
    print("📊 创建监控配置...")

    monitoring_config = {
        "metrics": {
            "system": {
                "cpu_usage": {"enabled": True, "interval": 60},
                "memory_usage": {"enabled": True, "interval": 60},
                "disk_usage": {"enabled": True, "interval": 300},
                "network_io": {"enabled": True, "interval": 60}
            },
            "business": {
                "data_loading_speed": {"enabled": True, "interval": 300},
                "feature_calculation_time": {"enabled": True, "interval": 300},
                "model_prediction_accuracy": {"enabled": True, "interval": 600},
                "strategy_signal_count": {"enabled": True, "interval": 60},
                "trade_execution_success_rate": {"enabled": True, "interval": 60}
            },
            "trading": {
                "position_value": {"enabled": True, "interval": 60},
                "pnl": {"enabled": True, "interval": 300},
                "risk_metrics": {"enabled": True, "interval": 300},
                "execution_latency": {"enabled": True, "interval": 60}
            }
        },
        "alerts": {
            "critical": {
                "system_down": {"enabled": True, "threshold": 0},
                "data_feed_failure": {"enabled": True, "threshold": 0},
                "model_error": {"enabled": True, "threshold": 0},
                "risk_limit_breach": {"enabled": True, "threshold": 0}
            },
            "warning": {
                "high_cpu_usage": {"enabled": True, "threshold": 80},
                "high_memory_usage": {"enabled": True, "threshold": 85},
                "low_disk_space": {"enabled": True, "threshold": 90},
                "high_latency": {"enabled": True, "threshold": 1000}
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "rotation": "daily",
            "retention": "30 days",
            "compression": True
        }
    }

    # 保存监控配置
    monitoring_path = "config/monitoring_config.json"
    with open(monitoring_path, 'w', encoding='utf-8') as f:
        json.dump(monitoring_config, f, indent=2, ensure_ascii=False)

    print(f"✅ 监控配置已创建: {monitoring_path}")
    return monitoring_config


def create_docker_compose():
    """创建Docker Compose配置"""
    print("🐳 创建Docker Compose配置...")

    docker_compose = {
        "version": "3.8",
        "services": {
            "rqa2025": {
                "build": ".",
                "image": "rqa2025:latest",
                "container_name": "rqa2025-app",
                "ports": ["8000:8000"],
                "environment": [
                    "ENVIRONMENT=production",
                    "LOG_LEVEL=INFO",
                    "DATABASE_URL=postgresql://user:pass@db:5432/rqa2025"
                ],
                "volumes": [
                    "./data:/app/data",
                    "./logs:/app/logs",
                    "./config:/app/config"
                ],
                "depends_on": ["db", "redis"],
                "restart": "unless-stopped"
            },
            "db": {
                "image": "postgres:13",
                "container_name": "rqa2025-db",
                "environment": [
                    "POSTGRES_DB=rqa2025",
                    "POSTGRES_USER=user",
                    "POSTGRES_PASSWORD=pass"
                ],
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"]
            },
            "redis": {
                "image": "redis:6-alpine",
                "container_name": "rqa2025-redis",
                "ports": ["6379:6379"],
                "volumes": ["redis_data:/data"]
            },
            "monitoring": {
                "image": "prom/prometheus:latest",
                "container_name": "rqa2025-monitoring",
                "ports": ["9090:9090"],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                    "prometheus_data:/prometheus"
                ]
            }
        },
        "volumes": {
            "postgres_data": {},
            "redis_data": {},
            "prometheus_data": {}
        }
    }

    # 保存Docker Compose配置
    docker_path = "docker-compose.yml"
    with open(docker_path, 'w', encoding='utf-8') as f:
        yaml.dump(docker_compose, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ Docker Compose配置已创建: {docker_path}")
    return docker_compose


def create_deployment_script():
    """创建部署脚本"""
    print("🚀 创建部署脚本...")

    deployment_script = """#!/bin/bash
# RQA2025 部署脚本

set -e

echo "🚀 开始部署 RQA2025..."

# 1. 环境检查
echo "📋 检查部署环境..."
python scripts/quick_validation.py

# 2. 构建Docker镜像
echo "🐳 构建Docker镜像..."
docker build -t rqa2025:latest .

# 3. 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 4. 健康检查
echo "🏥 执行健康检查..."
sleep 30
python scripts/health_check.py

# 5. 部署完成
echo "✅ 部署完成！"
echo "📊 监控面板: http://localhost:9090"
echo "🔗 API文档: http://localhost:8000/docs"
"""

    # 保存部署脚本
    script_path = "scripts/deploy.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(deployment_script)

    # 设置执行权限
    os.chmod(script_path, 0o755)

    print(f"✅ 部署脚本已创建: {script_path}")
    return script_path


def create_health_check_script():
    """创建健康检查脚本"""
    print("🏥 创建健康检查脚本...")

    health_check_script = """#!/usr/bin/env python3
\"\"\"
RQA2025 健康检查脚本
\"\"\"

import requests
import time
import sys

def check_service_health():
    \"\"\"检查服务健康状态\"\"\"
    services = {
        "API服务": "http://localhost:8000/health",
        "数据库": "http://localhost:5432",
        "Redis": "http://localhost:6379",
        "监控": "http://localhost:9090"
    }
    
    all_healthy = True
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}: 健康")
            else:
                print(f"❌ {service_name}: 异常 (状态码: {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"❌ {service_name}: 无法连接 ({e})")
            all_healthy = False
    
    return all_healthy

if __name__ == "__main__":
    print("🏥 开始健康检查...")
    
    # 等待服务启动
    time.sleep(10)
    
    if check_service_health():
        print("🎉 所有服务健康！")
        sys.exit(0)
    else:
        print("❌ 部分服务异常！")
        sys.exit(1)
"""

    # 保存健康检查脚本
    script_path = "scripts/health_check.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(health_check_script)

    print(f"✅ 健康检查脚本已创建: {script_path}")
    return script_path


def main():
    """主函数"""
    print("🚀 RQA2025 上线部署准备开始")
    print("=" * 60)

    try:
        # 创建部署配置
        deployment_config = create_deployment_config()

        # 创建生产环境检查清单
        checklist = create_production_checklist()

        # 创建监控配置
        monitoring_config = create_monitoring_config()

        # 创建Docker Compose配置
        docker_compose = create_docker_compose()

        # 创建部署脚本
        deploy_script = create_deployment_script()

        # 创建健康检查脚本
        health_script = create_health_check_script()

        print("\n" + "=" * 60)
        print("✅ 上线部署准备完成！")
        print("\n📋 已创建的文件:")
        print("  - config/deployment_config.json (部署配置)")
        print("  - config/production_checklist.json (检查清单)")
        print("  - config/monitoring_config.json (监控配置)")
        print("  - docker-compose.yml (Docker配置)")
        print("  - scripts/deploy.sh (部署脚本)")
        print("  - scripts/health_check.py (健康检查)")

        print("\n🚀 下一步操作:")
        print("  1. 运行: chmod +x scripts/deploy.sh")
        print("  2. 运行: ./scripts/deploy.sh")
        print("  3. 检查: ./scripts/health_check.py")

        return True

    except Exception as e:
        print(f"❌ 部署准备失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 上线部署准备成功完成！")
    else:
        print("\n❌ 上线部署准备失败！")
