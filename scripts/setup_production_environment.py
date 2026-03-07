#!/usr/bin/env python3
"""
生产环境模拟搭建脚本 - Phase 6.1 Day 1
用于搭建完整的生产环境模拟环境，包括数据库、缓存、监控等

搭建内容:
✅ 生产环境网络拓扑模拟
✅ PostgreSQL生产实例部署
✅ Redis集群环境搭建
✅ Docker容器化部署配置
✅ 监控系统部署配置

使用方法:
python scripts/setup_production_environment.py --setup all
python scripts/setup_production_environment.py --setup database
python scripts/setup_production_environment.py --setup redis
python scripts/setup_production_environment.py --setup monitoring
"""

import os
import yaml
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionEnvironmentConfig:
    """生产环境配置"""
    # 网络配置
    domain: str = "rqa2025.com"
    ssl_enabled: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # 数据库配置
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rqa2025_prod"
    db_user: str = "rqa_user"
    db_password: str = "secure_password_2025"
    db_ssl_mode: str = "require"

    # Redis配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "redis_secure_2025"
    redis_ssl: bool = True

    # 应用配置
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 4
    app_ssl: bool = True

    # 监控配置
    prometheus_port: int = 9090
    grafana_port: int = 3000
    alertmanager_port: int = 9093
    node_exporter_port: int = 9100

    # 环境标识
    environment: str = "production"
    debug: bool = False


class ProductionEnvironmentSetup:
    """生产环境搭建工具"""

    def __init__(self, config: ProductionEnvironmentConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.env_dir = self.project_root / "production_env"
        self.docker_dir = self.env_dir / "docker"
        self.configs_dir = self.env_dir / "configs"
        self.data_dir = self.env_dir / "data"

        # 创建目录结构
        for dir_path in [self.env_dir, self.docker_dir, self.configs_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_all(self):
        """完整生产环境搭建"""
        logger.info("🚀 开始生产环境完整搭建...")

        self.setup_network_config()
        self.setup_database_config()
        self.setup_redis_config()
        self.setup_application_config()
        self.setup_monitoring_config()
        self.setup_docker_compose()
        self.setup_deployment_scripts()

        logger.info("✅ 生产环境搭建完成！")
        logger.info("\\n📋 后续步骤:")
        logger.info("1. 运行: cd production_env && ./start_production.sh")
        logger.info("2. 验证: ./health_check.sh")
        logger.info("3. 监控: 访问 http://localhost:3000 (admin/admin)")

    def setup_network_config(self):
        """网络配置搭建"""
        logger.info("🌐 配置生产环境网络...")

        # Nginx配置文件
        nginx_config = f"""
# RQA2025生产环境Nginx配置
upstream rqa2025_app {{
    server app:8000;
}}

server {{
    listen 80;
    server_name {self.config.domain};

    # HTTP重定向到HTTPS
    return 301 https://$server_name$request_uri;
}}

server {{
    listen 443 ssl http2;
    server_name {self.config.domain};

    # SSL配置
    ssl_certificate {self.config.ssl_cert_path or '/etc/ssl/certs/rqa2025.crt'};
    ssl_certificate_key {self.config.ssl_key_path or '/etc/ssl/private/rqa2025.key'};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;

    # 安全头
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # API代理
    location /api/ {{
        proxy_pass http://rqa2025_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }}

    # 静态文件
    location /static/ {{
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}

    # 健康检查
    location /health {{
        proxy_pass http://rqa2025_app/health;
        access_log off;
    }}
}}
"""

        nginx_file = self.configs_dir / "nginx.conf"
        with open(nginx_file, 'w', encoding='utf-8') as f:
            f.write(nginx_config.strip())

        # Docker网络配置
        network_config = {
            'networks': {
                'rqa2025_network': {
                    'driver': 'bridge',
                    'ipam': {
                        'config': [
                            {
                                'subnet': '172.20.0.0/16',
                                'gateway': '172.20.0.1'
                            }
                        ]
                    }
                }
            }
        }

        network_file = self.configs_dir / "networks.yml"
        with open(network_file, 'w', encoding='utf-8') as f:
            yaml.dump(network_config, f, default_flow_style=False)

        logger.info(f"✅ 网络配置已保存到: {self.configs_dir}")

    def setup_database_config(self):
        """数据库配置搭建"""
        logger.info("🗄️ 配置生产环境数据库...")

        # PostgreSQL配置
        postgres_config = f"""
# PostgreSQL生产环境配置
listen_addresses = '*'
port = {self.config.db_port}
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 6990kB
min_wal_size = 1GB
max_wal_size = 4GB

# SSL配置
ssl = on
ssl_cert_file = '/var/lib/postgresql/data/ssl/server.crt'
ssl_key_file = '/var/lib/postgresql/data/ssl/server.key'
ssl_ca_file = '/var/lib/postgresql/data/ssl/ca.crt'

# 日志配置
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'ddl'
log_duration = on
log_lock_waits = on
log_min_duration_statement = 1000

# 监控配置
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all
pg_stat_statements.track_utility = off
"""

        postgres_file = self.configs_dir / "postgresql.conf"
        with open(postgres_file, 'w', encoding='utf-8') as f:
            f.write(postgres_config.strip())

        # 初始化脚本
        init_script = f"""
-- RQA2025生产数据库初始化脚本
-- 创建数据库和用户
CREATE DATABASE {self.config.db_name};
CREATE USER {self.config.db_user} WITH ENCRYPTED PASSWORD '{self.config.db_password}';
GRANT ALL PRIVILEGES ON DATABASE {self.config.db_name} TO {self.config.db_user};

-- 连接到应用数据库
\\c {self.config.db_name};

-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_buffercache";

-- 创建应用用户表
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    permissions JSONB DEFAULT '[]',
    balance DECIMAL(15,2) DEFAULT 10000.00,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);

-- 创建交易表
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    order_type VARCHAR(20) DEFAULT 'market',
    status VARCHAR(20) DEFAULT 'pending',
    executed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建投资组合表
CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
);

-- 插入测试数据
INSERT INTO users (username, email, password_hash, balance) VALUES
('admin', 'admin@rqa2025.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8lZRfWJ6G', 100000.00)
ON CONFLICT (username) DO NOTHING;
"""

        init_file = self.configs_dir / "init.sql"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_script.strip())

        logger.info(f"✅ 数据库配置已保存到: {self.configs_dir}")

    def setup_redis_config(self):
        """Redis配置搭建"""
        logger.info("🔴 配置生产环境Redis...")

        # Redis配置文件
        redis_config = f"""
# Redis生产环境配置
bind 0.0.0.0
port {self.config.redis_port}
timeout 0
tcp-keepalive 300
daemonize no
supervised systemd
loglevel notice
logfile ""
databases 16

# 安全配置
requirepass {self.config.redis_password}

# 内存配置
maxmemory 512mb
maxmemory-policy allkeys-lru

# 持久化配置
save 900 1
save 300 10
save 60 10000

appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec

# 集群配置 (生产环境扩展用)
# cluster-enabled yes
# cluster-config-file nodes.conf
# cluster-node-timeout 5000

# SSL配置
tls-port 6380
tls-cert-file /etc/redis/ssl/redis.crt
tls-key-file /etc/redis/ssl/redis.key
tls-ca-cert-file /etc/redis/ssl/ca.crt
tls-auth-clients optional

# 性能优化
tcp-backlog 511
databases 16
maxclients 10000
"""

        redis_file = self.configs_dir / "redis.conf"
        with open(redis_file, 'w', encoding='utf-8') as f:
            f.write(redis_config.strip())

        logger.info(f"✅ Redis配置已保存到: {self.configs_dir}")

    def setup_application_config(self):
        """应用配置搭建"""
        logger.info("🚀 配置生产环境应用...")

        # 环境变量配置
        env_config = f"""
# RQA2025生产环境变量配置
RQA_ENV={self.config.environment}
DEBUG={str(self.config.debug).lower()}

# 数据库配置
DATABASE_HOST={self.config.db_host}
DATABASE_PORT={self.config.db_port}
DATABASE_NAME={self.config.db_name}
DATABASE_USER={self.config.db_user}
DATABASE_PASSWORD={self.config.db_password}
DATABASE_SSL_MODE={self.config.db_ssl_mode}

# Redis配置
REDIS_HOST={self.config.redis_host}
REDIS_PORT={self.config.redis_port}
REDIS_PASSWORD={self.config.redis_password}
REDIS_SSL={str(self.config.redis_ssl).lower()}

# 应用配置
APP_HOST={self.config.app_host}
APP_PORT={self.config.app_port}
APP_WORKERS={self.config.app_workers}
APP_SSL={str(self.config.app_ssl).lower()}

# 安全配置
SECRET_KEY=your-super-secure-secret-key-change-in-production-2025
JWT_SECRET_KEY=your-jwt-secret-key-change-in-production-2025
BCRYPT_ROUNDS=12

# 监控配置
PROMETHEUS_MULTIPROC_DIR=/tmp
ENABLE_METRICS=true

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/app/logs/rqa2025.log

# 缓存配置
CACHE_TTL=3600
CACHE_MAX_MEMORY=512mb
"""

        env_file = self.env_dir / ".env.production"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_config.strip())

        # Docker配置文件
        dockerfile = """
# RQA2025生产环境Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    postgresql-client \\
    redis-tools \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""

        dockerfile_path = self.docker_dir / "Dockerfile.production"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile.strip())

        logger.info(f"✅ 应用配置已保存到: {self.env_dir}")

    def setup_monitoring_config(self):
        """监控配置搭建"""
        logger.info("📊 配置生产环境监控...")

        # 使用之前创建的监控配置
        monitoring_script = self.project_root / "scripts" / "setup_monitoring_alerts.py"
        if monitoring_script.exists():
            # 运行监控配置脚本
            result = subprocess.run([
                "python", str(monitoring_script), "--setup", "all"
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                # 复制监控配置到生产环境目录
                monitoring_src = self.project_root / "monitoring"
                monitoring_dst = self.env_dir / "monitoring"
                if monitoring_src.exists():
                    if monitoring_dst.exists():
                        shutil.rmtree(monitoring_dst)
                    shutil.copytree(monitoring_src, monitoring_dst)
                    logger.info(f"✅ 监控配置已复制到: {monitoring_dst}")
            else:
                logger.error(f"监控配置生成失败: {result.stderr}")
        else:
            logger.warning("监控配置脚本不存在，跳过监控配置")

    def setup_docker_compose(self):
        """Docker Compose配置"""
        logger.info("🐳 生成Docker Compose配置...")

        docker_compose = {
            'version': '3.8',
            'networks': {
                'rqa2025_network': {
                    'driver': 'bridge',
                    'ipam': {
                        'config': [
                            {
                                'subnet': '172.20.0.0/16',
                                'gateway': '172.20.0.1'
                            }
                        ]
                    }
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {},
                'grafana_data': {},
                'prometheus_data': {}
            },
            'services': {
                'postgres': {
                    'image': 'postgres:14-alpine',
                    'container_name': 'rqa2025_postgres',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'environment': {
                        'POSTGRES_DB': self.config.db_name,
                        'POSTGRES_USER': self.config.db_user,
                        'POSTGRES_PASSWORD': self.config.db_password,
                        'PGDATA': '/var/lib/postgresql/data/pgdata'
                    },
                    'volumes': [
                        './configs/postgresql.conf:/etc/postgresql/postgresql.conf',
                        './configs/init.sql:/docker-entrypoint-initdb.d/init.sql',
                        'postgres_data:/var/lib/postgresql/data'
                    ],
                    'ports': [f'{self.config.db_port}:5432'],
                    'command': 'postgres -c config_file=/etc/postgresql/postgresql.conf',
                    'healthcheck': {
                        'test': ['CMD-SHELL', f'pg_isready -U {self.config.db_user} -d {self.config.db_name}'],
                        'interval': '10s',
                        'timeout': '5s',
                        'retries': 5
                    }
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'container_name': 'rqa2025_redis',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'volumes': [
                        './configs/redis.conf:/etc/redis/redis.conf',
                        'redis_data:/data'
                    ],
                    'ports': [f'{self.config.redis_port}:6379'],
                    'command': 'redis-server /etc/redis/redis.conf',
                    'healthcheck': {
                        'test': ['CMD', 'redis-cli', '--raw', 'incr', 'ping'],
                        'interval': '10s',
                        'timeout': '3s',
                        'retries': 3
                    }
                },
                'app': {
                    'build': {
                        'context': '..',
                        'dockerfile': './production_env/docker/Dockerfile.production'
                    },
                    'container_name': 'rqa2025_app',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'env_file': './.env.production',
                    'volumes': [
                        '../src:/app/src:ro',
                        './logs:/app/logs'
                    ],
                    'ports': [f'{self.config.app_port}:8000'],
                    'depends_on': {
                        'postgres': {'condition': 'service_healthy'},
                        'redis': {'condition': 'service_healthy'}
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    }
                },
                'nginx': {
                    'image': 'nginx:alpine',
                    'container_name': 'rqa2025_nginx',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'volumes': [
                        './configs/nginx.conf:/etc/nginx/conf.d/default.conf'
                    ],
                    'ports': ['80:80', '443:443'],
                    'depends_on': ['app']
                }
            }
        }

        # 如果有监控配置，添加监控服务
        monitoring_dir = self.env_dir / "monitoring"
        if monitoring_dir.exists():
            docker_compose['services'].update({
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'container_name': 'rqa2025_prometheus',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml',
                        './monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml',
                        'prometheus_data:/prometheus'
                    ],
                    'ports': [f'{self.config.prometheus_port}:9090'],
                    'command': [
                        '--config.file=/etc/prometheus/prometheus.yml',
                        '--storage.tsdb.path=/prometheus',
                        '--web.console.libraries=/etc/prometheus/console_libraries',
                        '--web.console.templates=/etc/prometheus/consoles',
                        '--storage.tsdb.retention.time=200h',
                        '--web.enable-lifecycle'
                    ]
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'container_name': 'rqa2025_grafana',
                    'restart': 'unless-stopped',
                    'networks': ['rqa2025_network'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin',
                        'GF_USERS_ALLOW_SIGN_UP': 'false'
                    },
                    'volumes': [
                        './monitoring/grafana/provisioning:/etc/grafana/provisioning',
                        './monitoring/grafana/dashboards:/var/lib/grafana/dashboards',
                        'grafana_data:/var/lib/grafana'
                    ],
                    'ports': [f'{self.config.grafana_port}:3000'],
                    'depends_on': ['prometheus']
                }
            })

        compose_file = self.env_dir / "docker-compose.yml"
        with open(compose_file, 'w', encoding='utf-8') as f:
            yaml.dump(docker_compose, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"✅ Docker Compose配置已保存到: {compose_file}")

    def setup_deployment_scripts(self):
        """部署脚本搭建"""
        logger.info("📜 生成部署脚本...")

        # 启动脚本
        start_script = """#!/bin/bash
# RQA2025生产环境启动脚本

set -e

echo "🚀 启动RQA2025生产环境..."

# 检查Docker和docker-compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose未安装，请先安装docker-compose"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动服务
echo "🐳 启动Docker服务..."
docker-compose up -d

echo "⏳ 等待服务启动..."
sleep 60

# 运行健康检查
echo "🔍 执行健康检查..."
if ./health_check.sh; then
    echo ""
    echo "✅ 生产环境启动成功！"
    echo ""
    echo "📊 服务状态:"
    echo "  • API服务:    http://localhost:8000"
    echo "  • Grafana:    http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
    echo ""
    echo "🔧 管理命令:"
    echo "  • 查看日志:   docker-compose logs -f"
    echo "  • 停止服务:   docker-compose down"
    echo "  • 重启服务:   docker-compose restart"
else
    echo "❌ 健康检查失败，请检查服务状态"
    echo "查看日志: docker-compose logs"
    exit 1
fi
"""

        # 停止脚本
        stop_script = """#!/bin/bash
# RQA2025生产环境停止脚本

echo "🛑 停止RQA2025生产环境..."

# 优雅停止
docker-compose down

echo "✅ 生产环境已停止"
"""

        # 健康检查脚本
        health_check_script = """#!/bin/bash
# RQA2025生产环境健康检查脚本

set -e

echo "🔍 RQA2025生产环境健康检查"

# 检查服务状态
echo "📋 检查服务状态..."
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ 没有运行中的服务"
    exit 1
fi

# 检查PostgreSQL
echo "🗄️ 检查PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U rqa_user -d rqa2025_prod > /dev/null 2>&1; then
    echo "✅ PostgreSQL: 正常"
else
    echo "❌ PostgreSQL: 异常"
    exit 1
fi

# 检查Redis
echo "🔴 检查Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis: 正常"
else
    echo "❌ Redis: 异常"
    exit 1
fi

# 检查应用
echo "🚀 检查应用服务..."
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 应用服务: 正常"
else
    echo "❌ 应用服务: 异常"
    exit 1
fi

# 检查监控 (如果存在)
if docker-compose ps | grep -q prometheus; then
    echo "📊 检查Prometheus..."
    if curl -f -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo "✅ Prometheus: 正常"
    else
        echo "❌ Prometheus: 异常"
    fi
fi

if docker-compose ps | grep -q grafana; then
    echo "📈 检查Grafana..."
    if curl -f -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Grafana: 正常"
    else
        echo "❌ Grafana: 异常"
    fi
fi

echo ""
echo "🎉 所有服务健康检查通过！"
"""

        scripts = {
            'start_production.sh': start_script,
            'stop_production.sh': stop_script,
            'health_check.sh': health_check_script
        }

        for script_name, script_content in scripts.items():
            script_file = self.env_dir / script_name
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content.strip())

            # 设置执行权限
            os.chmod(script_file, 0o755)

        logger.info(f"✅ 部署脚本已保存到: {self.env_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生产环境模拟搭建工具')
    parser.add_argument('--setup', choices=['all', 'network', 'database', 'redis', 'app', 'monitoring', 'docker'],
                        default='all', help='搭建的组件')
    parser.add_argument('--domain', default='rqa2025.com', help='域名')
    parser.add_argument('--ssl', action='store_true', help='启用SSL')

    args = parser.parse_args()

    # 创建配置
    config = ProductionEnvironmentConfig(
        domain=args.domain,
        ssl_enabled=args.ssl
    )

    # 初始化搭建工具
    setup = ProductionEnvironmentSetup(config)

    try:
        if args.setup == 'all' or args.setup == 'network':
            setup.setup_network_config()

        if args.setup == 'all' or args.setup == 'database':
            setup.setup_database_config()

        if args.setup == 'all' or args.setup == 'redis':
            setup.setup_redis_config()

        if args.setup == 'all' or args.setup == 'app':
            setup.setup_application_config()

        if args.setup == 'all' or args.setup == 'monitoring':
            setup.setup_monitoring_config()

        if args.setup == 'all' or args.setup == 'docker':
            setup.setup_docker_compose()
            setup.setup_deployment_scripts()

        if args.setup == 'all':
            logger.info("🎉 生产环境模拟搭建完成！")
            logger.info("\\n📋 使用指南:")
            logger.info("1. cd production_env")
            logger.info("2. ./start_production.sh")
            logger.info("3. ./health_check.sh")

    except Exception as e:
        logger.error(f"生产环境搭建失败: {e}")
        raise


if __name__ == "__main__":
    main()
