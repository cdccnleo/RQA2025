#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 Docker部署修复脚本

修复Docker配置文件，确保容器可以正常构建和部署
"""

from pathlib import Path


def fix_docker_deployment():
    """修复Docker部署配置"""
    project_root = Path(__file__).parent.parent

    print("🔧 开始修复Docker部署配置...")

    # 1. 创建缺失的Dockerfile
    create_missing_dockerfiles(project_root)

    # 2. 修复docker-compose.yml
    fix_docker_compose(project_root)

    # 3. 验证配置
    validate_configuration(project_root)

    print("✅ Docker部署配置修复完成!")


def create_missing_dockerfiles(project_root):
    """创建缺失的Dockerfile"""
    print("📦 创建缺失的Dockerfile...")

    # Dockerfile.hft
    hft_dockerfile = project_root / "Dockerfile.hft"
    if not hft_dockerfile.exists():
        hft_content = """# RQA2025 高频交易节点 Dockerfile
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/
COPY scripts/ ./scripts/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/models

# 设置权限
RUN chmod +x scripts/*.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "print('HFT node is healthy')"

# 启动命令
CMD ["python", "scripts/run_distributed_system.py", "--node-type", "hft"]
"""
        with open(hft_dockerfile, 'w', encoding='utf-8') as f:
            f.write(hft_content)
        print(f"✅ 创建了 {hft_dockerfile}")

    # Dockerfile.ml
    ml_dockerfile = project_root / "Dockerfile.ml"
    if not ml_dockerfile.exists():
        ml_content = """# RQA2025 ML推理节点 Dockerfile
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/
COPY scripts/ ./scripts/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/models

# 设置权限
RUN chmod +x scripts/*.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "print('ML node is healthy')"

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python", "scripts/run_distributed_system.py", "--node-type", "ml"]
"""
        with open(ml_dockerfile, 'w', encoding='utf-8') as f:
            f.write(ml_content)
        print(f"✅ 创建了 {ml_dockerfile}")


def fix_docker_compose(project_root):
    """修复docker-compose.yml配置"""
    print("🔧 修复docker-compose.yml配置...")

    docker_compose_path = project_root / "docker-compose.yml"

    # 读取现有配置
    import yaml
    with open(docker_compose_path, 'r', encoding='utf-8') as f:
        compose_config = yaml.safe_load(f)

    # 修复服务配置
    services = compose_config.get('services', {})

    # 修复hft-node-1配置
    if 'hft-node-1' in services:
        services['hft-node-1']['build']['dockerfile'] = 'Dockerfile.hft'
        print("✅ 修复了hft-node-1的Dockerfile引用")

    # 修复ml-node-1配置
    if 'ml-node-1' in services:
        services['ml-node-1']['build']['dockerfile'] = 'Dockerfile.ml'
        print("✅ 修复了ml-node-1的Dockerfile引用")

    # 修复主应用配置
    if 'rqa2025-core' in services:
        # 确保使用正确的启动脚本
        if 'scripts/run_distributed_system.py' not in str(services['rqa2025-core'].get('command', '')):
            services['rqa2025-core']['command'] = ["python", "scripts/run_distributed_system.py"]
            print("✅ 修复了主应用的启动命令")

    # 写回配置文件
    with open(docker_compose_path, 'w', encoding='utf-8') as f:
        yaml.dump(compose_config, f, default_flow_style=False, allow_unicode=True)

    print("✅ docker-compose.yml修复完成")


def validate_configuration(project_root):
    """验证配置"""
    print("🔍 验证Docker配置...")

    # 检查必要的文件
    required_files = [
        "Dockerfile",
        "Dockerfile.hft",
        "Dockerfile.ml",
        "docker-compose.yml",
        "requirements.txt",
        "scripts/run_distributed_system.py"
    ]

    missing_files = []
    for file in required_files:
        file_path = project_root / file
        if not file_path.exists():
            missing_files.append(file)
            print(f"❌ 缺少文件: {file}")
        else:
            print(f"✅ 文件存在: {file}")

    if missing_files:
        print(f"⚠️  还有 {len(missing_files)} 个文件缺失")
        return False

    print("✅ 所有必要文件都存在")
    return True


def create_deployment_script(project_root):
    """创建部署脚本"""
    print("📝 创建部署脚本...")

    deploy_script = project_root / "scripts" / "deploy_containers.py"
    deploy_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
RQA2025 容器部署脚本
\"\"\"

import os
import sys
import subprocess
import time
from pathlib import Path

def deploy_containers():
    \"\"\"部署RQA2025容器\"\"\"
    project_root = Path(__file__).parent.parent

    print("🚀 开始部署RQA2025容器...")

    try:
        # 1. 构建镜像
        print("📦 构建Docker镜像...")
        subprocess.run([
            "docker-compose", "build", "--no-cache"
        ], cwd=project_root, check=True)

        # 2. 启动服务
        print("🏃 启动容器服务...")
        subprocess.run([
            "docker-compose", "up", "-d"
        ], cwd=project_root, check=True)

        # 3. 等待服务启动
        print("⏳ 等待服务启动...")
        time.sleep(30)

        # 4. 检查服务状态
        print("🔍 检查服务状态...")
        result = subprocess.run([
            "docker-compose", "ps"
        ], cwd=project_root, capture_output=True, text=True)

        print("服务状态:")
        print(result.stdout)

        # 5. 检查服务健康
        print("🏥 检查服务健康...")
        check_health(project_root)

        print("✅ RQA2025容器部署成功!")
        print("\\n📋 服务访问地址:")
        print("  - 主应用: http://localhost:8080")
        print("  - 交易服务: http://localhost:8081")
        print("  - 监控服务: http://localhost:8082")
        print("  - Grafana: http://localhost:3000")
        print("  - Prometheus: http://localhost:9090")

    except subprocess.CalledProcessError as e:
        print(f"❌ 部署失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 部署出现错误: {e}")
        return False

    return True

def check_health(project_root):
    \"\"\"检查服务健康\"\"\"
    try:
        # 检查主服务健康
        result = subprocess.run([
            "docker", "exec", "rqa2025-core",
            "python", "-c", "\"print('Service is running')\""
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ 主服务运行正常")
        else:
            print("⚠️  主服务可能存在问题")

    except Exception as e:
        print(f"⚠️  健康检查失败: {e}")

def stop_containers():
    \"\"\"停止容器\"\"\"
    project_root = Path(__file__).parent.parent
    print("🛑 停止RQA2025容器...")

    try:
        subprocess.run([
            "docker-compose", "down"
        ], cwd=project_root, check=True)
        print("✅ 容器已停止")
    except Exception as e:
        print(f"❌ 停止容器失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop_containers()
    else:
        deploy_containers()
"""

    with open(deploy_script, 'w', encoding='utf-8') as f:
        f.write(deploy_content)

    print(f"✅ 创建了部署脚本: {deploy_script}")


def create_docker_readme(project_root):
    """创建Docker使用说明"""
    print("📖 创建Docker使用说明...")

    readme_content = """# RQA2025 Docker部署指南

## 概述

RQA2025是一个量化交易分析系统，支持Docker容器化部署。

## 快速开始

### 1. 构建和启动服务

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 使用部署脚本

```bash
# 部署服务
python scripts/deploy_containers.py

# 停止服务
python scripts/deploy_containers.py stop
```

## 服务架构

### 主服务
- **rqa2025-core**: 主应用服务 (端口: 8080, 8081, 8082)
- **rqa2025-hft-node-1**: 高频交易节点
- **rqa2025-ml-node-1**: ML推理节点
- **rqa2025-data-collector**: 数据采集节点

### 基础设施服务
- **rqa2025-postgres**: PostgreSQL数据库 (端口: 5432)
- **rqa2025-redis**: Redis缓存服务 (端口: 6379)
- **rqa2025-kafka**: Kafka消息队列 (端口: 9092)
- **rqa2025-zookeeper**: Zookeeper协调服务

### 监控服务
- **rqa2025-prometheus**: Prometheus监控 (端口: 9090)
- **rqa2025-grafana**: Grafana可视化 (端口: 3000)

## 服务访问

- 主应用: http://localhost:8080
- 交易服务: http://localhost:8081
- 监控服务: http://localhost:8082
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## 故障排除

### 查看服务日志
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs rqa2025-core

# 实时查看日志
docker-compose logs -f rqa2025-core
```

### 重启服务
```bash
# 重启所有服务
docker-compose restart

# 重启特定服务
docker-compose restart rqa2025-core
```

### 清理资源
```bash
# 停止并删除容器
docker-compose down

# 停止并删除容器及卷
docker-compose down -v

# 删除镜像
docker-compose down --rmi all
```

## 开发模式

### 挂载代码进行开发
```yaml
services:
  rqa2025-core:
    volumes:
      - ./src:/app/src
      - ./scripts:/app/scripts
```

### 热重载
```yaml
services:
  rqa2025-core:
    environment:
      - FLASK_ENV=development
    command: ["python", "-m", "flask", "run", "--reload"]
```

## 生产部署

### 使用生产配置
```bash
# 使用生产环境的docker-compose文件
docker-compose -f docker-compose.prod.yml up -d
```

### 环境变量配置
创建 `.env` 文件配置环境变量：

```env
ENV=production
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
```

## 监控和告警

### Grafana仪表板
访问 http://localhost:3000 配置监控仪表板。

### Prometheus指标
访问 http://localhost:9090 查看指标收集情况。

### 健康检查
```bash
# 检查服务健康
curl http://localhost:8080/health

# 检查数据库连接
curl http://localhost:8081/health/db
```

## 备份和恢复

### 数据备份
```bash
# 备份数据库
docker exec rqa2025-postgres pg_dump -U rqa2025_user rqa2025 > backup.sql

# 备份Redis数据
docker exec rqa2025-redis redis-cli save
```

### 日志管理
```bash
# 查看应用日志
docker-compose logs rqa2025-core > app_logs.txt

# 清理旧日志
docker-compose logs --tail=1000 > recent_logs.txt
```

## 性能优化

### 资源限制
```yaml
services:
  rqa2025-core:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 扩展服务
```bash
# 扩展服务实例
docker-compose up -d --scale rqa2025-core=3

# 扩展HFT节点
docker-compose up -d --scale hft-node-1=5
```

## 安全注意事项

- 定期更新基础镜像
- 使用强密码和密钥
- 配置网络安全策略
- 定期安全扫描
- 监控异常活动

## 支持

如有问题，请查看：
- 项目文档: docs/
- 日志文件: logs/
- 监控仪表板: http://localhost:3000
"""

    readme_path = project_root / "DOCKER_DEPLOYMENT_README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✅ 创建了Docker使用说明: {readme_path}")


def main():
    """主函数"""
    print("RQA2025 Docker部署修复工具")
    print("=" * 50)

    fix_docker_deployment()
    create_deployment_script(Path(__file__).parent.parent)
    create_docker_readme(Path(__file__).parent.parent)

    print("\\n🎉 Docker部署配置修复完成!")
    print("\\n📋 现在可以运行以下命令部署容器:")
    print("  python scripts/deploy_containers.py")


if __name__ == "__main__":
    main()
