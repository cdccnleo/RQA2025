# RQA2025 Docker部署指南

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
