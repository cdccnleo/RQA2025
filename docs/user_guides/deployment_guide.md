# 部署指南

## 概述

本文档提供了RQA2025数据层的详细部署指南，包括环境准备、安装配置、部署验证等步骤。

## 环境要求

### 系统要求

- **操作系统**: Linux (Ubuntu 18.04+) / Windows 10+ / macOS 10.15+
- **Python**: 3.8+
- **内存**: 最少4GB，推荐8GB+
- **磁盘**: 最少10GB可用空间
- **网络**: 稳定的互联网连接

### 依赖软件

- **Docker**: 20.10+ (可选)
- **Docker Compose**: 1.29+ (可选)
- **Git**: 2.20+

## 安装步骤

### 1. 克隆代码

```bash
git clone https://github.com/your-org/rqa2025.git
cd rqa2025
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

环境变量配置示例:
```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/rqa2025

# Redis配置
REDIS_URL=redis://localhost:6379

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 监控配置
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

## 部署方式

### 方式一: 直接部署

```bash
# 启动应用
python -m src.infrastructure.web.app_factory

# 或者使用uvicorn
uvicorn src.infrastructure.web.app_factory:app --host 0.0.0.0 --port 8000
```

### 方式二: Docker部署

```bash
# 构建镜像
docker build -t rqa2025:latest .

# 运行容器
docker run -d -p 8000:8000 --name rqa2025 rqa2025:latest
```

### 方式三: Docker Compose部署

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 配置说明

### 应用配置

```yaml
# config/app.yaml
app:
  name: RQA2025
  version: 1.0.0
  debug: false

server:
  host: 0.0.0.0
  port: 8000
  workers: 4

database:
  url: postgresql://user:password@localhost:5432/rqa2025
  pool_size: 10
  max_overflow: 20

cache:
  type: redis
  url: redis://localhost:6379
  ttl: 3600

monitoring:
  enabled: true
  prometheus_port: 9090
  grafana_port: 3000
```

### 数据源配置

```yaml
# config/data_sources.yaml
crypto:
  enabled: true
  sources:
    - name: coingecko
      api_key: your_api_key
      rate_limit: 100
    - name: binance
      api_key: your_api_key
      secret: your_secret

macro:
  enabled: true
  sources:
    - name: fred
      api_key: your_api_key
    - name: worldbank
      api_key: your_api_key
```

## 验证部署

### 1. 健康检查

```bash
# 检查API健康状态
curl http://localhost:8000/api/v1/data/health

# 预期响应
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

### 2. 功能测试

```bash
# 测试数据源列表
curl http://localhost:8000/api/v1/data/sources

# 测试数据加载
curl -X POST http://localhost:8000/api/v1/data/load   -H "Content-Type: application/json"   -d '{"source": "crypto", "symbol": "BTC"}'
```

### 3. 监控验证

```bash
# 检查Prometheus指标
curl http://localhost:9090/metrics

# 检查Grafana (如果启用)
# 访问 http://localhost:3000
```

## 性能调优

### 系统调优

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化内核参数
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

### 应用调优

```python
# 优化配置示例
app_config = {
    'workers': 4,  # CPU核心数
    'max_connections': 1000,
    'timeout': 30,
    'keepalive': 5
}
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep 8000
   
   # 杀死进程
   kill -9 <pid>
   ```

2. **数据库连接失败**
   ```bash
   # 检查数据库状态
   systemctl status postgresql
   
   # 检查连接
   psql -h localhost -U user -d rqa2025
   ```

3. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   
   # 优化内存配置
   export PYTHONMALLOC=malloc
   ```

### 日志分析

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log

# 查看性能日志
grep "response_time" logs/app.log
```

## 维护

### 备份策略

```bash
# 数据库备份
pg_dump rqa2025 > backup_$(date +%Y%m%d).sql

# 配置文件备份
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### 更新部署

```bash
# 停止服务
docker-compose down

# 拉取最新代码
git pull origin main

# 重新构建
docker-compose build

# 启动服务
docker-compose up -d
```

### 监控维护

```bash
# 检查服务状态
docker-compose ps

# 查看资源使用
docker stats

# 清理日志
docker system prune -f
```
