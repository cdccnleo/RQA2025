# 🔍 RQA2025系统日志监控指南

## 📊 概述

本指南介绍如何查看和分析RQA2025量化交易系统的运行日志，以确保系统运行符合预期。

## 🐳 容器服务架构

RQA2025采用容器化部署，包含以下服务：

| 服务名称 | 容器名称 | 端口 | 功能描述 |
|----------|----------|------|----------|
| **RQA2025主应用** | `rqa2025-app-main` | 8000 | 量化交易核心服务 |
| **PostgreSQL数据库** | `rqa2025-postgres` | 5432 | 关系型数据存储 |
| **Redis缓存** | `rqa2025-redis` | 6379 | 高速缓存服务 |
| **Prometheus监控** | `rqa2025-prometheus` | 9090 | 指标收集和存储 |
| **Grafana可视化** | `rqa2025-grafana` | 3000 | 监控仪表板 |

## 🔍 日志查看方法

### 1. 查看所有容器状态

```bash
# 显示所有运行中的容器
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 显示所有容器（包括停止的）
docker ps -a
```

### 2. 查看特定服务日志

#### RQA2025主应用日志
```bash
# 查看最新日志
docker logs rqa2025-app-main --tail 20

# 实时监控日志
docker logs rqa2025-app-main -f

# 查看指定时间范围的日志
docker logs rqa2025-app-main --since "2025-12-27T00:00:00"
```

#### 数据库服务日志
```bash
# PostgreSQL日志
docker logs rqa2025-postgres --tail 10

# Redis日志
docker logs rqa2025-redis --tail 10
```

#### 监控服务日志
```bash
# Prometheus日志
docker logs rqa2025-prometheus --tail 10

# Grafana日志
docker logs rqa2025-grafana --tail 10
```

### 3. 使用自动化日志检查工具

```bash
# 运行完整的系统日志分析
python scripts/check_system_logs.py

# 定期监控（每30秒检查一次）
watch -n 30 'python scripts/check_system_logs.py'
```

## 📈 日志分析要点

### 正常运行指标

#### ✅ RQA2025主应用正常日志示例
```
INFO:     127.0.0.1:52466 - "GET /health HTTP/1.1" 200 OK
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### ✅ 数据库服务正常日志示例
```
PostgreSQL: 数据库系统已准备好接受连接
checkpoint complete: wrote 3 buffers (0.0%)
LOG:  database system is ready to accept connections
```

#### ✅ Redis缓存正常日志示例
```
Server initialized
Ready to accept connections tcp
```

#### ✅ Prometheus正常日志示例
```
Server is ready to receive web requests
```

#### ✅ Grafana正常日志示例
```
HTTP Server Listen address=[::]:3000
Plugin registered pluginId=grafana-piechart-panel
```

### ⚠️ 异常情况识别

#### 🚨 RQA2025应用异常日志
```
ERROR: Application startup failed
CRITICAL: Database connection failed
WARNING: High memory usage detected
```

#### 🚨 数据库异常日志
```
FATAL: database "rqa2025" does not exist
ERROR: connection to server failed
```

#### 🚨 Redis异常日志
```
Connection refused
OOM command not allowed when used memory > 'maxmemory'
```

#### 🚨 监控服务异常日志
```
Error loading config
Failed to start server
```

## 🔧 故障排查步骤

### 1. 服务无法启动
```bash
# 检查容器状态
docker ps -a | grep rqa2025

# 查看详细错误日志
docker logs <container_name>

# 检查容器资源使用
docker stats <container_name>

# 重新启动服务
docker restart <container_name>
```

### 2. API无响应
```bash
# 测试健康检查
curl http://localhost:8000/health

# 检查端口监听
docker exec rqa2025-app-main netstat -tlnp | grep 8000

# 查看应用日志
docker logs rqa2025-app-main --tail 50
```

### 3. 数据库连接问题
```bash
# 测试数据库连接
docker exec rqa2025-postgres pg_isready -U rqa2025 -d rqa2025

# 检查数据库日志
docker logs rqa2025-postgres

# 重启数据库
docker restart rqa2025-postgres
```

### 4. 监控数据异常
```bash
# 检查Prometheus状态
curl http://localhost:9090/-/ready

# 检查Grafana访问
curl http://localhost:3000/api/health

# 查看监控日志
docker logs rqa2025-prometheus
docker logs rqa2025-grafana
```

## 📊 性能监控指标

### 响应时间监控
- API响应时间应 < 500ms
- 数据库查询时间应 < 100ms
- 缓存命中率应 > 90%

### 资源使用监控
```bash
# 查看所有容器资源使用
docker stats

# 监控特定容器
docker stats rqa2025-app-main
```

### 业务指标监控
- 请求成功率：> 99.9%
- 错误率：< 0.1%
- 业务处理量：根据需求评估

## 🎯 日常运维建议

### 1. 定期日志检查
```bash
# 每日检查脚本
0 9 * * * /path/to/project/scripts/check_system_logs.py
```

### 2. 日志轮转配置
- 设置适当的日志保留时间
- 定期清理旧日志文件
- 监控日志文件大小

### 3. 告警设置
- 设置关键服务状态监控
- 配置错误日志告警
- 建立性能阈值告警

### 4. 备份策略
- 数据库定期备份
- 配置文件备份
- 日志文件归档

## 🛠️ 高级调试技巧

### 进入容器调试
```bash
# 进入RQA2025应用容器
docker exec -it rqa2025-app-main bash

# 进入数据库容器
docker exec -it rqa2025-postgres bash

# 进入Redis容器
docker exec -it rqa2025-redis redis-cli
```

### 网络连通性测试
```bash
# 测试服务间通信
docker exec rqa2025-app-main curl http://rqa2025-postgres:5432

# 检查网络配置
docker network inspect rqa2025
```

### 日志聚合分析
```bash
# 导出所有日志到文件
for container in $(docker ps --format "{{.Names}}" | grep rqa2025); do
    echo "=== $container ===" >> all_logs.txt
    docker logs $container >> all_logs.txt 2>&1
done
```

## 📞 技术支持

如果遇到无法解决的问题，请：

1. 收集完整的日志信息
2. 记录问题发生的时间和操作
3. 描述预期的行为和实际行为
4. 联系技术支持团队

---

*最后更新时间: 2025年12月27日*
*版本: v1.0*
