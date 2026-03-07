# RQA2025 容器部署指南

## 📋 部署概览

RQA2025量化交易系统现已支持完整的容器化部署，包含以下服务：

- **应用服务**: FastAPI数据采集API服务
- **数据库**: PostgreSQL + TimescaleDB时序优化
- **缓存**: Redis集群缓存
- **对象存储**: MinIO分布式存储
- **监控**: Prometheus + Grafana + Loki + Promtail
- **系统监控**: Node Exporter + cAdvisor

## 🏗️ 部署架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RQA2025 部署架构                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌─────────────────────────────────────┐     │
│  │   负载均衡      │     │        监控可视化                  │     │
│  │   Nginx         │     │   Grafana     Prometheus     Loki   │     │
│  │   (Port: 80)    │     │   (Port: 3000) (9090)        (3100)  │     │
│  └─────────────────┘     └─────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     应用服务层                                  │ │
│  │  ┌─────────────────┐     ┌─────────────────────────────────┐     │ │
│  │  │   FastAPI API   │     │      数据采集工作流              │     │ │
│  │  │   (Port: 8000)  │     │   历史数据采集 + 补全调度        │     │ │
│  │  └─────────────────┘     └─────────────────────────────────┘     │ │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     数据存储层                                  │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │ │
│  │  │ PostgreSQL  │    │    Redis    │    │     MinIO       │     │ │
│  │  │ +TimescaleDB│    │   缓存层    │    │   对象存储      │     │ │
│  │  │ (Port: 5432)│    │ (Port: 6379)│    │  (Port: 9000)   │     │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────┘     │ │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     系统监控层                                  │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐     │ │
│  │  │ Node Exporter│    │  cAdvisor   │    │   Promtail      │     │ │
│  │  │ 系统监控     │    │ 容器监控    │    │   日志收集      │     │ │
│  │  │ (Port: 9100)│    │ (Port: 8080)│    │                 │     │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────┘     │ │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 快速部署

### 1. 环境准备

确保系统已安装：
- Docker >= 20.0
- Docker Compose >= 2.0
- 至少 8GB RAM 和 50GB 磁盘空间

### 2. 克隆项目

```bash
git clone https://github.com/your-org/rqa2025.git
cd rqa2025
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.production.template .env.production

# 编辑生产环境配置
# 注意：修改所有默认密码和密钥！
nano .env.production
```

关键配置项：
```bash
# 数据库密码
DB_PASSWORD=your-secure-db-password

# Redis密码
REDIS_PASSWORD=your-secure-redis-password

# JWT密钥
JWT_SECRET=your-super-secure-jwt-secret

# 备份加密密钥
BACKUP_ENCRYPTION_KEY=your-32-char-encryption-key

# API密钥（根据需要）
AKSHARE_API_KEY=your-akshare-api-key
TUSHARE_TOKEN=your-tushare-token
```

### 4. 一键部署

```bash
# 构建并启动所有服务
docker-compose -f docker-compose.prod.yml up -d

# 或者使用部署脚本（Linux/Mac）
chmod +x scripts/deploy_containers.sh
./scripts/deploy_containers.sh
```

### 5. 验证部署

```bash
# 检查服务状态
docker-compose -f docker-compose.prod.yml ps

# 查看服务日志
docker-compose -f docker-compose.prod.yml logs -f app

# 健康检查
curl http://localhost:8000/health
```

## 📊 服务访问地址

部署成功后，可通过以下地址访问服务：

| 服务 | 地址 | 默认凭据 | 说明 |
|------|------|----------|------|
| **Web应用** | http://localhost | - | 主应用入口 |
| **API服务** | http://localhost:8000 | - | FastAPI数据采集API |
| **API文档** | http://localhost:8000/docs | - | 自动生成的API文档 |
| **Grafana** | http://localhost:3000 | admin/GrafanaAdmin123! | 监控仪表板 |
| **Prometheus** | http://localhost:9090 | - | 指标收集服务 |
| **MinIO** | http://localhost:9000 | minioadmin/minioadmin | 对象存储控制台 |
| **Loki** | http://localhost:3100 | - | 日志聚合服务 |

## 🔧 核心功能验证

### 数据采集API测试

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 启动数据采集任务
curl -X POST "http://localhost:8000/api/v1/acquisition/start" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["000001.SZ", "000002.SZ"],
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "data_types": ["stock"],
    "priority": "high",
    "quality_threshold": 0.85,
    "max_concurrent": 2
  }'

# 3. 查询任务状态
curl "http://localhost:8000/api/v1/acquisition/task_xxx/status"

# 4. 查询历史数据
curl "http://localhost:8000/api/v1/data/stock/000001.SZ?start_date=2020-01-01&end_date=2020-12-31"
```

### 监控面板配置

1. 访问 Grafana: http://localhost:3000
2. 登录: admin / GrafanaAdmin123!
3. 配置数据源:
   - 类型: Prometheus
   - URL: http://prometheus:9090
   - 访问: Server (默认)
4. 导入仪表板:
   - 上传 `monitoring/grafana/dashboards/data_collection_dashboard.json`

### MinIO存储验证

1. 访问 MinIO 控制台: http://localhost:9000
2. 登录: minioadmin / minioadmin
3. 验证存储桶:
   - rqa2025-data (主数据存储)
   - rqa2025-backups (备份存储)
   - rqa2025-temp (临时文件)

## 📋 部署后配置

### 1. 修改默认密码

```bash
# 修改Grafana密码
# 在Grafana界面: Configuration -> Users -> admin -> Change password

# 修改MinIO密码（生产环境）
# 在MinIO界面: Access Keys -> Create access key
```

### 2. 配置监控告警

在Grafana中配置告警规则：
- 数据采集失败率 > 5%
- 系统CPU使用率 > 80%
- 内存使用率 > 85%
- 磁盘空间不足

### 3. 设置备份策略

```bash
# 配置自动备份
# 1. 设置备份定时任务
# 2. 配置MinIO生命周期规则
# 3. 设置备份验证机制
```

### 4. 配置日志轮转

```bash
# 在docker-compose.prod.yml中添加日志配置
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## 🔍 故障排除

### 常见问题

#### 1. 容器启动失败
```bash
# 查看详细日志
docker-compose -f docker-compose.prod.yml logs <service_name>

# 检查端口冲突
netstat -tulpn | grep :8000
```

#### 2. 数据库连接失败
```bash
# 检查PostgreSQL状态
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U rqa2025_admin -d rqa2025_prod

# 查看数据库日志
docker-compose -f docker-compose.prod.yml logs postgres
```

#### 3. API服务无响应
```bash
# 检查应用日志
docker-compose -f docker-compose.prod.yml logs app

# 测试API端点
curl -v http://localhost:8000/health
```

#### 3. TimescaleDB验证
```bash
# 检查TimescaleDB扩展状态
docker-compose -f docker-compose.prod.yml exec postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT * FROM timescaledb_information.hypertables;"

# 检查TimescaleDB版本
docker-compose -f docker-compose.prod.yml exec postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"

# 验证超表创建
docker-compose -f docker-compose.prod.yml exec postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"

# 检查TimescaleDB性能统计
docker-compose -f docker-compose.prod.yml exec postgres psql -U rqa2025_admin -d rqa2025_prod -c "SELECT * FROM timescaledb_information.compressed_hypertable_stats;"
```

#### 4. 监控数据缺失
```bash
# 检查Prometheus配置
docker-compose -f docker-compose.prod.yml exec prometheus cat /etc/prometheus/prometheus.yml

# 验证指标收集
curl http://localhost:9090/api/v1/targets
```

### 性能优化

#### 1. 内存优化
```yaml
# 在docker-compose.prod.yml中调整
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

#### 2. CPU优化
```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
        reservations:
          cpus: '1.0'
```

#### 3. 数据库优化
```sql
-- 调整PostgreSQL配置
ALTER SYSTEM SET shared_buffers = '512MB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
```

## 🔄 升级部署

### 滚动升级
```bash
# 停止旧版本
docker-compose -f docker-compose.prod.yml up -d --scale app=0

# 构建新镜像
docker-compose -f docker-compose.prod.yml build app

# 启动新版本
docker-compose -f docker-compose.prod.yml up -d --scale app=1

# 验证新版本
curl http://localhost:8000/health

# 清理旧镜像
docker image prune -f
```

### 蓝绿部署
```bash
# 创建新环境
docker-compose -f docker-compose.prod.yml -p rqa2025-green up -d

# 测试新环境
curl http://localhost:8001/health  # 新环境端口

# 切换流量（更新Nginx配置）
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload

# 清理旧环境
docker-compose -f docker-compose.prod.yml -p rqa2025-blue down
```

## 📞 支持与维护

### 监控指标
- **系统健康**: CPU/内存/磁盘使用率
- **应用性能**: 请求响应时间、吞吐量
- **数据质量**: 采集成功率、数据完整性
- **业务指标**: 任务完成数、用户活跃度

### 日志管理
- **应用日志**: `/app/logs/rqa2025.log`
- **系统日志**: 通过Loki集中管理
- **审计日志**: 关键操作记录

### 备份策略
- **数据库**: 每日全量备份 + 实时增量备份
- **文件存储**: MinIO自动备份
- **配置**: 配置文件版本控制

---

## 🎯 部署检查清单

- [ ] 环境依赖检查 (Docker, Docker Compose)
- [ ] 环境变量配置 (.env.production)
- [ ] 磁盘空间充足 (至少50GB)
- [ ] 网络端口可用 (8000, 3000, 5432, 6379, 9000等)
- [ ] 服务启动顺序正确
- [ ] 健康检查通过
- [ ] API端点响应正常
- [ ] 监控面板可访问
- [ ] 默认密码已修改
- [ ] 备份策略已配置
- [ ] 日志轮转已设置

---

**部署完成标志**: ✅ 所有服务正常运行，监控面板显示正常，API调用成功！

**系统就绪状态**: 🟢 **生产环境部署完成，可投入使用**