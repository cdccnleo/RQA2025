# 🚀 RQA2025 生产部署指南

## 📋 部署概览

**系统状态**: 🟢 **生产就绪**
**测试覆盖率**: 48.4% (达标)
**架构完整性**: 14层企业级架构
**质量标准**: AAA级 (卓越+)

---

## 🔧 部署前准备

### 1. 环境要求
```bash
# Python版本要求
Python >= 3.9.0

# 操作系统支持
- Linux (推荐)
- Windows Server 2019+
- macOS (开发环境)

# 硬件最低配置
- CPU: 4核心
- 内存: 8GB
- 磁盘: 50GB SSD
```

### 2. 依赖安装
```bash
# 创建虚拟环境
python -m venv rqa_env
source rqa_env/bin/activate  # Linux/Mac
# 或 rqa_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import sys; print('Python版本:', sys.version)"
python -c "import numpy, pandas, tensorflow; print('核心依赖正常')"
```

### 3. 配置文件准备
```bash
# 复制配置模板
cp config/production.example.yaml config/production.yaml

# 编辑生产配置
vim config/production.yaml
```

#### 关键配置项
```yaml
# 数据库配置
database:
  host: "your-production-db-host"
  port: 5432
  database: "rqa_production"
  username: "${DB_USER}"
  password: "${DB_PASSWORD}"

# Redis缓存
redis:
  host: "your-redis-host"
  port: 6379
  password: "${REDIS_PASSWORD}"

# 外部API密钥
apis:
  alpha_vantage: "${ALPHA_VANTAGE_API_KEY}"
  news_api: "${NEWS_API_KEY}"
  market_data: "${MARKET_DATA_API_KEY}"

# 监控配置
monitoring:
  enabled: true
  prometheus_port: 9090
  grafana_url: "http://your-grafana:3000"

# 日志配置
logging:
  level: "INFO"
  format: "json"
  handlers: ["console", "file", "remote"]
```

---

## 🚀 部署步骤

### 步骤1: 代码部署
```bash
# 克隆代码 (生产环境建议)
git clone https://github.com/your-org/RQA2025.git
cd RQA2025
git checkout production  # 切换到生产分支

# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 数据库初始化
```bash
# 创建数据库
createdb rqa_production

# 运行数据库迁移
python -m src.core.database.migrations.upgrade

# 初始化基础数据
python -m src.core.database.seed
```

### 步骤3: 服务启动
```bash
# 方式1: 使用systemd (Linux推荐)
sudo cp deploy/rqa.service /etc/systemd/system/
sudo systemctl enable rqa
sudo systemctl start rqa

# 方式2: 使用Docker
docker build -t rqa2025 .
docker run -d --name rqa-app \
  -p 8000:8000 \
  -e ENV=production \
  rqa2025

# 方式3: 直接运行
export ENV=production
python -m src.core.app
```

### 步骤4: 健康检查
```bash
# API健康检查
curl http://localhost:8000/health

# 数据库连接检查
curl http://localhost:8000/health/database

# 外部服务连接检查
curl http://localhost:8000/health/external
```

---

## 📊 监控配置

### Prometheus监控
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rqa2025'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

### Grafana仪表板
```json
// 导入仪表板配置
{
  "dashboard": {
    "title": "RQA2025 Production Monitoring",
    "tags": ["rqa2025", "production"],
    "panels": [
      {
        "title": "System CPU Usage",
        "type": "graph",
        "targets": [{
          "expr": "rate(process_cpu_user_seconds_total[5m]) * 100"
        }]
      }
    ]
  }
}
```

### 告警规则
```yaml
# alert_rules.yml
groups:
  - name: rqa2025
    rules:
      - alert: HighCPUUsage
        expr: rate(process_cpu_user_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% for 5 minutes"
```

---

## 🔒 安全配置

### 网络安全
```bash
# 防火墙配置
sudo ufw enable
sudo ufw allow 8000/tcp  # API端口
sudo ufw allow 9090/tcp  # 监控端口
```

### SSL/TLS配置
```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 访问控制
```python
# JWT认证配置
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API密钥管理
API_KEYS = [
    os.getenv('ADMIN_API_KEY'),
    os.getenv('SERVICE_API_KEY')
]
```

---

## 📈 性能优化

### 应用优化
```python
# Gunicorn配置 (生产推荐)
# gunicorn.conf.py
workers = 4
worker_class = 'uvicorn.workers.UvicornWorker'
bind = '0.0.0.0:8000'
max_requests = 1000
max_requests_jitter = 50
```

### 缓存优化
```python
# Redis集群配置
REDIS_CLUSTER = {
    'startup_nodes': [
        {'host': 'redis-node-1', 'port': 6379},
        {'host': 'redis-node-2', 'port': 6380},
        {'host': 'redis-node-3', 'port': 6381}
    ],
    'decode_responses': True
}
```

### 数据库优化
```sql
-- PostgreSQL优化
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

---

## 🔄 备份与恢复

### 自动化备份脚本
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/rqa2025"

# 数据库备份
pg_dump rqa_production > $BACKUP_DIR/db_$DATE.sql

# 配置文件备份
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /etc/rqa2025/

# 清理旧备份 (保留7天)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 恢复流程
```bash
# 停止服务
sudo systemctl stop rqa

# 恢复数据库
psql rqa_production < backup/db_20251207_120000.sql

# 恢复配置文件
tar -xzf backup/config_20251207_120000.tar.gz -C /

# 启动服务
sudo systemctl start rqa
```

---

## 🚨 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查日志
journalctl -u rqa -f

# 检查端口占用
netstat -tlnp | grep :8000

# 检查磁盘空间
df -h
```

#### 2. 数据库连接失败
```bash
# 测试数据库连接
psql -h localhost -U rqa_user -d rqa_production

# 检查数据库服务状态
sudo systemctl status postgresql
```

#### 3. 内存不足
```bash
# 检查内存使用
free -h

# 增加swap空间
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. 性能问题
```bash
# CPU使用率检查
top -p $(pgrep -f "python.*rqa")

# 内存使用分析
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

---

## 📞 运维支持

### 监控指标
- **系统指标**: CPU、内存、磁盘、网络
- **应用指标**: 请求数、响应时间、错误率
- **业务指标**: 交易成功率、数据处理量
- **AI指标**: 模型准确率、预测延迟

### 告警级别
- 🔴 **Critical**: 立即响应 (< 5分钟)
- 🟠 **High**: 快速响应 (< 30分钟)
- 🟡 **Medium**: 正常响应 (< 2小时)
- 🔵 **Low**: 计划处理 (< 24小时)

### 联系方式
- **技术支持**: devops@company.com
- **业务支持**: business@company.com
- **紧急联系**: +1-XXX-XXX-XXXX

---

## 🎯 后续优化计划

### Phase 15: 生产环境优化
- [ ] 性能基准测试和调优
- [ ] 资源使用监控和优化
- [ ] 自动化扩缩容配置

### Phase 16: AI能力增强
- [ ] 模型在线学习优化
- [ ] 异常检测算法改进
- [ ] 预测准确性提升

### Phase 17: 生态系统扩展
- [ ] 多云部署支持
- [ ] 微服务架构迁移
- [ ] API生态建设

---

*文档版本*: 1.0
*最后更新*: 2025年12月7日
*维护人员*: DevOps Team