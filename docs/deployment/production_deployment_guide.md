# 生产环境部署指南

## 概述

本指南详细说明了RQA系统在生产环境中的部署流程，包括环境准备、部署步骤、配置说明和故障排除等内容。

## 目录

1. [环境准备](#环境准备)
2. [部署前检查](#部署前检查)
3. [部署步骤](#部署步骤)
4. [配置说明](#配置说明)
5. [监控设置](#监控设置)
6. [故障排除](#故障排除)
7. [维护指南](#维护指南)

## 环境准备

### 系统要求

#### 硬件要求
- **CPU**: 8核心以上，推荐16核心
- **内存**: 32GB以上，推荐64GB
- **存储**: 500GB以上SSD，推荐1TB
- **网络**: 千兆以太网，推荐万兆

#### 软件要求
- **操作系统**: Ubuntu 20.04 LTS 或 CentOS 8
- **Python**: 3.8+ (推荐3.9)
- **数据库**: PostgreSQL 12+
- **缓存**: Redis 6+
- **Web服务器**: Nginx 1.18+

#### 依赖软件
```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv
sudo apt-get install -y postgresql postgresql-contrib
sudo apt-get install -y redis-server nginx
sudo apt-get install -y git curl wget

# Python依赖
pip3 install --upgrade pip
pip3 install virtualenv
```

### 环境配置

#### 1. 创建项目目录
```bash
# 创建项目目录
sudo mkdir -p /opt/rqa
sudo chown $USER:$USER /opt/rqa
cd /opt/rqa

# 克隆项目
git clone https://github.com/your-org/rqa.git .
```

#### 2. 创建Python虚拟环境
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 3. 配置数据库
```bash
# 创建数据库用户
sudo -u postgres createuser --interactive rqa_user
sudo -u postgres createdb rqa_db

# 设置密码
sudo -u postgres psql
postgres=# ALTER USER rqa_user WITH PASSWORD 'your_secure_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE rqa_db TO rqa_user;
postgres=# \q
```

#### 4. 配置Redis
```bash
# 编辑Redis配置
sudo nano /etc/redis/redis.conf

# 设置密码
requirepass your_redis_password

# 设置内存限制
maxmemory 2gb
maxmemory-policy allkeys-lru

# 重启Redis
sudo systemctl restart redis
```

#### 5. 配置Nginx
```bash
# 创建Nginx配置
sudo nano /etc/nginx/sites-available/rqa

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /static/ {
        alias /opt/rqa/static/;
        expires 30d;
    }
}

# 启用站点
sudo ln -s /etc/nginx/sites-available/rqa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 部署前检查

### 1. 系统检查
```bash
# 检查系统资源
free -h
df -h
nproc
top

# 检查网络连接
ping -c 3 google.com
curl -I https://your-domain.com
```

### 2. 服务检查
```bash
# 检查PostgreSQL
sudo systemctl status postgresql
psql -h localhost -U rqa_user -d rqa_db -c "SELECT version();"

# 检查Redis
sudo systemctl status redis
redis-cli ping

# 检查Nginx
sudo systemctl status nginx
sudo nginx -t
```

### 3. 应用检查
```bash
# 激活虚拟环境
source venv/bin/activate

# 检查Python环境
python --version
pip list

# 检查应用配置
python -c "from src.infrastructure.core.config.environment_manager import default_environment_manager; print(default_environment_manager.get_environment_info())"
```

## 部署步骤

### 1. 配置环境变量
```bash
# 创建环境变量文件
cat > .env << EOF
# 数据库配置
DATABASE_URL=postgresql://rqa_user:your_secure_password@localhost:5432/rqa_db
REDIS_URL=redis://:your_redis_password@localhost:6379/0

# 应用配置
SECRET_KEY=your_secret_key_here
DEBUG=False
ALLOWED_HOSTS=your-domain.com,localhost

# 监控配置
MONITORING_ENABLED=True
ALERT_EMAIL=admin@your-domain.com
EOF
```

### 2. 初始化数据库
```bash
# 运行数据库迁移
python scripts/deployment/init_database.py

# 创建初始数据
python scripts/deployment/create_initial_data.py
```

### 3. 配置生产环境
```bash
# 切换到生产环境
python -c "
from src.infrastructure.core.config.environment_manager import default_environment_manager
default_environment_manager.switch_environment('production')
"

# 设置生产环境配置
python scripts/deployment/setup_production_config.py
```

### 4. 启动应用服务
```bash
# 使用Gunicorn启动应用
gunicorn --workers 4 --bind 127.0.0.1:8000 --user www-data --group www-data wsgi:app

# 或者使用systemd服务
sudo systemctl start rqa
sudo systemctl enable rqa
```

### 5. 启动监控系统
```bash
# 启动监控系统
python scripts/deployment/start_monitoring.py

# 检查监控状态
python scripts/deployment/check_monitoring.py
```

## 配置说明

### 环境配置管理

#### 开发环境配置
```yaml
# config/development/config.yaml
database:
  host: localhost
  port: 5432
  name: rqa_dev
  user: rqa_user
  password: dev_password

redis:
  host: localhost
  port: 6379
  password: dev_password

logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/dev.log

monitoring:
  enabled: False
  metrics_interval: 60
  alert_threshold: 0.8

security:
  encryption_enabled: False
  session_timeout: 1800
  max_login_attempts: 10

performance:
  cache_ttl: 300
  max_connections: 50
  timeout: 30
```

#### 生产环境配置
```yaml
# config/production/config.yaml
database:
  host: localhost
  port: 5432
  name: rqa_prod
  user: rqa_user
  password: encrypted_production_password

redis:
  host: localhost
  port: 6379
  password: encrypted_production_password

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/prod.log

monitoring:
  enabled: True
  metrics_interval: 30
  alert_threshold: 0.8

security:
  encryption_enabled: True
  session_timeout: 3600
  max_login_attempts: 5

performance:
  cache_ttl: 600
  max_connections: 200
  timeout: 60
```

### 监控配置

#### 系统监控配置
```python
# 系统监控配置
system_config = {
    "cpu_threshold": 80.0,        # CPU使用率阈值
    "memory_threshold": 80.0,     # 内存使用率阈值
    "disk_threshold": 85.0,       # 磁盘使用率阈值
    "network_threshold": 1000000, # 网络流量阈值
    "monitor_interval": 60        # 监控间隔（秒）
}
```

#### 应用监控配置
```python
# 应用监控配置
application_config = {
    "response_time_threshold": 1000,  # 响应时间阈值（毫秒）
    "error_rate_threshold": 0.05,     # 错误率阈值（5%）
    "memory_threshold": 80.0,         # 内存使用阈值
    "cpu_threshold": 80.0,            # CPU使用阈值
    "monitor_interval": 30            # 监控间隔（秒）
}
```

#### 告警配置
```python
# 告警配置
alert_config = {
    "suppression_time": 300,  # 告警抑制时间（秒）
    "email": {
        "smtp_server": "smtp.your-domain.com",
        "smtp_port": 587,
        "username": "alerts@your-domain.com",
        "password": "your_email_password",
        "recipients": ["admin@your-domain.com"]
    },
    "webhook": {
        "url": "https://webhook.your-domain.com/alerts"
    }
}
```

## 监控设置

### 1. 启动监控系统
```bash
# 启动监控系统
python scripts/deployment/start_monitoring.py

# 检查监控状态
python scripts/deployment/check_monitoring.py
```

### 2. 配置告警
```bash
# 配置邮件告警
python scripts/deployment/setup_email_alerts.py

# 配置Webhook告警
python scripts/deployment/setup_webhook_alerts.py
```

### 3. 监控仪表板
```bash
# 启动监控仪表板
python scripts/deployment/start_dashboard.py

# 访问仪表板
# http://your-domain.com:8080
```

## 故障排除

### 常见问题

#### 1. 数据库连接失败
```bash
# 检查PostgreSQL状态
sudo systemctl status postgresql

# 检查连接
psql -h localhost -U rqa_user -d rqa_db

# 检查日志
sudo tail -f /var/log/postgresql/postgresql-12-main.log
```

#### 2. Redis连接失败
```bash
# 检查Redis状态
sudo systemctl status redis

# 测试连接
redis-cli ping

# 检查配置
sudo cat /etc/redis/redis.conf | grep -E "(bind|port|requirepass)"
```

#### 3. 应用启动失败
```bash
# 检查应用日志
tail -f logs/app.log

# 检查Python环境
source venv/bin/activate
python --version
pip list

# 检查配置文件
python -c "from src.infrastructure.core.config.environment_manager import default_environment_manager; print(default_environment_manager.validate_config())"
```

#### 4. 监控系统问题
```bash
# 检查监控状态
python scripts/deployment/check_monitoring.py

# 检查告警配置
python scripts/deployment/check_alerts.py

# 重启监控系统
python scripts/deployment/restart_monitoring.py
```

#### 5. 性能问题
```bash
# 检查系统资源
htop
iostat -x 1
free -h

# 检查应用性能
python scripts/deployment/check_performance.py

# 优化配置
python scripts/deployment/optimize_config.py
```

### 日志分析

#### 应用日志
```bash
# 查看应用日志
tail -f logs/app.log

# 搜索错误
grep -i error logs/app.log

# 搜索警告
grep -i warning logs/app.log
```

#### 系统日志
```bash
# 查看系统日志
sudo journalctl -u rqa -f

# 查看Nginx日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

#### 数据库日志
```bash
# 查看PostgreSQL日志
sudo tail -f /var/log/postgresql/postgresql-12-main.log

# 查看Redis日志
sudo tail -f /var/log/redis/redis-server.log
```

## 维护指南

### 日常维护

#### 1. 备份
```bash
# 数据库备份
pg_dump -h localhost -U rqa_user rqa_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 配置文件备份
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz config/

# 应用备份
tar -czf app_backup_$(date +%Y%m%d_%H%M%S).tar.gz --exclude=venv --exclude=logs .
```

#### 2. 更新
```bash
# 代码更新
git pull origin main

# 依赖更新
source venv/bin/activate
pip install -r requirements.txt

# 数据库迁移
python scripts/deployment/migrate_database.py

# 重启服务
sudo systemctl restart rqa
```

#### 3. 监控维护
```bash
# 检查监控状态
python scripts/deployment/check_monitoring.py

# 清理旧日志
python scripts/deployment/cleanup_logs.py

# 优化监控配置
python scripts/deployment/optimize_monitoring.py
```

### 性能优化

#### 1. 数据库优化
```sql
-- 分析表统计信息
ANALYZE;

-- 重建索引
REINDEX DATABASE rqa_db;

-- 清理垃圾数据
VACUUM FULL;
```

#### 2. 缓存优化
```bash
# 清理Redis缓存
redis-cli FLUSHALL

# 预热缓存
python scripts/deployment/warmup_cache.py
```

#### 3. 应用优化
```bash
# 优化Python代码
python scripts/deployment/optimize_code.py

# 调整Gunicorn配置
python scripts/deployment/optimize_gunicorn.py
```

### 安全维护

#### 1. 安全更新
```bash
# 系统安全更新
sudo apt-get update
sudo apt-get upgrade

# Python包安全更新
source venv/bin/activate
pip list --outdated
pip install --upgrade package_name
```

#### 2. 安全检查
```bash
# 检查配置文件权限
ls -la config/

# 检查日志文件权限
ls -la logs/

# 检查数据库安全
python scripts/deployment/security_check.py
```

#### 3. 访问控制
```bash
# 更新防火墙规则
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
```

## 联系支持

如果在部署过程中遇到问题，请联系技术支持：

- **邮箱**: support@your-domain.com
- **电话**: +86-xxx-xxxx-xxxx
- **在线支持**: https://support.your-domain.com

## 附录

### A. 配置文件模板

#### 环境变量模板
```bash
# .env.template
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://:password@host:port/db
SECRET_KEY=your_secret_key
DEBUG=False
ALLOWED_HOSTS=domain.com,localhost
```

#### Nginx配置模板
```nginx
# nginx.conf.template
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### B. 常用命令

#### 服务管理
```bash
# 启动服务
sudo systemctl start rqa
sudo systemctl start postgresql
sudo systemctl start redis
sudo systemctl start nginx

# 停止服务
sudo systemctl stop rqa
sudo systemctl stop postgresql
sudo systemctl stop redis
sudo systemctl stop nginx

# 重启服务
sudo systemctl restart rqa
sudo systemctl restart postgresql
sudo systemctl restart redis
sudo systemctl restart nginx

# 查看状态
sudo systemctl status rqa
sudo systemctl status postgresql
sudo systemctl status redis
sudo systemctl status nginx
```

#### 日志查看
```bash
# 应用日志
tail -f logs/app.log
grep -i error logs/app.log

# 系统日志
sudo journalctl -u rqa -f
sudo journalctl -u postgresql -f

# Nginx日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

#### 数据库操作
```bash
# 连接数据库
psql -h localhost -U rqa_user -d rqa_db

# 备份数据库
pg_dump -h localhost -U rqa_user rqa_db > backup.sql

# 恢复数据库
psql -h localhost -U rqa_user -d rqa_db < backup.sql
```

### C. 故障排除检查清单

- [ ] 检查系统资源使用情况
- [ ] 检查服务状态
- [ ] 检查网络连接
- [ ] 检查数据库连接
- [ ] 检查Redis连接
- [ ] 检查应用日志
- [ ] 检查配置文件
- [ ] 检查权限设置
- [ ] 检查防火墙规则
- [ ] 检查SSL证书

---

**注意**: 本指南适用于RQA系统v1.0及以上版本。如有疑问，请参考最新版本的文档或联系技术支持。
