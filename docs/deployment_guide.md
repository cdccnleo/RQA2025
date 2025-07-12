# RQA2025 基础设施层部署指南

## 1. 环境要求

### 1.1 硬件配置
| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU  | 4核      | 8核      |
| 内存 | 8GB      | 16GB     |
| 磁盘 | 50GB     | 100GB SSD|

### 1.2 软件依赖
- Python 3.8+
- PostgreSQL 12+ (用于配置存储)
- Redis (用于监控缓存)
- NVIDIA驱动 (如需GPU监控)

## 2. 安装步骤

### 2.1 基础安装
```bash
# 创建虚拟环境
python -m venv /opt/rqa2025/env
source /opt/rqa2025/env/bin/activate

# 安装依赖
pip install -r requirements-infra.txt

# 安装监控代理 (可选)
curl -sSL https://monitoring-agent.install | bash
```

### 2.2 配置文件
1. 创建配置目录
```bash
mkdir -p /etc/rqa2025/config
```

2. 生成基础配置
```yaml
# /etc/rqa2025/config/base.yaml
logging:
  level: INFO
  path: /var/log/rqa2025

monitoring:
  interval: 5.0
  retention_days: 7
```

3. 环境特定配置
```yaml
# /etc/rqa2025/config/production.yaml
database:
  host: db.prod.example.com
  port: 5432
```

## 3. 系统初始化

### 3.1 服务启动
```bash
# 启动监控服务
python -m src.infrastructure.monitoring.service start

# 启动配置热更新监听
python -m src.infrastructure.config.watcher start
```

### 3.2 健康检查
验证服务状态：
```bash
curl http://localhost:8080/health
# 预期输出: {"status": "OK", "components": [...]}
```

## 4. 运维操作

### 4.1 日常维护
```bash
# 日志轮转
logrotate /etc/logrotate.d/rqa2025

# 监控数据备份
python -m src.infrastructure.monitoring.backup --output=/backup
```

### 4.2 故障处理

#### 配置加载失败
1. 检查配置语法：
```bash
python -m src.infrastructure.config.validate /etc/rqa2025/config/
```

2. 临时恢复默认配置：
```bash
cp src/infrastructure/config/defaults.yaml /etc/rqa2025/config/base.yaml
```

#### 监控数据异常
1. 重置监控缓存：
```bash
redis-cli FLUSHALL
```

## 5. 监控集成

### 5.1 Prometheus配置
```yaml
scrape_configs:
  - job_name: 'rqa2025'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8080']
```

### 5.4 资源监控看板

#### 功能概述
```text
- 实时系统资源监控 (CPU/内存/磁盘)
- GPU使用情况可视化
- 历史数据趋势分析
- 告警状态显示
```

#### 启动方式
```bash
# 安装依赖
pip install dash plotly requests

# 启动看板 (默认端口8050)
python -m src.infrastructure.dashboard.resource_dashboard
```

#### 配置选项
```python
# 自定义配置示例
dashboard = ResourceDashboard(
    api_base_url="http://your-api-server:port/api/v1",  # 资源API地址
)
dashboard.run(host="0.0.0.0", port=8050)  # 绑定IP和端口
```

#### API端点
```text
GET /api/v1/system          # 获取当前系统资源使用
GET /api/v1/gpu             # 获取GPU使用情况  
GET /api/v1/history         # 获取历史数据
GET /api/v1/strategies      # 获取策略资源使用情况
```

#### 配置策略配额
```python
# 在策略初始化时设置资源配额
resource_manager.set_strategy_quota(
    strategy="momentum_strategy",
    cpu=30,          # 最大CPU使用率%
    gpu_memory=2048, # 最大GPU显存(MB)
    max_workers=5     # 最大工作线程数
)

# 注册策略工作线程
resource_manager.register_strategy_worker(
    strategy="momentum_strategy",
    worker_id="worker1"
)
```

#### 新增功能
```text
- 策略资源使用表格
- 工作线程配额进度条
- 策略筛选功能
- 配额超限警示
```

### 5.5 告警规则示例
```yaml
groups:
- name: infrastructure
  rules:
  - alert: HighCPUUsage
    expr: system_cpu_usage > 80
    for: 5m
    labels:
      severity: warning
```

## 6. 升级指南

### 6.1 滚动升级步骤
1. 停止旧服务
```bash
systemctl stop rqa2025-infra
```

2. 备份配置
```bash
cp -r /etc/rqa2025/config /backup/config-$(date +%F)
```

3. 部署新版本
```bash
tar xzf rqa2025-infra-2.0.tar.gz -C /opt/rqa2025
```

4. 启动新服务
```bash
systemctl start rqa2025-infra
```

### 6.2 回滚流程
1. 恢复旧版本
```bash
rm -rf /opt/rqa2025/current
ln -s /opt/rqa2025/versions/1.2 /opt/rqa2025/current
```

2. 恢复配置
```bash
cp -r /backup/config-2023-01-01/* /etc/rqa2025/config/
```

## 7. 安全配置

### 7.1 配置加密最佳实践

1. **密钥管理**:
   - 不要将密钥存储在代码或配置文件中
   - 使用KMS或密钥管理服务保管主密钥
   - 为不同环境使用不同密钥

2. **加密范围**:
   ```yaml
   # 需要加密的典型字段
   database:
     password: encrypted:...  # 数据库密码
   redis:
     password: encrypted:...  # Redis密码
   api:
     secret_key: encrypted:... # API密钥
   ```

3. **密钥轮换**:
   ```bash
   # 1. 使用新密钥重新加密配置文件
   # 2. 更新环境变量/密钥管理服务
   # 3. 重启服务使新密钥生效
   ```

### 7.2 访问控制
```yaml
# security.yaml
admin_users:
  - username: admin
    role: superuser

api_auth:
  secret_key: !ENV ${API_SECRET}
```

### 7.2 审计日志
启用审计日志：
```bash
python -m src.infrastructure.logging.enable_audit --level=INFO
```