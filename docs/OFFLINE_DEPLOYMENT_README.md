# RQA2025 离线环境部署指南

## 概述

RQA2025支持在无法访问Docker Hub的离线环境中运行，通过本地Python环境替代容器化部署。

## 环境要求

- Python 3.9+
- pip包管理器
- 至少4GB可用磁盘空间
- 至少8GB可用内存

## 快速开始

### 1. 设置离线环境

```bash
# 设置Python虚拟环境和依赖
python scripts/offline_docker_setup.py
```

### 2. 启动服务

```bash
# 启动RQA2025服务
python start_rqa2025.py
```

### 3. 检查服务状态

```bash
# 检查服务健康
python start_rqa2025.py health
```

## 服务架构

### 核心服务
- **主服务**: 基于Flask的Web服务 (端口: 8080)
- **交易引擎**: 量化交易执行服务 (端口: 8081)
- **监控服务**: 系统监控和健康检查 (端口: 8082)

### 数据服务 (本地模拟)
- **SQLite**: 本地数据库替代PostgreSQL
- **本地缓存**: 基于文件的缓存替代Redis
- **本地消息队列**: 基于内存的消息队列替代Kafka

## 服务访问

- 主应用: http://localhost:8080
- 交易服务: http://localhost:8081
- 监控服务: http://localhost:8082

## 功能特性

### 支持的功能
- ✅ 量化策略开发和测试
- ✅ 数据分析和可视化
- ✅ 交易信号生成
- ✅ 风险管理
- ✅ 回测分析
- ✅ 实时监控

### 离线环境的限制
- ❌ 高并发处理能力受限
- ❌ 分布式计算能力有限
- ❌ 持久化存储容量有限
- ❌ 网络服务依赖本地

## 开发模式

### 代码热重载
```bash
# 修改代码后自动重载
export FLASK_ENV=development
python start_rqa2025.py
```

### 日志配置
```bash
# 查看详细日志
tail -f logs/rqa2025.log

# 查看错误日志
tail -f logs/error.log
```

## 配置管理

### 环境变量
编辑 `.env` 文件配置环境变量：

```env
# 开发环境
ENV=development
DEBUG=True

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 应用配置
编辑 `config.ini` 文件：

```ini
[app]
name = RQA2025
version = 1.0.0
debug = true

[database]
type = sqlite
path = ./data/rqa2025.db

[cache]
type = file
path = ./cache
```

## 监控和维护

### 系统监控
```bash
# 查看系统资源使用
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, 内存: {psutil.virtual_memory().percent}%')"

# 查看磁盘使用
python -c "import psutil; print(f'磁盘使用: {psutil.disk_usage(".").percent}%')"
```

### 日志管理
```bash
# 清理旧日志
find logs/ -name "*.log" -mtime +7 -delete

# 压缩日志文件
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### 备份数据
```bash
# 备份应用数据
cp -r data/ backup/data_$(date +%Y%m%d)/

# 备份模型文件
cp -r models/ backup/models_$(date +%Y%m%d)/
```

## 性能优化

### 内存优化
```python
# 在代码中使用内存优化
import gc
gc.collect()  # 手动垃圾回收
```

### 缓存策略
```python
# 使用本地文件缓存
from src.infrastructure.cache.local_cache import LocalCache
cache = LocalCache('./cache')
```

### 数据库优化
```python
# 使用SQLite优化
PRAGMA cache_size = 10000;
PRAGMA synchronous = OFF;
PRAGMA journal_mode = MEMORY;
```

## 故障排除

### 常见问题

#### 1. 端口占用
```bash
# 检查端口占用
netstat -tulpn | grep :8080

# 杀死占用进程
kill -9 <PID>
```

#### 2. 内存不足
```bash
# 清理系统缓存
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 增加交换空间
sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. 磁盘空间不足
```bash
# 清理临时文件
rm -rf temp/*
rm -rf cache/*
rm -rf logs/*.log

# 查看大文件
find . -type f -size +100M -exec ls -lh {} \;
```

### 调试模式
```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python start_rqa2025.py
```

## 扩展部署

### 多进程部署
```python
# 使用gunicorn部署
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### 负载均衡
```bash
# 使用nginx反向代理
sudo apt-get install nginx
# 配置nginx.conf
```

## 安全配置

### 本地安全
- 设置强密码
- 定期备份数据
- 监控系统日志
- 限制文件权限

### 网络安全
- 使用防火墙
- 配置HTTPS
- 定期更新系统
- 监控异常活动

## 维护计划

### 日常维护
- [ ] 检查系统资源使用
- [ ] 查看应用日志
- [ ] 备份重要数据
- [ ] 更新依赖包

### 每周维护
- [ ] 清理临时文件
- [ ] 分析性能指标
- [ ] 检查安全配置
- [ ] 更新文档

### 每月维护
- [ ] 完整系统备份
- [ ] 性能优化评估
- [ ] 安全评估
- [ ] 容量规划

## 支持

### 获取帮助
- 查看项目文档: docs/
- 查看日志文件: logs/
- 查看配置文件: config/
- 运行健康检查: python start_rqa2025.py health

### 社区支持
- 项目GitHub: https://github.com/your-org/rqa2025
- 技术论坛: https://forum.rqa2025.com
- 邮件支持: support@rqa2025.com
