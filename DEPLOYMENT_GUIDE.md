# RQA2025 部署指南

## 📋 概述

本指南将帮助您部署RQA2025量化研究和算法交易系统。系统已通过全面测试，达到生产就绪状态。

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- **操作系统**: Windows 10/11, Linux, macOS
- **Python版本**: 3.8+
- **内存**: 8GB+ (推荐16GB)
- **存储**: 10GB+ 可用空间
- **网络**: 稳定的互联网连接

#### 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd RQA2025

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

#### 基础配置
创建配置文件 `config/local_config.ini`:
```ini
[data]
cache_dir = cache
max_retries = 3
timeout = 30

[api]
host = 0.0.0.0
port = 8000
debug = false

[monitoring]
enabled = true
interval = 30

[quality]
enabled = true
threshold = 0.8
```

#### 数据源配置 (可选)
```ini
[crypto]
coingecko_api_key = your_api_key_here
binance_api_key = your_api_key_here
binance_api_secret = your_api_secret_here

[macro]
fred_api_key = your_api_key_here
worldbank_api_key = your_api_key_here
```

### 3. 启动服务

#### 开发环境
```bash
# 启动Web API服务
python -m uvicorn src.infrastructure.web.app_factory:create_app --host 0.0.0.0 --port 8000 --reload

# 启动性能监控
python scripts/performance/performance_optimization.py

# 运行功能测试
python scripts/feature_extension/test_feature_extension.py
```

#### 生产环境
```bash
# 使用Docker (推荐)
docker-compose up -d

# 或使用Gunicorn
gunicorn src.infrastructure.web.app_factory:create_app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🔧 详细部署步骤

### 步骤1: 环境检查

```bash
# 检查Python版本
python --version

# 检查依赖
pip list

# 运行健康检查
python -c "from src.engine.web.data_api import router; print('✅ API模块加载成功')"
```

### 步骤2: 数据层初始化

```bash
# 创建缓存目录
mkdir -p cache

# 初始化数据管理器
python -c "
from src.data.data_manager import DataManager
dm = DataManager()
print('✅ 数据管理器初始化成功')
"
```

### 步骤3: API服务启动

```bash
# 启动API服务
python -m uvicorn src.infrastructure.web.app_factory:create_app --host 0.0.0.0 --port 8000

# 验证服务
curl http://localhost:8000/api/v1/data/health
```

### 步骤4: 功能验证

```bash
# 运行完整测试套件
python scripts/feature_extension/test_api_integration.py

# 测试数据加载
python -c "
import asyncio
from src.data.loader.crypto_loader import CryptoDataLoader
loader = CryptoDataLoader({'cache_dir': 'cache'})
result = asyncio.run(loader.load_data())
print(f'✅ 数据加载测试成功: {len(result[\"data\"])} 条记录')
"
```

## 📊 监控和验证

### 1. 健康检查
```bash
# API健康检查
curl http://localhost:8000/api/v1/data/health

# 性能指标
curl http://localhost:8000/api/v1/data/performance

# 质量报告
curl http://localhost:8000/api/v1/data/quality/report
```

### 2. 日志监控
```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

### 3. 性能监控
```bash
# 运行性能优化
python scripts/performance/performance_optimization.py

# 查看性能报告
cat reports/performance_optimization_*.json
```

## 🔐 安全配置

### 1. API安全
```python
# 添加API密钥验证
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_api_key(credentials = Depends(security)):
    if credentials.credentials != "your_api_key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
```

### 2. 数据加密
```python
# 启用数据加密
from cryptography.fernet import Fernet

# 生成密钥
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

### 3. 访问控制
```python
# 添加CORS配置
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 🐳 Docker部署

### 1. 构建镜像
```bash
# 构建Docker镜像
docker build -t rqa2025:latest .

# 运行容器
docker run -d -p 8000:8000 --name rqa2025 rqa2025:latest
```

### 2. Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  rqa2025:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
    restart: unless-stopped
```

## 📈 性能优化

### 1. 缓存优化
```python
# 配置多级缓存
cache_config = CacheConfig(
    max_size=5000,
    ttl=1800,
    enable_disk_cache=True,
    compression=True
)
```

### 2. 并发优化
```python
# 配置异步处理
import asyncio
import aiohttp

# 设置并发限制
semaphore = asyncio.Semaphore(10)
```

### 3. 内存优化
```python
# 配置内存管理
import gc

# 定期垃圾回收
gc.collect()
```

## 🔍 故障排除

### 常见问题

#### 1. 端口占用
```bash
# 检查端口占用
netstat -tulpn | grep 8000

# 杀死进程
kill -9 <PID>
```

#### 2. 依赖问题
```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 清理缓存
pip cache purge
```

#### 3. 配置问题
```bash
# 验证配置
python -c "
import configparser
config = configparser.ConfigParser()
config.read('config/local_config.ini')
print('✅ 配置文件验证成功')
"
```

### 日志分析
```bash
# 查看错误日志
grep "ERROR" logs/app.log

# 查看性能日志
grep "performance" logs/app.log
```

## 📞 支持

### 联系方式
- **技术问题**: 查看项目文档
- **部署问题**: 检查日志文件
- **性能问题**: 运行性能测试

### 有用链接
- [API文档](http://localhost:8000/docs)
- [项目文档](docs/)
- [测试报告](reports/)

## 🎉 部署完成

恭喜！RQA2025系统已成功部署。您现在可以：

1. **访问API**: http://localhost:8000/api/v1/data/health
2. **查看文档**: http://localhost:8000/docs
3. **运行测试**: `python scripts/feature_extension/test_api_integration.py`
4. **监控性能**: `python scripts/performance/performance_optimization.py`

### 下一步
- 配置生产环境
- 设置监控告警
- 优化性能参数
- 扩展数据源

---

**部署指南版本**: v1.0  
**最后更新**: 2025-07-31  
**适用版本**: RQA2025 功能扩展完成版 