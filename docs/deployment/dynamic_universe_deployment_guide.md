# 动态宇宙管理系统部署指南

## 🚀 快速开始

### 系统要求
- Python 3.9+
- 内存: 4GB+
- 存储: 10GB+
- 操作系统: Windows/Linux/macOS

### 环境准备
```bash
# 创建conda环境
conda create -n rqa python=3.9
conda activate rqa

# 安装依赖
conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib seaborn
pip install backtrader transformers
```

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd RQA2025
```

### 2. 安装依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装可选依赖（用于高级功能）
pip install transformers seaborn backtrader
```

### 3. 配置环境
```bash
# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## ⚙️ 配置说明

### 1. 基础配置
创建配置文件 `config/dynamic_universe_config.json`:

```json
{
  "universe_manager": {
    "min_liquidity": 1000000,
    "min_market_cap": 1000000000,
    "max_volatility": 0.5,
    "update_interval": 3600
  },
  "intelligent_updater": {
    "performance_threshold": 0.1,
    "volatility_threshold": 0.3,
    "liquidity_threshold": 0.01,
    "time_threshold": 3600
  },
  "weight_adjuster": {
    "adjustment_sensitivity": 1.0,
    "min_weight": 0.05,
    "max_weight": 0.5,
    "base_weights": {
      "fundamental": 0.3,
      "liquidity": 0.25,
      "technical": 0.25,
      "sentiment": 0.1,
      "volatility": 0.1
    }
  }
}
```

### 2. 数据源配置
```json
{
  "data_sources": {
    "market_data": {
      "type": "csv",
      "path": "data/stock/",
      "update_frequency": "daily"
    },
    "fundamental_data": {
      "type": "api",
      "endpoint": "https://api.example.com/fundamental",
      "api_key": "your_api_key"
    }
  }
}
```

## 🔧 部署选项

### 1. 本地部署
```bash
# 运行演示
python examples/dynamic_universe_demo.py

# 运行测试
python -m pytest tests/unit/trading/ -v

# 启动服务
python src/main.py
```

### 2. Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/main.py"]
```

### 3. 生产环境部署
```bash
# 使用gunicorn部署
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 src.main:app

# 使用systemd服务
sudo systemctl enable dynamic-universe
sudo systemctl start dynamic-universe
```

## 📊 监控配置

### 1. 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dynamic_universe.log'),
        logging.StreamHandler()
    ]
)
```

### 2. 性能监控
```python
# 启用性能监控
from src.infrastructure.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()
```

### 3. 告警配置
```python
# 配置告警
from src.infrastructure.alerting import AlertManager

alert_manager = AlertManager(
    email_config={
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "password": "your_password"
    }
)
```

## 🔍 故障排除

### 1. 常见问题

#### 问题1: 依赖包安装失败
```bash
# 解决方案
conda install -c conda-forge transformers seaborn
pip install backtrader --no-deps
```

#### 问题2: 内存不足
```python
# 优化内存使用
import gc
gc.collect()

# 使用数据流处理
def process_data_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]
```

#### 问题3: 性能问题
```python
# 启用性能优化
import numpy as np
import pandas as pd

# 使用向量化操作
def optimize_calculation(data):
    return np.vectorize(calculation_function)(data)
```

### 2. 调试技巧
```python
# 启用调试模式
import logging
logging.getLogger().setLevel(logging.DEBUG)

# 使用性能分析
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# 运行代码
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

## 📈 性能优化

### 1. 代码优化
- 使用向量化操作
- 避免循环中的函数调用
- 使用缓存机制
- 并行处理大数据

### 2. 内存优化
- 及时释放大对象
- 使用生成器处理大数据
- 避免不必要的数据复制
- 使用内存映射文件

### 3. 数据库优化
- 使用索引加速查询
- 批量操作减少IO
- 连接池管理
- 定期清理旧数据

## 🔒 安全配置

### 1. 访问控制
```python
# 配置访问权限
ACCESS_CONTROL = {
    "admin": ["read", "write", "delete"],
    "user": ["read", "write"],
    "guest": ["read"]
}
```

### 2. 数据加密
```python
# 敏感数据加密
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()
```

### 3. 网络安全
```python
# HTTPS配置
SSL_CONTEXT = {
    "certfile": "path/to/cert.pem",
    "keyfile": "path/to/key.pem"
}
```

## 📋 维护计划

### 1. 日常维护
- 检查系统日志
- 监控性能指标
- 备份重要数据
- 更新依赖包

### 2. 定期维护
- 清理临时文件
- 优化数据库
- 更新安全补丁
- 性能调优

### 3. 应急响应
- 制定应急预案
- 建立备份恢复机制
- 配置监控告警
- 准备回滚方案

## ✅ 部署检查清单

### 环境检查
- [ ] Python版本 >= 3.9
- [ ] 依赖包安装完成
- [ ] 环境变量配置正确
- [ ] 数据源连接正常

### 功能检查
- [ ] 单元测试全部通过
- [ ] 集成测试全部通过
- [ ] 演示脚本运行正常
- [ ] 性能指标达标

### 安全检查
- [ ] 访问权限配置
- [ ] 数据加密启用
- [ ] 网络安全设置
- [ ] 日志记录完整

### 监控检查
- [ ] 性能监控启用
- [ ] 告警机制配置
- [ ] 日志轮转设置
- [ ] 备份策略实施

## 🎯 最佳实践

### 1. 开发环境
- 使用虚拟环境隔离依赖
- 定期更新依赖包
- 保持代码风格一致
- 编写完整的测试用例

### 2. 测试环境
- 模拟生产环境配置
- 进行压力测试
- 验证故障恢复能力
- 测试数据迁移

### 3. 生产环境
- 使用负载均衡
- 配置自动扩缩容
- 实施监控告警
- 定期安全审计

---

**文档版本**: v1.0.0  
**最后更新**: 2025-01-25  
**维护团队**: RQA开发团队 