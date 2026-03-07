# 🚀 RQA2025量化交易系统快速开始指南

## 🎯 概述

欢迎使用RQA2025量化交易系统！本指南将帮助您在5分钟内完成系统安装和基础配置，开始您的量化交易之旅。

RQA2025是一款企业级的量化交易系统，具有以下核心特性：

- ⚡ **高性能**: 毫秒级响应，支持高频交易
- 🛡️ **高安全**: 金融级安全认证和权限控制
- 🐳 **易部署**: 一键容器化部署
- 📊 **全监控**: 完整的监控和告警体系
- 🔧 **易扩展**: 插件化架构，支持自定义策略

## 📋 前置要求

### 系统要求
- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **内存**: 至少4GB RAM（推荐8GB+）
- **磁盘空间**: 至少10GB可用空间
- **网络**: 稳定的互联网连接

### 软件依赖
- **Docker & Docker Compose**: 用于容器化部署
- **Python 3.9+**: 用于开发和脚本运行（可选）
- **Git**: 用于代码管理（可选）

### 验证环境
```bash
# 检查Docker版本
docker --version
# Docker version 20.10.0+

# 检查Docker Compose版本
docker-compose --version
# docker-compose version 1.29.0+

# 检查Python版本（可选）
python --version
# Python 3.9.0+
```

## 🏗️ 快速安装

### 方法1: 一键容器化部署（推荐）

1. **下载项目**
```bash
git clone https://github.com/your-org/rqa2025.git
cd rqa2025
```

2. **运行快速启动脚本**
```bash
python quick_start.py
```

脚本将自动：
- ✅ 检查系统环境
- ✅ 构建Docker镜像
- ✅ 启动所有服务
- ✅ 配置监控面板
- ✅ 显示访问地址

3. **验证安装成功**
```bash
# 检查服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f rqa-app
```

### 方法2: 手动容器部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看启动日志
docker-compose logs -f
```

### 方法3: 开发环境安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 初始化系统
python -c "from src.main import init_system; init_system()"

# 4. 启动服务
python -m src.main
```

## 🌐 访问系统

安装完成后，您可以通过以下地址访问系统：

### 主应用界面
- **URL**: http://localhost:8000
- **功能**: 系统管理界面、策略配置、交易监控

### 监控面板
- **Grafana**: http://localhost:3000
  - 用户名: admin
  - 密码: admin
- **Prometheus**: http://localhost:9090

### 数据库连接
- **PostgreSQL**: localhost:5432
  - 用户名: rqa
  - 密码: rqa_password
  - 数据库: rqa_db
- **Redis**: localhost:6379

## ⚙️ 基础配置

### 1. 系统初始化

首次访问系统时，需要进行基础配置：

1. 访问 http://localhost:8000
2. 设置管理员账户
3. 配置交易参数
4. 设置监控告警

### 2. 添加交易账户

```python
from rqa2025 import TradingSystem

# 初始化交易系统
system = TradingSystem()

# 添加交易账户
account = {
    "name": "demo_account",
    "broker": "simulator",  # 模拟交易
    "initial_balance": 100000.0,
    "currency": "USD"
}

system.add_account(account)
```

### 3. 配置基础策略

```python
from rqa2025.strategies import MovingAverageStrategy

# 创建移动平均策略
strategy = MovingAverageStrategy(
    symbol="AAPL",
    short_window=5,
    long_window=20,
    position_size=1000
)

# 注册策略
system.register_strategy(strategy)
```

## 📊 核心功能使用

### 策略回测

```python
# 运行策略回测
backtest_config = {
    "strategy": "moving_average",
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_balance": 100000.0
}

result = system.run_backtest(backtest_config)
print(f"回测收益率: {result['total_return']:.2%}")
```

### 实时交易

```python
# 启动实时交易
system.start_trading(
    strategy_name="moving_average",
    account_name="demo_account",
    mode="paper_trading"  # 纸上交易模式
)

# 监控交易状态
status = system.get_trading_status()
print(f"当前持仓: {status['positions']}")
```

### 风险管理

```python
# 配置风险管理规则
risk_config = {
    "max_position_size": 0.1,  # 最大仓位10%
    "max_daily_loss": 0.05,   # 每日最大亏损5%
    "stop_loss": 0.02,        # 止损2%
    "take_profit": 0.05       # 止盈5%
}

system.set_risk_management(risk_config)
```

## 📈 监控和告警

### 系统监控

1. **Grafana面板**: http://localhost:3000
   - 系统性能指标
   - 交易统计数据
   - 风险监控面板

2. **Prometheus指标**: http://localhost:9090
   - 应用健康状态
   - 数据库连接状态
   - 缓存命中率

### 告警配置

```python
# 配置告警规则
alert_config = {
    "daily_loss_threshold": 0.03,  # 日亏损3%告警
    "connection_failure": True,     # 连接失败告警
    "system_cpu_threshold": 80,     # CPU使用率80%告警
    "email_notifications": "admin@company.com"
}

system.configure_alerts(alert_config)
```

## 🛠️ 故障排查

### 常见问题

#### 服务启动失败
```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 检查Docker服务状态
docker-compose ps

# 查看详细日志
docker-compose logs rqa-app
```

#### 数据库连接失败
```bash
# 检查数据库服务
docker-compose logs postgres

# 测试数据库连接
docker exec -it rqa2025-postgres psql -U rqa -d rqa_db
```

#### 内存不足错误
```bash
# 检查系统内存
free -h

# 增加Docker内存限制
# 编辑 docker-compose.yml 增加内存配置
services:
  rqa-app:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### 获取帮助

- 📖 **完整文档**: https://docs.rqa2025.com
- 💬 **社区支持**: https://community.rqa2025.com
- 📧 **技术支持**: support@rqa2025.com
- 🐛 **问题反馈**: https://github.com/rqa2025/issues

## 🎯 下一步

恭喜！您已经成功安装并配置了RQA2025量化交易系统。

### 推荐下一步学习：
1. 📖 阅读[用户指南](./user-guide/)了解详细功能
2. 🎓 学习[策略开发教程](../developer/strategy-development.md)
3. 📊 探索[监控面板使用指南](./monitoring-setup.md)
4. 🚀 尝试[高级配置选项](./advanced-configuration.md)

### 生产环境部署：
- 📋 参考[部署指南](../deployment/production-deployment.md)
- 🔒 配置[安全设置](../deployment/security-setup.md)
- 📈 设置[性能优化](../deployment/performance-tuning.md)

---

**🎉 祝您量化交易之旅顺利！如有问题，请随时联系我们。**

