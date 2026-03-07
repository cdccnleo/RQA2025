#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 离线Docker环境设置脚本

在无法访问Docker Hub的环境中设置项目运行环境
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_offline_environment():
    """设置离线运行环境"""
    project_root = Path(__file__).parent.parent

    print("🔧 设置RQA2025离线运行环境...")

    # 1. 创建本地Python环境
    setup_python_environment(project_root)

    # 2. 安装依赖
    install_dependencies(project_root)

    # 3. 配置项目环境
    configure_project_environment(project_root)

    # 4. 验证环境
    validate_environment(project_root)

    print("\\n✅ 离线环境设置完成!")
    print("\\n📋 现在可以运行以下命令启动服务:")
    print("  python scripts/run_distributed_system.py")


def setup_python_environment(project_root):
    """设置Python环境"""
    print("🐍 设置Python环境...")

    try:
        # 检查Python版本
        result = subprocess.run([sys.executable, "--version"],
                                capture_output=True, text=True)
        print(f"✅ Python版本: {result.stdout.strip()}")

        # 创建虚拟环境
        venv_path = project_root / "venv"
        if not venv_path.exists():
            print("📦 创建虚拟环境...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)],
                           check=True)
            print("✅ 虚拟环境创建完成")

        # 激活虚拟环境并升级pip
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:  # Linux/Mac
            pip_path = venv_path / "bin" / "pip"

        print("📦 升级pip...")
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"],
                       check=True)
        print("✅ pip升级完成")

    except subprocess.CalledProcessError as e:
        print(f"❌ Python环境设置失败: {e}")
        return False

    return True


def install_dependencies(project_root):
    """安装项目依赖"""
    print("📦 安装项目依赖...")

    venv_path = project_root / "venv"
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Linux/Mac
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    try:
        # 安装核心依赖
        requirements_path = project_root / "requirements.txt"
        if requirements_path.exists():
            print("📦 安装Python依赖...")
            subprocess.run([
                str(pip_path), "install", "-r", str(requirements_path)
            ], check=True)
            print("✅ 依赖安装完成")
        else:
            print("⚠️  未找到requirements.txt文件")

        # 安装项目本身
        setup_path = project_root / "setup.py"
        if setup_path.exists():
            print("📦 安装项目...")
            subprocess.run([
                str(pip_path), "install", "-e", str(project_root)
            ], check=True)
            print("✅ 项目安装完成")

    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

    return True


def configure_project_environment(project_root):
    """配置项目环境"""
    print("⚙️ 配置项目环境...")

    # 创建必要的目录
    directories = [
        "logs",
        "data",
        "models",
        "cache",
        "temp",
        "config"
    ]

    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

    # 创建配置文件
    create_config_files(project_root)

    print("✅ 项目环境配置完成")


def create_config_files(project_root):
    """创建配置文件"""
    print("📝 创建配置文件...")

    # 创建环境配置文件
    env_content = """# RQA2025 环境配置
ENV=development
DEBUG=True
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rqa2025
POSTGRES_USER=rqa2025_user
POSTGRES_PASSWORD=rqa2025_password

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Kafka配置
KAFKA_HOST=localhost
KAFKA_PORT=9092

# API配置
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4

# 监控配置
PROMETHEUS_ENABLED=True
GRAFANA_ENABLED=True

# 安全配置
JWT_SECRET=your-jwt-secret-here
JWT_EXPIRATION=3600

# 数据配置
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs
CACHE_DIR=./cache
"""

    env_file = project_root / ".env"
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)

    print(f"✅ 创建环境配置文件: {env_file}")


def validate_environment(project_root):
    """验证环境"""
    print("🔍 验证环境配置...")

    venv_path = project_root / "venv"
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Linux/Mac
        python_path = venv_path / "bin" / "python"

    try:
        # 测试Python环境
        result = subprocess.run([
            str(python_path), "-c",
            "import sys; print(f'Python版本: {sys.version}')"
        ], capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")

        # 测试关键依赖
        test_imports = [
            "import flask",
            "import pandas",
            "import numpy",
            "import sklearn",
            "import tensorflow"
        ]

        for import_test in test_imports:
            try:
                result = subprocess.run([
                    str(python_path), "-c", import_test
                ], capture_output=True, text=True, check=True)
                module_name = import_test.split()[-1]
                print(f"✅ 模块可用: {module_name}")
            except subprocess.CalledProcessError:
                module_name = import_test.split()[-1]
                print(f"⚠️  模块不可用: {module_name}")

        # 测试项目导入
        try:
            result = subprocess.run([
                str(python_path), "-c",
                "from src.infrastructure.health_check import HealthChecker; print('项目模块导入成功')"
            ], cwd=project_root, capture_output=True, text=True, check=True)
            print("✅ 项目模块导入成功")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  项目模块导入失败: {e}")

    except Exception as e:
        print(f"❌ 环境验证失败: {e}")
        return False

    return True


def create_startup_script(project_root):
    """创建启动脚本"""
    print("📝 创建启动脚本...")

    startup_script = project_root / "start_rqa2025.py"
    startup_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
RQA2025 本地启动脚本

在离线环境中启动RQA2025服务
\"\"\"

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def start_services():
    \"\"\"启动RQA2025服务\"\"\"    project_root = Path(__file__).parent

    print("🚀 启动RQA2025服务...")

    # 激活虚拟环境
    venv_path = project_root / "venv"
    if os.name == 'nt':  # Windows
        python_path = venv_path / "Scripts" / "python.exe"
        activate_script = venv_path / "Scripts" / "activate.bat"
    else:  # Linux/Mac
        python_path = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"

    if not python_path.exists():
        print("❌ 虚拟环境未找到，请先运行离线环境设置脚本")
        return False

    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root / "src")
        env['RQA2025_ENV'] = 'development'

        # 启动主服务
        print("📊 启动主服务...")
        main_service = subprocess.Popen([
            str(python_path), "scripts/run_distributed_system.py"
        ], cwd=project_root, env=env)

        print("✅ 主服务已启动")

        # 等待服务启动
        time.sleep(5)

        print("\\n🎉 RQA2025服务启动成功!")
        print("\\n📋 服务信息:")
        print("  - 主服务PID:", main_service.pid)
        print("  - 工作目录:", project_root)
        print("  - 环境: 离线开发环境")

        print("\\n📝 停止服务请按 Ctrl+C")

        try:
            main_service.wait()
        except KeyboardInterrupt:
            print("\\n🛑 正在停止服务...")
            main_service.terminate()
            main_service.wait()
            print("✅ 服务已停止")

    except Exception as e:
        print(f"❌ 启动服务失败: {e}")
        return False

    return True

def check_service_health():
    \"\"\"检查服务健康\"\"\"    try:
        import requests
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
            return True
        else:
            print("⚠️  服务健康检查失败")
            return False
    except Exception as e:
        print(f"⚠️  无法连接到服务: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        check_service_health()
    else:
        start_services()
"""

    with open(startup_script, 'w', encoding='utf-8') as f:
        f.write(startup_content)

    print(f"✅ 创建启动脚本: {startup_script}")


def create_offline_readme(project_root):
    """创建离线环境说明"""
    print("📖 创建离线环境使用说明...")

    readme_content = """# RQA2025 离线环境部署指南

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
python -c "import psutil; print(f'磁盘使用: {psutil.disk_usage(\".\").percent}%')"
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
"""

    readme_path = project_root / "OFFLINE_DEPLOYMENT_README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✅ 创建离线环境说明: {readme_path}")


def main():
    """主函数"""
    print("RQA2025 离线环境设置工具")
    print("=" * 50)

    success = setup_offline_environment()
    create_startup_script(Path(__file__).parent.parent)
    create_offline_readme(Path(__file__).parent.parent)

    if success:
        print("\\n🎉 离线环境设置完成!")
        print("\\n📋 现在可以运行以下命令启动服务:")
        print("  python start_rqa2025.py")
    else:
        print("\\n❌ 离线环境设置失败!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
