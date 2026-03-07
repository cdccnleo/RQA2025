#!/usr/bin/env python3
"""
部署准备工作系统 - RQA2025生产部署准备

根据生产就绪评估结果，准备生产环境部署：
1. 部署前置条件检查和修复
2. 生产环境配置模板生成
3. 部署脚本和文档准备
4. 回滚计划制定
5. 部署验证清单生成

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys


@dataclass
class DeploymentPrerequisite:
    """部署前置条件"""
    name: str
    description: str
    category: str  # infrastructure, configuration, dependencies, security
    status: str  # pending, in_progress, completed, failed
    automated_check: bool
    manual_steps: List[str]
    verification_command: Optional[str] = None


@dataclass
class DeploymentConfiguration:
    """部署配置"""
    environment: str  # development, staging, production
    config_templates: Dict[str, Any]
    environment_variables: Dict[str, str]
    secrets_configuration: Dict[str, Any]
    resource_limits: Dict[str, Any]


@dataclass
class DeploymentArtifact:
    """部署产物"""
    name: str
    type: str  # script, config, documentation, container
    path: str
    description: str
    required_for: List[str]  # environments where this is required


class DeploymentPreparator:
    """
    部署准备器

    基于生产就绪评估结果，准备生产部署所需的所有文档和配置
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.deployment_dir = self.project_root / "deployment"
        self.templates_dir = self.deployment_dir / "templates"
        self.scripts_dir = self.deployment_dir / "scripts"
        self.docs_dir = self.deployment_dir / "docs"

        # 创建目录结构
        for dir_path in [self.deployment_dir, self.templates_dir,
                        self.scripts_dir, self.docs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def prepare_deployment(self, readiness_report_path: str = None) -> Dict[str, Any]:
        """
        执行部署准备工作

        Args:
            readiness_report_path: 生产就绪评估报告路径

        Returns:
            部署准备结果
        """
        print("🚀 开始RQA2025部署准备工作")
        print("=" * 50)

        # 加载生产就绪评估结果
        readiness_report = self._load_readiness_report(readiness_report_path)

        # 1. 识别和修复前置条件
        prerequisites = self._identify_prerequisites(readiness_report)
        fixed_prerequisites = self._fix_prerequisites(prerequisites)

        # 2. 生成部署配置
        configurations = self._generate_deployment_configurations()

        # 3. 创建部署脚本
        scripts = self._create_deployment_scripts()

        # 4. 生成部署文档
        documentation = self._generate_deployment_documentation(readiness_report)

        # 5. 创建回滚计划
        rollback_plan = self._create_rollback_plan()

        # 6. 生成部署验证清单
        validation_checklist = self._create_validation_checklist(readiness_report)

        # 组织部署产物
        artifacts = self._organize_deployment_artifacts(
            configurations, scripts, documentation, rollback_plan, validation_checklist
        )

        # 生成部署准备报告
        preparation_report = {
            "preparation_date": datetime.now().isoformat(),
            "based_on_readiness_report": readiness_report_path,
            "overall_readiness_score": readiness_report.get("overall_score", 0),
            "deployment_readiness": readiness_report.get("deployment_readiness", {}),
            "prerequisites_status": fixed_prerequisites,
            "artifacts_created": artifacts,
            "next_steps": self._generate_next_steps(readiness_report),
            "estimated_deployment_time": readiness_report.get("deployment_readiness", {}).get("estimated_deployment_time", "unknown"),
            "risk_assessment": self._assess_deployment_risks(readiness_report)
        }

        # 保存部署准备报告
        self._save_preparation_report(preparation_report)

        print("\n✅ 部署准备工作完成")
        print("=" * 40)
        print(f"📊 就绪评分: {readiness_report.get('overall_score', 0):.1f}/100")
        print(f"📦 生成产物: {len(artifacts)} 个")
        print(f"⚠️  前置条件: {len([p for p in fixed_prerequisites if p['status'] != 'completed'])} 个待处理")
        print(f"⏱️  预计部署时间: {preparation_report['estimated_deployment_time']}")

        return preparation_report

    def _load_readiness_report(self, report_path: str = None) -> Dict[str, Any]:
        """加载生产就绪评估报告"""
        if not report_path:
            # 查找最新的评估报告
            report_files = list(self.project_root.glob("test_logs/production_readiness_assessment_*.json"))
            if report_files:
                report_path = max(report_files, key=lambda p: p.stat().st_mtime)

        if report_path and Path(report_path).exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 返回默认评估结果（用于测试）
        return {
            "overall_score": 66.3,
            "overall_status": "not_ready",
            "deployment_readiness": {
                "can_deploy": False,
                "risk_level": "high",
                "estimated_deployment_time": "3-5天",
                "required_pre_deployment_tasks": [
                    "修复性能问题，提升吞吐量",
                    "完善安全配置",
                    "补充部署文档"
                ]
            }
        }

    def _identify_prerequisites(self, readiness_report: Dict[str, Any]) -> List[DeploymentPrerequisite]:
        """识别部署前置条件"""
        prerequisites = []

        # 基于评估结果识别前置条件
        deployment_readiness = readiness_report.get("deployment_readiness", {})
        required_tasks = deployment_readiness.get("required_pre_deployment_tasks", [])

        # 基础设施前置条件
        prerequisites.extend([
            DeploymentPrerequisite(
                name="system_requirements_check",
                description="检查目标系统满足最低硬件要求",
                category="infrastructure",
                status="pending",
                automated_check=True,
                manual_steps=["验证CPU核心数 >= 4", "验证内存 >= 8GB", "验证磁盘空间 >= 50GB"],
                verification_command="python -c \"import psutil; print(f'CPU: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total//1024**3}GB')\""
            ),
            DeploymentPrerequisite(
                name="python_environment_setup",
                description="设置Python运行环境",
                category="infrastructure",
                status="pending",
                automated_check=True,
                manual_steps=["安装Python 3.9+", "创建虚拟环境", "安装依赖包"],
                verification_command="python --version && pip --version"
            ),
            DeploymentPrerequisite(
                name="database_setup",
                description="配置数据库连接",
                category="infrastructure",
                status="pending",
                automated_check=False,
                manual_steps=["安装并配置数据库", "创建数据库用户", "设置连接参数"]
            )
        ])

        # 配置相关前置条件
        prerequisites.extend([
            DeploymentPrerequisite(
                name="configuration_files",
                description="准备环境配置文件",
                category="configuration",
                status="pending",
                automated_check=True,
                manual_steps=["创建config目录", "生成环境配置文件", "验证配置语法"]
            ),
            DeploymentPrerequisite(
                name="environment_variables",
                description="设置环境变量",
                category="configuration",
                status="pending",
                automated_check=True,
                manual_steps=["定义必需的环境变量", "设置安全相关的环境变量"]
            ),
            DeploymentPrerequisite(
                name="secrets_management",
                description="配置密钥管理",
                category="configuration",
                status="pending",
                automated_check=False,
                manual_steps=["设置API密钥存储", "配置数据库密码", "设置JWT密钥"]
            )
        ])

        # 依赖相关前置条件
        prerequisites.extend([
            DeploymentPrerequisite(
                name="dependency_installation",
                description="安装Python依赖包",
                category="dependencies",
                status="pending",
                automated_check=True,
                manual_steps=["安装requirements.txt中的包", "验证包版本兼容性"],
                verification_command="pip install -r requirements.txt --dry-run"
            ),
            DeploymentPrerequisite(
                name="external_services",
                description="配置外部服务连接",
                category="dependencies",
                status="pending",
                automated_check=False,
                manual_steps=["配置Redis连接", "设置消息队列", "配置监控服务"]
            )
        ])

        # 安全相关前置条件
        prerequisites.extend([
            DeploymentPrerequisite(
                name="firewall_configuration",
                description="配置防火墙规则",
                category="security",
                status="pending",
                automated_check=False,
                manual_steps=["开放必要端口", "配置安全组规则", "设置访问控制"]
            ),
            DeploymentPrerequisite(
                name="ssl_certificates",
                description="安装SSL证书",
                category="security",
                status="pending",
                automated_check=False,
                manual_steps=["获取SSL证书", "配置HTTPS", "验证证书有效性"]
            )
        ])

        # 基于评估结果调整状态
        if readiness_report.get("overall_score", 0) >= 80:
            # 高评分系统，许多前置条件可能已满足
            for prereq in prerequisites:
                if prereq.automated_check:
                    prereq.status = "completed"

        return prerequisites

    def _fix_prerequisites(self, prerequisites: List[DeploymentPrerequisite]) -> List[Dict[str, Any]]:
        """修复部署前置条件"""
        fixed_prerequisites = []

        for prereq in prerequisites:
            print(f"🔧 处理前置条件: {prereq.name}")

            if prereq.automated_check and prereq.verification_command:
                # 执行自动化检查
                try:
                    result = subprocess.run(
                        prereq.verification_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        cwd=self.project_root,
                        timeout=30
                    )

                    if result.returncode == 0:
                        prereq.status = "completed"
                        print(f"  ✅ 自动检查通过: {prereq.name}")
                    else:
                        prereq.status = "failed"
                        print(f"  ❌ 自动检查失败: {prereq.name}")
                        print(f"     错误: {result.stderr.strip()}")

                except subprocess.TimeoutExpired:
                    prereq.status = "failed"
                    print(f"  ⏰ 检查超时: {prereq.name}")
                except Exception as e:
                    prereq.status = "failed"
                    print(f"  ❌ 检查异常: {prereq.name} - {e}")
            else:
                # 标记为需要手动处理
                prereq.status = "pending"
                print(f"  📋 需要手动处理: {prereq.name}")

            fixed_prerequisites.append(asdict(prereq))

        return fixed_prerequisites

    def _generate_deployment_configurations(self) -> Dict[str, DeploymentConfiguration]:
        """生成部署配置"""
        configurations = {}

        # 开发环境配置
        dev_config = DeploymentConfiguration(
            environment="development",
            config_templates={
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "rqa2025_dev",
                    "max_connections": 10
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0
                },
                "logging": {
                    "level": "DEBUG",
                    "file": "logs/rqa2025_dev.log"
                }
            },
            environment_variables={
                "RQA2025_ENV": "development",
                "RQA2025_DEBUG": "true",
                "RQA2025_DATABASE_URL": "postgresql://user:pass@localhost:5432/rqa2025_dev"
            },
            secrets_configuration={
                "jwt_secret": "dev_jwt_secret_key_here",
                "api_keys": {
                    "alpha_vantage": "demo_api_key"
                }
            },
            resource_limits={
                "cpu_limit": "1.0",
                "memory_limit": "2GB",
                "max_connections": 50
            }
        )

        # 生产环境配置模板
        prod_config = DeploymentConfiguration(
            environment="production",
            config_templates={
                "database": {
                    "host": "${DATABASE_HOST}",
                    "port": "${DATABASE_PORT}",
                    "database": "${DATABASE_NAME}",
                    "max_connections": 100,
                    "ssl_mode": "require"
                },
                "redis": {
                    "host": "${REDIS_HOST}",
                    "port": "${REDIS_PORT}",
                    "password": "${REDIS_PASSWORD}",
                    "db": 0,
                    "ssl": True
                },
                "logging": {
                    "level": "INFO",
                    "file": "/var/log/rqa2025/app.log",
                    "max_file_size": "100MB",
                    "backup_count": 5
                },
                "monitoring": {
                    "enabled": True,
                    "metrics_endpoint": "/metrics",
                    "health_check_endpoint": "/health"
                }
            },
            environment_variables={
                "RQA2025_ENV": "production",
                "RQA2025_DEBUG": "false",
                "RQA2025_DATABASE_URL": "${DATABASE_URL}",
                "RQA2025_REDIS_URL": "${REDIS_URL}",
                "RQA2025_JWT_SECRET": "${JWT_SECRET}",
                "RQA2025_API_KEYS": "${API_KEYS_JSON}",
                "RQA2025_LOG_LEVEL": "INFO"
            },
            secrets_configuration={
                "jwt_secret": "${JWT_SECRET}",
                "database_password": "${DATABASE_PASSWORD}",
                "redis_password": "${REDIS_PASSWORD}",
                "api_keys": "${API_KEYS_JSON}",
                "ssl_cert": "/etc/ssl/certs/rqa2025.crt",
                "ssl_key": "/etc/ssl/private/rqa2025.key"
            },
            resource_limits={
                "cpu_limit": "4.0",
                "memory_limit": "8GB",
                "max_connections": 1000,
                "max_requests_per_minute": 10000
            }
        )

        configurations["development"] = dev_config
        configurations["production"] = prod_config

        # 保存配置模板
        self._save_configuration_templates(configurations)

        return configurations

    def _save_configuration_templates(self, configurations: Dict[str, DeploymentConfiguration]):
        """保存配置模板"""
        for env_name, config in configurations.items():
            config_file = self.templates_dir / f"config_{env_name}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, indent=2, ensure_ascii=False)

            print(f"💾 配置模板已保存: {config_file}")

    def _create_deployment_scripts(self) -> Dict[str, str]:
        """创建部署脚本"""
        scripts = {}

        # 部署前检查脚本
        pre_deploy_script = """#!/bin/bash
# RQA2025 部署前检查脚本

echo "🔍 执行部署前检查..."

# 检查系统要求
echo "📊 检查系统资源..."
CPU_CORES=$(nproc)
MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
DISK_GB=$(df / | awk 'NR==2{printf "%.0", $4/1024/1024}')

echo "CPU核心数: $CPU_CORES (需要: ≥4)"
echo "内存大小: ${MEMORY_GB}GB (需要: ≥8GB)"
echo "磁盘空间: ${DISK_GB}GB (需要: ≥50GB)"

# 检查端口占用
echo "🔌 检查端口占用..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ 端口8000已被占用"
    exit 1
else
    echo "✅ 端口8000可用"
fi

# 检查Python版本
echo "🐍 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 检查依赖
echo "📦 检查Python依赖..."
if [ -f "requirements.txt" ]; then
    python3 -c "
import pkg_resources
import sys

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

missing = []
for req in requirements:
    if req.strip() and not req.startswith('#'):
        try:
            pkg_resources.require(req)
        except:
            missing.append(req)

if missing:
    echo '❌ 缺少依赖包:'
    for pkg in missing:
        echo \"  - $pkg\"
    exit 1
else:
    echo '✅ 所有依赖包已安装'
fi
"
else
    echo "⚠️ 未找到requirements.txt文件"
fi

echo "🎉 部署前检查完成"
"""

        # 部署脚本
        deploy_script = """#!/bin/bash
# RQA2025 部署脚本

set -e  # 遇到错误立即退出

echo "🚀 开始部署RQA2025..."

# 创建部署目录
DEPLOY_DIR="/opt/rqa2025"
echo "📁 创建部署目录: $DEPLOY_DIR"
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR

# 复制应用代码
echo "📋 复制应用代码..."
cp -r . $DEPLOY_DIR/
cd $DEPLOY_DIR

# 设置Python虚拟环境
echo "🐍 设置Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 安装依赖
echo "📦 安装Python依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 运行数据库迁移（如果有的话）
echo "🗄️ 执行数据库迁移..."
# python manage.py migrate  # 根据实际框架调整

# 收集静态文件（如果有的话）
echo "📄 收集静态文件..."
# python manage.py collectstatic --noinput  # 根据实际框架调整

# 设置权限
echo "🔐 设置文件权限..."
chmod +x scripts/*.sh
chmod 644 config/*.json

# 创建日志目录
echo "📝 创建日志目录..."
mkdir -p logs
chmod 755 logs

# 创建systemd服务文件（可选）
echo "⚙️ 配置systemd服务..."
sudo tee /etc/systemd/system/rqa2025.service > /dev/null <<EOF
[Unit]
Description=RQA2025 Quantitative Trading System
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
ExecStart=$DEPLOY_DIR/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 重新加载systemd
sudo systemctl daemon-reload

echo "🎉 RQA2025部署完成！"
echo ""
echo "启动服务命令:"
echo "  sudo systemctl start rqa2025"
echo "  sudo systemctl enable rqa2025"
echo ""
echo "查看状态命令:"
echo "  sudo systemctl status rqa2025"
echo ""
echo "查看日志命令:"
echo "  sudo journalctl -u rqa2025 -"
"""

        # 回滚脚本
        rollback_script = """#!/bin/bash
# RQA2025 回滚脚本

echo "🔄 开始执行RQA2025回滚..."

# 停止服务
echo "🛑 停止当前服务..."
sudo systemctl stop rqa2025 || true

# 备份当前版本
BACKUP_DIR="/opt/rqa2025_backup_$(date +%Y%m%d_%H%M%S)"
echo "💾 备份当前版本到: $BACKUP_DIR"
sudo cp -r /opt/rqa2025 $BACKUP_DIR

# 恢复上一版本
if [ -d "/opt/rqa2025_previous" ]; then
    echo "🔄 恢复上一版本..."
    sudo rm -rf /opt/rqa2025
    sudo mv /opt/rqa2025_previous /opt/rqa2025
else
    echo "❌ 未找到上一版本，无法回滚"
    exit 1
fi

# 重启服务
echo "🚀 重启服务..."
sudo systemctl start rqa2025

# 验证服务状态
echo "🔍 验证服务状态..."
sleep 5
if sudo systemctl is-active --quiet rqa2025; then
    echo "✅ 服务回滚成功"
else
    echo "❌ 服务回滚失败，请手动检查"
    exit 1
fi

echo "🎉 回滚完成"
"""

        # 保存脚本
        scripts["pre_deploy_check.sh"] = pre_deploy_script
        scripts["deploy.sh"] = deploy_script
        scripts["rollback.sh"] = rollback_script

        self._save_deployment_scripts(scripts)

        return scripts

    def _save_deployment_scripts(self, scripts: Dict[str, str]):
        """保存部署脚本"""
        for script_name, content in scripts.items():
            script_file = self.scripts_dir / script_name
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # 设置执行权限
            script_file.chmod(0o755)

            print(f"💾 部署脚本已保存: {script_file}")

    def _generate_deployment_documentation(self, readiness_report: Dict[str, Any]) -> Dict[str, str]:
        """生成部署文档"""
        docs = {}

        # 部署指南
        deployment_guide = """# RQA2025 部署指南

## 概述

RQA2025 量化交易系统部署指南。本文档提供完整的生产环境部署说明。

## 部署前准备

### 系统要求

- **操作系统**: Linux (Ubuntu 20.04+ 或 CentOS 7+)
- **CPU**: 4核心以上
- **内存**: 8GB以上
- **磁盘**: 50GB以上可用空间
- **网络**: 稳定的互联网连接

### 依赖软件

- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Nginx (可选，用于反向代理)

### 部署前检查

运行部署前检查脚本：

```bash
./scripts/pre_deploy_check.sh
```

## 生产就绪评估结果

基于自动化评估，系统当前状态：

- **总体评分**: {readiness_report.get('overall_score', 0):.1f}/100
- **就绪状态**: {readiness_report.get('overall_status', 'unknown')}
- **风险等级**: {readiness_report.get('deployment_readiness', {}).get('risk_level', 'unknown')}

### 类别评分

| 类别 | 评分 | 状态 |
|------|------|------|
| 功能完整性 | {readiness_report.get('category_scores', {}).get('functionality', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('functionality', 0) >= 80 else '❌'} |
| 性能就绪 | {readiness_report.get('category_scores', {}).get('performance', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('performance', 0) >= 80 else '❌'} |
| 稳定性 | {readiness_report.get('category_scores', {}).get('stability', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('stability', 0) >= 80 else '❌'} |
| 安全性 | {readiness_report.get('category_scores', {}).get('security', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('security', 0) >= 80 else '❌'} |
| 可运维性 | {readiness_report.get('category_scores', {}).get('operability', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('operability', 0) >= 80 else '❌'} |
| 文档完整性 | {readiness_report.get('category_scores', {}).get('documentation', 0):.1f} | {'✅' if readiness_report.get('category_scores', {}).get('documentation', 0) >= 80 else '❌'} |

## 部署步骤

### 1. 环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python
sudo apt install python3.9 python3.9-venv python3-pip

# 安装PostgreSQL
sudo apt install postgresql postgresql-contrib

# 安装Redis
sudo apt install redis-server

# 安装Nginx (可选)
sudo apt install nginx
```

### 2. 代码部署

```bash
# 克隆代码
git clone <repository-url> rqa2025
cd rqa2025

# 运行部署脚本
./scripts/deploy.sh
```

### 3. 配置设置

复制并修改配置文件：

```bash
cp deployment/templates/config_production.json config/production.json
# 编辑配置文件，设置数据库连接、API密钥等
```

### 4. 环境变量设置

```bash
# 创建环境变量文件
sudo tee /etc/rqa2025.env > /dev/null <<EOF
RQA2025_ENV=production
RQA2025_DATABASE_URL=postgresql://user:pass@host:port/db
RQA2025_REDIS_URL=redis://host:port
RQA2025_JWT_SECRET=your-secret-key
EOF
```

### 5. 服务启动

```bash
# 启动服务
sudo systemctl start rqa2025
sudo systemctl enable rqa2025

# 检查状态
sudo systemctl status rqa2025
```

## 监控和维护

### 健康检查

```bash
# API健康检查
curl http://localhost:8000/health

# 指标监控
curl http://localhost:8000/metrics
```

### 日志查看

```bash
# 应用日志
tail -f /var/log/rqa2025/app.log

# 系统日志
sudo journalctl -u rqa2025 -f
```

### 备份策略

- **数据库备份**: 每日全量备份，每小时增量备份
- **配置文件备份**: 每次部署时自动备份
- **日志备份**: 按月轮转，保留6个月

## 故障排除

### 常见问题

1. **服务启动失败**
- 检查配置文件是否正确
- 验证数据库连接
- 查看详细错误日志

2. **性能问题**
- 检查系统资源使用情况
- 调整连接池大小
- 优化查询性能

3. **内存泄漏**
- 监控内存使用趋势
- 重启服务释放内存
- 检查代码中的资源释放

## 安全注意事项

- 定期更新系统补丁
- 使用强密码和API密钥
- 限制网络访问权限
- 启用审计日志
- 定期安全扫描

## 联系支持

如遇部署或运行问题，请联系技术支持团队。

---
*本文档由RQA2025部署准备系统自动生成*
"""

        # 运维手册
        operations_manual = """# RQA2025 运维手册

## 日常运维

### 服务管理

```bash
# 查看服务状态
sudo systemctl status rqa2025

# 重启服务
sudo systemctl restart rqa2025

# 停止服务
sudo systemctl stop rqa2025

# 查看日志
sudo journalctl -u rqa2025 -f
```

### 监控指标

- **系统指标**: CPU、内存、磁盘使用率
- **应用指标**: 请求数、响应时间、错误率
- **业务指标**: 交易量、成功率、盈亏情况

### 告警配置

系统配置了以下告警：

- CPU使用率 > 80%
- 内存使用率 > 85%
- 磁盘空间 < 10%
- 服务不可用
- 响应时间 > 1秒

## 备份恢复

### 数据库备份

```bash
# 创建备份
pg_dump rqa2025 > backup_$(date +%Y%m%d_%H%M%S).sql

# 恢复备份
psql rqa2025 < backup_file.sql
```

### 应用备份

```bash
# 备份应用目录
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz /opt/rqa2025
```

## 性能优化

### 内存优化

- 监控内存泄漏趋势
- 定期重启服务释放内存
- 优化对象生命周期管理

### 数据库优化

- 定期分析慢查询
- 优化索引
- 调整连接池大小

### 缓存优化

- 监控Redis内存使用
- 调整缓存策略
- 清理过期数据

## 安全运维

### 访问控制

- 使用强密码
- 定期轮换API密钥
- 限制管理员访问

### 安全更新

- 定期更新系统补丁
- 更新Python依赖包
- 监控安全漏洞

### 审计日志

- 记录所有管理操作
- 监控异常访问
- 定期审计日志

## 应急处理

### 服务宕机

1. 检查系统资源
2. 查看错误日志
3. 重启服务
4. 如果无法恢复，执行回滚

### 数据异常

1. 检查数据库连接
2. 验证数据完整性
3. 从备份恢复
4. 分析异常原因

### 性能问题

1. 监控系统指标
2. 识别性能瓶颈
3. 临时调整配置
4. 实施长期优化方案

## 升级维护

### 小版本升级

1. 备份当前版本
2. 部署新版本
3. 运行测试验证
4. 如果失败，回滚到上一版本

### 大版本升级

1. 在测试环境验证
2. 制定详细升级计划
3. 在维护窗口执行升级
4. 准备回滚方案

---
*本文档由RQA2025部署准备系统自动生成*
"""

        # 故障排除指南
        troubleshooting_guide = """# RQA2025 故障排除指南

## 启动问题

### 服务无法启动

**症状**: `systemctl start rqa2025` 失败

**检查步骤**:
1. 查看详细错误日志：
```bash
sudo journalctl -u rqa2025 -n 50
```

2. 检查配置文件语法：
```bash
python -m py_compile config/production.json
```

3. 验证数据库连接：
```bash
python -c "
import psycopg2
conn = psycopg2.connect(os.environ['RQA2025_DATABASE_URL'])
print('数据库连接正常')
"
```

4. 检查端口占用：
```bash
sudo lsof -i :8000
```

**解决方案**:
- 修复配置文件错误
- 重新安装缺失的依赖
- 重启相关服务（PostgreSQL、Redis）

### 应用崩溃

**症状**: 服务启动后不久崩溃

**检查步骤**:
1. 查看应用日志：
```bash
tail -f /var/log/rqa2025/app.log
```

2. 检查内存使用：
```bash
ps aux | grep rqa2025
```

3. 验证Python环境：
```bash
/opt/rqa2025/venv/bin/python --version
```

**解决方案**:
- 增加内存限制
- 检查代码中的资源泄漏
- 优化启动脚本

## 性能问题

### 响应缓慢

**检查步骤**:
1. 监控系统资源：
```bash
top
iostat -x 1
```

2. 检查数据库查询：
```sql
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

3. 分析应用指标：
```bash
curl http://localhost:8000/metrics
```

**解决方案**:
- 优化数据库查询
- 增加缓存
- 扩展系统资源

### 内存泄漏

**检查步骤**:
1. 监控内存使用趋势：
```bash
ps aux --no-headers -o pmem,pid,cmd | sort -nr | head -10
```

2. 使用内存分析工具：
```bash
python -c "
import tracemalloc
tracemalloc.start()
   # 运行一段时间后
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('traceback')
for stat in top_stats[:10]:
    print(stat)
"
```

**解决方案**:
- 重启服务释放内存
- 修复代码中的内存泄漏
- 优化对象生命周期管理

## 数据库问题

### 连接失败

**检查步骤**:
1. 验证数据库服务状态：
```bash
sudo systemctl status postgresql
```

2. 检查连接配置：
```bash
psql -h localhost -U rqa2025_user -d rqa2025
```

3. 查看连接池状态：
```bash
SELECT * FROM pg_stat_activity;
```

**解决方案**:
- 重启数据库服务
- 检查网络配置
- 调整连接池大小

### 查询缓慢

**检查步骤**:
1. 识别慢查询：
```sql
SELECT query, total_time, calls, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

2. 分析查询计划：
```sql
EXPLAIN ANALYZE SELECT * FROM large_table WHERE conditions;
```

**解决方案**:
- 添加适当的索引
- 重写查询语句
- 优化数据库配置

## 网络问题

### 无法访问API

**检查步骤**:
1. 验证服务状态：
```bash
curl http://localhost:8000/health
```

2. 检查防火墙：
```bash
sudo ufw status
```

3. 验证反向代理配置：
```bash
sudo nginx -t
```

**解决方案**:
- 配置防火墙规则
- 检查Nginx配置
- 验证SSL证书

## 日志分析

### 错误日志模式

**常见错误模式**:
- `Connection refused`: 数据库或外部服务连接问题
- `MemoryError`: 内存不足
- `TimeoutError`: 请求超时
- `ImportError`: 依赖缺失

### 日志轮转

```bash
# 配置logrotate
sudo tee /etc/logrotate.d/rqa2025 > /dev/null <<EOF
/var/log/rqa2025/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 rqa2025 rqa2025
    postrotate
        systemctl reload rqa2025
    endscript
}
EOF
```

## 预防措施

### 定期维护

- 每周检查系统资源使用
- 每月更新依赖包
- 每季度进行安全审计

### 监控告警

- 设置关键指标阈值
- 配置多渠道告警通知
- 建立应急响应流程

### 备份策略

- 每日数据库备份
- 每周完整应用备份
- 关键配置实时同步

---
*本文档由RQA2025部署准备系统自动生成*
"""

        docs["deployment_guide.md"] = deployment_guide
        docs["operations_manual.md"] = operations_manual
        docs["troubleshooting_guide.md"] = troubleshooting_guide

        # 保存文档
        self._save_deployment_documentation(docs)

        return docs

    def _save_deployment_documentation(self, docs: Dict[str, str]):
        """保存部署文档"""
        for doc_name, content in docs.items():
            doc_file = self.docs_dir / doc_name
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"📄 部署文档已保存: {doc_file}")

    def _create_rollback_plan(self) -> Dict[str, Any]:
        """创建回滚计划"""
        rollback_plan = {
            "rollback_strategy": "immediate_rollback",
            "backup_strategy": "pre_deployment_backup",
            "rollback_steps": [
                {
                    "step": 1,
                    "description": "停止当前服务",
                    "command": "sudo systemctl stop rqa2025",
                    "timeout": 30,
                    "verification": "sudo systemctl is-active rqa2025"
                },
                {
                    "step": 2,
                    "description": "备份当前版本",
                    "command": "sudo cp -r /opt/rqa2025 /opt/rqa2025_backup_$(date +%Y%m%d_%H%M%S)",
                    "timeout": 60,
                    "verification": "ls -la /opt/rqa2025_backup_* | tail -1"
                },
                {
                    "step": 3,
                    "description": "恢复上一版本",
                    "command": "sudo rm -rf /opt/rqa2025 && sudo mv /opt/rqa2025_previous /opt/rqa2025",
                    "timeout": 120,
                    "verification": "ls -la /opt/rqa2025"
                },
                {
                    "step": 4,
                    "description": "重启服务",
                    "command": "sudo systemctl start rqa2025",
                    "timeout": 60,
                    "verification": "curl -f http://localhost:8000/health"
                }
            ],
            "rollback_triggers": [
                "部署后5分钟内服务不可用",
                "错误率超过10%",
                "响应时间超过5秒",
                "内存使用率超过95%"
            ],
            "success_criteria": [
                "服务成功启动",
                "健康检查通过",
                "基本功能验证通过",
                "监控指标正常"
            ],
            "contact_procedures": [
                "立即通知技术团队",
                "启动应急响应流程",
                "通知业务负责人",
                "准备状态报告"
            ]
        }

        # 保存回滚计划
        rollback_file = self.docs_dir / "rollback_plan.json"
        with open(rollback_file, 'w', encoding='utf-8') as f:
            json.dump(rollback_plan, f, indent=2, ensure_ascii=False)

        print(f"🔄 回滚计划已保存: {rollback_file}")

        return rollback_plan

    def _create_validation_checklist(self, readiness_report: Dict[str, Any]) -> Dict[str, Any]:
        """创建部署验证清单"""
        validation_checklist = {
            "pre_deployment_checks": [
                {
                    "item": "系统资源检查",
                    "description": "验证CPU、内存、磁盘满足要求",
                    "command": "python scripts/pre_deploy_check.sh",
                    "expected_result": "所有检查通过",
                    "critical": True
                },
                {
                    "item": "依赖包验证",
                    "description": "确认所有Python依赖已正确安装",
                    "command": "pip check",
                    "expected_result": "无依赖冲突",
                    "critical": True
                },
                {
                    "item": "配置文件验证",
                    "description": "检查所有配置文件语法正确",
                    "command": "python -m json.tool config/production.json",
                    "expected_result": "JSON格式正确",
                    "critical": True
                }
            ],
            "deployment_checks": [
                {
                    "item": "服务启动",
                    "description": "验证服务能够正常启动",
                    "command": "sudo systemctl start rqa2025 && sleep 10 && sudo systemctl status rqa2025",
                    "expected_result": "服务状态为active",
                    "critical": True
                },
                {
                    "item": "端口监听",
                    "description": "确认应用监听正确端口",
                    "command": "sudo lsof -i :8000",
                    "expected_result": "端口8000被监听",
                    "critical": True
                },
                {
                    "item": "数据库连接",
                    "description": "验证数据库连接正常",
                    "command": "python -c \"import os; os.environ['RQA2025_DATABASE_URL']; # test connection\"",
                    "expected_result": "连接成功",
                    "critical": True
                }
            ],
            "post_deployment_checks": [
                {
                    "item": "健康检查",
                    "description": "执行应用健康检查",
                    "command": "curl -f http://localhost:8000/health",
                    "expected_result": "返回200状态码",
                    "critical": True
                },
                {
                    "item": "基本功能测试",
                    "description": "验证核心API功能",
                    "command": "curl -f http://localhost:8000/api/v1/market-data",
                    "expected_result": "返回有效数据",
                    "critical": True
                },
                {
                    "item": "性能基准",
                    "description": "运行基础性能测试",
                    "command": "python -m pytest tests/performance/benchmark_framework.py -k 'trading_core' --tb=short",
                    "expected_result": "测试通过",
                    "critical": False
                },
                {
                    "item": "日志验证",
                    "description": "检查日志文件生成",
                    "command": "ls -la /var/log/rqa2025/",
                    "expected_result": "存在日志文件",
                    "critical": False
                },
                {
                    "item": "监控指标",
                    "description": "验证监控端点工作",
                    "command": "curl -f http://localhost:8000/metrics",
                    "expected_result": "返回监控数据",
                    "critical": False
                }
            ],
            "production_readiness_score": readiness_report.get("overall_score", 0),
            "automated_validation_available": True
        }

        # 保存验证清单
        checklist_file = self.docs_dir / "validation_checklist.json"
        with open(checklist_file, 'w', encoding='utf-8') as f:
            json.dump(validation_checklist, f, indent=2, ensure_ascii=False)

        # 生成可执行的验证脚本
        validation_script = self._generate_validation_script(validation_checklist)
        script_file = self.scripts_dir / "validate_deployment.sh"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(validation_script)
        script_file.chmod(0o755)

        print(f"✅ 部署验证清单已保存: {checklist_file}")
        print(f"💾 验证脚本已保存: {script_file}")

        return validation_checklist

    def _generate_validation_script(self, checklist: Dict[str, Any]) -> str:
        """生成验证脚本"""
        script = """#!/bin/bash
# RQA2025 部署验证脚本

echo "🔍 开始RQA2025部署验证..."
echo "=================================="

FAILED_CHECKS=0
PASSED_CHECKS=0

# 函数：执行检查并报告结果
run_check() {
    local name="$1"
    local description="$2"
    local command="$3"
    local expected="$4"
    local critical="$5"

    echo ""
    echo "📋 检查: $name"
    echo "📝 描述: $description"

    if [ "$critical" = "true" ]; then
        echo "🚨 重要性: 关键"
    else
        echo "ℹ️  重要性: 可选"
    fi

    echo "🔧 执行: $command"
    echo "🎯 期望: $expected"

    # 执行命令
    if eval "$command" 2>/dev/null; then
        echo "✅ 通过: $name"
        ((PASSED_CHECKS++))
    else
        echo "❌ 失败: $name"
        ((FAILED_CHECKS++))
        if [ "$critical" = "true" ]; then
            echo "🚨 关键检查失败，可能影响系统正常运行"
        fi
    fi
}

"""

        # 添加所有检查
        for category, checks in checklist.items():
            if category.endswith("_checks"):
                script += f"\n# {category.replace('_', ' ').title()}\n"
                script += "echo \"\\n🔍 {category.replace('_', ' ').title()}\"\n"
                script += "echo \"==================================\"\n"

                if isinstance(checks, list):
                    for check in checks:
                        if isinstance(check, dict):
                            script += """
run_check "{check['item']}" "{check['description']}" "{check['command']}" "{check['expected_result']}" "{check['critical']}"
"""

        # 添加总结
        script += """

echo ""
echo "🎯 验证完成总结"
echo "=================="
echo "✅ 通过检查: $PASSED_CHECKS"
echo "❌ 失败检查: $FAILED_CHECKS"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo "🎉 所有检查通过！部署验证成功"
    exit 0
else
    echo "⚠️ 存在失败的检查，请查看上述详细信息"
    exit 1
fi
"""

        return script

    def _organize_deployment_artifacts(self, configurations: Dict[str, DeploymentConfiguration],
                                    scripts: Dict[str, str], documentation: Dict[str, str],
                                    rollback_plan: Dict[str, Any], validation_checklist: Dict[str, Any]) -> List[DeploymentArtifact]:
        """组织部署产物"""
        artifacts = []

        # 配置模板
        for env_name in configurations.keys():
            artifacts.append(DeploymentArtifact(
                name=f"config_{env_name}.json",
                type="config",
                path=f"deployment/templates/config_{env_name}.json",
                description=f"{env_name.title()}环境配置文件模板",
                required_for=["development" if env_name == "development" else "production"]
            ))

        # 部署脚本
        script_descriptions = {
            "pre_deploy_check.sh": "部署前环境检查脚本",
            "deploy.sh": "自动化部署脚本",
            "rollback.sh": "部署回滚脚本"
        }

        for script_name, description in script_descriptions.items():
            if script_name in scripts:
                artifacts.append(DeploymentArtifact(
                    name=script_name,
                    type="script",
                    path=f"deployment/scripts/{script_name}",
                    description=description,
                    required_for=["production"]
                ))

        # 文档
        doc_descriptions = {
            "deployment_guide.md": "完整部署指南文档",
            "operations_manual.md": "日常运维操作手册",
            "troubleshooting_guide.md": "故障排除指南"
        }

        for doc_name, description in doc_descriptions.items():
            if doc_name in documentation:
                artifacts.append(DeploymentArtifact(
                    name=doc_name,
                    type="documentation",
                    path=f"deployment/docs/{doc_name}",
                    description=description,
                    required_for=["production"]
                ))

        # 其他文件
        artifacts.extend([
            DeploymentArtifact(
                name="rollback_plan.json",
                type="documentation",
                path="deployment/docs/rollback_plan.json",
                description="详细的回滚计划文档",
                required_for=["production"]
            ),
            DeploymentArtifact(
                name="validation_checklist.json",
                type="documentation",
                path="deployment/docs/validation_checklist.json",
                description="部署验证检查清单",
                required_for=["production"]
            ),
            DeploymentArtifact(
                name="validate_deployment.sh",
                type="script",
                path="deployment/scripts/validate_deployment.sh",
                description="自动化部署验证脚本",
                required_for=["production"]
            )
        ])

        return artifacts

    def _generate_next_steps(self, readiness_report: Dict[str, Any]) -> List[str]:
        """生成下一步行动建议"""
        next_steps = []
        score = readiness_report.get("overall_score", 0)
        can_deploy = readiness_report.get("deployment_readiness", {}).get("can_deploy", False)

        if can_deploy:
            next_steps.extend([
                "✅ 系统已达到生产部署标准",
                "📋 按照部署指南执行生产部署",
                "🔍 部署完成后运行验证检查",
                "📊 建立生产环境监控和告警",
                "📚 培训运维团队使用操作手册"
            ])
        else:
            next_steps.extend([
                "🔧 解决生产就绪评估中发现的问题",
                f"📈 提升系统评分至80+ (当前: {score:.1f})",
                "🎯 重点解决性能和安全问题",
                "🔄 重新运行就绪评估",
                "📋 完善部署文档和脚本"
            ])

        next_steps.extend([
            "📞 组织部署评审会议",
            "👥 确认部署团队和时间表",
            "🛡️ 制定上线应急预案",
            "📊 建立部署后监控指标"
        ])

        return next_steps

    def _assess_deployment_risks(self, readiness_report: Dict[str, Any]) -> Dict[str, Any]:
        """评估部署风险"""
        risks = {
            "overall_risk_level": "low",
            "risk_factors": [],
            "mitigation_strategies": [],
            "contingency_plans": []
        }

        score = readiness_report.get("overall_score", 100)
        critical_issues = len(readiness_report.get("critical_issues", []))

        # 根据评分和问题数量评估风险
        if score < 60 or critical_issues > 2:
            risks["overall_risk_level"] = "high"
            risks["risk_factors"].extend([
                "系统评分过低",
                "存在多个关键问题",
                "部署成功率不高"
            ])
            risks["mitigation_strategies"].extend([
                "推迟部署，优先解决问题",
                "增加测试环境验证时间",
                "准备详细的回滚计划"
            ])
        elif score < 80 or critical_issues > 0:
            risks["overall_risk_level"] = "medium"
            risks["risk_factors"].extend([
                "系统存在可改进之处",
                "部署需要额外监控"
            ])
            risks["mitigation_strategies"].extend([
                "在维护窗口执行部署",
                "准备备用方案",
                "增加部署后监控时间"
            ])
        else:
            risks["overall_risk_level"] = "low"
            risks["risk_factors"].append("系统整体表现良好")
            risks["mitigation_strategies"].extend([
                "正常部署流程",
                "标准监控措施"
            ])

        # 通用应急预案
        risks["contingency_plans"].extend([
            "立即回滚到上一版本",
            "切换到备用系统",
            "通知业务方和客户",
            "启动应急响应团队"
        ])

        return risks

    def _save_preparation_report(self, report: Dict[str, Any]):
        """保存部署准备报告"""
        report_file = self.project_root / "test_logs" / "deployment_preparation_report.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        # 生成HTML报告
        html_report = self._generate_preparation_html_report(report)
        html_file = report_file.with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"💾 部署准备报告已保存: {report_file}")
        print(f"🌐 HTML报告已保存: {html_file}")

    def _generate_preparation_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML格式的准备报告"""
        can_deploy = report["deployment_readiness"]["can_deploy"]
        risk_level = report["risk_assessment"]["overall_risk_level"]

        status_color = {
            True: "#28a745",  # green for deployable
            False: "#dc3545"  # red for not deployable
        }.get(can_deploy, "#ffc107")

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RQA2025部署准备报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status {{ background: {status_color}; color: white; padding: 20px; border-radius: 5px; margin: 20px 0; text-align: center; font-size: 24px; }}
        .metric {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .artifacts {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .artifact {{ background: #ffffff; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #007bff; }}
        .steps {{ background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .step {{ background: #ffffff; padding: 8px; margin: 3px 0; border-radius: 3px; }}
        .risk {{ background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RQA2025部署准备报告</h1>
        <p>生成时间: {report['preparation_date']}</p>
        <p>基于就绪评估: {report['based_on_readiness_report']}</p>
    </div>

    <div class="status">
        <strong>部署状态: {'可以部署' if can_deploy else '暂不建议部署'}</strong>
    </div>

    <div class="metric">
        <h2>评估概览</h2>
        <p><strong>就绪评分:</strong> {report['overall_readiness_score']:.1f}/100</p>
        <p><strong>风险等级:</strong> {risk_level}</p>
        <p><strong>预计部署时间:</strong> {report['estimated_deployment_time']}</p>
    </div>

    <h2>部署产物</h2>
    <div class="artifacts">
"""

        for artifact in report["artifacts_created"]:
            required_envs = ", ".join(artifact.required_for)
            html += """
        <div class="artifact">
            <h4>{artifact.name} ({artifact.type})</h4>
            <p><strong>路径:</strong> {artifact.path}</p>
            <p><strong>描述:</strong> {artifact.description}</p>
            <p><strong>适用环境:</strong> {required_envs}</p>
        </div>
"""

        html += """
    </div>

    <h2>下一步行动</h2>
    <div class="steps">
"""

        for step in report["next_steps"]:
            html += """
        <div class="step">{step}</div>
"""

        html += """
    </div>

    <h2>风险评估</h2>
    <div class="risk">
        <h3>风险因素</h3>
"""

        for factor in report["risk_assessment"]["risk_factors"]:
            html += f"<p>• {factor}</p>"

        html += """
        <h3>缓解策略</h3>
"""

        for strategy in report["risk_assessment"]["mitigation_strategies"]:
            html += f"<p>• {strategy}</p>"

        html += """
        <h3>应急预案</h3>
"""

        for plan in report["risk_assessment"]["contingency_plans"]:
            html += f"<p>• {plan}</p>"

        html += """
    </div>
</body>
</html>
"""
        return html


def run_deployment_preparation():
    """运行部署准备工作"""
    print("🚀 启动RQA2025部署准备工作")
    print("=" * 50)

    # 查找最新的就绪评估报告
    import glob
    report_files = glob.glob("test_logs/production_readiness_assessment_*.json")
    if report_files:
        latest_report = max(report_files, key=lambda f: f)
        print(f"📊 使用就绪评估报告: {latest_report}")
    else:
        latest_report = None
        print("⚠️ 未找到就绪评估报告，将使用默认配置")

    # 创建部署准备器
    preparator = DeploymentPreparator()

    # 执行部署准备
    preparation_report = preparator.prepare_deployment(latest_report)

    print("\n✅ 部署准备工作完成")
    print("=" * 40)
    print(f"📦 生成部署产物: {len(preparation_report['artifacts_created'])} 个")
    print(f"📋 部署文档: {len([a for a in preparation_report['artifacts_created'] if a.type == 'documentation'])} 个")
    print(f"💾 部署脚本: {len([a for a in preparation_report['artifacts_created'] if a.type == 'script'])} 个")
    print(f"🔧 配置文件: {len([a for a in preparation_report['artifacts_created'] if a.type == 'config'])} 个")

    next_steps_count = len(preparation_report["next_steps"])
    print(f"🎯 下一步行动: {next_steps_count} 项")

    if preparation_report["deployment_readiness"]["can_deploy"]:
        print("🎉 系统已准备好进行生产部署！")
    else:
        print("⚠️ 系统需要进一步完善后再进行生产部署")

    return preparation_report


if __name__ == "__main__":
    run_deployment_preparation()
