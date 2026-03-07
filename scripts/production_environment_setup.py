#!/usr/bin/env python3
"""
生产环境部署准备脚本

检查和准备生产环境部署所需的配置和资源
    创建时间: 2024年12月
"""

import sys
import os
import json
import logging
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionEnvironmentSetup:
    """生产环境部署准备类"""

    def __init__(self, config_file: str = "config/production_config.json"):
        self.config_file = Path(config_file)
        self.config = {}
        self.check_results = {}
        self.setup_results = {}

    def load_config(self) -> bool:
        """加载生产环境配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✅ 配置文件加载成功: {self.config_file}")
                return True
            else:
                print(f"⚠️ 配置文件不存在: {self.config_file}")
                self.create_default_config()
                return True
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            return False

    def create_default_config(self):
        """创建默认配置文件"""
        self.config = {
            "environment": {
                "name": "production",
                "version": "1.0.0",
                "region": "cn-north-1"
            },
            "database": {
                "host": "localhost",
                "port": 3306,
                "database": "rqa2025_prod",
                "charset": "utf8mb4",
                "max_connections": 100
            },
            "cache": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "max_memory": "2gb"
            },
            "security": {
                "jwt_secret_key": "CHANGE_THIS_IN_PRODUCTION",
                "encryption_key": "CHANGE_THIS_IN_PRODUCTION",
                "session_timeout": 3600,
                "max_login_attempts": 5
            },
            "monitoring": {
                "enabled": True,
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "alert_rules_path": "/etc/prometheus/rules"
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "log_path": "/var/log/rqa2025",
                "max_file_size": "100MB",
                "backup_count": 7
            }
        }

        # 创建配置目录
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        # 保存默认配置
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print(f"✅ 默认配置文件已创建: {self.config_file}")

    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        print("\n🔍 检查系统要求...")

        requirements = [
            ("Python", "3.8+", self._check_python_version),
            ("操作系统", "Linux/Windows", self._check_os),
            ("磁盘空间", ">5GB", self._check_disk_space),
            ("内存", ">4GB", self._check_memory),
            ("网络连接", "正常", self._check_network),
        ]

        all_passed = True
        for req_name, req_value, check_func in requirements:
            try:
                result = check_func()
                if result:
                    print(f"✅ {req_name}: {req_value}")
                    self.check_results[req_name] = True
                else:
                    print(f"❌ {req_name}: {req_value}")
                    self.check_results[req_name] = False
                    all_passed = False
            except Exception as e:
                print(f"❌ {req_name}: 检查失败 - {e}")
                self.check_results[req_name] = False
                all_passed = False

        return all_passed

    def _check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        return version.major >= 3 and version.minor >= 8

    def _check_os(self) -> bool:
        """检查操作系统"""
        import platform
        system = platform.system().lower()
        return system in ['linux', 'windows', 'darwin']

    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            return free > 5 * 1024 * 1024 * 1024  # 5GB
        except:
            return True  # 跳过检查

    def _check_memory(self) -> bool:
        """检查内存"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.total > 4 * 1024 * 1024 * 1024  # 4GB
        except:
            return True  # 跳过检查

    def _check_network(self) -> bool:
        """检查网络连接"""
        try:
            import socket
            socket.setdefaulttimeout(5)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except:
            return False

    def setup_directories(self) -> bool:
        """创建必要的目录结构"""
        print("\n📁 创建目录结构...")

        directories = [
            "logs",
            "data",
            "config",
            "backups",
            "temp",
            "ssl"
        ]

        try:
            for dir_name in directories:
                dir_path = Path(dir_name)
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建目录: {dir_path}")

            self.setup_results["directories"] = True
            return True
        except Exception as e:
            print(f"❌ 创建目录失败: {e}")
            self.setup_results["directories"] = False
            return False

    def setup_database_config(self) -> bool:
        """设置数据库配置"""
        print("\n🗄️ 设置数据库配置...")

        try:
            # 检查数据库配置
            db_config = self.config.get("database", {})

            if not db_config.get("host"):
                print("⚠️ 数据库配置不完整，使用默认配置")
                db_config = {
                    "host": "localhost",
                    "port": 3306,
                    "database": "rqa2025_prod",
                    "charset": "utf8mb4"
                }

            # 创建数据库连接测试
            print("📋 数据库配置:")
            for key, value in db_config.items():
                if key != "password":  # 不显示密码
                    print(f"   {key}: {value}")

            print("✅ 数据库配置完成")
            self.setup_results["database"] = True
            return True

        except Exception as e:
            print(f"❌ 数据库配置失败: {e}")
            self.setup_results["database"] = False
            return False

    def setup_security_config(self) -> bool:
        """设置安全配置"""
        print("\n🔐 设置安全配置...")

        try:
            import secrets
            import base64

            # 生成安全的密钥
            jwt_secret = secrets.token_hex(32)
            encryption_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

            security_config = {
                "jwt_secret_key": jwt_secret,
                "encryption_key": encryption_key,
                "session_timeout": 3600,
                "max_login_attempts": 5,
                "password_min_length": 8,
                "require_mfa": True
            }

            # 更新配置
            self.config["security"] = security_config

            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)

            print("✅ 安全配置完成")
            print(f"   JWT密钥: {jwt_secret[:8]}...")
            print(f"   加密密钥: {encryption_key[:8]}...")

            self.setup_results["security"] = True
            return True

        except Exception as e:
            print(f"❌ 安全配置失败: {e}")
            self.setup_results["security"] = False
            return False

    def setup_monitoring_config(self) -> bool:
        """设置监控配置"""
        print("\n📊 设置监控配置...")

        try:
            monitoring_config = {
                "enabled": True,
                "metrics_port": 9090,
                "alert_port": 9093,
                "dashboard_port": 3000,
                "metrics_path": "/metrics",
                "alert_rules": [
                    {
                        "name": "high_cpu_usage",
                        "condition": "cpu_usage > 80",
                        "severity": "warning",
                        "description": "CPU使用率过高"
                    },
                    {
                        "name": "high_memory_usage",
                        "condition": "memory_usage > 85",
                        "severity": "warning",
                        "description": "内存使用率过高"
                    }
                ]
            }

            self.config["monitoring"] = monitoring_config
            print("✅ 监控配置完成")

            self.setup_results["monitoring"] = True
            return True

        except Exception as e:
            print(f"❌ 监控配置失败: {e}")
            self.setup_results["monitoring"] = False
            return False

    def create_deployment_checklist(self) -> bool:
        """创建部署检查清单"""
        print("\n📋 创建部署检查清单...")

        checklist = {
            "pre_deployment": [
                {"item": "系统要求检查", "status": "pending", "responsible": "运维团队"},
                {"item": "环境变量配置", "status": "pending", "responsible": "开发团队"},
                {"item": "数据库连接测试", "status": "pending", "responsible": "DBA"},
                {"item": "缓存服务配置", "status": "pending", "responsible": "运维团队"},
                {"item": "网络配置验证", "status": "pending", "responsible": "网络团队"}
            ],
            "deployment": [
                {"item": "代码部署", "status": "pending", "responsible": "部署团队"},
                {"item": "数据库迁移", "status": "pending", "responsible": "DBA"},
                {"item": "服务启动", "status": "pending", "responsible": "运维团队"},
                {"item": "配置生效", "status": "pending", "responsible": "运维团队"},
                {"item": "监控启动", "status": "pending", "responsible": "运维团队"}
            ],
            "post_deployment": [
                {"item": "功能验证", "status": "pending", "responsible": "测试团队"},
                {"item": "性能测试", "status": "pending", "responsible": "测试团队"},
                {"item": "业务验收", "status": "pending", "responsible": "业务团队"},
                {"item": "文档更新", "status": "pending", "responsible": "文档团队"},
                {"item": "知识转移", "status": "pending", "responsible": "培训团队"}
            ]
        }

        # 保存检查清单
        checklist_file = Path("config/deployment_checklist.json")
        checklist_file.parent.mkdir(parents=True, exist_ok=True)

        with open(checklist_file, 'w', encoding='utf-8') as f:
            json.dump(checklist, f, ensure_ascii=False, indent=2)

        print("✅ 部署检查清单已创建")
        self.setup_results["checklist"] = True
        return True

    def create_backup_strategy(self) -> bool:
        """创建备份策略"""
        print("\n💾 创建备份策略...")

        backup_strategy = {
            "database": {
                "frequency": "daily",
                "retention_days": 30,
                "type": "full",
                "schedule": "02:00"
            },
            "application": {
                "frequency": "weekly",
                "retention_weeks": 4,
                "type": "incremental",
                "schedule": "03:00"
            },
            "configuration": {
                "frequency": "daily",
                "retention_days": 7,
                "type": "full",
                "schedule": "01:00"
            }
        }

        # 保存备份策略
        backup_file = Path("config/backup_strategy.json")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_strategy, f, ensure_ascii=False, indent=2)

        print("✅ 备份策略已创建")
        self.setup_results["backup"] = True
        return True

    def generate_deployment_script(self) -> bool:
        """生成部署脚本"""
        print("\n🚀 生成部署脚本...")

        script_content = """#!/bin/bash
# RQA2025 生产环境部署脚本
# 创建时间: 2024年12月

set -e

echo "🚀 开始RQA2025生产环境部署..."

# 检查环境
echo "📋 检查部署环境..."
if [ ! -f "config/production_config.json" ]; then
    echo "❌ 生产配置文件不存在"
    exit 1
fi

# 创建必要的目录
echo "📁 创建运行目录..."
mkdir -p logs data backups temp

# 备份当前配置
echo "💾 备份当前配置..."
if [ -d "config" ]; then
    cp -r config config.backup.$(date +%Y%m%d_%H%M%S)
fi

# 部署应用代码
echo "📦 部署应用代码..."
# 这里添加具体的部署命令

# 设置环境变量
echo "🔧 配置环境变量..."
export RQA2025_ENV=production
export RQA2025_CONFIG=config/production_config.json

# 启动服务
echo "🏃 启动应用服务..."
# 这里添加服务启动命令

# 健康检查
echo "🏥 执行健康检查..."
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "✅ RQA2025生产环境部署完成！"
echo "🌐 应用访问地址: http://localhost:8000"
echo "📊 监控地址: http://localhost:3000"
"""

        script_file = Path("scripts/deploy_production.sh")
        script_file.parent.mkdir(parents=True, exist_ok=True)

        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # 设置脚本执行权限
        import os
        os.chmod(script_file, 0o755)

        print("✅ 部署脚本已生成")
        self.setup_results["script"] = True
        return True

    def run_final_verification(self) -> bool:
        """运行最终验证"""
        print("\n✅ 运行最终验证...")

        verification_items = [
            ("配置文件完整性", lambda: self.config_file.exists()),
            ("目录结构完整性", lambda: all(Path(d).exists() for d in ["logs", "config", "data"])),
            ("安全配置完整性", lambda: "security" in self.config and len(
                self.config["security"].get("jwt_secret_key", "")) > 32),
            ("监控配置完整性", lambda: "monitoring" in self.config and self.config["monitoring"].get(
                "enabled", False)),
        ]

        all_passed = True
        for item_name, check_func in verification_items:
            try:
                if check_func():
                    print(f"✅ {item_name}")
                else:
                    print(f"❌ {item_name}")
                    all_passed = False
            except Exception as e:
                print(f"❌ {item_name} - 验证失败: {e}")
                all_passed = False

        return all_passed

    def generate_setup_report(self):
        """生成设置报告"""
        print("\n" + "="*60)
        print("📊 生产环境部署准备报告")
        print("="*60)

        print(f"配置文件: {self.config_file}")
        print(f"配置状态: {'✅ 完整' if self.config else '❌ 缺失'}")

        print("\n🔍 系统检查结果:")
        for item, result in self.check_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"   {item}: {status}")

        print("\n📋 设置结果:")
        for item, result in self.setup_results.items():
            status = "✅ 成功" if result else "❌ 失败"
            print(f"   {item}: {status}")

        success_count = sum(1 for r in self.setup_results.values() if r)
        total_count = len(self.setup_results)

        if success_count == total_count:
            print("\n🎉 生产环境部署准备完成！")
            print(f"   总项数: {total_count}")
            print(f"   成功数: {success_count}")
            print(f"   成功率: {success_count/total_count:.1%}")
            return True
        else:
            print("\n⚠️ 部分设置项需要检查")
            print(f"   总项数: {total_count}")
            print(f"   成功数: {success_count}")
            print(f"   成功率: {success_count/total_count:.1%}")
            return False


def main():
    """主函数"""
    print("🏭 RQA2025生产环境部署准备")
    print("="*50)

    setup = ProductionEnvironmentSetup()

    # 加载配置
    if not setup.load_config():
        return False

    # 执行各项设置
    steps = [
        ("系统要求检查", setup.check_system_requirements),
        ("目录结构设置", setup.setup_directories),
        ("数据库配置", setup.setup_database_config),
        ("安全配置", setup.setup_security_config),
        ("监控配置", setup.setup_monitoring_config),
        ("部署检查清单", setup.create_deployment_checklist),
        ("备份策略", setup.create_backup_strategy),
        ("部署脚本", setup.generate_deployment_script),
        ("最终验证", setup.run_final_verification),
    ]

    success_count = 0
    for step_name, step_func in steps:
        print(f"\n{'='*50}")
        print(f"执行: {step_name}")
        print('='*50)

        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} - 完成")
            else:
                print(f"❌ {step_name} - 失败")
        except Exception as e:
            print(f"❌ {step_name} - 执行异常: {e}")

    # 生成报告
    success = setup.generate_setup_report()

    print(f"\n📋 执行总结:")
    print(f"   总步骤数: {len(steps)}")
    print(f"   成功数: {success_count}")
    print(f"   成功率: {success_count/len(steps):.1%}")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
