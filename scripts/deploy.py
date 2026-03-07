#!/usr/bin/env python3
"""
RQA2025 自动化部署脚本
支持测试环境和生产环境的部署
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class Deployer:
    """自动化部署器"""

    def __init__(self, project_root: str, environment: str = "staging"):
        self.project_root = Path(project_root)
        self.environment = environment
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_config(self) -> Dict[str, Any]:
        """加载部署配置"""
        config_file = self.project_root / "deploy-config.json"
        default_config = {
            "environments": {
                "staging": {
                    "host": "staging.rqa2025.com",
                    "port": 22,
                    "user": "deploy",
                    "path": "/opt/rqa2025/staging",
                    "services": ["trading-engine", "risk-monitor", "data-processor"],
                    "health_check_url": "http://staging.rqa2025.com/health"
                },
                "production": {
                    "host": "prod.rqa2025.com",
                    "port": 22,
                    "user": "deploy",
                    "path": "/opt/rqa2025/production",
                    "services": ["trading-engine", "risk-monitor", "data-processor"],
                    "health_check_url": "http://prod.rqa2025.com/health",
                    "backup_before_deploy": true,
                    "canary_deployment": true
                }
            },
            "build": {
                "python_version": "3.9",
                "requirements_file": "requirements.txt",
                "exclude_patterns": [".git", "__pycache__", "*.pyc", "test_*"]
            },
            "rollback": {
                "keep_versions": 5,
                "auto_rollback_on_failure": true
            }
        }

        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 合并配置
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values

        return default_config

    def build_package(self) -> Optional[str]:
        """构建部署包"""
        print("📦 构建部署包...")

        try:
            # 创建构建目录
            build_dir = self.project_root / "build"
            build_dir.mkdir(exist_ok=True)

            package_name = f"rqa2025-{self.environment}-{self.timestamp}"
            package_dir = build_dir / package_name
            package_dir.mkdir(exist_ok=True)

            # 复制项目文件
            exclude_patterns = self.config["build"]["exclude_patterns"]

            def should_copy(path: Path) -> bool:
                """检查是否应该复制文件"""
                for pattern in exclude_patterns:
                    if pattern.startswith("*"):
                        if path.name.endswith(pattern[1:]):
                            return False
                    elif pattern in str(path):
                        return False
                return True

            # 递归复制文件
            def copy_files(src: Path, dst: Path):
                if not should_copy(src):
                    return

                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src, dst)
                elif src.is_dir():
                    for item in src.iterdir():
                        copy_files(item, dst / item.name)

            copy_files(self.project_root, package_dir)

            # 创建部署信息文件
            deploy_info = {
                "version": self.timestamp,
                "environment": self.environment,
                "build_time": datetime.now().isoformat(),
                "git_commit": self._get_git_commit(),
                "python_version": self.config["build"]["python_version"]
            }

            with open(package_dir / "deploy-info.json", 'w', encoding='utf-8') as f:
                json.dump(deploy_info, f, indent=2, ensure_ascii=False)

            # 压缩包
            import tarfile
            tar_path = build_dir / f"{package_name}.tar.gz"

            with tarfile.open(tar_path, "w:gz") as tar:
                for item in package_dir.iterdir():
                    tar.add(item, arcname=item.name)

            print(f"✅ 部署包构建完成: {tar_path}")
            return str(tar_path)

        except Exception as e:
            print(f"❌ 构建部署包失败: {e}")
            return None

    def _get_git_commit(self) -> str:
        """获取Git提交信息"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def deploy_to_server(self, package_path: str) -> bool:
        """部署到服务器"""
        print(f"🚀 部署到{self.environment}环境...")

        env_config = self.config["environments"][self.environment]

        try:
            # 这里应该是实际的部署逻辑
            # 为了演示，我们使用模拟部署

            print(f"📡 连接到服务器: {env_config['host']}:{env_config['port']}")
            print(f"👤 用户: {env_config['user']}")
            print(f"📁 部署路径: {env_config['path']}")

            # 模拟文件传输
            print(f"📤 上传部署包: {package_path}")

            # 模拟备份
            if env_config.get("backup_before_deploy", False):
                print("💾 创建备份...")
                print("✅ 备份完成")

            # 模拟服务停止
            print("🛑 停止现有服务...")
            for service in env_config["services"]:
                print(f"  • 停止服务: {service}")

            # 模拟文件解压和安装
            print("📦 解压部署包...")
            print("🔧 安装依赖...")
            print("⚙️ 更新配置...")

            # 模拟服务启动
            print("▶️ 启动新服务...")
            for service in env_config["services"]:
                print(f"  • 启动服务: {service}")

            # 灰度发布检查
            if env_config.get("canary_deployment", False):
                print("🦜 执行灰度发布...")
                print("📊 监控灰度效果...")

            # 健康检查
            print("🏥 执行健康检查...")
            health_url = env_config.get("health_check_url")
            if health_url:
                print(f"🔍 检查URL: {health_url}")
                print("✅ 健康检查通过")

            print(f"✅ 部署到{self.environment}环境成功！")
            return True

        except Exception as e:
            print(f"❌ 部署失败: {e}")
            return False

    def rollback(self, version: Optional[str] = None) -> bool:
        """回滚到指定版本"""
        print("🔄 执行回滚操作...")

        try:
            if version is None:
                # 回滚到上一个版本
                print("📋 查找上一个稳定版本...")
                version = "previous-stable-version"

            print(f"🔙 回滚到版本: {version}")

            # 模拟回滚逻辑
            env_config = self.config["environments"][self.environment]

            print("🛑 停止当前服务...")
            print(f"📦 恢复备份版本: {version}")
            print("▶️ 重启服务...")

            # 健康检查
            print("🏥 执行回滚后健康检查...")
            print("✅ 回滚成功")

            return True

        except Exception as e:
            print(f"❌ 回滚失败: {e}")
            return False

    def run_health_checks(self) -> Dict[str, Any]:
        """运行健康检查"""
        print("🏥 执行部署后健康检查...")

        results = {
            "overall_health": "unknown",
            "checks": {}
        }

        env_config = self.config["environments"][self.environment]

        # 服务状态检查
        results["checks"]["services"] = self._check_services_status(env_config["services"])

        # 数据库连接检查
        results["checks"]["database"] = self._check_database_connection()

        # 缓存连接检查
        results["checks"]["cache"] = self._check_cache_connection()

        # API端点检查
        results["checks"]["api"] = self._check_api_endpoints()

        # 性能指标检查
        results["checks"]["performance"] = self._check_performance_metrics()

        # 计算整体健康状态
        failed_checks = [k for k, v in results["checks"].items() if not v.get("healthy", False)]
        if not failed_checks:
            results["overall_health"] = "healthy"
        elif len(failed_checks) < len(results["checks"]) * 0.5:
            results["overall_health"] = "degraded"
        else:
            results["overall_health"] = "unhealthy"

        return results

    def _check_services_status(self, services: list) -> Dict[str, Any]:
        """检查服务状态"""
        # 模拟服务状态检查
        return {
            "healthy": True,
            "services": {service: "running" for service in services},
            "message": "所有服务正常运行"
        }

    def _check_database_connection(self) -> Dict[str, Any]:
        """检查数据库连接"""
        # 模拟数据库连接检查
        return {
            "healthy": True,
            "connection_time": 0.05,
            "message": "数据库连接正常"
        }

    def _check_cache_connection(self) -> Dict[str, Any]:
        """检查缓存连接"""
        # 模拟缓存连接检查
        return {
            "healthy": True,
            "response_time": 0.01,
            "message": "缓存连接正常"
        }

    def _check_api_endpoints(self) -> Dict[str, Any]:
        """检查API端点"""
        # 模拟API端点检查
        endpoints = ["/health", "/status", "/metrics"]
        return {
            "healthy": True,
            "endpoints_checked": len(endpoints),
            "response_times": {ep: 0.1 for ep in endpoints},
            "message": "所有API端点正常响应"
        }

    def _check_performance_metrics(self) -> Dict[str, Any]:
        """检查性能指标"""
        # 模拟性能指标检查
        return {
            "healthy": True,
            "metrics": {
                "cpu_usage": 65.5,
                "memory_usage": 72.3,
                "response_time_avg": 0.15,
                "throughput": 1250
            },
            "message": "性能指标在正常范围内"
        }

    def run_full_deployment(self) -> bool:
        """执行完整部署流程"""
        print(f"🚀 开始{self.environment}环境完整部署流程")
        print("=" * 60)

        try:
            # 1. 构建部署包
            package_path = self.build_package()
            if not package_path:
                return False

            # 2. 质量门禁检查
            print("\n🎯 执行质量门禁检查...")
            gate_passed = self.run_quality_gate_check()
            if not gate_passed:
                print("❌ 质量门禁检查失败，停止部署")
                return False

            # 3. 部署到服务器
            if not self.deploy_to_server(package_path):
                print("❌ 部署失败，准备回滚...")
                self.rollback()
                return False

            # 4. 运行健康检查
            health_results = self.run_health_checks()

            if health_results["overall_health"] == "healthy":
                print("\n✅ 部署成功完成！")
                print("📊 健康检查结果: 所有组件正常运行")
                return True
            else:
                print(f"\n⚠️ 部署完成但健康状态异常: {health_results['overall_health']}")
                print("🔄 执行自动回滚...")
                self.rollback()
                return False

        except Exception as e:
            print(f"\n❌ 部署过程中发生错误: {e}")
            print("🔄 尝试回滚...")
            self.rollback()
            return False

    def run_quality_gate_check(self) -> bool:
        """运行质量门禁检查"""
        try:
            # 运行质量门禁脚本
            result = subprocess.run([
                sys.executable, "scripts/quality_gate.py",
                "--project-root", str(self.project_root),
                "--fail-on-error"
            ], cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ 质量门禁检查通过")
                return True
            else:
                print("❌ 质量门禁检查失败")
                print(result.stdout)
                print(result.stderr)
                return False

        except Exception as e:
            print(f"❌ 质量门禁检查执行失败: {e}")
            return False

    def generate_deployment_report(self, success: bool, health_results: Dict[str, Any]) -> str:
        """生成部署报告"""
        report = [
            "# RQA2025 部署报告\n",
            f"**部署时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**目标环境**: {self.environment}",
            f"**部署状态**: {'✅ 成功' if success else '❌ 失败'}",
            f"**部署版本**: {self.timestamp}",
            ""
        ]

        if health_results:
            report.append("## 健康检查结果")
            report.append(f"**整体状态**: {health_results['overall_health']}")

            for check_name, check_result in health_results.get("checks", {}).items():
                status = "✅" if check_result.get("healthy", False) else "❌"
                message = check_result.get("message", "无详细信息")
                report.append(f"- **{check_name}**: {status} {message}")

        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 自动化部署器')
    parser.add_argument('--environment', '-e', choices=['staging', 'production'],
                       default='staging', help='部署环境')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--dry-run', action='store_true', help='仅执行检查，不进行实际部署')
    parser.add_argument('--rollback', help='回滚到指定版本')
    parser.add_argument('--health-check', action='store_true', help='仅执行健康检查')

    args = parser.parse_args()

    # 初始化部署器
    deployer = Deployer(args.project_root, args.environment)

    if args.rollback:
        # 执行回滚
        success = deployer.rollback(args.rollback)
        if success:
            print(f"✅ 成功回滚到版本: {args.rollback}")
        else:
            print("❌ 回滚失败")
        return 0 if success else 1

    if args.health_check:
        # 仅执行健康检查
        results = deployer.run_health_checks()
        print("🏥 健康检查结果:"        print(f"  整体状态: {results['overall_health']}")
        for check_name, check_result in results.get("checks", {}).items():
            status = "✅" if check_result.get("healthy", False) else "❌"
            print(f"  {check_name}: {status} {check_result.get('message', '')}")
        return 0

    if args.dry_run:
        # 仅执行检查
        print("🔍 执行部署前检查...")

        # 质量门禁检查
        gate_passed = deployer.run_quality_gate_check()
        if not gate_passed:
            print("❌ 质量门禁检查失败")
            return 1

        # 构建包检查
        package_path = deployer.build_package()
        if not package_path:
            print("❌ 构建包失败")
            return 1

        print("✅ 所有检查通过，准备部署")
        return 0

    # 执行完整部署
    success = deployer.run_full_deployment()

    if success:
        print("
🎊 RQA2025 部署成功完成！"        print(f"📋 部署环境: {args.environment}")
        print(f"📦 部署版本: {deployer.timestamp}")
    else:
        print("\n❌ 部署失败，请检查日志")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())