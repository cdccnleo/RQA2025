#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025生产环境部署验证脚本
在部署前验证所有配置、依赖和服务可用性

Author: RQA2025 Development Team
Date: 2025-12-02
"""

import os
import sys
import json
import socket
import logging
import argparse
import subprocess
import requests
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_deployment_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeploymentValidator:
    """生产环境部署验证器"""

    def __init__(self, config_dir: str = "config/production"):
        self.config_dir = Path(config_dir)
        self.validation_results = {
            "config_validation": [],
            "service_connectivity": [],
            "dependency_checks": [],
            "security_validation": [],
            "performance_baselines": []
        }

    def validate_config_files(self) -> bool:
        """验证配置文件完整性"""
        logger.info("🔍 验证配置文件完整性...")

        required_configs = [
            "database.json", "redis.json", "api.json",
            "monitoring.json", "logging.json", "security.json"
        ]

        all_valid = True
        for config_file in required_configs:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                self.validation_results["config_validation"].append({
                    "check": f"配置文件存在性: {config_file}",
                    "status": "FAILED",
                    "message": f"配置文件不存在: {config_path}"
                })
                all_valid = False
                continue

            # 验证JSON格式
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                self.validation_results["config_validation"].append({
                    "check": f"配置文件格式: {config_file}",
                    "status": "PASSED",
                    "message": "JSON格式正确"
                })
            except json.JSONDecodeError as e:
                self.validation_results["config_validation"].append({
                    "check": f"配置文件格式: {config_file}",
                    "status": "FAILED",
                    "message": f"JSON格式错误: {e}"
                })
                all_valid = False

        # 验证Nginx配置
        nginx_config = self.config_dir / "nginx.conf"
        if nginx_config.exists():
            self.validation_results["config_validation"].append({
                "check": "Nginx配置文件",
                "status": "PASSED",
                "message": "Nginx配置文件存在"
            })
        else:
            self.validation_results["config_validation"].append({
                "check": "Nginx配置文件",
                "status": "FAILED",
                "message": "Nginx配置文件不存在"
            })
            all_valid = False

        return all_valid

    def validate_service_connectivity(self) -> bool:
        """验证服务连接性"""
        logger.info("🔗 验证服务连接性...")

        # 加载配置
        try:
            with open(self.config_dir / "database.json", 'r') as f:
                db_config = json.load(f)["database"]
            with open(self.config_dir / "redis.json", 'r') as f:
                redis_config = json.load(f)["redis"]
            with open(self.config_dir / "api.json", 'r') as f:
                api_config = json.load(f)["api"]
        except Exception as e:
            self.validation_results["service_connectivity"].append({
                "check": "配置加载",
                "status": "FAILED",
                "message": f"无法加载配置: {e}"
            })
            return False

        all_connected = True

        # 验证数据库连接
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((db_config["host"], db_config["port"]))
            sock.close()

            if result == 0:
                self.validation_results["service_connectivity"].append({
                    "check": "数据库连接",
                    "status": "PASSED",
                    "message": f"数据库连接成功: {db_config['host']}:{db_config['port']}"
                })
            else:
                self.validation_results["service_connectivity"].append({
                    "check": "数据库连接",
                    "status": "FAILED",
                    "message": f"数据库连接失败: {db_config['host']}:{db_config['port']}"
                })
                all_connected = False
        except Exception as e:
            self.validation_results["service_connectivity"].append({
                "check": "数据库连接",
                "status": "FAILED",
                "message": f"数据库连接异常: {e}"
            })
            all_connected = False

        # 验证Redis连接
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((redis_config["host"], redis_config["port"]))
            sock.close()

            if result == 0:
                self.validation_results["service_connectivity"].append({
                    "check": "Redis连接",
                    "status": "PASSED",
                    "message": f"Redis连接成功: {redis_config['host']}:{redis_config['port']}"
                })
            else:
                self.validation_results["service_connectivity"].append({
                    "check": "Redis连接",
                    "status": "FAILED",
                    "message": f"Redis连接失败: {redis_config['host']}:{redis_config['port']}"
                })
                all_connected = False
        except Exception as e:
            self.validation_results["service_connectivity"].append({
                "check": "Redis连接",
                "status": "FAILED",
                "message": f"Redis连接异常: {e}"
            })
            all_connected = False

        return all_connected

    def validate_dependencies(self) -> bool:
        """验证系统依赖"""
        logger.info("📦 验证系统依赖...")

        dependencies = [
            ("python", "--version", "Python"),
            ("docker", "--version", "Docker"),
            ("docker-compose", "--version", "Docker Compose"),
            ("kubectl", "version --client", "Kubernetes CLI"),
            ("nginx", "-v", "Nginx")
        ]

        all_available = True
        for cmd, args, name in dependencies:
            try:
                result = subprocess.run([cmd] + args.split(),
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0] if result.stdout.strip() else "可用"
                    self.validation_results["dependency_checks"].append({
                        "check": f"{name}可用性",
                        "status": "PASSED",
                        "message": f"{name}可用: {version}"
                    })
                else:
                    self.validation_results["dependency_checks"].append({
                        "check": f"{name}可用性",
                        "status": "FAILED",
                        "message": f"{name}不可用或版本不兼容"
                    })
                    all_available = False
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                self.validation_results["dependency_checks"].append({
                    "check": f"{name}可用性",
                    "status": "FAILED",
                    "message": f"{name}检查失败: {e}"
                })
                all_available = False

        return all_available

    def validate_security_config(self) -> bool:
        """验证安全配置"""
        logger.info("🔒 验证安全配置...")

        try:
            with open(self.config_dir / "security.json", 'r') as f:
                security_config = json.load(f)["security"]
        except Exception as e:
            self.validation_results["security_validation"].append({
                "check": "安全配置加载",
                "status": "FAILED",
                "message": f"无法加载安全配置: {e}"
            })
            return False

        security_valid = True

        # 验证加密配置
        encryption = security_config.get("encryption", {})
        if encryption.get("algorithm") in ["AES-256-GCM", "AES-256-CBC"]:
            self.validation_results["security_validation"].append({
                "check": "加密算法配置",
                "status": "PASSED",
                "message": f"使用安全的加密算法: {encryption['algorithm']}"
            })
        else:
            self.validation_results["security_validation"].append({
                "check": "加密算法配置",
                "status": "FAILED",
                "message": "加密算法配置不安全"
            })
            security_valid = False

        # 验证SSL配置
        ssl_config = security_config.get("ssl", {})
        if ssl_config.get("min_version") in ["TLSv1.2", "TLSv1.3"]:
            self.validation_results["security_validation"].append({
                "check": "SSL/TLS配置",
                "status": "PASSED",
                "message": f"使用安全的TLS版本: {ssl_config['min_version']}"
            })
        else:
            self.validation_results["security_validation"].append({
                "check": "SSL/TLS配置",
                "status": "FAILED",
                "message": "SSL/TLS版本配置不安全"
            })
            security_valid = False

        return security_valid

    def validate_performance_baselines(self) -> bool:
        """验证性能基准"""
        logger.info("⚡ 验证性能基准...")

        try:
            with open(self.config_dir / "api.json", 'r') as f:
                api_config = json.load(f)["api"]
        except Exception as e:
            self.validation_results["performance_baselines"].append({
                "check": "API配置加载",
                "status": "FAILED",
                "message": f"无法加载API配置: {e}"
            })
            return False

        performance_valid = True

        # 验证资源配置合理性
        workers = api_config.get("workers", 1)
        if workers >= 2:
            self.validation_results["performance_baselines"].append({
                "check": "API工作进程配置",
                "status": "PASSED",
                "message": f"配置了 {workers} 个工作进程，适合生产环境"
            })
        else:
            self.validation_results["performance_baselines"].append({
                "check": "API工作进程配置",
                "status": "WARNING",
                "message": f"工作进程数较少: {workers}，可能影响性能"
            })

        # 验证端口配置
        port = api_config.get("port", 8080)
        if 1024 <= port <= 65535:
            self.validation_results["performance_baselines"].append({
                "check": "端口配置",
                "status": "PASSED",
                "message": f"使用有效端口: {port}"
            })
        else:
            self.validation_results["performance_baselines"].append({
                "check": "端口配置",
                "status": "FAILED",
                "message": f"端口配置无效: {port}"
            })
            performance_valid = False

        return performance_valid

    def run_health_checks(self) -> bool:
        """运行健康检查"""
        logger.info("🏥 运行健康检查...")

        health_checks = []

        # 检查配置文件语法
        try:
            with open(self.config_dir / "nginx.conf", 'r') as f:
                nginx_config = f.read()
            # 基本语法检查：确保有server块
            if "server {" in nginx_config and "listen" in nginx_config:
                health_checks.append({
                    "check": "Nginx配置语法",
                    "status": "PASSED",
                    "message": "Nginx配置语法正确"
                })
            else:
                health_checks.append({
                    "check": "Nginx配置语法",
                    "status": "FAILED",
                    "message": "Nginx配置缺少必要的server块或listen指令"
                })
        except Exception as e:
            health_checks.append({
                "check": "Nginx配置语法",
                "status": "FAILED",
                "message": f"Nginx配置检查失败: {e}"
            })

        # 检查Docker Compose配置
        try:
            import yaml
            with open(self.config_dir / "docker-compose.yml", 'r') as f:
                docker_config = yaml.safe_load(f)
            if "services" in docker_config and "version" in docker_config:
                health_checks.append({
                    "check": "Docker Compose配置",
                    "status": "PASSED",
                    "message": "Docker Compose配置有效"
                })
            else:
                health_checks.append({
                    "check": "Docker Compose配置",
                    "status": "FAILED",
                    "message": "Docker Compose配置缺少必要的服务或版本信息"
                })
        except Exception as e:
            health_checks.append({
                "check": "Docker Compose配置",
                "status": "FAILED",
                "message": f"Docker Compose配置检查失败: {e}"
            })

        # 检查Kubernetes配置
        try:
            with open(self.config_dir / "kubernetes.yml", 'r') as f:
                k8s_config = yaml.safe_load(f)
            if k8s_config.get("kind") == "Deployment" and "spec" in k8s_config:
                health_checks.append({
                    "check": "Kubernetes配置",
                    "status": "PASSED",
                    "message": "Kubernetes配置有效"
                })
            else:
                health_checks.append({
                    "check": "Kubernetes配置",
                    "status": "FAILED",
                    "message": "Kubernetes配置无效"
                })
        except Exception as e:
            health_checks.append({
                "check": "Kubernetes配置",
                "status": "FAILED",
                "message": f"Kubernetes配置检查失败: {e}"
            })

        # 将健康检查结果添加到验证结果中
        for check in health_checks:
            if check["check"] not in [r["check"] for r in self.validation_results["dependency_checks"]]:
                self.validation_results["dependency_checks"].append(check)

        return all(check["status"] == "PASSED" for check in health_checks)

    def run_all_validations(self) -> Dict[str, Any]:
        """运行所有验证"""
        logger.info("🚀 开始生产环境部署验证...")

        validation_status = {
            "config_validation": self.validate_config_files(),
            "service_connectivity": self.validate_service_connectivity(),
            "dependency_checks": self.validate_dependencies(),
            "security_validation": self.validate_security_config(),
            "performance_baselines": self.validate_performance_baselines(),
            "health_checks": self.run_health_checks()
        }

        validation_status["overall_success"] = all(validation_status.values())

        # 生成验证报告
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_status": validation_status,
            "validation_results": self.validation_results,
            "summary": {
                "total_checks": sum(len(results) for results in self.validation_results.values()),
                "passed_checks": sum(1 for results in self.validation_results.values()
                                   for result in results if result["status"] == "PASSED"),
                "failed_checks": sum(1 for results in self.validation_results.values()
                                   for result in results if result["status"] == "FAILED"),
                "warning_checks": sum(1 for results in self.validation_results.values()
                                    for result in results if result["status"] == "WARNING")
            }
        }

        return report

    def save_validation_report(self, report: Dict[str, Any], output_file: str = None):
        """保存验证报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"production_deployment_validation_report_{timestamp}.json"

        report_path = Path("reports") / output_file
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"验证报告已保存到: {report_path}")
        return report_path

    def print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        print("\n" + "="*80)
        print("🎯 RQA2025生产环境部署验证报告")
        print("="*80)

        status = report["validation_status"]
        summary = report["summary"]

        print(f"\n📊 验证概览:")
        print(f"   总检查项: {summary['total_checks']}")
        print(f"   通过检查: {summary['passed_checks']}")
        print(f"   失败检查: {summary['failed_checks']}")
        print(f"   警告检查: {summary['warning_checks']}")

        print(f"\n🔍 验证状态:")
        for check_type, passed in status.items():
            if check_type != "overall_success":
                status_icon = "✅" if passed else "❌"
                print(f"   {status_icon} {check_type.replace('_', ' ').title()}: {'通过' if passed else '失败'}")

        overall_status = "✅ 验证通过，可以开始部署" if status["overall_success"] else "❌ 验证失败，需要修复问题后重新验证"

        print(f"\n🏆 总体结果: {overall_status}")

        if not status["overall_success"]:
            print(f"\n⚠️  需要修复的问题:")
            for check_type, results in report["validation_results"].items():
                failed_results = [r for r in results if r["status"] == "FAILED"]
                if failed_results:
                    print(f"\n   {check_type.replace('_', ' ').title()}:")
                    for result in failed_results:
                        print(f"     ❌ {result['check']}: {result['message']}")

        print(f"\n📅 验证时间: {report['validation_timestamp']}")
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025生产环境部署验证器")
    parser.add_argument("--config-dir", default="config/production",
                       help="配置文件目录")
    parser.add_argument("--output-file", default=None,
                       help="验证报告输出文件")
    parser.add_argument("--skip-connectivity", action="store_true",
                       help="跳过服务连接性验证")

    args = parser.parse_args()

    validator = ProductionDeploymentValidator(args.config_dir)

    # 运行验证
    report = validator.run_all_validations()

    # 保存报告
    report_path = validator.save_validation_report(report, args.output_file)

    # 打印摘要
    validator.print_validation_summary(report)

    # 返回退出码
    success = report["validation_status"]["overall_success"]
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


