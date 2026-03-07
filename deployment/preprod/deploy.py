#!/usr/bin/env python3
"""
RQA2025 预投产环境部署脚本 (Python版本)

支持Windows和Linux环境
"""

import sys
import time
import subprocess
import requests
from pathlib import Path


class PreprodDeployer:
    """预投产环境部署器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.compose_file = self.project_root / "deployment" / "preprod" / "docker-compose.yml"
        self.env_file = self.project_root / "deployment" / "preprod" / ".env"
        self.project_name = "rqa2025-preprod"

        # 检测操作系统
        self.is_windows = sys.platform.startswith('win')

        # 设置命令
        if self.is_windows:
            self.compose_cmd = "docker compose"
        else:
            self.compose_cmd = "docker-compose"

    def log_info(self, message):
        print(f"[INFO] {message}")

    def log_success(self, message):
        print(f"[SUCCESS] {message}")

    def log_warning(self, message):
        print(f"[WARNING] {message}")

    def log_error(self, message):
        print(f"[ERROR] {message}")

    def check_dependencies(self):
        """检查系统依赖"""
        self.log_info("检查系统依赖...")

        # 检查Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"Docker版本: {result.stdout.strip()}")
            else:
                raise Exception("Docker不可用")
        except Exception as e:
            self.log_error(f"Docker检查失败: {e}")
            return False

        # 检查Docker Compose
        try:
            result = subprocess.run([self.compose_cmd, "version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"Docker Compose版本: {result.stdout.strip()}")
            else:
                raise Exception("Docker Compose不可用")
        except Exception as e:
            self.log_error(f"Docker Compose检查失败: {e}")
            return False

        return True

    def create_env_file(self):
        """创建环境配置文件"""
        if not self.env_file.exists():
            self.log_info("创建环境配置文件...")

            env_content = """# RQA2025 预投产环境配置

# PostgreSQL 配置
POSTGRES_PASSWORD=rqa2025_secure_pass_preprod

# InfluxDB 配置
INFLUXDB_PASSWORD=rqa2025_influx_pass_preprod
INFLUXDB_TOKEN=rqa2025_token_preprod_12345

# Redis 配置
REDIS_PASSWORD=rqa2025_redis_pass_preprod

# Grafana 配置
GRAFANA_USER=admin
GRAFANA_PASSWORD=rqa2025_grafana_pass_preprod

# Elasticsearch 配置
ELASTIC_PASSWORD=rqa2025_elastic_pass_preprod

# 应用配置
ENV=preprod
DEBUG=false
LOG_LEVEL=INFO
"""

            self.env_file.write_text(env_content)
            self.log_success(f"环境配置文件创建完成: {self.env_file}")
        else:
            self.log_info(f"环境配置文件已存在: {self.env_file}")

    def start_services(self):
        """启动服务"""
        self.log_info("启动预投产环境服务...")

        cmd = [
            self.compose_cmd,
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "up", "-d"
        ]

        try:
            result = subprocess.run(cmd, cwd=self.compose_file.parent)
            if result.returncode == 0:
                self.log_success("服务启动命令已执行")
                return True
            else:
                self.log_error("服务启动失败")
                return False
        except Exception as e:
            self.log_error(f"服务启动异常: {e}")
            return False

    def wait_for_services(self, timeout=600):
        """等待服务就绪"""
        self.log_info("等待服务启动完成...")

        services = ["postgres", "influxdb", "redis", "prometheus",
                    "grafana", "elasticsearch", "kibana", "health-monitor"]
        start_time = time.time()

        while time.time() - start_time < timeout:
            healthy_count = 0

            for service in services:
                try:
                    cmd = [
                        self.compose_cmd,
                        "-f", str(self.compose_file),
                        "-p", self.project_name,
                        "ps", service
                    ]
                    result = subprocess.run(cmd, capture_output=True,
                                            text=True, cwd=self.compose_file.parent)

                    if "healthy" in result.stdout or "running" in result.stdout:
                        healthy_count += 1

                except Exception:
                    continue

            if healthy_count == len(services):
                self.log_success(f"所有服务已就绪! ({healthy_count}/{len(services)} 服务健康)")
                return True

            self.log_info(f"等待服务就绪... ({healthy_count}/{len(services)} 服务就绪)")
            time.sleep(10)

        self.log_error("服务启动超时")
        return False

    def validate_deployment(self):
        """验证部署状态"""
        self.log_info("验证部署状态...")

        # 验证关键端点
        endpoints = [
            ("http://localhost:8000/health", "Health Monitor"),
            ("http://localhost:9090/-/healthy", "Prometheus"),
            ("http://localhost:3000/api/health", "Grafana"),
            ("http://localhost:9200/_cluster/health", "Elasticsearch"),
            ("http://localhost:5601/api/status", "Kibana")
        ]

        self.log_info("验证关键端点:")
        for url, name in endpoints:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    self.log_success(f"✅ {name}: {url}")
                else:
                    self.log_warning(f"⚠️  {name}: {url} (HTTP {response.status_code})")
            except Exception as e:
                self.log_warning(f"⚠️  {name}: {url} ({e})")

    def show_service_info(self):
        """显示服务信息"""
        self.log_info("预投产环境服务信息:")
        print()
        print("🌐 Web 界面:")
        print("  Health Monitor: http://localhost:8000")
        print("  Grafana:        http://localhost:3000 (admin / rqa2025_grafana_pass_preprod)")
        print("  Kibana:         http://localhost:5601")
        print("  Prometheus:     http://localhost:9090")
        print()
        print("💾 数据库服务:")
        print("  PostgreSQL:     localhost:5432 (rqa2025_user / rqa2025_secure_pass_preprod)")
        print("  InfluxDB:       localhost:8086 (rqa2025_admin / rqa2025_influx_pass_preprod)")
        print("  Redis:          localhost:6379 (password: rqa2025_redis_pass_preprod)")
        print("  Elasticsearch:  localhost:9200")
        print()
        print("📊 监控指标:")
        print("  Health Metrics: http://localhost:8000/metrics")
        print("  Prometheus API: http://localhost:9090/api/v1/query")

    def show_status(self):
        """显示服务状态"""
        self.log_info("预投产环境状态:")

        try:
            cmd = [
                self.compose_cmd,
                "-f", str(self.compose_file),
                "-p", self.project_name,
                "ps"
            ]
            result = subprocess.run(cmd, cwd=self.compose_file.parent)
        except Exception as e:
            self.log_error(f"获取状态失败: {e}")

    def stop_services(self):
        """停止服务"""
        self.log_info("停止预投产环境服务...")

        try:
            cmd = [
                self.compose_cmd,
                "-f", str(self.compose_file),
                "-p", self.project_name,
                "down"
            ]
            result = subprocess.run(cmd, cwd=self.compose_file.parent)
            if result.returncode == 0:
                self.log_success("服务已停止")
                return True
            else:
                self.log_error("停止服务失败")
                return False
        except Exception as e:
            self.log_error(f"停止服务异常: {e}")
            return False

    def cleanup(self, full=False):
        """清理环境"""
        self.log_warning("清理预投产环境...")

        try:
            # 停止并删除容器
            cmd = [
                self.compose_cmd,
                "-f", str(self.compose_file),
                "-p", self.project_name,
                "down", "-v", "--remove-orphans"
            ]
            subprocess.run(cmd, cwd=self.compose_file.parent)

            # 删除环境文件
            if self.env_file.exists():
                self.env_file.unlink()
                self.log_info("已删除环境配置文件")

            self.log_success("环境清理完成")

        except Exception as e:
            self.log_error(f"清理失败: {e}")

    def deploy(self):
        """完整部署流程"""
        self.log_info("🚀 开始RQA2025预投产环境部署")

        # 1. 检查依赖
        if not self.check_dependencies():
            return False

        # 2. 创建环境配置
        self.create_env_file()

        # 3. 启动服务
        if not self.start_services():
            return False

        # 4. 等待服务就绪
        if not self.wait_for_services():
            return False

        # 5. 验证部署
        self.validate_deployment()

        # 6. 显示服务信息
        self.show_service_info()

        self.log_success("🎉 预投产环境部署完成！")
        return True


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("RQA2025 预投产环境部署工具")
        print()
        print("使用方法:")
        print("  python deploy.py [command]")
        print()
        print("可用命令:")
        print("  deploy    完整部署预投产环境")
        print("  status    显示环境状态")
        print("  stop      停止预投产环境")
        print("  cleanup   清理环境")
        return

    command = sys.argv[1]
    deployer = PreprodDeployer()

    if command == "deploy":
        success = deployer.deploy()
        sys.exit(0 if success else 1)
    elif command == "status":
        deployer.show_status()
    elif command == "stop":
        deployer.stop_services()
    elif command == "cleanup":
        deployer.cleanup()
    else:
        print(f"未知命令: {command}")


if __name__ == "__main__":
    main()
