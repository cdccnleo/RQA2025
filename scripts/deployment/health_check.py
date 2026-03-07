#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
"""
RQA2025 生产环境健康检查脚本
"""

import os
import sys
import time
import json
import requests
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """服务健康状态"""
    name: str
    status: HealthStatus
    response_time: float
    last_check: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthChecker:
    """健康检查器"""

    def __init__(self, config_file: str = "config/production/monitoring.yaml"):
        self.config_file = Path(config_file)
        self.logger = self._setup_logging()
        self.monitoring_config = self._load_monitoring_config()
        self.services: List[ServiceHealth] = []

        # 健康检查配置
        self.health_check_config = {
            "timeout": 10,
            "retries": 3,
            "interval": 30,
            "endpoints": {
                "api_gateway": "http://localhost:8080/health",
                "data_service": "http://localhost:8081/health",
                "features_service": "http://localhost:8082/health",
                "model_service": "http://localhost:8083/health",
                "trading_service": "http://localhost:8084/health",
                "risk_service": "http://localhost:8085/health"
            }
        }

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_dir = Path("logs/health_check")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"health_check_{int(time.time())}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def _load_monitoring_config(self) -> Dict[str, Any]:
        """加载监控配置"""
        if not self.config_file.exists():
            self.logger.warning(f"监控配置文件不存在: {self.config_file}")
            return {}

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"加载监控配置失败: {str(e)}")
            return {}

    def check_all_services(self) -> List[ServiceHealth]:
        """检查所有服务健康状态"""
        self.logger.info("开始健康检查...")

        for service_name, endpoint in self.health_check_config["endpoints"].items():
            self.logger.info(f"检查服务: {service_name}")
            health = self._check_service_health(service_name, endpoint)
            self.services.append(health)

            # 检查间隔
            time.sleep(1)

        self.logger.info("健康检查完成")
        return self.services

    def _check_service_health(self, service_name: str, endpoint: str) -> ServiceHealth:
        """检查单个服务健康状态"""
        start_time = time.time()
        error_message = None
        details = {}

        for attempt in range(self.health_check_config["retries"]):
            try:
                self.logger.debug(f"检查 {service_name} (尝试 {attempt + 1})")

                response = requests.get(
                    endpoint,
                    timeout=self.health_check_config["timeout"],
                    headers={"User-Agent": "RQA2025-HealthChecker/1.0"}
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        details = health_data

                        # 检查健康状态
                        if health_data.get("status") == "healthy":
                            status = HealthStatus.HEALTHY
                        elif health_data.get("status") == "degraded":
                            status = HealthStatus.DEGRADED
                        else:
                            status = HealthStatus.UNHEALTHY

                        self.logger.info(f"{service_name} 健康检查通过: {status.value}")
                        break

                    except json.JSONDecodeError:
                        # 如果不是JSON响应，检查HTTP状态
                        if response.status_code == 200:
                            status = HealthStatus.HEALTHY
                        else:
                            status = HealthStatus.UNHEALTHY
                        break

                else:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                    status = HealthStatus.UNHEALTHY

                    if attempt < self.health_check_config["retries"] - 1:
                        time.sleep(2)  # 重试前等待

            except requests.exceptions.Timeout:
                error_message = f"请求超时 (>{self.health_check_config['timeout']}s)"
                status = HealthStatus.UNHEALTHY

            except requests.exceptions.ConnectionError:
                error_message = "连接失败"
                status = HealthStatus.UNHEALTHY

            except Exception as e:
                error_message = f"检查异常: {str(e)}"
                status = HealthStatus.UNKNOWN

        else:
            # 所有重试都失败了
            if not error_message:
                error_message = "所有重试都失败了"
            status = HealthStatus.UNHEALTHY
            response_time = time.time() - start_time

        return ServiceHealth(
            name=service_name,
            status=status,
            response_time=response_time,
            last_check=time.time(),
            error_message=error_message,
            details=details
        )

    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        self.logger.info("检查系统资源...")

        resources = {}

        try:
            # 检查CPU使用率
            cpu_result = subprocess.run(
                ["top", "-bn1", "|", "grep", "Cpu(s)", "|", "awk", "{print $2}"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if cpu_result.returncode == 0:
                cpu_usage = cpu_result.stdout.strip().replace('%', '')
                resources["cpu_usage"] = float(cpu_usage) if cpu_usage else 0

            # 检查内存使用率
            memory_result = subprocess.run(
                ["free", "-m"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if memory_result.returncode == 0:
                lines = memory_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    memory_line = lines[1]
                    parts = memory_line.split()
                    if len(parts) >= 3:
                        total = int(parts[1])
                        used = int(parts[2])
                        memory_usage = (used / total) * 100 if total > 0 else 0
                        resources["memory_usage"] = memory_usage
                        resources["memory_total_mb"] = total
                        resources["memory_used_mb"] = used

            # 检查磁盘使用率
            disk_result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if disk_result.returncode == 0:
                lines = disk_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    disk_line = lines[1]
                    parts = disk_line.split()
                    if len(parts) >= 5:
                        disk_usage = parts[4].replace('%', '')
                        resources["disk_usage"] = int(disk_usage) if disk_usage.isdigit() else 0

            # 检查网络连接
            network_result = subprocess.run(
                ["netstat", "-i"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if network_result.returncode == 0:
                lines = network_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # 简单统计网络接口数量
                    interface_count = len(
                        [line for line in lines if line.strip() and not line.startswith('Iface')])
                    resources["network_interfaces"] = interface_count

            self.logger.info("系统资源检查完成")

        except Exception as e:
            self.logger.warning(f"系统资源检查失败: {str(e)}")

        return resources

    def check_database_health(self) -> Dict[str, Any]:
        """检查数据库健康状态"""
        self.logger.info("检查数据库健康状态...")

        db_health = {}

        try:
            # 检查PostgreSQL连接
            pg_result = subprocess.run(
                ["pg_isready", "-h", "localhost", "-p", "5432"],
                capture_output=True,
                text=True,
                timeout=30
            )

            db_health["postgresql"] = {
                "status": "healthy" if pg_result.returncode == 0 else "unhealthy",
                "message": pg_result.stdout.strip() if pg_result.returncode == 0 else pg_result.stderr.strip()
            }

            # 检查Redis连接
            redis_result = subprocess.run(
                ["redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=30
            )

            db_health["redis"] = {
                "status": "healthy" if redis_result.returncode == 0 else "unhealthy",
                "message": redis_result.stdout.strip() if redis_result.returncode == 0 else redis_result.stderr.strip()
            }

            self.logger.info("数据库健康检查完成")

        except Exception as e:
            self.logger.warning(f"数据库健康检查失败: {str(e)}")
            db_health["error"] = str(e)

        return db_health

    def check_docker_services(self) -> Dict[str, Any]:
        """检查Docker服务状态"""
        self.logger.info("检查Docker服务状态...")

        docker_health = {}

        try:
            # 检查Docker服务状态
            docker_result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if docker_result.returncode == 0:
                lines = docker_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    # 跳过标题行
                    for line in lines[1:]:
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                service_name = parts[0]
                                status = parts[1]
                                ports = parts[2] if len(parts) > 2 else ""

                                docker_health[service_name] = {
                                    "status": "running" if "Up" in status else "stopped",
                                    "details": status,
                                    "ports": ports
                                }

            # 检查Docker Compose状态
            compose_result = subprocess.run(
                ["docker-compose", "-f", "deploy/docker-compose.prod.yml", "ps"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if compose_result.returncode == 0:
                docker_health["compose_status"] = "available"
                docker_health["compose_output"] = compose_result.stdout.strip()
            else:
                docker_health["compose_status"] = "unavailable"
                docker_health["compose_error"] = compose_result.stderr.strip()

            self.logger.info("Docker服务检查完成")

        except Exception as e:
            self.logger.warning(f"Docker服务检查失败: {str(e)}")
            docker_health["error"] = str(e)

        return docker_health

    def generate_health_report(self) -> str:
        """生成健康检查报告"""
        report = ["# 生产环境健康检查报告\n"]

        # 总体健康状态
        total_services = len(self.services)
        healthy_services = len([s for s in self.services if s.status == HealthStatus.HEALTHY])
        unhealthy_services = len([s for s in self.services if s.status == HealthStatus.UNHEALTHY])
        degraded_services = len([s for s in self.services if s.status == HealthStatus.DEGRADED])

        overall_status = "healthy" if unhealthy_services == 0 else "unhealthy"
        status_icon = "✅" if overall_status == "healthy" else "❌"

        report.append(f"## 总体健康状态 {status_icon}\n")
        report.append(f"- **状态**: {overall_status}")
        report.append(f"- **总服务数**: {total_services}")
        report.append(f"- **健康服务**: {healthy_services}")
        report.append(f"- **不健康服务**: {unhealthy_services}")
        report.append(f"- **降级服务**: {degraded_services}")
        report.append(f"- **健康率**: {healthy_services/total_services*100:.1f}%\n")

        # 服务健康详情
        report.append("## 服务健康详情\n")
        for service in self.services:
            status_icon = "✅" if service.status == HealthStatus.HEALTHY else "❌"
            report.append(f"### {service.name} {status_icon}")
            report.append(f"- **状态**: {service.status.value}")
            report.append(f"- **响应时间**: {service.response_time:.3f}s")
            report.append(
                f"- **最后检查**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(service.last_check))}")

            if service.error_message:
                report.append(f"- **错误信息**: {service.error_message}")

            if service.details:
                report.append(
                    f"- **详细信息**: ```json\n{json.dumps(service.details, indent=2, ensure_ascii=False)}\n```")

            report.append("")

        # 系统资源状态
        system_resources = self.check_system_resources()
        if system_resources:
            report.append("## 系统资源状态\n")
            for resource, value in system_resources.items():
                if isinstance(value, float):
                    report.append(f"- **{resource}**: {value:.2f}")
                else:
                    report.append(f"- **{resource}**: {value}")
            report.append("")

        # 数据库健康状态
        db_health = self.check_database_health()
        if db_health:
            report.append("## 数据库健康状态\n")
            for db_name, db_status in db_health.items():
                if isinstance(db_status, dict):
                    status_icon = "✅" if db_status.get("status") == "healthy" else "❌"
                    report.append(f"### {db_name} {status_icon}")
                    report.append(f"- **状态**: {db_status.get('status', 'unknown')}")
                    if db_status.get("message"):
                        report.append(f"- **消息**: {db_status['message']}")
                    report.append("")

        # Docker服务状态
        docker_health = self.check_docker_services()
        if docker_health:
            report.append("## Docker服务状态\n")
            for service_name, service_status in docker_health.items():
                if isinstance(service_status, dict):
                    status_icon = "✅" if service_status.get("status") == "running" else "❌"
                    report.append(f"### {service_name} {status_icon}")
                    report.append(f"- **状态**: {service_status.get('status', 'unknown')}")
                    if service_status.get("details"):
                        report.append(f"- **详情**: {service_status['details']}")
                    if service_status.get("ports"):
                        report.append(f"- **端口**: {service_status['ports']}")
                    report.append("")

        # 建议和行动项
        report.append("## 建议和行动项\n")
        if unhealthy_services > 0:
            report.append("⚠️ **需要立即关注的问题:**")
            for service in self.services:
                if service.status == HealthStatus.UNHEALTHY:
                    report.append(f"- 检查服务 {service.name} 的状态和日志")
            report.append("")

        if degraded_services > 0:
            report.append("⚠️ **需要监控的问题:**")
            for service in self.services:
                if service.status == HealthStatus.DEGRADED:
                    report.append(f"- 监控服务 {service.name} 的性能指标")
            report.append("")

        if overall_status == "healthy":
            report.append("🎉 **系统运行正常** 建议继续监控关键指标。")

        return "\n".join(report)

    def save_health_report(self, report: str):
        """保存健康检查报告"""
        report_dir = Path("reports/health_check")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        report_file = report_dir / f"health_check_report_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"健康检查报告已保存: {report_file}")

    def continuous_monitoring(self, duration: int = 3600, interval: int = 30):
        """持续监控"""
        self.logger.info(f"开始持续监控，持续时间: {duration}s，检查间隔: {interval}s")

        start_time = time.time()
        check_count = 0

        try:
            while time.time() - start_time < duration:
                check_count += 1
                self.logger.info(f"执行第 {check_count} 次健康检查...")

                # 执行健康检查
                self.check_all_services()

                # 生成报告
                report = self.generate_health_report()

                # 保存报告
                timestamp = int(time.time())
                report_file = f"reports/health_check/continuous_check_{timestamp}.md"
                os.makedirs(os.path.dirname(report_file), exist_ok=True)

                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)

                # 等待下次检查
                if time.time() - start_time < duration:
                    time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("持续监控被用户中断")

        self.logger.info(f"持续监控结束，共执行 {check_count} 次检查")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025 生产环境健康检查工具")
    parser.add_argument("--continuous", "-c", action="store_true", help="启用持续监控")
    parser.add_argument("--duration", "-d", type=int, default=3600, help="持续监控持续时间(秒)")
    parser.add_argument("--interval", "-i", type=int, default=30, help="检查间隔(秒)")

    args = parser.parse_args()

    checker = HealthChecker()

    if args.continuous:
        # 持续监控模式
        checker.continuous_monitoring(args.duration, args.interval)
    else:
        # 单次检查模式
        checker.check_all_services()
        report = checker.generate_health_report()
        print(report)
        checker.save_health_report(report)


if __name__ == "__main__":
    main()
