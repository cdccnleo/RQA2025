#!/usr/bin/env python3
"""
生产环境部署执行脚本

用于自动化部署RQA2025到生产环境
包括：服务启动、配置验证、健康检查等
"""

import os
import sys
import time
import json
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """生产环境部署器"""

    def __init__(self, config_path: str):
        """
        初始化部署器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.deployment_status = {}
        self.start_time = time.time()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)

    def check_prerequisites(self) -> bool:
        """检查部署前置条件"""
        logger.info("检查部署前置条件...")

        prerequisites = [
            ("Docker", self._check_docker),
            ("Docker Compose", self._check_docker_compose),
            ("磁盘空间", self._check_disk_space),
            ("网络连通性", self._check_network_connectivity),
            ("配置文件", self._check_config_files)
        ]

        all_passed = True
        for name, check_func in prerequisites:
            try:
                if check_func():
                    logger.info(f"✅ {name} 检查通过")
                else:
                    logger.error(f"❌ {name} 检查失败")
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {name} 检查异常: {e}")
                all_passed = False

        if all_passed:
            logger.info("所有前置条件检查通过")
        else:
            logger.error("部分前置条件检查失败，无法继续部署")

        return all_passed

    def _check_docker(self) -> bool:
        """检查Docker是否可用"""
        try:
            result = subprocess.run(['docker', '--version'],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    def _check_docker_compose(self) -> bool:
        """检查Docker Compose是否可用"""
        try:
            result = subprocess.run(['docker-compose', '--version'],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False

    def _check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            return free_gb > 50  # 至少需要50GB可用空间
        except ImportError:
            logger.warning("psutil未安装，跳过磁盘空间检查")
            return True

    def _check_network_connectivity(self) -> bool:
        """检查网络连通性"""
        try:
            # 检查基本网络连通性
            response = requests.get('http://www.google.com', timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_config_files(self) -> bool:
        """检查配置文件"""
        required_files = [
            'docker-compose.yml',
            'Dockerfile',
            'config/deployment_config.json'
        ]

        for file_path in required_files:
            if not Path(f"deploy/{file_path}").exists():
                logger.error(f"缺少必需文件: {file_path}")
                return False

        return True

    def deploy_services(self) -> bool:
        """部署服务"""
        logger.info("开始部署服务...")

        try:
            # 切换到部署目录
            os.chdir('deploy')

            # 停止现有服务
            logger.info("停止现有服务...")
            subprocess.run(['docker-compose', 'down'], check=True)

            # 构建镜像
            logger.info("构建Docker镜像...")
            subprocess.run(['docker-compose', 'build'], check=True)

            # 启动服务
            logger.info("启动服务...")
            subprocess.run(['docker-compose', 'up', '-d'], check=True)

            # 等待服务启动
            logger.info("等待服务启动...")
            time.sleep(30)

            # 检查服务状态
            logger.info("检查服务状态...")
            result = subprocess.run(['docker-compose', 'ps'],
                                    capture_output=True, text=True, check=True)

            logger.info("服务状态:\n" + result.stdout)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"部署服务失败: {e}")
            return False
        except Exception as e:
            logger.error(f"部署服务异常: {e}")
            return False

    def wait_for_services(self, timeout: int = 300) -> bool:
        """等待服务就绪"""
        logger.info(f"等待服务就绪，超时时间: {timeout}秒")

        start_time = time.time()
        services_ready = {
            'postgres': False,
            'redis': False,
            'api': False,
            'monitoring': False
        }

        while time.time() - start_time < timeout:
            # 检查PostgreSQL
            if not services_ready['postgres']:
                if self._check_service_health('postgres', 5432):
                    services_ready['postgres'] = True
                    logger.info("✅ PostgreSQL服务就绪")

            # 检查Redis
            if not services_ready['redis']:
                if self._check_service_health('redis', 6379):
                    services_ready['redis'] = True
                    logger.info("✅ Redis服务就绪")

            # 检查API服务
            if not services_ready['api']:
                if self._check_service_health('api', 8000):
                    services_ready['api'] = True
                    logger.info("✅ API服务就绪")

            # 检查监控服务
            if not services_ready['monitoring']:
                if self._check_service_health('prometheus', 9090):
                    services_ready['monitoring'] = True
                    logger.info("✅ 监控服务就绪")

            # 检查是否所有服务都就绪
            if all(services_ready.values()):
                logger.info("🎉 所有服务都已就绪！")
                return True

            # 等待一段时间再检查
            time.sleep(10)

        logger.error("服务启动超时")
        return False

    def _check_service_health(self, service: str, port: int) -> bool:
        """检查服务健康状态"""
        try:
            if service == 'postgres':
                # PostgreSQL健康检查
                import psycopg2
                conn = psycopg2.connect(
                    host='localhost',
                    port=port,
                    database='rqa2025',
                    user='postgres',
                    password='password',
                    connect_timeout=5
                )
                conn.close()
                return True

            elif service == 'redis':
                # Redis健康检查
                import redis
                r = redis.Redis(host='localhost', port=port, decode_responses=True)
                r.ping()
                return True

            elif service == 'api':
                # API服务健康检查
                response = requests.get(f'http://localhost:{port}/health', timeout=5)
                return response.status_code == 200

            elif service == 'prometheus':
                # Prometheus健康检查
                response = requests.get(f'http://localhost:{port}/-/healthy', timeout=5)
                return response.status_code == 200

        except Exception:
            return False

        return False

    def run_health_checks(self) -> Dict[str, Any]:
        """运行健康检查"""
        logger.info("运行健康检查...")

        health_results = {}

        # 检查服务状态
        try:
            result = subprocess.run(['docker-compose', 'ps'],
                                    capture_output=True, text=True, check=True)
            health_results['services_status'] = result.stdout
        except Exception as e:
            health_results['services_status_error'] = str(e)

        # 检查端口监听
        health_results['ports'] = self._check_ports()

        # 检查日志
        health_results['logs'] = self._check_logs()

        return health_results

    def _check_ports(self) -> Dict[str, bool]:
        """检查端口监听状态"""
        ports_to_check = {
            'postgres': 5432,
            'redis': 6379,
            'api': 8000,
            'inference': 8001,
            'prometheus': 9090,
            'grafana': 3000
        }

        port_status = {}
        for service, port in ports_to_check.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                port_status[service] = result == 0
            except Exception:
                port_status[service] = False

        return port_status

    def _check_logs(self) -> Dict[str, str]:
        """检查服务日志"""
        logs = {}
        services = ['postgres', 'redis', 'api', 'inference', 'prometheus', 'grafana']

        for service in services:
            try:
                result = subprocess.run(['docker-compose', 'logs', '--tail=10', service],
                                        capture_output=True, text=True, check=True)
                logs[service] = result.stdout
            except Exception as e:
                logs[service] = f"获取日志失败: {e}"

        return logs

    def generate_deployment_report(self) -> Dict[str, Any]:
        """生成部署报告"""
        deployment_time = time.time() - self.start_time

        report = {
            'deployment_status': 'success' if all(self.deployment_status.values()) else 'failed',
            'deployment_time': deployment_time,
            'deployment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': str(self.config_path),
            'deployment_status_details': self.deployment_status,
            'timestamp': time.time()
        }

        # 保存部署报告
        timestamp = int(time.time())
        report_file = f"deploy/reports/deployment_report_{timestamp}.json"

        try:
            Path(report_file).parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"部署报告已保存到: {report_file}")
        except Exception as e:
            logger.error(f"保存部署报告失败: {e}")

        return report

    def deploy(self) -> bool:
        """执行完整部署流程"""
        logger.info("🚀 开始生产环境部署...")

        try:
            # 1. 检查前置条件
            if not self.check_prerequisites():
                return False

            # 2. 部署服务
            if not self.deploy_services():
                return False

            # 3. 等待服务就绪
            if not self.wait_for_services():
                return False

            # 4. 运行健康检查
            health_results = self.run_health_checks()

            # 5. 生成部署报告
            deployment_report = self.generate_deployment_report()

            # 6. 输出部署结果
            self._print_deployment_summary(deployment_report, health_results)

            logger.info("🎉 生产环境部署完成！")
            return True

        except Exception as e:
            logger.error(f"部署过程中发生异常: {e}")
            return False

    def _print_deployment_summary(self, deployment_report: Dict[str, Any],
                                  health_results: Dict[str, Any]):
        """打印部署摘要"""
        print("\n" + "="*60)
        print("生产环境部署摘要")
        print("="*60)

        print(f"部署状态: {deployment_report['deployment_status'].upper()}")
        print(f"部署时间: {deployment_report['deployment_date']}")
        print(f"部署耗时: {deployment_report['deployment_time']:.2f} 秒")
        print(f"配置文件: {deployment_report['config_file']}")

        print("\n端口监听状态:")
        print("-"*30)
        for service, status in health_results.get('ports', {}).items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {service}: {'监听中' if status else '未监听'}")

        print("\n服务状态:")
        print("-"*30)
        print(health_results.get('services_status', '无法获取服务状态'))


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python deploy_production.py <config_file>")
        print("示例: python deploy_production.py ../config/deployment_config.json")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建部署器并执行部署
    deployer = ProductionDeployer(config_path)

    if deployer.deploy():
        print("\n✅ 部署成功！")
        sys.exit(0)
    else:
        print("\n❌ 部署失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
