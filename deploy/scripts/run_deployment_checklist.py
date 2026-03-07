#!/usr/bin/env python3
"""
部署检查清单执行脚本

自动化执行生产环境部署检查清单中的各项检查
包括：基础设施检查、数据库检查、缓存服务检查、监控系统检查等
"""

import sys
import time
import json
import subprocess
import requests
import psutil
from pathlib import Path
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_checklist.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeploymentChecklistExecutor:
    """部署检查清单执行器"""

    def __init__(self, config_path: str):
        """
        初始化检查清单执行器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.check_results = {}
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

    def run_infrastructure_checks(self) -> Dict[str, Any]:
        """运行基础设施检查"""
        logger.info("🔧 开始基础设施检查...")

        results = {
            'server_resources': self._check_server_resources(),
            'network_config': self._check_network_config(),
            'security_config': self._check_security_config()
        }

        return results

    def _check_server_resources(self) -> Dict[str, Any]:
        """检查服务器资源"""
        logger.info("检查服务器资源...")

        try:
            # CPU资源检查
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存资源检查
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent

            # 磁盘资源检查
            disk = psutil.disk_usage('/')
            disk_total_gb = disk.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            disk_percent = disk.percent

            # 评估结果
            cpu_status = 'pass' if cpu_percent < 70 else 'warning'
            memory_status = 'pass' if memory_percent < 80 else 'warning'
            disk_status = 'pass' if disk_percent < 90 else 'warning'

            results = {
                'status': 'pass' if all(s == 'pass' for s in [cpu_status, memory_status, disk_status]) else 'warning',
                'cpu': {
                    'status': cpu_status,
                    'cores': cpu_count,
                    'usage_percent': cpu_percent,
                    'recommendation': 'CPU使用率正常' if cpu_percent < 70 else 'CPU使用率过高，建议优化'
                },
                'memory': {
                    'status': memory_status,
                    'total_gb': round(memory_total_gb, 2),
                    'available_gb': round(memory_available_gb, 2),
                    'usage_percent': memory_percent,
                    'recommendation': '内存使用率正常' if memory_percent < 80 else '内存使用率过高，建议增加内存'
                },
                'disk': {
                    'status': disk_status,
                    'total_gb': round(disk_total_gb, 2),
                    'free_gb': round(disk_free_gb, 2),
                    'usage_percent': disk_percent,
                    'recommendation': '磁盘空间充足' if disk_percent < 90 else '磁盘空间不足，建议清理或扩容'
                }
            }

            logger.info("服务器资源检查完成")
            return results

        except Exception as e:
            logger.error(f"服务器资源检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _check_network_config(self) -> Dict[str, Any]:
        """检查网络配置"""
        logger.info("检查网络配置...")

        try:
            # 检查网络连通性
            network_tests = [
                ('Google DNS', '8.8.8.8', 53),
                ('Localhost', '127.0.0.1', 80),
                ('External HTTP', 'http://www.google.com', None)
            ]

            network_results = {}
            all_passed = True

            for name, target, port in network_tests:
                if port:
                    # TCP端口检查
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        result = sock.connect_ex((target, port))
                        sock.close()
                        status = 'pass' if result == 0 else 'fail'
                        if status == 'fail':
                            all_passed = False
                        network_results[name] = {'status': status, 'port': port}
                    except Exception as e:
                        network_results[name] = {'status': 'fail', 'error': str(e)}
                        all_passed = False
                else:
                    # HTTP连接检查
                    try:
                        response = requests.get(target, timeout=5)
                        status = 'pass' if response.status_code == 200 else 'fail'
                        if status == 'fail':
                            all_passed = False
                        network_results[name] = {'status': status,
                                                 'response_code': response.status_code}
                    except Exception as e:
                        network_results[name] = {'status': 'fail', 'error': str(e)}
                        all_passed = False

            # 检查端口占用
            ports_to_check = [5432, 6379, 8000, 8001, 9090, 3000]
            port_status = {}

            for port in ports_to_check:
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    sock.close()
                    port_status[port] = 'occupied' if result == 0 else 'free'
                except Exception:
                    port_status[port] = 'unknown'

            results = {
                'status': 'pass' if all_passed else 'fail',
                'connectivity': network_results,
                'port_status': port_status,
                'recommendation': '网络配置正常' if all_passed else '网络配置存在问题，需要检查'
            }

            logger.info("网络配置检查完成")
            return results

        except Exception as e:
            logger.error(f"网络配置检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _check_security_config(self) -> Dict[str, Any]:
        """检查安全配置"""
        logger.info("检查安全配置...")

        try:
            # 检查SSH配置
            ssh_config_path = Path('/etc/ssh/sshd_config')
            ssh_status = 'unknown'
            ssh_recommendations = []

            if ssh_config_path.exists():
                try:
                    with open(ssh_config_path, 'r') as f:
                        ssh_content = f.read()

                    # 检查关键安全配置
                    if 'PasswordAuthentication no' in ssh_content:
                        ssh_status = 'pass'
                    else:
                        ssh_status = 'warning'
                        ssh_recommendations.append('建议禁用密码认证')

                    if 'PermitRootLogin no' in ssh_content:
                        ssh_status = 'pass'
                    else:
                        ssh_status = 'warning'
                        ssh_recommendations.append('建议禁用root登录')

                except Exception:
                    ssh_status = 'unknown'

            # 检查防火墙状态
            firewall_status = 'unknown'
            try:
                # 尝试检查Windows防火墙状态
                result = subprocess.run(['netsh', 'advfirewall', 'show', 'allprofiles'],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    firewall_status = 'enabled'
                else:
                    firewall_status = 'disabled'
            except Exception:
                firewall_status = 'unknown'

            results = {
                'status': 'pass' if ssh_status == 'pass' and firewall_status == 'enabled' else 'warning',
                'ssh': {
                    'status': ssh_status,
                    'recommendations': ssh_recommendations
                },
                'firewall': {
                    'status': firewall_status,
                    'recommendation': '防火墙已启用' if firewall_status == 'enabled' else '建议启用防火墙'
                }
            }

            logger.info("安全配置检查完成")
            return results

        except Exception as e:
            logger.error(f"安全配置检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def run_database_checks(self) -> Dict[str, Any]:
        """运行数据库检查"""
        logger.info("🗄️ 开始数据库检查...")

        try:
            # 尝试连接PostgreSQL
            postgres_status = self._check_postgresql()

            # 尝试连接Redis
            redis_status = self._check_redis()

            results = {
                'postgresql': postgres_status,
                'redis': redis_status,
                'overall_status': 'pass' if all(s['status'] == 'pass' for s in [postgres_status, redis_status]) else 'fail'
            }

            return results

        except Exception as e:
            logger.error(f"数据库检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _check_postgresql(self) -> Dict[str, Any]:
        """检查PostgreSQL"""
        try:
            import psycopg2

            # 尝试连接数据库
            conn = psycopg2.connect(
                host=self.config['services']['postgres']['host'],
                port=self.config['services']['postgres']['port'],
                database=self.config['services']['postgres']['database'],
                user=self.config['services']['postgres']['user'],
                password=self.config['services']['postgres']['password'],
                connect_timeout=10
            )

            # 检查数据库版本和状态
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            cursor.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cursor.fetchone()[0]

            cursor.execute("SELECT count(*) FROM pg_stat_activity;")
            connection_count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                'status': 'pass',
                'version': version,
                'table_count': table_count,
                'connection_count': connection_count,
                'recommendation': 'PostgreSQL连接正常'
            }

        except ImportError:
            return {
                'status': 'warning',
                'error': 'psycopg2未安装',
                'recommendation': '请安装psycopg2: pip install psycopg2-binary'
            }
        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查PostgreSQL服务状态和连接配置'
            }

    def _check_redis(self) -> Dict[str, Any]:
        """检查Redis"""
        try:
            import redis

            # 尝试连接Redis
            r = redis.Redis(
                host=self.config['services']['redis']['host'],
                port=self.config['services']['redis']['port'],
                password=self.config['services']['redis']['password'],
                decode_responses=True,
                socket_timeout=10
            )

            # 检查Redis状态
            info = r.info()
            version = info.get('redis_version', 'unknown')
            memory_used = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)

            # 测试基本操作
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            r.delete('test_key')

            return {
                'status': 'pass',
                'version': version,
                'memory_used': memory_used,
                'connected_clients': connected_clients,
                'basic_operations': 'working',
                'recommendation': 'Redis连接正常'
            }

        except ImportError:
            return {
                'status': 'warning',
                'error': 'redis未安装',
                'recommendation': '请安装redis: pip install redis'
            }
        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查Redis服务状态和连接配置'
            }

    def run_monitoring_checks(self) -> Dict[str, Any]:
        """运行监控系统检查"""
        logger.info("📊 开始监控系统检查...")

        try:
            # 检查Prometheus
            prometheus_status = self._check_prometheus()

            # 检查Grafana
            grafana_status = self._check_grafana()

            results = {
                'prometheus': prometheus_status,
                'grafana': grafana_status,
                'overall_status': 'pass' if all(s['status'] == 'pass' for s in [prometheus_status, grafana_status]) else 'fail'
            }

            return results

        except Exception as e:
            logger.error(f"监控系统检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _check_prometheus(self) -> Dict[str, Any]:
        """检查Prometheus"""
        try:
            base_url = f"http://{self.config['services']['prometheus']['host']}:{self.config['services']['prometheus']['port']}"

            # 健康检查
            health_response = requests.get(f"{base_url}/-/healthy", timeout=10)
            if health_response.status_code != 200:
                return {
                    'status': 'fail',
                    'error': f'健康检查失败: {health_response.status_code}',
                    'recommendation': '检查Prometheus服务状态'
                }

            # 状态检查
            status_response = requests.get(f"{base_url}/api/v1/status/targets", timeout=10)
            if status_response.status_code == 200:
                targets_data = status_response.json()
                active_targets = targets_data.get('data', {}).get('activeTargets', [])

                return {
                    'status': 'pass',
                    'health': 'healthy',
                    'active_targets': len(active_targets),
                    'recommendation': 'Prometheus运行正常'
                }
            else:
                return {
                    'status': 'warning',
                    'error': f'状态检查失败: {status_response.status_code}',
                    'recommendation': '检查Prometheus配置'
                }

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查Prometheus服务状态和网络连接'
            }

    def _check_grafana(self) -> Dict[str, Any]:
        """检查Grafana"""
        try:
            base_url = f"http://{self.config['services']['grafana']['host']}:{self.config['services']['grafana']['port']}"

            # 健康检查
            health_response = requests.get(f"{base_url}/api/health", timeout=10)
            if health_response.status_code != 200:
                return {
                    'status': 'fail',
                    'error': f'健康检查失败: {health_response.status_code}',
                    'recommendation': '检查Grafana服务状态'
                }

            health_data = health_response.json()
            database_status = health_data.get('database', 'unknown')

            if database_status == 'ok':
                return {
                    'status': 'pass',
                    'health': 'healthy',
                    'database': 'connected',
                    'recommendation': 'Grafana运行正常'
                }
            else:
                return {
                    'status': 'warning',
                    'error': f'数据库连接异常: {database_status}',
                    'recommendation': '检查Grafana数据库配置'
                }

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查Grafana服务状态和网络连接'
            }

    def run_api_checks(self) -> Dict[str, Any]:
        """运行API服务检查"""
        logger.info("🔌 开始API服务检查...")

        try:
            # 检查API服务
            api_status = self._check_api_service()

            # 检查推理服务
            inference_status = self._check_inference_service()

            results = {
                'api': api_status,
                'inference': inference_status,
                'overall_status': 'pass' if all(s['status'] == 'pass' for s in [api_status, inference_status]) else 'fail'
            }

            return results

        except Exception as e:
            logger.error(f"API服务检查失败: {e}")
            return {
                'status': 'fail',
                'error': str(e)
            }

    def _check_api_service(self) -> Dict[str, Any]:
        """检查API服务"""
        try:
            base_url = f"http://{self.config['services']['api']['host']}:{self.config['services']['api']['port']}"

            # 健康检查
            health_response = requests.get(f"{base_url}/health", timeout=10)
            if health_response.status_code != 200:
                return {
                    'status': 'fail',
                    'error': f'健康检查失败: {health_response.status_code}',
                    'recommendation': '检查API服务状态'
                }

            health_data = health_response.json()
            status = health_data.get('status', 'unknown')

            # 测试核心API端点
            try:
                api_response = requests.get(f"{base_url}/api/v1/status", timeout=10)
                api_status = 'working' if api_response.status_code == 200 else 'failed'
            except Exception:
                api_status = 'failed'

            return {
                'status': 'pass',
                'health': status,
                'api_endpoint': api_status,
                'response_time': health_response.elapsed.total_seconds(),
                'recommendation': 'API服务运行正常'
            }

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查API服务状态和网络连接'
            }

    def _check_inference_service(self) -> Dict[str, Any]:
        """检查推理服务"""
        try:
            base_url = f"http://{self.config['services']['inference']['host']}:{self.config['services']['inference']['port']}"

            # 健康检查
            health_response = requests.get(f"{base_url}/health", timeout=10)
            if health_response.status_code != 200:
                return {
                    'status': 'fail',
                    'error': f'健康检查失败: {health_response.status_code}',
                    'recommendation': '检查推理服务状态'
                }

            health_data = health_response.json()
            status = health_data.get('status', 'unknown')

            return {
                'status': 'pass',
                'health': status,
                'gpu_enabled': self.config['services']['inference']['gpu_enabled'],
                'recommendation': '推理服务运行正常'
            }

        except Exception as e:
            return {
                'status': 'fail',
                'error': str(e),
                'recommendation': '检查推理服务状态和网络连接'
            }

    def run_full_checklist(self) -> Dict[str, Any]:
        """运行完整检查清单"""
        logger.info("🚀 开始执行生产环境部署检查清单...")

        # 执行各项检查
        self.check_results['infrastructure'] = self.run_infrastructure_checks()
        self.check_results['database'] = self.run_database_checks()
        self.check_results['monitoring'] = self.run_monitoring_checks()
        self.check_results['api_services'] = self.run_api_checks()

        # 计算总体结果
        overall_status = self._calculate_overall_status()

        # 生成检查报告
        checklist_report = {
            'overall_status': overall_status,
            'check_results': self.check_results,
            'execution_time': time.time() - self.start_time,
            'timestamp': time.time(),
            'recommendations': self._generate_recommendations()
        }

        # 保存检查报告
        self._save_checklist_report(checklist_report)

        # 输出检查结果
        self._print_checklist_summary(checklist_report)

        return checklist_report

    def _calculate_overall_status(self) -> str:
        """计算总体状态"""
        all_results = []

        # 收集所有检查结果
        for category, results in self.check_results.items():
            if isinstance(results, dict):
                if 'overall_status' in results:
                    all_results.append(results['overall_status'])
                elif 'status' in results:
                    all_results.append(results['status'])
                else:
                    # 递归查找状态
                    for key, value in results.items():
                        if isinstance(value, dict) and 'status' in value:
                            all_results.append(value['status'])

        # 判断总体状态
        if not all_results:
            return 'unknown'
        elif 'fail' in all_results:
            return 'fail'
        elif 'warning' in all_results:
            return 'warning'
        else:
            return 'pass'

    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for category, results in self.check_results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'recommendation' in value:
                        if value['status'] != 'pass':
                            recommendations.append(f"{category}.{key}: {value['recommendation']}")

        return recommendations

    def _save_checklist_report(self, report: Dict[str, Any]):
        """保存检查清单报告"""
        try:
            timestamp = int(time.time())
            filename = f"deployment_checklist_report_{timestamp}.json"
            filepath = Path("deploy/reports") / filename

            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"检查清单报告已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存检查清单报告失败: {e}")

    def _print_checklist_summary(self, report: Dict[str, Any]):
        """打印检查清单摘要"""
        print("\n" + "="*80)
        print("生产环境部署检查清单执行报告")
        print("="*80)

        print(f"总体状态: {report['overall_status'].upper()}")
        print(f"执行时间: {report['execution_time']:.2f} 秒")
        print(f"检查时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n详细检查结果:")
        print("-"*80)

        for category, results in report['check_results'].items():
            print(f"\n📋 {category.upper()} 检查结果:")

            if isinstance(results, dict):
                if 'overall_status' in results:
                    status_icon = "✅" if results['overall_status'] == 'pass' else "⚠️" if results['overall_status'] == 'warning' else "❌"
                    print(f"   {status_icon} 总体状态: {results['overall_status']}")

                for key, value in results.items():
                    if key != 'overall_status':
                        if isinstance(value, dict) and 'status' in value:
                            status_icon = "✅" if value['status'] == 'pass' else "⚠️" if value['status'] == 'warning' else "❌"
                            print(f"   {status_icon} {key}: {value['status']}")

                            if 'recommendation' in value:
                                print(f"       💡 建议: {value['recommendation']}")

        if report['recommendations']:
            print(f"\n🔧 改进建议:")
            print("-"*40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")

        print("="*80)


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python run_deployment_checklist.py <config_file>")
        print("示例: python run_deployment_checklist.py ../config/deployment_config.json")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建检查清单执行器并运行检查
    executor = DeploymentChecklistExecutor(config_path)
    checklist_report = executor.run_full_checklist()

    # 根据检查结果设置退出码
    if checklist_report['overall_status'] == 'pass':
        print("\n✅ 所有检查项目通过！")
        sys.exit(0)
    elif checklist_report['overall_status'] == 'warning':
        print("\n⚠️ 部分检查项目需要关注！")
        sys.exit(2)
    else:
        print("\n❌ 存在严重问题，需要解决！")
        sys.exit(1)


if __name__ == "__main__":
    main()
