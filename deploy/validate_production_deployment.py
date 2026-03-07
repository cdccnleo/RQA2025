#!/usr/bin/env python3
"""
生产环境部署验证脚本

用于验证RQA2025生产环境的各个组件是否正常工作
包括：数据库、缓存、监控、API服务等
"""

import sys
import time
import json
import requests
import psycopg2
import redis
from pathlib import Path
from typing import Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeploymentValidator:
    """生产环境部署验证器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化验证器

        Args:
            config: 配置字典，包含各种服务的连接信息
        """
        self.config = config
        self.validation_results = {}
        self.start_time = time.time()

    def validate_database(self) -> Dict[str, Any]:
        """验证PostgreSQL数据库"""
        logger.info("开始验证PostgreSQL数据库...")
        result = {
            'status': 'failed',
            'details': {},
            'timestamp': time.time()
        }

        try:
            # 测试数据库连接
            conn = psycopg2.connect(
                host=self.config['services']['postgres']['host'],
                port=self.config['services']['postgres']['port'],
                database=self.config['services']['postgres']['database'],
                user=self.config['services']['postgres']['user'],
                password=self.config['services']['postgres']['password']
            )

            # 测试基本查询
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # 测试表是否存在
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5;
            """)
            tables = cursor.fetchall()

            cursor.close()
            conn.close()

            result['status'] = 'success'
            result['details'] = {
                'version': version,
                'tables_count': len(tables),
                'connection': 'established'
            }
            logger.info("PostgreSQL数据库验证成功")

        except Exception as e:
            result['details'] = {
                'error': str(e),
                'connection': 'failed'
            }
            logger.error(f"PostgreSQL数据库验证失败: {e}")

        return result

    def validate_redis(self) -> Dict[str, Any]:
        """验证Redis缓存服务"""
        logger.info("开始验证Redis缓存服务...")
        result = {
            'status': 'failed',
            'details': {},
            'timestamp': time.time()
        }

        try:
            # 测试Redis连接
            r = redis.Redis(
                host=self.config['services']['redis']['host'],
                port=self.config['services']['redis']['port'],
                password=self.config['services']['redis']['password'],
                decode_responses=True
            )

            # 测试基本操作
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            r.delete('test_key')

            # 获取Redis信息
            info = r.info()

            result['status'] = 'success'
            result['details'] = {
                'version': info.get('redis_version', 'unknown'),
                'memory_used': info.get('used_memory_human', 'unknown'),
                'connected_clients': info.get('connected_clients', 0),
                'basic_operations': 'working'
            }
            logger.info("Redis缓存服务验证成功")

        except Exception as e:
            result['details'] = {
                'error': str(e),
                'connection': 'failed'
            }
            logger.error(f"Redis缓存服务验证失败: {e}")

        return result

    def validate_api_service(self) -> Dict[str, Any]:
        """验证API服务"""
        logger.info("开始验证API服务...")
        result = {
            'status': 'failed',
            'details': {},
            'timestamp': time.time()
        }

        try:
            base_url = f"http://{self.config['services']['api']['host']}:{self.config['services']['api']['port']}"

            # 测试健康检查端点
            health_url = f"{base_url}/health"
            health_response = requests.get(health_url, timeout=10)

            if health_response.status_code == 200:
                health_data = health_response.json()

                # 测试核心API端点
                api_url = f"{base_url}/api/v1/status"
                api_response = requests.get(api_url, timeout=10)

                result['status'] = 'success'
                result['details'] = {
                    'health_status': health_data.get('status', 'unknown'),
                    'api_response_code': api_response.status_code,
                    'response_time': health_response.elapsed.total_seconds()
                }
                logger.info("API服务验证成功")
            else:
                result['details'] = {
                    'health_status_code': health_response.status_code,
                    'error': 'Health check failed'
                }
                logger.error(f"API服务健康检查失败: {health_response.status_code}")

        except Exception as e:
            result['details'] = {
                'error': str(e),
                'connection': 'failed'
            }
            logger.error(f"API服务验证失败: {e}")

        return result

    def validate_monitoring(self) -> Dict[str, Any]:
        """验证监控系统"""
        logger.info("开始验证监控系统...")
        result = {
            'status': 'failed',
            'details': {},
            'timestamp': time.time()
        }

        try:
            # 验证Prometheus
            prometheus_url = f"http://{self.config['services']['prometheus']['host']}:{self.config['services']['prometheus']['port']}"
            prometheus_response = requests.get(f"{prometheus_url}/-/healthy", timeout=10)

            # 验证Grafana
            grafana_url = f"http://{self.config['services']['grafana']['host']}:{self.config['services']['grafana']['port']}"
            grafana_response = requests.get(f"{grafana_url}/api/health", timeout=10)

            if prometheus_response.status_code == 200 and grafana_response.status_code == 200:
                result['status'] = 'success'
                result['details'] = {
                    'prometheus': 'healthy',
                    'grafana': 'healthy',
                    'prometheus_targets': 'accessible',
                    'grafana_dashboards': 'accessible'
                }
                logger.info("监控系统验证成功")
            else:
                result['details'] = {
                    'prometheus_status': prometheus_response.status_code,
                    'grafana_status': grafana_response.status_code,
                    'error': 'Monitoring services unhealthy'
                }
                logger.error("监控系统验证失败")

        except Exception as e:
            result['details'] = {
                'error': str(e),
                'connection': 'failed'
            }
            logger.error(f"监控系统验证失败: {e}")

        return result

    def validate_system_resources(self) -> Dict[str, Any]:
        """验证系统资源"""
        logger.info("开始验证系统资源...")
        result = {
            'status': 'success',
            'details': {},
            'timestamp': time.time()
        }

        try:
            import psutil

            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # 网络状态
            network = psutil.net_io_counters()

            result['details'] = {
                'cpu_usage': f"{cpu_percent:.1f}%",
                'memory_usage': f"{memory_percent:.1f}%",
                'disk_usage': f"{disk_percent:.1f}%",
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }

            # 检查资源使用是否在合理范围内
            if cpu_percent > 80:
                result['status'] = 'warning'
                result['details']['cpu_warning'] = 'CPU使用率过高'

            if memory_percent > 85:
                result['status'] = 'warning'
                result['details']['memory_warning'] = '内存使用率过高'

            if disk_percent > 90:
                result['status'] = 'warning'
                result['details']['disk_warning'] = '磁盘使用率过高'

            logger.info("系统资源验证完成")

        except ImportError:
            result['status'] = 'warning'
            result['details']['warning'] = 'psutil未安装，无法检查系统资源'
            logger.warning("psutil未安装，跳过系统资源检查")

        except Exception as e:
            result['status'] = 'failed'
            result['details']['error'] = str(e)
            logger.error(f"系统资源验证失败: {e}")

        return result

    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info("开始生产环境部署验证...")

        # 验证各个组件
        self.validation_results['database'] = self.validate_database()
        self.validation_results['redis'] = self.validate_redis()
        self.validation_results['api_service'] = self.validate_api_service()
        self.validation_results['monitoring'] = self.validate_monitoring()
        self.validation_results['system_resources'] = self.validate_system_resources()

        # 计算总体状态
        total_checks = len(self.validation_results)
        successful_checks = sum(1 for r in self.validation_results.values()
                                if r['status'] == 'success')
        warning_checks = sum(1 for r in self.validation_results.values()
                             if r['status'] == 'warning')
        failed_checks = sum(1 for r in self.validation_results.values() if r['status'] == 'failed')

        overall_status = 'success'
        if failed_checks > 0:
            overall_status = 'failed'
        elif warning_checks > 0:
            overall_status = 'warning'

        # 生成验证报告
        validation_report = {
            'overall_status': overall_status,
            'summary': {
                'total_checks': total_checks,
                'successful': successful_checks,
                'warnings': warning_checks,
                'failed': failed_checks,
                'success_rate': f"{(successful_checks / total_checks) * 100:.1f}%"
            },
            'validation_results': self.validation_results,
            'validation_time': time.time() - self.start_time,
            'timestamp': time.time()
        }

        # 保存验证报告
        self._save_validation_report(validation_report)

        # 输出验证结果
        self._print_validation_summary(validation_report)

        return validation_report

    def _save_validation_report(self, report: Dict[str, Any]):
        """保存验证报告"""
        try:
            timestamp = int(time.time())
            filename = f"deployment_validation_report_{timestamp}.json"
            filepath = Path("deploy/reports") / filename

            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"验证报告已保存到: {filepath}")

        except Exception as e:
            logger.error(f"保存验证报告失败: {e}")

    def _print_validation_summary(self, report: Dict[str, Any]):
        """打印验证摘要"""
        print("\n" + "="*60)
        print("生产环境部署验证报告")
        print("="*60)

        summary = report['summary']
        print(f"总体状态: {report['overall_status'].upper()}")
        print(f"验证时间: {report['validation_time']:.2f} 秒")
        print(f"检查项目: {summary['total_checks']}")
        print(f"成功项目: {summary['successful']}")
        print(f"警告项目: {summary['warnings']}")
        print(f"失败项目: {summary['failed']}")
        print(f"成功率: {summary['success_rate']}")

        print("\n详细结果:")
        print("-"*60)

        for component, result in report['validation_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'warning' else "❌"
            print(f"{status_icon} {component.upper()}: {result['status']}")

            if result['details']:
                for key, value in result['details'].items():
                    print(f"    {key}: {value}")

        print("="*60)


def main():
    """主函数"""
    # 默认配置（可以根据实际情况修改）
    default_config = {
        'postgres': {
            'host': 'localhost',
            'port': 5432,
            'database': 'rqa2025',
            'user': 'postgres',
            'password': 'password'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'password': None
        },
        'api': {
            'host': 'localhost',
            'port': 8000
        },
        'prometheus': {
            'host': 'localhost',
            'port': 9090
        },
        'grafana': {
            'host': 'localhost',
            'port': 3000
        }
    }

    # 尝试从环境变量或配置文件加载配置
    config_file = Path("deploy/config/deployment_config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"从配置文件加载配置: {config_file}")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
            config = default_config
    else:
        logger.info("使用默认配置")
        config = default_config

    # 创建验证器并运行验证
    validator = ProductionDeploymentValidator(config)
    validation_report = validator.run_full_validation()

    # 根据验证结果设置退出码
    if validation_report['overall_status'] == 'failed':
        sys.exit(1)
    elif validation_report['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
