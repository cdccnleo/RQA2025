#!/usr/bin/env python3
"""
RQA2025 环境检查器
自动化检查生产环境配置和依赖
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EnvironmentChecker:
    """环境检查器"""

    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = project_root
        self.checks = {
            'python': self._check_python_environment,
            'dependencies': self._check_dependencies,
            'system': self._check_system_requirements,
            'network': self._check_network_connectivity,
            'storage': self._check_storage_space,
            'permissions': self._check_permissions,
            'services': self._check_services,
            'config': self._check_configuration
        }

    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有环境检查"""
        print(f"🔍 开始 {self.environment} 环境检查...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'checks': {},
            'overall_success': True
        }

        for check_name, check_func in self.checks.items():
            print(f"\n📋 检查 {check_name}...")

            try:
                check_result = check_func()
                results['checks'][check_name] = check_result

                if not check_result['success']:
                    results['overall_success'] = False

                status = "✅" if check_result['success'] else "❌"
                print(f"  {status} {check_result['message']}")

                if check_result.get('details'):
                    for detail in check_result['details']:
                        print(f"    - {detail}")

            except Exception as e:
                error_result = {
                    'success': False,
                    'message': f"检查失败: {str(e)}",
                    'error': str(e)
                }
                results['checks'][check_name] = error_result
                results['overall_success'] = False
                print(f"  ❌ {error_result['message']}")

        return results

    def _check_python_environment(self) -> Dict[str, Any]:
        """检查Python环境"""
        try:
            # 检查Python版本
            python_version = sys.version_info
            version_ok = python_version >= (3, 8)

            # 检查虚拟环境
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

            # 检查conda环境
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
            conda_ok = conda_env == 'rqa' if self.environment == 'production' else True

            success = version_ok and (in_venv or conda_ok)

            details = [
                f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}",
                f"虚拟环境: {'是' if in_venv else '否'}",
                f"Conda环境: {conda_env}"
            ]

            return {
                'success': success,
                'message': f"Python环境检查 {'通过' if success else '失败'}",
                'details': details,
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'in_venv': in_venv,
                'conda_env': conda_env
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"Python环境检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖包"""
        try:
            required_packages = [
                'numpy', 'pandas', 'sklearn', 'matplotlib',
                'redis', 'psycopg2', 'requests', 'yaml',
                'pytest', 'pytest-cov', 'docker'
            ]

            missing_packages = []
            installed_packages = []

            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                    installed_packages.append(package)
                except ImportError:
                    missing_packages.append(package)

            success = len(missing_packages) == 0

            details = [
                f"已安装包: {len(installed_packages)}/{len(required_packages)}",
                f"缺失包: {', '.join(missing_packages) if missing_packages else '无'}"
            ]

            return {
                'success': success,
                'message': f"依赖包检查 {'通过' if success else '失败'}",
                'details': details,
                'installed_packages': installed_packages,
                'missing_packages': missing_packages
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"依赖包检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_system_requirements(self) -> Dict[str, Any]:
        """检查系统要求"""
        try:
            import psutil

            # 检查CPU
            cpu_count = psutil.cpu_count()
            cpu_ok = cpu_count >= 4

            # 检查内存
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_ok = memory_gb >= 8

            # 检查磁盘
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            disk_ok = disk_gb >= 10

            success = cpu_ok and memory_ok and disk_ok

            details = [
                f"CPU核心数: {cpu_count} (最低要求: 4)",
                f"内存: {memory_gb:.1f}GB (最低要求: 8GB)",
                f"可用磁盘: {disk_gb:.1f}GB (最低要求: 10GB)"
            ]

            return {
                'success': success,
                'message': f"系统要求检查 {'通过' if success else '失败'}",
                'details': details,
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'disk_gb': disk_gb
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"系统要求检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """检查网络连接"""
        try:
            import requests

            test_urls = [
                'https://www.baidu.com',
                'https://www.google.com',
                'https://pypi.org'
            ]

            successful_connections = []
            failed_connections = []

            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        successful_connections.append(url)
                    else:
                        failed_connections.append(url)
                except:
                    failed_connections.append(url)

            success = len(successful_connections) >= 1

            details = [
                f"成功连接: {len(successful_connections)}/{len(test_urls)}",
                f"失败连接: {', '.join(failed_connections) if failed_connections else '无'}"
            ]

            return {
                'success': success,
                'message': f"网络连接检查 {'通过' if success else '失败'}",
                'details': details,
                'successful_connections': successful_connections,
                'failed_connections': failed_connections
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"网络连接检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_storage_space(self) -> Dict[str, Any]:
        """检查存储空间"""
        try:
            import psutil

            # 检查项目目录空间
            project_path = str(self.project_root)
            disk_usage = psutil.disk_usage(project_path)

            # 检查日志目录空间
            log_path = self.project_root / 'logs'
            log_disk_usage = psutil.disk_usage(str(log_path)) if log_path.exists() else disk_usage

            # 检查数据目录空间
            data_path = self.project_root / 'data'
            data_disk_usage = psutil.disk_usage(
                str(data_path)) if data_path.exists() else disk_usage

            project_gb = disk_usage.free / (1024**3)
            log_gb = log_disk_usage.free / (1024**3)
            data_gb = data_disk_usage.free / (1024**3)

            success = project_gb >= 5 and log_gb >= 2 and data_gb >= 10

            details = [
                f"项目目录可用空间: {project_gb:.1f}GB",
                f"日志目录可用空间: {log_gb:.1f}GB",
                f"数据目录可用空间: {data_gb:.1f}GB"
            ]

            return {
                'success': success,
                'message': f"存储空间检查 {'通过' if success else '失败'}",
                'details': details,
                'project_gb': project_gb,
                'log_gb': log_gb,
                'data_gb': data_gb
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"存储空间检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_permissions(self) -> Dict[str, Any]:
        """检查文件权限"""
        try:
            # 检查关键目录权限
            critical_paths = [
                'logs',
                'data',
                'config',
                'reports'
            ]

            permission_issues = []
            accessible_paths = []

            for path_name in critical_paths:
                path = self.project_root / path_name
                try:
                    # 检查读写权限
                    if path.exists():
                        # 检查读权限
                        os.access(path, os.R_OK)
                        # 检查写权限
                        os.access(path, os.W_OK)
                        accessible_paths.append(path_name)
                    else:
                        # 尝试创建目录
                        path.mkdir(parents=True, exist_ok=True)
                        accessible_paths.append(path_name)
                except Exception as e:
                    permission_issues.append(f"{path_name}: {str(e)}")

            success = len(permission_issues) == 0

            details = [
                f"可访问路径: {len(accessible_paths)}/{len(critical_paths)}",
                f"权限问题: {', '.join(permission_issues) if permission_issues else '无'}"
            ]

            return {
                'success': success,
                'message': f"文件权限检查 {'通过' if success else '失败'}",
                'details': details,
                'accessible_paths': accessible_paths,
                'permission_issues': permission_issues
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"文件权限检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_services(self) -> Dict[str, Any]:
        """检查服务状态"""
        try:
            # 检查Docker服务
            docker_ok = False
            try:
                result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
                docker_ok = result.returncode == 0
            except:
                pass

            # 检查Docker Compose
            compose_ok = False
            try:
                result = subprocess.run(['docker-compose', '--version'],
                                        capture_output=True, text=True)
                compose_ok = result.returncode == 0
            except:
                pass

            # 检查Redis连接
            redis_ok = False
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                redis_ok = True
            except:
                pass

            success = docker_ok and compose_ok

            details = [
                f"Docker: {'可用' if docker_ok else '不可用'}",
                f"Docker Compose: {'可用' if compose_ok else '不可用'}",
                f"Redis: {'可用' if redis_ok else '不可用'}"
            ]

            return {
                'success': success,
                'message': f"服务状态检查 {'通过' if success else '失败'}",
                'details': details,
                'docker_ok': docker_ok,
                'compose_ok': compose_ok,
                'redis_ok': redis_ok
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"服务状态检查失败: {str(e)}",
                'error': str(e)
            }

    def _check_configuration(self) -> Dict[str, Any]:
        """检查配置文件"""
        try:
            # 检查配置文件存在性
            config_files = [
                'config/default.json',
                'config/production.json',
                'deploy/docker-compose.yml'
            ]

            existing_files = []
            missing_files = []

            for config_file in config_files:
                file_path = self.project_root / config_file
                if file_path.exists():
                    existing_files.append(config_file)
                else:
                    missing_files.append(config_file)
                    print(f"    Debug: 文件不存在: {file_path}")

            success = len(missing_files) == 0

            details = [
                f"配置文件存在: {len(existing_files)}/{len(config_files)}",
                f"缺失文件: {', '.join(missing_files) if missing_files else '无'}"
            ]

            return {
                'success': success,
                'message': f"配置文件检查 {'通过' if success else '失败'}",
                'details': details,
                'existing_files': existing_files,
                'missing_files': missing_files
            }

        except Exception as e:
            return {
                'success': False,
                'message': f"配置文件检查失败: {str(e)}",
                'error': str(e)
            }

    def generate_report(self, results: Dict[str, Any], output_file: str = "") -> str:
        """生成检查报告"""
        if not output_file:
            output_file = f"reports/environment_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 确保报告目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n📋 环境检查报告已生成: {output_file}")

        return output_file

    def get_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """获取检查摘要"""
        total_checks = len(results['checks'])
        passed_checks = sum(1 for check in results['checks'].values() if check['success'])
        failed_checks = total_checks - passed_checks

        summary = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'overall_success': results['overall_success'],
            'failed_checks_list': []
        }

        for check_name, check_result in results['checks'].items():
            if not check_result['success']:
                summary['failed_checks_list'].append({
                    'check': check_name,
                    'message': check_result['message'],
                    'error': check_result.get('error', '')
                })

        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 环境检查器')
    parser.add_argument('--env', type=str, default='production', help='环境类型')
    parser.add_argument('--output', type=str, help='输出文件路径')

    args = parser.parse_args()

    # 创建检查器
    checker = EnvironmentChecker(environment=args.env)

    # 运行检查
    results = checker.run_all_checks()

    # 生成报告
    output_file = args.output or f"reports/environment_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    checker.generate_report(results, output_file)

    # 显示摘要
    summary = checker.get_summary(results)

    print(f"\n📈 检查摘要:")
    print(f"  总检查项: {summary['total_checks']}")
    print(f"  通过检查: {summary['passed_checks']}")
    print(f"  失败检查: {summary['failed_checks']}")
    print(f"  整体状态: {'✅ 通过' if summary['overall_success'] else '❌ 需要修复'}")

    if summary['failed_checks_list']:
        print(f"\n⚠️ 失败的检查项:")
        for failed_check in summary['failed_checks_list']:
            print(f"  - {failed_check['check']}: {failed_check['message']}")

    # 返回适当的退出码
    sys.exit(0 if summary['overall_success'] else 1)


if __name__ == "__main__":
    main()
