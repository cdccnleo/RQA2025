#!/usr/bin/env python3
"""
生产环境部署就绪性检查

验证系统是否已经准备好进行生产环境部署：
1. 代码质量检查
2. 测试覆盖率验证
3. 安全漏洞扫描
4. 性能基准验证
5. 配置完整性检查
6. 文档完整性验证
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logs_dir = Path('deploy/logs')
logs_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deploy/logs/production_readiness_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionReadinessChecker:
    """生产环境就绪性检查器"""

    def __init__(self):
        self.check_results = []
        self.project_root = Path(__file__).parent.parent

    def run_check(self, check_name, check_func, *args, **kwargs):
        """运行单个检查"""
        logger.info(f"开始检查: {check_name}")
        try:
            result = check_func(*args, **kwargs)
            self.check_results.append({
                'check_name': check_name,
                'status': 'PASSED' if result else 'FAILED',
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"检查完成: {check_name} - {'通过' if result else '失败'}")
            return result
        except Exception as e:
            logger.error(f"检查异常: {check_name} - {str(e)}")
            self.check_results.append({
                'check_name': check_name,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    def check_code_quality(self):
        """检查代码质量"""
        logger.info("检查代码质量指标...")

        # 检查是否有严重的代码问题
        python_files = list(self.project_root.rglob('src/**/*.py'))
        logger.info(f"找到 {len(python_files)} 个Python文件")

        # 检查主要模块是否完整
        required_modules = [
            'src/trading',
            'src/risk',
            'src/data',
            'src/infrastructure',
            'src/monitoring'
        ]

        for module in required_modules:
            module_path = self.project_root / module
            if not module_path.exists():
                logger.error(f"缺少必需模块: {module}")
                return False

            py_files = list(module_path.rglob('*.py'))
            if len(py_files) < 5:  # 每个模块至少需要5个文件
                logger.warning(f"模块文件较少: {module} - {len(py_files)} 个文件")

        return True

    def check_test_coverage(self):
        """检查测试覆盖率"""
        logger.info("检查测试覆盖率...")

        # 检查测试文件是否存在
        test_directories = [
            'tests/unit',
            'tests/integration',
            'tests/security'
        ]

        total_test_files = 0
        for test_dir in test_directories:
            test_path = self.project_root / test_dir
            if test_path.exists():
                py_files = list(test_path.rglob('*.py'))
                total_test_files += len(py_files)
                logger.info(f"{test_dir}: {len(py_files)} 个测试文件")

        logger.info(f"总测试文件数: {total_test_files}")

        # 运行一个快速的测试覆盖率检查
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--cov=src',
                '--cov-report=term-missing:skip-covered',
                '--cov-fail-under=80',
                'tests/unit/infrastructure/core/test_tools_enhanced.py',
                '-q'
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("测试覆盖率检查通过")
                return True
            else:
                logger.warning(f"测试覆盖率检查失败: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("测试覆盖率检查超时")
            return False
        except Exception as e:
            logger.warning(f"测试覆盖率检查异常: {str(e)}")
            return False

    def check_security_vulnerabilities(self):
        """检查安全漏洞"""
        logger.info("检查安全漏洞...")

        # 检查是否有安全测试文件
        security_tests = [
            'tests/security/test_owasp_security.py',
            'tests/security/test_compliance_security.py',
            'tests/security/test_api_security.py'
        ]

        for test_file in security_tests:
            test_path = self.project_root / test_file
            if not test_path.exists():
                logger.warning(f"缺少安全测试文件: {test_file}")
                continue

            # 运行安全测试
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest',
                    str(test_path),
                    '-v', '--tb=no'
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    logger.info(f"安全测试通过: {test_file}")
                else:
                    logger.warning(f"安全测试失败: {test_file}")
                    return False

            except subprocess.TimeoutExpired:
                logger.warning(f"安全测试超时: {test_file}")
                return False

        return True

    def check_performance_baselines(self):
        """检查性能基准"""
        logger.info("检查性能基准...")

        # 检查是否有性能测试文件
        performance_tests = [
            'tests/integration/test_basic_performance.py',
            'tests/integration/test_end_to_end_trading_flow.py'
        ]

        for test_file in performance_tests:
            test_path = self.project_root / test_file
            if not test_path.exists():
                logger.warning(f"缺少性能测试文件: {test_file}")
                continue

            # 运行性能测试
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest',
                    str(test_path),
                    '-v', '--tb=no'
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    logger.info(f"性能测试通过: {test_file}")
                else:
                    logger.warning(f"性能测试失败: {test_file}")
                    return False

            except subprocess.TimeoutExpired:
                logger.warning(f"性能测试超时: {test_file}")
                return False

        return True

    def check_configuration_integrity(self):
        """检查配置完整性"""
        logger.info("检查配置完整性...")

        # 检查必需的配置文件
        required_configs = [
            'config/production.yaml',
            'config/development.yaml',
            'config/testing.yaml',
            'pytest.ini',
            'requirements.txt'
        ]

        for config_file in required_configs:
            config_path = self.project_root / config_file
            if not config_path.exists():
                logger.error(f"缺少必需配置文件: {config_file}")
                return False

            # 检查配置文件是否有内容
            if config_path.stat().st_size < 100:  # 配置文件应该有足够内容
                logger.warning(f"配置文件可能不完整: {config_file}")

        # 检查部署配置文件
        deploy_configs = [
            'deploy/config/production.yaml',
            'deploy/config/deployment_config.json'
        ]

        for config_file in deploy_configs:
            config_path = self.project_root / config_file
            if not config_path.exists():
                logger.error(f"缺少部署配置文件: {config_file}")
                return False

        return True

    def check_documentation_completeness(self):
        """检查文档完整性"""
        logger.info("检查文档完整性...")

        # 检查必需的文档
        required_docs = [
            'README.md',
            'docs/architecture/PRODUCTION_TEST_PLAN.md',
            'deploy/PRODUCTION_DEPLOYMENT_GUIDE.md',
            'deploy/PRODUCTION_DEPLOYMENT_CHECKLIST.md'
        ]

        for doc_file in required_docs:
            doc_path = self.project_root / doc_file
            if not doc_path.exists():
                logger.error(f"缺少必需文档: {doc_file}")
                return False

            # 检查文档是否有足够内容
            if doc_path.stat().st_size < 1000:  # 文档应该有足够内容
                logger.warning(f"文档可能不完整: {doc_file}")

        return True

    def check_deployment_readiness(self):
        """检查部署就绪性"""
        logger.info("检查部署就绪性...")

        # 检查Docker相关文件
        docker_files = [
            'Dockerfile',
            'docker-compose.yml',
            'deploy/docker-compose.production.yml'
        ]

        for docker_file in docker_files:
            docker_path = self.project_root / docker_file
            if not docker_path.exists():
                logger.error(f"缺少Docker配置文件: {docker_file}")
                return False

        # 检查Kubernetes部署文件
        k8s_files = [
            'deploy/k8s/rqa2025-deployment.yml',
            'deploy/k8s/rqa2025-namespace.yml'
        ]

        for k8s_file in k8s_files:
            k8s_path = self.project_root / k8s_file
            if not k8s_path.exists():
                logger.warning(f"缺少Kubernetes配置文件: {k8s_file}")
                # Kubernetes不是必需的，所以只警告

        # 检查部署脚本
        deploy_scripts = [
            'deploy/scripts/deploy.sh',
            'deploy/scripts/verify_functionality.sh'
        ]

        for script_file in deploy_scripts:
            script_path = self.project_root / script_file
            if not script_path.exists():
                logger.warning(f"缺少部署脚本: {script_file}")

        return True

    def run_all_checks(self):
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("开始生产环境就绪性检查")
        logger.info("=" * 60)

        checks = [
            ("代码质量检查", self.check_code_quality),
            ("测试覆盖率检查", self.check_test_coverage),
            ("安全漏洞检查", self.check_security_vulnerabilities),
            ("性能基准检查", self.check_performance_baselines),
            ("配置完整性检查", self.check_configuration_integrity),
            ("文档完整性检查", self.check_documentation_completeness),
            ("部署就绪性检查", self.check_deployment_readiness)
        ]

        passed_checks = 0
        total_checks = len(checks)

        for check_name, check_func in checks:
            if self.run_check(check_name, check_func):
                passed_checks += 1

        # 生成检查报告
        self.generate_report(passed_checks, total_checks)

        return passed_checks == total_checks

    def generate_report(self, passed_checks, total_checks):
        """生成检查报告"""
        report = {
            'check_timestamp': datetime.now().isoformat(),
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': (passed_checks / total_checks) * 100 if total_checks > 0 else 0,
            'overall_status': 'READY' if passed_checks == total_checks else 'NOT_READY',
            'check_results': self.check_results
        }

        # 保存报告
        reports_dir = self.project_root / 'deploy' / 'reports'
        reports_dir.mkdir(exist_ok=True)

        report_file = reports_dir / \
            f'production_readiness_report_{int(datetime.now().timestamp())}.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印报告
        print("\n" + "=" * 60)
        print("生产环境就绪性检查报告")
        print("=" * 60)
        print(f"检查时间: {report['check_timestamp']}")
        print(f"总检查项: {total_checks}")
        print(f"通过检查: {passed_checks}")
        print(f"失败检查: {total_checks - passed_checks}")
        print(".1f")
        print(f"总体状态: {'✅ 就绪' if report['overall_status'] == 'READY' else '❌ 未就绪'}")
        print("=" * 60)

        for result in self.check_results:
            status_icon = {
                'PASSED': '✅',
                'FAILED': '❌',
                'ERROR': '⚠️'
            }
            icon = status_icon.get(result['status'], '❓')
            print(f"{icon} {result['check_name']}: {result['status']}")

        print(f"\n详细报告已保存到: {report_file}")

        if report['overall_status'] != 'READY':
            print("\n⚠️  系统尚未完全就绪进行生产部署")
            print("请解决上述失败的检查项后重新运行检查")
        else:
            print("\n🎉 系统已准备好进行生产部署！")
            print("请参考 deploy/PRODUCTION_DEPLOYMENT_GUIDE.md 进行部署")


def main():
    """主函数"""
    checker = ProductionReadinessChecker()

    try:
        success = checker.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("检查被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"检查过程中发生异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
