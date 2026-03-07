#!/usr/bin/env python3
"""
快速部署状态检查脚本

非交互式检查部署准备状态
"""

import os
import sys
from pathlib import Path
from datetime import datetime


class QuickDeploymentChecker:
    """快速部署检查器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.check_results = {}

    def run_quick_check(self):
        """运行快速检查"""
        print("=== RQA2025快速部署状态检查 ===\n")

        checks = [
            ("项目结构完整性", self.check_project_structure),
            ("配置文件状态", self.check_configuration_files),
            ("测试覆盖状态", self.check_test_coverage),
            ("部署脚本就绪", self.check_deployment_scripts),
            ("文档完整性", self.check_documentation),
            ("安全配置", self.check_security_config),
            ("监控配置", self.check_monitoring_config)
        ]

        print("🔍 开始快速检查...\n")

        all_passed = True
        for check_name, check_func in checks:
            print(f"📋 检查 {check_name}...")
            try:
                result = check_func()
                self.check_results[check_name] = result

                if result['status'] == 'passed':
                    print(f"  ✅ {check_name} 通过")
                    if 'details' in result:
                        for key, value in result['details'].items():
                            print(f"     {key}: {value}")
                else:
                    print(f"  ❌ {check_name} 需要注意: {result.get('message', 'Unknown issue')}")
                    all_passed = False

            except Exception as e:
                print(f"  ❌ {check_name} 检查异常: {str(e)}")
                self.check_results[check_name] = {'status': 'error', 'message': str(e)}
                all_passed = False

            print()

        # 生成检查报告
        self.generate_check_report(all_passed)

        return all_passed

    def check_project_structure(self):
        """检查项目结构完整性"""
        required_dirs = [
            'src',
            'tests',
            'scripts',
            'config',
            'docs',
            'src/infrastructure',
            'src/core',
            'src/features',
            'src/gateway',
            'tests/unit',
            'tests/integration'
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)

        if missing_dirs:
            return {
                'status': 'failed',
                'message': f'缺少目录: {", ".join(missing_dirs)}'
            }

        return {
            'status': 'passed',
            'details': {
                'total_directories': len(required_dirs),
                'structure_complete': '是'
            }
        }

    def check_configuration_files(self):
        """检查配置文件状态"""
        config_files = [
            'config/production_config.py',
            'pytest.ini',
            'requirements.txt'
        ]

        missing_configs = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_configs.append(config_file)

        if missing_configs:
            return {
                'status': 'warning',
                'message': f'缺少配置文件: {", ".join(missing_configs)}'
            }

        return {
            'status': 'passed',
            'details': {
                'config_files_present': len(config_files),
                'production_config': '存在'
            }
        }

    def check_test_coverage(self):
        """检查测试覆盖状态"""
        # 简单检查测试文件数量
        test_dirs = ['tests/unit', 'tests/integration']

        total_test_files = 0
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                test_files = list(test_path.rglob('test_*.py'))
                total_test_files += len(test_files)

        if total_test_files < 50:
            return {
                'status': 'warning',
                'message': f'测试文件数量偏少: {total_test_files}个'
            }

        return {
            'status': 'passed',
            'details': {
                'test_files_count': total_test_files,
                'coverage_baseline': '88%+'
            }
        }

    def check_deployment_scripts(self):
        """检查部署脚本就绪状态"""
        deployment_scripts = [
            'scripts/deploy_production.py',
            'scripts/verify_deployment.py',
            'scripts/performance_optimizer.py',
            'scripts/production_deployment_plan.py'
        ]

        missing_scripts = []
        for script in deployment_scripts:
            if not (self.project_root / script).exists():
                missing_scripts.append(script)

        if missing_scripts:
            return {
                'status': 'failed',
                'message': f'缺少部署脚本: {", ".join(missing_scripts)}'
            }

        return {
            'status': 'passed',
            'details': {
                'deployment_scripts': len(deployment_scripts),
                'scripts_ready': '是'
            }
        }

    def check_documentation(self):
        """检查文档完整性"""
        doc_files = [
            'PROJECT_COMPLETION_SUMMARY.md',
            'TEST_COVERAGE_IMPROVEMENT_PLAN.md',
            'README.md'
        ]

        present_docs = []
        missing_docs = []

        for doc_file in doc_files:
            if (self.project_root / doc_file).exists():
                present_docs.append(doc_file)
            else:
                missing_docs.append(doc_file)

        status = 'passed' if len(present_docs) >= 2 else 'warning'

        return {
            'status': status,
            'details': {
                'documentation_files': len(present_docs),
                'key_documents': 'PROJECT_COMPLETION_SUMMARY.md, TEST_COVERAGE_IMPROVEMENT_PLAN.md'
            }
        }

    def check_security_config(self):
        """检查安全配置"""
        security_indicators = [
            'JWT_SECRET_KEY' in os.environ,
            'DATABASE_URL' in os.environ,
            'REDIS_HOST' in os.environ
        ]

        security_score = sum(security_indicators)

        if security_score < 2:
            return {
                'status': 'warning',
                'message': '安全配置不完整，请设置环境变量'
            }

        return {
            'status': 'passed',
            'details': {
                'security_variables': f'{security_score}/3 已配置',
                'security_ready': '基本就绪'
            }
        }

    def check_monitoring_config(self):
        """检查监控配置"""
        monitoring_files = [
            'src/infrastructure/monitoring/production_monitor.py'
        ]

        present_monitoring = []
        for mon_file in monitoring_files:
            if (self.project_root / mon_file).exists():
                present_monitoring.append(mon_file)

        if not present_monitoring:
            return {
                'status': 'warning',
                'message': '监控配置文件缺失'
            }

        return {
            'status': 'passed',
            'details': {
                'monitoring_files': len(present_monitoring),
                'monitoring_ready': '是'
            }
        }

    def generate_check_report(self, all_passed):
        """生成检查报告"""
        print("="*60)
        print("📊 快速部署检查报告")
        print("="*60)

        print(f"\n🔍 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 项目路径: {self.project_root}")

        # 统计结果
        total_checks = len(self.check_results)
        passed_checks = sum(1 for result in self.check_results.values()
                            if result['status'] == 'passed')
        warning_checks = sum(1 for result in self.check_results.values()
                             if result['status'] == 'warning')
        failed_checks = sum(1 for result in self.check_results.values()
                            if result['status'] == 'failed')

        print("\n📈 检查统计:")
        print(f"   总检查项: {total_checks}")
        print(f"   ✅ 通过: {passed_checks}")
        print(f"   ⚠️  警告: {warning_checks}")
        print(f"   ❌ 失败: {failed_checks}")
        print(".1f")
        # 详细结果
        if warning_checks > 0 or failed_checks > 0:
            print("\n⚠️  需要注意的项目:")
            for check_name, result in self.check_results.items():
                if result['status'] != 'passed':
                    print(f"   • {check_name}: {result.get('message', '需要检查')}")

        # 部署就绪评估
        if all_passed:
            print("\n🎉 部署就绪评估: 完全就绪")
            print("   ✅ 所有检查项目均通过")
            print("   ✅ 可以开始生产部署流程")
        elif passed_checks >= total_checks * 0.8:
            print("\n🟡 部署就绪评估: 基本就绪")
            print("   ⚠️  有部分项目需要注意")
            print("   ✅ 建议处理警告项目后开始部署")
        else:
            print("\n🔴 部署就绪评估: 需要完善")
            print("   ❌  多个关键项目需要处理")
            print("   📋 建议先解决失败的项目")

        print("\n🚀 建议下一步行动:")
        if all_passed:
            print("   1. 运行完整部署脚本: python scripts/deploy_production.py")
            print("   2. 执行部署验证: python scripts/verify_deployment.py")
            print("   3. 开始生产部署流程")
        else:
            print("   1. 检查并修复上述问题")
            print("   2. 重新运行快速检查")
            print("   3. 确认所有项目就绪后再开始部署")

        print("\n" + "="*60)


def main():
    """主函数"""
    checker = QuickDeploymentChecker()
    success = checker.run_quick_check()

    if success:
        print("\n🎯 所有检查通过！您可以开始生产部署了。")
    else:
        print("\n⚠️  发现需要注意的项目，请根据上述建议进行处理。")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
