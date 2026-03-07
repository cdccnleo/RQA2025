#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试状态验证脚本
用于检查当前测试环境和覆盖率状态，为下一步测试计划提供依据
"""

import subprocess
import json
from pathlib import Path
from typing import Dict
import time
from datetime import datetime


class TestStatusValidator:
    """测试状态验证器"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"

        # 各层关键模块
        self.layer_modules = {
            'infrastructure': {
                'priority': 'critical',
                'modules': [
                    'config/config_manager.py',
                    'm_logging/logger.py',
                    'cache/cache_manager.py',
                    'database/database_manager.py',
                    'monitoring/system_monitor.py',
                    'error/error_handler.py',
                    'error/circuit_breaker.py',
                    'deployment/service_launcher.py'
                ]
            },
            'data': {
                'priority': 'high',
                'modules': [
                    'data_loader.py',
                    'data_manager.py',
                    'validator.py',
                    'base_loader.py',
                    'parallel_loader.py'
                ]
            },
            'features': {
                'priority': 'medium',
                'modules': [
                    'feature_engineer.py',
                    'feature_manager.py',
                    'feature_engine.py',
                    'signal_generator.py',
                    'sentiment_analyzer.py'
                ]
            },
            'models': {
                'priority': 'critical',
                'modules': [
                    'model_manager.py',
                    'inference_engine.py',
                    'model_optimizer.py',
                    'base_model.py'
                ]
            },
            'trading': {
                'priority': 'high',
                'modules': [
                    'trading_engine.py',
                    'execution_engine.py',
                    'live_trading.py',
                    'order_manager.py',
                    'risk_manager.py'
                ]
            },
            'backtest': {
                'priority': 'medium',
                'modules': [
                    'backtest_engine.py',
                    'parameter_optimizer.py',
                    'performance_analyzer.py',
                    'strategy_evaluator.py'
                ]
            }
        }

    def check_environment(self) -> Dict:
        """检查测试环境状态"""
        print("🔍 检查测试环境状态...")

        status = {
            'conda_env': False,
            'python_version': None,
            'pytest_available': False,
            'coverage_available': False,
            'key_dependencies': {},
            'test_directories': {},
            'issues': []
        }

        try:
            # 检查conda环境
            result = subprocess.run(['conda', 'info', '--json'],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                conda_info = json.loads(result.stdout)
                active_env = conda_info.get('active_prefix_name', '')
                status['conda_env'] = active_env == 'test'
                print(f"✅ Conda环境: {active_env}")
            else:
                status['issues'].append("Conda环境检查失败")
                print("❌ Conda环境检查失败")
        except Exception as e:
            status['issues'].append(f"Conda环境检查异常: {str(e)}")
            print(f"❌ Conda环境检查异常: {str(e)}")

        try:
            # 检查Python版本
            result = subprocess.run(['python', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                status['python_version'] = result.stdout.strip()
                print(f"✅ Python版本: {status['python_version']}")
        except Exception as e:
            status['issues'].append(f"Python版本检查异常: {str(e)}")
            print(f"❌ Python版本检查异常: {str(e)}")

        try:
            # 检查pytest
            result = subprocess.run(['python', '-m', 'pytest', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                status['pytest_available'] = True
                print("✅ Pytest可用")
            else:
                status['issues'].append("Pytest不可用")
                print("❌ Pytest不可用")
        except Exception as e:
            status['issues'].append(f"Pytest检查异常: {str(e)}")
            print(f"❌ Pytest检查异常: {str(e)}")

        try:
            # 检查coverage
            result = subprocess.run(['python', '-m', 'coverage', '--version'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                status['coverage_available'] = True
                print("✅ Coverage可用")
            else:
                status['issues'].append("Coverage不可用")
                print("❌ Coverage不可用")
        except Exception as e:
            status['issues'].append(f"Coverage检查异常: {str(e)}")
            print(f"❌ Coverage检查异常: {str(e)}")

        # 检查关键依赖
        key_deps = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pytest', 'coverage']
        for dep in key_deps:
            try:
                result = subprocess.run(['python', '-c', f'import {dep}'],
                                        capture_output=True, text=True, timeout=10)
                status['key_dependencies'][dep] = result.returncode == 0
                if result.returncode == 0:
                    print(f"✅ {dep} 可用")
                else:
                    print(f"❌ {dep} 不可用")
                    status['issues'].append(f"{dep} 不可用")
            except Exception as e:
                status['key_dependencies'][dep] = False
                status['issues'].append(f"{dep} 检查异常: {str(e)}")
                print(f"❌ {dep} 检查异常: {str(e)}")

        # 检查测试目录
        test_dirs = ['tests/unit/infrastructure', 'tests/unit/data', 'tests/unit/features',
                     'tests/unit/models', 'tests/unit/trading', 'tests/unit/backtest']
        for test_dir in test_dirs:
            dir_path = self.project_root / test_dir
            status['test_directories'][test_dir] = dir_path.exists()
            if dir_path.exists():
                test_files = list(dir_path.glob('test_*.py'))
                status['test_directories'][f"{test_dir}_count"] = len(test_files)
                print(f"✅ {test_dir}: {len(test_files)} 个测试文件")
            else:
                print(f"❌ {test_dir}: 目录不存在")

        return status

    def check_module_coverage(self, layer: str) -> Dict:
        """检查指定层的模块覆盖率"""
        print(f"🔍 检查 {layer} 层模块覆盖率...")

        coverage_status = {
            'layer': layer,
            'modules': {},
            'total_files': 0,
            'covered_files': 0,
            'issues': []
        }

        if layer not in self.layer_modules:
            coverage_status['issues'].append(f"未知层: {layer}")
            return coverage_status

        layer_config = self.layer_modules[layer]
        modules = layer_config['modules']

        for module in modules:
            module_path = self.src_path / layer / module
            test_path = self.tests_path / 'unit' / layer / \
                f"test_{module.replace('/', '_').replace('.py', '.py')}"

            module_status = {
                'exists': module_path.exists(),
                'has_test': test_path.exists(),
                'test_files': []
            }

            if test_path.exists():
                # 查找相关的测试文件
                test_dir = test_path.parent
                if test_dir.exists():
                    test_files = list(test_dir.glob(
                        f"test_{module.split('/')[-1].replace('.py', '')}*.py"))
                    module_status['test_files'] = [str(f) for f in test_files]
                    module_status['test_count'] = len(test_files)
                else:
                    module_status['test_count'] = 0
            else:
                module_status['test_count'] = 0

            coverage_status['modules'][module] = module_status

            if module_path.exists():
                coverage_status['total_files'] += 1
                if module_status['has_test'] or module_status['test_count'] > 0:
                    coverage_status['covered_files'] += 1

            # 输出状态
            if module_path.exists():
                if module_status['has_test'] or module_status['test_count'] > 0:
                    print(f"✅ {module}: {module_status['test_count']} 个测试文件")
                else:
                    print(f"❌ {module}: 无测试文件")
                    coverage_status['issues'].append(f"{module} 无测试文件")
            else:
                print(f"⚠️ {module}: 模块文件不存在")
                coverage_status['issues'].append(f"{module} 模块文件不存在")

        # 计算覆盖率
        if coverage_status['total_files'] > 0:
            coverage_status['coverage_percentage'] = (
                coverage_status['covered_files'] / coverage_status['total_files']) * 100
        else:
            coverage_status['coverage_percentage'] = 0

        print(f"📊 {layer} 层覆盖率: {coverage_status['coverage_percentage']:.1f}%")

        return coverage_status

    def run_simple_test(self, test_file: str) -> Dict:
        """运行单个测试文件"""
        print(f"🧪 运行测试: {test_file}")

        test_result = {
            'file': test_file,
            'success': False,
            'duration': 0,
            'output': '',
            'error': '',
            'issues': []
        }

        if not Path(test_file).exists():
            test_result['issues'].append("测试文件不存在")
            print(f"❌ 测试文件不存在: {test_file}")
            return test_result

        start_time = time.time()

        try:
            # 使用run_tests.py脚本运行测试
            cmd = [
                'python', 'scripts/testing/run_tests.py',
                '--test-file', test_file,
                '--skip-coverage',
                '--pytest-args', '-v', '--tb=short'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=self.project_root
            )

            test_result['duration'] = time.time() - start_time
            test_result['output'] = result.stdout
            test_result['error'] = result.stderr
            test_result['success'] = result.returncode == 0

            if result.returncode == 0:
                print(f"✅ 测试成功: {test_file} ({test_result['duration']:.1f}s)")
            else:
                print(f"❌ 测试失败: {test_file} ({test_result['duration']:.1f}s)")
                test_result['issues'].append(f"测试执行失败，退出码: {result.returncode}")

        except subprocess.TimeoutExpired:
            test_result['duration'] = time.time() - start_time
            test_result['issues'].append("测试执行超时")
            print(f"⏰ 测试超时: {test_file}")
        except Exception as e:
            test_result['duration'] = time.time() - start_time
            test_result['issues'].append(f"测试执行异常: {str(e)}")
            print(f"❌ 测试异常: {test_file} - {str(e)}")

        return test_result

    def generate_report(self) -> Dict:
        """生成完整的测试状态报告"""
        print("📋 生成测试状态报告...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.check_environment(),
            'layer_coverage': {},
            'test_results': [],
            'summary': {},
            'recommendations': []
        }

        # 检查各层覆盖率
        for layer in self.layer_modules.keys():
            report['layer_coverage'][layer] = self.check_module_coverage(layer)

        # 运行一些关键测试
        key_tests = [
            'tests/unit/infrastructure/test_config_manager.py',
            'tests/unit/infrastructure/test_error_handler.py',
            'tests/unit/infrastructure/test_circuit_breaker.py'
        ]

        for test_file in key_tests:
            if Path(test_file).exists():
                result = self.run_simple_test(test_file)
                report['test_results'].append(result)

        # 生成摘要
        total_modules = sum(len(self.layer_modules[layer]['modules'])
                            for layer in self.layer_modules)
        covered_modules = sum(report['layer_coverage'][layer]['covered_files']
                              for layer in report['layer_coverage'])

        report['summary'] = {
            'total_modules': total_modules,
            'covered_modules': covered_modules,
            'overall_coverage': (covered_modules / total_modules * 100) if total_modules > 0 else 0,
            'environment_issues': len(report['environment']['issues']),
            'test_success_rate': len([r for r in report['test_results'] if r['success']]) / len(report['test_results']) if report['test_results'] else 0
        }

        # 生成建议
        if report['environment']['issues']:
            report['recommendations'].append("修复测试环境问题")

        for layer, coverage in report['layer_coverage'].items():
            if coverage['coverage_percentage'] < 50:
                report['recommendations'].append(f"提升 {layer} 层测试覆盖率")

        if report['summary']['test_success_rate'] < 0.8:
            report['recommendations'].append("修复失败的测试用例")

        return report

    def save_report(self, report: Dict, filename: str = None):
        """保存报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/testing/test_status_report_{timestamp}.json"

        report_path = self.project_root / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 报告已保存: {report_path}")
        return report_path

    def print_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "="*60)
        print("📊 测试状态报告摘要")
        print("="*60)

        # 环境状态
        env = report['environment']
        print(f"🔧 环境状态:")
        print(f"  - Conda环境: {'✅' if env['conda_env'] else '❌'}")
        print(f"  - Python版本: {env['python_version'] or '未知'}")
        print(f"  - Pytest: {'✅' if env['pytest_available'] else '❌'}")
        print(f"  - Coverage: {'✅' if env['coverage_available'] else '❌'}")
        print(f"  - 环境问题: {len(env['issues'])} 个")

        # 各层覆盖率
        print(f"\n📈 各层覆盖率:")
        for layer, coverage in report['layer_coverage'].items():
            print(f"  - {layer}: {coverage['coverage_percentage']:.1f}% "
                  f"({coverage['covered_files']}/{coverage['total_files']})")

        # 测试结果
        test_results = report['test_results']
        if test_results:
            success_count = len([r for r in test_results if r['success']])
            print(f"\n🧪 测试执行结果:")
            print(f"  - 成功: {success_count}/{len(test_results)}")
            print(f"  - 成功率: {report['summary']['test_success_rate']*100:.1f}%")

        # 总体摘要
        summary = report['summary']
        print(f"\n📋 总体摘要:")
        print(f"  - 总模块数: {summary['total_modules']}")
        print(f"  - 已覆盖模块: {summary['covered_modules']}")
        print(f"  - 整体覆盖率: {summary['overall_coverage']:.1f}%")

        # 建议
        if report['recommendations']:
            print(f"\n💡 建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        print("="*60)


def main():
    """主函数"""
    print("🚀 开始测试状态验证...")

    validator = TestStatusValidator()

    # 生成报告
    report = validator.generate_report()

    # 保存报告
    report_path = validator.save_report(report)

    # 打印摘要
    validator.print_summary(report)

    print(f"\n✅ 测试状态验证完成!")
    print(f"📄 详细报告: {report_path}")


if __name__ == "__main__":
    main()
