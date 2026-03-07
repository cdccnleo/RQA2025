#!/usr/bin/env python3
"""
RQA2025 CI检查自动化脚本

执行完整的CI检查流程，包括：
1. 代码质量检查
2. 单元测试
3. 集成测试
4. 性能基准测试
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any


class CICheckRunner:
    """CI检查运行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {}

    def run_all_checks(self) -> Dict[str, Any]:
        """运行所有CI检查"""
        print('🚀 RQA2025 CI检查启动')
        print('=' * 50)

        start_time = time.time()

        # 1. 代码质量检查
        print('📏 1. 代码质量检查...')
        self.results['code_quality'] = self._run_code_quality_checks()

        # 2. 单元测试
        print('🧪 2. 单元测试...')
        self.results['unit_tests'] = self._run_unit_tests()

        # 3. 集成测试
        print('🔗 3. 集成测试...')
        self.results['integration_tests'] = self._run_integration_tests()

        # 4. 系统集成验证
        print('🔄 4. 系统集成验证...')
        self.results['system_integration'] = self._run_system_integration()

        # 5. 性能基准测试
        print('⚡ 5. 性能基准测试...')
        self.results['performance'] = self._run_performance_tests()

        end_time = time.time()
        total_time = end_time - start_time

        # 生成总结报告
        summary = self._generate_summary_report(total_time)

        print('✅ CI检查完成！')
        print(f'总耗时: {total_time:.1f}秒')
        return summary

    def _run_code_quality_checks(self) -> Dict[str, Any]:
        """运行代码质量检查"""
        checks = {}

        # Black格式检查
        try:
            result = subprocess.run([
                sys.executable, '-m', 'black', '--check', '--diff', 'src/', 'tests/', 'scripts/'
            ], capture_output=True, text=True, cwd=self.project_root)
            checks['black'] = {
                'passed': result.returncode == 0,
                'output': result.stdout + result.stderr
            }
        except Exception as e:
            checks['black'] = {'passed': False, 'error': str(e)}

        # isort导入排序检查
        try:
            result = subprocess.run([
                sys.executable, '-m', 'isort', '--check-only', '--diff', 'src/', 'tests/', 'scripts/'
            ], capture_output=True, text=True, cwd=self.project_root)
            checks['isort'] = {
                'passed': result.returncode == 0,
                'output': result.stdout + result.stderr
            }
        except Exception as e:
            checks['isort'] = {'passed': False, 'error': str(e)}

        # flake8代码质量检查
        try:
            result = subprocess.run([
                sys.executable, '-m', 'flake8', 'src/', 'tests/', 'scripts/',
                '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'
            ], capture_output=True, text=True, cwd=self.project_root)
            checks['flake8'] = {
                'passed': result.returncode == 0,
                'output': result.stdout + result.stderr
            }
        except Exception as e:
            checks['flake8'] = {'passed': False, 'error': str(e)}

        return checks

    def _run_unit_tests(self) -> Dict[str, Any]:
        """运行单元测试"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/unit/',
                '-v', '--tb=short', '--cov=src/', '--cov-report=term-missing',
                '--cov-fail-under=70'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            return {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Unit tests timed out'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _run_integration_tests(self) -> Dict[str, Any]:
        """运行集成测试"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/integration/',
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=600)

            return {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Integration tests timed out'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _run_system_integration(self) -> Dict[str, Any]:
        """运行系统集成验证"""
        try:
            result = subprocess.run([
                sys.executable, 'phase29_system_integration_test_simplified.py'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            return {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'System integration test timed out'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _run_performance_tests(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        try:
            result = subprocess.run([
                sys.executable, 'phase29_2_end_to_end_performance_test.py'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=300)

            return {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Performance tests timed out'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _generate_summary_report(self, total_time: float) -> Dict[str, Any]:
        """生成总结报告"""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time': round(total_time, 2),
            'checks': {}
        }

        # 计算各检查的通过情况
        for check_name, check_results in self.results.items():
            if check_name == 'code_quality':
                # 代码质量检查特殊处理
                passed_checks = sum(1 for c in check_results.values() if c.get('passed', False))
                total_checks = len(check_results)
                summary['checks'][check_name] = {
                    'passed': passed_checks,
                    'total': total_checks,
                    'pass_rate': f'{passed_checks}/{total_checks}',
                    'status': 'PASSED' if passed_checks == total_checks else 'FAILED'
                }
            else:
                # 其他测试
                summary['checks'][check_name] = {
                    'passed': check_results.get('passed', False),
                    'status': 'PASSED' if check_results.get('passed', False) else 'FAILED'
                }

        # 计算总体状态
        all_passed = all(
            check['status'] == 'PASSED'
            for check in summary['checks'].values()
        )
        summary['overall_status'] = 'SUCCESS' if all_passed else 'FAILURE'

        return summary


def main():
    """主函数"""
    runner = CICheckRunner()
    report = runner.run_all_checks()

    # 输出结果
    print('\n📊 CI检查结果汇总')
    print('=' * 50)
    print(f"总耗时: {report['total_time']}秒")
    print(f"总体状态: {report['overall_status']}")

    print('\n详细结果:')
    for check_name, check_result in report['checks'].items():
        status_icon = '✅' if check_result['status'] == 'PASSED' else '❌'
        check_display_name = {
            'code_quality': '代码质量',
            'unit_tests': '单元测试',
            'integration_tests': '集成测试',
            'system_integration': '系统集成',
            'performance': '性能测试'
        }.get(check_name, check_name)

        if 'pass_rate' in check_result:
            print(f"  {status_icon} {check_display_name}: {check_result['pass_rate']}")
        else:
            print(f"  {status_icon} {check_display_name}: {check_result['status']}")

    # 保存详细报告
    import json
    with open('ci_check_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细报告已保存到: ci_check_report.json")

    # 返回退出码
    return 0 if report['overall_status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    sys.exit(main())
