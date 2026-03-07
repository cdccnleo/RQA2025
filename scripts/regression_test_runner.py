#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 回归测试执行器
Regression Test Runner

执行全面的回归测试，确保系统功能正常，性能达标
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'test_logs' / 'regression_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RegressionTestRunner:
    """回归测试执行器"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.test_logs_dir = PROJECT_ROOT / 'test_logs'
        self.regression_reports_dir = self.test_logs_dir / 'regression_reports'

        # 确保目录存在
        self.regression_reports_dir.mkdir(parents=True, exist_ok=True)

        # 回归测试配置
        self.test_suites = {
            'unit_tests': {
                'name': '单元测试回归',
                'paths': ['tests/unit/'],
                'timeout': 900,  # 15分钟
                'expected_pass_rate': 90.0
            },
            'integration_tests': {
                'name': '集成测试回归',
                'paths': ['tests/integration/'],
                'timeout': 1200,  # 20分钟
                'expected_pass_rate': 85.0
            },
            'e2e_tests': {
                'name': '端到端测试回归',
                'paths': ['tests/e2e/'],
                'timeout': 1800,  # 30分钟
                'expected_pass_rate': 80.0
            },
            'performance_tests': {
                'name': '性能测试回归',
                'paths': ['tests/performance/'],
                'timeout': 2400,  # 40分钟
                'expected_pass_rate': 75.0
            }
        }

        # 性能基准
        self.performance_baselines = {
            'test_execution_time': 300,  # 最大执行时间（秒）
            'memory_usage': 1024 * 1024 * 1024,  # 最大内存使用（1GB）
            'coverage_rate': 85.0  # 最低覆盖率
        }

    def run_test_suite(self, suite_name: str, with_coverage: bool = True) -> Dict[str, Any]:
        """运行指定的测试套件"""
        if suite_name not in self.test_suites:
            raise ValueError(f"未知的测试套件: {suite_name}")

        suite_config = self.test_suites[suite_name]
        logger.info(f"开始运行{suite_config['name']}...")

        cmd = [
            sys.executable, '-m', 'pytest',
            '--tb=short',
            '--json-report',
            f'--json-report-file=regression_{suite_name}_results.json',
            '--junitxml', f'regression_{suite_name}_results.xml',
            '--maxfail=50',
            '--durations=20',
            '-n', 'auto',  # 并行执行
            '--dist=loadscope'
        ]

        if with_coverage:
            cmd.extend([
                '--cov=src',
                '--cov-report=json:coverage.json',
                '--cov-report=term-missing',
                '--cov-report=html:test_logs/regression_reports',
                '--cov-fail-under=80'
            ])

        # 添加测试路径
        cmd.extend(suite_config['paths'])

        start_time = time.time()

        try:
            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=suite_config['timeout']
            )

            execution_time = time.time() - start_time

            # 解析结果
            test_results = self._parse_test_results(f'regression_{suite_name}_results.json')
            coverage_data = self._parse_coverage_data() if with_coverage else {}

            report = {
                'suite_name': suite_name,
                'suite_display_name': suite_config['name'],
                'timestamp': datetime.now().isoformat(),
                'exit_code': result.returncode,
                'execution_time': execution_time,
                'stdout': result.stdout[-5000:],  # 只保留最后5000字符
                'stderr': result.stderr[-5000:],  # 只保留最后5000字符
                'test_results': test_results,
                'coverage': coverage_data,
                'expected_pass_rate': suite_config['expected_pass_rate']
            }

            logger.info(f"{suite_config['name']}执行完成，退出码: {result.returncode}")
            return report

        except subprocess.TimeoutExpired:
            logger.error(f"{suite_config['name']}执行超时")
            return {
                'suite_name': suite_name,
                'error': 'timeout',
                'execution_time': suite_config['timeout'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"{suite_config['name']}执行失败: {e}")
            return {
                'suite_name': suite_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _parse_test_results(self, results_file: str) -> Dict[str, Any]:
        """解析测试结果"""
        results_path = self.project_root / results_file
        if not results_path.exists():
            return {}

        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            summary = data.get('summary', {})
            return {
                'total': summary.get('num_tests', 0),
                'passed': summary.get('passed', 0),
                'failed': summary.get('failed', 0),
                'skipped': summary.get('skipped', 0),
                'errors': summary.get('errors', 0),
                'duration': summary.get('duration', 0),
                'success_rate': (summary.get('passed', 0) / max(summary.get('num_tests', 1), 1)) * 100
            }
        except Exception as e:
            logger.warning(f"解析测试结果失败: {e}")
            return {}

    def _parse_coverage_data(self) -> Dict[str, Any]:
        """解析覆盖率数据"""
        coverage_file = self.project_root / 'coverage.json'
        if not coverage_file.exists():
            return {}

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            totals = data.get('totals', {})
            return {
                'covered_lines': totals.get('covered_lines', 0),
                'num_statements': totals.get('num_statements', 0),
                'percent_covered': totals.get('percent_covered', 0),
                'missing_lines': totals.get('missing_lines', 0),
                'excluded_lines': totals.get('excluded_lines', 0)
            }
        except Exception as e:
            logger.warning(f"解析覆盖率数据失败: {e}")
            return {}

    def analyze_regression_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析回归测试结果"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'passed',
            'total_suites': len(results),
            'passed_suites': 0,
            'failed_suites': 0,
            'issues': [],
            'performance_metrics': {},
            'recommendations': []
        }

        total_execution_time = 0
        total_tests = 0
        total_passed = 0

        for result in results:
            suite_name = result.get('suite_name', 'unknown')
            suite_display_name = result.get('suite_display_name', suite_name)

            if 'error' in result:
                analysis['overall_status'] = 'failed'
                analysis['failed_suites'] += 1
                analysis['issues'].append({
                    'type': 'execution_error',
                    'suite': suite_display_name,
                    'message': f"执行失败: {result['error']}"
                })
                continue

            # 检查测试结果
            test_results = result.get('test_results', {})
            success_rate = test_results.get('success_rate', 0)
            expected_rate = result.get('expected_pass_rate', 90.0)

            if success_rate < expected_rate:
                analysis['overall_status'] = 'failed'
                analysis['failed_suites'] += 1
                analysis['issues'].append({
                    'type': 'pass_rate_below_threshold',
                    'suite': suite_display_name,
                    'message': f"成功率 {success_rate:.1f}% 低于阈值 {expected_rate:.1f}%"
                })
            else:
                analysis['passed_suites'] += 1

            # 累积统计
            total_execution_time += result.get('execution_time', 0)
            total_tests += test_results.get('total', 0)
            total_passed += test_results.get('passed', 0)

            # 检查性能
            execution_time = result.get('execution_time', 0)
            if execution_time > self.performance_baselines['test_execution_time']:
                analysis['issues'].append({
                    'type': 'performance_issue',
                    'suite': suite_display_name,
                    'message': f"执行时间 {execution_time:.1f}s 超过基准 {self.performance_baselines['test_execution_time']}s"
                })

        # 计算总体指标
        if total_tests > 0:
            overall_success_rate = (total_passed / total_tests) * 100
            analysis['performance_metrics'] = {
                'total_execution_time': round(total_execution_time, 2),
                'total_tests': total_tests,
                'total_passed': total_passed,
                'overall_success_rate': round(overall_success_rate, 2),
                'average_execution_time': round(total_execution_time / len(results), 2) if results else 0
            }

            # 检查总体成功率
            if overall_success_rate < 85.0:
                analysis['overall_status'] = 'failed'
                analysis['issues'].append({
                    'type': 'overall_pass_rate_low',
                    'message': f"总体成功率 {overall_success_rate:.1f}% 过低"
                })

        # 生成建议
        if analysis['issues']:
            analysis['recommendations'].extend([
                "修复发现的测试失败",
                "优化测试执行性能",
                "检查测试环境稳定性"
            ])
        else:
            analysis['recommendations'].append("回归测试全部通过，系统运行正常")

        return analysis

    def generate_regression_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """生成回归测试报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.regression_reports_dir / f'regression_test_report_{timestamp}.md'

        content = f"""# RQA2025 回归测试报告
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 总体状态

**测试结果**: {'✅ 通过' if analysis['overall_status'] == 'passed' else '❌ 失败'}

### 测试套件执行概览
| 测试套件 | 状态 | 执行时间 | 成功率 |
|---------|------|----------|-------|
"""

        for result in results:
            suite_display_name = result.get('suite_display_name', result.get('suite_name', 'unknown'))
            status = "❌" if 'error' in result else "✅"
            execution_time = result.get('execution_time', 0)
            success_rate = result.get('test_results', {}).get('success_rate', 0)

            content += f"| {suite_display_name} | {status} | {execution_time:.1f}s | {success_rate:.1f}% |\n"

        content += f"""
## 📈 性能指标

- **总执行时间**: {analysis['performance_metrics'].get('total_execution_time', 0):.1f}秒
- **总测试数**: {analysis['performance_metrics'].get('total_tests', 0)}
- **通过测试**: {analysis['performance_metrics'].get('total_passed', 0)}
- **总体成功率**: {analysis['performance_metrics'].get('overall_success_rate', 0):.1f}%
- **平均执行时间**: {analysis['performance_metrics'].get('average_execution_time', 0):.1f}秒/套件

## ⚠️ 发现的问题

"""

        if analysis['issues']:
            for issue in analysis['issues']:
                issue_type = issue.get('type', 'unknown')
                suite = issue.get('suite', '')
                message = issue.get('message', '')

                content += f"- **{issue_type}**"
                if suite:
                    content += f" ({suite})"
                content += f": {message}\n"
        else:
            content += "暂无严重问题\n"

        content += "\n## 💡 改进建议\n\n"

        for rec in analysis.get('recommendations', []):
            content += f"- {rec}\n"

        content += f"""
## 📋 测试套件详情

"""

        for result in results:
            suite_display_name = result.get('suite_display_name', result.get('suite_name', 'unknown'))
            content += f"### {suite_display_name}\n"

            if 'error' in result:
                content += f"**状态**: ❌ 执行失败 - {result['error']}\n"
            else:
                test_results = result.get('test_results', {})
                content += f"""**状态**: ✅ 执行成功
**测试总数**: {test_results.get('total', 0)}
**通过**: {test_results.get('passed', 0)}
**失败**: {test_results.get('failed', 0)}
**跳过**: {test_results.get('skipped', 0)}
**成功率**: {test_results.get('success_rate', 0):.1f}%
**执行时间**: {result.get('execution_time', 0):.1f}秒
"""

            content += "\n"

        content += "---\n*此报告由回归测试执行器自动生成*"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"回归测试报告已生成: {report_file}")
            return str(report_file)
        except Exception as e:
            logger.error(f"生成回归测试报告失败: {e}")
            return ""

    def run_full_regression(self, suites: Optional[List[str]] = None,
                           generate_report: bool = True) -> Dict[str, Any]:
        """运行完整的回归测试"""
        if suites is None:
            suites = list(self.test_suites.keys())

        logger.info(f"开始运行完整回归测试，包含套件: {', '.join(suites)}")

        results = []

        for suite_name in suites:
            try:
                result = self.run_test_suite(suite_name, with_coverage=True)
                results.append(result)
                time.sleep(2)  # 短暂暂停，避免资源冲突
            except Exception as e:
                logger.error(f"运行{suite_name}失败: {e}")
                results.append({
                    'suite_name': suite_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # 分析结果
        analysis = self.analyze_regression_results(results)

        # 生成报告
        if generate_report:
            report_path = self.generate_regression_report(results, analysis)
            analysis['report_path'] = report_path

        logger.info(f"完整回归测试完成: {analysis['overall_status']}")

        return {
            'results': results,
            'analysis': analysis
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 回归测试执行器')
    parser.add_argument('--suites', nargs='+', help='指定要运行的测试套件')
    parser.add_argument('--no-report', action='store_true', help='不生成报告')
    parser.add_argument('--json', action='store_true', help='输出JSON格式结果')

    args = parser.parse_args()

    runner = RegressionTestRunner()

    try:
        result = runner.run_full_regression(
            suites=args.suites,
            generate_report=not args.no_report
        )

        analysis = result['analysis']

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n🎯 回归测试执行结果:")
            print(f"   总体状态: {'✅ 通过' if analysis['overall_status'] == 'passed' else '❌ 失败'}")
            print(f"   测试套件: {analysis['passed_suites']}/{analysis['total_suites']} 通过")

            if analysis['issues']:
                print("   发现问题:")
                for issue in analysis['issues']:
                    print(f"   - {issue.get('message', '未知问题')}")

            if analysis.get('report_path'):
                print(f"   📄 详细报告: {analysis['report_path']}")

            if analysis['overall_status'] == 'passed':
                sys.exit(0)
            else:
                sys.exit(1)

    except Exception as e:
        logger.error(f"回归测试执行异常: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
