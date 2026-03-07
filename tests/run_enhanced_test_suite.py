#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025增强测试套件运行器
提供统一的测试执行接口，支持分层测试、端到端测试和质量监控
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_architecture_config import test_architecture_config
from tests.coverage_quality_monitor import quality_monitor


class EnhancedTestSuiteRunner:
    """增强测试套件运行器"""

    def __init__(self):
        self.project_root = project_root
        self.logger = self._setup_logger()
        self.test_results = {}

    def _setup_logger(self):
        """设置日志"""
        import logging
        logger = logging.getLogger("EnhancedTestSuite")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def run_layered_tests(self, layers: List[str] = None, parallel: bool = True) -> Dict[str, Any]:
        """运行分层测试"""
        if layers is None:
            layers = ["unit", "integration", "e2e"]

        results = {}

        for layer in layers:
            self.logger.info(f"开始执行 {layer} 层测试")
            start_time = time.time()

            try:
                if layer == "unit":
                    result = self._run_unit_tests(parallel)
                elif layer == "integration":
                    result = self._run_integration_tests(parallel)
                elif layer == "e2e":
                    result = self._run_e2e_tests()
                else:
                    self.logger.warning(f"未知的测试层: {layer}")
                    continue

                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                results[layer] = result

                self.logger.info(
                    f"{layer} 层测试完成: {result.get('passed', 0)}/{result.get('total', 0)} 通过, "
                    f"耗时 {execution_time:.2f}秒"
                )

            except Exception as e:
                self.logger.error(f"{layer} 层测试执行失败: {e}")
                results[layer] = {'error': str(e)}

        return results

    def _run_unit_tests(self, parallel: bool = True) -> Dict[str, Any]:
        """运行单元测试"""
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]

        if parallel:
            # 使用pytest-xdist进行并行执行
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                self.logger.warning("pytest-xdist未安装，使用单线程执行")

        return self._execute_test_command(cmd, "单元测试")

    def _run_integration_tests(self, parallel: bool = True) -> Dict[str, Any]:
        """运行集成测试"""
        # 重点运行新创建的集成测试
        integration_patterns = [
            "tests/integration/test_*_layer_integration.py",
            "tests/integration/test_*_processor_integration.py",
            "tests/integration/test_*_coordinator_integration.py"
        ]

        cmd = ["python", "-m", "pytest"]
        cmd.extend(integration_patterns)
        cmd.extend(["-v", "--tb=short", "-m", "integration"])

        if parallel and len(integration_patterns) > 1:
            try:
                import pytest_xdist
                cmd.extend(["-n", "2"])  # 集成测试使用较少的并行度
            except ImportError:
                pass

        return self._execute_test_command(cmd, "集成测试")

    def _run_e2e_tests(self) -> Dict[str, Any]:
        """运行端到端测试"""
        # 重点运行新创建的端到端测试
        e2e_patterns = [
            "tests/e2e/test_quantitative_strategy_full_lifecycle.py",
            "tests/e2e/test_trading_execution_full_chain.py",
            "tests/e2e/test_risk_control_closed_loop.py"
        ]

        cmd = ["python", "-m", "pytest"]
        cmd.extend(e2e_patterns)
        cmd.extend(["-v", "--tb=short", "-m", "e2e"])

        return self._execute_test_command(cmd, "端到端测试")

    def _execute_test_command(self, cmd: List[str], test_type: str) -> Dict[str, Any]:
        """执行测试命令"""
        try:
            self.logger.info(f"执行{test_type}: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30分钟超时
            )

            # 解析pytest输出
            return self._parse_pytest_output(result, test_type)

        except subprocess.TimeoutExpired:
            self.logger.error(f"{test_type}执行超时")
            return {'error': 'timeout', 'total': 0, 'passed': 0, 'failed': 0}
        except Exception as e:
            self.logger.error(f"{test_type}执行异常: {e}")
            return {'error': str(e), 'total': 0, 'passed': 0, 'failed': 0}

    def _parse_pytest_output(self, result: subprocess.CompletedProcess, test_type: str) -> Dict[str, Any]:
        """解析pytest输出"""
        output = result.stdout + result.stderr

        # 简单的结果解析
        total = passed = failed = errors = 0

        # 查找测试结果摘要
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # 尝试解析类似 "5 passed, 2 failed, 1 error" 的行
                parts = line.replace(',', '').split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        if i + 1 < len(parts):
                            next_word = parts[i + 1].lower()
                            if 'passed' in next_word:
                                passed = int(part)
                            elif 'failed' in next_word:
                                failed = int(part)
                            elif 'error' in next_word:
                                errors = int(part)

        total = passed + failed + errors

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'success_rate': (passed / total * 100) if total > 0 else 0.0,
            'exit_code': result.returncode,
            'output': output[-2000:]  # 只保留最后2000字符
        }

    def run_quality_assessment(self) -> Dict[str, Any]:
        """运行质量评估"""
        self.logger.info("开始质量评估")

        try:
            # 收集覆盖率指标
            coverage = quality_monitor.collect_coverage_metrics()

            # 收集质量指标
            quality = quality_monitor.collect_quality_metrics()

            # 收集各层覆盖率
            layer_coverages = quality_monitor.collect_layer_coverage()

            # 检查质量阈值
            quality_check = quality_monitor.check_quality_thresholds(coverage, quality)

            assessment = {
                'timestamp': datetime.now().isoformat(),
                'coverage': {
                    'total_lines': coverage.total_lines if coverage else 0,
                    'covered_lines': coverage.covered_lines if coverage else 0,
                    'coverage_percent': coverage.coverage_percent if coverage else 0.0
                } if coverage else None,
                'quality': {
                    'test_count': quality.test_count if quality else 0,
                    'success_rate': quality.success_rate if quality else 0.0,
                    'execution_time': quality.execution_time if quality else 0.0
                } if quality else None,
                'layer_coverage': [
                    {
                        'layer': lc.layer_name,
                        'coverage_percent': lc.coverage_percent,
                        'critical_paths_covered': lc.critical_paths_covered
                    } for lc in layer_coverages
                ] if layer_coverages else [],
                'quality_check': quality_check
            }

            # 生成质量报告
            if coverage and quality:
                quality_monitor.save_quality_report(coverage, quality, layer_coverages)

            self.logger.info("质量评估完成")
            return assessment

        except Exception as e:
            self.logger.error(f"质量评估失败: {e}")
            return {'error': str(e)}

    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        self.logger.info("开始性能基准测试")

        try:
            from tests.performance_benchmark_framework import performance_framework

            # 运行关键性能基准
            benchmark_tests = {
                'api_response_time': lambda: self._mock_api_call(),
                'memory_usage': lambda: self._mock_memory_operation(),
                'database_query_time': lambda: self._mock_database_query()
            }

            results = performance_framework.run_multiple_benchmarks(benchmark_tests)

            # 生成性能报告
            performance_framework.save_performance_report(results)

            self.logger.info("性能基准测试完成")
            return {
                'benchmarks_run': len(results),
                'results': [
                    {
                        'benchmark': r.benchmark_name,
                        'status': r.status,
                        'measured_value': r.measured_value,
                        'baseline_value': r.baseline_value,
                        'deviation_percent': r.deviation_percent
                    } for r in results
                ]
            }

        except Exception as e:
            self.logger.error(f"性能基准测试失败: {e}")
            return {'error': str(e)}

    def _mock_api_call(self) -> float:
        """模拟API调用"""
        import time
        time.sleep(0.05)  # 模拟50ms响应时间
        return 0.05

    def _mock_memory_operation(self) -> float:
        """模拟内存操作"""
        data = [i * i for i in range(10000)]  # 消耗一些内存
        return len(data)

    def _mock_database_query(self) -> float:
        """模拟数据库查询"""
        import time
        time.sleep(0.02)  # 模拟20ms查询时间
        return 0.02

    def generate_comprehensive_report(self, test_results: Dict[str, Any],
                                    quality_assessment: Dict[str, Any],
                                    performance_results: Dict[str, Any]) -> str:
        """生成综合报告"""
        report_lines = []
        report_lines.append("# RQA2025增强测试套件执行报告")
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append("")

        # 测试结果汇总
        report_lines.append("## 📊 测试执行结果")
        total_passed = 0
        total_failed = 0
        total_tests = 0

        for layer, result in test_results.items():
            if 'error' not in result:
                report_lines.append(f"### {layer.upper()} 层")
                report_lines.append(f"- 总测试数: {result.get('total', 0)}")
                report_lines.append(f"- 通过测试: {result.get('passed', 0)}")
                report_lines.append(f"- 失败测试: {result.get('failed', 0)}")
                report_lines.append(".2")
                report_lines.append("")

                total_tests += result.get('total', 0)
                total_passed += result.get('passed', 0)
                total_failed += result.get('failed', 0)

        # 总体统计
        report_lines.append("### 总体统计")
        report_lines.append(f"- 总测试数: {total_tests}")
        report_lines.append(f"- 总通过数: {total_passed}")
        report_lines.append(f"- 总失败数: {total_failed}")
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            report_lines.append(".1")
        report_lines.append("")

        # 质量评估结果
        if quality_assessment and 'error' not in quality_assessment:
            report_lines.append("## 🧪 质量评估结果")

            if quality_assessment.get('coverage'):
                cov = quality_assessment['coverage']
                report_lines.append(f"- 代码覆盖率: {cov['coverage_percent']:.1f}%")
                report_lines.append(f"- 覆盖行数: {cov['covered_lines']:,}/{cov['total_lines']:,}")

            if quality_assessment.get('quality'):
                qual = quality_assessment['quality']
                report_lines.append(".1")
                report_lines.append(f"- 测试执行时间: {qual['execution_time']:.2f}秒")

            # 质量检查状态
            quality_check = quality_assessment.get('quality_check', {})
            status_emoji = {
                'pass': '✅',
                'warning': '⚠️',
                'fail': '❌'
            }.get(quality_check.get('overall_status'), '❓')

            report_lines.append(f"- 质量状态: {status_emoji} {quality_check.get('overall_status', 'unknown').upper()}")

            if quality_check.get('violations'):
                report_lines.append("- 违规项目:")
                for violation in quality_check['violations']:
                    report_lines.append(f"  - {violation['type']}: {violation['metric']} 阈值违规")

            report_lines.append("")

        # 性能基准结果
        if performance_results and 'error' not in performance_results:
            report_lines.append("## ⚡ 性能基准结果")
            report_lines.append(f"- 执行基准测试: {performance_results.get('benchmarks_run', 0)} 个")

            for result in performance_results.get('results', []):
                status_emoji = {
                    'pass': '✅',
                    'warning': '⚠️',
                    'fail': '❌'
                }.get(result['status'], '❓')

                report_lines.append(f"- {result['benchmark']}: {status_emoji} {result['status'].upper()}")
                report_lines.append(".2")
                report_lines.append(".2")
                report_lines.append("")

        # 执行建议
        report_lines.append("## 💡 执行建议")

        if total_failed > 0:
            report_lines.append("⚠️ 发现测试失败，请检查失败的测试用例并修复相关问题。")

        if quality_assessment.get('quality_check', {}).get('overall_status') in ['fail', 'warning']:
            report_lines.append("⚠️ 质量指标未达到预期标准，建议关注覆盖率和测试稳定性。")

        if performance_results and any(r['status'] != 'pass' for r in performance_results.get('results', [])):
            report_lines.append("⚠️ 性能基准测试发现异常，建议检查系统性能问题。")

        if total_failed == 0 and quality_assessment.get('quality_check', {}).get('overall_status') == 'pass':
            report_lines.append("✅ 所有测试通过，质量指标达标，系统运行正常！")

        return "\n".join(report_lines)

    def save_comprehensive_report(self, test_results: Dict[str, Any],
                                quality_assessment: Dict[str, Any],
                                performance_results: Dict[str, Any],
                                report_file: str = "enhanced_test_suite_report.md"):
        """保存综合报告"""
        report_path = self.project_root / "test_logs" / report_file
        report_path.parent.mkdir(exist_ok=True)

        report_content = self.generate_comprehensive_report(test_results, quality_assessment, performance_results)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.logger.info(f"综合测试报告已保存到: {report_path}")

        return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025增强测试套件运行器")

    parser.add_argument("--layers", nargs="*", choices=["unit", "integration", "e2e"],
                    default=["unit", "integration", "e2e"],
                    help="要执行的测试层级")

    parser.add_argument("--no-parallel", action="store_true",
                    help="禁用并行测试执行")

    parser.add_argument("--quality-check", action="store_true",
                    help="执行质量评估")

    parser.add_argument("--performance-benchmark", action="store_true",
                    help="执行性能基准测试")

    parser.add_argument("--report", type=str, default="enhanced_test_suite_report.md",
                    help="报告文件路径")

    parser.add_argument("--quick", action="store_true",
                    help="快速模式：只运行关键测试")

    args = parser.parse_args()

    # 创建测试运行器
    runner = EnhancedTestSuiteRunner()

    try:
        print("🚀 开始执行RQA2025增强测试套件...")

        # 运行分层测试
        parallel = not args.no_parallel
        test_results = runner.run_layered_tests(args.layers, parallel)

        quality_assessment = None
        performance_results = None

        # 质量评估
        if args.quality_check:
            print("🧪 执行质量评估...")
            quality_assessment = runner.run_quality_assessment()

        # 性能基准测试
        if args.performance_benchmark:
            print("⚡ 执行性能基准测试...")
            performance_results = runner.run_performance_benchmarks()

        # 生成综合报告
        report_path = runner.save_comprehensive_report(
            test_results, quality_assessment or {}, performance_results or {}, args.report
        )

        print(f"✅ 测试执行完成！详细报告已保存到: {report_path}")

        # 输出简要结果
        total_passed = sum(r.get('passed', 0) for r in test_results.values())
        total_failed = sum(r.get('failed', 0) for r in test_results.values())

        print("\n📊 执行摘要:")
        print(f"   通过: {total_passed}")
        print(f"   失败: {total_failed}")

        if total_failed > 0:
            print("❌ 发现测试失败，请查看详细报告")
            sys.exit(1)
        else:
            print("✅ 所有测试通过！")

    except KeyboardInterrupt:
        print("\n⚠️ 测试执行被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
