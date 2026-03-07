#!/usr/bin/env python3
"""
Phase 14.4: pytest执行性能监控脚本
监控测试执行时间、内存使用、成功率等关键指标
"""

import time
import subprocess
import sys
import psutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class PytestPerformanceMonitor:
    """pytest性能监控器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.metrics = []

    def monitor_execution(self, test_command: List[str], description: str) -> Dict[str, Any]:
        """监控pytest执行性能"""
        print(f"📊 监控: {description}")

        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        start_cpu = psutil.cpu_percent(interval=None)

        # 执行测试命令
        result = subprocess.run(
            test_command,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)  # MB
        end_cpu = psutil.cpu_percent(interval=None)

        # 解析结果
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'execution_time_seconds': end_time - start_time,
            'memory_used_mb': end_memory - start_memory,
            'cpu_usage_percent': (start_cpu + end_cpu) / 2,
            'return_code': result.returncode,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'errors': []
        }

        # 解析stdout
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if 'passed' in line and 'failed' in line:
                    # 解析类似 "10 passed, 2 failed, 1 skipped" 的行
                    parts = line.replace(',', '').split()
                    for i, part in enumerate(parts):
                        if part == 'passed,':
                            metrics['tests_passed'] = int(parts[i-1])
                        elif part == 'failed,':
                            metrics['tests_failed'] = int(parts[i-1])
                        elif part == 'skipped,':
                            metrics['tests_skipped'] = int(parts[i-1])

        # 记录错误
        if result.stderr:
            metrics['errors'] = result.stderr.strip().split('\n')

        # 计算派生指标
        total_tests = metrics['tests_passed'] + metrics['tests_failed'] + metrics['tests_skipped']
        metrics['total_tests'] = total_tests
        metrics['success_rate'] = (metrics['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        metrics['failure_rate'] = (metrics['tests_failed'] / total_tests * 100) if total_tests > 0 else 0

        print(".1f"        print(f"  ✅ 通过: {metrics['tests_passed']}")
        print(f"  ❌ 失败: {metrics['tests_failed']}")
        print(f"  ⏭️ 跳过: {metrics['tests_skipped']}")
        print(".1f"
        self.metrics.append(metrics)
        return metrics

    def run_monitoring_session(self) -> Dict[str, Any]:
        """运行完整的监控会话"""
        print("🚀 Phase 14.4: pytest执行性能监控会话")
        print("=" * 60)

        # 1. 基础功能测试
        self.monitor_execution([
            sys.executable, '-m', 'pytest',
            'tests/unit/infrastructure/test_config_low_coverage.py',
            'tests/unit/infrastructure/test_cache_low_coverage.py',
            '-v', '--tb=no'
        ], "基础功能测试 (串行)")

        # 2. 并行执行测试
        self.monitor_execution([
            sys.executable, '-m', 'pytest',
            'tests/unit/infrastructure/test_config_low_coverage.py',
            'tests/unit/infrastructure/test_cache_low_coverage.py',
            '-n=2', '--dist=loadscope', '-v', '--tb=no'
        ], "基础功能测试 (并行)")

        # 3. 集成测试性能
        self.monitor_execution([
            sys.executable, '-m', 'pytest',
            'tests/integration/infrastructure/test_infrastructure_core_integration.py',
            '-v', '--tb=no'
        ], "集成测试性能")

        # 4. 端到端测试性能
        self.monitor_execution([
            sys.executable, '-m', 'pytest',
            'tests/e2e/test_error_handling_e2e.py::TestErrorHandlingE2E::test_network_failure_e2e',
            '-v', '--tb=no'
        ], "端到端测试性能")

        # 5. 覆盖率测试性能
        self.monitor_execution([
            sys.executable, '-m', 'pytest',
            'tests/unit/infrastructure/',
            '--cov=src', '--cov-report=', '-x'
        ], "覆盖率测试性能")

        # 生成报告
        report = self.generate_report()
        self.save_report(report)

        print("\n" + "=" * 60)
        print("✅ Phase 14.4 性能监控完成")
        print("=" * 60)

        return report

    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.metrics:
            return {'error': 'No metrics collected'}

        report = {
            'session_timestamp': datetime.now().isoformat(),
            'total_executions': len(self.metrics),
            'summary': {},
            'performance_trends': [],
            'recommendations': []
        }

        # 计算汇总指标
        execution_times = [m['execution_time_seconds'] for m in self.metrics]
        memory_usage = [m['memory_used_mb'] for m in self.metrics]
        success_rates = [m['success_rate'] for m in self.metrics]

        report['summary'] = {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'avg_memory_usage': sum(memory_usage) / len(memory_usage),
            'avg_success_rate': sum(success_rates) / len(success_rates),
            'total_tests_run': sum(m['total_tests'] for m in self.metrics),
            'total_tests_passed': sum(m['tests_passed'] for m in self.metrics),
            'total_tests_failed': sum(m['tests_failed'] for m in self.metrics)
        }

        # 性能趋势分析
        for i, metric in enumerate(self.metrics):
            trend = {
                'execution_id': i + 1,
                'description': metric['description'],
                'execution_time': metric['execution_time_seconds'],
                'success_rate': metric['success_rate'],
                'memory_usage': metric['memory_used_mb']
            }
            report['performance_trends'].append(trend)

        # 生成建议
        avg_time = report['summary']['avg_execution_time']
        avg_success = report['summary']['avg_success_rate']

        if avg_time > 60:
            report['recommendations'].append('测试执行时间较长，建议优化测试或增加并行度')

        if avg_success < 95:
            report['recommendations'].append('测试成功率偏低，建议修复测试失败问题')

        if report['summary']['avg_memory_usage'] > 500:
            report['recommendations'].append('内存使用较高，建议优化测试或减少并发数')

        return report

    def save_report(self, report: Dict[str, Any]):
        """保存性能报告"""
        report_file = self.project_root / 'test_logs' / 'phase14_performance_report.json'
        metrics_file = self.project_root / 'test_logs' / 'phase14_performance_metrics.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"📄 性能报告已保存: {report_file}")
        print(f"📄 详细指标已保存: {metrics_file}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    monitor = PytestPerformanceMonitor(project_root)
    report = monitor.run_monitoring_session()

    # 打印摘要
    if 'summary' in report:
        summary = report['summary']
        print("
📊 性能监控摘要:"        print(".1f"        print(".1f"        print(".1f"        print(".1f"        print(f"  总测试数: {summary['total_tests_run']}")
        print(f"  通过/失败: {summary['total_tests_passed']}/{summary['total_tests_failed']}")


if __name__ == '__main__':
    main()
