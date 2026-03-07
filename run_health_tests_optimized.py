#!/usr/bin/env python3
"""
健康管理模块测试优化执行脚本

支持并行执行、性能监控和详细报告
"""

import subprocess
import sys
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any
import json


class TestRunner:
    """优化测试运行器"""

    def __init__(self):
        self.start_time = time.time()
        self.process = None
        self.monitoring_active = False
        self.performance_data = []

    def monitor_system_resources(self):
        """监控系统资源使用情况"""
        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                self.performance_data.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'disk_percent': disk.percent
                })

                time.sleep(2)  # 每2秒收集一次数据
            except Exception as e:
                print(f"资源监控错误: {e}")
                break

    def run_tests_parallel(self, test_files: List[str], workers: int = 4) -> Dict[str, Any]:
        """并行运行测试"""
        print(f"🚀 启动并行测试执行 (工作进程数: {workers})")

        # 启动资源监控
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self.monitor_system_resources, daemon=True)
        monitor_thread.start()

        try:
            # 构建pytest命令
            cmd = [
                sys.executable, '-m', 'pytest',
                '-n', str(workers),  # 并行工作进程
                '--tb=short',  # 简短错误信息
                '--cov=src/infrastructure/health',  # 覆盖率分析
                '--cov-report=term-missing',  # 终端覆盖率报告
                '--cov-report=json:coverage_parallel.json',  # JSON覆盖率报告
                '--durations=10',  # 显示最慢的10个测试
                '-q'  # 安静模式
            ] + test_files

            print(f"执行命令: {' '.join(cmd)}")

            # 运行测试
            self.start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=Path.cwd())

            execution_time = time.time() - self.start_time

            # 停止资源监控
            self.monitoring_active = False
            monitor_thread.join(timeout=5)

            return {
                'return_code': result.returncode,
                'stdout': result.stdout or '',
                'stderr': result.stderr or '',
                'execution_time': execution_time,
                'performance_data': self.performance_data.copy()
            }

        except Exception as e:
            self.monitoring_active = False
            monitor_thread.join(timeout=5)
            return {
                'return_code': 1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': time.time() - self.start_time,
                'performance_data': self.performance_data.copy()
            }

    def run_tests_sequential(self, test_files: List[str]) -> Dict[str, Any]:
        """顺序运行测试（用于调试）"""
        print("🔧 启动顺序测试执行（调试模式）")

        cmd = [
            sys.executable, '-m', 'pytest',
            '--tb=long',  # 详细错误信息
            '--cov=src/infrastructure/health',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_sequential.json'
        ] + test_files

        print(f"执行命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=Path.cwd())

        return {
            'return_code': result.returncode,
            'stdout': result.stdout or '',
            'stderr': result.stderr or '',
            'execution_time': time.time() - self.start_time
        }

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            'success': results['return_code'] == 0,
            'execution_time_seconds': results['execution_time'],
            'has_performance_data': 'performance_data' in results,
            'warnings': [],
            'recommendations': []
        }

        # 分析执行时间
        if results['execution_time'] > 300:  # 超过5分钟
            analysis['warnings'].append("测试执行时间过长")
            analysis['recommendations'].append("考虑增加并行工作进程或优化测试")

        # 分析性能数据
        if 'performance_data' in results and results['performance_data']:
            perf_data = results['performance_data']
            avg_cpu = sum(d['cpu_percent'] for d in perf_data) / len(perf_data)
            max_memory = max(d['memory_percent'] for d in perf_data)

            analysis['avg_cpu_usage'] = avg_cpu
            analysis['max_memory_usage'] = max_memory

            if avg_cpu > 80:
                analysis['warnings'].append("CPU使用率较高")
                analysis['recommendations'].append("考虑减少并行工作进程")

            if max_memory > 85:
                analysis['warnings'].append("内存使用率较高")
                analysis['recommendations'].append("考虑增加系统内存或优化测试")

        # 分析输出
        stdout_lines = results['stdout'].split('\n')
        stderr_lines = results['stderr'].split('\n')

        # 查找失败的测试
        failed_tests = []
        for line in stdout_lines + stderr_lines:
            if 'FAILED' in line or 'ERROR' in line:
                failed_tests.append(line.strip())

        analysis['failed_tests_count'] = len(failed_tests)
        analysis['failed_tests'] = failed_tests[:10]  # 只显示前10个

        return analysis

    def generate_report(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("🧪 健康管理模块测试执行报告")
        report.append("=" * 60)

        report.append(f"执行状态: {'✅ 成功' if analysis['success'] else '❌ 失败'}")
        report.append(f"执行时间: {analysis['execution_time_seconds']:.2f} 秒")
        report.append(f"失败测试数: {analysis['failed_tests_count']}")

        if analysis['has_performance_data']:
            report.append(f"平均CPU使用率: {analysis['avg_cpu_usage']:.1f}%")
            report.append(f"最大内存使用率: {analysis['max_memory_usage']:.1f}%")

        if analysis['warnings']:
            report.append("\n⚠️  警告:")
            for warning in analysis['warnings']:
                report.append(f"  • {warning}")

        if analysis['recommendations']:
            report.append("\n💡 建议:")
            for rec in analysis['recommendations']:
                report.append(f"  • {rec}")

        if analysis['failed_tests_count'] > 0:
            report.append(f"\n❌ 失败测试 (前{len(analysis['failed_tests'])}个):")
            for test in analysis['failed_tests']:
                report.append(f"  • {test}")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """主函数"""
    # 定义测试文件列表
    health_test_files = [
        'tests/unit/infrastructure/health/test_probe_components_comprehensive.py',
        'tests/unit/infrastructure/health/test_status_components_comprehensive.py',
        'tests/unit/infrastructure/health/test_model_monitor_plugin_comprehensive.py',
        'tests/unit/infrastructure/health/test_zero_coverage_special.py',
        'tests/unit/infrastructure/health/test_corrected_components.py',
        'tests/unit/infrastructure/health/test_prometheus_integration_deep.py',
        'tests/unit/infrastructure/health/test_real_business_logic.py',
        'tests/unit/infrastructure/health/test_health_checker_comprehensive.py',
        'tests/unit/infrastructure/health/test_alert_components_core.py',
        'tests/unit/infrastructure/health/test_app_monitor_core_methods.py',
        'tests/unit/infrastructure/health/test_critical_low_coverage.py',
        'tests/unit/infrastructure/health/test_database_health_monitor.py',
        'tests/unit/infrastructure/health/test_disaster_monitor_enhanced.py',
        'tests/unit/infrastructure/health/test_enhanced_health_checker.py',
        'tests/unit/infrastructure/health/test_health_base.py',
        'tests/unit/infrastructure/health/test_health_data_api.py',
        'tests/unit/infrastructure/health/test_health_interfaces.py',
        'tests/unit/infrastructure/health/test_low_coverage_modules.py',
        'tests/unit/infrastructure/health/test_health_checker_simple.py',
        'tests/unit/infrastructure/health/test_automation_monitor_comprehensive.py',
        'tests/unit/infrastructure/health/test_backtest_monitor_plugin_comprehensive.py',
        'tests/unit/infrastructure/health/test_basic_health_checker_comprehensive.py',
        'tests/unit/infrastructure/health/test_application_monitor_comprehensive.py',
        'tests/unit/infrastructure/health/test_performance_monitor_comprehensive.py',
        'tests/unit/infrastructure/health/test_health_system_integration.py',  # 新增集成测试
    ]

    runner = TestRunner()

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--sequential':
        # 顺序执行模式
        results = runner.run_tests_sequential(health_test_files)
    else:
        # 并行执行模式
        results = runner.run_tests_parallel(health_test_files, workers=4)

    # 分析结果
    analysis = runner.analyze_results(results)

    # 生成报告
    report = runner.generate_report(results, analysis)

    # 输出报告
    print(report)

    # 保存详细结果到文件
    output_file = 'test_execution_report.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'analysis': analysis,
            'timestamp': time.time()
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存到: {output_file}")

    # 返回适当的退出码
    sys.exit(results['return_code'])


if __name__ == '__main__':
    main()
