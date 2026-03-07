#!/usr/bin/env python3
"""
基础设施层质量监控系统

定期审查测试覆盖情况，建立质量趋势分析
"""

import subprocess
import sys
import time
import os
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse


class QualityMonitor:
    """质量监控器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports"
        self.history_file = self.reports_dir / "quality_history.json"
        self.thresholds = {
            'min_coverage': 30.0,  # 最低覆盖率阈值
            'min_success_rate': 70.0,  # 最低成功率阈值
            'max_avg_duration': 10.0,  # 最大平均执行时间（秒）
            'critical_tests': ['cache', 'logging', 'config']  # 关键测试模块
        }

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """运行完整的测试套件"""
        print("🏃 运行完整测试套件...")

        # 运行基础设施层测试
        test_cmd = [
            "python", "-m", "pytest",
            "tests/unit/infrastructure/",
            "--cov=src/infrastructure",
            "--cov-report=json:reports/coverage.json",
            "--cov-report=term-missing",
            "--durations=20",
            "--durations-min=1.0",
            "--tb=no",
            "-q"
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                test_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )

            end_time = time.time()
            total_duration = end_time - start_time

            # 解析结果
            output_lines = result.stdout.split('\n')
            passed = failed = errors = 0

            for line in output_lines:
                if 'passed' in line and 'failed' in line:
                    # 解析类似 "19 passed, 3 warnings, 2 errors in 5.43s" 的行
                    parts = line.split(',')
                    for part in parts:
                        part = part.strip()
                        if 'passed' in part:
                            passed = int(part.split()[0])
                        elif 'failed' in part:
                            failed = int(part.split()[0])
                        elif 'errors' in part:
                            errors = int(part.split()[0])

            return {
                'timestamp': datetime.now().isoformat(),
                'total_tests': passed + failed + errors,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'success_rate': (passed / (passed + failed + errors)) * 100 if (passed + failed + errors) > 0 else 0,
                'total_duration': total_duration,
                'exit_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'timeout',
                'total_duration': 600,
                'exit_code': -1
            }

    def analyze_coverage(self) -> Dict[str, Any]:
        """分析覆盖率"""
        print("📊 分析测试覆盖率...")

        coverage_file = self.reports_dir / "coverage.json"

        if not coverage_file.exists():
            return {'error': 'Coverage file not found'}

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

            totals = coverage_data.get('totals', {})
            files = coverage_data.get('files', {})

            # 计算各模块的覆盖率
            module_coverage = {}
            for file_path, file_data in files.items():
                if 'src/infrastructure' in file_path:
                    # 提取模块名
                    parts = file_path.replace('src/infrastructure/', '').split('/')
                    module = parts[0] if parts[0] else 'root'

                    if module not in module_coverage:
                        module_coverage[module] = {'files': 0, 'covered_lines': 0, 'total_lines': 0}

                    summary = file_data.get('summary', {})
                    module_coverage[module]['files'] += 1
                    module_coverage[module]['covered_lines'] += summary.get('covered_lines', 0)
                    module_coverage[module]['total_lines'] += summary.get('num_statements', 0)

            # 计算模块覆盖率
            for module, data in module_coverage.items():
                if data['total_lines'] > 0:
                    data['coverage_percent'] = (data['covered_lines'] / data['total_lines']) * 100
                else:
                    data['coverage_percent'] = 0.0

            return {
                'timestamp': datetime.now().isoformat(),
                'overall_coverage': totals.get('percent_covered', 0),
                'module_coverage': module_coverage,
                'files_covered': totals.get('num_statements', 0),
                'missing_lines': totals.get('missing_lines', 0)
            }

        except Exception as e:
            return {'error': f'Failed to analyze coverage: {e}'}

    def check_quality_thresholds(self, test_results: Dict, coverage_results: Dict) -> Dict[str, Any]:
        """检查质量阈值"""
        print("🔍 检查质量阈值...")

        issues = []
        warnings = []

        # 检查测试成功率
        success_rate = test_results.get('success_rate', 0)
        if success_rate < self.thresholds['min_success_rate']:
            issues.append({
                'type': 'critical',
                'metric': 'success_rate',
                'value': success_rate,
                'threshold': self.thresholds['min_success_rate'],
                'message': '.1f'
            })

        # 检查覆盖率
        overall_coverage = coverage_results.get('overall_coverage', 0)
        if overall_coverage < self.thresholds['min_coverage']:
            issues.append({
                'type': 'critical',
                'metric': 'coverage',
                'value': overall_coverage,
                'threshold': self.thresholds['min_coverage'],
                'message': '.1f'
            })

        # 检查关键模块覆盖率
        module_coverage = coverage_results.get('module_coverage', {})
        for module in self.thresholds['critical_tests']:
            if module in module_coverage:
                module_cov = module_coverage[module].get('coverage_percent', 0)
                if module_cov < self.thresholds['min_coverage']:
                    warnings.append({
                        'type': 'warning',
                        'metric': f'{module}_coverage',
                        'value': module_cov,
                        'threshold': self.thresholds['min_coverage'],
                        'message': f'{module}模块覆盖率过低: {module_cov:.1f}%'
                    })

        # 检查平均执行时间
        total_duration = test_results.get('total_duration', 0)
        total_tests = test_results.get('total_tests', 1)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0

        if avg_duration > self.thresholds['max_avg_duration']:
            warnings.append({
                'type': 'warning',
                'metric': 'avg_duration',
                'value': avg_duration,
                'threshold': self.thresholds['max_avg_duration'],
                'message': '.2f'
            })

        return {
            'issues': issues,
            'warnings': warnings,
            'overall_status': 'critical' if issues else ('warning' if warnings else 'healthy')
        }

    def load_history(self) -> List[Dict]:
        """加载历史数据"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_history(self, history: List[Dict]):
        """保存历史数据"""
        self.reports_dir.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def generate_trend_analysis(self, history: List[Dict], current: Dict) -> Dict[str, Any]:
        """生成趋势分析"""
        if len(history) < 2:
            return {'message': '需要至少2次测量才能进行趋势分析'}

        # 计算趋势
        recent = history[-5:] + [current]  # 最近5次 + 当前
        success_rates = [h.get('test_results', {}).get('success_rate', 0) for h in recent]
        coverages = [h.get('coverage_results', {}).get('overall_coverage', 0) for h in recent]
        durations = [h.get('test_results', {}).get('total_duration', 0) for h in recent]

        def calculate_trend(values):
            if len(values) < 2:
                return 0
            return (values[-1] - values[0]) / len(values)  # 平均变化率

        return {
            'success_rate_trend': calculate_trend(success_rates),
            'coverage_trend': calculate_trend(coverages),
            'duration_trend': calculate_trend(durations),
            'data_points': len(recent)
        }

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """运行完整的监控周期"""
        print("🔄 开始质量监控周期")
        print("=" * 50)

        # 1. 运行测试
        test_results = self.run_comprehensive_test_suite()

        # 2. 分析覆盖率
        coverage_results = self.analyze_coverage()

        # 3. 检查质量阈值
        quality_check = self.check_quality_thresholds(test_results, coverage_results)

        # 4. 加载历史数据
        history = self.load_history()

        # 5. 生成趋势分析
        current_data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'coverage_results': coverage_results,
            'quality_check': quality_check
        }

        trend_analysis = self.generate_trend_analysis(history, current_data)

        # 6. 保存到历史
        history.append(current_data)
        # 只保留最近30天的记录
        cutoff_date = datetime.now() - timedelta(days=30)
        history = [h for h in history if datetime.fromisoformat(h['timestamp']) > cutoff_date]

        self.save_history(history)

        # 7. 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'coverage_results': coverage_results,
            'quality_check': quality_check,
            'trend_analysis': trend_analysis,
            'history_size': len(history)
        }

        return report

    def print_report(self, report: Dict):
        """打印监控报告"""
        print("\n" + "=" * 60)
        print("📋 基础设施层质量监控报告")
        print("=" * 60)

        # 测试结果
        test_results = report.get('test_results', {})
        print(f"🧪 测试结果:")
        print(f"   总测试数: {test_results.get('total_tests', 0)}")
        print(f"   通过: {test_results.get('passed', 0)}")
        print(f"   失败: {test_results.get('failed', 0)}")
        print(f"   错误: {test_results.get('errors', 0)}")
        print(".1f")
        print(".1f")
        # 覆盖率结果
        coverage_results = report.get('coverage_results', {})
        if 'overall_coverage' in coverage_results:
            print("\n📊 覆盖率结果:")
            print(".1f")
            print(f"   文件数: {coverage_results.get('files_covered', 0)}")
            print(f"   缺失行数: {coverage_results.get('missing_lines', 0)}")

            # 模块覆盖率
            module_coverage = coverage_results.get('module_coverage', {})
            if module_coverage:
                print("\n🏗️  模块覆盖率:")
                for module, data in sorted(module_coverage.items()):
                    cov = data.get('coverage_percent', 0)
                    files = data.get('files', 0)
                    print(".1f"
        # 质量检查
        quality_check=report.get('quality_check', {})
        issues=quality_check.get('issues', [])
        warnings=quality_check.get('warnings', [])

        if issues:
            print("
❌ 质量问题: " for issue in issues:
                print(f"   🚨 {issue['message']}")

        if warnings:
            print("
⚠️  质量警告: " for warning in warnings:
                print(f"   ⚠️  {warning['message']}")

        status=quality_check.get('overall_status', 'unknown')
        if status == 'healthy':
            print("
✅ 整体状态: 健康" elif status == 'warning':
            print("
⚠️  整体状态: 需要注意" else:
            print("
❌ 整体状态: 需要改进"
        # 趋势分析
        trend=report.get('trend_analysis', {})
        if 'success_rate_trend' in trend:
            print("
📈 趋势分析: "            print(".3f"            print(".3f"            print(".3f"            print(f"   数据点数: {trend.get('data_points', 0)}")

        print(f"\n📁 历史记录数: {report.get('history_size', 0)}")
        print(f"📅 生成时间: {report.get('timestamp', 'unknown')}")

def main():
    """主函数"""
    parser=argparse.ArgumentParser(description='基础设施层质量监控系统')
    parser.add_argument('--continuous', action='store_true',
                       help='持续监控模式')
    parser.add_argument('--interval', type=int, default=3600,
                       help='监控间隔(秒，默认1小时)')
    parser.add_argument('--once', action='store_true',
                       help='只执行一次监控')

    args=parser.parse_args()

    project_root=Path(__file__).resolve().parent
    monitor=QualityMonitor(project_root)

    if args.once:
        # 执行一次监控
        report=monitor.run_monitoring_cycle()
        monitor.print_report(report)
        return 0

    if args.continuous:
        # 持续监控模式
        print(f"🔄 启动持续监控模式 (间隔: {args.interval}秒)")
        try:
            while True:
                report=monitor.run_monitoring_cycle()
                monitor.print_report(report)

                print(f"\n⏰ 等待 {args.interval} 秒后进行下次监控...")
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
            return 0
    else:
        # 默认执行一次
        report=monitor.run_monitoring_cycle()
        monitor.print_report(report)
        return 0

if __name__ == "__main__":
    exit(main())
