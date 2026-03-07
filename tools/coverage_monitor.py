#!/usr/bin/env python3
"""
RQA2025 覆盖率监控工具
提供全面的测试覆盖率监控和报告功能
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CoverageMonitor:
    """覆盖率监控器"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.reports_dir = os.path.join(project_root, "coverage_reports")
        self.history_file = os.path.join(self.reports_dir, "coverage_history.json")
        self.baseline_file = os.path.join(self.reports_dir, "coverage_baseline.json")

        # 创建报告目录
        os.makedirs(self.reports_dir, exist_ok=True)

        # 设置matplotlib样式
        plt.style.use('default')
        sns.set_palette("husl")

    def run_coverage_analysis(self, include_integration: bool = True) -> Dict[str, Any]:
        """运行覆盖率分析"""
        print("🔍 开始覆盖率分析...")

        # 运行单元测试覆盖率
        unit_coverage = self._run_unit_test_coverage()

        # 运行集成测试覆盖率（如果启用）
        integration_coverage = {}
        if include_integration:
            integration_coverage = self._run_integration_test_coverage()

        # 合并结果
        coverage_data = {
            'timestamp': datetime.now().isoformat(),
            'unit_tests': unit_coverage,
            'integration_tests': integration_coverage,
            'summary': self._calculate_summary(unit_coverage, integration_coverage)
        }

        # 保存历史记录
        self._save_history(coverage_data)

        print("✅ 覆盖率分析完成")
        return coverage_data

    def _run_unit_test_coverage(self) -> Dict[str, Any]:
        """运行单元测试覆盖率"""
        print("📊 运行单元测试覆盖率分析...")

        try:
            # 运行pytest-cov获取覆盖率数据
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/unit/",
                "--cov=src",
                "--cov-report=json:coverage_unit.json",
                "--cov-report=term-missing",
                "-q", "--tb=no", "--disable-warnings"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )

            # 解析覆盖率JSON文件
            coverage_file = os.path.join(self.project_root, "coverage_unit.json")
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)

                # 提取关键指标
                totals = coverage_data.get('totals', {})

                unit_coverage = {
                    'overall_coverage': totals.get('percent_covered', 0),
                    'covered_lines': totals.get('covered_lines', 0),
                    'num_statements': totals.get('num_statements', 0),
                    'missing_lines': totals.get('missing_lines', 0),
                    'excluded_lines': totals.get('excluded_lines', 0),
                    'files': self._extract_file_coverage(coverage_data),
                    'test_result': 'passed' if result.returncode == 0 else 'failed'
                }

                # 清理临时文件
                os.remove(coverage_file)

                return unit_coverage
            else:
                return {
                    'overall_coverage': 0,
                    'error': 'Coverage file not generated',
                    'test_result': 'failed'
                }

        except Exception as e:
            print(f"❌ 单元测试覆盖率分析失败: {e}")
            return {
                'overall_coverage': 0,
                'error': str(e),
                'test_result': 'error'
            }

    def _run_integration_test_coverage(self) -> Dict[str, Any]:
        """运行集成测试覆盖率"""
        print("🔗 运行集成测试覆盖率分析...")

        try:
            # 运行集成测试覆盖率
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/integration/",
                "--cov=src",
                "--cov-report=json:coverage_integration.json",
                "--cov-report=term-missing",
                "-q", "--tb=no", "--disable-warnings"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )

            # 解析集成测试覆盖率
            coverage_file = os.path.join(self.project_root, "coverage_integration.json")
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get('totals', {})

                integration_coverage = {
                    'overall_coverage': totals.get('percent_covered', 0),
                    'covered_lines': totals.get('covered_lines', 0),
                    'num_statements': totals.get('num_statements', 0),
                    'files': self._extract_file_coverage(coverage_data),
                    'test_result': 'passed' if result.returncode == 0 else 'failed'
                }

                # 清理临时文件
                os.remove(coverage_file)

                return integration_coverage
            else:
                return {
                    'overall_coverage': 0,
                    'error': 'Integration coverage file not generated',
                    'test_result': 'failed'
                }

        except Exception as e:
            print(f"❌ 集成测试覆盖率分析失败: {e}")
            return {
                'overall_coverage': 0,
                'error': str(e),
                'test_result': 'error'
            }

    def _extract_file_coverage(self, coverage_data: Dict) -> Dict[str, Any]:
        """提取文件级覆盖率数据"""
        files_data = {}

        for file_path, file_data in coverage_data.get('files', {}).items():
            # 只关注src目录下的文件
            if file_path.startswith('src/'):
                files_data[file_path] = {
                    'coverage': file_data.get('summary', {}).get('percent_covered', 0),
                    'covered_lines': file_data.get('summary', {}).get('covered_lines', 0),
                    'num_statements': file_data.get('summary', {}).get('num_statements', 0),
                    'missing_lines': file_data.get('summary', {}).get('missing_lines', 0)
                }

        return files_data

    def _calculate_summary(self, unit_coverage: Dict, integration_coverage: Dict) -> Dict[str, Any]:
        """计算汇总指标"""
        summary = {
            'overall_coverage': unit_coverage.get('overall_coverage', 0),
            'unit_test_coverage': unit_coverage.get('overall_coverage', 0),
            'integration_test_coverage': integration_coverage.get('overall_coverage', 0),
            'test_pass_rate': 100.0 if unit_coverage.get('test_result') == 'passed' else 0.0,
            'module_coverage': {}
        }

        # 计算各模块覆盖率
        unit_files = unit_coverage.get('files', {})
        for file_path, file_data in unit_files.items():
            module = file_path.split('/')[1] if '/' in file_path else 'unknown'
            if module not in summary['module_coverage']:
                summary['module_coverage'][module] = {
                    'files': 0,
                    'total_coverage': 0,
                    'avg_coverage': 0
                }

            summary['module_coverage'][module]['files'] += 1
            summary['module_coverage'][module]['total_coverage'] += file_data['coverage']

        # 计算平均覆盖率
        for module_data in summary['module_coverage'].values():
            if module_data['files'] > 0:
                module_data['avg_coverage'] = module_data['total_coverage'] / module_data['files']

        return summary

    def _save_history(self, coverage_data: Dict[str, Any]):
        """保存覆盖率历史记录"""
        try:
            # 读取现有历史
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            # 添加新记录
            history.append(coverage_data)

            # 只保留最近100条记录
            if len(history) > 100:
                history = history[-100:]

            # 保存历史
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 保存覆盖率历史失败: {e}")

    def generate_coverage_report(self, output_file: Optional[str] = None) -> str:
        """生成覆盖率报告"""
        print("📋 生成覆盖率报告...")

        # 获取最新覆盖率数据
        coverage_data = self.run_coverage_analysis()

        # 生成报告
        report = self._generate_detailed_report(coverage_data)

        # 保存报告
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.reports_dir, f"coverage_report_{timestamp}.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 覆盖率报告已保存到: {output_file}")
        return output_file

    def _generate_detailed_report(self, coverage_data: Dict[str, Any]) -> str:
        """生成详细的覆盖率报告"""
        timestamp = coverage_data['timestamp']
        summary = coverage_data['summary']

        report = f"""# RQA2025 覆盖率分析报告

**生成时间**: {timestamp}

## 📊 总体概况

- **整体覆盖率**: {summary['overall_coverage']:.1f}%
- **单元测试覆盖率**: {summary['unit_test_coverage']:.1f}%
- **集成测试覆盖率**: {summary['integration_test_coverage']:.1f}%
- **测试通过率**: {summary['test_pass_rate']:.1f}%

## 📈 模块覆盖率详情

| 模块 | 文件数 | 平均覆盖率 | 状态 |
|------|--------|------------|------|
"""

        for module, data in summary['module_coverage'].items():
            status = "✅" if data['avg_coverage'] >= 80 else "⚠️" if data['avg_coverage'] >= 60 else "❌"
            report += f"| {module} | {data['files']} | {data['avg_coverage']:.1f}% | {status} |\n"

        report += "\n## 🔍 详细文件覆盖率\n\n"

        unit_files = coverage_data['unit_tests'].get('files', {})
        sorted_files = sorted(unit_files.items(), key=lambda x: x[1]['coverage'])

        for file_path, file_data in sorted_files:
            coverage = file_data['coverage']
            status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
            report += f"- {status} **{file_path}**: {coverage:.1f}% ({file_data['covered_lines']}/{file_data['num_statements']})\n"

        report += "\n## 🎯 覆盖率目标达成情况\n\n"

        overall_target = 80.0
        current = summary['overall_coverage']

        if current >= overall_target:
            report += f"🎉 **恭喜！整体覆盖率已达到{overall_target}%目标！**\n"
            report += f"当前覆盖率: {current:.1f}%\n"
        else:
            gap = overall_target - current
            report += f"⚠️ **整体覆盖率距离{overall_target}%目标还差{gap:.1f}%**\n"
            report += f"当前覆盖率: {current:.1f}%\n"

        report += "\n## 📋 改进建议\n\n"

        # 分析低覆盖率文件
        low_coverage_files = [f for f, d in unit_files.items() if d['coverage'] < 60]
        if low_coverage_files:
            report += "### 🔧 需要重点关注的低覆盖率文件:\n"
            for file_path in low_coverage_files[:10]:  # 只显示前10个
                report += f"- `{file_path}`: {unit_files[file_path]['coverage']:.1f}%\n"
            report += "\n"

        # 覆盖率趋势分析
        report += "### 📈 覆盖率趋势建议:\n"
        report += "- 继续补充边界条件测试\n"
        report += "- 增加异常处理路径覆盖\n"
        report += "- 完善集成测试场景\n"
        report += "- 建立定期覆盖率监控机制\n"

        report += "\n---\n"
        report += "*报告由覆盖率监控工具自动生成*\n"

        return report

    def generate_trend_chart(self, days: int = 30):
        """生成覆盖率趋势图表"""
        try:
            if not os.path.exists(self.history_file):
                print("❌ 没有找到覆盖率历史数据")
                return

            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # 提取最近N天的趋势数据
            recent_data = history[-days:] if len(history) > days else history

            dates = []
            coverages = []

            for entry in recent_data:
                try:
                    date = datetime.fromisoformat(entry['timestamp'])
                    coverage = entry['summary']['overall_coverage']
                    dates.append(date)
                    coverages.append(coverage)
                except:
                    continue

            if not dates:
                print("❌ 没有有效的历史数据")
                return

            # 创建趋势图
            plt.figure(figsize=(12, 6))
            plt.plot(dates, coverages, marker='o', linewidth=2, markersize=4)
            plt.title('RQA2025 覆盖率趋势图', fontsize=16, fontweight='bold')
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('覆盖率 (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)

            # 添加目标线
            plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='目标: 80%')
            plt.legend()

            # 保存图表
            chart_file = os.path.join(self.reports_dir, "coverage_trend.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 覆盖率趋势图已保存到: {chart_file}")

        except Exception as e:
            print(f"❌ 生成趋势图失败: {e}")

    def set_baseline(self, coverage_data: Dict[str, Any]):
        """设置覆盖率基准"""
        try:
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'baseline': coverage_data
                }, f, indent=2, ensure_ascii=False)

            print("✅ 覆盖率基准已设置")
        except Exception as e:
            print(f"❌ 设置基准失败: {e}")

    def compare_with_baseline(self) -> Dict[str, Any]:
        """与基准比较覆盖率变化"""
        try:
            if not os.path.exists(self.baseline_file):
                return {'error': 'No baseline found'}

            with open(self.baseline_file, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)

            # 获取当前覆盖率
            current_data = self.run_coverage_analysis()

            baseline_coverage = baseline_data['baseline']['summary']['overall_coverage']
            current_coverage = current_data['summary']['overall_coverage']

            change = current_coverage - baseline_coverage

            return {
                'baseline_coverage': baseline_coverage,
                'current_coverage': current_coverage,
                'change': change,
                'change_percent': (change / baseline_coverage * 100) if baseline_coverage > 0 else 0,
                'status': 'improved' if change > 0 else 'declined' if change < 0 else 'stable'
            }

        except Exception as e:
            return {'error': str(e)}


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 覆盖率监控工具')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--report', action='store_true', help='生成覆盖率报告')
    parser.add_argument('--trend', action='store_true', help='生成趋势图表')
    parser.add_argument('--set-baseline', action='store_true', help='设置覆盖率基准')
    parser.add_argument('--compare-baseline', action='store_true', help='与基准比较')

    args = parser.parse_args()

    # 获取项目根目录
    project_root = os.path.abspath(args.project_root)

    # 创建监控器
    monitor = CoverageMonitor(project_root)

    if args.set_baseline:
        print("📊 设置覆盖率基准...")
        coverage_data = monitor.run_coverage_analysis()
        monitor.set_baseline(coverage_data)
        print("✅ 基准设置完成")

    elif args.compare_baseline:
        print("📊 比较覆盖率变化...")
        comparison = monitor.compare_with_baseline()
        if 'error' in comparison:
            print(f"❌ 比较失败: {comparison['error']}")
        else:
            print(f"基准覆盖率: {comparison['baseline_coverage']:.1f}%")
            print(f"当前覆盖率: {comparison['current_coverage']:.1f}%")
            print(f"变化: {comparison['change']:+.1f}% ({comparison['change_percent']:+.1f}%)")
            print(f"状态: {comparison['status']}")

    elif args.trend:
        print("📈 生成覆盖率趋势图...")
        monitor.generate_trend_chart()

    elif args.report:
        print("📋 生成覆盖率报告...")
        report_file = monitor.generate_coverage_report()
        print(f"报告已生成: {report_file}")

    else:
        # 默认操作：运行分析并生成报告
        print("🔍 运行完整覆盖率分析...")
        coverage_data = monitor.run_coverage_analysis()
        report_file = monitor.generate_coverage_report()

        # 显示关键指标
        summary = coverage_data['summary']
        print("\n📊 关键指标:")
        print(f"  整体覆盖率: {summary['overall_coverage']:.1f}%")
        print(f"  测试通过率: {summary['test_pass_rate']:.1f}%")
        print(f"  覆盖文件数: {len(summary['module_coverage'])}")

        # 检查目标达成
        target_coverage = 80.0
        if summary['overall_coverage'] >= target_coverage:
            print(f"\n🎉 恭喜！覆盖率已达到{target_coverage}%目标！")
        else:
            gap = target_coverage - summary['overall_coverage']
            print(f"\n⚠️ 距离{target_coverage}%目标还差{gap:.1f}%")


if __name__ == "__main__":
    main()
