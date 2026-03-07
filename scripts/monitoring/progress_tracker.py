#!/usr/bin/env python3
"""
RQA2025 模型落地进度跟踪脚本

功能：
1. 跟踪各层测试进度
2. 监控覆盖率变化
3. 生成进度报告
4. 识别瓶颈和风险
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.progress_file = self.project_root / "docs" / "progress_tracking.json"
        self.progress_file.parent.mkdir(exist_ok=True)

        # 各层目标配置
        self.layer_targets = {
            'infrastructure': {
                'target_coverage': 90,
                'priority': 'highest',
                'estimated_days': 3,
                'dependencies': []
            },
            'data': {
                'target_coverage': 80,
                'priority': 'high',
                'estimated_days': 4,
                'dependencies': ['infrastructure']
            },
            'features': {
                'target_coverage': 80,
                'priority': 'high',
                'estimated_days': 3,
                'dependencies': ['data']
            },
            'models': {
                'target_coverage': 80,
                'priority': 'medium',
                'estimated_days': 3,
                'dependencies': ['features']
            },
            'trading': {
                'target_coverage': 80,
                'priority': 'medium',
                'estimated_days': 3,
                'dependencies': ['models']
            },
            'backtest': {
                'target_coverage': 80,
                'priority': 'low',
                'estimated_days': 3,
                'dependencies': ['trading']
            }
        }

        # 加载历史进度
        self.progress_history = self.load_progress()

    def load_progress(self) -> List[Dict]:
        """加载历史进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载进度文件失败: {e}")
                return []
        return []

    def save_progress(self):
        """保存进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_history, f, indent=2, ensure_ascii=False)
            logger.info(f"进度已保存到: {self.progress_file}")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")

    def add_progress_entry(self, layer_results: List[Dict]):
        """添加进度条目"""
        timestamp = datetime.datetime.now()

        entry = {
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M:%S'),
            'layers': {}
        }

        for result in layer_results:
            layer_name = result['layer']
            test_result = result['test_result']
            coverage_result = result['coverage_result']

            entry['layers'][layer_name] = {
                'coverage': coverage_result.get('coverage', 0),
                'target_coverage': self.layer_targets[layer_name]['target_coverage'],
                'test_passed': test_result.get('passed', 0),
                'test_failed': test_result.get('failed', 0),
                'test_error': test_result.get('error', 0),
                'success': result['success']
            }

        self.progress_history.append(entry)
        self.save_progress()

        logger.info(f"添加进度条目: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def get_latest_progress(self) -> Optional[Dict]:
        """获取最新进度"""
        if self.progress_history:
            return self.progress_history[-1]
        return None

    def get_layer_progress(self, layer_name: str) -> List[Dict]:
        """获取指定层的进度历史"""
        layer_progress = []

        for entry in self.progress_history:
            if layer_name in entry['layers']:
                layer_data = entry['layers'][layer_name].copy()
                layer_data['timestamp'] = entry['timestamp']
                layer_data['date'] = entry['date']
                layer_progress.append(layer_data)

        return layer_progress

    def calculate_completion_percentage(self, layer_name: str) -> float:
        """计算完成百分比"""
        latest = self.get_latest_progress()
        if not latest or layer_name not in latest['layers']:
            return 0.0

        layer_data = latest['layers'][layer_name]
        current_coverage = layer_data['coverage']
        target_coverage = layer_data['target_coverage']

        # 基于覆盖率计算完成度
        coverage_completion = min(current_coverage / target_coverage, 1.0)

        # 基于测试通过率计算完成度
        total_tests = layer_data['test_passed'] + \
            layer_data['test_failed'] + layer_data['test_error']
        test_completion = layer_data['test_passed'] / total_tests if total_tests > 0 else 0.0

        # 综合完成度
        return (coverage_completion * 0.7 + test_completion * 0.3) * 100

    def identify_bottlenecks(self) -> List[Dict]:
        """识别瓶颈"""
        bottlenecks = []
        latest = self.get_latest_progress()

        if not latest:
            return bottlenecks

        for layer_name, layer_data in latest['layers'].items():
            completion = self.calculate_completion_percentage(layer_name)
            target = self.layer_targets[layer_name]

            # 识别低完成度的层
            if completion < 50:
                bottlenecks.append({
                    'layer': layer_name,
                    'type': 'low_completion',
                    'completion': completion,
                    'description': f'{layer_name}层完成度仅为{completion:.1f}%',
                    'priority': target['priority']
                })

            # 识别覆盖率差距大的层
            coverage_gap = target['target_coverage'] - layer_data['coverage']
            if coverage_gap > 20:
                bottlenecks.append({
                    'layer': layer_name,
                    'type': 'coverage_gap',
                    'gap': coverage_gap,
                    'description': f'{layer_name}层覆盖率差距{coverage_gap}个百分点',
                    'priority': target['priority']
                })

            # 识别测试失败的层
            if layer_data['test_failed'] > 0 or layer_data['test_error'] > 0:
                bottlenecks.append({
                    'layer': layer_name,
                    'type': 'test_failure',
                    'failed': layer_data['test_failed'],
                    'error': layer_data['test_error'],
                    'description': f'{layer_name}层有{layer_data["test_failed"]}个失败测试，{layer_data["test_error"]}个错误测试',
                    'priority': target['priority']
                })

        return bottlenecks

    def estimate_completion_date(self) -> str:
        """估算完成日期"""
        latest = self.get_latest_progress()
        if not latest:
            return "无法估算"

        remaining_days = 0
        total_completion = 0
        total_layers = 0

        for layer_name, layer_data in latest['layers'].items():
            completion = self.calculate_completion_percentage(layer_name)
            target = self.layer_targets[layer_name]

            if completion < 100:
                # 基于剩余工作量和历史进度估算
                remaining_work = (100 - completion) / 100
                estimated_days = target['estimated_days'] * remaining_work
                remaining_days = max(remaining_days, estimated_days)

            total_completion += completion
            total_layers += 1

        if total_layers > 0:
            avg_completion = total_completion / total_layers
            if avg_completion < 50:
                remaining_days = max(remaining_days, 10)  # 至少需要10天

        if remaining_days > 0:
            completion_date = datetime.datetime.now() + datetime.timedelta(days=remaining_days)
            return completion_date.strftime('%Y-%m-%d')
        else:
            return "已完成"

    def generate_progress_report(self) -> str:
        """生成进度报告"""
        report = []
        report.append("# RQA2025 模型落地进度报告")
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        latest = self.get_latest_progress()
        if not latest:
            report.append("暂无进度数据")
            return "\n".join(report)

        # 总体进度
        total_completion = 0
        total_layers = 0

        report.append("## 各层完成情况")
        for layer_name, layer_data in latest['layers'].items():
            completion = self.calculate_completion_percentage(layer_name)
            target = self.layer_targets[layer_name]

            total_completion += completion
            total_layers += 1

            status_emoji = "✅" if completion >= 100 else "🟡" if completion >= 50 else "🔴"

            report.append(f"### {status_emoji} {layer_name.capitalize()}层")
            report.append(f"- **完成度**: {completion:.1f}%")
            report.append(
                f"- **覆盖率**: {layer_data['coverage']:.1f}% / {target['target_coverage']}%")
            report.append(f"- **测试通过**: {layer_data['test_passed']}")
            report.append(f"- **测试失败**: {layer_data['test_failed']}")
            report.append(f"- **测试错误**: {layer_data['test_error']}")
            report.append(f"- **优先级**: {target['priority']}")
            report.append("")

        # 总体统计
        if total_layers > 0:
            avg_completion = total_completion / total_layers
            report.append("## 总体统计")
            report.append(f"- **平均完成度**: {avg_completion:.1f}%")
            report.append(f"- **预计完成日期**: {self.estimate_completion_date()}")
            report.append("")

        # 瓶颈分析
        bottlenecks = self.identify_bottlenecks()
        if bottlenecks:
            report.append("## 瓶颈识别")
            for bottleneck in bottlenecks:
                priority_emoji = {
                    'highest': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(bottleneck['priority'], '🟡')

                report.append(f"- {priority_emoji} {bottleneck['description']}")
            report.append("")

        # 趋势分析
        report.append("## 趋势分析")
        if len(self.progress_history) >= 2:
            # 计算最近两次的进度变化
            prev_entry = self.progress_history[-2]
            for layer_name in latest['layers']:
                if layer_name in prev_entry['layers']:
                    prev_coverage = prev_entry['layers'][layer_name]['coverage']
                    curr_coverage = latest['layers'][layer_name]['coverage']
                    change = curr_coverage - prev_coverage

                    if change > 0:
                        report.append(f"- {layer_name}层覆盖率提升 {change:.1f}%")
                    elif change < 0:
                        report.append(f"- {layer_name}层覆盖率下降 {abs(change):.1f}%")
                    else:
                        report.append(f"- {layer_name}层覆盖率无变化")
        else:
            report.append("- 数据不足，无法进行趋势分析")

        return "\n".join(report)

    def save_progress_report(self):
        """保存进度报告"""
        report_content = self.generate_progress_report()
        report_file = self.project_root / "docs" / "progress_report.md"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"进度报告已保存: {report_file}")
        except Exception as e:
            logger.error(f"保存进度报告失败: {e}")

    def create_progress_chart(self):
        """创建进度图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # 准备数据
            dates = []
            coverages = {}

            for entry in self.progress_history:
                date = datetime.datetime.fromisoformat(entry['timestamp'])
                dates.append(date)

                for layer_name, layer_data in entry['layers'].items():
                    if layer_name not in coverages:
                        coverages[layer_name] = []
                    coverages[layer_name].append(layer_data['coverage'])

            if not dates:
                logger.warning("没有足够的数据创建图表")
                return

            # 创建图表
            plt.figure(figsize=(12, 8))

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            for i, (layer_name, coverage_data) in enumerate(coverages.items()):
                color = colors[i % len(colors)]
                plt.plot(dates, coverage_data, marker='o',
                         label=layer_name, color=color, linewidth=2)

            plt.xlabel('日期')
            plt.ylabel('覆盖率 (%)')
            plt.title('RQA2025 模型落地进度跟踪')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 格式化x轴
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.xticks(rotation=45)

            # 保存图表
            chart_file = self.project_root / "reports" / "progress_chart.png"
            chart_file.parent.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"进度图表已保存: {chart_file}")

        except ImportError:
            logger.warning("matplotlib未安装，跳过图表生成")
        except Exception as e:
            logger.error(f"创建进度图表失败: {e}")


def main():
    """主函数"""
    tracker = ProgressTracker()

    # 示例：添加一些进度数据
    if len(tracker.progress_history) == 0:
        logger.info("添加示例进度数据...")

        # 模拟一些进度数据
        sample_results = [
            {
                'layer': 'infrastructure',
                'test_result': {'passed': 421, 'failed': 3, 'error': 0},
                'coverage_result': {'coverage': 36.30},
                'success': True
            },
            {
                'layer': 'data',
                'test_result': {'passed': 50, 'failed': 10, 'error': 5},
                'coverage_result': {'coverage': 11.97},
                'success': False
            },
            {
                'layer': 'features',
                'test_result': {'passed': 41, 'failed': 5, 'error': 0},
                'coverage_result': {'coverage': 45.0},
                'success': True
            },
            {
                'layer': 'models',
                'test_result': {'passed': 200, 'failed': 0, 'error': 0},
                'coverage_result': {'coverage': 82.0},
                'success': True
            },
            {
                'layer': 'trading',
                'test_result': {'passed': 30, 'failed': 15, 'error': 5},
                'coverage_result': {'coverage': 45.0},
                'success': False
            },
            {
                'layer': 'backtest',
                'test_result': {'passed': 20, 'failed': 10, 'error': 0},
                'coverage_result': {'coverage': 30.0},
                'success': True
            }
        ]

        tracker.add_progress_entry(sample_results)

    # 生成报告
    tracker.save_progress_report()
    tracker.create_progress_chart()

    print("✅ 进度跟踪完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
