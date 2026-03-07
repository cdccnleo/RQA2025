#!/usr/bin/env python3
"""
质量趋势报告生成器

生成代码质量趋势分析报告，包括：
- 质量评分趋势图
- 克隆组数量变化
- 复杂度指标变化
- 改进建议跟踪
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class QualityTrendReporter:
    """质量趋势报告生成器"""

    def __init__(self, reports_dir: str = "quality_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

    def generate_trend_report(self, days: int = 30) -> Dict[str, Any]:
        """
        生成质量趋势报告

        Args:
            days: 分析的时间范围（天数）

        Returns:
            Dict[str, Any]: 趋势分析报告
        """
        print(f"📊 生成{days}天质量趋势报告...")

        # 收集历史报告
        historical_reports = self._collect_historical_reports(days)

        if not historical_reports:
            print("⚠️ 未找到历史质量报告")
            return self._create_empty_report()

        # 分析趋势
        trend_analysis = self._analyze_trends(historical_reports)

        # 生成可视化图表
        charts_paths = self._generate_charts(trend_analysis)

        # 生成综合报告
        report = {
            'summary': {
                'analysis_period_days': days,
                'total_reports_analyzed': len(historical_reports),
                'generated_at': datetime.now().isoformat(),
                'trend_direction': trend_analysis['overall_trend']
            },
            'trend_analysis': trend_analysis,
            'charts': charts_paths,
            'recommendations': self._generate_trend_recommendations(trend_analysis)
        }

        return report

    def _collect_historical_reports(self, days: int) -> List[Dict[str, Any]]:
        """收集历史质量报告"""
        cutoff_date = datetime.now() - timedelta(days=days)
        historical_reports = []

        # 查找所有质量报告文件
        for file_path in self.reports_dir.glob("quality_report_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                report_date = datetime.fromisoformat(report['timestamp'])
                if report_date >= cutoff_date:
                    report['_file_date'] = report_date
                    historical_reports.append(report)

            except Exception as e:
                print(f"⚠️ 读取报告文件失败 {file_path}: {e}")

        # 按日期排序
        historical_reports.sort(key=lambda x: x['_file_date'])

        return historical_reports

    def _analyze_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析质量趋势"""
        if not reports:
            return {}

        # 提取时间序列数据
        dates = []
        quality_scores = []
        clone_counts = []
        refactoring_opportunities = []

        for report in reports:
            dates.append(report['_file_date'])
            summary = report.get('summary', {})
            duplicate_check = report.get('duplicate_check', {})

            quality_scores.append(summary.get('overall_quality_score', 0))
            clone_counts.append(duplicate_check.get('total_groups', 0))
            refactoring_opportunities.append(duplicate_check.get('refactoring_opportunities', 0))

        # 计算趋势
        trend_analysis = {
            'time_range': {
                'start': dates[0].isoformat() if dates else None,
                'end': dates[-1].isoformat() if dates else None,
                'data_points': len(dates)
            },
            'quality_score_trend': {
                'current': quality_scores[-1] if quality_scores else 0,
                'average': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'min': min(quality_scores) if quality_scores else 0,
                'max': max(quality_scores) if quality_scores else 0,
                'trend': self._calculate_trend(quality_scores),
                'change_percent': self._calculate_change_percentage(quality_scores)
            },
            'clone_count_trend': {
                'current': clone_counts[-1] if clone_counts else 0,
                'average': sum(clone_counts) / len(clone_counts) if clone_counts else 0,
                'min': min(clone_counts) if clone_counts else 0,
                'max': max(clone_counts) if clone_counts else 0,
                'trend': self._calculate_trend(clone_counts),
                'change_percent': self._calculate_change_percentage(clone_counts)
            },
            'refactoring_opportunities_trend': {
                'current': refactoring_opportunities[-1] if refactoring_opportunities else 0,
                'average': sum(refactoring_opportunities) / len(refactoring_opportunities) if refactoring_opportunities else 0,
                'trend': self._calculate_trend(refactoring_opportunities)
            },
            'overall_trend': self._determine_overall_trend(quality_scores, clone_counts)
        }

        return trend_analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return "insufficient_data"

        # 计算线性趋势
        n = len(values)
        if n < 2:
            return "stable"

        # 简单线性回归斜率
        x = list(range(n))
        slope = self._calculate_slope(x, values)

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """计算线性回归斜率"""
        n = len(x)
        if n < 2:
            return 0

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _calculate_change_percentage(self, values: List[float]) -> float:
        """计算变化百分比"""
        if len(values) < 2:
            return 0.0

        first_value = values[0]
        last_value = values[-1]

        if first_value == 0:
            return 0.0

        return ((last_value - first_value) / first_value) * 100

    def _determine_overall_trend(self, quality_scores: List[float],
                                 clone_counts: List[float]) -> str:
        """确定整体趋势"""
        quality_trend = self._calculate_trend(quality_scores)
        clone_trend = self._calculate_trend(clone_counts)

        if quality_trend == "improving" and clone_trend in ["improving", "stable"]:
            return "improving"
        elif quality_trend == "declining" or clone_trend == "declining":
            return "declining"
        else:
            return "stable"

    def _generate_charts(self, trend_analysis: Dict[str, Any]) -> Dict[str, str]:
        """生成趋势图表"""
        charts_dir = self.reports_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        chart_files = {}

        try:
            # 这里可以添加matplotlib图表生成代码
            # 暂时返回空字典
            pass

        except Exception as e:
            print(f"⚠️ 生成图表失败: {e}")

        return chart_files

    def _generate_trend_recommendations(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """生成趋势分析建议"""
        recommendations = []

        quality_trend = trend_analysis.get('quality_score_trend', {})
        clone_trend = trend_analysis.get('clone_count_trend', {})

        # 基于质量趋势的建议
        if quality_trend.get('trend') == 'improving':
            recommendations.append("✅ 质量趋势向好，继续保持当前改进节奏")
        elif quality_trend.get('trend') == 'declining':
            recommendations.append("🔴 质量评分下降，需要立即采取改进措施")
        else:
            recommendations.append("🟡 质量评分稳定，考虑进一步优化")

        # 基于克隆趋势的建议
        if clone_trend.get('trend') == 'declining':
            recommendations.append("✅ 克隆组数量减少，重复代码清理效果良好")
        elif clone_trend.get('trend') == 'improving':
            recommendations.append("🟡 克隆组数量增加，需要加强重复代码管理")
        else:
            recommendations.append("🟡 克隆组数量稳定，继续监控变化")

        # 基于当前值的建议
        current_quality = quality_trend.get('current', 0)
        current_clones = clone_trend.get('current', 0)

        if current_quality < 0.7:
            recommendations.append("🔴 质量评分偏低，建议优先处理高影响问题")
        if current_clones > 50:
            recommendations.append("🟡 克隆组数量较多，建议安排专门的重构任务")

        return recommendations

    def _create_empty_report(self) -> Dict[str, Any]:
        """创建空报告"""
        return {
            'summary': {
                'analysis_period_days': 0,
                'total_reports_analyzed': 0,
                'generated_at': datetime.now().isoformat(),
                'trend_direction': 'no_data'
            },
            'trend_analysis': {},
            'charts': {},
            'recommendations': ['暂无历史数据，请运行更多质量检查']
        }

    def save_report(self, report: Dict[str, Any], output_path: str):
        """保存趋势报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"📄 趋势报告已保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="质量趋势报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/generate_quality_trend_report.py
  python scripts/generate_quality_trend_report.py --days 90 --output trend_report.json
        """
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=30,
        help='分析的时间范围（天数，默认30天）'
    )

    parser.add_argument(
        '--output', '-o',
        help='输出报告文件路径'
    )

    parser.add_argument(
        '--reports-dir',
        default='quality_reports',
        help='质量报告存储目录'
    )

    args = parser.parse_args()

    # 创建趋势报告生成器
    reporter = QualityTrendReporter(args.reports_dir)

    # 生成趋势报告
    report = reporter.generate_trend_report(args.days)

    # 保存报告
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'quality_trend_report_{timestamp}.json'

    reporter.save_report(report, output_path)

    # 打印摘要
    summary = report['summary']
    trend_analysis = report.get('trend_analysis', {})

    print(f"\n📊 质量趋势报告摘要")
    print(f"分析周期: {summary['analysis_period_days']}天")
    print(f"报告数量: {summary['total_reports_analyzed']}个")
    print(f"整体趋势: {summary['trend_direction']}")

    if trend_analysis:
        quality = trend_analysis.get('quality_score_trend', {})
        clones = trend_analysis.get('clone_count_trend', {})

        print(f"当前质量评分: {quality.get('current', 0):.3f}")
        print(f"质量变化: {quality.get('change_percent', 0):+.1f}%")
        print(f"当前克隆组数: {clones.get('current', 0)}")
        print(f"克隆变化: {clones.get('change_percent', 0):+.1f}%")

    print(f"\n💡 关键建议:")
    for rec in report.get('recommendations', []):
        print(f"  • {rec}")

    print(f"\n📄 详细报告已保存至: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
