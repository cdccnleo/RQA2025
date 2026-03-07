#!/usr/bin/env python3
"""
RQA2025 覆盖率监控系统
持续监控和分析测试覆盖率趋势
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CoverageMonitor:
    """覆盖率监控器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.history_file = project_root / "coverage_history.json"
        self.reports_dir = project_root / "coverage_reports"
        self.reports_dir.mkdir(exist_ok=True)

        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def load_history(self) -> List[Dict[str, Any]]:
        """加载历史数据"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_history(self, history: List[Dict[str, Any]]):
        """保存历史数据"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def record_coverage(self, coverage_data: Dict[str, Any]):
        """记录覆盖率数据"""
        history = self.load_history()

        record = {
            "timestamp": datetime.now().isoformat(),
            "commit_sha": os.getenv("GITHUB_SHA", "unknown"),
            "run_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
            "branch": os.getenv("GITHUB_REF", "local"),
            "coverage": coverage_data
        }

        history.append(record)

        # 保留最近100条记录
        if len(history) > 100:
            history = history[-100:]

        self.save_history(history)
        print(f"📊 覆盖率记录已保存: {record['timestamp']}")

    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """分析覆盖率趋势"""
        history = self.load_history()

        if not history:
            return {"error": "没有历史数据"}

        # 转换为DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # 过滤最近N天的数据
        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = df[df.index >= cutoff_date]

        if df_recent.empty:
            return {"error": f"最近{days}天没有数据"}

        analysis = {
            "period_days": days,
            "total_records": len(df_recent),
            "coverage_trend": {
                "current": df_recent.iloc[-1]['coverage'].get('weighted_coverage', 0),
                "average": df_recent['coverage'].apply(lambda x: x.get('weighted_coverage', 0)).mean(),
                "min": df_recent['coverage'].apply(lambda x: x.get('weighted_coverage', 0)).min(),
                "max": df_recent['coverage'].apply(lambda x: x.get('weighted_coverage', 0)).max(),
                "trend": self.calculate_trend(df_recent)
            },
            "pass_rate_trend": {
                "current": df_recent.iloc[-1]['coverage'].get('pass_rate', 0),
                "average": df_recent['coverage'].apply(lambda x: x.get('pass_rate', 0)).mean(),
                "improvement": self.calculate_improvement(df_recent, 'pass_rate')
            }
        }

        return analysis

    def calculate_trend(self, df: pd.DataFrame) -> str:
        """计算趋势"""
        coverage_values = df['coverage'].apply(lambda x: x.get('weighted_coverage', 0))

        if len(coverage_values) < 2:
            return "insufficient_data"

        # 线性回归计算趋势
        x = range(len(coverage_values))
        y = coverage_values.values

        if len(x) > 1:
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            if slope > 0.1:
                return "improving"
            elif slope < -0.1:
                return "declining"
            else:
                return "stable"
        else:
            return "stable"

    def calculate_improvement(self, df: pd.DataFrame, metric: str) -> float:
        """计算改进幅度"""
        values = df['coverage'].apply(lambda x: x.get(metric, 0))

        if len(values) < 2:
            return 0.0

        return values.iloc[-1] - values.iloc[0]

    def generate_trend_chart(self, days: int = 30):
        """生成趋势图表"""
        history = self.load_history()

        if not history:
            print("❌ 没有历史数据")
            return

        # 转换为DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # 过滤最近N天
        cutoff_date = datetime.now() - timedelta(days=days)
        df_recent = df[df.index >= cutoff_date]

        if df_recent.empty:
            print(f"❌ 最近{days}天没有数据")
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 覆盖率趋势
        coverage_values = df_recent['coverage'].apply(lambda x: x.get('weighted_coverage', 0))
        ax1.plot(df_recent.index, coverage_values, 'b-o', linewidth=2, markersize=4)
        ax1.set_title(f'RQA2025 覆盖率趋势 (最近{days}天)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('覆盖率 (%)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='目标: 70%')
        ax1.legend()

        # 通过率趋势
        pass_rate_values = df_recent['coverage'].apply(lambda x: x.get('pass_rate', 0))
        ax2.plot(df_recent.index, pass_rate_values, 'g-s', linewidth=2, markersize=4)
        ax2.set_title(f'测试通过率趋势 (最近{days}天)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('通过率 (%)', fontsize=12)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=98, color='r', linestyle='--', alpha=0.7, label='目标: 98%')
        ax2.legend()

        plt.tight_layout()

        # 保存图表
        chart_path = self.reports_dir / f"coverage_trend_{days}days.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📈 趋势图表已保存: {chart_path}")

        plt.close()

    def generate_report(self, days: int = 30) -> Dict[str, Any]:
        """生成监控报告"""
        analysis = self.analyze_trends(days)

        if "error" in analysis:
            return analysis

        # 生成图表
        self.generate_trend_chart(days)

        report = {
            "report_type": "RQA2025 Coverage Monitoring Report",
            "generated_at": datetime.now().isoformat(),
            "monitoring_period_days": days,
            "analysis": analysis,
            "recommendations": self.generate_recommendations(analysis),
            "charts_generated": True
        }

        # 保存报告
        report_path = self.reports_dir / f"coverage_monitor_report_{days}days.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 监控报告已保存: {report_path}")

        return report

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        coverage_trend = analysis.get("coverage_trend", {})
        pass_rate_trend = analysis.get("pass_rate_trend", {})

        current_coverage = coverage_trend.get("current", 0)
        current_pass_rate = pass_rate_trend.get("current", 0)

        if coverage_trend.get("trend") == "declining":
            recommendations.append("⚠️ 覆盖率下降：需要增加测试覆盖")

        if current_coverage < 70:
            recommendations.append(f"🎯 距离70%目标还差{70-current_coverage:.1f}个百分点")

        if current_pass_rate < 98:
            recommendations.append("🔧 通过率不足：需要修复失败的测试")

        if coverage_trend.get("trend") == "improving" and current_coverage >= 70:
            recommendations.append("✅ 覆盖率稳步提升：保持当前节奏")

        if not recommendations:
            recommendations.append("🎉 覆盖率监控正常：继续保持高质量")

        return recommendations


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025 覆盖率监控")
    parser.add_argument("--record", action="store_true", help="记录当前覆盖率")
    parser.add_argument("--analyze", type=int, default=30, help="分析最近N天的趋势")
    parser.add_argument("--report", action="store_true", help="生成监控报告")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    monitor = CoverageMonitor(project_root)

    if args.record:
        # 记录示例数据（实际使用时从pytest-cov获取）
        sample_coverage = {
            "weighted_coverage": 71.8,
            "pass_rate": 98.4,
            "total_tests": 5772,
            "passed": 5675,
            "failed": 97
        }
        monitor.record_coverage(sample_coverage)
        print("✅ 覆盖率数据已记录")

    if args.report or args.analyze:
        report = monitor.generate_report(args.analyze)

        print("\n📊 覆盖率监控报告")
        print(f"📅 监控周期: {args.analyze}天")
        print(f"🎯 当前覆盖率: {report.get('analysis', {}).get('coverage_trend', {}).get('current', 0):.1f}%")
        print(f"📈 通过率: {report.get('analysis', {}).get('pass_rate_trend', {}).get('current', 0):.1f}%")

        print("\n💡 建议:")
        for rec in report.get('recommendations', []):
            print(f"  • {rec}")


if __name__ == "__main__":
    main()