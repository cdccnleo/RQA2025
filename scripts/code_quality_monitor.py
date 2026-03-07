#!/usr/bin/env python3
"""
代码质量监控脚本

定期检查代码质量，生成质量报告并监控趋势变化。
用于防止代码质量回退，持续维护高质量代码。

作者：AI Assistant
版本：1.0
更新日期：2025-10-27
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 项目根目录
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ai_intelligent_code_analyzer import IntelligentCodeAnalyzer


class CodeQualityMonitor:
    """
    代码质量监控器

    提供定期质量检查、趋势分析和报告生成功能。
    """

    def __init__(self, target_path: str, reports_dir: str = "test_logs"):
        self.target_path = target_path
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)

        self.analyzer = IntelligentCodeAnalyzer()
        self.history_file = self.reports_dir / "quality_history.json"

    def run_quality_check(self) -> Dict[str, Any]:
        """
        执行代码质量检查

        Returns:
            Dict[str, Any]: 检查结果
        """
        print("🔍 开始代码质量检查...")

        # 执行分析
        result = self.analyzer.analyze_project(self.target_path, deep_analysis=True)

        # 保存结果
        report_file = self.reports_dir / f"quality_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': result.timestamp.isoformat(),
                'metrics': result.metrics,
                'quality_score': result.quality_score,
                'overall_score': result.overall_score,
                'risk_assessment': result.risk_assessment,
                'opportunities_count': len(result.opportunities),
                'organization_score': result.organization_analysis.quality_score if result.organization_analysis else None
            }, f, indent=2, ensure_ascii=False)

        # 更新历史记录
        self._update_history(result)

        print(f"✅ 质量检查完成，报告已保存到: {report_file}")
        return result

    def _update_history(self, result):
        """更新质量历史记录"""
        history = self._load_history()

        record = {
            'timestamp': result.timestamp.isoformat(),
            'quality_score': result.quality_score,
            'overall_score': result.overall_score,
            'risk_level': result.risk_assessment.get('overall_risk'),
            'opportunities_count': len(result.opportunities),
            'total_files': result.metrics.get('total_files', 0),
            'total_lines': result.metrics.get('total_lines', 0),
            'organization_score': result.organization_analysis.quality_score if result.organization_analysis else None
        }

        history.append(record)

        # 只保留最近50条记录
        if len(history) > 50:
            history = history[-50:]

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def _load_history(self) -> List[Dict[str, Any]]:
        """加载历史记录"""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def generate_trend_report(self, days: int = 30) -> str:
        """
        生成趋势报告

        Args:
            days: 分析最近多少天的趋势

        Returns:
            str: 趋势报告内容
        """
        history = self._load_history()
        if not history:
            return "暂无历史数据，无法生成趋势报告"

        report = []
        report.append("# 📊 代码质量趋势报告")
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"分析时间范围: 最近{days}天")
        report.append("")

        # 显示最新数据
        if history:
            latest = history[-1]
            report.append("## 🎯 当前质量指标")
            report.append(f"- 综合评分: {latest.get('overall_score', 0):.3f}")
            report.append(f"- 代码质量评分: {latest.get('quality_score', 0):.3f}")
            report.append(f"- 风险等级: {latest.get('risk_level', 'unknown')}")
            report.append(f"- 重构机会: {latest.get('opportunities_count', 0)}")
            report.append("")

        # 统计信息
        report.append("## 📈 统计信息")
        report.append(f"- 历史记录数: {len(history)}")
        if history:
            report.append(f"- 总文件数: {latest.get('total_files', 0)}")
            report.append(f"- 总代码行: {latest.get('total_lines', 0)}")
        report.append("")

        return "\n".join(report)

    def check_quality_thresholds(self, thresholds: Dict[str, float] = None) -> List[str]:
        """
        检查质量阈值

        Args:
            thresholds: 质量阈值配置

        Returns:
            List[str]: 告警信息列表
        """
        if thresholds is None:
            thresholds = {
                'min_quality_score': 0.8,
                'min_overall_score': 0.85,
                'max_opportunities': 300
            }

        history = self._load_history()
        if not history:
            return ["暂无历史数据"]

        latest = history[-1]
        alerts = []

        quality_score = latest.get('quality_score', 0)
        overall_score = latest.get('overall_score', 0)
        opportunities_count = latest.get('opportunities_count', 0)
        risk_level = latest.get('risk_level')

        if quality_score < thresholds['min_quality_score']:
            alerts.append(f"⚠️ 代码质量评分过低: {quality_score:.3f} < {thresholds['min_quality_score']}")

        if overall_score < thresholds['min_overall_score']:
            alerts.append(f"⚠️ 综合质量评分过低: {overall_score:.3f} < {thresholds['min_overall_score']}")

        if opportunities_count > thresholds['max_opportunities']:
            alerts.append(f"⚠️ 重构机会过多: {opportunities_count} > {thresholds['max_opportunities']}")

        if risk_level in ['high', 'very_high']:
            alerts.append(f"⚠️ 风险等级过高: {risk_level}")

        if not alerts:
            alerts.append("✅ 所有质量指标均在阈值范围内")

        return alerts


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="代码质量监控工具")
    parser.add_argument('target', help='监控目标路径')
    parser.add_argument('--check', action='store_true', help='执行质量检查')
    parser.add_argument('--trend', type=int, default=30, help='生成趋势报告（天数）')
    parser.add_argument('--alerts', action='store_true', help='检查质量告警')
    parser.add_argument('--reports-dir', default='test_logs', help='报告存储目录')

    args = parser.parse_args()

    monitor = CodeQualityMonitor(args.target, args.reports_dir)

    if args.check:
        result = monitor.run_quality_check()
        print(f"质量评分: {result.quality_score:.3f}")
        print(f"综合评分: {result.overall_score:.3f}")
        print(f"风险等级: {result.risk_assessment.get('overall_risk')}")

    if args.trend:
        report = monitor.generate_trend_report(args.trend)
        print(report)

    if args.alerts:
        alerts = monitor.check_quality_thresholds()
        print("\n".join(alerts))


if __name__ == '__main__':
    main()
