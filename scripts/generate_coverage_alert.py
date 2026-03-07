#!/usr/bin/env python3
"""
生成覆盖率告警脚本

当覆盖率下降时生成详细的告警报告
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class CoverageAlertGenerator:
    """覆盖率告警生成器"""

    def __init__(self):
        self.alert_file = os.path.join(project_root, 'coverage_alert.json')
        self.report_file = os.path.join(project_root, 'coverage_alert_report.md')

    def generate_alert(self, coverage_data: Dict) -> Dict:
        """生成覆盖率告警"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'coverage_regression',
            'severity': self._determine_severity(coverage_data),
            'summary': self._generate_summary(coverage_data),
            'details': coverage_data,
            'recommendations': self._generate_recommendations(coverage_data)
        }

        # 保存告警到文件
        self._save_alert(alert)

        # 生成报告
        self._generate_report(alert)

        return alert

    def _determine_severity(self, coverage_data: Dict) -> str:
        """确定告警严重程度"""
        alerts = coverage_data.get('alerts', [])

        if not alerts:
            return 'low'

        # 检查是否有高严重程度告警
        high_severity_count = sum(1 for alert in alerts if alert.get('severity') == 'high')

        if high_severity_count > 0:
            return 'high'
        elif len(alerts) > 2:
            return 'medium'
        else:
            return 'low'

    def _generate_summary(self, coverage_data: Dict) -> str:
        """生成告警摘要"""
        alerts = coverage_data.get('alerts', [])
        total_coverage = coverage_data.get('total_coverage', 0)

        if not alerts:
            return f"覆盖率正常: {total_coverage:.1f}%"

        alert_count = len(alerts)
        layers_affected = [alert['layer'] for alert in alerts]

        return f"覆盖率下降告警: {alert_count}个层级受影响 - {', '.join(layers_affected)}"

    def _generate_recommendations(self, coverage_data: Dict) -> List[str]:
        """生成建议措施"""
        alerts = coverage_data.get('alerts', [])
        recommendations = []

        if not alerts:
            return ["保持当前测试覆盖率水平"]

        # 基于受影响的层级生成建议
        affected_layers = set(alert['layer'] for alert in alerts)

        for layer in affected_layers:
            if layer == 'strategy':
                recommendations.extend([
                    f"为策略层({layer})添加更多边界条件测试",
                    f"为策略层({layer})添加异常处理测试",
                    "检查策略层的参数验证逻辑测试覆盖"
                ])
            elif layer == 'risk':
                recommendations.extend([
                    f"为风险管理层({layer})添加压力测试场景",
                    f"为风险管理层({layer})添加VaR计算测试",
                    "检查风险层的合规性测试覆盖"
                ])
            elif layer == 'ml':
                recommendations.extend([
                    f"为机器学习层({layer})添加模型验证测试",
                    f"为机器学习层({layer})添加特征工程测试",
                    "检查ML层的数据预处理测试覆盖"
                ])
            else:
                recommendations.append(f"为{layer}层添加更多单元测试")

        # 通用建议
        recommendations.extend([
            "运行完整测试套件检查是否有测试失败",
            "审查最近的代码更改是否引入了未测试的代码路径",
            "考虑添加集成测试以提高覆盖率"
        ])

        return recommendations

    def _save_alert(self, alert: Dict):
        """保存告警到文件"""
        try:
            with open(self.alert_file, 'w', encoding='utf-8') as f:
                json.dump(alert, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"保存告警文件失败: {e}")

    def _generate_report(self, alert: Dict):
        """生成告警报告"""
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write("# 覆盖率告警报告\n\n")
                f.write(f"**生成时间:** {alert['timestamp']}\n")
                f.write(f"**告警类型:** {alert['alert_type']}\n")
                f.write(f"**严重程度:** {alert['severity'].upper()}\n")
                f.write(f"**摘要:** {alert['summary']}\n\n")

                if 'details' in alert and 'alerts' in alert['details']:
                    f.write("## 详细告警\n\n")
                    for alert_item in alert['details']['alerts']:
                        f.write(f"- **{alert_item['severity'].upper()}**: {alert_item['message']}\n")

                if alert.get('recommendations'):
                    f.write("\n## 建议措施\n\n")
                    for i, rec in enumerate(alert['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")

                f.write("\n## 覆盖率变化详情\n\n")
                if 'details' in alert and 'changes' in alert['details']:
                    changes = alert['details']['changes']
                    f.write("| 层级 | 当前覆盖率 | 之前覆盖率 | 变化 | 变化百分比 |\n")
                    f.write("|------|----------|----------|------|----------|\n")

                    for layer, change_data in changes.items():
                        f.write(f"| {layer} | {change_data['current']:.1f}% | {change_data['previous']:.1f}% | {change_data['change']:+.1f}% | {change_data['change_percent']:+.1f}% |\n")

        except IOError as e:
            print(f"生成报告失败: {e}")


def main():
    """主函数"""
    # 从环境变量或命令行参数获取覆盖率数据
    # 这里简化处理，实际应该从check_coverage_trends.py的输出获取

    sample_coverage_data = {
        'total_coverage': 85.5,
        'alerts': [
            {
                'layer': 'strategy',
                'severity': 'high',
                'message': 'strategy层覆盖率下降3.2% (从82.1%到78.9%)'
            },
            {
                'layer': 'ml',
                'severity': 'medium',
                'message': 'ml层覆盖率下降1.8% (从77.4%到75.6%)'
            }
        ],
        'changes': {
            'strategy': {'current': 78.9, 'previous': 82.1, 'change': -3.2, 'change_percent': -3.9},
            'ml': {'current': 75.6, 'previous': 77.4, 'change': -1.8, 'change_percent': -2.3}
        }
    }

    generator = CoverageAlertGenerator()
    alert = generator.generate_alert(sample_coverage_data)

    print("=== 覆盖率告警生成完成 ===")
    print(f"告警文件: {generator.alert_file}")
    print(f"报告文件: {generator.report_file}")
    print(f"严重程度: {alert['severity']}")
    print(f"告警数量: {len(alert['details'].get('alerts', []))}")


if __name__ == "__main__":
    main()
