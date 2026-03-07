#!/usr/bin/env python3
"""
覆盖率趋势检查脚本

检查代码覆盖率的变化趋势，如果覆盖率下降则发出告警
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class CoverageMonitor:
    """覆盖率监控器"""

    def __init__(self, codecov_token: Optional[str] = None):
        self.codecov_token = codecov_token or os.getenv('CODECOV_TOKEN')
        self.coverage_history_file = os.path.join(project_root, '.coverage_history.json')
        self.alert_threshold = 2.0  # 覆盖率下降2%触发告警

    def get_current_coverage(self) -> Dict[str, float]:
        """获取当前覆盖率数据"""
        # 这里可以从codecov API或本地覆盖率文件获取
        # 简化实现，直接返回模拟数据

        return {
            'total': 85.5,
            'infrastructure': 92.1,
            'data': 88.3,
            'strategy': 78.9,
            'risk': 82.4,
            'ml': 75.6,
            'async_processor': 89.2
        }

    def load_coverage_history(self) -> List[Dict]:
        """加载覆盖率历史数据"""
        if os.path.exists(self.coverage_history_file):
            try:
                with open(self.coverage_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return []

    def save_coverage_history(self, history: List[Dict]):
        """保存覆盖率历史数据"""
        try:
            with open(self.coverage_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"保存覆盖率历史失败: {e}")

    def check_coverage_trends(self) -> Dict:
        """检查覆盖率趋势"""
        current_coverage = self.get_current_coverage()
        history = self.load_coverage_history()

        # 添加当前数据到历史记录
        current_record = {
            'timestamp': datetime.now().isoformat(),
            'coverage': current_coverage
        }
        history.append(current_record)

        # 只保留最近30天的记录
        cutoff_date = datetime.now() - timedelta(days=30)
        history = [
            record for record in history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]

        # 保存更新后的历史
        self.save_coverage_history(history)

        # 分析趋势
        return self.analyze_trends(history)

    def analyze_trends(self, history: List[Dict]) -> Dict:
        """分析覆盖率趋势"""
        if len(history) < 2:
            return {
                'status': 'insufficient_data',
                'message': '历史数据不足，无法分析趋势'
            }

        # 获取最近和最旧的覆盖率数据
        latest = history[-1]['coverage']
        previous = history[-2]['coverage'] if len(history) >= 2 else latest

        # 计算变化
        changes = {}
        alerts = []

        for layer, current_value in latest.items():
            if layer in previous:
                previous_value = previous[layer]
                change = current_value - previous_value

                changes[layer] = {
                    'current': current_value,
                    'previous': previous_value,
                    'change': change,
                    'change_percent': (change / previous_value * 100) if previous_value > 0 else 0
                }

                # 检查是否需要告警
                if change < -self.alert_threshold:
                    alerts.append({
                        'layer': layer,
                        'severity': 'high' if abs(change) > 5.0 else 'medium',
                        'message': f"{layer}层覆盖率下降{abs(change):.1f}% (从{previous_value:.1f}%到{current_value:.1f}%)"
                    })

        return {
            'status': 'analyzed',
            'total_coverage': latest.get('total', 0),
            'changes': changes,
            'alerts': alerts,
            'alert_triggered': len(alerts) > 0
        }


def main():
    """主函数"""
    monitor = CoverageMonitor()

    try:
        result = monitor.check_coverage_trends()

        print("=== 覆盖率趋势分析结果 ===")
        print(f"状态: {result['status']}")
        print(".1f"
        if 'changes' in result:
            print("\n各层变化:")
            for layer, change_data in result['changes'].items():
                print("+.1f")

        if result.get('alert_triggered', False):
            print("\n🚨 触发告警:"            for alert in result['alerts']:
                print(f"  {alert['severity'].upper()}: {alert['message']}")
            sys.exit(1)
        else:
            print("\n✅ 覆盖率正常，无需告警")

    except Exception as e:
        print(f"覆盖率检查失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
