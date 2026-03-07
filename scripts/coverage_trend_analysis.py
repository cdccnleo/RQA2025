#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试覆盖率趋势分析
监控RQA2025量化交易系统测试覆盖率变化趋势
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoverageTrendAnalyzer:
    """测试覆盖率趋势分析器"""

    def __init__(self, data_dir: str = "reports/coverage_history"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 趋势分析阈值
        self.decline_threshold = 2.0  # 2%下降触发警告
        self.critical_decline = 5.0   # 5%下降触发严重警告

        # 目标覆盖率
        self.target_coverage = 90.0
        self.minimum_coverage = 80.0

    def record_coverage_data(self, coverage_data: Dict[str, Any]) -> None:
        """记录覆盖率数据"""
        timestamp = datetime.now()
        filename = f"coverage_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename

        # 添加时间戳
        coverage_data['timestamp'] = timestamp.isoformat()
        coverage_data['date'] = timestamp.strftime('%Y-%m-%d')
        coverage_data['time'] = timestamp.strftime('%H:%M:%S')

        # 保存数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(coverage_data, f, indent=2, ensure_ascii=False)

        logger.info(f"📊 覆盖率数据已记录: {filepath}")

    def load_historical_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """加载历史数据"""
        cutoff_date = datetime.now() - timedelta(days=days)
        historical_data = []

        for filepath in self.data_dir.glob("coverage_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查时间戳
                if 'timestamp' in data:
                    data_time = datetime.fromisoformat(data['timestamp'])
                    if data_time >= cutoff_date:
                        historical_data.append(data)
                else:
                    # 从文件名提取时间戳
                    filename = filepath.stem
                    if filename.startswith('coverage_'):
                        try:
                            time_str = filename[9:]  # 去掉'coverage_'前缀
                            data_time = datetime.strptime(time_str, '%Y%m%d_%H%M%S')
                            if data_time >= cutoff_date:
                                data['timestamp'] = data_time.isoformat()
                                historical_data.append(data)
                        except ValueError:
                            continue
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"无法加载历史数据文件 {filepath}: {e}")
                continue

        # 按时间排序
        historical_data.sort(key=lambda x: x.get('timestamp', ''))
        logger.info(f"📈 加载了 {len(historical_data)} 条历史记录")
        return historical_data

    def analyze_coverage_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析覆盖率趋势"""
        if len(historical_data) < 2:
            return {
                'trend': 'INSUFFICIENT_DATA',
                'message': '历史数据不足，无法分析趋势',
                'data_points': len(historical_data)
            }

        # 提取覆盖率数据
        coverage_points = []
        for data in historical_data:
            if 'enforcement_result' in data:
                coverage = data['enforcement_result'].get('overall_coverage', 0)
            elif 'totals' in data:
                coverage = data['totals'].get('percent_covered', 0)
            else:
                coverage = data.get('overall_coverage', 0)

            coverage_points.append({
                'timestamp': data.get('timestamp'),
                'coverage': coverage
            })

        # 计算趋势
        trend_analysis = self._calculate_trend(coverage_points)

        # 检测异常
        anomalies = self._detect_anomalies(coverage_points)

        # 生成预测
        prediction = self._predict_future_coverage(coverage_points)

        return {
            'trend': trend_analysis['direction'],
            'trend_slope': trend_analysis['slope'],
            'current_coverage': coverage_points[-1]['coverage'],
            'previous_coverage': coverage_points[-2]['coverage'] if len(coverage_points) >= 2 else 0,
            'change': coverage_points[-1]['coverage'] - coverage_points[-2]['coverage'] if len(coverage_points) >= 2 else 0,
            'data_points': len(coverage_points),
            'period_days': len(historical_data),
            'anomalies': anomalies,
            'prediction': prediction,
            'alerts': self._generate_trend_alerts(trend_analysis, coverage_points)
        }

    def _calculate_trend(self, coverage_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算趋势"""
        if len(coverage_points) < 2:
            return {'direction': 'STABLE', 'slope': 0.0}

        # 简单线性回归计算趋势
        n = len(coverage_points)
        x_values = [float(i) for i in range(n)]
        y_values = [float(point['coverage']) for point in coverage_points]

        # 计算斜率
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # 判断趋势方向
        if abs(slope) < 0.1:
            direction = 'STABLE'
        elif slope > 0:
            direction = 'IMPROVING'
        else:
            direction = 'DECLINING'

        return {
            'direction': direction,
            'slope': round(slope, 3),
            'correlation': self._calculate_correlation(x_values, y_values)
        }

    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """计算相关系数"""
        n = len(x_values)
        if n < 2:
            return 0.0

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        x_variance = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        y_variance = sum((y_values[i] - y_mean) ** 2 for i in range(n))

        denominator = (x_variance * y_variance) ** 0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _detect_anomalies(self, coverage_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测异常值"""
        anomalies = []

        if len(coverage_points) < 3:
            return anomalies

        # 计算移动平均和标准差
        window_size = min(5, len(coverage_points) // 2)

        for i in range(window_size, len(coverage_points)):
            window_data = [coverage_points[j]['coverage'] for j in range(i - window_size, i)]
            window_mean = sum(window_data) / len(window_data)
            window_std = (sum((x - window_mean) ** 2 for x in window_data) /
                          len(window_data)) ** 0.5

            current_coverage = coverage_points[i]['coverage']

            # 检测异常（超过2个标准差）
            if abs(current_coverage - window_mean) > 2 * window_std:
                anomalies.append({
                    'timestamp': coverage_points[i]['timestamp'],
                    'coverage': current_coverage,
                    'expected_range': [window_mean - 2 * window_std, window_mean + 2 * window_std],
                    'deviation': abs(current_coverage - window_mean),
                    'type': 'SUDDEN_DROP' if current_coverage < window_mean else 'SUDDEN_INCREASE'
                })

        return anomalies

    def _predict_future_coverage(self, coverage_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测未来覆盖率"""
        if len(coverage_points) < 3:
            return {'prediction': 'INSUFFICIENT_DATA'}

        # 使用最近的趋势预测
        recent_points = coverage_points[-5:]  # 最近5个数据点
        trend = self._calculate_trend(recent_points)

        current_coverage = coverage_points[-1]['coverage']
        predicted_coverage = current_coverage + trend['slope'] * 7  # 预测一周后

        # 预测目标达成时间
        if trend['slope'] > 0 and current_coverage < self.target_coverage:
            days_to_target = (self.target_coverage - current_coverage) / trend['slope']
            target_date = datetime.now() + timedelta(days=days_to_target)
        else:
            days_to_target = None
            target_date = None

        return {
            'predicted_coverage_7days': round(predicted_coverage, 2),
            'trend_direction': trend['direction'],
            'days_to_target': round(days_to_target) if days_to_target else None,
            'target_date': target_date.strftime('%Y-%m-%d') if target_date else None,
            'confidence': 'HIGH' if abs(trend['correlation']) > 0.7 else 'MEDIUM' if abs(trend['correlation']) > 0.3 else 'LOW'
        }

    def _generate_trend_alerts(self, trend_analysis: Dict[str, Any],
                               coverage_points: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """生成趋势告警"""
        alerts = []

        if len(coverage_points) < 2:
            return alerts

        current_coverage = coverage_points[-1]['coverage']
        previous_coverage = coverage_points[-2]['coverage']
        change = current_coverage - previous_coverage

        # 覆盖率下降告警
        if change <= -self.critical_decline:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'SEVERE_DECLINE',
                'message': f'覆盖率严重下降 {abs(change):.1f}%',
                'action': '立即调查原因并采取补救措施'
            })
        elif change <= -self.decline_threshold:
            alerts.append({
                'level': 'WARNING',
                'type': 'COVERAGE_DECLINE',
                'message': f'覆盖率下降 {abs(change):.1f}%',
                'action': '监控趋势并考虑改进措施'
            })

        # 低于最低要求告警
        if current_coverage < self.minimum_coverage:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'BELOW_MINIMUM',
                'message': f'覆盖率({current_coverage:.1f}%)低于最低要求({self.minimum_coverage}%)',
                'action': '阻止生产部署，立即提升覆盖率'
            })

        # 趋势持续下降告警
        if (trend_analysis['direction'] == 'DECLINING' and
            len(coverage_points) >= 3 and
            all(coverage_points[i]['coverage'] > coverage_points[i+1]['coverage']
                for i in range(len(coverage_points)-3, len(coverage_points)-1))):
            alerts.append({
                'level': 'WARNING',
                'type': 'SUSTAINED_DECLINE',
                'message': '覆盖率持续下降趋势',
                'action': '分析下降原因，制定改进计划'
            })

        return alerts

    def generate_trend_report(self, target_coverage: float = 90.0,
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成趋势报告"""
        self.target_coverage = target_coverage

        # 加载历史数据
        historical_data = self.load_historical_data()

        # 分析趋势
        trend_analysis = self.analyze_coverage_trends(historical_data)

        # 生成报告
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_period_days': 30,
                'target_coverage': self.target_coverage,
                'minimum_coverage': self.minimum_coverage
            },
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_recommendations(trend_analysis),
            'summary': self._generate_summary(trend_analysis)
        }

        # 保存报告
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 趋势分析报告已保存: {output_file}")

        return report

    def _generate_recommendations(self, trend_analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        trend_direction = trend_analysis.get('trend', 'STABLE')
        current_coverage = trend_analysis.get('current_coverage', 0)
        alerts = trend_analysis.get('alerts', [])

        # 基于趋势的建议
        if trend_direction == 'DECLINING':
            recommendations.extend([
                '分析覆盖率下降的根本原因',
                '审查最近的代码变更和新增功能',
                '加强代码审查中的测试要求',
                '考虑增加测试覆盖率的CI/CD检查'
            ])
        elif trend_direction == 'STABLE' and current_coverage < self.target_coverage:
            recommendations.extend([
                '制定系统性的测试改进计划',
                '优先覆盖核心业务逻辑',
                '增加边界条件和异常处理测试',
                '建立测试覆盖率提升的激励机制'
            ])
        elif trend_direction == 'IMPROVING':
            recommendations.extend([
                '保持当前的测试改进势头',
                '关注测试质量而非仅仅数量',
                '建立可持续的测试维护机制'
            ])

        # 基于告警的建议
        critical_alerts = [a for a in alerts if a.get('level') == 'CRITICAL']
        if critical_alerts:
            recommendations.insert(0, '立即处理严重告警，暂停非关键功能开发')

        return recommendations[:5]  # 限制建议数量

    def _generate_summary(self, trend_analysis: Dict[str, Any]) -> str:
        """生成摘要"""
        trend_direction = trend_analysis.get('trend', 'STABLE')
        current_coverage = trend_analysis.get('current_coverage', 0)
        change = trend_analysis.get('change', 0)
        alerts = trend_analysis.get('alerts', [])

        # 基础摘要
        if trend_direction == 'IMPROVING':
            summary = f"测试覆盖率呈上升趋势，当前{current_coverage:.1f}%"
            if change > 0:
                summary += f"，较上次提升{change:.1f}%"
        elif trend_direction == 'DECLINING':
            summary = f"测试覆盖率呈下降趋势，当前{current_coverage:.1f}%"
            if change < 0:
                summary += f"，较上次下降{abs(change):.1f}%"
        else:
            summary = f"测试覆盖率保持稳定，当前{current_coverage:.1f}%"

        # 添加状态评估
        if current_coverage >= self.target_coverage:
            summary += "，已达到目标要求"
        elif current_coverage >= self.minimum_coverage:
            summary += "，达到最低要求但需继续改进"
        else:
            summary += "，低于最低要求，需要紧急改进"

        # 添加告警信息
        critical_count = len([a for a in alerts if a.get('level') == 'CRITICAL'])
        warning_count = len([a for a in alerts if a.get('level') == 'WARNING'])

        if critical_count > 0:
            summary += f"，存在{critical_count}个严重告警"
        elif warning_count > 0:
            summary += f"，存在{warning_count}个警告"

        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025测试覆盖率趋势分析')
    parser.add_argument('--generate-trends', action='store_true',
                        help='生成趋势分析报告')
    parser.add_argument('--record-coverage', type=str,
                        help='记录覆盖率数据（提供JSON文件路径）')
    parser.add_argument('--target-coverage', type=float, default=90.0,
                        help='目标覆盖率 (默认: 90%)')
    parser.add_argument('--alert-on-decline', action='store_true',
                        help='检查覆盖率下降并发出告警')
    parser.add_argument('--output', type=str,
                        help='输出报告文件路径')

    args = parser.parse_args()

    # 创建趋势分析器
    analyzer = CoverageTrendAnalyzer()

    if args.record_coverage:
        # 记录覆盖率数据
        try:
            with open(args.record_coverage, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)
            analyzer.record_coverage_data(coverage_data)
            print(f"✅ 覆盖率数据已记录")
        except Exception as e:
            logger.error(f"❌ 记录覆盖率数据失败: {e}")
            sys.exit(1)

    if args.generate_trends:
        # 生成趋势报告
        report = analyzer.generate_trend_report(
            target_coverage=args.target_coverage,
            output_file=args.output
        )

        # 输出摘要
        print(f"\n📈 覆盖率趋势分析报告")
        print(f"📊 {report['summary']}")

        # 输出告警
        alerts = report['trend_analysis'].get('alerts', [])
        if alerts:
            print(f"\n🚨 告警信息:")
            for alert in alerts:
                level_icon = "🔴" if alert['level'] == 'CRITICAL' else "🟡"
                print(f"  {level_icon} {alert['message']}")

    if args.alert_on_decline:
        # 检查覆盖率下降
        historical_data = analyzer.load_historical_data(days=7)  # 检查最近一周
        if len(historical_data) >= 2:
            trend_analysis = analyzer.analyze_coverage_trends(historical_data)
            alerts = trend_analysis.get('alerts', [])

            critical_alerts = [a for a in alerts if a.get('level') == 'CRITICAL']
            if critical_alerts:
                print("🚨 检测到严重覆盖率问题!")
                for alert in critical_alerts:
                    print(f"  🔴 {alert['message']}")
                sys.exit(1)
            elif alerts:
                print("⚠️ 检测到覆盖率警告")
                for alert in alerts:
                    print(f"  🟡 {alert['message']}")

        print("✅ 覆盖率检查通过")


if __name__ == "__main__":
    main()
