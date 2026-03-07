#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 智能告警系统
提供多渠道告警通知和异常检测功能

告警功能:
1. 阈值告警 (系统资源、性能指标)
2. 异常检测 (基于统计模型的异常识别)
3. 服务状态告警 (服务宕机、响应异常)
4. 趋势分析告警 (性能下降趋势)
5. 多渠道通知 (日志、控制台、文件)
"""

import os
import json
import time
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import logging
import statistics

class AlertSystem:
    """智能告警系统"""

    def __init__(self):
        self.alerts = deque(maxlen=1000)  # 保留最近1000个告警
        self.alert_rules = self.load_alert_rules()
        self.notification_channels = {
            'console': self.notify_console,
            'file': self.notify_file,
            'email': self.notify_email
        }
        self.active_channels = ['console', 'file']  # 默认启用控制台和文件通知

        # 异常检测历史数据
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        self.anomaly_threshold = 3.0  # 3倍标准差作为异常阈值

    def load_alert_rules(self):
        """加载告警规则"""
        default_rules = {
            'system_cpu_high': {
                'metric': 'cpu_percent',
                'threshold': 80.0,
                'operator': '>',
                'level': 'warning',
                'message': 'CPU使用率过高: {value:.1f}%',
                'cooldown': 300  # 5分钟冷却期
            },
            'system_memory_high': {
                'metric': 'memory_percent',
                'threshold': 85.0,
                'operator': '>',
                'level': 'critical',
                'message': '内存使用率过高: {value:.1f}%',
                'cooldown': 300
            },
            'system_disk_high': {
                'metric': 'disk_percent',
                'threshold': 90.0,
                'operator': '>',
                'level': 'warning',
                'message': '磁盘使用率过高: {value:.1f}%',
                'cooldown': 600
            },
            'service_down': {
                'metric': 'service_healthy',
                'threshold': False,
                'operator': '==',
                'level': 'critical',
                'message': '服务 {service_name} 不可用',
                'cooldown': 60
            },
            'response_time_high': {
                'metric': 'avg_response_time',
                'threshold': 1.0,
                'operator': '>',
                'level': 'warning',
                'message': '响应时间过高: {value:.3f}s',
                'cooldown': 180
            }
        }

        # 尝试加载自定义规则
        rules_file = Path('monitoring_system/alert_rules.json')
        if rules_file.exists():
            with open(rules_file, 'r', encoding='utf-8') as f:
                custom_rules = json.load(f)
            default_rules.update(custom_rules)

        return default_rules

    def check_alerts(self, metrics_data):
        """检查告警条件"""
        current_time = datetime.now()

        for rule_name, rule in self.alert_rules.items():
            try:
                if self._should_check_rule(rule_name, rule, current_time):
                    if self._evaluate_rule(rule, metrics_data):
                        alert = self._create_alert(rule_name, rule, metrics_data, current_time)
                        self._trigger_alert(alert)

            except Exception as e:
                logging.error(f"检查告警规则 {rule_name} 时出错: {e}")

    def _should_check_rule(self, rule_name, rule, current_time):
        """检查是否应该执行告警规则"""
        # 检查冷却期
        recent_alerts = [a for a in self.alerts
                        if a['rule_name'] == rule_name and
                        current_time - datetime.fromisoformat(a['timestamp']) < timedelta(seconds=rule.get('cooldown', 300))]

        return len(recent_alerts) == 0

    def _evaluate_rule(self, rule, metrics_data):
        """评估告警规则条件"""
        metric_name = rule['metric']
        threshold = rule['threshold']
        operator = rule['operator']

        # 获取指标值
        value = self._extract_metric_value(metric_name, metrics_data)

        if value is None:
            return False

        # 评估条件
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '==':
            return value == threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        else:
            return False

    def _extract_metric_value(self, metric_name, metrics_data):
        """从指标数据中提取值"""
        # 系统指标
        if metric_name in ['cpu_percent', 'memory_percent', 'disk_percent']:
            return metrics_data.get('system_status', {}).get('summary', {}).get(metric_name.replace('_percent', '_avg'))

        # 服务指标
        if metric_name == 'service_healthy':
            services = metrics_data.get('service_health', {}).get('services', {})
            # 检查是否有服务不健康
            unhealthy_count = sum(1 for s in services.values() if not s.get('healthy', True))
            return unhealthy_count > 0

        # 性能指标
        if metric_name == 'avg_response_time':
            performance = metrics_data.get('performance_metrics', {})
            response_times = [p.get('avg_response_time', 0) for p in performance.values()]
            return max(response_times) if response_times else 0

        return None

    def _create_alert(self, rule_name, rule, metrics_data, timestamp):
        """创建告警"""
        value = self._extract_metric_value(rule['metric'], metrics_data)

        alert = {
            'id': f"{rule_name}_{int(timestamp.timestamp())}",
            'rule_name': rule_name,
            'level': rule['level'],
            'message': rule['message'].format(value=value, **metrics_data),
            'value': value,
            'threshold': rule['threshold'],
            'timestamp': timestamp.isoformat(),
            'resolved': False,
            'acknowledged': False
        }

        return alert

    def _trigger_alert(self, alert):
        """触发告警"""
        # 添加到告警队列
        self.alerts.append(alert)

        # 发送通知
        for channel in self.active_channels:
            try:
                self.notification_channels[channel](alert)
            except Exception as e:
                logging.error(f"发送告警通知失败 ({channel}): {e}")

        logging.warning(f"告警触发: {alert['level']} - {alert['message']}")

    def check_anomalies(self, metrics_data):
        """检查异常模式"""
        current_time = datetime.now()

        for metric_name, value in self._flatten_metrics(metrics_data):
            if metric_name not in ['cpu_percent', 'memory_percent', 'avg_response_time']:
                continue

            # 记录历史数据
            self.metric_history[metric_name].append(value)

            # 需要足够的历史数据才能检测异常
            if len(self.metric_history[metric_name]) < 10:
                continue

            # 计算统计特征
            values = list(self.metric_history[metric_name])
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0

            if stdev > 0:
                z_score = abs(value - mean) / stdev

                if z_score > self.anomaly_threshold:
                    anomaly_alert = {
                        'id': f"anomaly_{metric_name}_{int(current_time.timestamp())}",
                        'rule_name': 'anomaly_detection',
                        'level': 'warning',
                        'message': f'异常检测: {metric_name} 值 {value:.2f} 偏离正常范围 (Z-Score: {z_score:.1f})',
                        'value': value,
                        'threshold': mean + self.anomaly_threshold * stdev,
                        'timestamp': current_time.isoformat(),
                        'metric': metric_name,
                        'z_score': z_score,
                        'resolved': False,
                        'acknowledged': False
                    }

                    self._trigger_alert(anomaly_alert)

    def _flatten_metrics(self, metrics_data):
        """扁平化指标数据"""
        flattened = []

        # 系统指标
        system = metrics_data.get('system_status', {}).get('summary', {})
        flattened.extend([
            ('cpu_percent', system.get('cpu_avg', 0)),
            ('memory_percent', system.get('memory_avg', 0)),
            ('disk_percent', system.get('disk_usage', 0))
        ])

        # 性能指标
        performance = metrics_data.get('performance_metrics', {})
        for service, metrics in performance.items():
            flattened.append((f'{service}_response_time', metrics.get('avg_response_time', 0)))

        return flattened

    def notify_console(self, alert):
        """控制台通知"""
        level_icon = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨'
        }

        icon = level_icon.get(alert['level'], '❓')
        print(f"{icon} [{alert['level'].upper()}] {alert['message']}")

    def notify_file(self, alert):
        """文件通知"""
        alerts_dir = Path('monitoring_system/alerts')
        alerts_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime('%Y-%m-%d')
        alert_file = alerts_dir / f'alerts_{today}.log'

        with open(alert_file, 'a', encoding='utf-8') as f:
            f.write(f"[{alert['timestamp']}] {alert['level'].upper()}: {alert['message']}\n")

    def notify_email(self, alert):
        """邮件通知 (需要配置SMTP)"""
        # 这里可以实现邮件通知
        # 需要SMTP服务器配置
        pass

    def get_alert_summary(self):
        """获取告警摘要"""
        total_alerts = len(self.alerts)
        critical_alerts = len([a for a in self.alerts if a['level'] == 'critical'])
        warning_alerts = len([a for a in self.alerts if a['level'] == 'warning'])
        unresolved_alerts = len([a for a in self.alerts if not a['resolved']])

        recent_alerts = [a for a in self.alerts if
                        datetime.now() - datetime.fromisoformat(a['timestamp']) < timedelta(hours=24)]

        return {
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'warning_alerts': warning_alerts,
            'unresolved_alerts': unresolved_alerts,
            'recent_alerts_24h': len(recent_alerts),
            'alerts_by_level': {
                'critical': critical_alerts,
                'warning': warning_alerts,
                'info': total_alerts - critical_alerts - warning_alerts
            }
        }

    def acknowledge_alert(self, alert_id):
        """确认告警"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                logging.info(f"告警已确认: {alert_id}")
                break

    def resolve_alert(self, alert_id):
        """解决告警"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['resolved'] = True
                logging.info(f"告警已解决: {alert_id}")
                break


class LogAnalyzer:
    """日志分析器"""

    def __init__(self):
        self.log_patterns = {
            'error': [r'ERROR', r'Exception', r'Failed', r'Error'],
            'warning': [r'WARNING', r'Warn'],
            'info': [r'INFO'],
            'debug': [r'DEBUG']
        }

    def analyze_logs(self, log_file_path, hours=24):
        """分析日志文件"""
        if not Path(log_file_path).exists():
            return {'error': '日志文件不存在'}

        analysis = {
            'period': f'{hours}小时',
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'recent_errors': [],
            'top_error_patterns': {},
            'log_levels_distribution': {}
        }

        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    analysis['total_lines'] += 1

                    # 解析时间戳 (假设格式为YYYY-MM-DD HH:MM:SS)
                    try:
                        if len(line) > 19:
                            line_time = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                            if line_time < cutoff_time:
                                continue
                    except:
                        pass  # 如果无法解析时间戳，仍然处理该行

                    # 分析日志级别
                    line_lower = line.lower()
                    if any(pattern.lower() in line_lower for pattern in self.log_patterns['error']):
                        analysis['error_count'] += 1
                        if len(analysis['recent_errors']) < 10:
                            analysis['recent_errors'].append(line.strip())
                    elif any(pattern.lower() in line_lower for pattern in self.log_patterns['warning']):
                        analysis['warning_count'] += 1
                    elif any(pattern.lower() in line_lower for pattern in self.log_patterns['info']):
                        analysis['info_count'] += 1

        except Exception as e:
            return {'error': f'日志分析失败: {str(e)}'}

        # 计算分布
        total_classified = analysis['error_count'] + analysis['warning_count'] + analysis['info_count']
        if total_classified > 0:
            analysis['log_levels_distribution'] = {
                'error': analysis['error_count'] / total_classified,
                'warning': analysis['warning_count'] / total_classified,
                'info': analysis['info_count'] / total_classified
            }

        return analysis


def create_alert_report(alert_system, log_analyzer):
    """创建告警和日志分析报告"""
    alert_summary = alert_system.get_alert_summary()

    # 分析主要日志文件
    log_files = [
        'monitoring_system/monitoring.log',
        'web_interface/app.log',
        'deployment_scripts/deployment.log'
    ]

    log_analysis = {}
    for log_file in log_files:
        if Path(log_file).exists():
            log_analysis[log_file] = log_analyzer.analyze_logs(log_file)

    report = {
        'report_time': datetime.now().isoformat(),
        'alert_summary': alert_summary,
        'log_analysis': log_analysis,
        'system_health_score': calculate_health_score(alert_summary, log_analysis),
        'recommendations': generate_recommendations(alert_summary, log_analysis)
    }

    return report


def calculate_health_score(alert_summary, log_analysis):
    """计算系统健康评分 (0-100)"""
    score = 100

    # 基于告警数量扣分
    critical_penalty = alert_summary['critical_alerts'] * 20
    warning_penalty = alert_summary['warning_alerts'] * 5

    score -= min(critical_penalty + warning_penalty, 50)

    # 基于错误日志扣分
    total_errors = 0
    for analysis in log_analysis.values():
        if isinstance(analysis, dict) and 'error_count' in analysis:
            total_errors += analysis['error_count']

    error_penalty = min(total_errors * 2, 30)
    score -= error_penalty

    return max(0, score)


def generate_recommendations(alert_summary, log_analysis):
    """生成建议"""
    recommendations = []

    if alert_summary['critical_alerts'] > 0:
        recommendations.append("🔴 紧急: 处理关键告警，检查系统稳定性")

    if alert_summary['warning_alerts'] > 5:
        recommendations.append("🟡 注意: 监控告警数量较多，建议优化系统配置")

    for log_file, analysis in log_analysis.items():
        if isinstance(analysis, dict) and analysis.get('error_count', 0) > 10:
            recommendations.append(f"📋 检查日志文件 {log_file} 中的错误模式")

    if not recommendations:
        recommendations.append("✅ 系统运行正常，继续监控")

    return recommendations


def main():
    """主函数"""
    print("🚨 启动 RQA2026 智能告警系统")
    print("=" * 60)

    # 创建告警系统和日志分析器
    alert_system = AlertSystem()
    log_analyzer = LogAnalyzer()

    # 模拟一些告警检查
    print("🔍 执行告警检查...")

    # 模拟指标数据
    sample_metrics = {
        'system_status': {
            'summary': {
                'cpu_avg': 45.2,
                'memory_avg': 62.8,
                'disk_usage': 34.1
            }
        },
        'service_health': {
            'services': {
                'fusion_engine': {'healthy': True},
                'quantum_engine': {'healthy': True},
                'ai_engine': {'healthy': False},
                'bci_engine': {'healthy': True},
                'web_interface': {'healthy': True}
            }
        },
        'performance_metrics': {
            'fusion_engine': {'avg_response_time': 0.05},
            'quantum_engine': {'avg_response_time': 0.08},
            'ai_engine': {'avg_response_time': 0.15},
            'bci_engine': {'avg_response_time': 0.03},
            'web_interface': {'avg_response_time': 0.12}
        }
    }

    # 检查告警
    alert_system.check_alerts(sample_metrics)
    alert_system.check_anomalies(sample_metrics)

    # 生成告警报告
    print("📋 生成告警和日志分析报告...")
    report = create_alert_report(alert_system, log_analyzer)

    # 保存报告
    report_file = Path('monitoring_system/alert_report.json')
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("✅ 告警报告已保存: {}".format(report_file))

    # 显示摘要
    print("\\n🎯 告警系统摘要:")
    print("  🚨 总告警数: {}".format(report['alert_summary']['total_alerts']))
    print("  🔴 关键告警: {}".format(report['alert_summary']['critical_alerts']))
    print("  🟡 警告告警: {}".format(report['alert_summary']['warning_alerts']))
    print("  💚 系统健康评分: {}/100".format(report['system_health_score']))

    if report['recommendations']:
        print("\\n💡 建议:")
        for rec in report['recommendations']:
            print("  {}".format(rec))


if __name__ == "__main__":
    main()
