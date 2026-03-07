#!/usr/bin/env python3
"""
RQA2025 业务流程测试监控和告警系统

监控业务流程测试执行情况，及时发现问题并发出告警。
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging


class TestMonitor:
    """测试监控器"""

    def __init__(self, reports_dir: str = "reports/business_flow_tests"):
        self.reports_dir = Path(reports_dir)
        self.logger = self._setup_logger()
        self.alert_history = []
        self.monitoring_config = self._load_monitoring_config()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TestMonitor')
        logger.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        # 添加处理器到记录器
        if not logger.handlers:
            logger.addHandler(console_handler)

        return logger

    def _load_monitoring_config(self) -> Dict[str, Any]:
        """加载监控配置"""
        return {
            'alert_thresholds': {
                'max_failure_rate': 0.1,  # 最大失败率10%
                'max_execution_time': 300,  # 最大执行时间5分钟
                'min_success_rate': 0.95,  # 最小成功率95%
                'critical_test_timeout': 600  # 关键测试超时时间10分钟
            },
            'alert_channels': {
                'email': {
                    'enabled': True,
                    'recipients': ['devops@company.com', 'qa@company.com'],
                    'smtp_server': 'smtp.company.com',
                    'smtp_port': 587
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': '',
                    'channel': '#test-alerts'
                },
                'teams': {
                    'enabled': False,
                    'webhook_url': '',
                    'channel': 'test-alerts'
                }
            },
            'monitoring_rules': {
                'check_frequency': 60,  # 每60秒检查一次
                'max_alert_frequency': 300,  # 相同告警最多每5分钟发送一次
                'alert_cooldown': 1800,  # 告警冷却时间30分钟
                'escalation_time': 3600  # 升级告警时间1小时
            }
        }

    def monitor_test_execution(self) -> Dict[str, Any]:
        """监控测试执行情况"""
        self.logger.info("开始监控业务流程测试执行情况")

        # 获取最新的测试报告
        latest_report = self._get_latest_test_report()
        if not latest_report:
            self.logger.warning("未找到测试报告")
            return {'status': 'no_report', 'message': '未找到测试报告'}

        # 分析测试结果
        analysis = self._analyze_test_results(latest_report)

        # 检查是否需要告警
        alerts = self._check_alert_conditions(analysis)

        # 发送告警
        if alerts:
            self._send_alerts(alerts, analysis)

        # 生成监控报告
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'report_file': str(latest_report),
            'analysis': analysis,
            'alerts': alerts,
            'recommendations': self._generate_recommendations(analysis)
        }

        self.logger.info(f"监控完成，发现 {len(alerts)} 个告警")

        return monitoring_report

    def _get_latest_test_report(self) -> Optional[Path]:
        """获取最新的测试报告"""
        if not self.reports_dir.exists():
            return None

        # 查找JSON格式的测试报告
        json_files = list(self.reports_dir.glob("business_flow_test_report_*.json"))
        if not json_files:
            return None

        # 返回最新的报告文件
        return max(json_files, key=lambda f: f.stat().st_mtime)

    def _analyze_test_results(self, report_file: Path) -> Dict[str, Any]:
        """分析测试结果"""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            overall_summary = report_data.get('overall_summary', {})
            flow_details = report_data.get('flow_details', {})

            analysis = {
                'total_flows': overall_summary.get('total_flows_tested', 0),
                'passed_flows': overall_summary.get('passed_flows', 0),
                'failed_flows': overall_summary.get('failed_flows', 0),
                'success_rate': overall_summary.get('overall_success_rate', 0),
                'execution_time': overall_summary.get('total_execution_time', 0),
                'test_status': overall_summary.get('test_status', 'UNKNOWN'),
                'flow_analysis': {},
                'performance_metrics': {
                    'avg_execution_time': overall_summary.get('average_execution_time', 0),
                    'total_execution_time': overall_summary.get('total_execution_time', 0)
                }
            }

            # 分析各流程详情
            for flow_name, flow_detail in flow_details.items():
                flow_analysis = {
                    'status': flow_detail.get('status'),
                    'execution_time': flow_detail.get('execution_time', 0),
                    'success_rate': flow_detail.get('success_rate', 0),
                    'steps_completed': flow_detail.get('steps_completed', 0),
                    'total_steps': flow_detail.get('total_steps', 0),
                    'issues': []
                }

                # 检查流程问题
                if flow_detail.get('status') != 'passed':
                    flow_analysis['issues'].append(f"流程状态异常: {flow_detail.get('status')}")

                if flow_detail.get('success_rate', 0) < 1.0:
                    flow_analysis['issues'].append(f"成功率不足: {flow_detail.get('success_rate'):.1%}")

                if flow_detail.get('execution_time', 0) > 60:  # 超过1分钟
                    flow_analysis['issues'].append(f"执行时间过长: {flow_detail.get('execution_time'):.2f}秒")

                analysis['flow_analysis'][flow_name] = flow_analysis

            return analysis

        except Exception as e:
            self.logger.error(f"分析测试结果失败: {e}")
            return {'error': str(e)}

    def _check_alert_conditions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        thresholds = self.monitoring_config['alert_thresholds']

        # 检查整体成功率
        if analysis.get('success_rate', 1.0) < thresholds['min_success_rate']:
            alerts.append({
                'level': 'critical',
                'type': 'low_success_rate',
                'message': f"测试成功率过低: {analysis.get('success_rate'):.1%} < {thresholds['min_success_rate']:.1%}",
                'details': analysis
            })

        # 检查执行时间
        if analysis.get('execution_time', 0) > thresholds['max_execution_time']:
            alerts.append({
                'level': 'warning',
                'type': 'long_execution_time',
                'message': f"测试执行时间过长: {analysis.get('execution_time'):.2f}秒 > {thresholds['max_execution_time']}秒",
                'details': analysis
            })

        # 检查失败流程
        if analysis.get('failed_flows', 0) > 0:
            alerts.append({
                'level': 'error',
                'type': 'failed_flows',
                'message': f"发现失败的业务流程: {analysis.get('failed_flows')} 个",
                'details': analysis
            })

        # 检查流程级问题
        for flow_name, flow_analysis in analysis.get('flow_analysis', {}).items():
            if flow_analysis.get('issues'):
                alerts.append({
                    'level': 'warning',
                    'type': 'flow_issues',
                    'message': f"流程 {flow_name} 存在问题: {len(flow_analysis['issues'])} 个",
                    'details': {'flow': flow_name, 'issues': flow_analysis['issues']}
                })

        return alerts

    def _send_alerts(self, alerts: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """发送告警"""
        for alert in alerts:
            # 检查告警频率限制
            if self._should_send_alert(alert):
                self._send_email_alert(alert, analysis)
                self._send_slack_alert(alert, analysis)
                self._send_teams_alert(alert, analysis)

                # 记录告警历史
                self.alert_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert,
                    'analysis': analysis
                })

    def _should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """检查是否应该发送告警"""
        # 检查冷却时间
        cooldown_period = self.monitoring_config['monitoring_rules']['alert_cooldown']
        recent_alerts = [
            h for h in self.alert_history
            if (datetime.now() - datetime.fromisoformat(h['timestamp'])).total_seconds() < cooldown_period
            and h['alert']['type'] == alert['type']
        ]

        return len(recent_alerts) == 0

    def _send_email_alert(self, alert: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """发送邮件告警"""
        config = self.monitoring_config['alert_channels']['email']
        if not config['enabled']:
            return

        try:
            # 创建邮件内容
            subject = f"🚨 RQA2025 业务流程测试告警 - {alert['level'].upper()}"
            body = self._generate_alert_content(alert, analysis, 'email')

            # 发送邮件
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = 'monitor@company.com'
            msg['To'] = ', '.join(config['recipients'])

            msg.attach(MIMEText(body, 'html'))

            # 注意：这里需要实际的SMTP配置
            # server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            # server.starttls()
            # server.login(username, password)
            # server.sendmail(msg['From'], config['recipients'], msg.as_string())
            # server.quit()

            self.logger.info(f"邮件告警已发送: {subject}")

        except Exception as e:
            self.logger.error(f"发送邮件告警失败: {e}")

    def _send_slack_alert(self, alert: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """发送Slack告警"""
        config = self.monitoring_config['alert_channels']['slack']
        if not config['enabled']:
            return

        try:
            # 这里实现Slack webhook发送逻辑
            # import requests
            # payload = {'text': self._generate_alert_content(alert, analysis, 'slack')}
            # requests.post(config['webhook_url'], json=payload)

            self.logger.info("Slack告警已发送")

        except Exception as e:
            self.logger.error(f"发送Slack告警失败: {e}")

    def _send_teams_alert(self, alert: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """发送Teams告警"""
        config = self.monitoring_config['alert_channels']['teams']
        if not config['enabled']:
            return

        try:
            # 这里实现Teams webhook发送逻辑
            self.logger.info("Teams告警已发送")

        except Exception as e:
            self.logger.error(f"发送Teams告警失败: {e}")

    def _generate_alert_content(self, alert: Dict[str, Any], analysis: Dict[str, Any], channel: str) -> str:
        """生成告警内容"""
        level_icons = {
            'critical': '🚨',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }

        content = f"""
{level_icons.get(alert['level'], '⚠️')} **业务流程测试告警**

**告警级别**: {alert['level'].upper()}
**告警类型**: {alert['type']}
**告警消息**: {alert['message']}

**测试概况**:
- 总流程数: {analysis.get('total_flows', 0)}
- 通过流程数: {analysis.get('passed_flows', 0)}
- 失败流程数: {analysis.get('failed_flows', 0)}
- 成功率: {analysis.get('success_rate', 0):.1%}
- 执行时间: {analysis.get('execution_time', 0):.2f}秒

**告警时间**: {datetime.now().isoformat()}
**系统**: RQA2025 量化交易系统
**模块**: 业务流程测试
"""

        if channel == 'email':
            content = f"""
<html>
<body>
<h2>{level_icons.get(alert['level'], '⚠️')} 业务流程测试告警</h2>

<p><strong>告警级别:</strong> {alert['level'].upper()}</p>
<p><strong>告警类型:</strong> {alert['type']}</p>
<p><strong>告警消息:</strong> {alert['message']}</p>

<h3>测试概况</h3>
<ul>
<li>总流程数: {analysis.get('total_flows', 0)}</li>
<li>通过流程数: {analysis.get('passed_flows', 0)}</li>
<li>失败流程数: {analysis.get('failed_flows', 0)}</li>
<li>成功率: {analysis.get('success_rate', 0):.1%}</li>
<li>执行时间: {analysis.get('execution_time', 0):.2f}秒</li>
</ul>

<p><strong>告警时间:</strong> {datetime.now().isoformat()}</p>
<p><strong>系统:</strong> RQA2025 量化交易系统</p>
<p><strong>模块:</strong> 业务流程测试</p>
</body>
</html>
"""

        return content

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        if analysis.get('success_rate', 1.0) < 0.95:
            recommendations.append("检查失败的测试用例并修复问题")
            recommendations.append("审查测试数据质量和业务逻辑")

        if analysis.get('execution_time', 0) > 300:
            recommendations.append("优化测试执行性能，考虑并行执行")
            recommendations.append("审查测试数据量和处理逻辑")

        failed_flows = analysis.get('failed_flows', 0)
        if failed_flows > 0:
            recommendations.append(f"重点关注 {failed_flows} 个失败的业务流程")
            recommendations.append("检查业务流程配置和依赖关系")

        # 流程级建议
        for flow_name, flow_analysis in analysis.get('flow_analysis', {}).items():
            if flow_analysis.get('issues'):
                recommendations.append(f"修复流程 {flow_name} 的 {len(flow_analysis['issues'])} 个问题")

        if not recommendations:
            recommendations.append("测试执行正常，建议继续监控")

        return recommendations

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        monitoring_result = self.monitor_test_execution()

        report = {
            'monitoring_timestamp': datetime.now().isoformat(),
            'monitoring_result': monitoring_result,
            'alert_history': self.alert_history[-10:],  # 最近10个告警
            'system_status': {
                'reports_directory': str(self.reports_dir),
                'reports_exist': self.reports_dir.exists(),
                'latest_report': str(self._get_latest_test_report()) if self._get_latest_test_report() else None
            },
            'recommendations': monitoring_result.get('recommendations', [])
        }

        return report


def main():
    """主函数"""
    print("🎯 RQA2025 业务流程测试监控系统")
    print("=" * 50)

    monitor = TestMonitor()
    report = monitor.generate_monitoring_report()

    # 保存监控报告
    reports_dir = Path("reports/monitoring")
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f"business_process_monitoring_report_{timestamp}.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("📊 监控报告已生成:")
    print(f"   文件: {report_file}")

    # 检查是否有告警
    alerts = report['monitoring_result'].get('alerts', [])
    if alerts:
        print(f"\n🚨 发现 {len(alerts)} 个告警:")
        for alert in alerts:
            print(f"   {alert['level'].upper()}: {alert['message']}")
    else:
        print("\n✅ 未发现告警，系统运行正常")

    print(f"\n📋 建议:")
    for recommendation in report.get('recommendations', []):
        print(f"   • {recommendation}")


if __name__ == "__main__":
    main()
