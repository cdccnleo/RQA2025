#!/usr/bin/env python3
"""
RQA2025 质量保障调度器

提供自动化的质量保障任务调度，包括：
- 定期一致性检查
- 自动文档同步
- 版本管理任务
- 质量报告生成
- 告警通知机制
"""

import os
import json
import logging
import schedule
import time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QualityScheduler:
    """质量保障调度器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.config_file = self.project_root / "scripts" / "quality_assurance" / "scheduler_config.json"
        self.reports_dir = self.project_root / "reports" / "scheduled"

        # 创建目录
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.config = self._load_config()

        # 初始化组件
        self.consistency_checker = None
        self.doc_sync = None
        self.version_manager = None
        self.automated_pipeline = None

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'schedule': {
                'consistency_check': 'daily',  # daily, weekly, monthly
                'doc_sync': 'daily',
                'version_check': 'weekly',
                'quality_report': 'weekly'
            },
            'time': {
                'consistency_check': '09:00',
                'doc_sync': '10:00',
                'version_check': '09:00',
                'quality_report': '08:00'
            },
            'notification': {
                'enabled': True,
                'email': {
                    'smtp_server': 'smtp.example.com',
                    'smtp_port': 587,
                    'username': 'your-email@example.com',
                    'password': 'your-password',
                    'recipients': ['team@example.com']
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': '',
                    'channel': '#quality'
                }
            },
            'thresholds': {
                'consistency_score': 95.0,
                'sync_success_rate': 90.0,
                'version_consistency': True
            }
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 合并配置
                self._merge_config(default_config, user_config)
                return default_config
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")

        return default_config

    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """合并配置"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def setup_schedules(self):
        """设置调度任务"""
        logger.info("设置质量保障调度任务...")

        # 一致性检查
        check_time = self.config['time']['consistency_check']
        if self.config['schedule']['consistency_check'] == 'daily':
            schedule.every().day.at(check_time).do(self.run_consistency_check)
        elif self.config['schedule']['consistency_check'] == 'weekly':
            schedule.every().monday.at(check_time).do(self.run_consistency_check)

        # 文档同步
        sync_time = self.config['time']['doc_sync']
        if self.config['schedule']['doc_sync'] == 'daily':
            schedule.every().day.at(sync_time).do(self.run_doc_sync)
        elif self.config['schedule']['doc_sync'] == 'weekly':
            schedule.every().monday.at(sync_time).do(self.run_doc_sync)

        # 版本检查
        version_time = self.config['time']['version_check']
        if self.config['schedule']['version_check'] == 'weekly':
            schedule.every().monday.at(version_time).do(self.run_version_check)

        # 自动化质量流水线
        pipeline_time = self.config['time']['automated_pipeline']
        if self.config['schedule']['automated_pipeline'] == 'daily':
            schedule.every().day.at(pipeline_time).do(self.run_automated_pipeline)
        elif self.config['schedule']['automated_pipeline'] == 'weekly':
            schedule.every().monday.at(pipeline_time).do(self.run_automated_pipeline)

        # 质量报告
        report_time = self.config['time']['quality_report']
        if self.config['schedule']['quality_report'] == 'weekly':
            schedule.every().monday.at(report_time).do(self.generate_quality_report)

        logger.info("调度任务设置完成")

    def run_consistency_check(self):
        """运行一致性检查"""
        logger.info("开始定期一致性检查...")

        try:
            # 导入一致性检查器
            if self.consistency_checker is None:
                from .consistency_checker import ConsistencyChecker
                self.consistency_checker = ConsistencyChecker(str(self.project_root))

            # 运行检查
            results = self.consistency_checker.run_quick_check()

            # 检查是否需要告警
            summary = results.get('summary', {})
            consistency_score = summary.get('consistency_score', 100)

            if consistency_score < self.config['thresholds']['consistency_score']:
                self._send_alert(
                    'consistency_check',
                    f'一致性评分过低: {consistency_score:.1f}%',
                    results
                )

            logger.info(f"一致性检查完成，评分: {consistency_score:.1f}%")

        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            self._send_alert('consistency_check', f'一致性检查失败: {e}', None)

    def run_doc_sync(self):
        """运行文档同步"""
        logger.info("开始定期文档同步...")

        try:
            # 导入文档同步器
            if self.doc_sync is None:
                from ..documentation_automation.doc_sync import DocSync
                self.doc_sync = DocSync(str(self.project_root))

            # 运行同步
            results = self.doc_sync.sync_all_docs()

            # 检查同步成功率
            summary = results.get('summary', {})
            sync_rate = summary.get('sync_rate', 100)

            if sync_rate < self.config['thresholds']['sync_success_rate']:
                self._send_alert(
                    'doc_sync',
                    f'文档同步成功率过低: {sync_rate:.1f}%',
                    results
                )

            logger.info(f"文档同步完成，成功率: {sync_rate:.1f}%")

        except Exception as e:
            logger.error(f"文档同步失败: {e}")
            self._send_alert('doc_sync', f'文档同步失败: {e}', None)

    def run_version_check(self):
        """运行版本检查"""
        logger.info("开始定期版本检查...")

        try:
            # 导入版本管理器
            if self.version_manager is None:
                from ..version_management.version_manager import VersionManager
                self.version_manager = VersionManager(str(self.project_root))

            # 检查版本一致性
            consistency_result = self.version_manager.check_version_consistency()

            if not consistency_result.get('consistent', True):
                self._send_alert(
                    'version_check',
                    '发现版本不一致问题',
                    consistency_result
                )

            # 生成版本报告
            report = self.version_manager.generate_version_report()

            logger.info("版本检查完成")

        except Exception as e:
            logger.error(f"版本检查失败: {e}")
            self._send_alert('version_check', f'版本检查失败: {e}', None)

    def generate_quality_report(self):
        """生成质量报告"""
        logger.info("开始生成质量报告...")

        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'period': 'weekly',
                'sections': {}
            }

            # 收集各组件的最新报告
            report_files = list(self.reports_dir.glob("*.json"))
            report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # 读取最近的报告
            for report_file in report_files[:10]:  # 最近10个报告
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    report_type = self._infer_report_type(report_file.name)
                    if report_type not in report['sections']:
                        report['sections'][report_type] = []

                    report['sections'][report_type].append({
                        'file': report_file.name,
                        'data': report_data
                    })

                except Exception as e:
                    logger.warning(f"读取报告失败 {report_file}: {e}")

            # 生成汇总报告
            summary = self._generate_quality_summary(report)

            # 保存质量报告
            report_file = self.reports_dir / \
                f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # 生成HTML报告
            html_report = self._generate_html_quality_report(report, summary)
            html_file = self.reports_dir / \
                f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)

            logger.info(f"质量报告已生成: {report_file}")

            # 发送报告
            self._send_quality_report(summary)

        except Exception as e:
            logger.error(f"生成质量报告失败: {e}")

    def run_automated_pipeline(self):
        """运行自动化质量流水线"""
        logger.info("开始执行自动化质量流水线...")

        try:
            # 导入自动化流水线
            if self.automated_pipeline is None:
                from .automated_quality_pipeline import AutomatedQualityPipeline
                self.automated_pipeline = AutomatedQualityPipeline(str(self.project_root))

            # 运行流水线
            results = self.automated_pipeline.run_pipeline()

            # 检查流水线执行结果
            summary = results.get('summary', {})
            overall_success = summary.get('overall_success', False)

            if not overall_success:
                # 发送告警
                failed_stages = [name for name, result in results.get('stages', {}).items()
                                 if not result.get('success', False)]
                self._send_alert(
                    'automated_pipeline',
                    f'自动化流水线执行失败: {", ".join(failed_stages)}',
                    results
                )

            logger.info(f"自动化质量流水线执行完成，状态: {'成功' if overall_success else '失败'}")

        except Exception as e:
            logger.error(f"自动化质量流水线执行失败: {e}")
            self._send_alert('automated_pipeline', f'自动化流水线执行失败: {e}', None)

    def _infer_report_type(self, filename: str) -> str:
        """推断报告类型"""
        if 'consistency' in filename:
            return 'consistency'
        elif 'sync' in filename:
            return 'sync'
        elif 'version' in filename:
            return 'version'
        else:
            return 'other'

    def _generate_quality_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量汇总"""
        summary = {
            'overall_score': 100,
            'issues_count': 0,
            'trends': {},
            'recommendations': []
        }

        # 计算总体评分
        total_score = 0
        count = 0

        for section_name, section_reports in report.get('sections', {}).items():
            for report_item in section_reports:
                data = report_item.get('data', {})

                if section_name == 'consistency':
                    score = data.get('summary', {}).get('consistency_score', 100)
                    total_score += score
                    count += 1

                    issues = len(data.get('summary', {}).get('errors', [])) + \
                        len(data.get('summary', {}).get('warnings', []))
                    summary['issues_count'] += issues

                elif section_name == 'sync':
                    score = data.get('summary', {}).get('sync_rate', 100)
                    total_score += score
                    count += 1

        if count > 0:
            summary['overall_score'] = total_score / count

        # 生成推荐建议
        if summary['overall_score'] < 95:
            summary['recommendations'].append("建议提高代码与文档的一致性")
        if summary['issues_count'] > 5:
            summary['recommendations'].append("建议及时处理发现的质量问题")

        return summary

    def _generate_html_quality_report(self, report: Dict[str, Any], summary: Dict[str, Any]) -> str:
        """生成HTML质量报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RQA2025 质量报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .section {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .score-high {{ color: #28a745; font-weight: bold; }}
                .score-medium {{ color: #ffc107; font-weight: bold; }}
                .score-low {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RQA2025 质量保障报告</h1>
                <p>报告生成时间: {report['generated_at']}</p>
                <p>报告周期: {report.get('period', 'weekly')}</p>
            </div>

            <div class="summary">
                <h2>📊 质量概览</h2>
                <p>总体评分: <span class="score-{'high' if summary['overall_score'] >= 95 else 'medium' if summary['overall_score'] >= 85 else 'low'}">{summary['overall_score']:.1f}%</span></p>
                <p>发现问题: {summary['issues_count']} 个</p>
            </div>
        """

        # 添加各部分报告
        for section_name, section_reports in report.get('sections', {}).items():
            html += f"<div class=\"section\"><h3>{section_name.title()} 报告</h3>"

            for report_item in section_reports[:3]:  # 只显示最近3个
                data = report_item.get('data', {})
                html += f"<p><strong>{report_item['file']}</strong></p>"

                if 'summary' in data:
                    summary_data = data['summary']
                    html += "<ul>"
                    for key, value in summary_data.items():
                        if isinstance(value, (int, float)):
                            html += f"<li>{key}: {value}</li>"
                    html += "</ul>"

            html += "</div>"

        # 添加推荐建议
        if summary.get('recommendations'):
            html += "<div class=\"section\"><h3>💡 改进建议</h3><ul>"
            for rec in summary['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"

        html += "</body></html>"
        return html

    def _send_alert(self, alert_type: str, message: str, data: Any = None):
        """发送告警"""
        if not self.config['notification']['enabled']:
            return

        logger.warning(f"发送告警: {alert_type} - {message}")

        # 邮件告警
        if self.config['notification']['email']['smtp_server']:
            self._send_email_alert(alert_type, message, data)

        # Slack告警
        if self.config['notification']['slack']['enabled']:
            self._send_slack_alert(alert_type, message, data)

    def _send_email_alert(self, alert_type: str, message: str, data: Any = None):
        """发送邮件告警"""
        try:
            email_config = self.config['notification']['email']

            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f'RQA2025 质量告警 - {alert_type}'

            body = f"""
            RQA2025 质量保障系统告警

            告警类型: {alert_type}
            消息: {message}
            时间: {datetime.now().isoformat()}

            详细信息:
            {json.dumps(data, indent=2, ensure_ascii=False) if data else '无'}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.sendmail(email_config['username'], email_config['recipients'], msg.as_string())
            server.quit()

            logger.info("邮件告警已发送")

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")

    def _send_slack_alert(self, alert_type: str, message: str, data: Any = None):
        """发送Slack告警"""
        try:
            import requests

            slack_config = self.config['notification']['slack']
            webhook_url = slack_config['webhook_url']

            payload = {
                'channel': slack_config['channel'],
                'text': f':warning: RQA2025 质量告警\n类型: {alert_type}\n消息: {message}',
                'attachments': [{
                    'text': json.dumps(data, indent=2, ensure_ascii=False) if data else '无详细信息'
                }] if data else []
            }

            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Slack告警已发送")
            else:
                logger.error(f"发送Slack告警失败: {response.status_code}")

        except ImportError:
            logger.warning("requests库未安装，跳过Slack告警")
        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")

    def _send_quality_report(self, summary: Dict[str, Any]):
        """发送质量报告"""
        if summary['overall_score'] < 95 or summary['issues_count'] > 3:
            self._send_alert(
                'quality_report',
                f'质量评分: {summary["overall_score"]:.1f}%, 问题数量: {summary["issues_count"]}',
                summary
            )

    def start_scheduler(self):
        """启动调度器"""
        logger.info("启动质量保障调度器...")

        # 设置调度
        self.setup_schedules()

        # 启动调度循环
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次

        except KeyboardInterrupt:
            logger.info("调度器已停止")
        except Exception as e:
            logger.error(f"调度器运行出错: {e}")

    def run_manual_task(self, task: str):
        """手动运行任务"""
        task_map = {
            'consistency': self.run_consistency_check,
            'sync': self.run_doc_sync,
            'version': self.run_version_check,
            'report': self.generate_quality_report,
            'pipeline': self.run_automated_pipeline
        }

        if task in task_map:
            logger.info(f"手动执行任务: {task}")
            task_map[task]()
        else:
            logger.error(f"未知任务: {task}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 质量保障调度器')
    parser.add_argument('--project-root', help='项目根目录路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('command', choices=['start', 'run', 'config'],
                        help='执行命令')
    parser.add_argument('--task', choices=['consistency', 'sync', 'version', 'report', 'pipeline'],
                        help='手动执行的任务')

    args = parser.parse_args()

    # 创建调度器
    scheduler = QualityScheduler(args.project_root)

    if args.command == 'start':
        # 启动调度器
        scheduler.start_scheduler()

    elif args.command == 'run' and args.task:
        # 手动运行任务
        scheduler.run_manual_task(args.task)

    elif args.command == 'config':
        # 显示配置
        print(json.dumps(scheduler.config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
