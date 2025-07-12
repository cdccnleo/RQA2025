import logging
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
import shutil
import os
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProjectFinalizer:
    """项目正式结项处理器"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.final_report = {
            'project': self.config['project_name'],
            'version': self.config['version'],
            'closure_date': datetime.now().isoformat()
        }

    def _load_config(self, path: str) -> Dict:
        """加载结项配置"""
        logger.info(f"Loading finalization config from {path}")
        with open(path) as f:
            return json.load(f)

    def execute_finalization(self):
        """执行项目结项流程"""
        logger.info("Starting project finalization process")

        try:
            # 1. 召开结项评审会
            self._conduct_closure_meeting()

            # 2. 完成财务结算
            self._finalize_financials()

            # 3. 归档项目资料
            self._archive_project_assets()

            # 4. 发送结项通知
            self._send_closure_notice()

            # 5. 移交维护团队
            self._handover_to_maintenance()

            logger.info("Project finalized successfully")

        except Exception as e:
            logger.error(f"Project finalization failed: {e}")
            raise

    def _conduct_closure_meeting(self):
        """召开结项评审会"""
        logger.info("Conducting project closure meeting")

        # 模拟会议记录
        meeting_minutes = {
            'date': datetime.now().isoformat(),
            'attendees': self.config['stakeholders'],
            'agenda': [
                "项目成果回顾",
                "经验教训总结",
                "后续维护计划",
                "结项确认"
            ],
            'decisions': [
                "一致同意项目正式结项",
                "批准维护团队交接方案",
                "通过经验教训报告"
            ],
            'signatures': {
                'pm': "张明",
                'tech_lead': "李雷",
                'product_owner': "王芳",
                'finance': "刘伟"
            }
        }

        # 保存会议纪要
        minutes_file = f"docs/meetings/closure_{datetime.now().strftime('%Y%m%d')}.json"
        with open(minutes_file, 'w') as f:
            json.dump(meeting_minutes, f, indent=2, ensure_ascii=False)

        self.final_report['closure_meeting'] = minutes_file
        logger.info(f"Meeting minutes saved to {minutes_file}")

    def _finalize_financials(self):
        """完成财务结算"""
        logger.info("Finalizing project financials")

        # 模拟财务结算
        financial_report = {
            'total_budget': self.config['budget'],
            'actual_spending': self.config['actual_cost'],
            'savings': round(self.config['budget'] - self.config['actual_cost'], 2),
            'vendor_payments': {
                'data_services': "Paid",
                'cloud_services': "Paid",
                'consulting': "Paid"
            },
            'asset_depreciation': {
                'equipment': "Processed",
                'licenses': "Terminated"
            }
        }

        # 保存财务报告
        finance_file = "reports/financial_closure.json"
        with open(finance_file, 'w') as f:
            json.dump(financial_report, f, indent=2)

        self.final_report['financial_closure'] = finance_file
        logger.info(f"Financial report saved to {finance_file}")

    def _archive_project_assets(self):
        """归档项目资料"""
        logger.info("Archiving project assets")

        archive_dir = f"archives/{self.config['project_name']}_v{self.config['version']}"
        os.makedirs(archive_dir, exist_ok=True)

        # 归档关键资产
        assets_to_archive = [
            ('src', 'code'),
            ('docs', 'documentation'),
            ('reports', 'reports'),
            ('config', 'configurations')
        ]

        archived = []
        for src, dest in assets_to_archive:
            dest_path = os.path.join(archive_dir, dest)
            shutil.copytree(src, dest_path)
            archived.append(dest_path)
            logger.info(f"Archived {src} to {dest_path}")

        # 生成归档清单
        archive_manifest = {
            'archive_date': datetime.now().isoformat(),
            'location': archive_dir,
            'contents': archived,
            'backup_verified': False  # 将由备份系统更新
        }

        manifest_file = os.path.join(archive_dir, "manifest.json")
        with open(manifest_file, 'w') as f:
            json.dump(archive_manifest, f, indent=2)

        self.final_report['archive_location'] = archive_dir
        logger.info(f"Archive manifest created at {manifest_file}")

    def _send_closure_notice(self):
        """发送结项通知"""
        logger.info("Sending project closure notice")

        # 准备邮件内容
        email_content = f"""
        <html>
        <body>
            <h2>项目结项通知</h2>
            <p><strong>项目名称:</strong> {self.config['project_name']}</p>
            <p><strong>版本号:</strong> {self.config['version']}</p>
            <p><strong>结项日期:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
            
            <h3>关键成果</h3>
            <ul>
                <li>交付功能模块: {len(self.config['delivered_modules'])}个</li>
                <li>测试覆盖率: {self.config['test_coverage']}%</li>
                <li>实际支出: ¥{self.config['actual_cost']:,}</li>
                <li>较预算节省: {round((self.config['budget']-self.config['actual_cost'])/self.config['budget']*100,1)}%</li>
            </ul>
            
            <h3>后续计划</h3>
            <p>项目将转入维护阶段，由{self.config['maintenance_team']}团队负责。</p>
            
            <p>详细信息请查阅附件结项报告。</p>
        </body>
        </html>
        """

        # 模拟发送邮件
        msg = MIMEText(email_content, 'html')
        msg['Subject'] = f"[结项通知] {self.config['project_name']} v{self.config['version']}"
        msg['From'] = self.config['email_sender']
        msg['To'] = ", ".join(self.config['email_recipients'])

        try:
            # 实际实现会调用邮件服务器发送
            logger.info(f"Would send email to: {self.config['email_recipients']}")
            logger.debug(f"Email content:\n{email_content}")

            self.final_report['closure_notice'] = {
                'sent_to': self.config['email_recipients'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to send closure notice: {e}")
            raise

    def _handover_to_maintenance(self):
        """移交维护团队"""
        logger.info("Handing over to maintenance team")

        # 准备移交清单
        handover_items = [
            {
                'item': '代码仓库',
                'access_granted': True,
                'location': 'git@company.com/rqa2025.git',
                'contact': 'devops@company.com'
            },
            {
                'item': '监控系统',
                'access_granted': True,
                'location': 'grafana.company.com/d/rqa2025',
                'contact': 'monitoring@company.com'
            },
            {
                'item': '文档库',
                'access_granted': True,
                'location': 'confluence.company.com/rqa2025',
                'contact': 'tech_writing@company.com'
            },
            {
                'item': '生产访问权限',
                'access_granted': False,  # 需要单独审批
                'location': 'N/A',
                'contact': 'security@company.com'
            }
        ]

        # 保存移交记录
        handover_file = "reports/maintenance_handover.json"
        with open(handover_file, 'w') as f:
            json.dump(handover_items, f, indent=2)

        self.final_report['maintenance_handover'] = handover_file
        logger.info(f"Handover report saved to {handover_file}")

        # 更新项目状态
        self._update_project_status('closed')

    def _update_project_status(self, status: str):
        """更新项目状态"""
        logger.info(f"Updating project status to '{status}'")

        # 实际实现会更新项目管理系统
        self.final_report['final_status'] = status
        logger.info(f"Project status set to '{status}'")

def main():
    """主执行流程"""
    try:
        # 初始化结项处理器
        print("正在加载配置文件...")
        finalizer = ProjectFinalizer("config/finalization.json")
        print("配置文件加载成功")

        # 执行结项流程
        print("开始执行结项流程...")
        finalizer.execute_finalization()
        print("结项流程执行完成")

        # 保存最终报告
        report_file = "reports/project_final_report.json"
        print(f"正在保存最终报告到 {report_file}")
        with open(report_file, 'w') as f:
            json.dump(finalizer.final_report, f, indent=2)

        logger.info(f"Final project report saved to {report_file}")
        logger.info("Project closure process completed successfully")
        print("项目结项流程完成")

    except Exception as e:
        logger.error(f"Project finalization failed: {e}")
        print(f"发生错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
