#!/usr/bin/env python3
"""
动态宇宙管理系统培训通知脚本
"""

import os
import json
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_notification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingNotification:
    """培训通知管理器"""

    def __init__(self, training_date=None):
        self.training_date = training_date or datetime.now()
        self.notification_id = f"TRAIN_NOTIFY_{self.training_date.strftime('%Y%m%d')}"

    def create_training_announcement(self):
        """创建培训公告"""
        logger.info("创建培训公告...")

        announcement = {
            "notification_id": self.notification_id,
            "title": "动态宇宙管理系统技术培训通知",
            "publish_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_info": {
                "title": "动态宇宙管理系统技术培训",
                "start_date": (self.training_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                "duration": "5天",
                "schedule": "09:00-11:00, 14:00-16:00",
                "location": "线上/线下混合",
                "trainer": "系统架构师"
            },
            "target_audience": [
                "开发人员",
                "运维人员",
                "业务分析师",
                "项目经理",
                "技术负责人"
            ],
            "prerequisites": [
                "Python基础编程能力",
                "金融数据基础知识",
                "Linux基础操作能力",
                "Git版本控制基础"
            ],
            "training_objectives": [
                "掌握动态宇宙管理系统架构",
                "学会系统配置和部署",
                "理解智能更新机制",
                "掌握性能监控和优化",
                "能够独立运维系统"
            ],
            "registration_info": {
                "deadline": (self.training_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "max_participants": 20,
                "registration_method": "邮件报名",
                "contact_email": "training@company.com",
                "contact_phone": "400-123-4567"
            },
            "materials_provided": [
                "技术研讨会指南",
                "实践练习手册",
                "评估指南",
                "演示脚本",
                "部署指南"
            ],
            "certification": {
                "type": "内部技术认证",
                "levels": ["初级", "中级", "高级"],
                "validity": "2年",
                "renewal": "需要重新评估"
            }
        }

        # 保存公告
        announcement_file = f"reports/training/announcement_{self.notification_id}.json"
        os.makedirs(os.path.dirname(announcement_file), exist_ok=True)

        with open(announcement_file, 'w', encoding='utf-8') as f:
            json.dump(announcement, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 培训公告已创建: {announcement_file}")
        return announcement

    def create_email_template(self):
        """创建邮件模板"""
        logger.info("创建邮件模板...")

        email_template = {
            "subject": "【重要通知】动态宇宙管理系统技术培训报名",
            "sender": "training@company.com",
            "recipients": [
                "dev-team@company.com",
                "ops-team@company.com",
                "pm-team@company.com",
                "tech-leads@company.com"
            ],
            "cc": ["hr@company.com", "training-admin@company.com"],
            "body_template": """
尊敬的同事：

您好！

我们很高兴地通知您，动态宇宙管理系统技术培训即将开始。这是一次重要的技术能力提升机会，诚邀您参加。

【培训详情】
• 培训名称：动态宇宙管理系统技术培训
• 培训时间：{start_date} 至 {end_date}（共5天）
• 培训方式：线上/线下混合
• 每日安排：上午 09:00-11:00，下午 14:00-16:00
• 培训讲师：系统架构师

【培训目标】
• 掌握动态宇宙管理系统架构和核心组件
• 学会系统配置、部署和运维
• 理解智能更新机制和性能优化
• 获得内部技术认证

【报名要求】
• 报名截止：{registration_deadline}
• 人数限制：最多20人
• 报名方式：回复此邮件，注明姓名、部门、联系方式

【培训材料】
• 技术研讨会指南
• 实践练习手册  
• 评估指南
• 演示脚本
• 部署指南

【认证体系】
• 认证类型：内部技术认证
• 认证等级：初级、中级、高级
• 有效期：2年

请有意参加培训的同事尽快报名，我们将根据报名情况安排培训。

如有疑问，请联系：
培训管理员：training-admin@company.com
技术咨询：tech-support@company.com

谢谢！

培训团队
{company_name}
            """,
            "variables": {
                "start_date": (self.training_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                "end_date": (self.training_date + timedelta(days=11)).strftime("%Y-%m-%d"),
                "registration_deadline": (self.training_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "company_name": "XX科技有限公司"
            }
        }

        # 保存邮件模板
        email_file = f"reports/training/email_template_{self.notification_id}.json"
        os.makedirs(os.path.dirname(email_file), exist_ok=True)

        with open(email_file, 'w', encoding='utf-8') as f:
            json.dump(email_template, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 邮件模板已创建: {email_file}")
        return email_template

    def create_registration_form(self):
        """创建报名表"""
        logger.info("创建报名表...")

        registration_form = {
            "form_id": f"REG_{self.notification_id}",
            "title": "动态宇宙管理系统技术培训报名表",
            "fields": [
                {
                    "name": "employee_id",
                    "label": "员工编号",
                    "type": "text",
                    "required": True,
                    "validation": "alphanumeric"
                },
                {
                    "name": "name",
                    "label": "姓名",
                    "type": "text",
                    "required": True,
                    "validation": "chinese_name"
                },
                {
                    "name": "department",
                    "label": "部门",
                    "type": "select",
                    "required": True,
                    "options": [
                        "技术部",
                        "运维部",
                        "产品部",
                        "项目管理部",
                        "其他"
                    ]
                },
                {
                    "name": "position",
                    "label": "职位",
                    "type": "text",
                    "required": True
                },
                {
                    "name": "email",
                    "label": "邮箱",
                    "type": "email",
                    "required": True,
                    "validation": "email_format"
                },
                {
                    "name": "phone",
                    "label": "联系电话",
                    "type": "tel",
                    "required": True,
                    "validation": "phone_format"
                },
                {
                    "name": "experience_level",
                    "label": "技术经验水平",
                    "type": "select",
                    "required": True,
                    "options": [
                        "初级（1-3年）",
                        "中级（3-5年）",
                        "高级（5年以上）"
                    ]
                },
                {
                    "name": "python_skill",
                    "label": "Python编程能力",
                    "type": "select",
                    "required": True,
                    "options": [
                        "基础",
                        "熟练",
                        "精通"
                    ]
                },
                {
                    "name": "motivation",
                    "label": "参加培训的动机",
                    "type": "textarea",
                    "required": True,
                    "max_length": 500
                },
                {
                    "name": "availability",
                    "label": "培训时间安排",
                    "type": "checkbox",
                    "required": True,
                    "options": [
                        "可以参加全部培训",
                        "需要调整部分时间",
                        "只能参加部分培训"
                    ]
                }
            ],
            "submission_info": {
                "deadline": (self.training_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "submission_method": "email",
                "contact_email": "training@company.com",
                "auto_reply": True
            }
        }

        # 保存报名表
        form_file = f"reports/training/registration_form_{self.notification_id}.json"
        os.makedirs(os.path.dirname(form_file), exist_ok=True)

        with open(form_file, 'w', encoding='utf-8') as f:
            json.dump(registration_form, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 报名表已创建: {form_file}")
        return registration_form

    def create_training_calendar(self):
        """创建培训日历"""
        logger.info("创建培训日历...")

        calendar_events = []
        training_start = self.training_date + timedelta(days=7)

        for day in range(5):
            current_date = training_start + timedelta(days=day)

            # 上午课程
            morning_event = {
                "date": current_date.strftime("%Y-%m-%d"),
                "time": "09:00-11:00",
                "title": f"第{day+1}天上午 - 理论学习",
                "description": f"动态宇宙管理系统第{day+1}天上午课程",
                "location": "线上会议室",
                "type": "training",
                "day_number": day + 1,
                "session": "morning"
            }
            calendar_events.append(morning_event)

            # 下午课程
            afternoon_event = {
                "date": current_date.strftime("%Y-%m-%d"),
                "time": "14:00-16:00",
                "title": f"第{day+1}天下午 - 实践操作",
                "description": f"动态宇宙管理系统第{day+1}天下午课程",
                "location": "线上会议室",
                "type": "training",
                "day_number": day + 1,
                "session": "afternoon"
            }
            calendar_events.append(afternoon_event)

        calendar = {
            "calendar_id": f"CAL_{self.notification_id}",
            "title": "动态宇宙管理系统培训日历",
            "description": "培训期间的所有课程安排",
            "events": calendar_events,
            "reminders": [
                {
                    "type": "email",
                    "timing": "1_day_before",
                    "recipients": "all_participants"
                },
                {
                    "type": "calendar",
                    "timing": "30_minutes_before",
                    "recipients": "all_participants"
                }
            ]
        }

        # 保存日历
        calendar_file = f"reports/training/training_calendar_{self.notification_id}.json"
        os.makedirs(os.path.dirname(calendar_file), exist_ok=True)

        with open(calendar_file, 'w', encoding='utf-8') as f:
            json.dump(calendar, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 培训日历已创建: {calendar_file}")
        return calendar

    def generate_notification_report(self):
        """生成通知报告"""
        logger.info("生成通知报告...")

        report = {
            "report_info": {
                "title": "培训通知发布报告",
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notification_id": self.notification_id,
                "status": "published"
            },
            "notification_components": {
                "announcement": "created",
                "email_template": "created",
                "registration_form": "created",
                "training_calendar": "created"
            },
            "distribution_channels": [
                "公司内部邮件系统",
                "企业微信群",
                "公司公告栏",
                "部门会议通知"
            ],
            "target_audience_stats": {
                "total_departments": 5,
                "estimated_participants": 15,
                "response_rate_expected": 80
            },
            "timeline": {
                "notification_sent": self.training_date.strftime("%Y-%m-%d"),
                "registration_deadline": (self.training_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                "training_start": (self.training_date + timedelta(days=7)).strftime("%Y-%m-%d"),
                "training_end": (self.training_date + timedelta(days=11)).strftime("%Y-%m-%d")
            },
            "next_actions": [
                "监控报名情况",
                "准备培训环境",
                "安排培训讲师",
                "准备培训材料",
                "进行预培训测试"
            ]
        }

        # 保存报告
        report_file = f"reports/training/notification_report_{self.notification_id}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"[OK] 通知报告已生成: {report_file}")
        return report

    def publish_notification(self):
        """发布培训通知"""
        logger.info("开始发布培训通知...")

        steps = [
            ("创建培训公告", self.create_training_announcement),
            ("创建邮件模板", self.create_email_template),
            ("创建报名表", self.create_registration_form),
            ("创建培训日历", self.create_training_calendar),
            ("生成通知报告", self.generate_notification_report)
        ]

        results = {}
        for step_name, step_func in steps:
            logger.info(f"\n=== {step_name} ===")
            try:
                result = step_func()
                results[step_name] = "success"
                logger.info(f"[OK] {step_name} 完成")
            except Exception as e:
                results[step_name] = f"failed: {str(e)}"
                logger.error(f"[ERROR] {step_name} 失败: {e}")

        # 生成总结
        success_count = sum(1 for result in results.values() if result == "success")
        total_count = len(results)

        logger.info(f"\n=== 培训通知发布完成 ===")
        logger.info(f"成功步骤: {success_count}/{total_count}")

        if success_count == total_count:
            logger.info("[SUCCESS] 培训通知发布完成！")
            logger.info("下一步建议:")
            logger.info("1. 通过邮件系统发送培训通知")
            logger.info("2. 在企业微信群发布培训信息")
            logger.info("3. 在公司公告栏张贴培训海报")
            logger.info("4. 在部门会议上口头通知")
            logger.info("5. 开始收集报名信息")
        else:
            logger.warning("[WARN] 部分步骤失败，请检查并修复")

        return results


def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1:
        training_date_str = sys.argv[1]
        try:
            training_date = datetime.strptime(training_date_str, "%Y-%m-%d")
        except ValueError:
            logger.error("日期格式错误，请使用 YYYY-MM-DD 格式")
            sys.exit(1)
    else:
        training_date = datetime.now()

    notifier = TrainingNotification(training_date)
    results = notifier.publish_notification()

    return results


if __name__ == "__main__":
    main()
