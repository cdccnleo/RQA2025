#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全培训和意识提升脚本
"""

import json


class SecurityTrainingManager:
    """安全培训管理器"""

    def __init__(self):
        self.training_programs = {}
        self.awareness_campaigns = {}
        self.assessments = {}

    def create_security_training_program(self):
        """创建安全培训课程"""
        training_program = {
            "program_name": "RQA2025安全培训计划",
            "duration": "3个月",
            "target_audience": ["开发团队", "运维团队", "业务团队", "管理层"],
            "training_modules": [
                {
                    "module_id": "SEC_101",
                    "title": "安全基础知识",
                    "duration": "2小时",
                    "format": "在线课程",
                    "content": [
                        "安全基本概念",
                        "常见安全威胁",
                        "安全最佳实践",
                        "安全意识重要性"
                    ],
                    "target_audience": ["全体员工"],
                    "frequency": "入职培训 + 年度复习"
                },
                {
                    "module_id": "SEC_201",
                    "title": "开发安全编码",
                    "duration": "4小时",
                    "format": "工作坊",
                    "content": [
                        "OWASP Top 10",
                        "安全编码标准",
                        "代码审查技巧",
                        "安全测试方法"
                    ],
                    "target_audience": ["开发团队"],
                    "frequency": "每季度一次"
                },
                {
                    "module_id": "SEC_301",
                    "title": "安全运维实践",
                    "duration": "3小时",
                    "format": "实操培训",
                    "content": [
                        "系统加固",
                        "监控和告警",
                        "应急响应",
                        "合规要求"
                    ],
                    "target_audience": ["运维团队"],
                    "frequency": "每半年一次"
                },
                {
                    "module_id": "SEC_401",
                    "title": "安全管理与治理",
                    "duration": "2小时",
                    "format": "研讨会",
                    "content": [
                        "安全策略制定",
                        "风险管理",
                        "合规管理",
                        "安全文化建设"
                    ],
                    "target_audience": ["管理层"],
                    "frequency": "每半年一次"
                }
            ],
            "assessment_methods": [
                {
                    "method": "在线考试",
                    "coverage": "理论知识",
                    "passing_score": "80%",
                    "frequency": "培训后立即"
                },
                {
                    "method": "实操考核",
                    "coverage": "实践技能",
                    "passing_score": "90%",
                    "frequency": "培训后一个月"
                },
                {
                    "method": "模拟演练",
                    "coverage": "应急响应",
                    "passing_score": "合格",
                    "frequency": "每季度一次"
                }
            ],
            "certification": {
                "certificates": [
                    "安全意识认证",
                    "安全编码认证",
                    "安全运维认证",
                    "安全管理认证"
                ],
                "renewal_period": "2年",
                "recertification_requirements": "继续教育 + 考核"
            }
        }

        return training_program

    def create_awareness_campaign(self):
        """创建安全意识宣传活动"""
        awareness_campaign = {
            "campaign_name": "安全月宣传活动",
            "duration": "1个月",
            "target_audience": "全体员工",
            "campaign_activities": [
                {
                    "activity": "安全海报展览",
                    "description": "在办公区域展示安全宣传海报",
                    "frequency": "持续展示",
                    "materials": ["安全威胁海报", "最佳实践海报", "案例分享海报"]
                },
                {
                    "activity": "安全知识竞赛",
                    "description": "在线安全知识问答竞赛",
                    "frequency": "每周一次",
                    "prizes": ["礼品卡", "荣誉证书", "团队奖励"]
                },
                {
                    "activity": "安全故事分享",
                    "description": "分享真实安全事件和教训",
                    "frequency": "每周二",
                    "format": "15分钟线上分享"
                },
                {
                    "activity": "钓鱼邮件演练",
                    "description": "模拟钓鱼邮件测试员工警惕性",
                    "frequency": "每月一次",
                    "follow_up": "培训反馈"
                },
                {
                    "activity": "安全早餐会",
                    "description": "早餐时间安全话题讨论",
                    "frequency": "每周五",
                    "topics": ["最新安全威胁", "最佳实践分享", "Q&A环节"]
                }
            ],
            "communication_channels": [
                {
                    "channel": "企业微信",
                    "content": ["每日安全提示", "活动通知", "知识分享"],
                    "frequency": "每日"
                },
                {
                    "channel": "内部网站",
                    "content": ["安全资源库", "培训资料", "案例库"],
                    "frequency": "持续更新"
                },
                {
                    "channel": "邮件通讯",
                    "content": ["周安全摘要", "重要安全通知"],
                    "frequency": "每周"
                },
                {
                    "channel": "线下活动",
                    "content": ["培训课程", "研讨会", "应急演练"],
                    "frequency": "每月"
                }
            ],
            "metrics_and_evaluation": [
                {
                    "metric": "参与率",
                    "target": ">80%",
                    "measurement": "活动报名和参与统计"
                },
                {
                    "metric": "知识提升",
                    "target": "+20%",
                    "measurement": "前后测试对比"
                },
                {
                    "metric": "行为改变",
                    "target": "+30%",
                    "measurement": "安全事件报告率"
                },
                {
                    "metric": "满意度",
                    "target": ">4.0/5.0",
                    "measurement": "活动满意度调查"
                }
            ]
        }

        return awareness_campaign

    def create_security_assessment(self):
        """创建安全评估"""
        assessment = {
            "assessment_name": "年度安全意识评估",
            "assessment_type": "综合评估",
            "target_audience": "全体员工",
            "assessment_components": [
                {
                    "component": "知识测试",
                    "weight": 40,
                    "questions": 30,
                    "topics": ["安全基本概念", "威胁识别", "最佳实践"],
                    "format": "在线选择题"
                },
                {
                    "component": "行为观察",
                    "weight": 30,
                    "criteria": [
                        "密码管理",
                        "设备安全",
                        "信息处理",
                        "异常报告"
                    ],
                    "format": "主管评估"
                },
                {
                    "component": "实践考核",
                    "weight": 30,
                    "scenarios": [
                        "钓鱼邮件识别",
                        "安全配置检查",
                        "事件响应演练"
                    ],
                    "format": "实操测试"
                }
            ],
            "scoring_system": {
                "excellent": "90-100分",
                "good": "80-89分",
                "satisfactory": "70-79分",
                "needs_improvement": "60-69分",
                "unsatisfactory": "<60分"
            },
            "remediation_plan": {
                "needs_improvement": [
                    "参加补习课程",
                    "增加实践练习",
                    "接受一对一辅导"
                ],
                "unsatisfactory": [
                    "强制参加培训",
                    "增加监督检查",
                    "制定改进计划"
                ]
            },
            "frequency": "年度评估 + 季度抽样"
        }

        return assessment

    def generate_training_materials(self):
        """生成培训资料"""
        training_materials = {
            "material_categories": {
                "presentations": [
                    {
                        "title": "安全基础知识培训",
                        "duration": "2小时",
                        "slides": 45,
                        "target_audience": "全体员工",
                        "language": "中文"
                    },
                    {
                        "title": "OWASP Top 10详解",
                        "duration": "3小时",
                        "slides": 60,
                        "target_audience": "开发团队",
                        "language": "中文"
                    }
                ],
                "videos": [
                    {
                        "title": "密码安全最佳实践",
                        "duration": "10分钟",
                        "format": "动画视频",
                        "target_audience": "全体员工"
                    },
                    {
                        "title": "钓鱼邮件识别指南",
                        "duration": "15分钟",
                        "format": "情景剧",
                        "target_audience": "全体员工"
                    }
                ],
                "handouts": [
                    {
                        "title": "安全检查清单",
                        "pages": 5,
                        "format": "PDF",
                        "target_audience": "全体员工"
                    },
                    {
                        "title": "应急响应指南",
                        "pages": 12,
                        "format": "PDF",
                        "target_audience": "关键岗位"
                    }
                ],
                "interactive_content": [
                    {
                        "title": "安全知识问答系统",
                        "questions": 200,
                        "difficulty_levels": 3,
                        "target_audience": "全体员工"
                    },
                    {
                        "title": "安全情景模拟器",
                        "scenarios": 10,
                        "difficulty_levels": 3,
                        "target_audience": "关键岗位"
                    }
                ]
            },
            "distribution_plan": {
                "online_platform": "企业学习管理系统",
                "download_access": "内部文件共享系统",
                "print_materials": "培训教室和会议室",
                "mobile_access": "企业APP和微信小程序"
            },
            "maintenance_plan": {
                "content_review": "每季度更新",
                "accuracy_check": "每半年验证",
                "feedback_collection": "每次培训后",
                "improvement_implementation": "每季度实施"
            }
        }

        return training_materials


def run_training_program():
    """运行培训计划"""
    print("开始制定安全培训和意识提升计划...")

    manager = SecurityTrainingManager()

    # 创建培训课程
    print("\n1. 创建安全培训课程:")
    training_program = manager.create_security_training_program()
    print(f"   培训课程数量: {len(training_program['training_modules'])}")
    print(f"   目标受众: {len(training_program['target_audience'])}类")
    print(f"   认证数量: {len(training_program['certification']['certificates'])}")

    # 创建意识宣传活动
    print("\n2. 创建安全意识宣传活动:")
    awareness_campaign = manager.create_awareness_campaign()
    print(f"   宣传活动数量: {len(awareness_campaign['campaign_activities'])}")
    print(f"   沟通渠道数量: {len(awareness_campaign['communication_channels'])}")
    print(f"   评估指标数量: {len(awareness_campaign['metrics_and_evaluation'])}")

    # 创建安全评估
    print("\n3. 创建安全评估:")
    assessment = manager.create_security_assessment()
    print(f"   评估组件数量: {len(assessment['assessment_components'])}")
    print(f"   评分等级数量: {len(assessment['scoring_system'])}")

    # 生成培训资料
    print("\n4. 生成培训资料:")
    training_materials = manager.generate_training_materials()
    print(f"   资料类别数量: {len(training_materials['material_categories'])}")
    print(f"   演示文稿数量: {len(training_materials['material_categories']['presentations'])}")
    print(f"   视频资料数量: {len(training_materials['material_categories']['videos'])}")

    return {
        "training_program": training_program,
        "awareness_campaign": awareness_campaign,
        "assessment": assessment,
        "training_materials": training_materials
    }


def generate_training_report(results):
    """生成培训报告"""
    print("生成安全培训和意识提升报告...")

    report = {
        "security_training_report": {
            "program_summary": {
                "total_modules": len(results["training_program"]["training_modules"]),
                "total_audience_groups": len(results["training_program"]["target_audience"]),
                "total_certificates": len(results["training_program"]["certification"]["certificates"]),
                "total_materials": sum(
                    len(materials) for materials in results["training_materials"]["material_categories"].values()
                )
            },
            "implementation_plan": {
                "phase1": {
                    "name": "基础培训阶段",
                    "duration": "4月20日-4月30日",
                    "focus": "安全意识基础培训",
                    "modules": ["SEC_101"],
                    "participants": "全体员工"
                },
                "phase2": {
                    "name": "专业培训阶段",
                    "duration": "5月1日-5月15日",
                    "focus": "专业技能提升培训",
                    "modules": ["SEC_201", "SEC_301"],
                    "participants": "开发和运维团队"
                },
                "phase3": {
                    "name": "管理培训阶段",
                    "duration": "5月16日-5月31日",
                    "focus": "安全管理和治理培训",
                    "modules": ["SEC_401"],
                    "participants": "管理层和关键岗位"
                }
            },
            "awareness_activities": {
                "campaign_duration": "4月20日-5月20日",
                "total_activities": len(results["awareness_campaign"]["campaign_activities"]),
                "communication_channels": len(results["awareness_campaign"]["communication_channels"]),
                "expected_participation": "85%",
                "budget_allocation": "5万元"
            },
            "assessment_schedule": {
                "baseline_assessment": "4月25日",
                "mid_term_assessment": "5月10日",
                "final_assessment": "5月25日",
                "improvement_target": "+25%知识提升"
            },
            "success_metrics": {
                "training_completion_rate": ">90%",
                "assessment_passing_rate": ">85%",
                "awareness_participation_rate": ">80%",
                "incident_reporting_rate": "+50%",
                "security_culture_score": ">4.0/5.0"
            },
            "budget_and_resources": {
                "total_budget": "15万元",
                "breakdown": {
                    "培训师资": "5万元",
                    "培训材料": "3万元",
                    "活动策划": "4万元",
                    "评估工具": "2万元",
                    "平台建设": "1万元"
                },
                "resource_allocation": {
                    "培训专员": "2人",
                    "内容开发": "1人",
                    "活动策划": "1人",
                    "评估分析": "1人"
                }
            },
            "risk_mitigation": {
                "low_participation": "强制培训 + 激励措施",
                "content_outdated": "季度内容更新",
                "assessment_cheating": "多格式评估组合",
                "resource_shortage": "外包培训服务"
            }
        }
    }

    return report


def main():
    """主函数"""
    print("开始安全培训和意识提升计划制定...")

    # 运行培训计划
    training_results = run_training_program()

    # 生成培训报告
    training_report = generate_training_report(training_results)

    # 合并结果
    final_results = {
        "training_results": training_results,
        "training_report": training_report
    }

    # 保存结果
    with open('security_training_awareness_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\n安全培训和意识提升计划制定完成，结果已保存到 security_training_awareness_results.json")

    # 输出关键指标
    summary = training_report["security_training_report"]["program_summary"]
    print("\n培训计划总结:")
    print(f"  培训模块数量: {summary['total_modules']}")
    print(f"  受众群体数量: {summary['total_audience_groups']}")
    print(f"  认证数量: {summary['total_certificates']}")
    print(f"  培训资料总数: {summary['total_materials']}")

    implementation = training_report["security_training_report"]["implementation_plan"]
    print(f"\n实施计划:")
    for phase_name, phase_info in implementation.items():
        print(f"  {phase_name}: {phase_info['name']} ({phase_info['duration']})")

    return final_results


if __name__ == '__main__':
    main()
