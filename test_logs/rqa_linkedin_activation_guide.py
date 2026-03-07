#!/usr/bin/env python3
"""
RQA LinkedIn Premium激活执行指南
详细步骤指导LinkedIn账户升级和公司页面设置
"""

from typing import Dict, List, Any
from datetime import datetime
import json

class RQALinkedInActivationGuide:
    """RQA LinkedIn激活执行指南"""

    def __init__(self):
        self.activation_date = datetime(2027, 1, 1)
        self.account_manager = "Sarah Chen"
        self.company_name = "RQA Technologies Inc."
        self.target_positions = ["CEO", "CTO", "CPO"]

        self.activation_steps = self._define_activation_steps()
        self.company_page_content = self._define_company_page_content()
        self.job_posting_templates = self._define_job_posting_templates()
        self.targeting_strategy = self._define_targeting_strategy()

    def _define_activation_steps(self) -> Dict[str, Any]:
        """定义激活步骤"""
        return {
            "step_1_account_upgrade": {
                "step_number": 1,
                "title": "LinkedIn Premium Business Account Upgrade",
                "estimated_time": "30 minutes",
                "status": "pending",
                "prerequisites": [
                    "Valid company email address",
                    "Company credit card for billing",
                    "Basic LinkedIn account (if not existing, create one)"
                ],
                "detailed_instructions": [
                    {
                        "action": "Navigate to LinkedIn Premium",
                        "description": "Open web browser and go to https://www.linkedin.com/premium",
                        "screenshot_needed": False,
                        "expected_result": "Premium plans page loads"
                    },
                    {
                        "action": "Select Business Plan",
                        "description": "Click on 'Business' plan option ($8,000/month)",
                        "screenshot_needed": True,
                        "expected_result": "Business plan selected"
                    },
                    {
                        "action": "Enter Billing Information",
                        "description": "Fill in company billing details using company credit card",
                        "screenshot_needed": False,
                        "expected_result": "Billing information accepted"
                    },
                    {
                        "action": "Confirm Upgrade",
                        "description": "Review terms and click 'Upgrade Now' button",
                        "screenshot_needed": True,
                        "expected_result": "Payment processed successfully"
                    },
                    {
                        "action": "Verify Activation",
                        "description": "Check account settings to confirm Premium Business status",
                        "screenshot_needed": True,
                        "expected_result": "Premium Business badge visible"
                    }
                ],
                "success_criteria": [
                    "Payment confirmation email received",
                    "Premium Business badge shows in profile",
                    "Access to business analytics and insights",
                    "Ability to create company pages"
                ],
                "troubleshooting": [
                    {
                        "issue": "Payment declined",
                        "solution": "Verify card details, try different card, or contact bank"
                    },
                    {
                        "issue": "Page not loading",
                        "solution": "Clear browser cache, try different browser, or check internet connection"
                    }
                ]
            },
            "step_2_company_page_creation": {
                "step_number": 2,
                "title": "RQA Technologies Inc. Company Page Creation",
                "estimated_time": "60 minutes",
                "status": "pending",
                "prerequisites": [
                    "Premium Business account activated",
                    "Company logo file (PNG/JPG format)",
                    "Company banner image (recommended 1536x768px)",
                    "Approved company description",
                    "Company website URL"
                ],
                "detailed_instructions": [
                    {
                        "action": "Access Company Page Creation",
                        "description": "From LinkedIn homepage, click 'Create company page' from admin menu or search",
                        "screenshot_needed": False,
                        "expected_result": "Company page creation form opens"
                    },
                    {
                        "action": "Enter Basic Information",
                        "description": "Fill in: Company name 'RQA Technologies Inc.', Industry 'Software Development', Company size '1-10 employees'",
                        "screenshot_needed": True,
                        "expected_result": "Basic information saved"
                    },
                    {
                        "action": "Upload Logo and Banner",
                        "description": "Upload company logo (400x400px) and banner image (1536x768px)",
                        "screenshot_needed": True,
                        "expected_result": "Images uploaded and displayed correctly"
                    },
                    {
                        "action": "Write Company Description",
                        "description": "Enter compelling company description highlighting AI-driven QA innovation",
                        "screenshot_needed": True,
                        "expected_result": "Description saved and formatted properly"
                    },
                    {
                        "action": "Add Company Links",
                        "description": "Add company website URL and social media links",
                        "screenshot_needed": False,
                        "expected_result": "Links saved and clickable"
                    },
                    {
                        "action": "Set Page Visibility",
                        "description": "Set page to public and enable job postings",
                        "screenshot_needed": True,
                        "expected_result": "Page goes live publicly"
                    }
                ],
                "success_criteria": [
                    "Company page live and accessible",
                    "Logo and banner display correctly",
                    "Company description compelling and professional",
                    "All links functional",
                    "Job posting capability enabled"
                ],
                "content_requirements": {
                    "company_description": "RQA Technologies Inc. is revolutionizing software quality assurance through AI/ML technologies. We're building the future of automated testing and intelligent quality management.",
                    "website_url": "https://www.rqa.tech (placeholder - to be updated)",
                    "specialties": ["AI/ML", "Quality Assurance", "DevOps", "SaaS", "Software Testing"]
                }
            },
            "step_3_job_posting_templates": {
                "step_number": 3,
                "title": "Job Posting Templates Configuration",
                "estimated_time": "90 minutes",
                "status": "pending",
                "prerequisites": [
                    "Company page created and active",
                    "Approved job descriptions",
                    "Company branding guidelines",
                    "Salary range information"
                ],
                "detailed_instructions": [
                    {
                        "action": "Access Job Posting Section",
                        "description": "From company page admin, click 'Post a job' to access job posting tools",
                        "screenshot_needed": False,
                        "expected_result": "Job posting interface opens"
                    },
                    {
                        "action": "Create Executive Template",
                        "description": "Create template for CEO/CTO/CPO positions with standard sections",
                        "screenshot_needed": True,
                        "expected_result": "Executive template saved"
                    },
                    {
                        "action": "Create Technical Template",
                        "description": "Create template for engineering positions with technical focus",
                        "screenshot_needed": True,
                        "expected_result": "Technical template saved"
                    },
                    {
                        "action": "Configure Salary Transparency",
                        "description": "Set up salary range display and compensation philosophy",
                        "screenshot_needed": True,
                        "expected_result": "Salary settings configured"
                    },
                    {
                        "action": "Set Up Automated Workflows",
                        "description": "Configure auto-responses, follow-up sequences, and integration with CRM",
                        "screenshot_needed": True,
                        "expected_result": "Automation rules active"
                    }
                ],
                "template_structure": {
                    "executive_template": [
                        "Company Overview",
                        "Role Summary",
                        "Key Responsibilities",
                        "Requirements & Qualifications",
                        "Benefits & Compensation",
                        "How to Apply"
                    ],
                    "technical_template": [
                        "Problem Statement",
                        "Technical Challenges",
                        "Role Expectations",
                        "Tech Stack & Tools",
                        "Growth Opportunities",
                        "Application Process"
                    ]
                },
                "automation_features": [
                    "Auto-acknowledgment emails",
                    "Application status updates",
                    "Interview scheduling integration",
                    "Rejection email templates"
                ]
            },
            "step_4_targeting_configuration": {
                "step_number": 4,
                "title": "Audience Targeting Setup",
                "estimated_time": "45 minutes",
                "status": "pending",
                "prerequisites": [
                    "Premium Business account",
                    "Job posting templates ready",
                    "Target candidate profiles defined"
                ],
                "detailed_instructions": [
                    {
                        "action": "Access Targeting Settings",
                        "description": "From job posting creation, click 'Target candidates' section",
                        "screenshot_needed": False,
                        "expected_result": "Targeting interface opens"
                    },
                    {
                        "action": "Set Geographic Targeting",
                        "description": "Configure locations: San Francisco Bay Area, Seattle, Austin, Remote",
                        "screenshot_needed": True,
                        "expected_result": "Geographic filters applied"
                    },
                    {
                        "action": "Configure Seniority Levels",
                        "description": "Set target levels: Director, VP, C-level, Senior Individual Contributor",
                        "screenshot_needed": True,
                        "expected_result": "Seniority filters active"
                    },
                    {
                        "action": "Set Industry Targeting",
                        "description": "Target: Software Development, AI/ML, SaaS, Quality Assurance, DevOps",
                        "screenshot_needed": True,
                        "expected_result": "Industry filters configured"
                    },
                    {
                        "action": "Configure Skills-Based Targeting",
                        "description": "Add skills: Python, AI/ML, Quality Assurance, Leadership, Product Management",
                        "screenshot_needed": True,
                        "expected_result": "Skills targeting enabled"
                    },
                    {
                        "action": "Set Up Retargeting Campaigns",
                        "description": "Configure campaigns for passive candidates who view company page",
                        "screenshot_needed": True,
                        "expected_result": "Retargeting campaigns active"
                    }
                ],
                "targeting_strategy": {
                    "primary_target": {
                        "geography": "San Francisco Bay Area, Seattle, Austin",
                        "seniority": "VP/Director level and above",
                        "industries": "Software, AI, SaaS",
                        "skills": "Leadership, AI/ML, Product Management"
                    },
                    "secondary_target": {
                        "geography": "US-wide remote",
                        "seniority": "Senior level",
                        "industries": "Technology, DevOps, Quality Assurance",
                        "skills": "Technical leadership, AI engineering"
                    }
                }
            },
            "step_5_initial_job_posts": {
                "step_number": 5,
                "title": "Initial Job Postings",
                "estimated_time": "45 minutes",
                "status": "pending",
                "prerequisites": [
                    "All templates configured",
                    "Targeting parameters set",
                    "Job descriptions finalized",
                    "Company page live"
                ],
                "detailed_instructions": [
                    {
                        "action": "Post CEO Position",
                        "description": "Create and publish CEO job posting using executive template",
                        "screenshot_needed": True,
                        "expected_result": "CEO position live on LinkedIn"
                    },
                    {
                        "action": "Post CTO Position",
                        "description": "Create and publish CTO job posting using executive template",
                        "screenshot_needed": True,
                        "expected_result": "CTO position live on LinkedIn"
                    },
                    {
                        "action": "Post CPO Position",
                        "description": "Create and publish CPO job posting using executive template",
                        "screenshot_needed": True,
                        "expected_result": "CPO position live on LinkedIn"
                    },
                    {
                        "action": "Verify Post Appearance",
                        "description": "Check that all posts display correctly on company page",
                        "screenshot_needed": True,
                        "expected_result": "All posts visible and formatted properly"
                    },
                    {
                        "action": "Share Posts on Company Page",
                        "description": "Share job postings as updates on company page feed",
                        "screenshot_needed": True,
                        "expected_result": "Job posts shared to company feed"
                    }
                ],
                "posting_checklist": [
                    "Job title and company name correct",
                    "Salary range displayed (where applicable)",
                    "Job description complete and compelling",
                    "Application process clearly explained",
                    "Company branding consistent",
                    "Call-to-action buttons functional",
                    "Targeting parameters applied"
                ]
            }
        }

    def _define_company_page_content(self) -> Dict[str, Any]:
        """定义公司页面内容"""
        return {
            "basic_information": {
                "company_name": "RQA Technologies Inc.",
                "tagline": "AI-Powered Quality Assurance for the Future",
                "industry": "Software Development",
                "company_size": "1-10 employees",
                "headquarters": "San Francisco Bay Area, CA",
                "founded": "2027"
            },
            "company_description": """
            RQA Technologies Inc. is revolutionizing software quality assurance through cutting-edge AI/ML technologies.
            We're building the next generation of intelligent testing platforms that automate quality assurance,
            predict defects before they occur, and ensure software reliability at scale.

            Our mission is to empower developers and QA teams with AI-driven tools that make software quality
            assurance more efficient, accurate, and scalable than ever before.

            Join us in shaping the future of software quality!
            """,
            "specialties": [
                "AI/ML-Powered Testing",
                "Automated Quality Assurance",
                "Predictive Defect Analysis",
                "DevOps Integration",
                "SaaS Quality Platforms",
                "Intelligent Test Generation"
            ],
            "website": "https://www.rqa.tech",
            "social_links": {
                "twitter": "@RQATech",
                "github": "https://github.com/rqa-tech",
                "linkedin": "https://linkedin.com/company/rqa-technologies"
            }
        }

    def _define_job_posting_templates(self) -> Dict[str, Any]:
        """定义职位发布模板"""
        return {
            "executive_template": {
                "structure": [
                    {
                        "section": "Company Overview",
                        "content": "Brief, compelling description of RQA and our mission"
                    },
                    {
                        "section": "Role Summary",
                        "content": "2-3 sentence overview of the position and impact"
                    },
                    {
                        "section": "Key Responsibilities",
                        "content": "5-7 bullet points of main duties and expectations"
                    },
                    {
                        "section": "Requirements & Qualifications",
                        "content": "Must-have skills, experience, and qualifications"
                    },
                    {
                        "section": "Benefits & Compensation",
                        "content": "Salary range, equity, benefits package"
                    },
                    {
                        "section": "How to Apply",
                        "content": "Clear application instructions and timeline"
                    }
                ],
                "formatting": {
                    "font": "Clean, professional sans-serif",
                    "colors": "RQA brand colors (#0066CC primary, #00CC66 accent)",
                    "images": "Include company logo and relevant graphics",
                    "length": "800-1000 words"
                }
            },
            "technical_template": {
                "structure": [
                    {
                        "section": "Problem Statement",
                        "content": "The technical challenge we're solving"
                    },
                    {
                        "section": "Technical Challenges",
                        "content": "Specific problems the role will address"
                    },
                    {
                        "section": "Role Expectations",
                        "content": "What success looks like in this role"
                    },
                    {
                        "section": "Tech Stack & Tools",
                        "content": "Technologies and tools used"
                    },
                    {
                        "section": "Growth Opportunities",
                        "content": "Learning and career development paths"
                    },
                    {
                        "section": "Application Process",
                        "content": "Interview process and timeline"
                    }
                ],
                "formatting": {
                    "font": "Monospace for code examples, clean sans-serif for text",
                    "colors": "Tech-focused color scheme",
                    "images": "Include architecture diagrams, code snippets",
                    "length": "600-800 words"
                }
            }
        }

    def _define_targeting_strategy(self) -> Dict[str, Any]:
        """定义定位策略"""
        return {
            "geographic_targeting": {
                "primary_markets": [
                    "San Francisco Bay Area, CA",
                    "Seattle, WA",
                    "Austin, TX"
                ],
                "secondary_markets": [
                    "New York, NY",
                    "Boston, MA",
                    "Denver, CO",
                    "Remote (US-wide)"
                ],
                "international": [
                    "London, UK",
                    "Toronto, Canada",
                    "Remote (Global)"
                ]
            },
            "seniority_targeting": {
                "executive_roles": ["Director", "VP", "C-Level", "Partner"],
                "technical_roles": ["Senior", "Lead", "Principal", "Staff"],
                "emerging_talent": ["Mid-level with high potential"]
            },
            "industry_targeting": {
                "primary": ["Software Development", "SaaS", "AI/ML"],
                "secondary": ["DevOps", "Quality Assurance", "Cloud Computing"],
                "adjacent": ["FinTech", "HealthTech", "E-commerce"]
            },
            "skills_targeting": {
                "leadership": ["Strategic Planning", "Team Leadership", "Executive Management"],
                "technical": ["AI/ML", "Python", "Cloud Architecture", "DevOps"],
                "domain": ["Quality Assurance", "Software Testing", "Product Management"],
                "soft_skills": ["Communication", "Problem Solving", "Innovation"]
            },
            "retargeting_strategy": {
                "page_visitors": "Target people who viewed our company page",
                "job_seekers": "Target active job seekers in our industry",
                "competitor_employees": "Target professionals from competing companies",
                "network_connections": "Leverage existing network connections"
            }
        }

    def generate_activation_guide(self) -> Dict[str, Any]:
        """生成激活指南"""
        return {
            "guide_overview": {
                "title": "RQA LinkedIn Premium Activation Guide",
                "date": self.activation_date.strftime("%Y-%m-%d"),
                "author": self.account_manager,
                "version": "1.0",
                "total_steps": 5,
                "estimated_total_time": "4 hours"
            },
            "activation_steps": self.activation_steps,
            "company_page_content": self.company_page_content,
            "job_posting_templates": self.job_posting_templates,
            "targeting_strategy": self.targeting_strategy,
            "progress_tracking": self._create_progress_tracking(),
            "quality_checklist": self._create_quality_checklist(),
            "troubleshooting_guide": self._create_troubleshooting_guide()
        }

    def _create_progress_tracking(self) -> Dict[str, Any]:
        """创建进度跟踪"""
        return {
            "step_completion_status": {
                "step_1_account_upgrade": {"completed": False, "completion_time": None, "notes": ""},
                "step_2_company_page_creation": {"completed": False, "completion_time": None, "notes": ""},
                "step_3_job_posting_templates": {"completed": False, "completion_time": None, "notes": ""},
                "step_4_targeting_configuration": {"completed": False, "completion_time": None, "notes": ""},
                "step_5_initial_job_posts": {"completed": False, "completion_time": None, "notes": ""}
            },
            "overall_progress": {
                "steps_completed": 0,
                "total_steps": 5,
                "completion_percentage": 0,
                "estimated_time_remaining": "4 hours",
                "next_action": "Begin Step 1: Account Upgrade"
            },
            "milestone_tracking": [
                {"milestone": "Premium account active", "completed": False, "target_date": "Day 1, 10:00 AM"},
                {"milestone": "Company page live", "completed": False, "target_date": "Day 1, 11:30 AM"},
                {"milestone": "Templates configured", "completed": False, "target_date": "Day 1, 2:00 PM"},
                {"milestone": "Targeting set up", "completed": False, "target_date": "Day 1, 3:00 PM"},
                {"milestone": "Initial jobs posted", "completed": False, "target_date": "Day 1, 4:00 PM"}
            ]
        }

    def _create_quality_checklist(self) -> Dict[str, Any]:
        """创建质量检查清单"""
        return {
            "account_setup_checks": [
                {"check": "Premium Business badge visible in profile", "status": "pending"},
                {"check": "Access to business analytics dashboard", "status": "pending"},
                {"check": "Company page creation capability enabled", "status": "pending"},
                {"check": "Job posting credits available", "status": "pending"}
            ],
            "company_page_checks": [
                {"check": "Page loads correctly and displays professionally", "status": "pending"},
                {"check": "Logo and banner images display at correct sizes", "status": "pending"},
                {"check": "Company description is compelling and error-free", "status": "pending"},
                {"check": "All links (website, social) are functional", "status": "pending"},
                {"check": "Page is set to public visibility", "status": "pending"}
            ],
            "posting_quality_checks": [
                {"check": "Job descriptions are complete and accurate", "status": "pending"},
                {"check": "Salary ranges are competitive and clearly stated", "status": "pending"},
                {"check": "Company branding is consistent across posts", "status": "pending"},
                {"check": "Application instructions are clear", "status": "pending"},
                {"check": "Posts display correctly on mobile devices", "status": "pending"}
            ],
            "targeting_effectiveness_checks": [
                {"check": "Geographic filters applied correctly", "status": "pending"},
                {"check": "Seniority levels targeted appropriately", "status": "pending"},
                {"check": "Industry and skills filters active", "status": "pending"},
                {"check": "Retargeting campaigns configured", "status": "pending"}
            ]
        }

    def _create_troubleshooting_guide(self) -> Dict[str, Any]:
        """创建故障排除指南"""
        return {
            "common_issues": [
                {
                    "issue": "Payment processing failed",
                    "symptoms": "Error message during checkout, card declined",
                    "solutions": [
                        "Verify card details and billing address",
                        "Try alternative payment method",
                        "Contact bank to confirm card status",
                        "Contact LinkedIn support if issue persists"
                    ],
                    "prevention": "Prepare valid payment method in advance"
                },
                {
                    "issue": "Company page creation blocked",
                    "symptoms": "Unable to create page, error messages",
                    "solutions": [
                        "Ensure Premium Business account is fully active",
                        "Verify email domain ownership",
                        "Wait 24 hours after account upgrade",
                        "Contact LinkedIn support for assistance"
                    ],
                    "prevention": "Complete account upgrade 24 hours before page creation"
                },
                {
                    "issue": "Job posts not displaying correctly",
                    "symptoms": "Formatting issues, missing information",
                    "solutions": [
                        "Clear browser cache and reload",
                        "Check for unsupported characters in text",
                        "Verify image file formats and sizes",
                        "Republish post if formatting is corrupted"
                    ],
                    "prevention": "Use plain text formatting and standard image formats"
                },
                {
                    "issue": "Targeting options not available",
                    "symptoms": "Cannot access targeting settings",
                    "solutions": [
                        "Confirm Premium Business status",
                        "Refresh page and try different browser",
                        "Contact LinkedIn support for account verification",
                        "Wait for account to fully propagate (up to 48 hours)"
                    ],
                    "prevention": "Complete all account setup before configuring targeting"
                }
            ],
            "emergency_contacts": {
                "linkedin_support": "https://www.linkedin.com/help/linkedin",
                "account_executive": "Contact details for Premium account manager",
                "technical_support": "Internal IT support for browser/network issues",
                "backup_channels": "Alternative job posting platforms if LinkedIn unavailable"
            },
            "escalation_procedures": {
                "level_1": "Try self-service solutions from troubleshooting guide",
                "level_2": "Contact LinkedIn support directly",
                "level_3": "Escalate to account executive or senior management",
                "level_4": "Switch to backup recruitment channels"
            }
        }

def main():
    """主函数：生成RQA LinkedIn激活指南"""
    print("=" * 80)
    print("🔗 RQA LinkedIn Premium激活执行指南")
    print("=" * 80)

    guide = RQALinkedInActivationGuide()
    activation_guide = guide.generate_activation_guide()

    # 保存为JSON格式
    json_file = "test_logs/rqa_linkedin_activation_guide.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(activation_guide, f, ensure_ascii=False, indent=2)

    # 输出关键信息
    print("\n🎯 激活概览:")
    overview = activation_guide['guide_overview']
    print(f"  指南标题: {overview['title']}")
    print(f"  执行日期: {overview['date']}")
    print(f"  负责人: {overview['author']}")
    print(f"  总步骤: {overview['total_steps']}步")
    print(f"  预估时间: {overview['estimated_total_time']}")

    print("\n📋 激活步骤:")
    steps = activation_guide['activation_steps']
    for step_key, step in steps.items():
        status_icon = "✅" if step['status'] == 'completed' else "⏳"
        print(f"  {status_icon} 步骤{step['step_number']}: {step['title']} ({step['estimated_time']})")

    print("\n🏢 公司页面内容:")
    content = activation_guide['company_page_content']
    print(f"  公司名称: {content['basic_information']['company_name']}")
    print(f"  标语: {content['basic_information']['tagline']}")
    print(f"  行业: {content['basic_information']['industry']}")
    print(f"  公司规模: {content['basic_information']['company_size']}")

    print("\n📝 职位模板:")
    templates = activation_guide['job_posting_templates']
    print(f"  执行模板: {len(templates['executive_template']['structure'])}个部分")
    print(f"  技术模板: {len(templates['technical_template']['structure'])}个部分")

    print("\n🎯 定位策略:")
    targeting = activation_guide['targeting_strategy']
    print(f"  主要市场: {len(targeting['geographic_targeting']['primary_markets'])}个地区")
    print(f"  职级定位: {len(targeting['seniority_targeting']['executive_roles'])}个级别")
    print(f"  行业定位: {len(targeting['industry_targeting']['primary'])}个行业")

    print("\n📊 进度跟踪:")
    progress = activation_guide['progress_tracking']
    overall = progress['overall_progress']
    print(f"  已完成步骤: {overall['steps_completed']}/{overall['total_steps']}")
    print(f"  完成百分比: {overall['completion_percentage']}%")
    print(f"  剩余时间: {overall['estimated_time_remaining']}")
    print(f"  下一步行动: {overall['next_action']}")

    print("\n✅ 质量检查清单:")
    checklist = activation_guide['quality_checklist']
    total_checks = sum(len(checks) for checks in checklist.values())
    completed_checks = sum(len([check for check in checks if check['status'] == 'completed']) for checks in checklist.values())
    print(f"  总检查项: {total_checks}项")
    print(f"  已完成: {completed_checks}项")
    print(f"  完成率: {(completed_checks/total_checks)*100:.1f}%")

    print("\n🛠️ 故障排除:")
    troubleshooting = activation_guide['troubleshooting_guide']
    print(f"  常见问题: {len(troubleshooting['common_issues'])}个")
    print(f"  紧急联系人: {len(troubleshooting['emergency_contacts'])}个")
    print(f"  升级流程: {len(troubleshooting['escalation_procedures'])}级")

    print("\n🎊 LinkedIn激活指南生成成功！")
    print("从账户升级到职位发布，从公司页面到定位配置，开启RQA LinkedIn招聘之旅！")
    print("=" * 80)

if __name__ == "__main__":
    main()
