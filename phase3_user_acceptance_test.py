#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 用户验收测试 - UAT脚本
模拟最终用户的使用体验和验收流程
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor


@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    user_type: str  # 'trader', 'portfolio_manager', 'risk_officer', 'compliance_officer', 'admin'
    experience_level: str  # 'novice', 'intermediate', 'expert'
    primary_use_case: str
    acceptance_criteria: List[str]


@dataclass
class UserJourney:
    """用户旅程"""
    journey_id: str
    user_profile: UserProfile
    journey_name: str
    steps: List[Dict[str, Any]]
    expected_duration_minutes: int
    success_criteria: List[str]


@dataclass
class JourneyResult:
    """旅程测试结果"""
    journey_id: str
    journey_name: str
    user_type: str
    start_time: datetime
    end_time: datetime
    total_duration_minutes: float
    steps_completed: int
    steps_failed: int
    steps_total: int
    completion_rate: float
    user_satisfaction_score: float  # 1-10
    issues_reported: List[str]
    recommendations: List[str]
    status: str  # 'passed', 'conditional_pass', 'failed'


@dataclass
class UserAcceptanceReport:
    """用户验收报告"""
    timestamp: datetime
    total_users_tested: int
    total_journeys_executed: int
    journeys_passed: int
    journeys_failed: int
    average_satisfaction_score: float
    overall_acceptance_rate: float
    user_readiness_score: float
    critical_blockers: List[str]
    improvement_areas: List[str]
    go_live_recommendation: str
    journey_results: List[JourneyResult]


class UserAcceptanceTester:
    """用户验收测试器"""

    def __init__(self):
        self.user_profiles: List[UserProfile] = []
        self.user_journeys: List[UserJourney] = []
        self.journey_results: List[JourneyResult] = []
        self.setup_logging()
        self.initialize_user_profiles()
        self.initialize_user_journeys()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('user_acceptance_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_user_profiles(self):
        """初始化用户档案"""
        self.user_profiles = [
            UserProfile(
                user_id="trader_001",
                user_type="trader",
                experience_level="expert",
                primary_use_case="High-frequency trading execution",
                acceptance_criteria=[
                    "Can execute orders within 100ms",
                    "Real-time P&L updates accurate",
                    "Order book visualization clear",
                    "Risk limits enforced automatically"
                ]
            ),
            UserProfile(
                user_id="pm_001",
                user_type="portfolio_manager",
                experience_level="intermediate",
                primary_use_case="Portfolio rebalancing and performance monitoring",
                acceptance_criteria=[
                    "Portfolio analytics load within 3 seconds",
                    "Rebalancing suggestions accurate",
                    "Performance attribution clear",
                    "Reporting export functions work"
                ]
            ),
            UserProfile(
                user_id="risk_001",
                user_type="risk_officer",
                experience_level="expert",
                primary_use_case="Risk monitoring and limit management",
                acceptance_criteria=[
                    "Risk dashboard updates in real-time",
                    "Alert notifications work reliably",
                    "Limit breaches detected immediately",
                    "Risk reports generate correctly"
                ]
            ),
            UserProfile(
                user_id="compliance_001",
                user_type="compliance_officer",
                experience_level="intermediate",
                primary_use_case="Regulatory compliance checking",
                acceptance_criteria=[
                    "Trade surveillance works effectively",
                    "Compliance reports generate on schedule",
                    "Audit trails complete and accessible",
                    "Regulatory filings submit successfully"
                ]
            ),
            UserProfile(
                user_id="admin_001",
                user_type="admin",
                experience_level="expert",
                primary_use_case="System administration and user management",
                acceptance_criteria=[
                    "User onboarding process smooth",
                    "System configuration changes apply correctly",
                    "Backup and recovery procedures work",
                    "Performance monitoring dashboards functional"
                ]
            )
        ]

    def initialize_user_journeys(self):
        """初始化用户旅程"""
        self.user_journeys = [
            UserJourney(
                journey_id="trader_daily_workflow",
                user_profile=self.user_profiles[0],  # trader
                journey_name="交易员日常工作流程",
                expected_duration_minutes=45,
                success_criteria=[
                    "Complete a full trading cycle from market analysis to order execution",
                    "Monitor positions and P&L throughout the session",
                    "Handle market volatility appropriately",
                    "End-of-day reporting and reconciliation"
                ],
                steps=[
                    {
                        "step_id": "login",
                        "name": "系统登录",
                        "description": "使用交易员账号登录系统",
                        "expected_duration_seconds": 30,
                        "critical": True
                    },
                    {
                        "step_id": "market_overview",
                        "name": "查看市场概览",
                        "description": "查看主要市场指数和个股表现",
                        "expected_duration_seconds": 60,
                        "critical": False
                    },
                    {
                        "step_id": "strategy_selection",
                        "name": "选择交易策略",
                        "description": "从可用策略中选择合适的交易策略",
                        "expected_duration_seconds": 45,
                        "critical": True
                    },
                    {
                        "step_id": "order_placement",
                        "name": "下单交易",
                        "description": "根据策略信号下达买卖订单",
                        "expected_duration_seconds": 120,
                        "critical": True
                    },
                    {
                        "step_id": "position_monitoring",
                        "name": "持仓监控",
                        "description": "实时监控持仓和盈亏情况",
                        "expected_duration_seconds": 180,
                        "critical": True
                    },
                    {
                        "step_id": "risk_check",
                        "name": "风险检查",
                        "description": "检查持仓是否符合风险限制",
                        "expected_duration_seconds": 60,
                        "critical": True
                    },
                    {
                        "step_id": "performance_review",
                        "name": "业绩回顾",
                        "description": "查看当日交易业绩和统计",
                        "expected_duration_seconds": 90,
                        "critical": False
                    }
                ]
            ),

            UserJourney(
                journey_id="pm_portfolio_management",
                user_profile=self.user_profiles[1],  # portfolio_manager
                journey_name="组合经理投资组合管理",
                expected_duration_minutes=60,
                success_criteria=[
                    "Review portfolio performance against benchmarks",
                    "Execute portfolio rebalancing trades",
                    "Generate client reporting",
                    "Update investment strategy parameters"
                ],
                steps=[
                    {
                        "step_id": "portfolio_dashboard",
                        "name": "查看组合仪表板",
                        "description": "查看投资组合整体表现和关键指标",
                        "expected_duration_seconds": 90,
                        "critical": True
                    },
                    {
                        "step_id": "performance_analysis",
                        "name": "业绩分析",
                        "description": "分析组合相对于基准的业绩表现",
                        "expected_duration_seconds": 120,
                        "critical": True
                    },
                    {
                        "step_id": "rebalancing_decision",
                        "name": "再平衡决策",
                        "description": "决定是否需要调整组合权重",
                        "expected_duration_seconds": 180,
                        "critical": False
                    },
                    {
                        "step_id": "trade_execution",
                        "name": "执行交易",
                        "description": "执行再平衡所需的交易",
                        "expected_duration_seconds": 240,
                        "critical": True
                    },
                    {
                        "step_id": "client_reporting",
                        "name": "客户报告",
                        "description": "生成和发送客户业绩报告",
                        "expected_duration_seconds": 150,
                        "critical": True
                    }
                ]
            ),

            UserJourney(
                journey_id="risk_monitoring",
                user_profile=self.user_profiles[2],  # risk_officer
                journey_name="风险官风险监控",
                expected_duration_minutes=40,
                success_criteria=[
                    "Monitor all risk metrics in real-time",
                    "Respond to risk limit breaches",
                    "Review risk reports and analytics",
                    "Update risk parameters as needed"
                ],
                steps=[
                    {
                        "step_id": "risk_dashboard_review",
                        "name": "风险仪表板检查",
                        "description": "检查所有风险指标的状态",
                        "expected_duration_seconds": 120,
                        "critical": True
                    },
                    {
                        "step_id": "alert_response",
                        "name": "告警响应",
                        "description": "处理任何活跃的风险告警",
                        "expected_duration_seconds": 90,
                        "critical": True
                    },
                    {
                        "step_id": "limit_adjustment",
                        "name": "限额调整",
                        "description": "根据市场条件调整风险限额",
                        "expected_duration_seconds": 60,
                        "critical": False
                    },
                    {
                        "step_id": "risk_reporting",
                        "name": "风险报告",
                        "description": "生成每日风险报告",
                        "expected_duration_seconds": 180,
                        "critical": True
                    }
                ]
            ),

            UserJourney(
                journey_id="compliance_checking",
                user_profile=self.user_profiles[3],  # compliance_officer
                journey_name="合规官合规检查",
                expected_duration_minutes=50,
                success_criteria=[
                    "Review trading activity for compliance",
                    "Check regulatory filings",
                    "Monitor for suspicious activity",
                    "Generate compliance reports"
                ],
                steps=[
                    {
                        "step_id": "trade_surveillance",
                        "name": "交易监控",
                        "description": "监控交易活动是否符合法规",
                        "expected_duration_seconds": 150,
                        "critical": True
                    },
                    {
                        "step_id": "regulatory_reporting",
                        "name": "监管报告",
                        "description": "准备和提交监管要求的报告",
                        "expected_duration_seconds": 120,
                        "critical": True
                    },
                    {
                        "step_id": "audit_trail_review",
                        "name": "审计轨迹检查",
                        "description": "检查交易记录的完整性和准确性",
                        "expected_duration_seconds": 90,
                        "critical": True
                    },
                    {
                        "step_id": "exception_handling",
                        "name": "异常处理",
                        "description": "处理任何合规异常情况",
                        "expected_duration_seconds": 60,
                        "critical": False
                    }
                ]
            ),

            UserJourney(
                journey_id="admin_system_management",
                user_profile=self.user_profiles[4],  # admin
                journey_name="管理员系统管理",
                expected_duration_minutes=35,
                success_criteria=[
                    "Manage user accounts and permissions",
                    "Configure system parameters",
                    "Monitor system health",
                    "Handle system maintenance tasks"
                ],
                steps=[
                    {
                        "step_id": "user_management",
                        "name": "用户管理",
                        "description": "创建和管理用户账号及权限",
                        "expected_duration_seconds": 120,
                        "critical": True
                    },
                    {
                        "step_id": "system_configuration",
                        "name": "系统配置",
                        "description": "调整系统参数和设置",
                        "expected_duration_seconds": 90,
                        "critical": True
                    },
                    {
                        "step_id": "health_monitoring",
                        "name": "健康监控",
                        "description": "检查系统各组件的运行状态",
                        "expected_duration_seconds": 60,
                        "critical": False
                    },
                    {
                        "step_id": "backup_verification",
                        "name": "备份验证",
                        "description": "验证备份过程和数据完整性",
                        "expected_duration_seconds": 80,
                        "critical": True
                    }
                ]
            )
        ]

    async def run_user_acceptance_test(self) -> UserAcceptanceReport:
        """运行用户验收测试"""
        self.logger.info("开始用户验收测试...")
        start_time = datetime.now()

        # 并行执行用户旅程测试
        tasks = []
        for journey in self.user_journeys:
            task = asyncio.create_task(self.execute_user_journey(journey))
            tasks.append(task)

        # 等待所有旅程完成
        await asyncio.gather(*tasks)

        end_time = datetime.now()
        self.logger.info("用户验收测试完成")

        # 生成综合报告
        report = self.generate_acceptance_report()
        report.timestamp = start_time

        return report

    async def execute_user_journey(self, journey: UserJourney) -> None:
        """执行用户旅程"""
        self.logger.info(f"开始执行用户旅程: {journey.journey_name} ({journey.user_profile.user_type})")
        start_time = datetime.now()

        result = JourneyResult(
            journey_id=journey.journey_id,
            journey_name=journey.journey_name,
            user_type=journey.user_profile.user_type,
            start_time=start_time,
            end_time=None,
            total_duration_minutes=0,
            steps_completed=0,
            steps_failed=0,
            steps_total=len(journey.steps),
            completion_rate=0.0,
            user_satisfaction_score=0.0,
            issues_reported=[],
            recommendations=[],
            status="running"
        )

        try:
            # 执行旅程步骤
            for step in journey.steps:
                success, duration, issues = await self.execute_journey_step(journey, step)

                if success:
                    result.steps_completed += 1
                else:
                    result.steps_failed += 1
                    result.issues_reported.extend(issues)

                # 记录步骤持续时间（模拟）
                step_duration = duration if duration > 0 else step['expected_duration_seconds']
                await asyncio.sleep(step_duration / 60)  # 转换为分钟的睡眠

            # 计算完成率
            result.completion_rate = result.steps_completed / result.steps_total if result.steps_total > 0 else 0

            # 评估用户满意度
            result.user_satisfaction_score = self.calculate_user_satisfaction(journey, result)

            # 确定旅程状态
            critical_steps_failed = sum(1 for step in journey.steps
                                      if step['critical'] and
                                      any(issue in result.issues_reported for issue in
                                          [f"{step['step_id']}: " + issue for issue in result.issues_reported]))

            if result.completion_rate == 1.0 and result.user_satisfaction_score >= 8:
                result.status = "passed"
            elif result.completion_rate >= 0.8 and result.user_satisfaction_score >= 6:
                result.status = "conditional_pass"
            else:
                result.status = "failed"

            # 生成用户建议
            result.recommendations = self.generate_user_recommendations(journey, result)

        except Exception as e:
            result.status = "failed"
            result.issues_reported.append(f"旅程执行异常: {str(e)}")
            self.logger.error(f"用户旅程 {journey.journey_name} 执行失败: {e}")

        finally:
            end_time = datetime.now()
            result.end_time = end_time
            result.total_duration_minutes = (end_time - start_time).total_seconds() / 60

            self.journey_results.append(result)
            self.logger.info(f"用户旅程 {journey.journey_name} 完成: {result.status} ({result.steps_completed}/{result.steps_total})")

    async def execute_journey_step(self, journey: UserJourney, step: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """执行旅程步骤"""
        step_id = step['step_id']
        step_name = step['name']

        self.logger.info(f"执行步骤: {step_name}")

        try:
            # 模拟步骤执行时间和成功率
            await asyncio.sleep(np.random.uniform(0.1, 0.5))  # 短暂延迟

            # 基于用户类型和步骤的成功率
            success_rate = self.get_step_success_rate(journey.user_profile, step)
            success = np.random.choice([True, False], p=[success_rate, 1-success_rate])

            # 模拟执行时间（有些随机性）
            duration = np.random.normal(step['expected_duration_seconds'],
                                      step['expected_duration_seconds'] * 0.2)

            issues = []
            if not success:
                issues = self.generate_step_issues(journey.user_profile, step)

            return success, duration, issues

        except Exception as e:
            return False, step['expected_duration_seconds'], [f"步骤执行异常: {str(e)}"]

    def get_step_success_rate(self, user_profile: UserProfile, step: Dict[str, Any]) -> float:
        """获取步骤成功率（基于用户类型和系统当前状态）"""
        base_success_rate = 0.85  # 基础成功率

        # 根据用户经验调整
        if user_profile.experience_level == "expert":
            base_success_rate += 0.1
        elif user_profile.experience_level == "novice":
            base_success_rate -= 0.15

        # 根据业务场景验证结果调整（模拟系统问题）
        if user_profile.user_type == "trader":
            base_success_rate -= 0.2  # 交易功能有问题
        elif user_profile.user_type == "portfolio_manager":
            base_success_rate -= 0.15  # 组合管理有问题
        elif user_profile.user_type == "risk_officer":
            base_success_rate -= 0.1  # 风险管理部分有问题

        # 确保在合理范围内
        return max(0.3, min(0.95, base_success_rate))

    def generate_step_issues(self, user_profile: UserProfile, step: Dict[str, Any]) -> List[str]:
        """生成步骤问题"""
        issues_templates = {
            "login": [
                "登录界面响应慢",
                "密码重置功能不可用",
                "双因子认证失败"
            ],
            "market_overview": [
                "市场数据加载失败",
                "图表显示不正确",
                "数据刷新延迟"
            ],
            "strategy_selection": [
                "策略列表加载失败",
                "策略参数无法修改",
                "策略回测结果不显示"
            ],
            "order_placement": [
                "订单提交失败",
                "价格验证错误",
                "数量限制检查失败"
            ],
            "position_monitoring": [
                "持仓数据不更新",
                "盈亏计算错误",
                "实时更新延迟"
            ],
            "risk_check": [
                "风险限额检查失败",
                "告警通知不工作",
                "风险报告生成错误"
            ],
            "performance_review": [
                "业绩数据不准确",
                "报告导出失败",
                "历史数据缺失"
            ],
            "portfolio_dashboard": [
                "组合数据加载慢",
                "图表渲染失败",
                "指标计算错误"
            ],
            "performance_analysis": [
                "基准比较数据缺失",
                "归因分析失败",
                "自定义日期范围无效"
            ],
            "rebalancing_decision": [
                "再平衡建议不准确",
                "交易成本估算错误",
                "执行计划生成失败"
            ],
            "trade_execution": [
                "交易执行延迟",
                "订单状态不更新",
                "执行确认丢失"
            ],
            "client_reporting": [
                "报告模板不可用",
                "数据导出格式错误",
                "邮件发送失败"
            ],
            "risk_dashboard_review": [
                "风险指标更新延迟",
                "可视化图表错误",
                "阈值设置不保存"
            ],
            "alert_response": [
                "告警通知延迟",
                "响应工作流复杂",
                "升级流程不清晰"
            ],
            "limit_adjustment": [
                "参数修改权限不足",
                "更改历史记录缺失",
                "生效时间设置混乱"
            ],
            "risk_reporting": [
                "报告生成时间过长",
                "数据准确性问题",
                "分发列表管理困难"
            ],
            "trade_surveillance": [
                "监控规则配置复杂",
                "误报率过高",
                "调查工具不够强大"
            ],
            "regulatory_reporting": [
                "报告格式不符合要求",
                "数据验证失败",
                "提交截止时间紧迫"
            ],
            "audit_trail_review": [
                "审计日志搜索慢",
                "数据完整性问题",
                "导出功能受限"
            ],
            "exception_handling": [
                "异常处理流程不清晰",
                "上报机制复杂",
                "解决时间过长"
            ],
            "user_management": [
                "用户创建流程复杂",
                "权限分配混乱",
                "批量操作不支持"
            ],
            "system_configuration": [
                "配置界面不够直观",
                "参数验证不完善",
                "更改需要重启服务"
            ],
            "health_monitoring": [
                "监控指标不够全面",
                "告警阈值设置复杂",
                "问题诊断困难"
            ],
            "backup_verification": [
                "备份验证过程繁琐",
                "恢复测试不完整",
                "备份文件管理混乱"
            ]
        }

        step_issues = issues_templates.get(step['step_id'], ["未知问题"])
        # 随机选择1-2个问题
        num_issues = np.random.randint(1, min(3, len(step_issues) + 1))
        selected_issues = np.random.choice(step_issues, num_issues, replace=False)

        return [f"{step['step_id']}: {issue}" for issue in selected_issues]

    def calculate_user_satisfaction(self, journey: UserJourney, result: JourneyResult) -> float:
        """计算用户满意度"""
        # 基于完成率和问题数量计算满意度
        base_satisfaction = result.completion_rate * 10

        # 问题数量的惩罚
        issue_penalty = len(result.issues_reported) * 0.5

        # 关键步骤失败的额外惩罚
        critical_penalty = result.steps_failed * 0.3

        satisfaction = max(1, min(10, base_satisfaction - issue_penalty - critical_penalty))

        return satisfaction

    def generate_user_recommendations(self, journey: UserJourney, result: JourneyResult) -> List[str]:
        """生成用户建议"""
        recommendations = []

        if result.completion_rate < 0.8:
            recommendations.append(f"简化 {journey.journey_name} 的工作流程")

        if result.user_satisfaction_score < 7:
            recommendations.append(f"改进 {journey.user_profile.user_type} 用户的使用体验")

        if result.steps_failed > 0:
            failed_steps = [issue.split(':')[0] for issue in result.issues_reported[:3]]
            recommendations.append(f"优先修复步骤: {', '.join(set(failed_steps))}")

        if result.total_duration_minutes > journey.expected_duration_minutes * 1.5:
            recommendations.append(f"优化 {journey.journey_name} 的响应时间")

        return recommendations if recommendations else ["用户体验基本满意"]

    def generate_acceptance_report(self) -> UserAcceptanceReport:
        """生成验收报告"""
        total_journeys = len(self.journey_results)
        journeys_passed = sum(1 for r in self.journey_results if r.status == "passed")
        journeys_conditional = sum(1 for r in self.journey_results if r.status == "conditional_pass")
        journeys_failed = sum(1 for r in self.journey_results if r.status == "failed")

        average_satisfaction = np.mean([r.user_satisfaction_score for r in self.journey_results])
        overall_acceptance_rate = (journeys_passed + journeys_conditional * 0.5) / total_journeys if total_journeys > 0 else 0

        # 计算用户就绪评分
        user_readiness_score = self.calculate_user_readiness_score()

        # 识别关键阻塞点
        critical_blockers = self.identify_critical_blockers()

        # 确定改进领域
        improvement_areas = self.identify_improvement_areas()

        # 生成上线建议
        go_live_recommendation = self.generate_go_live_recommendation(user_readiness_score)

        return UserAcceptanceReport(
            timestamp=datetime.now(),
            total_users_tested=len(self.user_profiles),
            total_journeys_executed=total_journeys,
            journeys_passed=journeys_passed,
            journeys_failed=journeys_failed,
            average_satisfaction_score=average_satisfaction,
            overall_acceptance_rate=overall_acceptance_rate,
            user_readiness_score=user_readiness_score,
            critical_blockers=critical_blockers,
            improvement_areas=improvement_areas,
            go_live_recommendation=go_live_recommendation,
            journey_results=self.journey_results
        )

    def calculate_user_readiness_score(self) -> float:
        """计算用户就绪评分"""
        if not self.journey_results:
            return 0.0

        # 基于满意度和完成率计算
        satisfaction_scores = [r.user_satisfaction_score for r in self.journey_results]
        completion_rates = [r.completion_rate for r in self.journey_results]

        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0
        avg_completion = np.mean(completion_rates) if completion_rates else 0

        # 加权计算 (满意度 60%, 完成率 40%)
        readiness_score = (avg_satisfaction / 10 * 0.6 + avg_completion * 0.4) * 100

        return readiness_score

    def identify_critical_blockers(self) -> List[str]:
        """识别关键阻塞点"""
        blockers = []

        for result in self.journey_results:
            if result.status == "failed":
                blockers.append(f"{result.user_type}用户的主要工作流 '{result.journey_name}' 无法正常使用")

            if result.user_satisfaction_score < 5:
                blockers.append(f"{result.user_type}用户对系统满意度极低 ({result.user_satisfaction_score:.1f}/10)")

            if result.completion_rate < 0.5:
                blockers.append(f"{result.journey_name} 完成率过低 ({result.completion_rate:.1%})")

        # 检查是否有系统性的问题
        failed_journeys = sum(1 for r in self.journey_results if r.status == "failed")
        if failed_journeys > len(self.journey_results) * 0.5:
            blockers.append("超过50%的用户旅程测试失败，存在系统性可用性问题")

        return blockers

    def identify_improvement_areas(self) -> List[str]:
        """识别改进领域"""
        areas = []

        # 分析常见问题
        all_issues = []
        for result in self.journey_results:
            all_issues.extend(result.issues_reported)

        # 统计问题类型
        issue_counts = {}
        for issue in all_issues:
            category = issue.split(':')[0] if ':' in issue else '其他'
            issue_counts[category] = issue_counts.get(category, 0) + 1

        # 找出最常见的问题
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        for issue_type, count in top_issues:
            if count > 2:  # 出现多次的问题
                areas.append(f"改进{issue_type}相关功能 (出现{count}次)")

        # 检查用户满意度
        low_satisfaction_users = [r for r in self.journey_results if r.user_satisfaction_score < 7]
        if low_satisfaction_users:
            user_types = list(set(r.user_type for r in low_satisfaction_users))
            areas.append(f"提升{', '.join(user_types)}用户的使用体验")

        return areas if areas else ["整体用户体验需要进一步优化"]

    def generate_go_live_recommendation(self, readiness_score: float) -> str:
        """生成上线建议"""
        if readiness_score >= 80:
            return "系统可以投入生产使用，用户体验基本满意"
        elif readiness_score >= 70:
            return "系统可以有限制地投入生产，但需要持续监控和快速响应团队"
        elif readiness_score >= 60:
            return "建议推迟上线，优先解决关键用户体验问题"
        else:
            return "强烈建议重新评估系统设计，不建议在当前状态下投入生产"


async def main():
    """主函数"""
    print('👥 Phase 3 用户验收测试开始')
    print('=' * 60)

    # 创建验收测试器
    tester = UserAcceptanceTester()

    print('👤 用户类型和旅程:')
    for journey in tester.user_journeys:
        print(f'• {journey.user_profile.user_type}: {journey.journey_name}')
        print(f'  预期时长: {journey.expected_duration_minutes}分钟')
        print(f'  步骤数量: {len(journey.steps)}')
    print()

    try:
        # 运行用户验收测试
        report = await tester.run_user_acceptance_test()

        print('\n📊 用户验收测试结果:')
        print(f'测试用户数: {report.total_users_tested}')
        print(f'执行旅程数: {report.total_journeys_executed}')
        print(f'完全通过: {report.journeys_passed}')
        print(f'条件通过: {sum(1 for r in report.journey_results if r.status == "conditional_pass")}')
        print(f'失败: {report.journeys_failed}')
        print(f'平均满意度: {report.average_satisfaction_score:.1f}/10')
        print(f'整体验收率: {report.overall_acceptance_rate:.1%}')
        print(f'用户就绪评分: {report.user_readiness_score:.1f}/100')

        # 评估就绪状态
        if report.user_readiness_score >= 80:
            readiness_status = "READY_FOR_PRODUCTION"
            status_msg = "用户验收通过，可以投入生产"
        elif report.user_readiness_score >= 70:
            readiness_status = "CONDITIONAL_GO_LIVE"
            status_msg = "可以有限制地上线，需持续改进"
        elif report.user_readiness_score >= 60:
            readiness_status = "NEEDS_IMPROVEMENT"
            status_msg = "需要进一步改进后才能上线"
        else:
            readiness_status = "NOT_READY"
            status_msg = "存在严重问题，不建议上线"

        print(f'就绪状态: {readiness_status}')
        print(f'评估结果: {status_msg}')
        print(f'上线建议: {report.go_live_recommendation}')

        print('\n📋 旅程详细结果:')
        for result in report.journey_results:
            status_icon = {"passed": "✅", "conditional_pass": "⚠️", "failed": "❌"}.get(result.status, "❓")
            print(f'{status_icon} {result.user_type}: {result.journey_name}')
            print(f'   完成率: {result.completion_rate:.1%}, 满意度: {result.user_satisfaction_score:.1f}/10')

        if report.critical_blockers:
            print('\n🚨 关键阻塞点:')
            for i, blocker in enumerate(report.critical_blockers, 1):
                print(f'{i}. {blocker}')

        if report.improvement_areas:
            print('\n🔧 改进领域:')
            for i, area in enumerate(report.improvement_areas, 1):
                print(f'{i}. {area}')

        # 保存详细报告
        os.makedirs('test_logs', exist_ok=True)
        report_file = f'phase3_user_acceptance_test_{int(datetime.now().timestamp())}.json'
        with open(f'test_logs/{report_file}', 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str, ensure_ascii=False)

        print('=' * 60)
        print('✅ Phase 3 用户验收测试完成')
        print(f'📄 详细报告已保存: test_logs/{report_file}')
        print('=' * 60)

        return readiness_status, report.user_readiness_score

    except Exception as e:
        print(f'\n❌ 用户验收测试过程中发生错误: {e}')
        return "error", 0.0


if __name__ == "__main__":
    asyncio.run(main())
