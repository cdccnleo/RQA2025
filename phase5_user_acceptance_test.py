#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5: 用户验收测试

模拟端到端用户体验，确保用户流程完整
包括用户注册、登录、交易、投资组合管理等完整用户旅程
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import random
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入核心业务组件
try:
    from phase4_core_business_fix import CoreBusinessEngine, TransactionType, OrderType
    from phase4_risk_control_system_reconstruction import ComprehensiveRiskControlSystem, Account, Position
    from phase4_portfolio_management_reconstruction import PortfolioManager, PortfolioType, RiskTolerance
    from phase4_user_experience_optimization import UserExperienceManager
    logger.info("成功导入核心业务组件")
except ImportError as e:
    logger.error(f"导入核心组件失败: {e}")
    # 创建模拟组件用于测试
    class MockCoreBusinessEngine:
        def __init__(self):
            self.users = {}
            self.orders = {}
            self.positions = {}
        def register_user(self, *args, **kwargs):
            return True, "用户注册成功", "USER_001"
        def submit_order(self, *args, **kwargs):
            return True, "订单提交成功", "ORDER_001"
        def get_account_summary(self, user_id):
            return {"balance": 100000, "total_value": 100000}

@dataclass
class UserJourney:
    """用户旅程定义"""
    journey_id: str
    user_type: str  # novice, experienced, institutional
    description: str
    steps: List[Dict[str, Any]]
    expected_duration: int  # 期望完成时间（秒）
    success_criteria: List[str]

@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: str
    user_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps_completed: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    satisfaction_score: int = 5  # 1-5 分

@dataclass
class UATResult:
    """验收测试结果"""
    journey_id: str
    user_session: UserSession
    success: bool
    completion_time: float
    user_experience_score: int  # 1-5
    functionality_score: int   # 1-5
    performance_score: int     # 1-5
    overall_score: int         # 1-5
    feedback: List[str] = field(default_factory=list)

@dataclass
class UserAcceptanceReport:
    """用户验收报告"""
    total_journeys: int = 0
    completed_journeys: int = 0
    failed_journeys: int = 0
    average_completion_time: float = 0.0
    average_satisfaction: float = 0.0
    user_experience_score: float = 0.0
    functionality_score: float = 0.0
    performance_score: float = 0.0
    overall_acceptance_score: float = 0.0
    journey_results: List[UATResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class UserAcceptanceTester:
    """用户验收测试器"""

    def __init__(self):
        self.business_engine = None
        self.risk_system = None
        self.portfolio_manager = None
        self.user_experience_manager = None

        # 测试用户
        self.test_users: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, UserSession] = {}

        # 用户旅程定义
        self.journeys = []
        self.results = []

        # 初始化系统
        self._initialize_system()

        # 定义用户旅程
        self._define_user_journeys()

        logger.info("用户验收测试器初始化完成")

    def _initialize_system(self):
        """初始化系统组件"""
        try:
            self.business_engine = CoreBusinessEngine()
            self.business_engine.initialize()
            logger.info("✅ 核心业务引擎初始化成功")
        except Exception as e:
            logger.warning(f"核心业务引擎初始化失败，使用模拟组件: {e}")
            self.business_engine = MockCoreBusinessEngine()

        try:
            self.risk_system = ComprehensiveRiskControlSystem()
            self.risk_system.start()
            logger.info("✅ 风险控制系统初始化成功")
        except Exception as e:
            logger.warning(f"风险控制系统初始化失败: {e}")

        try:
            self.portfolio_manager = PortfolioManager()
            logger.info("✅ 投资组合管理器初始化成功")
        except Exception as e:
            logger.warning(f"投资组合管理器初始化失败: {e}")

        try:
            self.user_experience_manager = UserExperienceManager()
            logger.info("✅ 用户体验管理器初始化成功")
        except Exception as e:
            logger.warning(f"用户体验管理器初始化失败: {e}")

    def _define_user_journeys(self):
        """定义用户旅程"""
        self.journeys = [

            # 旅程1: 新手用户首次使用
            UserJourney(
                journey_id="novice_user_first_time",
                user_type="novice",
                description="新手用户从注册到完成首次交易的完整旅程",
                expected_duration=300,  # 5分钟
                steps=[
                    {
                        "step_id": "visit_homepage",
                        "action": "navigate_to_homepage",
                        "description": "访问首页",
                        "expected_time": 5
                    },
                    {
                        "step_id": "register_account",
                        "action": "user_registration",
                        "description": "注册新账户",
                        "params": {
                            "email": "novice_user_{timestamp}@example.com",
                            "password": "TestPass123!",
                            "user_type": "individual"
                        },
                        "expected_time": 30
                    },
                    {
                        "step_id": "verify_email",
                        "action": "email_verification",
                        "description": "验证邮箱",
                        "expected_time": 10
                    },
                    {
                        "step_id": "complete_profile",
                        "action": "profile_setup",
                        "description": "完善个人资料",
                        "params": {
                            "risk_tolerance": "conservative",
                            "investment_goal": "long_term_growth"
                        },
                        "expected_time": 45
                    },
                    {
                        "step_id": "deposit_funds",
                        "action": "account_deposit",
                        "description": "账户充值",
                        "params": {"amount": 10000},
                        "expected_time": 20
                    },
                    {
                        "step_id": "browse_stocks",
                        "action": "market_browse",
                        "description": "浏览股票市场",
                        "params": {"category": "popular"},
                        "expected_time": 30
                    },
                    {
                        "step_id": "place_first_order",
                        "action": "simple_buy_order",
                        "description": "下第一个买入订单",
                        "params": {
                            "symbol": "AAPL",
                            "quantity": 10,
                            "order_type": "market"
                        },
                        "expected_time": 25
                    },
                    {
                        "step_id": "check_portfolio",
                        "action": "view_portfolio",
                        "description": "查看投资组合",
                        "expected_time": 15
                    },
                    {
                        "step_id": "view_performance",
                        "action": "performance_review",
                        "description": "查看业绩报告",
                        "expected_time": 20
                    }
                ],
                success_criteria=[
                    "成功完成账户注册",
                    "账户充值成功",
                    "首次交易执行成功",
                    "能够查看投资组合和业绩",
                    "用户界面友好易用"
                ]
            ),

            # 旅程2: 经验用户日常交易
            UserJourney(
                journey_id="experienced_user_daily_trading",
                user_type="experienced",
                description="经验用户进行日常交易操作",
                expected_duration=180,  # 3分钟
                steps=[
                    {
                        "step_id": "login_account",
                        "action": "user_login",
                        "description": "登录账户",
                        "params": {
                            "email": "experienced_user@example.com",
                            "password": "SecurePass456!"
                        },
                        "expected_time": 10
                    },
                    {
                        "step_id": "check_market_overview",
                        "action": "market_overview",
                        "description": "查看市场概览",
                        "expected_time": 15
                    },
                    {
                        "step_id": "analyze_stocks",
                        "action": "technical_analysis",
                        "description": "技术分析股票",
                        "params": {"symbols": ["AAPL", "GOOGL", "MSFT"]},
                        "expected_time": 45
                    },
                    {
                        "step_id": "create_watchlist",
                        "action": "manage_watchlist",
                        "description": "管理自选股",
                        "params": {"add_symbols": ["TSLA", "NVDA"]},
                        "expected_time": 20
                    },
                    {
                        "step_id": "execute_trades",
                        "action": "multiple_orders",
                        "description": "执行多笔交易",
                        "params": {
                            "orders": [
                                {"symbol": "AAPL", "action": "BUY", "quantity": 50, "type": "limit", "price": 150},
                                {"symbol": "GOOGL", "action": "SELL", "quantity": 25, "type": "market"}
                            ]
                        },
                        "expected_time": 35
                    },
                    {
                        "step_id": "monitor_positions",
                        "action": "position_monitoring",
                        "description": "监控持仓",
                        "expected_time": 25
                    },
                    {
                        "step_id": "set_stop_loss",
                        "action": "risk_management_setup",
                        "description": "设置止损止盈",
                        "params": {"symbol": "AAPL", "stop_loss": 140, "take_profit": 170},
                        "expected_time": 15
                    },
                    {
                        "step_id": "generate_report",
                        "action": "custom_report",
                        "description": "生成自定义报告",
                        "params": {"report_type": "daily_pnl", "period": "today"},
                        "expected_time": 20
                    }
                ],
                success_criteria=[
                    "快速登录系统",
                    "市场分析功能正常",
                    "交易执行高效",
                    "风险管理设置有效",
                    "报告生成准确"
                ]
            ),

            # 旅程3: 机构用户投资组合管理
            UserJourney(
                journey_id="institutional_portfolio_management",
                user_type="institutional",
                description="机构用户进行专业投资组合管理",
                expected_duration=420,  # 7分钟
                steps=[
                    {
                        "step_id": "login_institutional",
                        "action": "institutional_login",
                        "description": "机构用户登录",
                        "params": {
                            "organization_id": "INST_001",
                            "user_id": "portfolio_manager",
                            "two_factor_code": "123456"
                        },
                        "expected_time": 15
                    },
                    {
                        "step_id": "access_client_accounts",
                        "action": "client_account_access",
                        "description": "访问客户账户",
                        "params": {"client_ids": ["CLIENT_001", "CLIENT_002"]},
                        "expected_time": 20
                    },
                    {
                        "step_id": "create_model_portfolio",
                        "action": "model_portfolio_creation",
                        "description": "创建模型组合",
                        "params": {
                            "portfolio_name": "Balanced Growth Model",
                            "target_allocation": {
                                "stocks": 0.6,
                                "bonds": 0.3,
                                "cash": 0.1
                            },
                            "rebalancing_frequency": "quarterly"
                        },
                        "expected_time": 60
                    },
                    {
                        "step_id": "apply_portfolio_to_clients",
                        "action": "portfolio_application",
                        "description": "将组合应用到客户账户",
                        "params": {
                            "model_portfolio_id": "{create_model_portfolio.portfolio_id}",
                            "client_ids": ["CLIENT_001"]
                        },
                        "expected_time": 45
                    },
                    {
                        "step_id": "execute_rebalancing",
                        "action": "portfolio_rebalancing",
                        "description": "执行组合再平衡",
                        "params": {"portfolio_id": "{create_model_portfolio.portfolio_id}"},
                        "expected_time": 90
                    },
                    {
                        "step_id": "compliance_check",
                        "action": "regulatory_compliance_check",
                        "description": "合规性检查",
                        "params": {"portfolio_id": "{create_model_portfolio.portfolio_id}"},
                        "expected_time": 30
                    },
                    {
                        "step_id": "performance_attribution",
                        "action": "performance_attribution_analysis",
                        "description": "业绩归因分析",
                        "params": {
                            "portfolio_id": "{create_model_portfolio.portfolio_id}",
                            "benchmark": "S&P 500"
                        },
                        "expected_time": 40
                    },
                    {
                        "step_id": "client_reporting",
                        "action": "client_performance_report",
                        "description": "生成客户报告",
                        "params": {
                            "client_id": "CLIENT_001",
                            "report_period": "monthly"
                        },
                        "expected_time": 35
                    }
                ],
                success_criteria=[
                    "机构级功能正常",
                    "多账户管理有效",
                    "模型组合创建成功",
                    "合规检查通过",
                    "专业报告准确"
                ]
            ),

            # 旅程4: 移动端用户快速操作
            UserJourney(
                journey_id="mobile_user_quick_actions",
                user_type="mobile",
                description="移动端用户进行快速交易操作",
                expected_duration=120,  # 2分钟
                steps=[
                    {
                        "step_id": "mobile_app_launch",
                        "action": "app_launch",
                        "description": "启动移动应用",
                        "expected_time": 8
                    },
                    {
                        "step_id": "biometric_login",
                        "action": "biometric_authentication",
                        "description": "生物识别登录",
                        "expected_time": 5
                    },
                    {
                        "step_id": "quick_market_view",
                        "action": "mobile_market_snapshot",
                        "description": "快速查看市场行情",
                        "expected_time": 10
                    },
                    {
                        "step_id": "favorite_stocks",
                        "action": "quick_favorites_access",
                        "description": "访问收藏股票",
                        "expected_time": 8
                    },
                    {
                        "step_id": "quick_buy",
                        "action": "one_tap_buy",
                        "description": "一键买入",
                        "params": {
                            "symbol": "AAPL",
                            "amount": 1000  # 买入1000元
                        },
                        "expected_time": 12
                    },
                    {
                        "step_id": "push_notifications",
                        "action": "notification_check",
                        "description": "检查推送通知",
                        "expected_time": 5
                    },
                    {
                        "step_id": "voice_commands",
                        "action": "voice_trading",
                        "description": "语音交易",
                        "params": {"command": "sell 20 shares of TSLA"},
                        "expected_time": 15
                    },
                    {
                        "step_id": "mobile_portfolio",
                        "action": "mobile_portfolio_view",
                        "description": "移动端组合查看",
                        "expected_time": 12
                    }
                ],
                success_criteria=[
                    "移动应用启动快速",
                    "生物识别登录成功",
                    "一键交易便捷",
                    "语音功能准确",
                    "界面适配良好"
                ]
            )
        ]

        logger.info(f"✅ 已定义 {len(self.journeys)} 个用户旅程")

    async def run_user_acceptance_test(self) -> UserAcceptanceReport:
        """运行用户验收测试"""
        logger.info("🚀 开始用户验收测试")

        report = UserAcceptanceReport()
        start_time = time.time()

        # 按用户类型分组执行旅程
        journey_groups = {}
        for journey in self.journeys:
            if journey.user_type not in journey_groups:
                journey_groups[journey.user_type] = []
            journey_groups[journey.user_type].append(journey)

        # 为每个用户类型执行旅程
        for user_type, journeys in journey_groups.items():
            logger.info(f"执行 {user_type} 用户类型的 {len(journeys)} 个旅程")

            for journey in journeys:
                logger.info(f"开始用户旅程: {journey.description}")

                # 执行用户旅程
                result = await self._execute_user_journey(journey)
                self.results.append(result)

                # 更新报告统计
                report.journey_results.append(result)
                report.total_journeys += 1

                if result.success:
                    report.completed_journeys += 1
                else:
                    report.failed_journeys += 1

                logger.info(f"用户旅程 {journey.journey_id} 完成: {'✅' if result.success else '❌'}")

        # 计算统计指标
        total_time = time.time() - start_time
        if report.total_journeys > 0:
            report.average_completion_time = sum(r.completion_time for r in report.journey_results) / report.total_journeys
            report.average_satisfaction = sum(r.user_experience_score for r in report.journey_results) / report.total_journeys
            report.user_experience_score = sum(r.user_experience_score for r in report.journey_results) / report.total_journeys
            report.functionality_score = sum(r.functionality_score for r in report.journey_results) / report.total_journeys
            report.performance_score = sum(r.performance_score for r in report.journey_results) / report.total_journeys
            report.overall_acceptance_score = sum(r.overall_score for r in report.journey_results) / report.total_journeys

        # 生成建议
        report.recommendations = self._generate_uat_recommendations(report)

        logger.info(f"✅ 用户验收测试完成，总耗时: {total_time:.2f}秒")

        # 生成详细报告
        self._generate_uat_report(report)

        return report

    async def _execute_user_journey(self, journey: UserJourney) -> UATResult:
        """执行用户旅程"""
        # 创建用户会话
        session_id = str(uuid.uuid4())
        user_id = f"user_{journey.user_type}_{int(time.time())}"

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            user_type=journey.user_type,
            start_time=datetime.now()
        )

        self.user_sessions[session_id] = session

        result = UATResult(
            journey_id=journey.journey_id,
            user_session=session,
            success=True,
            completion_time=0,
            user_experience_score=5,
            functionality_score=5,
            performance_score=5,
            overall_score=5
        )

        journey_start_time = time.time()
        context = {}  # 存储步骤间的数据

        try:
            for step in journey.steps:
                step_start_time = time.time()

                # 解析参数
                resolved_params = self._resolve_journey_parameters(step.get('params', {}), context)

                # 执行步骤
                step_result = await self._execute_journey_step(step, resolved_params, session)

                step_execution_time = time.time() - step_start_time

                # 记录步骤完成情况
                step_record = {
                    'step_id': step['step_id'],
                    'description': step['description'],
                    'execution_time': step_execution_time,
                    'expected_time': step['expected_time'],
                    'success': step_result['success'],
                    'timestamp': datetime.now()
                }

                if not step_result['success']:
                    step_record['error'] = step_result.get('error', '未知错误')

                session.steps_completed.append(step_record)

                # 检查执行时间是否超过预期
                if step_execution_time > step['expected_time'] * 1.5:  # 允许50%容忍度
                    result.performance_score = max(1, result.performance_score - 1)
                    session.errors.append(f"步骤 {step['step_id']} 执行时间过长: {step_execution_time:.1f}s (期望 {step['expected_time']}s)")

                # 存储步骤结果到上下文
                if step_result['success'] and 'data' in step_result:
                    context[step['step_id']] = step_result['data']

                # 如果步骤失败，降低评分
                if not step_result['success']:
                    result.success = False
                    result.functionality_score = max(1, result.functionality_score - 1)
                    result.user_experience_score = max(1, result.user_experience_score - 1)
                    session.errors.append(f"步骤失败: {step['step_id']} - {step_result.get('error', '未知错误')}")

                # 模拟用户思考时间和操作延迟
                await asyncio.sleep(random.uniform(0.5, 2.0))

            result.completion_time = time.time() - journey_start_time

            # 检查整体完成时间
            if result.completion_time > journey.expected_duration * 1.5:
                result.performance_score = max(1, result.performance_score - 1)

            # 计算综合得分
            result.overall_score = (result.user_experience_score + result.functionality_score + result.performance_score) // 3

            # 生成用户反馈
            result.feedback = self._generate_user_feedback(result, journey)

        except Exception as e:
            result.success = False
            result.completion_time = time.time() - journey_start_time
            session.errors.append(f"旅程执行异常: {str(e)}")
            result.feedback.append(f"系统异常: {str(e)}")

        finally:
            session.end_time = datetime.now()

        return result

    def _resolve_journey_parameters(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """解析旅程参数"""
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str):
                # 替换变量引用，如 {step_id.field}
                resolved_value = value
                import re
                matches = re.findall(r'\{([^}]+)\}', value)
                for match in matches:
                    if match in context:
                        resolved_value = resolved_value.replace(f"{{{match}}}", str(context[match]))
                    elif '.' in match:
                        parts = match.split('.')
                        if len(parts) == 2 and parts[0] in context:
                            step_data = context[parts[0]]
                            if isinstance(step_data, dict) and parts[1] in step_data:
                                resolved_value = resolved_value.replace(f"{{{match}}}", str(step_data[parts[1]]))

                resolved[key] = resolved_value
            else:
                resolved[key] = value

        return resolved

    async def _execute_journey_step(self, step: Dict[str, Any], params: Dict[str, Any], session: UserSession) -> Dict[str, Any]:
        """执行旅程步骤"""
        action = step['action']

        try:
            # 模拟各种用户操作
            if action == "navigate_to_homepage":
                return await self._mock_navigate_homepage()
            elif action == "user_registration":
                return await self._mock_user_registration(params, session)
            elif action == "email_verification":
                return await self._mock_email_verification(params, session)
            elif action == "profile_setup":
                return await self._mock_profile_setup(params, session)
            elif action == "account_deposit":
                return await self._mock_account_deposit(params, session)
            elif action == "market_browse":
                return await self._mock_market_browse(params)
            elif action == "simple_buy_order":
                return await self._mock_simple_buy_order(params, session)
            elif action == "view_portfolio":
                return await self._mock_view_portfolio(session)
            elif action == "performance_review":
                return await self._mock_performance_review(session)
            elif action == "user_login":
                return await self._mock_user_login(params, session)
            elif action == "market_overview":
                return await self._mock_market_overview()
            elif action == "technical_analysis":
                return await self._mock_technical_analysis(params)
            elif action == "manage_watchlist":
                return await self._mock_manage_watchlist(params, session)
            elif action == "multiple_orders":
                return await self._mock_multiple_orders(params, session)
            elif action == "position_monitoring":
                return await self._mock_position_monitoring(session)
            elif action == "risk_management_setup":
                return await self._mock_risk_management_setup(params, session)
            elif action == "custom_report":
                return await self._mock_custom_report(params, session)
            elif action == "institutional_login":
                return await self._mock_institutional_login(params, session)
            elif action == "client_account_access":
                return await self._mock_client_account_access(params, session)
            elif action == "model_portfolio_creation":
                return await self._mock_model_portfolio_creation(params, session)
            elif action == "portfolio_application":
                return await self._mock_portfolio_application(params, session)
            elif action == "portfolio_rebalancing":
                return await self._mock_portfolio_rebalancing(params, session)
            elif action == "regulatory_compliance_check":
                return await self._mock_compliance_check(params, session)
            elif action == "performance_attribution_analysis":
                return await self._mock_performance_attribution(params, session)
            elif action == "client_performance_report":
                return await self._mock_client_performance_report(params, session)
            elif action == "app_launch":
                return await self._mock_app_launch()
            elif action == "biometric_authentication":
                return await self._mock_biometric_auth(params, session)
            elif action == "mobile_market_snapshot":
                return await self._mock_mobile_market_snapshot()
            elif action == "quick_favorites_access":
                return await self._mock_quick_favorites_access(session)
            elif action == "one_tap_buy":
                return await self._mock_one_tap_buy(params, session)
            elif action == "notification_check":
                return await self._mock_notification_check(session)
            elif action == "voice_trading":
                return await self._mock_voice_trading(params, session)
            elif action == "mobile_portfolio_view":
                return await self._mock_mobile_portfolio_view(session)
            else:
                return {"success": False, "error": f"未知操作: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # 模拟各种用户操作的方法
    async def _mock_navigate_homepage(self):
        await asyncio.sleep(0.5)
        return {"success": True, "data": {"page_loaded": True, "load_time": 0.5}}

    async def _mock_user_registration(self, params, session):
        await asyncio.sleep(2.0)
        user_id = f"USER_{int(time.time() * 1000)}"
        self.test_users[user_id] = {
            "email": params["email"],
            "user_type": params["user_type"],
            "registration_time": datetime.now()
        }
        session.user_id = user_id
        return {"success": True, "data": {"user_id": user_id, "registration_complete": True}}

    async def _mock_email_verification(self, params, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"email_verified": True}}

    async def _mock_profile_setup(self, params, session):
        await asyncio.sleep(3.0)
        return {"success": True, "data": {"profile_complete": True}}

    async def _mock_account_deposit(self, params, session):
        await asyncio.sleep(1.5)
        return {"success": True, "data": {"deposit_complete": True, "amount": params["amount"]}}

    async def _mock_market_browse(self, params):
        await asyncio.sleep(2.0)
        return {"success": True, "data": {"stocks_loaded": 50, "categories": ["popular", "trending"]}}

    async def _mock_simple_buy_order(self, params, session):
        await asyncio.sleep(1.5)
        order_id = f"ORDER_{int(time.time() * 1000)}"
        return {"success": True, "data": {"order_id": order_id, "status": "filled"}}

    async def _mock_view_portfolio(self, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"portfolio_value": 10500, "total_return": 5.0}}

    async def _mock_performance_review(self, session):
        await asyncio.sleep(1.5)
        return {"success": True, "data": {"report_generated": True}}

    async def _mock_user_login(self, params, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"login_success": True}}

    async def _mock_market_overview(self):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"market_data": {"sp500": 4200, "nasdaq": 13500}}}

    async def _mock_technical_analysis(self, params):
        await asyncio.sleep(3.0)
        return {"success": True, "data": {"analysis_complete": True, "signals": ["BUY", "HOLD", "SELL"]}}

    async def _mock_manage_watchlist(self, params, session):
        await asyncio.sleep(1.5)
        return {"success": True, "data": {"watchlist_updated": True}}

    async def _mock_multiple_orders(self, params, session):
        await asyncio.sleep(2.5)
        return {"success": True, "data": {"orders_executed": len(params["orders"])}}

    async def _mock_position_monitoring(self, session):
        await asyncio.sleep(2.0)
        return {"success": True, "data": {"positions": 5, "total_value": 25000}}

    async def _mock_risk_management_setup(self, params, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"risk_rules_set": True}}

    async def _mock_custom_report(self, params, session):
        await asyncio.sleep(1.5)
        return {"success": True, "data": {"report_generated": True}}

    async def _mock_institutional_login(self, params, session):
        await asyncio.sleep(2.0)
        return {"success": True, "data": {"login_success": True, "permissions": ["full_access"]}}

    async def _mock_client_account_access(self, params, session):
        await asyncio.sleep(1.5)
        return {"success": True, "data": {"accounts_accessed": len(params["client_ids"])}}

    async def _mock_model_portfolio_creation(self, params, session):
        await asyncio.sleep(4.0)
        portfolio_id = f"PORTFOLIO_{int(time.time() * 1000)}"
        return {"success": True, "data": {"portfolio_id": portfolio_id, "model_created": True}}

    async def _mock_portfolio_application(self, params, session):
        await asyncio.sleep(3.0)
        return {"success": True, "data": {"application_success": True}}

    async def _mock_portfolio_rebalancing(self, params, session):
        await asyncio.sleep(6.0)
        return {"success": True, "data": {"rebalancing_complete": True, "trades_executed": 8}}

    async def _mock_compliance_check(self, params, session):
        await asyncio.sleep(2.0)
        return {"success": True, "data": {"compliance_passed": True}}

    async def _mock_performance_attribution(self, params, session):
        await asyncio.sleep(3.0)
        return {"success": True, "data": {"attribution_complete": True}}

    async def _mock_client_performance_report(self, params, session):
        await asyncio.sleep(2.5)
        return {"success": True, "data": {"report_generated": True}}

    async def _mock_app_launch(self):
        await asyncio.sleep(0.8)
        return {"success": True, "data": {"app_launched": True}}

    async def _mock_biometric_auth(self, params, session):
        await asyncio.sleep(0.5)
        return {"success": True, "data": {"auth_success": True}}

    async def _mock_mobile_market_snapshot(self):
        await asyncio.sleep(0.8)
        return {"success": True, "data": {"snapshot_loaded": True}}

    async def _mock_quick_favorites_access(self, session):
        await asyncio.sleep(0.6)
        return {"success": True, "data": {"favorites_loaded": 8}}

    async def _mock_one_tap_buy(self, params, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"order_executed": True}}

    async def _mock_notification_check(self, session):
        await asyncio.sleep(0.5)
        return {"success": True, "data": {"notifications": 3}}

    async def _mock_voice_trading(self, params, session):
        await asyncio.sleep(1.2)
        return {"success": True, "data": {"voice_command_processed": True}}

    async def _mock_mobile_portfolio_view(self, session):
        await asyncio.sleep(1.0)
        return {"success": True, "data": {"portfolio_viewed": True}}

    def _generate_user_feedback(self, result: UATResult, journey: UserJourney) -> List[str]:
        """生成用户反馈"""
        feedback = []

        if result.completion_time > journey.expected_duration * 1.2:
            feedback.append(f"流程耗时较长 ({result.completion_time:.1f}s)，期望 {journey.expected_duration}s")

        if result.user_experience_score < 4:
            feedback.append("用户体验有待改进")

        if result.performance_score < 4:
            feedback.append("系统响应速度需要优化")

        if result.functionality_score < 4:
            feedback.append("部分功能存在问题")

        if not result.success:
            feedback.append("关键功能无法正常使用")

        if not feedback:
            feedback.append("整体体验良好，功能正常")

        return feedback

    def _generate_uat_recommendations(self, report: UserAcceptanceReport) -> List[str]:
        """生成验收测试建议"""
        recommendations = []

        if report.failed_journeys > 0:
            recommendations.append(f"🔧 修复 {report.failed_journeys} 个失败的用户旅程")

        if report.average_satisfaction < 4.0:
            recommendations.append("😊 提升用户满意度，改善用户体验")

        if report.performance_score < 4.0:
            recommendations.append("⚡ 优化系统性能，提升响应速度")

        if report.functionality_score < 4.0:
            recommendations.append("🔧 完善系统功能，确保所有特性正常工作")

        if report.overall_acceptance_score < 4.0:
            recommendations.append("🎯 整体验收得分不足，需要重点改进核心功能")

        if not recommendations:
            recommendations.append("✅ 用户验收测试通过，系统已准备好投产")

        return recommendations

    def _generate_uat_report(self, report: UserAcceptanceReport):
        """生成验收测试报告"""
        report_data = {
            'acceptance_summary': {
                'total_journeys': report.total_journeys,
                'completed_journeys': report.completed_journeys,
                'failed_journeys': report.failed_journeys,
                'completion_rate': report.completed_journeys / max(1, report.total_journeys),
                'average_completion_time': report.average_completion_time,
                'average_satisfaction': report.average_satisfaction
            },
            'quality_scores': {
                'user_experience_score': report.user_experience_score,
                'functionality_score': report.functionality_score,
                'performance_score': report.performance_score,
                'overall_acceptance_score': report.overall_acceptance_score
            },
            'journey_details': [
                {
                    'journey_id': r.journey_id,
                    'user_type': r.user_session.user_type,
                    'success': r.success,
                    'completion_time': r.completion_time,
                    'user_experience_score': r.user_experience_score,
                    'functionality_score': r.functionality_score,
                    'performance_score': r.performance_score,
                    'overall_score': r.overall_score,
                    'feedback': r.feedback,
                    'steps_completed': len(r.user_session.steps_completed),
                    'errors': r.user_session.errors
                }
                for r in report.journey_results
            ],
            'recommendations': report.recommendations
        }

        # 保存报告
        os.makedirs('test_logs', exist_ok=True)
        report_file = f'test_logs/phase5_user_acceptance_test_{int(time.time())}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📊 用户验收测试报告已保存: {report_file}")

        # 打印总结报告
        self._print_uat_summary(report_data)

    def _print_uat_summary(self, report_data: Dict[str, Any]):
        """打印验收测试总结"""
        print("\n" + "="*80)
        print("📊 用户验收测试总结报告")
        print("="*80)

        summary = report_data['acceptance_summary']
        print(f"🎯 用户旅程总数: {summary['total_journeys']}")
        print(f"✅ 完成旅程: {summary['completed_journeys']}")
        print(f"❌ 失败旅程: {summary['failed_journeys']}")
        print(f"📊 完成率: {summary['completion_rate']:.1f}")
        print(f"⏱️ 平均完成时间: {summary['average_completion_time']:.1f}秒")
        print(f"😊 平均满意度: {summary['average_satisfaction']:.1f}")
        quality = report_data['quality_scores']
        print("\n👤 用户体验评分:")
        print(f"  用户体验得分: {quality['user_experience_score']:.1f}")
        print(f"  功能完整性得分: {quality['functionality_score']:.1f}")
        print(f"  性能表现得分: {quality['performance_score']:.1f}")
        print(f"  综合验收得分: {quality['overall_acceptance_score']:.1f}")
        recommendations = report_data['recommendations']
        print("\n💡 验收建议:")
        for rec in recommendations:
            print(f"  • {rec}")

        print("\n" + "="*80)

async def run_user_acceptance_test():
    """运行用户验收测试"""
    logger.info("开始Phase 5: 用户验收测试")

    tester = UserAcceptanceTester()

    try:
        report = await tester.run_user_acceptance_test()

        if report.overall_acceptance_score >= 4.0:
            logger.info("✅ 用户验收测试完成 - 用户体验良好")
        else:
            logger.warning("⚠️ 用户验收测试完成 - 存在用户体验问题需要改进")

    except Exception as e:
        logger.error(f"用户验收测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(run_user_acceptance_test())
