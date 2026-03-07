"""
交易服务测试构建器

负责构建交易服务相关的测试套件
"""

from typing import List
from .base_builder import BaseTestBuilder, TestSuite, TestScenario, TestCase


class TradingServiceTestBuilder(BaseTestBuilder):
    """
    交易服务测试构建器
    
    职责：
    - 构建交易服务API的测试场景
    - 生成下单、撤单、查询订单等测试用例
    """
    
    def build_test_suite(self) -> TestSuite:
        """构建交易服务测试套件"""
        suite = TestSuite(
            id="trading_service_tests",
            name="交易服务API测试",
            description="RQA2025交易服务的完整API测试套件"
        )
        
        # 添加各类测试场景
        suite.scenarios.append(self._build_order_placement_scenario())
        suite.scenarios.append(self._build_order_cancellation_scenario())
        suite.scenarios.append(self._build_order_query_scenario())
        suite.scenarios.append(self._build_position_management_scenario())
        
        return suite
    
    def _build_order_placement_scenario(self) -> TestScenario:
        """构建订单下单测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="order_placement",
            name="订单下单",
            description="测试订单下单功能",
            endpoint="/api/v1/trading/orders",
            method="POST",
            variables={
                "symbol": "BTC/USDT",
                "order_type": "limit",
                "side": "buy",
                "quantity": 0.1,
                "price": 50000.0
            }
        )
        
        # 正常下单
        scenario.test_cases.append(self._create_test_case(
            case_id="order_placement_normal",
            title="正常限价买单",
            description="下一个正常的限价买单",
            priority="critical",
            category="functional",
            preconditions=["交易服务运行正常", "账户余额充足", "市场开放交易"],
            test_steps=[
                {"step": 1, "action": "准备订单参数"},
                {"step": 2, "action": "发送POST请求"},
                {"step": 3, "action": "验证订单响应"},
                {"step": 4, "action": "查询订单状态确认"}
            ],
            expected_results=[
                "返回状态码201",
                "返回订单ID",
                "订单状态为pending或filled"
            ],
            tags=["trading", "order", "buy", "limit"]
        ))
        
        # 余额不足场景
        scenario.test_cases.append(self._create_test_case(
            case_id="order_placement_insufficient_balance",
            title="余额不足处理",
            description="余额不足时的订单拒绝处理",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码400",
                "返回错误码INSUFFICIENT_BALANCE",
                "订单未被创建"
            ],
            tags=["trading", "order", "error_handling", "balance"]
        ))
        
        # 市场单测试
        scenario.test_cases.append(self._create_test_case(
            case_id="order_placement_market",
            title="市价单下单",
            description="测试市价单的下单和成交",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码201",
                "订单立即成交",
                "成交价格在合理范围内"
            ],
            tags=["trading", "order", "market"]
        ))
        
        return scenario
    
    def _build_order_cancellation_scenario(self) -> TestScenario:
        """构建订单撤单测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="order_cancellation",
            name="订单撤单",
            description="测试订单撤销功能",
            endpoint="/api/v1/trading/orders/{order_id}",
            method="DELETE",
            variables={
                "order_id": "test_order_001"
            },
            setup_steps=["创建一个pending状态的订单"]
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="order_cancel_normal",
            title="正常撤销订单",
            description="撤销一个pending状态的订单",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "订单状态变更为cancelled",
                "冻结资金解冻"
            ],
            tags=["trading", "cancel", "normal"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="order_cancel_filled",
            title="已成交订单撤销",
            description="尝试撤销已成交的订单",
            priority="medium",
            category="functional",
            expected_results=[
                "返回状态码400",
                "返回错误码ORDER_ALREADY_FILLED",
                "订单状态保持filled"
            ],
            tags=["trading", "cancel", "error_handling"]
        ))
        
        return scenario
    
    def _build_order_query_scenario(self) -> TestScenario:
        """构建订单查询测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="order_query",
            name="订单查询",
            description="测试订单查询功能",
            endpoint="/api/v1/trading/orders",
            method="GET",
            variables={
                "symbol": "BTC/USDT",
                "status": "all",
                "limit": 100
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="order_query_all",
            title="查询所有订单",
            description="查询账户的所有历史订单",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回订单列表",
                "支持分页和过滤"
            ],
            tags=["trading", "query", "list"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="order_query_by_id",
            title="根据ID查询单个订单",
            description="通过订单ID查询特定订单详情",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回完整订单信息",
                "包含订单状态和成交记录"
            ],
            tags=["trading", "query", "detail"]
        ))
        
        return scenario
    
    def _build_position_management_scenario(self) -> TestScenario:
        """构建持仓管理测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="position_management",
            name="持仓管理",
            description="测试持仓查询和管理功能",
            endpoint="/api/v1/trading/positions",
            method="GET",
            variables={
                "symbol": "BTC/USDT"
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="position_query_all",
            title="查询所有持仓",
            description="查询账户的所有当前持仓",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回持仓列表",
                "包含持仓数量、均价、盈亏等信息"
            ],
            tags=["trading", "position", "query"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="position_realtime_update",
            title="持仓实时更新",
            description="测试持仓数据的实时更新",
            priority="high",
            category="functional",
            expected_results=[
                "订单成交后持仓立即更新",
                "持仓数量和均价计算正确",
                "盈亏计算准确"
            ],
            tags=["trading", "position", "realtime"]
        ))
        
        return scenario

