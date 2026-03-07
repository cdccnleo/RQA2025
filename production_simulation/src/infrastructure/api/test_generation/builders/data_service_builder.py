"""
数据服务测试构建器

负责构建数据服务相关的测试套件
"""

from typing import List
from .base_builder import BaseTestBuilder, TestSuite, TestScenario, TestCase


class DataServiceTestBuilder(BaseTestBuilder):
    """
    数据服务测试构建器
    
    职责：
    - 构建数据服务API的测试场景
    - 生成市场数据、K线数据、实时行情等测试用例
    """
    
    def build_test_suite(self) -> TestSuite:
        """构建数据服务测试套件"""
        suite = TestSuite(
            id="data_service_tests",
            name="数据服务API测试",
            description="RQA2025数据服务的完整API测试套件"
        )
        
        # 添加各类测试场景
        suite.scenarios.append(self._build_market_data_scenario())
        suite.scenarios.append(self._build_kline_data_scenario())
        suite.scenarios.append(self._build_realtime_data_scenario())
        suite.scenarios.append(self._build_historical_data_scenario())
        
        return suite
    
    def _build_market_data_scenario(self) -> TestScenario:
        """构建市场数据获取测试场景"""
        scenario = self._create_market_data_scenario_base()

        # 添加各种测试用例
        scenario.test_cases.extend(self._create_market_data_normal_cases())
        scenario.test_cases.extend(self._create_market_data_error_cases())
        scenario.test_cases.extend(self._create_market_data_performance_cases())

        return scenario

    def _create_market_data_scenario_base(self) -> TestScenario:
        """创建市场数据测试场景的基础配置"""
        return self._create_test_scenario(
            scenario_id="market_data_retrieval",
            name="市场数据获取",
            description="测试市场数据的获取功能",
            endpoint="/api/v1/data/market/{symbol}",
            method="GET",
            variables={
                "symbol": "BTC/USDT",
                "valid_symbol": "BTC/USDT",
                "invalid_symbol": "INVALID/SYMBOL"
            }
        )

    def _create_market_data_normal_cases(self) -> List[TestCase]:
        """创建市场数据正常测试用例"""
        return [
            self._create_test_case(
                case_id="market_data_normal",
                title="正常获取市场数据",
                description="使用有效交易对获取市场数据的正常流程",
                priority="high",
                category="functional",
                preconditions=["数据服务运行正常", "交易对BTC/USDT存在"],
                test_steps=[
                    {"step": 1, "action": "发送GET请求到/api/v1/data/market/BTC/USDT"},
                    {"step": 2, "action": "验证响应状态码为200"},
                    {"step": 3, "action": "验证返回的市场数据结构"}
                ],
                expected_results=[
                    "返回状态码200",
                    "返回JSON格式的市场数据",
                    "包含价格、成交量等关键字段"
                ],
                tags=["market_data", "normal", "high_priority"]
            )
        ]

    def _create_market_data_error_cases(self) -> List[TestCase]:
        """创建市场数据异常测试用例"""
        return [
            self._create_test_case(
                case_id="market_data_invalid_symbol",
                title="无效交易对处理",
                description="使用无效交易对请求市场数据的异常处理",
                priority="high",
                category="functional",
                preconditions=["数据服务运行正常"],
                test_steps=[
                    {"step": 1, "action": "发送GET请求到/api/v1/data/market/INVALID/SYMBOL"},
                    {"step": 2, "action": "验证响应状态码为404"},
                    {"step": 3, "action": "验证错误信息"}
                ],
                expected_results=[
                    "返回状态码404",
                    "返回错误信息SYMBOL_NOT_FOUND",
                    "错误信息清晰说明交易对不存在"
                ],
                tags=["market_data", "error_handling", "invalid_input"]
            )
        ]

    def _create_market_data_performance_cases(self) -> List[TestCase]:
        """创建市场数据性能测试用例"""
        return [
            self._create_test_case(
                case_id="market_data_performance",
                title="市场数据获取性能",
                description="验证市场数据获取的响应时间",
                priority="medium",
                category="performance",
                preconditions=["数据服务运行正常", "交易对BTC/USDT存在"],
                test_steps=[
                    {"step": 1, "action": "连续发送100个请求"},
                    {"step": 2, "action": "记录每个请求的响应时间"},
                    {"step": 3, "action": "计算P95响应时间"}
                ],
                expected_results=[
                    "平均响应时间<100ms",
                    "P95响应时间<200ms",
                    "无超时请求"
                ],
                tags=["market_data", "performance"]
            )
        ]
    
    def _build_kline_data_scenario(self) -> TestScenario:
        """构建K线数据测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="kline_data_retrieval",
            name="K线数据获取",
            description="测试K线数据的获取功能",
            endpoint="/api/v1/data/kline",
            method="GET",
            variables={
                "symbol": "BTC/USDT",
                "interval": "1m",
                "valid_intervals": ["1m", "5m", "15m", "1h", "1d"]
            }
        )
        
        # 正常场景
        scenario.test_cases.append(self._create_test_case(
            case_id="kline_data_normal",
            title="正常获取K线数据",
            description="使用有效参数获取K线数据",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回K线数组",
                "K线数据包含OHLCV字段"
            ],
            tags=["kline", "normal"]
        ))
        
        # 参数验证场景
        scenario.test_cases.append(self._create_test_case(
            case_id="kline_data_invalid_interval",
            title="无效时间周期处理",
            description="使用无效的时间周期参数",
            priority="medium",
            category="functional",
            expected_results=[
                "返回状态码400",
                "返回错误码INVALID_INTERVAL",
                "提示有效的时间周期选项"
            ],
            tags=["kline", "validation", "error_handling"]
        ))
        
        return scenario
    
    def _build_realtime_data_scenario(self) -> TestScenario:
        """构建实时数据测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="realtime_data_subscription",
            name="实时数据订阅",
            description="测试实时数据WebSocket订阅功能",
            endpoint="/ws/v1/data/realtime",
            method="WS",
            variables={
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "channels": ["ticker", "depth", "trades"]
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="realtime_data_subscribe",
            title="订阅实时数据",
            description="建立WebSocket连接并订阅实时数据",
            priority="high",
            category="functional",
            expected_results=[
                "WebSocket连接成功建立",
                "订阅确认消息返回",
                "开始接收实时数据推送"
            ],
            tags=["realtime", "websocket", "subscription"]
        ))
        
        return scenario
    
    def _build_historical_data_scenario(self) -> TestScenario:
        """构建历史数据测试场景"""
        scenario = self._create_test_scenario(
            scenario_id="historical_data_query",
            name="历史数据查询",
            description="测试历史数据的查询和分页功能",
            endpoint="/api/v1/data/history",
            method="GET",
            variables={
                "symbol": "BTC/USDT",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "page_size": 100
            }
        )
        
        scenario.test_cases.append(self._create_test_case(
            case_id="historical_data_normal",
            title="正常查询历史数据",
            description="使用有效的时间范围查询历史数据",
            priority="high",
            category="functional",
            expected_results=[
                "返回状态码200",
                "返回历史数据列表",
                "支持分页查询"
            ],
            tags=["historical", "query", "pagination"]
        ))
        
        scenario.test_cases.append(self._create_test_case(
            case_id="historical_data_large_range",
            title="大时间范围查询",
            description="测试查询大时间范围的性能和限制",
            priority="medium",
            category="performance",
            expected_results=[
                "返回状态码200或400",
                "如果范围过大应返回错误",
                "响应时间在合理范围内"
            ],
            tags=["historical", "performance", "large_query"]
        ))
        
        return scenario

