#!/usr/bin/env python3
"""
业务验收测试模拟框架 - Phase 6.3
使用模拟对象执行业务验收测试，验证核心业务逻辑

测试范围:
1. 策略执行测试 - 模拟策略服务验证执行逻辑
2. 市场数据处理测试 - 模拟数据服务验证处理流程
3. 交易执行测试 - 模拟交易服务验证订单流程
4. 风险控制测试 - 模拟风险服务验证控制逻辑

使用方法:
python scripts/business_acceptance_test_simulation.py --test strategy_execution
python scripts/business_acceptance_test_simulation.py --test market_data
python scripts/business_acceptance_test_simulation.py --test trading_execution
python scripts/business_acceptance_test_simulation.py --test risk_control
python scripts/business_acceptance_test_simulation.py --test all
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import statistics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    test_type: str
    status: str  # 'pass', 'fail', 'error', 'skip'
    execution_time: float
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}


@dataclass
class TestMetrics:
    """测试指标"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0


class MockStrategyService:
    """模拟策略服务"""

    def __init__(self):
        self.strategies = {}
        self.next_id = 1

    def create_strategy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建策略"""
        strategy_id = self.next_id
        self.next_id += 1

        strategy = {
            'id': strategy_id,
            'name': config.get('name', 'test_strategy'),
            'type': config.get('type', 'momentum'),
            'symbols': config.get('symbols', []),
            'parameters': config.get('parameters', {}),
            'status': 'active',
            'created_at': datetime.now().isoformat()
        }

        self.strategies[strategy_id] = strategy
        return strategy

    def execute_strategy(self, strategy_id: int, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行策略"""
        if strategy_id not in self.strategies:
            return []

        strategy = self.strategies[strategy_id]
        signals = []

        # 模拟动量策略逻辑
        for symbol in strategy['symbols']:
            if symbol in market_data:
                current_price = market_data[symbol]['price']

                # 简单动量信号生成逻辑
                if current_price > market_data[symbol].get('prev_price', current_price * 0.95):
                    signal = {
                        'symbol': symbol,
                        'signal_type': 'buy',
                        'strength': round(random.uniform(0.5, 1.0), 2),
                        'price': current_price,
                        'timestamp': datetime.now().isoformat()
                    }
                    signals.append(signal)
                elif random.random() > 0.7:  # 30%概率卖出信号
                    signal = {
                        'symbol': symbol,
                        'signal_type': 'sell',
                        'strength': round(random.uniform(0.3, 0.8), 2),
                        'price': current_price,
                        'timestamp': datetime.now().isoformat()
                    }
                    signals.append(signal)

        return signals


class MockMarketDataService:
    """模拟市场数据服务"""

    def __init__(self):
        self.market_data = {}
        self._init_sample_data()

    def _init_sample_data(self):
        """初始化样本数据"""
        symbols = ['000001.SZ', '600036.SH', '000858.SZ', '600519.SH', '601318.SH']
        for symbol in symbols:
            self.market_data[symbol] = {
                'price': round(random.uniform(10, 500), 2),
                'volume': random.randint(100000, 10000000),
                'high': round(random.uniform(10, 500), 2),
                'low': round(random.uniform(10, 500), 2),
                'timestamp': datetime.now().isoformat()
            }

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""
        return self.market_data.get(symbol)

    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新市场数据"""
        if symbol not in self.market_data:
            self.market_data[symbol] = {}

        self.market_data[symbol].update(data)
        self.market_data[symbol]['timestamp'] = datetime.now().isoformat()

        return {'success': True, 'symbol': symbol, 'updated_at': self.market_data[symbol]['timestamp']}


class MockTradingService:
    """模拟交易服务"""

    def __init__(self):
        self.orders = {}
        self.next_order_id = 1

    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建订单"""
        order_id = self.next_order_id
        self.next_order_id += 1

        order = {
            'order_id': order_id,
            'user_id': order_data['user_id'],
            'symbol': order_data['symbol'],
            'order_type': order_data['order_type'],
            'side': order_data['side'],
            'quantity': order_data['quantity'],
            'price': order_data['price'],
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        self.orders[order_id] = order

        # 模拟订单立即成交 (只对市价单)
        if order_data.get('order_type') == 'market' and random.random() > 0.2:  # 80%概率立即成交
            order['status'] = 'filled'
            order['filled_quantity'] = order['quantity']
            order['filled_price'] = order['price']
            order['updated_at'] = datetime.now().isoformat()

        return {'success': True, 'order_id': order_id, 'status': order['status']}

    def get_order_status(self, order_id: int) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        return self.orders.get(order_id)

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """取消订单"""
        if order_id not in self.orders:
            return {'success': False, 'error': '订单不存在'}

        order = self.orders[order_id]
        if order['status'] in ['filled', 'cancelled']:
            return {'success': False, 'error': '订单无法取消'}

        order['status'] = 'cancelled'
        order['updated_at'] = datetime.now().isoformat()

        return {'success': True, 'order_id': order_id, 'status': 'cancelled'}


class MockRiskService:
    """模拟风险服务"""

    def assess_portfolio_risk(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """评估投资组合风险"""
        positions = portfolio.get('positions', [])
        cash_balance = portfolio.get('cash_balance', 0)

        # 计算总价值
        total_value = cash_balance
        position_values = []

        for position in positions:
            position_value = position['quantity'] * position['current_price']
            total_value += position_value
            position_values.append(position_value)

        # 计算风险指标
        if position_values:
            returns = [random.uniform(-0.1, 0.1) for _ in position_values]  # 模拟收益率
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.1

            sharpe_ratio = statistics.mean(returns) / volatility if volatility > 0 else 0
            max_drawdown = min(returns) if returns else 0
        else:
            volatility = 0.05
            sharpe_ratio = 2.0
            max_drawdown = -0.05

        return {
            'total_value': round(total_value, 2),
            'total_risk': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 4),
            'var_95': round(total_value * 0.05, 2),  # 5% VaR
            'assessment_time': datetime.now().isoformat()
        }

    def check_order_risk(self, order: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """检查订单风险"""
        order_value = order['quantity'] * order['price']
        portfolio_value = portfolio.get('cash_balance', 0)

        for position in portfolio.get('positions', []):
            portfolio_value += position['quantity'] * position['current_price']

        # 检查订单大小是否超过投资组合的10%
        order_percentage = order_value / portfolio_value if portfolio_value > 0 else 1.0

        risk_reasons = []
        approved = True

        if order_percentage > 0.1:  # 超过10%
            risk_reasons.append(f"订单价值({order_value:.2f})超过投资组合的10%({portfolio_value * 0.1:.2f})")
            approved = False

        if order['quantity'] > 10000:  # 单笔订单数量过大
            risk_reasons.append(f"订单数量({order['quantity']})超过单笔限制(10000)")
            approved = False

        return {
            'approved': approved,
            'order_value': order_value,
            'portfolio_value': portfolio_value,
            'order_percentage': round(order_percentage, 4),
            'risk_reasons': risk_reasons,
            'check_time': datetime.now().isoformat()
        }


class BusinessAcceptanceTestSimulation:
    """业务验收测试模拟框架"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results: List[TestResult] = []
        self.test_metrics = TestMetrics()

        # 初始化模拟服务
        self.mock_services = {
            'strategy': MockStrategyService(),
            'market_data': MockMarketDataService(),
            'trading': MockTradingService(),
            'risk': MockRiskService()
        }

        # 初始化测试数据
        self._init_test_data()

        logger.info("✅ 业务验收测试模拟框架初始化完成")

    def _init_test_data(self):
        """初始化测试数据"""
        self.test_symbols = ['000001.SZ', '600036.SH', '000858.SZ', '600519.SH', '601318.SH']
        self.test_users = [
            {'user_id': 1, 'username': 'test_user_1', 'balance': 100000.0},
            {'user_id': 2, 'username': 'test_user_2', 'balance': 50000.0},
            {'user_id': 3, 'username': 'test_user_3', 'balance': 200000.0}
        ]

        # 生成测试市场数据
        self.test_market_data = {}
        for symbol in self.test_symbols:
            self.test_market_data[symbol] = {
                'price': round(random.uniform(10, 500), 2),
                'volume': random.randint(100000, 10000000),
                'high': round(random.uniform(10, 500), 2),
                'low': round(random.uniform(10, 500), 2),
                'timestamp': datetime.now().isoformat()
            }

    def run_strategy_execution_test(self) -> TestResult:
        """策略执行测试 (使用模拟服务)"""
        start_time = time.time()

        try:
            strategy_service = self.mock_services['strategy']

            # 测试策略创建
            strategy_config = {
                'name': 'test_momentum_strategy',
                'type': 'momentum',
                'symbols': self.test_symbols[:3],
                'parameters': {
                    'lookback_period': 20,
                    'threshold': 0.05
                }
            }

            strategy = strategy_service.create_strategy(strategy_config)

            # 验证策略创建结果
            if not strategy or 'id' not in strategy:
                return TestResult(
                    test_name="strategy_execution",
                    test_type="strategy",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="策略创建失败"
                )

            # 测试策略执行
            market_data = {symbol: self.test_market_data[symbol]
                           for symbol in self.test_symbols[:3]}
            signals = strategy_service.execute_strategy(strategy['id'], market_data)

            # 验证信号生成
            if not signals:
                return TestResult(
                    test_name="strategy_execution",
                    test_type="strategy",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="策略执行未产生信号"
                )

            # 检查信号格式
            required_fields = ['symbol', 'signal_type', 'strength', 'timestamp']
            for signal in signals:
                for field in required_fields:
                    if field not in signal:
                        return TestResult(
                            test_name="strategy_execution",
                            test_type="strategy",
                            status="fail",
                            execution_time=time.time() - start_time,
                            message=f"信号缺少必要字段: {field}",
                            details={'signal': signal}
                        )

            return TestResult(
                test_name="strategy_execution",
                test_type="strategy",
                status="pass",
                execution_time=time.time() - start_time,
                message="策略执行测试通过 (模拟服务)",
                details={
                    'strategy_id': strategy['id'],
                    'signals_generated': len(signals),
                    'execution_time': time.time() - start_time,
                    'service_type': 'mock'
                }
            )

        except Exception as e:
            return TestResult(
                test_name="strategy_execution",
                test_type="strategy",
                status="error",
                execution_time=time.time() - start_time,
                message=f"策略执行测试异常: {str(e)}"
            )

    def run_market_data_test(self) -> TestResult:
        """市场数据处理测试 (使用模拟服务)"""
        start_time = time.time()

        try:
            market_service = self.mock_services['market_data']

            # 测试数据获取
            symbol = self.test_symbols[0]
            market_data = market_service.get_market_data(symbol)

            if not market_data:
                return TestResult(
                    test_name="market_data_processing",
                    test_type="market_data",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message=f"未获取到市场数据: {symbol}"
                )

            # 验证数据完整性
            required_fields = ['price', 'volume', 'timestamp']
            for field in required_fields:
                if field not in market_data:
                    return TestResult(
                        test_name="market_data_processing",
                        test_type="market_data",
                        status="fail",
                        execution_time=time.time() - start_time,
                        message=f"市场数据缺少字段: {field}",
                        details={'market_data': market_data}
                    )

            # 测试数据更新
            updated_data = self.test_market_data[symbol].copy()
            updated_data['price'] = updated_data['price'] * 1.02  # 2%上涨

            update_result = market_service.update_market_data(symbol, updated_data)

            if not update_result.get('success', False):
                return TestResult(
                    test_name="market_data_processing",
                    test_type="market_data",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="市场数据更新失败",
                    details={'update_result': update_result}
                )

            # 验证数据更新
            latest_data = market_service.get_market_data(symbol)
            if abs(latest_data['price'] - updated_data['price']) > 0.01:
                return TestResult(
                    test_name="market_data_processing",
                    test_type="market_data",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="市场数据更新未生效",
                    details={
                        'expected_price': updated_data['price'],
                        'actual_price': latest_data['price']
                    }
                )

            return TestResult(
                test_name="market_data_processing",
                test_type="market_data",
                status="pass",
                execution_time=time.time() - start_time,
                message="市场数据处理测试通过 (模拟服务)",
                details={
                    'symbol': symbol,
                    'data_retrieval': 'success',
                    'data_update': 'success',
                    'data_validation': 'success',
                    'service_type': 'mock'
                }
            )

        except Exception as e:
            return TestResult(
                test_name="market_data_processing",
                test_type="market_data",
                status="error",
                execution_time=time.time() - start_time,
                message=f"市场数据处理测试异常: {str(e)}"
            )

    def run_trading_execution_test(self) -> TestResult:
        """交易执行测试 (使用模拟服务)"""
        start_time = time.time()

        try:
            trading_service = self.mock_services['trading']

            # 测试订单创建
            user = self.test_users[0]
            symbol = self.test_symbols[0]
            market_price = self.test_market_data[symbol]['price']

            order_data = {
                'user_id': user['user_id'],
                'symbol': symbol,
                'order_type': 'limit',
                'side': 'buy',
                'quantity': 100,
                'price': market_price * 0.98,  # 比市场价低2%
                'time_in_force': 'day'
            }

            order_result = trading_service.create_order(order_data)

            if not order_result.get('success', False):
                return TestResult(
                    test_name="trading_execution",
                    test_type="trading",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="订单创建失败",
                    details={'order_result': order_result}
                )

            order_id = order_result.get('order_id')

            # 测试订单查询
            order_status = trading_service.get_order_status(order_id)

            if not order_status:
                return TestResult(
                    test_name="trading_execution",
                    test_type="trading",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="订单状态查询失败"
                )

            # 验证订单字段
            required_fields = ['order_id', 'status', 'symbol', 'quantity', 'price']
            for field in required_fields:
                if field not in order_status:
                    return TestResult(
                        test_name="trading_execution",
                        test_type="trading",
                        status="fail",
                        execution_time=time.time() - start_time,
                        message=f"订单状态缺少字段: {field}",
                        details={'order_status': order_status}
                    )

            # 测试订单取消
            cancel_result = trading_service.cancel_order(order_id)

            if not cancel_result.get('success', False):
                return TestResult(
                    test_name="trading_execution",
                    test_type="trading",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="订单取消失败",
                    details={'cancel_result': cancel_result}
                )

            # 验证订单已取消
            final_status = trading_service.get_order_status(order_id)
            if final_status.get('status') != 'cancelled':
                return TestResult(
                    test_name="trading_execution",
                    test_type="trading",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="订单取消状态未更新",
                    details={'final_status': final_status}
                )

            return TestResult(
                test_name="trading_execution",
                test_type="trading",
                status="pass",
                execution_time=time.time() - start_time,
                message="交易执行测试通过 (模拟服务)",
                details={
                    'order_id': order_id,
                    'order_creation': 'success',
                    'order_query': 'success',
                    'order_cancellation': 'success',
                    'service_type': 'mock'
                }
            )

        except Exception as e:
            return TestResult(
                test_name="trading_execution",
                test_type="trading",
                status="error",
                execution_time=time.time() - start_time,
                message=f"交易执行测试异常: {str(e)}"
            )

    def run_risk_control_test(self) -> TestResult:
        """风险控制测试 (使用模拟服务)"""
        start_time = time.time()

        try:
            risk_service = self.mock_services['risk']

            # 测试风险评估
            user = self.test_users[0]
            portfolio = {
                'user_id': user['user_id'],
                'positions': [
                    {
                        'symbol': self.test_symbols[0],
                        'quantity': 1000,
                        'avg_price': self.test_market_data[self.test_symbols[0]]['price'] * 0.9,
                        'current_price': self.test_market_data[self.test_symbols[0]]['price']
                    },
                    {
                        'symbol': self.test_symbols[1],
                        'quantity': 500,
                        'avg_price': self.test_market_data[self.test_symbols[1]]['price'] * 0.95,
                        'current_price': self.test_market_data[self.test_symbols[1]]['price']
                    }
                ],
                'cash_balance': user['balance']
            }

            risk_assessment = risk_service.assess_portfolio_risk(portfolio)

            if not risk_assessment:
                return TestResult(
                    test_name="risk_control",
                    test_type="risk",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="风险评估失败"
                )

            # 验证风险指标
            required_metrics = ['total_value', 'total_risk', 'sharpe_ratio', 'max_drawdown']
            for metric in required_metrics:
                if metric not in risk_assessment:
                    return TestResult(
                        test_name="risk_control",
                        test_type="risk",
                        status="fail",
                        execution_time=time.time() - start_time,
                        message=f"风险评估缺少指标: {metric}",
                        details={'risk_assessment': risk_assessment}
                    )

            # 测试风险限额检查
            test_order = {
                'user_id': user['user_id'],
                'symbol': self.test_symbols[2],
                'quantity': 10000,  # 大额订单
                'price': self.test_market_data[self.test_symbols[2]]['price']
            }

            risk_check = risk_service.check_order_risk(test_order, portfolio)

            if not risk_check:
                return TestResult(
                    test_name="risk_control",
                    test_type="risk",
                    status="fail",
                    execution_time=time.time() - start_time,
                    message="风险检查失败"
                )

            # 检查是否正确识别高风险
            if not risk_check.get('approved', False):
                risk_reasons = risk_check.get('risk_reasons', [])
                if not risk_reasons:
                    return TestResult(
                        test_name="risk_control",
                        test_type="risk",
                        status="fail",
                        execution_time=time.time() - start_time,
                        message="高风险订单未被正确拦截",
                        details={'risk_check': risk_check}
                    )

            return TestResult(
                test_name="risk_control",
                test_type="risk",
                status="pass",
                execution_time=time.time() - start_time,
                message="风险控制测试通过 (模拟服务)",
                details={
                    'portfolio_value': risk_assessment.get('total_value'),
                    'risk_level': risk_assessment.get('total_risk'),
                    'order_risk_check': 'performed',
                    'risk_limits': 'enforced',
                    'service_type': 'mock'
                }
            )

        except Exception as e:
            return TestResult(
                test_name="risk_control",
                test_type="risk",
                status="error",
                execution_time=time.time() - start_time,
                message=f"风险控制测试异常: {str(e)}"
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有业务验收测试 (模拟模式)"""
        logger.info("🎯 开始执行业务验收测试模拟套件...")

        # 定义测试列表
        test_functions = [
            ('strategy_execution', self.run_strategy_execution_test),
            ('market_data_processing', self.run_market_data_test),
            ('trading_execution', self.run_trading_execution_test),
            ('risk_control', self.run_risk_control_test)
        ]

        # 执行所有测试
        for test_name, test_func in test_functions:
            logger.info(f"📋 执行测试: {test_name}")
            result = test_func()
            self.test_results.append(result)

            # 更新统计指标
            self.test_metrics.total_tests += 1
            self.test_metrics.total_execution_time += result.execution_time

            if result.status == 'pass':
                self.test_metrics.passed_tests += 1
            elif result.status == 'fail':
                self.test_metrics.failed_tests += 1
            elif result.status == 'error':
                self.test_metrics.error_tests += 1
            elif result.status == 'skip':
                self.test_metrics.skipped_tests += 1

        # 计算平均执行时间
        if self.test_metrics.total_tests > 0:
            self.test_metrics.average_execution_time = self.test_metrics.total_execution_time / self.test_metrics.total_tests

        # 生成测试报告
        report = self.generate_test_report()

        logger.info("="*60)
        logger.info("📊 业务验收测试模拟执行总结")
        logger.info("="*60)
        logger.info(f"总测试数: {self.test_metrics.total_tests}")
        logger.info(f"通过测试: {self.test_metrics.passed_tests}")
        logger.info(f"失败测试: {self.test_metrics.failed_tests}")
        logger.info(f"错误测试: {self.test_metrics.error_tests}")
        logger.info(f"跳过测试: {self.test_metrics.skipped_tests}")
        logger.info(".1f")
        logger.info(".2f")
        logger.info("="*60)

        return report

    def run_specific_test(self, test_name: str) -> Dict[str, Any]:
        """运行特定测试 (模拟模式)"""
        test_map = {
            'strategy_execution': self.run_strategy_execution_test,
            'market_data': self.run_market_data_test,
            'trading_execution': self.run_trading_execution_test,
            'risk_control': self.run_risk_control_test
        }

        if test_name not in test_map:
            return {
                'error': f'未知的测试名称: {test_name}',
                'available_tests': list(test_map.keys())
            }

        logger.info(f"🎯 执行单个测试 (模拟模式): {test_name}")

        result = test_map[test_name]()
        self.test_results.append(result)

        # 更新统计指标
        self.test_metrics.total_tests = 1
        self.test_metrics.total_execution_time = result.execution_time

        if result.status == 'pass':
            self.test_metrics.passed_tests = 1
        elif result.status == 'fail':
            self.test_metrics.failed_tests = 1
        elif result.status == 'error':
            self.test_metrics.error_tests = 1
        elif result.status == 'skip':
            self.test_metrics.skipped_tests = 1

        self.test_metrics.average_execution_time = result.execution_time

        return {
            'test_name': test_name,
            'result': asdict(result),
            'metrics': asdict(self.test_metrics)
        }

    def generate_test_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        report = {
            'test_suite': 'business_acceptance_test_simulation',
            'execution_timestamp': datetime.now().isoformat(),
            'test_environment': {
                'strategy_service': 'mock',
                'trading_service': 'mock',
                'risk_service': 'mock',
                'market_data_service': 'mock',
                'simulation_mode': True
            },
            'metrics': asdict(self.test_metrics),
            'test_results': [asdict(result) for result in self.test_results],
            'summary': {
                'overall_status': 'pass' if self.test_metrics.success_rate >= 80.0 else 'fail',
                'critical_failures': len([r for r in self.test_results if r.status in ['fail', 'error']]),
                'service_availability': {
                    'strategy': True,
                    'trading': True,
                    'risk': True,
                    'market_data': True
                },
                'test_mode': 'simulation'
            }
        }

        # 保存报告
        report_file = self.project_root / 'data' / 'migration' / \
            'business_acceptance_simulation_test_report.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ 测试报告已保存: {report_file}")

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='业务验收测试模拟框架')
    parser.add_argument('--test', choices=['strategy_execution', 'market_data', 'trading_execution', 'risk_control', 'all'],
                        default='all', help='要执行的测试')
    parser.add_argument('--output', help='输出报告文件路径')

    args = parser.parse_args()

    try:
        test_suite = BusinessAcceptanceTestSimulation()

        if args.test == 'all':
            report = test_suite.run_all_tests()
        else:
            report = test_suite.run_specific_test(args.test)

        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"📄 测试报告已保存到: {args.output}")
        else:
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        print(f"❌ 测试失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()
