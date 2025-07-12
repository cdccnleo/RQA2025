import unittest
from datetime import datetime, time
from src.risk.risk_engine import (
    RiskEngine,
    RuleValidator,
    MarketRuleValidator,
    RiskLevel,
    liquidity_risk_check
)

class TestRiskEngine(unittest.TestCase):
    """风险控制引擎测试"""

    def setUp(self):
        self.engine = RiskEngine()

        # 配置基础验证规则
        pre_check = self.engine.validators['pre_check']
        pre_check.required_fields = ['trade_id', 'symbol', 'amount']
        pre_check.type_checks = {
            'trade_id': str,
            'amount': (int, float)
        }
        pre_check.range_checks = {
            'amount': (0, 10_000_000)
        }

        # 添加业务规则
        self.engine.add_business_rule(
            'market_rules',
            MarketRuleValidator()
        )

        # 添加风险检查
        self.engine.add_risk_check(
            liquidity_risk_check
        )

    def test_pre_check_pass(self):
        """测试预验证通过"""
        data = {
            'trade_id': 'TRADE123',
            'symbol': '600000',
            'amount': 1000
        }
        result = self.engine.validate(data)
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['risk_level'], 'SAFE')

    def test_pre_check_fail(self):
        """测试预验证失败"""
        data = {
            'trade_id': 123,  # 类型错误
            'amount': -100    # 范围错误
        }
        result = self.engine.validate(data)
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['risk_level'], 'BLOCKED')
        self.assertEqual(len(result['errors']), 3)  # 缺少symbol + 两个错误

    def test_market_rules(self):
        """测试市场规则"""
        data = {
            'trade_id': 'TRADE123',
            'symbol': '600000',
            'amount': 1000,
            'trade_time': time(8, 30)  # 非交易时间
        }
        result = self.engine.validate(data)
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['risk_level'], 'BLOCKED')

    def test_risk_check(self):
        """测试风险检查"""
        data = {
            'trade_id': 'TRADE123',
            'symbol': '600000',
            'amount': 2_000_000  # 触发大额检查
        }
        result = self.engine.validate(data)
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['risk_level'], 'DANGER')

    def test_audit_log(self):
        """测试审计日志"""
        test_data = {
            'trade_id': 'TRADE123',
            'symbol': '600000',
            'amount': 1000
        }

        # 首次验证
        self.engine.validate(test_data)
        logs = self.engine.get_audit_log()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['decision'], 'APPROVED')

        # 失败验证
        test_data['amount'] = -100
        self.engine.validate(test_data)
        logs = self.engine.get_audit_log()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[1]['decision'], 'REJECTED')

class TestMarketRuleValidator(unittest.TestCase):
    """市场规则验证器测试"""

    def setUp(self):
        self.validator = MarketRuleValidator()

    def test_trade_hours(self):
        """测试交易时间验证"""
        data = {
            'trade_time': time(15, 30)  # 合法时间
        }
        result = self.validator.validate(data)
        self.assertTrue(result.is_valid)

        data['trade_time'] = time(8, 30)  # 非交易时间
        result = self.validator.validate(data)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.risk_level, RiskLevel.BLOCKED)

class TestRiskChecks(unittest.TestCase):
    """风险检查函数测试"""

    def test_liquidity_check(self):
        """测试流动性风险检查"""
        data = {'amount': 500_000}
        result = liquidity_risk_check(data)
        self.assertTrue(result.is_valid)

        data['amount'] = 1_500_000
        result = liquidity_risk_check(data)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.risk_level, RiskLevel.DANGER)

if __name__ == '__main__':
    unittest.main()
