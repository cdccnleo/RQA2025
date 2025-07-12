#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监管验收测试框架
用于验证系统是否符合监管要求
"""

import unittest
from datetime import datetime
from typing import Dict, List, Any
from src.infrastructure.utils.logger import get_logger
from src.trading.execution.order_manager import OrderManager
from src.trading.risk.china.risk_controller import ChinaRiskController
from src.infrastructure.compliance.report_generator import ComplianceReportGenerator

logger = get_logger(__name__)

class RegulatoryTestFramework(unittest.TestCase):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化测试框架
        :param config: 配置参数
        """
        super().__init__()
        self.config = config
        self.order_manager = OrderManager(config)
        self.risk_controller = ChinaRiskController(config)
        self.report_generator = ComplianceReportGenerator(config)

    def setUp(self):
        """测试前准备"""
        self.test_start_time = datetime.now()
        logger.info(f"开始监管验收测试: {self.test_start_time}")

    def tearDown(self):
        """测试后清理"""
        test_duration = datetime.datetime.now() - self.test_start_time
        logger.info(f"监管验收测试完成, 耗时: {test_duration.total_seconds():.2f}秒")

    def test_t1_restriction(self):
        """测试T+1限制合规性"""
        # 模拟买入订单
        buy_order = {
            "symbol": "600519.SH",
            "price": 1800.00,
            "quantity": 100,
            "side": "buy",
            "account": "test_account"
        }

        # 执行买入
        self.order_manager.execute(buy_order)

        # 尝试当日卖出
        sell_order = {
            "symbol": "600519.SH",
            "price": 1801.00,
            "quantity": 100,
            "side": "sell",
            "account": "test_account"
        }

        # 验证是否被风控拦截
        check_result = self.risk_controller.check(sell_order)
        self.assertFalse(check_result["passed"], "T+1限制检查失败: 允许当日卖出")

        # 验证拒绝原因
        self.assertEqual(check_result["reason"], "T+1_RESTRICTION",
                        "T+1限制检查失败: 错误拒绝原因")

        logger.info("T+1限制测试通过")

    def test_price_limit(self):
        """测试涨跌停限制合规性"""
        # 获取昨日收盘价
        symbol = "600519.SH"
        last_close = self.order_manager.get_last_close_price(symbol)

        # 计算涨停价(假设10%涨跌幅限制)
        upper_limit = round(last_close * 1.1, 2)
        lower_limit = round(last_close * 0.9, 2)

        # 测试超过涨停价
        buy_order = {
            "symbol": symbol,
            "price": upper_limit + 0.01,
            "quantity": 100,
            "side": "buy"
        }

        check_result = self.risk_controller.check(buy_order)
        self.assertFalse(check_result["passed"], "涨停限制检查失败: 允许超过涨停价买入")
        self.assertEqual(check_result["reason"], "PRICE_LIMIT",
                        "涨停限制检查失败: 错误拒绝原因")

        # 测试低于跌停价
        sell_order = {
            "symbol": symbol,
            "price": lower_limit - 0.01,
            "quantity": 100,
            "side": "sell"
        }

        check_result = self.risk_controller.check(sell_order)
        self.assertFalse(check_result["passed"], "跌停限制检查失败: 允许低于跌停价卖出")
        self.assertEqual(check_result["reason"], "PRICE_LIMIT",
                        "跌停限制检查失败: 错误拒绝原因")

        logger.info("涨跌停限制测试通过")

    def test_star_market_rules(self):
        """测试科创板特殊规则合规性"""
        # 科创板股票
        symbol = "688981.SH"

        # 测试盘后固定价格交易
        after_hours_order = {
            "symbol": symbol,
            "price": 150.00,
            "quantity": 100,
            "side": "buy",
            "time_in_force": "DAY"
        }

        # 设置盘后交易时间
        self.order_manager.set_mock_time("15:15:00")

        check_result = self.risk_controller.check(after_hours_order)
        self.assertTrue(check_result["passed"], "科创板盘后交易检查失败: 拒绝有效盘后交易")

        # 验证价格调整为固定价格
        self.assertEqual(after_hours_order["price"], self.order_manager.get_fixed_price(symbol),
                        "科创板盘后交易检查失败: 未调整到固定价格")

        logger.info("科创板特殊规则测试通过")

    def test_circuit_breaker(self):
        """测试熔断机制合规性"""
        # 模拟触发5%熔断
        self.risk_controller.trigger_circuit_breaker(0.05)

        # 尝试下单
        order = {
            "symbol": "600519.SH",
            "price": 1800.00,
            "quantity": 100,
            "side": "buy"
        }

        # 验证是否被熔断拦截
        check_result = self.risk_controller.check(order)
        self.assertFalse(check_result["passed"], "熔断机制检查失败: 允许熔断期间交易")
        self.assertEqual(check_result["reason"], "CIRCUIT_BREAKER",
                        "熔断机制检查失败: 错误拒绝原因")

        logger.info("熔断机制测试通过")

    def test_compliance_reports(self):
        """测试合规报告完整性"""
        # 生成各类报告
        daily_report = self.report_generator.generate_daily_report()
        weekly_report = self.report_generator.generate_weekly_report()
        monthly_report = self.report_generator.generate_monthly_report()

        # 验证报告结构
        self._validate_report_structure(daily_report)
        self._validate_report_structure(weekly_report)
        self._validate_report_structure(monthly_report)

        # 验证数据完整性
        self.assertGreater(daily_report["total_orders"], 0, "日报数据不完整")
        self.assertGreater(weekly_report["weekly_trades"], 0, "周报数据不完整")
        self.assertGreater(monthly_report["monthly_volume"], 0, "月报数据不完整")

        logger.info("合规报告测试通过")

    def _validate_report_structure(self, report: Dict[str, Any]):
        """验证报告结构完整性"""
        self.assertIn("metadata", report, "报告缺少元数据")
        self.assertIn("title", report, "报告缺少标题")
        self.assertIn("sections", report, "报告缺少章节")
        for section in report["sections"]:
            self.assertIn("name", section, "章节缺少名称")
            self.assertIn("data", section, "章节缺少数据")

    def run_all_tests(self):
        """运行所有监管验收测试"""
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": []
        }

        tests = [
            self.test_t1_restriction,
            self.test_price_limit,
            self.test_star_market_rules,
            self.test_circuit_breaker,
            self.test_compliance_reports
        ]

        for test in tests:
            try:
                test()
                test_results["passed"] += 1
            except AssertionError as e:
                test_results["failed"] += 1
                test_results["errors"].append({
                    "test": test.__name__,
                    "error": str(e)
                })
                logger.error(f"测试失败: {test.__name__} - {str(e)}")
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append({
                    "test": test.__name__,
                    "error": f"意外错误: {str(e)}"
                })
                logger.error(f"测试错误: {test.__name__} - {str(e)}")

        return test_results
