"""监管接口测试模块"""
import unittest
import logging
from src.compliance.regulatory_compliance import RegulatoryCompliance
from unittest.mock import patch, MagicMock

class TestRegulatoryInterface(unittest.TestCase):
    """监管接口测试类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        cls.compliance = RegulatoryCompliance()

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.logger.info("\n" + "="*50)
        self.logger.info(f"开始测试: {self._testMethodName}")

    def test_report_generation(self):
        """测试报告生成功能"""
        test_cases = [
            ("daily_transaction", "每日交易报告"),
            ("position_holding", "持仓情况报告"),
            ("risk_control", "风控事件报告"),
            ("abnormal_transaction", "异常交易报告")
        ]

        for report_type, report_name in test_cases:
            with self.subTest(report_type=report_type):
                try:
                    report = self.compliance.generate_daily_report(report_type)
                    self.assertIsInstance(report, dict)
                    self.assertEqual(report["report_type"], report_type)
                    self.assertIn("data", report)
                    self.assertIn("signature", report)
                    self.logger.info(f"{report_name}生成测试通过")
                except Exception as e:
                    self.fail(f"{report_name}生成失败: {str(e)}")

    @patch('src.compliance.regulatory_compliance.RegulatoryCompliance._collect_report_data')
    def test_data_collection(self, mock_collect):
        """测试数据收集功能"""
        mock_collect.return_value = {"test": "data"}
        report = self.compliance.generate_daily_report("daily_transaction")
        self.assertEqual(report["data"], {"test": "data"})
        self.assertTrue(mock_collect.called)

    def test_digital_signature(self):
        """测试数字签名功能"""
        test_data = {"key": "value"}
        signature = self.compliance._generate_digital_signature(test_data)
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 64)  # SHA-256哈希长度

    @patch('src.compliance.regulatory_compliance.RegulatoryCompliance.logger')
    def test_report_submission(self, mock_logger):
        """测试报告提交功能"""
        test_report = {
            "report_type": "daily_transaction",
            "data": {"test": "data"},
            "signature": "test_signature"
        }

        # 测试成功提交
        result = self.compliance.submit_to_regulator(test_report)
        self.assertTrue(result)
        self.assertTrue(mock_logger.info.called)

        # 测试提交失败
        with patch.object(self.compliance, 'submit_to_regulator', side_effect=Exception("提交失败")):
            result = self.compliance.submit_to_regulator(test_report)
            self.assertFalse(result)
            self.assertTrue(mock_logger.error.called)

    def test_compliance_validation(self):
        """测试合规性验证功能"""
        test_rules = [
            {"name": "position_limit", "description": "持仓限制检查"},
            {"name": "trade_velocity", "description": "交易频率检查"}
        ]

        results = self.compliance.validate_compliance(test_rules)
        self.assertEqual(len(results), 2)
        for rule in test_rules:
            self.assertIn(rule["name"], results)
            self.assertIn(results[rule["name"]]["status"], ["compliant", "violation", "error"])

    @patch('tests.compliance.test_regulatory_interface.RegulatoryCompliance')
    def test_report_scheduler(self, mock_compliance):
        """测试报告调度器"""
        from src.compliance.regulatory_compliance import ReportScheduler

        # 设置模拟返回值
        mock_instance = mock_compliance.return_value
        mock_instance.required_reports = ["report1", "report2"]
        mock_instance.generate_daily_report.return_value = {"status": "success"}
        mock_instance.submit_to_regulator.return_value = True

        # 测试调度器
        scheduler = ReportScheduler()
        scheduler.compliance = mock_instance
        success_count = scheduler.run_daily_schedule()

        self.assertEqual(success_count, 2)
        self.assertEqual(mock_instance.generate_daily_report.call_count, 2)
        self.assertEqual(mock_instance.submit_to_regulator.call_count, 2)

if __name__ == "__main__":
    unittest.main()
