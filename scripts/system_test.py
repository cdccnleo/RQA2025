import unittest
import time
import pandas as pd
from src.data.data_manager import DataManager
from src.features.feature_manager import FeatureManager
from src.models.model_manager import ModelManager
from src.trading.live_trading_engine import TradingEngine, CTPService
from src.trading.enhanced_trading_strategy import EnhancedTradingStrategy

class SystemIntegrationTest(unittest.TestCase):
    """系统全链路测试用例"""

    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        cls.data_mgr = DataManager()
        cls.feature_mgr = FeatureManager()
        cls.model_mgr = ModelManager()
        cls.strategy = EnhancedTradingStrategy()

        # 模拟券商接口
        cls.broker = CTPService()
        cls.broker.connect({
            'front': 'tcp://180.168.146.187:10130',
            'broker_id': '9999',
            'investor_id': '000001',
            'password': '123456'
        })

        cls.trading_engine = TradingEngine(cls.broker, cls.strategy)

    def test_data_to_trading_flow(self):
        """测试从数据加载到交易执行全流程"""
        # 1. 数据加载
        test_symbols = ['600519.SH', '000858.SZ']
        data = self.data_mgr.load_data(test_symbols)
        self.assertGreater(len(data), 0, "数据加载失败")

        # 2. 特征生成
        features = self.feature_mgr.generate_features(data)
        self.assertEqual(features.shape[0], len(data), "特征生成数量不匹配")

        # 3. 模型预测
        predictions = self.model_mgr.predict(features)
        self.assertEqual(len(predictions), len(data), "预测结果数量不匹配")

        # 4. 策略信号
        signals = self.strategy.generate_signals(predictions)
        self.assertFalse(signals.empty, "信号生成失败")

        # 5. 交易执行
        start_time = time.time()
        self.trading_engine.execute_signals(signals)
        execution_time = time.time() - start_time

        print(f"\n全链路执行时间: {execution_time:.2f}秒")
        self.assertLess(execution_time, 5.0, "执行超时")

class PerformanceTest(unittest.TestCase):
    """性能压力测试用例"""

    def test_high_frequency_trading(self):
        """高频交易压力测试"""
        from concurrent.futures import ThreadPoolExecutor

        symbols = [f"{i:06d}.SH" for i in range(1, 101)]  # 100只测试股票
        test_cycles = 1000
        success_count = 0

        def single_cycle():
            try:
                data = DataManager().load_data(symbols[:10])
                features = FeatureManager().generate_features(data)
                predictions = ModelManager().predict(features)
                signals = EnhancedTradingStrategy().generate_signals(predictions)
                TradingEngine(CTPService(), EnhancedTradingStrategy()).execute_signals(signals)
                return True
            except:
                return False

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: single_cycle(), range(test_cycles)))

        success_count = sum(results)
        success_rate = success_count / test_cycles * 100

        print(f"\n压力测试成功率: {success_rate:.1f}%")
        self.assertGreaterEqual(success_rate, 95.0, "压力测试失败")

class FailureRecoveryTest(unittest.TestCase):
    """异常恢复测试用例"""

    def test_network_failure_recovery(self):
        """网络中断恢复测试"""
        from unittest.mock import patch

        broker = CTPService()
        broker.connect({
            'front': 'tcp://180.168.146.187:10130',
            'broker_id': '9999',
            'investor_id': '000001',
            'password': '123456'
        })

        with patch.object(broker, 'place_order', side_effect=ConnectionError("Network failure")):
            engine = TradingEngine(broker, EnhancedTradingStrategy())

            # 第一次调用应失败
            with self.assertRaises(ConnectionError):
                engine.execute_signals(pd.DataFrame([{
                    'symbol': '600519.SH',
                    'price': 1800.0,
                    'quantity': 100
                }]))

            # 恢复后应成功
            broker.connected = True
            with patch.object(broker, 'place_order', return_value="CTP_123456"):
                try:
                    engine.execute_signals(pd.DataFrame([{
                        'symbol': '600519.SH',
                        'price': 1800.0,
                        'quantity': 100
                    }]))
                    success = True
                except:
                    success = False

                self.assertTrue(success, "恢复后执行失败")

class SecurityTest(unittest.TestCase):
    """安全合规测试用例"""

    def test_data_encryption(self):
        """敏感数据加密测试"""
        from src.common.crypto_utils import is_encrypted

        test_cases = [
            ('password123', True),
            ('credit_card', True),
            ('stock_data', False),
            ('trade_record', False)
        ]

        for data, should_encrypt in test_cases:
            encrypted = is_encrypted(data.encode())
            if should_encrypt:
                self.assertTrue(encrypted, f"{data} 未加密")
            else:
                self.assertFalse(encrypted, f"{data} 被错误加密")

def generate_deployment_checklist():
    """生成上线检查清单"""
    checklist = [
        ("代码版本", "Git Tag v1.0.0", "确认"),
        ("依赖包", "requirements.txt更新", "确认"),
        ("配置检查", "生产环境配置", "确认"),
        ("数据库", "迁移脚本执行", "确认"),
        ("监控", "告警阈值设置", "确认"),
        ("备份", "灾备方案验证", "确认"),
        ("文档", "操作手册更新", "确认")
    ]

    df = pd.DataFrame(checklist, columns=["检查项", "内容", "状态"])
    df.to_markdown("deployment_checklist.md", index=False)
    print("上线检查清单已生成: deployment_checklist.md")

if __name__ == "__main__":
    # 执行测试用例
    unittest.main(exit=False)

    # 生成上线文档
    generate_deployment_checklist()

    # 输出测试覆盖率报告
    print("\n生成测试覆盖率报告...")
    import os
    os.system("pytest --cov=src --cov-report=html")
    print("覆盖率报告已生成: htmlcov/index.html")
