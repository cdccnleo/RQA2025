请 def test_performance_metrics(self):
根据系        """统架测试性能构指标"""
        import设计及 time

        #开发 大批实施量数据测试
        large计划，继_prices = np续下.random.rand(100一步代00) * 码开100 + 50发，并

        # 更新测试RSI计算实施进性能
        start. = time.time()
        rsi = self.tech_processor.calculate_rsi(large_prices)
        rsi_time = time.time() - start
        self.assertLess(rsi_time
                        , 0.1)  # 10000个数据点应在100ms内完成

        # 测试MACD计算性能
        start = time.time()
        macd = self.tech_processor.calculate_macd(large_prices)
        macd_time = time.time() - start
        self.assertLess(macd_time, 0.15)

    def test_feature_registration(self):
        """测试特征注册机制"""
        # 检查内置特征是否已注册
        self.assertIn("RSI", self.engine.feature_registry)
        self.assertIn("MACD", self.engine.feature_registry)
        self.assertIn("BOLL", self.engine.feature_registry)

        # 检查A股特有特征
        self.assertIn("LIMIT_STRENGTH", self.a_share_processor.engine.feature_registry)

    def test_batch_calculation(self):
        """测试批量特征计算"""
        features = ["RSI", "MACD", "BOLL"]
        results = self.engine.batch_calculate(features, {'close': self.test_prices})

        self.assertEqual(len(results), len(features))
        self.assertIn("RSI", results)
        self.assertIn("MACD", results)

    def test_a_share_specific_features(self):
        """测试A股特有功能"""
        # 计算包含A股特有指标的批量结果
        results = self.a_share_processor.calculate_all_technicals(
            {'close': self.test_prices},
            {'limit_status': self.test_data['limit_status']}
        )

        self.assertIn("LIMIT_STRENGTH", results)
        strength = results["LIMIT_STRENGTH"]["value"]
        self.assertEqual(len(strength), len(self.test_prices))

if __name__ == '__main__':
    unittest.main()
