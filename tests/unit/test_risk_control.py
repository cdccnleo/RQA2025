from src.risk.risk_controller import ChinaRiskController

class TestChinaRiskController:
    def test_price_limit_check(self, china_market_data):
        """测试A股涨跌停检查"""
        controller = ChinaRiskController()
        # 测试涨停价
        assert not controller.check_price_limit(
            china_market_data["symbol"],
            china_market_data["limit_up"] + 0.01
        )
        # 测试跌停价
        assert not controller.check_price_limit(
            china_market_data["symbol"],
            china_market_data["limit_down"] - 0.01
        )
        # 测试正常价格
        assert controller.check_price_limit(
            china_market_data["symbol"],
            china_market_data["price"]
        )

    def test_circuit_breaker(self):
        """测试熔断机制"""
        controller = ChinaRiskController()
        # 模拟5%下跌
        assert controller.check_circuit_breaker(-0.051) == "level_1"
        # 模拟7%下跌
        assert controller.check_circuit_breaker(-0.071) == "level_2"
