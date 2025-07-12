import pytest
from trading.risk.china.circuit_breaker import CircuitBreakerChecker

class TestCircuitBreaker:
    @pytest.fixture
    def checker(self):
        return CircuitBreakerChecker()

    def test_trigger_circuit_breaker(self, checker):
        # 测试触发5%熔断
        market_data = {
            'index_drop': 0.051,
            'timestamp': '2024-04-15 14:30:00'
        }
        assert checker.check(market_data) == True

    def test_no_trigger(self, checker):
        # 测试未触发熔断
        market_data = {
            'index_drop': 0.049,
            'timestamp': '2024-04-15 14:30:00'
        }
        assert checker.check(market_data) == False

    def test_recovery_period(self, checker):
        # 测试熔断恢复期
        market_data = {
            'index_drop': 0.03,
            'timestamp': '2024-04-15 14:35:00',
            'last_trigger_time': '2024-04-15 14:30:00'
        }
        assert checker.check(market_data) == True
