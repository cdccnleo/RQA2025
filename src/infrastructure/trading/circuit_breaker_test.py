import pytest
from datetime import datetime, timedelta
from time import sleep
from .circuit_breaker import InstrumentedCircuitBreaker, CircuitBreakerManager
from unittest.mock import Mock

class TestInstrumentedCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return InstrumentedCircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=1
        )

    def test_initial_state(self, breaker):
        assert breaker.state == breaker.STATE_CLOSED
        assert breaker.failure_threshold == 3

    def test_trip_mechanism(self, breaker):
        # 模拟连续失败
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == breaker.STATE_OPEN

    def test_auto_recovery(self, breaker):
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()  # 触发熔断

        sleep(1.1)  # 等待恢复超时
        breaker.record_success()  # 尝试恢复
        assert breaker.state == breaker.STATE_CLOSED

    def test_trading_hours_impact(self):
        trading_breaker = InstrumentedCircuitBreaker(
            name="trading_breaker",
            failure_threshold=2,
            recovery_timeout=10,
            trading_hours={
                'morning': {'start': '09:30', 'end': '11:30'},
                'afternoon': {'start': '13:00', 'end': '15:00'}
            }
        )

        # 模拟交易时段内触发熔断
        trading_breaker.record_failure()
        trading_breaker.record_failure()
        assert trading_breaker.state == trading_breaker.STATE_OPEN

        # 验证非交易时段才会尝试恢复
        assert trading_breaker._should_attempt_recovery() == False

class TestCircuitBreakerManager:
    def test_breaker_management(self):
        manager = CircuitBreakerManager()

        # 获取不存在的熔断器会自动创建
        breaker1 = manager.get_breaker("api_service")
        assert isinstance(breaker1, InstrumentedCircuitBreaker)

        # 再次获取同一个名称返回相同实例
        breaker2 = manager.get_breaker("api_service")
        assert breaker1 is breaker2

    def test_status_reporting(self):
        manager = CircuitBreakerManager()
        breaker = manager.get_breaker("db_service")

        # 记录失败
        breaker.record_failure(error_type="timeout", instrument="SH600000")

        # 获取状态报告
        status = manager.get_all_status()
        assert "db_service" in status
        assert status["db_service"]["failure_count"] == 1

@pytest.mark.parametrize("concurrency", [10, 50, 100])
def test_thread_safety(concurrency):
    """高并发场景下的线程安全测试"""
    from threading import Thread
    breaker = InstrumentedCircuitBreaker(
        name="concurrent_breaker",
        failure_threshold=1000
    )

    def stress_test():
        for _ in range(100):
            breaker.record_failure()
            breaker.record_success()

    threads = [Thread(target=stress_test) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert breaker.state == breaker.STATE_CLOSED
    assert 0 <= breaker._failure_count <= 100 * concurrency

def test_alert_integration(mocker):
    """测试告警系统集成"""
    mock_alert = mocker.patch(
        'infrastructure.trading.circuit_breaker.InstrumentedCircuitBreaker._send_alert'
    )

    breaker = InstrumentedCircuitBreaker(name="alert_breaker", failure_threshold=2)
    breaker.record_failure()
    breaker.record_failure()  # 触发熔断

    # 验证告警发送
    assert mock_alert.called
    alert_call = mock_alert.call_args[1]
    assert alert_call['action'] == "TRIPPED"
    assert alert_call['details']['failure_count'] == 2
