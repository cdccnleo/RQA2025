import pytest
from trading.risk.china.circuit_breaker import CircuitBreakerChecker
from trading.risk.china.t1_restriction import T1RestrictionChecker
from trading.risk.china.position_limits import STARMarketChecker

@pytest.fixture
def circuit_breaker_checker():
    return CircuitBreakerChecker()

@pytest.fixture
def t1_restriction_checker():
    return T1RestrictionChecker()

@pytest.fixture
def star_market_checker():
    return STARMarketChecker()
