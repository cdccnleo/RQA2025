import pytest
from src.infrastructure.auto_recovery import AutoRecovery

def test_auto_recovery_initialization():
    recovery = AutoRecovery(max_retries=3)
    assert recovery.max_retries == 3
    assert recovery.current_retries == 0

class TestAutoRecovery:
    def test_initialization(self):
        recovery = AutoRecovery(max_retries=3)
        assert recovery.max_retries == 3
        assert recovery.current_retries == 0

    def test_execute_success(self):
        recovery = AutoRecovery(max_retries=3)
        result = recovery.execute(lambda: "success")
        assert result == "success"
        assert recovery.current_retries == 0

    def test_execute_retry(self):
        recovery = AutoRecovery(max_retries=3)
        counter = 0
        def failing_op():
            nonlocal counter
            counter += 1
            if counter < 2:
                raise Exception("Temporary failure")
            return "success"

        result = recovery.execute(failing_op)
        assert result == "success"
        assert recovery.current_retries == 1
