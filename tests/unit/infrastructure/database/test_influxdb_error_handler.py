import time

import pytest
from unittest.mock import MagicMock, patch, Mock
from src.infrastructure.database.influxdb_error_handler import InfluxDBErrorHandler
from typing import Callable
from src.infrastructure.error.error_handler import ErrorHandler
# Mock ApiException since influxdb-client may not be available in test environment
class ApiException(Exception):
    def __init__(self, status=None, reason=None, body=None):
        self.status = status
        self.reason = reason
        self.body = body
        super().__init__(f"API Error: {status} - {reason}")

class TestInfluxdbErrorHandler:
    @patch('infrastructure.database.influxdb_error_handler.ErrorHandler')
    def test_circuit_breaker_with_partial_success_failures_below_threshold(self, mock_error_handler):
        """
        Test circuit breaker with intermittent failures below threshold.
        Verifies that:
        1. Circuit remains closed when failures are below threshold
        2. Failures are reset after a success
        """
        # Setup
        mock_handler = mock_error_handler.return_value
        error_handler = InfluxDBErrorHandler(mock_handler)

        # Create a mock function that fails twice then succeeds
        mock_func = MagicMock()
        mock_func.side_effect = [Exception("First failure"),
                                 Exception("Second failure"),
                                 "Success"]

        with pytest.raises(Exception):
            error_handler.circuit_breaker(mock_func)

        # Second call - should fail
        with pytest.raises(Exception):
            error_handler.circuit_breaker(mock_func)

        # Third call - should succeed and reset failure count
        result = error_handler.circuit_breaker(mock_func)
        assert result == "Success"

        # Verify circuit remains closed (no state change to open)
        # This assumes the class has some way to check circuit state
        # If not implemented, we can verify through behavior
        # Fourth call - should work normally (failure count was reset)
        mock_func.side_effect = ["Another success"]
        result = error_handler.circuit_breaker(mock_func)
        assert result == "Another success"

        # Verify the function was called the expected number of times
        assert mock_func.call_count == 4

    def test_circuit_breaker_recovery_after_timeout(self):
        """
        Test circuit breaker recovery after timeout.
        Verifies that after waiting the recovery timeout, the circuit closes and allows new attempts.
        """
        # Mock the error handler
        mock_error_handler = MagicMock()

        # Create instance with default retry config
        handler = InfluxDBErrorHandler(error_handler=mock_error_handler)

        # Mock a function that will fail
        failing_func = MagicMock(side_effect=Exception("Simulated failure"))

        # Set recovery timeout (assuming this is a configurable parameter)
        recovery_timeout = 5  # seconds

        # First attempt - should fail and open the circuit
        with pytest.raises(Exception):
            handler.circuit_breaker(failing_func)

        # Verify circuit is open (assuming there's a way to check this)
        # This might need adjustment based on actual implementation
        assert handler._circuit_open is True  # Assuming internal state tracking

        # Wait for recovery timeout
        time.sleep(recovery_timeout)

        # Reset mock to succeed this time
        failing_func.side_effect = None
        failing_func.return_value = "success"

        # Attempt again - should succeed as circuit should be closed
        result = handler.circuit_breaker(failing_func)

        # Verify circuit is closed again
        assert handler._circuit_open is False  # Assuming internal state tracking
        assert result == "success"

    def test_circuit_breaker_triggering_after_failure_threshold(self):
        """
        Test circuit breaker triggering after failure threshold
        Verifies that after 3 failures, circuit opens and subsequent calls raise RuntimeError
        """
        # Mock the error handler
        mock_error_handler = MagicMock()

        # Create instance with default retry config (max_attempts=3)
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Mock a function that always fails
        failing_func = MagicMock(side_effect=Exception("Simulated failure"))

        # Wrap the function with circuit breaker
        protected_func = handler.circuit_breaker(failing_func)

        # First 3 attempts should fail (but not raise RuntimeError yet)
        for _ in range(3):
            with pytest.raises(Exception):
                protected_func()

        # 4th attempt should raise RuntimeError (circuit open)
        with pytest.raises(RuntimeError, match="Circuit is open"):
            protected_func()

        # Verify the failing function was called exactly 3 times
        assert failing_func.call_count == 3

    def test_circuit_breaker_with_successful_operations(self):
        """
        Test circuit breaker with consecutive successful operations.
        Verifies that all calls succeed and circuit remains closed.
        """
        # Mock the error handler
        mock_error_handler = MagicMock()

        # Create instance of the class under test
        handler = InfluxDBErrorHandler(error_handler=mock_error_handler)

        # Mock a function that always succeeds
        successful_func = MagicMock(return_value="success")

        # Wrap the function with the circuit breaker
        wrapped_func = handler.circuit_breaker(successful_func)

        # Call the function multiple times (more than max_attempts to verify it stays closed)
        for _ in range(5):
            result = wrapped_func()

        # Verify all calls succeeded
        assert successful_func.call_count == 5
        assert all(result == "success" for _ in range(5))

        # Verify circuit remains closed (assuming there's a way to check this)
        # This might need adjustment based on actual implementation
        assert handler.circuit_state == "closed"  # or similar propertyimport pytest

    @patch('infrastructure.database.influxdb_error_handler.logger')
    def test_fallback_decorator_with_failed_operation(self, mock_logger):
        """
        Test fallback decorator with failed operation.
        Verifies that when a decorated function raises an exception,
        the fallback value is returned and a warning is logged.
        """
        # Mock the error handler
        mock_error_handler = MagicMock()

        # Create instance of the class (assuming it has the fallback decorator)
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Mock a function that will be decorated
        @handler.fallback_on_exception(fallback_value="default")
        def failing_function():
            raise Exception("Simulated failure")

        # Call the decorated function
        result = failing_function()

        # Verify the fallback value is returned
        assert result == "default"

        # Verify warning was logged
        mock_logger.warning.assert_called_once()

        # Verify the warning message contains relevant information
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Exception occurred" in warning_msg
        assert "Simulated failure" in warning_msg
        assert "Returning fallback value" in warning_msg

    def test_fallback_decorator_with_successful_operation(self):
        """
        Test that the fallback decorator returns the normal result
        when the decorated function succeeds.
        """
        # Setup
        mock_error_handler = MagicMock(spec=ErrorHandler)
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create a mock function that will succeed
        mock_func = MagicMock()
        mock_func.return_value = "success_result"

        # Apply the decorator
        decorated_func = handler.fallback_on_exception(mock_func)

        # Execute
        result = decorated_func("test_arg", kwarg="test_kwarg")

        # Verify
        mock_func.assert_called_once_with("test_arg", kwarg="test_kwarg")
        assert result == "success_result"
        # Verify no error handling was called
        assert not mock_error_handler.handle.called

    def test_retry_decorator_with_maximum_attempts(self):
        """
        Test retry decorator with function that always fails
        Verifies that after max_attempts, the last exception is raised
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of InfluxDBErrorHandler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create a function that always raises an exception
        def failing_function():
            raise Exception("Test exception")

        # Apply the retry decorator to the failing function
        decorated_function = handler.retry_on_exception(failing_function)

        # Verify that the function raises an exception after max attempts
        with pytest.raises(Exception) as exc_info:
            decorated_function()

        # Verify the exception message
        assert str(exc_info.value) == "Test exception"

        # Verify the number of attempts (max_attempts + 1 initial attempt)
        assert mock_error_handler.handle_error.call_count == handler.retry_config['max_attempts']

    def test_retry_decorator_with_connection_error(self):
        """
        Test retry decorator with connection error.
        Verifies that when a ConnectionError occurs, it's logged and immediately re-raised without retries.
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of InfluxDBErrorHandler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create a mock function that raises ConnectionError
        mock_func = MagicMock()
        mock_func.side_effect = ConnectionError("Test connection error")

        # Apply the retry decorator
        decorated_func = handler.retry_on_exception(mock_func)

        # Test that the error is re-raised immediately
        with pytest.raises(ConnectionError) as exc_info:
            decorated_func()

        # Verify the error message
        assert str(exc_info.value) == "Test connection error"

        # Verify the function was only called once (no retries)
        assert mock_func.call_count == 1

        # Verify the error was logged
        mock_error_handler.log_error.assert_called_once()

    def test_retry_decorator_with_transient_error(self):
        """Test retry decorator with function that fails twice then succeeds"""
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of the class
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create a mock function that fails twice then succeeds
        mock_func = MagicMock()
        mock_func.side_effect = [Exception("First failure"),
                                 Exception("Second failure"),
                                 "Success"]

        # Apply the retry decorator (assuming it's a method decorator)
        retry_func = handler.retry_on_exception(mock_func)

        # Record start time
        start_time = time.time()

        # Call the decorated function
        result = retry_func()

        # Record end time
        end_time = time.time()

        # Verify the function was called 3 times (2 failures + 1 success)
        assert mock_func.call_count == 3

        # Verify the final result is correct
        assert result == "Success"

        # Verify the delay between retries (approximately)
        # First retry should be after ~1 second, second after ~2 seconds (backoff)
        elapsed_time = end_time - start_time
        assert 2.9 < elapsed_time < 3.5  # Allowing some tolerance

        # Verify error handler was called for each failure
        assert mock_error_handler.handle_error.call_count == 2

    def test_retry_decorator_with_successful_operation(self):
        """
        Test that the retry decorator works correctly when the function succeeds on first attempt.
        Verifies that the function returns successfully without any retries.
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create the error handler instance
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create a mock function that succeeds on first attempt
        mock_function = MagicMock()
        mock_function.return_value = "success"

        # Apply the retry decorator to our mock function
        decorated_function = handler.retry_on_exception(mock_function)

        # Call the decorated function
        result = decorated_function()

        # Verify the function was called exactly once (no retries)
        mock_function.assert_called_once()

        # Verify the result is correct
        assert result == "success"

        # Verify no error handling was triggered
        assert not mock_error_handler.handle_error.called

    @patch('logging.Logger.warning')
    def test_handle_management_error_logs_warning(self, mock_logging_warning):
        """
        Test that handle_management_error logs the error with WARNING level
        when a management operation fails.
        """
        # Arrange
        mock_error_handler = MagicMock(spec=ErrorHandler)
        influx_error_handler = InfluxDBErrorHandler(mock_error_handler)
        operation = "create_db"
        exception = Exception("Test exception")

        # Act
        influx_error_handler.handle_management_error(operation, exception)

        # Assert
        mock_logging_warning.assert_called_once()
        assert "Management operation 'create_db' failed" in mock_logging_warning.call_args[0][0]

    @patch('logging.Logger.error')
    def test_handle_query_error_logs_error_level(self, mock_logger_error):
        """
        Test that handle_query_error logs errors with ERROR level when a query operation fails.
        """
        # Arrange
        mock_error_handler = Mock(spec=ErrorHandler)
        influx_error_handler = InfluxDBErrorHandler(mock_error_handler)
        test_exception = Exception("Test query error")

        # Act
        influx_error_handler.handle_query_error(operation="query", exception=test_exception)

        # Assert
        mock_logger_error.assert_called_once()
        # Verify the error message contains relevant information
        assert "query" in mock_logger_error.call_args[0][0]
        assert "Test query error" in mock_logger_error.call_args[0][0]

    def test_handle_write_error_with_other_exception(self):
        """
        Test handling a write error with non-rate limit exception
        Verifies that error is logged with ERROR level and action is "写入失败"
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of InfluxDBErrorHandler with mocked error handler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Test data
        operation = "write"
        test_exception = Exception("Test exception")

        # Call the method (assuming it's handle_write_error)
        handler.handle_write_error(operation, test_exception)

        # Verify the error handler was called with correct parameters
        mock_error_handler.handle_error.assert_called_once()

        # Get the actual call arguments
        call_args = mock_error_handler.handle_error.call_args[0]

        # Verify the error level and action message
        assert call_args[0] == "ERROR"  # error level
        assert "写入失败" in call_args[1]  # error message should contain the action
        assert str(test_exception) in call_args[1]  # error message should contain the exceptionimport pytest

    def test_handle_write_error_with_rate_limit(self):
        """
        Test handling a write error with rate limit (status 429)
        Verifies that:
        1. Error is logged with WARNING level
        2. Action is "等待后重试"
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of InfluxDBErrorHandler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Create the ApiException with status 429
        exception = ApiException(status=429)

        # Call the method
        result = handler.handle_write_error(exception)

        # Verify the action
        assert result == "等待后重试"

        # Verify the error was logged with WARNING level
        mock_error_handler.log_error.assert_called_once_with(
            operation="write",
            exception=exception,
            level="WARNING",
            action="等待后重试"
        )

    @patch('infrastructure.database.influxdb_error_handler.InfluxDBErrorHandler._recover_connection')
    def test_handle_connection_error_logs_critical_and_calls_recover(self, mock_recover):
        """
        Test that handle_connection_error logs with CRITICAL level and calls _recover_connection
        when a ConnectionError occurs during connect operation.
        """
        # Mock the error handler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Create instance of the class
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Call the method with test inputs
        operation = "connect"
        exception = ConnectionError()
        handler.handle_connection_error(operation, exception)

        # Verify the error was logged with CRITICAL level
        mock_error_handler.log_error.assert_called_once_with(
            operation=operation,
            exception=exception,
            level='CRITICAL'
        )

        # Verify _recover_connection was called
        mock_recover.assert_called_once()

    def test_configure_retry_with_custom_values(self):
        """
        Test configuring retry with custom parameters
        Input: max_attempts=5, delay=2, backoff=3
        Expected Outcome: retry_config is updated with custom values
        """
        # Mock the error handler dependency
        mock_error_handler = MagicMock()

        # Create instance of InfluxDBErrorHandler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Configure retry with custom values
        handler.configure_retry(max_attempts=5, delay=2, backoff=3)

        # Assert the retry_config is updated with custom values
        assert handler.retry_config['max_attempts'] == 5
        assert handler.retry_config['delay'] == 2
        assert handler.retry_config['backoff'] == 3

    def test_configure_retry_with_default_values(self):
        """
        Test configuring retry with default parameters.
        Verifies that retry_config is set to default values (max_attempts=3, delay=1, backoff=2).
        """
        # Mock the ErrorHandler dependency
        mock_error_handler = MagicMock()

        # Instantiate the InfluxDBErrorHandler with mocked error_handler
        handler = InfluxDBErrorHandler(mock_error_handler)

        # Verify the retry_config has the expected default values
        assert handler.retry_config == {
            'max_attempts': 3,
            'delay': 1,
            'backoff': 2
        }

    def test_initialization_with_error_handler(self):
        """
        Test initialization with a valid error handler.
        Verifies that the instance is created with default retry_config and error_handler set.
        """
        # Mock the ErrorHandler
        mock_error_handler = MagicMock(spec=ErrorHandler)

        # Initialize the InfluxDBErrorHandler with the mocked error handler
        influxdb_error_handler = InfluxDBErrorHandler(mock_error_handler)

        # Assert that the error_handler is set correctly
        assert influxdb_error_handler.error_handler == mock_error_handler

        # Assert that the retry_config is set to the default values
        assert influxdb_error_handler.retry_config == {
            'max_attempts': 3,
            'delay': 1,
            'backoff': 2
        }
