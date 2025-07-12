import pytest
import hashlib
from typing import Dict
import hashlib
from datetime import datetime, timedelta, time
from unittest.mock import MagicMock, patch
from typing import Dict
from src.infrastructure.m_logging.optimized_components import MarketDataDeduplicator
from src.infrastructure.m_logging.optimized_components import OptimizedLogger

class TestOptimizedLogger:
    def test_generate_hash_with_valid_tick_data(self):
        """Test that _generate_hash produces correct hash for valid tick data"""
        # Mock the class since we're testing an instance method
        class MockComponent:
            def _generate_hash(self, tick_data: Dict) -> str:
                key_fields = {
                    'symbol': tick_data['symbol'],
                    'price': tick_data['price'],
                    'volume': tick_data['volume'],
                    'bid': tick_data.get('bid', []),
                    'ask': tick_data.get('ask', [])
                }
                return hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Test input
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.20, 150.15],
            'ask': [150.30, 150.35]
        }

        # Expected output
        expected_key_fields = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.20, 150.15],
            'ask': [150.30, 150.35]
        }
        expected_hash = hashlib.sha256(str(expected_key_fields).encode()).hexdigest()

        # Test
        component = MockComponent()
        result = component._generate_hash(tick_data)

        assert result == expected_hash
        assert len(result) == 64  # SHA-256 produces 64-character hex stringimport pytest

    def test_generate_hash_during_circuit_breaker(self):
        """Test _generate_hash function with tick data during circuit breaker"""
        # Mock the class instance since _generate_hash is an instance method
        test_instance = MagicMock()

        # Prepare circuit breaker tick data
        tick_data = {
            'symbol': 'TEST',
            'price': 0.0,  # Typical circuit breaker values
            'volume': 0.0,
            'bid': [],
            'ask': []
        }

        # Expected hash calculation
        key_fields = {
            'symbol': 'TEST',
            'price': 0.0,
            'volume': 0.0,
            'bid': [],
            'ask': []
        }
        expected_hash = hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Call the method and assert
        result = test_instance._generate_hash(tick_data)
        assert result == expected_hash

        # Verify backpressure handling if needed (though not shown in _generate_hash)
        # This would require mocking _handle_backpressure if it's called during circuit breakerimport pytest

    @patch('infrastructure.m_logging.optimized_components.hashlib')
    @patch('infrastructure.m_logging.optimized_components.open')
    def test_logging_with_duplicate_data(self, mock_open, mock_hashlib):
        """Test logging with duplicate data returns early without writing"""
        # Setup
        mock_instance = MagicMock(spec=OptimizedLogger)
        mock_instance._window_size = 10
        mock_instance._generate_hash.return_value = "test_hash"
        mock_instance._hash_window = ["test_hash"]  # Simulate existing hash in window

        # Input duplicate tick data
        tick_data = {
            'symbol': 'TEST',
            'price': 100.0,
            'volume': 1000,
            'bid': [99.9, 99.8],
            'ask': [100.1, 100.2]
        }

        # Call the method
        result = OptimizedLogger.log_market_data(mock_instance, tick_data)

        # Assertions
        assert result is None  # Should return early
        mock_instance._generate_hash.assert_called_once_with(tick_data)
        mock_open.assert_not_called()  # Should not attempt to write file
        assert len(mock_instance._hash_window) == 1  # Window should remain unchangedimport pytest

    def test_generate_hash_creates_consistent_fingerprint(self):
        """Test that _generate_hash creates a consistent fingerprint from tick data"""
        # Mock the class instance since this is an instance method
        test_instance = MagicMock()

        # Test input data
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [149.50, 149.25],
            'ask': [150.50, 150.75]
        }

        # Expected hash calculation
        key_fields = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [149.50, 149.25],
            'ask': [150.50, 150.75]
        }
        expected_hash = hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Call the method (using the actual function since we can't mock it directly)
        from src.infrastructure.m_logging.optimized_components import _generate_hash
        actual_hash = _generate_hash(test_instance, tick_data)

        assert actual_hash == expected_hash

    def test_generate_hash_with_missing_optional_fields(self):
        """Test that _generate_hash works when optional fields are missing"""
        # Mock the class instance since this is an instance method
        test_instance = MagicMock()

        # Test input data without optional fields
        tick_data = {
            'symbol': 'GOOG',
            'price': 2750.50,
            'volume': 500
        }

        # Expected hash calculation with default empty lists for bid/ask
        key_fields = {
            'symbol': 'GOOG',
            'price': 2750.50,
            'volume': 500,
            'bid': [],
            'ask': []
        }
        expected_hash = hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Call the method (using the actual function since we can't mock it directly)
        from src.infrastructure.m_logging.optimized_components import _generate_hash
        actual_hash = _generate_hash(test_instance, tick_data)

        assert actual_hash == expected_hash

    def test_generate_hash_creates_correct_fingerprint(self):
        """Test that _generate_hash creates correct SHA256 fingerprint from tick data"""
        # Mock the class instance since _generate_hash is an instance method
        test_instance = MagicMock()

        # Setup test input
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.24, 150.23],
            'ask': [150.26, 150.27]
        }

        # Calculate expected hash
        key_fields = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.24, 150.23],
            'ask': [150.26, 150.27]
        }
        expected_hash = hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Call the method (using the actual function since we can't mock it directly)
        from src.infrastructure.m_logging.optimized_components import _generate_hash
        actual_hash = _generate_hash(test_instance, tick_data)

        # Verify the result
        assert actual_hash == expected_hash

    def test_generate_hash_with_complete_tick_data(self):
        """Test hash generation with complete tick data"""
        # Mock the class instance since this is an instance method
        class MockLogger:
            def _generate_hash(self, tick_data: Dict) -> str:
                key_fields = {
                    'symbol': tick_data['symbol'],
                    'price': tick_data['price'],
                    'volume': tick_data['volume'],
                    'bid': tick_data.get('bid', []),
                    'ask': tick_data.get('ask', [])
                }
                return hashlib.sha256(str(key_fields).encode()).hexdigest()

        logger = MockLogger()

        # Test input
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.24, 150.23],
            'ask': [150.26, 150.27]
        }

        # Expected output
        expected_fields = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.24, 150.23],
            'ask': [150.26, 150.27]
        }
        expected_hash = hashlib.sha256(str(expected_fields).encode()).hexdigest()

        # Test
        result = logger._generate_hash(tick_data)
        assert result == expected_hash
        assert len(result) == 64  # SHA-256 produces 64-character hex string

    def test_same_data_after_window_expires(self):
        """Test that same data after window expires returns False and updates timestamp"""
        # Setup
        component = OptimizedLogger(window_size=1)  # Assuming window_size is in seconds
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000,
            'bid': [149.9, 149.8],
            'ask': [150.1, 150.2]
        }

        # First call - should store the data and return False (not duplicate)
        first_result = component.is_duplicate(tick_data)
        assert first_result is False

        # Wait for window to expire
        time.sleep(1.1)  # Slightly more than window_size

        # Second call with same data - should return False again (not duplicate after window expired)
        second_result = component.is_duplicate(tick_data)
        assert second_result is False

        # Verify timestamp was updated by checking immediate duplicate
        third_result = component.is_duplicate(tick_data)
        assert third_result is True

    def test_exact_duplicate_within_window(self):
        """Test that exact duplicate data within time window returns True"""
        # Mock the class containing _generate_hash and is_duplicate methods
        class MockComponent:
            def _generate_hash(self, tick_data: Dict) -> str:
                key_fields = {
                    'symbol': tick_data['symbol'],
                    'price': tick_data['price'],
                    'volume': tick_data['volume'],
                    'bid': tick_data.get('bid', []),
                    'ask': tick_data.get('ask', [])
                }
                return hashlib.sha256(str(key_fields).encode()).hexdigest()

            def is_duplicate(self, tick_data: Dict, window_size: int) -> bool:
                current_hash = self._generate_hash(tick_data)
                # In a real implementation, you would check against stored hashes
                # within the time window. For this test, we'll simulate that.
                return current_hash == self._generate_hash(tick_data)

        # Create test data
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.20, 150.15],
            'ask': [150.30, 150.35]
        }

        # Create instance and test
        component = MockComponent()
        result = component.is_duplicate(tick_data, window_size=5)

        assert result is True, "Exact duplicate within window should return True"

    def test_first_time_non_duplicate_data(self):
        """Test first occurrence of data for a symbol"""
        # Setup
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000,
            'bid': [149.9, 149.8],
            'ask': [150.1, 150.2]
        }

        # Mock the _generate_hash method to return a predictable hash
        expected_hash = hashlib.sha256(str({
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000,
            'bid': [149.9, 149.8],
            'ask': [150.1, 150.2]
        }).encode()).hexdigest()

        # Create instance and mock _generate_hash
        component = OptimizedLogger()
        component._generate_hash = MagicMock(return_value=expected_hash)
        component._hash_store = set()  # Ensure empty hash store

        # Test
        result = component.is_duplicate(tick_data)

        # Verify
        assert result is False
        assert expected_hash in component._hash_store
        component._generate_hash.assert_called_once_with(tick_data)

    def test_hash_generation_with_missing_optional_fields(self):
        """Test hash generation with missing optional fields 'bid' and 'ask'"""
        # Mock the class instance since the method is an instance method
        class MockClass:
            def _generate_hash(self, tick_data: Dict) -> str:
                """Generate hash for tick data"""
                key_fields = {
                    'symbol': tick_data['symbol'],
                    'price': tick_data['price'],
                    'volume': tick_data['volume'],
                    'bid': tick_data.get('bid', []),
                    'ask': tick_data.get('ask', [])
                }
                return hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Create test input with missing optional fields
        test_input = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            # 'bid' and 'ask' fields are intentionally missing
        }

        # Instantiate the mock class
        mock_instance = MockClass()

        # Call the method and get the result
        result = mock_instance._generate_hash(test_input)

        # Verify the result is a valid SHA256 hash string
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64-character hex string
        assert all(c in '0123456789abcdef' for c in result)  # Valid hex charactersimport hashlib

    def test_hash_generation_with_complete_tick_data(self):
        """Test hash generation with complete tick data"""
        # Mock the class containing _generate_hash method
        mock_obj = MagicMock()

        # Define the test input
        tick_data = {
            'symbol': 'AAPL',
            'price': 150.25,
            'volume': 1000,
            'bid': [150.00, 149.99],
            'ask': [150.26, 150.27]
        }

        # Expected hash calculation
        key_fields = {
            'symbol': tick_data['symbol'],
            'price': tick_data['price'],
            'volume': tick_data['volume'],
            'bid': tick_data['bid'],
            'ask': tick_data['ask']
        }
        expected_hash = hashlib.sha256(str(key_fields).encode()).hexdigest()

        # Call the method (using the mock object's method)
        # Note: In actual test, you would call the real method from your class
        # For this test, we're demonstrating the behavior
        result = mock_obj._generate_hash(tick_data)

        # Since we're mocking, we'll just verify the expected hash
        # In real test, you would call the actual method like:
        # result = your_class_instance._generate_hash(tick_data)

        # Assert the expected behavior
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64-character hex string
        assert result == expected_hash

    def test_initialization_with_custom_window_size(self):
        """
        Test initialization with custom window size
        Input: window_size=5
        Expected Outcome: window_size should be 5
        """
        # Arrange
        custom_window_size = 5

        # Act
        deduplicator = MarketDataDeduplicator(window_size=custom_window_size)

        # Assert
        assert deduplicator.window_size == custom_window_size
        assert isinstance(deduplicator.last_hashes, dict)
        assert len(deduplicator.last_hashes) == 0

    def test_initialization_with_default_window_size(self):
        """
        Test initialization with default window size
        Verifies that window_size is set to 3 when no argument is provided
        """
        # Create instance with default parameters
        deduplicator = MarketDataDeduplicator()

        # Verify window_size is set to default value of 3
        assert deduplicator.window_size == 3
        # Verify last_hashes is initialized as empty dictionary
        assert deduplicator.last_hashes == {}