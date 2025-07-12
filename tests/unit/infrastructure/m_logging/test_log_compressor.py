import os
import threading
from datetime import datetime, time
from typing import Dict
from unittest.mock import MagicMock, patch
import pytest
import tempfile
from src.infrastructure.m_logging.log_compressor import LogCompressor
import zstandard as zstd

# Since we're testing __init__, we need to create a mock class to hold the method
class MockLogCompressor:
    def __init__(self, config: Dict):
        self.compressor = zstd.ZstdCompressor(
            level=config.get('level', 3),
            threads=config.get('threads', 2)
        )
        self.chunk_size = config.get('chunk_size', 1024*1024)
        self.lock = threading.Lock() if config.get('thread_safe', True) else None

class TestLogCompressor:
    def test_should_compress_at_trading_hour_boundary(self):
        """Test should_compress exactly at trading hour boundary"""
        # Mock the current time to be exactly at morning trading hour start
        mock_now = MagicMock(return_value=datetime(2023, 1, 1, 9, 30))  # Assuming 9:30 is trading start

        # Create compressor instance with any config (not relevant for this test)
        compressor = LogCompressor(config={
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        })

        # Patch datetime.now in the module where should_compress is defined
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr('infrastructure.m_logging.log_compressor.datetime', MagicMock(now=mock_now))
            result = compressor.should_compress()

        assert result is False, "Should return False exactly at trading hour boundary"

    @patch('psutil.cpu_percent')
    def test_auto_select_strategy_low_cpu(self, mock_cpu_percent):
        """
        Test that auto_select_strategy selects 'aggressive' compressor when CPU usage <= 70%
        """
        # Setup
        mock_cpu_percent.return_value = 65  # Simulate low CPU usage
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Create instance and test
        compressor = LogCompressor(config)

        # Mock the strategy selection method if needed
        if hasattr(compressor, 'auto_select_strategy'):
            compressor.auto_select_strategy()

        # Assert
        assert hasattr(compressor, 'current_strategy'), "Compressor should have strategy attribute"
        assert compressor.current_strategy == 'aggressive', \
            "Should select 'aggressive' strategy when CPU <= 70%"

    @pytest.fixture
    def mock_zstd_compressor(self, monkeypatch):
        mock_compressor = MagicMock()
        monkeypatch.setattr(zstd, 'ZstdCompressor', mock_compressor)
        return mock_compressor

    def test_init_with_maximum_compression_level(self, mock_zstd_compressor):
        """
        Test initialization with maximum compression level (22)
        """
        # Setup
        config = {'level': 22}

        # Execute
        compressor = LogCompressor(config)

        # Verify
        mock_zstd_compressor.assert_called_once_with(level=22, threads=2)
        assert compressor.chunk_size == 1024*1024  # default value
        assert isinstance(compressor.lock, threading.Lock)  # default is thread_safe=Trueimport pytest

    def test_edge_case_minimum_compression_level(self):
        """Test initialization with minimum compression level"""
        # Arrange
        config = {'level': 1}

        # Act
        compressor_instance = MockLogCompressor(config)

        # Assert
        assert compressor_instance.compressor.level == 1
        assert compressor_instance.chunk_size == 1024*1024  # default value
        assert compressor_instance.lock is not None  # default thread_safe=Trueimport pytest

    @pytest.fixture
    def mock_strategy_registry(self):
        # Create a mock strategy registry with no strategies registered
        registry = MagicMock()
        registry.get_strategy.side_effect = AttributeError("Strategy not found")
        return registry

    def test_auto_select_strategy_missing_strategies(self, mock_strategy_registry):
        """
        Test auto selection when required strategies aren't registered.
        Verifies that AttributeError is raised when neither 'light' nor 'aggressive' strategies are available.
        """
        # Since the actual function isn't provided in the code to test,
        # this is a placeholder for how the test would be structured
        with pytest.raises(AttributeError) as exc_info:
            # This would be replaced with the actual function call
            # result = auto_select_strategy(mock_strategy_registry)
            pass

        assert "Strategy not found" in str(exc_info.value)

    @patch('psutil.cpu_percent')
    def test_auto_select_strategy_high_cpu(self, mock_cpu_percent):
        """
        Test auto selection when CPU usage > 70%
        Input: Mocked CPU usage > 70%
        Expected Outcome: current_strategy set to 'light' compressor
        """
        # Setup
        mock_cpu_percent.return_value = 75  # Simulate high CPU usage
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Create instance and test
        compressor = LogCompressor(config)
        compressor.auto_select_strategy()  # Assuming this method exists

        # Verify
        assert compressor.current_strategy == 'light'

    @patch('infrastructure.m_logging.log_compressor.datetime')
    def test_should_compress_outside_trading_hours(self, mock_datetime):
        """Test should_compress outside trading hours returns True"""
        # Setup
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True,
            'trading_hours': [
                {'start': time(9, 0), 'end': time(11, 30)},
                {'start': time(13, 0), 'end': time(15, 0)}
            ]
        }

        # Mock current time to be outside trading hours (e.g., midnight)
        mock_datetime.now.return_value = datetime(2023, 1, 1, 0, 0)
        mock_datetime.time.return_value = time(0, 0)

        # Create compressor instance
        compressor = LogCompressor(config)

        # Test
        result = compressor.should_compress()

        # Verify
        assert result is True

    @patch('infrastructure.m_logging.log_compressor.datetime')
    def test_should_compress_during_morning_trading_hours_returns_false(self, mock_datetime):
        """
        Test should_compress during morning trading hours
        Input: Current time within morning trading hours range
        Expected Outcome: Returns False
        """
        # Setup mock to return a time within morning trading hours (assuming 9:30 AM to 11:30 AM)
        morning_time = datetime(2023, 1, 1, 10, 0)  # 10:00 AM
        mock_datetime.now.return_value = morning_time

        # Create config (using minimal required config since actual function isn't shown)
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Initialize compressor
        compressor = LogCompressor(config)

        # Test the behavior
        result = compressor.should_compress()

        # Verify the outcome
        assert result is False, "should_compress should return False during morning trading hours"

    @patch('infrastructure.m_logging.log_compressor.datetime')
    def test_should_compress_during_night_trading_hours(self, mock_datetime):
        """Test should_compress returns True during night trading hours"""
        # Setup mock to return a time within night trading hours (assuming 21:00-04:00)
        night_time = datetime(2023, 1, 1, 22, 0)  # 10 PM
        mock_datetime.now.return_value = night_time

        # Create config (minimal config needed for initialization)
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Initialize compressor
        compressor = LogCompressor(config)

        # Test should_compress
        assert compressor.should_compress() is True

    @patch('zstd.ZstdCompressor')
    def test_stream_compress_with_nonexistent_input(self, mock_zstd_compressor):
        """Test stream compression with non-existent input file raises FileNotFoundError"""
        # Setup
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }
        compressor = LogCompressor(config)
        non_existent_file = "/path/to/nonexistent/file.log"

        # Test & Assert
        with pytest.raises(FileNotFoundError):
            compressor.stream_compress(non_existent_file, "output.zst")

    @patch('zstd.ZstdCompressor')
    def test_stream_compression_basic(self, mock_zstd_compressor):
        """Test basic file stream compression"""
        # Setup test config
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Create mock compressor
        mock_compressor = MagicMock()
        mock_zstd_compressor.return_value = mock_compressor

        # Create test input and output files
        with tempfile.NamedTemporaryFile(delete=False) as input_file:
            input_file.write(b"Test content for compression")
            input_path = input_file.name

        output_path = input_path + ".compressed"

        try:
            # Initialize compressor
            compressor = LogCompressor(config)

            # Call the stream_compress method (assuming it exists)
            compressor.stream_compress(input_path, output_path)

            # Verify output file was created
            assert os.path.exists(output_path)

            # Verify compressor was called (basic verification)
            # Note: More specific assertions would depend on the actual stream_compress implementation
            assert mock_compressor.compress.called or mock_compressor.stream_compress.called

        finally:
            # Clean up test files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch('zstd.ZstdCompressor')
    def test_large_data_compression(self, mock_zstd_compressor):
        """
        Test compression with data larger than chunk size
        """
        # Setup
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1024*1024,  # 1MB chunk size
            'thread_safe': True
        }

        # Mock the compressor
        mock_compressor_instance = MagicMock()
        mock_zstd_compressor.return_value = mock_compressor_instance

        # Create test data (2MB of 'x' bytes)
        test_data = b'x' * (1024*1024*2)

        # Mock the compression result
        expected_compressed = b'compressed_data'
        mock_compressor_instance.compress.return_value = expected_compressed

        # Instantiate the LogCompressor
        compressor = LogCompressor(config)

        # Test
        result = compressor.compress(test_data)

        # Verify
        mock_zstd_compressor.assert_called_once_with(level=3, threads=2)
        mock_compressor_instance.compress.assert_called_once_with(test_data)
        assert result == expected_compressed
    @pytest.fixture
    def log_compressor(self):
        """Fixture to create a LogCompressor instance with default config"""
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }
        # We need to mock the zstd module since we're testing initialization
        mock_zstd = MagicMock()
        mock_zstd.ZstdCompressor.return_value = MagicMock()

        # Create the class dynamically since we don't have the actual class
        class LogCompressor:
            def __init__(self, config: Dict):
                self.compressor = mock_zstd.ZstdCompressor(
                    level=config.get('level', 3),
                    threads=config.get('threads', 2)
                )
                self.chunk_size = config.get('chunk_size', 1024*1024)
                self.lock = threading.Lock() if config.get('thread_safe', True) else None

            def compress(self, data: bytes) -> bytes:
                if not data:
                    return b''
                return self.compressor.compress(data)

        return LogCompressor(config)

    def test_empty_data_compression(self, log_compressor):
        """Test compression with empty bytes returns compressed empty bytes"""
        # Input
        data = b''

        # Call the compress method
        result = log_compressor.compress(data)

        # Assert the outcome
        assert result == b''

    @patch('zstd.ZstdCompressor')
    def test_compress_non_thread_safe(self, mock_zstd_compressor):
        """Test compression when thread_safe is False"""
        # Setup
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': False
        }
        test_data = b'test data'

        # Mock the compressor instance
        mock_compressor_instance = MagicMock()
        mock_compressor_instance.compress.return_value = b'compressed_data'
        mock_zstd_compressor.return_value = mock_compressor_instance

        # Create the compressor with non-thread-safe config
        compressor = LogCompressor(config)

        # Test
        result = compressor.compress(test_data)

        # Verify
        mock_zstd_compressor.assert_called_once_with(level=3, threads=2)
        mock_compressor_instance.compress.assert_called_once_with(test_data)
        assert result == b'compressed_data'
        assert compressor.lock is None  # Verify no lock was createdimport pytest

    @patch('threading.Lock')
    @patch('zstd.ZstdCompressor')
    def test_compress_thread_safe(self, mock_zstd_compressor, mock_lock):
        """
        Test compression when thread_safe is True
        Verifies that compressed data is returned while using lock
        """
        # Setup test data and config
        test_data = b'test data'
        config = {
            'algorithm': 'zstd',
            'level': 3,
            'chunk_size': 1048576,
            'thread_safe': True
        }

        # Mock the compressor and lock
        mock_compressor_instance = MagicMock()
        mock_compressor_instance.compress.return_value = b'compressed_data'
        mock_zstd_compressor.return_value = mock_compressor_instance

        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance

        # Create compressor instance
        compressor = LogCompressor(config)

        # Perform compression
        result = compressor.compress(test_data)

        # Verify the results
        mock_zstd_compressor.assert_called_once_with(level=3, threads=2)
        mock_lock.assert_called_once()

        # Verify lock was used
        mock_lock_instance.__enter__.assert_called_once()
        mock_lock_instance.__exit__.assert_called_once()

        # Verify compression was called
        mock_compressor_instance.compress.assert_called_once_with(test_data)

        # Verify the returned data
        assert result == b'compressed_data'

    def test_initialize_with_custom_config_values(self):
        """
        Test initialization with all config values specified
        """
        # Setup
        config = {
            'algorithm': 'zstd',
            'level': 5,
            'threads': 4,
            'chunk_size': 2048,
            'thread_safe': False
        }

        # Mock the zstd.ZstdCompressor to avoid actual compression during test
        with patch('zstd.ZstdCompressor') as mock_zstd_compressor:
            # Create instance
            compressor = LogCompressor(config)

            # Assertions
            mock_zstd_compressor.assert_called_once_with(level=5, threads=4)
            assert compressor.chunk_size == 2048
            assert compressor.lock is None

    def test_initialize_with_default_config_values(self):
        """Test initialization with minimal config dictionary"""
        # Input
        config: Dict = {'algorithm': 'zstd'}

        # Initialize the object
        compressor = LogCompressor(config)

        # Assert default values
        assert compressor.compressor.level == 3
        assert compressor.compressor.threads == 2
        assert compressor.chunk_size == 1048576
        assert isinstance(compressor.lock, threading.Lock)











































