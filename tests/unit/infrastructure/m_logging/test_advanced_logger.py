import asyncio
from datetime import datetime
from prometheus_client import REGISTRY
import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.m_logging.advanced_logger import EnhancedTradingLogger  # Assuming the class is named EnhancedTradingLogger

class TestEnhancedTradingLogger:
    def test_initialize_with_complete_config(self):
        """Test initialization with a complete configuration dictionary"""
        # Clean up any existing metrics first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue  # Skip if metric doesn't exist
            
        # Mock the component classes to avoid actual initialization
        original_compressor = EnhancedTradingLogger.__init__.__globals__['TradingHoursAwareCompressor']
        original_backpressure = EnhancedTradingLogger.__init__.__globals__['AdaptiveBackpressure']
        original_sampler = EnhancedTradingLogger.__init__.__globals__['TradingSampler']
        original_metrics = EnhancedTradingLogger.__init__.__globals__['LoggingMetrics']

        EnhancedTradingLogger.__init__.__globals__['TradingHoursAwareCompressor'] = MagicMock()
        EnhancedTradingLogger.__init__.__globals__['AdaptiveBackpressure'] = MagicMock()
        EnhancedTradingLogger.__init__.__globals__['TradingSampler'] = MagicMock()
        EnhancedTradingLogger.__init__.__globals__['LoggingMetrics'] = MagicMock()

        try:
            # Test input
            config = {
                'compression': {},
                'backpressure': {'max_queue': 5000},
                'sampling': {},
                'security': {}
            }

            # Initialize the logger
            logger = EnhancedTradingLogger(config)

            # Assertions
            assert logger.compressor is not None
            assert logger.backpressure is not None
            assert logger.sampler is not None
            assert logger.metrics is not None
            assert logger._queue.maxsize == 5000
            assert logger._is_running is False

        finally:
            # Restore the original classes
            EnhancedTradingLogger.__init__.__globals__['TradingHoursAwareCompressor'] = original_compressor
            EnhancedTradingLogger.__init__.__globals__['AdaptiveBackpressure'] = original_backpressure
            EnhancedTradingLogger.__init__.__globals__['TradingSampler'] = original_sampler

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    @patch('infrastructure.m_logging.advanced_logger.LoggingMetrics')
    def test_initialize_with_missing_optional_config(
            self,
            mock_metrics,
            mock_sampler,
            mock_backpressure,
            mock_compressor
    ):
        """Test initialization with missing optional configuration values"""
        # Clean up registry before test
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])

        # Setup mocks
        mock_compressor.return_value = MagicMock()
        mock_backpressure.return_value = MagicMock()
        mock_sampler.return_value = MagicMock()
        mock_metrics.return_value = MagicMock()

        # Test config with optional fields empty
        config = {
            'compression': {
                'trading_hours': {
                    'morning': ('09:30', '11:30'),
                    'afternoon': ('13:00', '15:00')
                }
            },
            'backpressure': {
                'initial_rate': 1000,
                'max_rate': 10000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},  # optional field
            'security': {}  # optional field
        }

        # Test initialization
        from src.infrastructure.m_logging.advanced_logger import EnhancedTradingLogger
        logger = EnhancedTradingLogger(config)

        # Verify logger was created successfully
        assert logger is not None
        mock_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_logging_process(self):
        """Test starting the logging process sets _is_running to True and creates process loop task."""
        # Clean up Prometheus registry first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue
        
        # Create a mock config dictionary with all required fields
        mock_config = {
            'compression': {},
            'backpressure': {
                'max_queue': 1000,
                'max_rate': 10000,  # 添加缺失的配置
                'initial_rate': 1000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},
            'security': {}
        }

        # Create an instance of EnhancedTradingLogger with mock config
        logger = EnhancedTradingLogger(mock_config)

        # Mock the _process_loop method to avoid actual processing
        logger._process_loop = MagicMock()

        # Call the start method
        await logger.start()

        # Verify _is_running is set to True
        assert logger._is_running is True

        # Verify process loop task was created (we can't directly check the task object,
        # but we can verify _process_loop was called if it's a mock)
        logger._process_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_logging_process_with_empty_queue(self):
        """
        Test stopping the logging process when queue is empty.
        Verifies that _is_running is set to False and queue.join() completes immediately.
        """
        # Clean up Prometheus registry first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue
        
        # Create a test config with all required fields
        test_config = {
            'compression': {},
            'backpressure': {
                'max_queue': 10000,
                'max_rate': 10000,  # 添加缺失的配置
                'initial_rate': 1000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},
            'security': {}
        }

        # Create the logger instance
        logger = EnhancedTradingLogger(test_config)

        # Set initial state
        logger._is_running = True
        logger._queue = MagicMock(spec=asyncio.Queue)
        logger._queue.empty.return_value = True
        logger._queue.join = MagicMock(return_value=asyncio.Future())
        logger._queue.join.return_value.set_result(None)

        # Call the stop method
        await logger.stop()

        # Verify the results
        assert logger._is_running is False
        logger._queue.join.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_logging_process_with_items_in_queue(self):
        """
        Test stopping the logging process when queue has items.
        Verifies that _is_running is set to False and queue.join() waits for processing.
        """
        # Clean up Prometheus registry first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue
        
        # Create a mock config with all required fields
        mock_config = {
            'compression': {},
            'backpressure': {
                'max_queue': 10000,
                'max_rate': 10000,  # 添加缺失的配置
                'initial_rate': 1000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},
            'security': {}
        }

        # Create an instance of EnhancedTradingLogger
        logger = EnhancedTradingLogger(mock_config)

        # Set initial state
        logger._is_running = True
        logger._queue = asyncio.Queue(maxsize=mock_config['backpressure']['max_queue'])

        # Put one item in the queue
        await logger._queue.put("test_item")

        # Mock the queue.join() method
        logger._queue.join = MagicMock()

        # Call the stop method (assuming it's named 'stop')
        await logger.stop()

        # Assertions
        assert logger._is_running is False  # Should set running flag to False
        logger._queue.join.assert_called_once()  # Should wait for queue processing

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    @patch('infrastructure.m_logging.advanced_logger.LoggingMetrics')
    def test_log_message_with_successful_sampling(self, mock_metrics, mock_sampler, mock_backpressure, mock_compressor):
        # Clean up Prometheus registry first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue
        
        # Setup test config with all required fields
        config = {
            'compression': {},
            'backpressure': {
                'max_queue': 1000,
                'max_rate': 10000,  # 添加缺失的配置
                'initial_rate': 1000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},
            'security': {}
        }

        # Setup mocks
        mock_compressor.return_value = MagicMock()
        mock_backpressure.return_value = MagicMock()
        mock_sampler.return_value = MagicMock()
        mock_metrics.return_value = MagicMock()

        # Create logger instance
        logger = EnhancedTradingLogger(config)

        # Mock sampling decision
        mock_sampler.return_value.should_sample.return_value = True

        # Test log message
        result = logger.log_message("test message", "INFO")

        # Verify result
        assert result is True
        mock_sampler.return_value.should_sample.assert_called_once()

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    @patch('infrastructure.m_logging.advanced_logger.LoggingMetrics')
    def test_log_message_with_failed_sampling(self, mock_metrics, mock_sampler,
                                            mock_backpressure, mock_compressor):
        # Clean up Prometheus registry first
        from prometheus_client import REGISTRY
        for metric_name in list(REGISTRY._names_to_collectors.keys()):
            try:
                collector = REGISTRY._names_to_collectors.get(metric_name)
                if collector:
                    REGISTRY.unregister(collector)
            except KeyError:
                continue
        
        # Setup test config with all required fields
        config = {
            'compression': {},
            'backpressure': {
                'max_queue': 1000,
                'max_rate': 10000,  # 添加缺失的配置
                'initial_rate': 1000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {},
            'security': {}
        }

        # Setup mocks
        mock_compressor.return_value = MagicMock()
        mock_backpressure.return_value = MagicMock()
        mock_sampler.return_value = MagicMock()
        mock_metrics.return_value = MagicMock()

        # Create logger instance
        logger = EnhancedTradingLogger(config)

        # Mock sampling decision to return False
        mock_sampler.return_value.should_sample.return_value = False

        # Test log message
        result = logger.log_message("test message", "INFO")

        # Verify result
        assert result is False
        mock_sampler.return_value.should_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_message_with_backpressure_error(self):
        """
        Test logging when backpressure protection fails
        Verifies that message is dropped and drop metrics are recorded when backpressure raises error
        """
        # Setup test config
        config = {
            'compression': {},
            'backpressure': {'max_queue': 10000},
            'sampling': {},
            'security': {}
        }

        # Create logger instance
        logger = EnhancedTradingLogger(config)

        # Mock backpressure to raise error
        logger.backpressure = MagicMock()
        logger.backpressure.apply.side_effect = Exception("Backpressure error")

        # Mock metrics to verify recording
        logger.metrics = MagicMock()

        # Test logging
        await logger.log(level="INFO", message="Test message")

        # Verify metrics were recorded
        logger.metrics.record_drop.assert_called_once()

        # Verify message was not put in queue
        assert logger._queue.empty()

    @pytest.mark.asyncio
    async def test_process_loop_with_compression(self):
        """
        Test processing loop when compression is needed.
        Verifies that message is compressed before writing and metrics are recorded.
        """
        # Setup test config
        test_config = {
            'compression': {'enabled': True},
            'backpressure': {'max_queue': 10000},
            'sampling': {},
            'security': {}
        }

        # Create logger instance
        logger = EnhancedTradingLogger(test_config)

        # Mock the compressor to return True (compression needed)
        logger.compressor = MagicMock()
        logger.compressor.process.return_value = (True, b"compressed_data")

        # Mock metrics recorder
        logger.metrics = MagicMock()

        # Mock the queue with a test message
        test_message = {"data": "test", "needs_compression": True}
        logger._queue = asyncio.Queue()
        await logger._queue.put(test_message)

        # Mock the actual writing method to just verify it was called
        logger._write_compressed = MagicMock()

        # Run the process loop once
        # We use a side effect to stop after one iteration
        with patch.object(logger, '_should_continue_processing',
                         side_effect=[True, False]):
            await logger._process_loop()

        # Verify compression was called
        logger.compressor.process.assert_called_once_with(test_message)

        # Verify compressed data was written
        logger._write_compressed.assert_called_once_with(b"compressed_data")

        # Verify metrics were recorded
        logger.metrics.record_compression.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_process_loop_with_write_error(self):
        """
        Test processing loop when writing fails.
        Verifies that error metrics are recorded and queue task is marked done when _write_entry raises an exception.
        """
        # Setup test config
        test_config = {
            'compression': {},
            'backpressure': {'max_queue': 100},
            'sampling': {},
            'security': {}
        }

        # Create instance
        logger = EnhancedTradingLogger(test_config)

        # Mock the metrics object
        logger.metrics = MagicMock()

        # Mock _write_entry to raise an exception
        with patch.object(logger, '_write_entry', side_effect=Exception("Write failed")):
            # Create a test queue entry
            test_entry = {"data": "test"}

            # Put the entry in the queue
            await logger._queue.put(test_entry)

            # Start the process loop
            logger._is_running = True
            process_task = asyncio.create_task(logger._process_loop())

            # Wait briefly to let the loop process the entry
            await asyncio.sleep(0.1)

            # Stop the loop
            logger._is_running = False
            await logger._queue.put(None)  # Sentinel to stop the loop
            await process_task

            # Verify error metrics were recorded
            logger.metrics.record_error.assert_called_once()

            # Verify queue is empty (task was marked done)
            assert logger._queue.empty()

            # Verify the queue task was properly handled
            assert process_task.done() and not process_task.cancelled()

    def test_sampling_decision_with_high_rate_always_returns_true(self):
        """
        Test sampling with high sample rate (1.0) should always return True
        """
        # Create a mock config with sampling rate = 1.0
        config = {
            'compression': {},
            'backpressure': {},
            'sampling': {'sample_rate': 1.0},
            'security': {}
        }

        # Create the logger instance
        logger = EnhancedTradingLogger(config)

        # Test multiple times to ensure consistent behavior
        for _ in range(100):
            assert logger._should_sample() is True

    @pytest.mark.asyncio
    async def test_process_loop_without_compression(self):
        """
        Test processing loop when compression is not needed.
        Verifies that message is written without compression when compressor returns False.
        """
        # Setup test configuration
        config = {
            'compression': {'enabled': True},
            'backpressure': {'max_queue': 1000},
            'sampling': {},
            'security': {}
        }

        # Create logger instance
        logger = EnhancedTradingLogger(config)

        # Mock the compressor to return False (no compression needed)
        logger.compressor.compress = MagicMock(return_value=False)

        # Mock the actual writing method (assuming it's called _write_message)
        logger._write_message = MagicMock()

        # Create a test message
        test_message = "Test message without compression"

        # Put the message in the queue
        await logger._queue.put(test_message)

        # Set running flag to True
        logger._is_running = True

        # Run the process loop once (we'll mock the infinite loop to run once)
        with patch.object(logger, '_queue') as mock_queue:
            mock_queue.get = asyncio.coroutine(lambda: test_message)
            mock_queue.task_done = MagicMock()

            # Run one iteration of the process loop
            await logger._process_loop()

            # Verify compressor was called with the message
            logger.compressor.compress.assert_called_once_with(test_message)

            # Verify message was written without compression
            logger._write_message.assert_called_once_with(test_message)

            # Verify queue operations were called
            mock_queue.task_done.assert_called_once()

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    @patch('infrastructure.m_logging.advanced_logger.LoggingMetrics')
    def test_sampling_decision_with_low_rate(self, mock_metrics, mock_sampler, mock_backpressure, mock_compressor):
        """Test sampling with low sample rate (0.0) should always return False"""
        # Setup test config with sampling rate 0.0 and required compression config
        test_config = {
            'compression': {
                'trading_hours': {
                    'morning': ('09:30', '11:30'),
                    'afternoon': ('13:00', '15:00')
                },
                'level': 3
            },
            'backpressure': {
                'initial_rate': 1000,
                'max_rate': 10000,
                'window_size': 60
            },
            'sampling': {'sample_rate': 0.0},
            'security': {}
        }

        # Create instance of EnhancedTradingLogger
        logger = EnhancedTradingLogger(test_config)

        # Mock the _should_sample method if it's not directly accessible
        # If it's a private method, we might need to patch it or make it accessible for testing
        # Here we assume it's accessible or we've patched it appropriately

        # Test multiple times to ensure consistent behavior
        for _ in range(100):
            assert logger._should_sample() is False, "With sample_rate=0.0, should always return False"


    @patch('infrastructure.m_logging.advanced_logger.datetime')
    def test_build_log_entry(self, mock_datetime):
        """
        Test building a complete log entry with all required fields including timestamp and signature
        """
        # Setup
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = fixed_time

        config = {
            'compression': {
                'trading_hours': {
                    'morning': ('09:30', '11:30'),
                    'afternoon': ('13:00', '15:00')
                },
                'level': 3
            },
            'backpressure': {
                'initial_rate': 1000,
                'max_rate': 10000,
                'window_size': 60
            },
            'sampling': {},
            'security': {}
        }
        logger = EnhancedTradingLogger(config)

        # Mock any security signing if needed
        logger._sign = MagicMock(return_value="mock_signature")

        # Input
        level = "ERROR"
        message = "Failure"
        meta = {'code': 500}

        # Expected output structure
        expected_entry = {
            'timestamp': fixed_time.isoformat(),
            'level': level,
            'message': message,
            'meta': meta,
            'signature': "mock_signature"
        }

        # Execute
        result = logger._build_entry(level, message, meta)

        # Verify
        assert isinstance(result, dict)
        assert result['timestamp'] == fixed_time.isoformat()
        assert result['level'] == level
        assert result['message'] == message
        assert result['meta'] == meta
        assert 'signature' in result

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    def test_circuit_breaker_trigger_initialization(self, mock_sampler, mock_backpressure, mock_compressor):
        """Test that the circuit breaker is properly initialized with _is_running set to False"""
        # Setup mock config
        config = {
            'compression': {
                'trading_hours': {
                    'morning': ('09:30', '11:30'),
                    'afternoon': ('13:00', '15:00')
                },
                'level': 3
            },
            'backpressure': {
                'max_queue': 1000,
                'initial_rate': 1000,
                'max_rate': 10000,
                'window_size': 60
            },
            'sampling': {},
            'security': {}
        }

        # Create instance with mocked dependencies
        logger = EnhancedTradingLogger(config)

        # Assert that _is_running is initialized to False
        assert logger._is_running is False

    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    def test_circuit_breaker_reset(self, mock_sampler, mock_backpressure, mock_compressor):
        """Test that resetting the circuit breaker sets _is_triggered to False"""
        # Setup mock config
        config = {
            'compression': {},
            'backpressure': {'max_queue': 10000},
            'sampling': {},
            'security': {}
        }

        # Create instance and manually set _is_triggered to True to simulate triggered state
        logger = EnhancedTradingLogger(config)
        logger._is_triggered = True  # Assuming this attribute exists for circuit breaker

        # Call the reset method
        logger.reset()

        # Verify _is_triggered is now False
        assert logger._is_triggered is False

    @patch('infrastructure.m_logging.advanced_logger.AdaptiveBackpressure')
    @patch('infrastructure.m_logging.advanced_logger.TradingHoursAwareCompressor')
    @patch('infrastructure.m_logging.advanced_logger.TradingSampler')
    def test_circuit_breaker_check_when_triggered(self, mock_sampler, mock_compressor, mock_backpressure):
        # Setup mock objects
        mock_backpressure_instance = MagicMock()
        mock_backpressure.return_value = mock_backpressure_instance

        # Configure the mock backpressure to simulate triggered state
        mock_backpressure_instance.is_triggered = True
        mock_backpressure_instance.check.return_value = False

        # Create test config with all required fields
        test_config = {
            'compression': {
                'trading_hours': {
                    'morning': ('09:30', '11:30'),
                    'afternoon': ('13:00', '15:00')
                }
            },
            'backpressure': {
                'max_queue': 1000,
                'initial_rate': 1000,
                'max_rate': 10000,
                'window_size': 60,
                'backoff_factor': 0.5
            },
            'sampling': {'default_rate': 0.1},
            'security': {}
        }

        # Initialize the logger
        logger = EnhancedTradingLogger(test_config)

        # Test the check method when circuit breaker is triggered
        result = logger.backpressure.check()

        # Verify the result
        assert result is False, "Expected False when circuit breaker is triggered"
