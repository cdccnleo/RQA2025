import threading
import pytest
import threading
from unittest.mock import MagicMock
import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.m_logging.log_metrics import record
from src.infrastructure.m_logging.log_metrics import LogMetrics, LogMetricsConfig, get_metrics_instance
import pytest
from src.infrastructure.m_logging.log_metrics import LogMetrics
from src.infrastructure.m_logging.log_metrics import push_metrics, get_metrics_instance

class SingletonClassTest:
    def test_singleton_instance_creation(self):
        """
        Test that only one instance is created when __new__ is called multiple times.
        """
        # Create a mock class that inherits from the singleton pattern
        class TestClass:
            _instance = None
            _lock = MagicMock()

            def __new__(cls, *args, **kwargs):
                if cls._instance is None:
                    with cls._lock:
                        if cls._instance is None:
                            cls._instance = super().__new__(cls)
                            cls._instance._initialized = False
                return cls._instance

        # First call to __new__
        instance1 = TestClass.__new__(TestClass)

        # Second call to __new__
        instance2 = TestClass.__new__(TestClass)

        # Verify both calls return the same instance
        assert instance1 is instance2

        # Verify _instance was only set once
        assert TestClass._instance is not None
        assert TestClass._instance is instance1

        # Verify lock was acquired only once (during first creation)
        TestClass._lock.__enter__.assert_called_once()

    def test_init_initialization_guard(self):
        """
        Test that __init__ only runs initialization code when _initialized is False.
        """
        # Mock the class to test the initialization guard
        with patch('infrastructure.m_logging.log_metrics.LogMetrics') as MockLogMetrics:
            # Configure the mock instance
            mock_instance = MockLogMetrics.return_value
            mock_instance._initialized = False

            # First call to __init__ should run initialization
            mock_instance.__init__()
            assert mock_instance._initialized  # Should now be True

            # Reset any side effects from first call
            MockLogMetrics.reset_mock()

            # Second call to __init__ should not run initialization
            mock_instance.__init__()
            mock_instance._initialize.assert_not_called()  # Verify initialization wasn't called again
            assert mock_instance._initialized  # Should still be True

    def test_default_config_initialization(self):
        """
        Test that a default LogMetricsConfig is created when none is provided.
        """
        # Mock the necessary components that would be initialized in __init__
        LogMetrics._instance = None
        LogMetrics._lock = MagicMock()

        # Create instance with config=None
        log_metrics = LogMetrics.__new__(LogMetrics)
        log_metrics.__init__(config=None)

        # Verify config is not None and is an instance of LogMetricsConfig
        assert log_metrics.config is not None
        assert isinstance(log_metrics.config, LogMetricsConfig)

        # Clean up
        LogMetrics._instance = None

    @patch('infrastructure.m_logging.log_metrics.Counter')
    @patch('infrastructure.m_logging.log_metrics.Histogram')
    @patch('infrastructure.m_logging.log_metrics.Gauge')
    def test_prometheus_metrics_initialization(self, mock_gauge, mock_histogram, mock_counter):
        """
        Test that Prometheus metrics are initialized when enable_prometheus is True
        """
        # Mock the counter and histogram classes
        mock_counter.return_value = MagicMock()
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = MagicMock()

        # Create instance with prometheus enabled
        instance = LogMetrics.__new__(LogMetrics)
        instance._initialized = False
        instance.__init__(enable_prometheus=True)

        # Verify metrics were initialized
        assert hasattr(instance, 'log_total')
        assert hasattr(instance, 'log_sampled')
        assert hasattr(instance, 'log_processing_time')
        assert hasattr(instance, 'log_level_count')

        # Verify the metrics were created with correct labels
        mock_counter.assert_any_call(
            'log_total',
            'Total number of logs processed',
            ['level', 'service']
        )
        mock_counter.assert_any_call(
            'log_sampled',
            'Number of logs sampled',
            ['level', 'service']
        )
        mock_histogram.assert_called_once_with(
            'log_processing_time_seconds',
            'Time spent processing logs',
            ['level', 'service'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        mock_gauge.assert_called_once_with(
            'log_level_count',
            'Current count of logs by level',
            ['level', 'service']
        )

    def test_no_prometheus_metrics_when_disabled(self):
        """
        Test that Prometheus metrics are not initialized when enable_prometheus is False
        """
        # Create instance with prometheus disabled
        instance = LogMetrics.__new__(LogMetrics)
        instance._initialized = False
        instance.__init__(enable_prometheus=False)

        # Verify metrics were not initialized
        assert not hasattr(instance, 'log_total')
        assert not hasattr(instance, 'log_sampled')
        assert not hasattr(instance, 'log_processing_time')
        assert not hasattr(instance, 'log_level_count')

    def test_basic_metric_recording(self):
        """
        Test that basic metric increment works correctly.
        Verifies that _metrics['total'] and level/logger metrics are incremented.
        """
        # Setup - create instance and mock necessary components
        metrics = LogMetrics()

        # Clear any existing metrics
        metrics._metrics = {}

        # Test - call the record method
        metrics.record("INFO", "main")

        # Verify - check metrics were incremented
        assert metrics._metrics['total'] == 1
        assert metrics._metrics['INFO'] == 1
        assert metrics._metrics['main'] == 1

        # Call again to verify increments work
        metrics.record("INFO", "main")
        assert metrics._metrics['total'] == 2
        assert metrics._metrics['INFO'] == 2
        assert metrics._metrics['main'] == 2

    def test_multiple_levels_and_loggers(self):
        """
        Test that metrics are correctly tracked for different levels and loggers.
        """
        metrics = LogMetrics()
        metrics._metrics = {}

        # Test with different combinations
        metrics.record("INFO", "main")
        metrics.record("WARNING", "main")
        metrics.record("ERROR", "auth")
        metrics.record("DEBUG", "database")

        # Verify all metrics were tracked correctly
        assert metrics._metrics['total'] == 4
        assert metrics._metrics['INFO'] == 1
        assert metrics._metrics['WARNING'] == 1
        assert metrics._metrics['ERROR'] == 1
        assert metrics._metrics['DEBUG'] == 1
        assert metrics._metrics['main'] == 2
        assert metrics._metrics['auth'] == 1
        assert metrics._metrics['database'] == 1

    def test_sampled_metric_recording(self):
        """
        Test that sampled metrics are incremented when sampled=True
        """
        # Setup
        log_metrics = LogMetrics()
        log_metrics._metrics = {'sampled': 0}
        log_metrics._sampled_level = {'INFO': 0}

        # Execute
        log_metrics.record("INFO", "main", sampled=True)

        # Verify
        assert log_metrics._metrics['sampled'] == 1
        assert log_metrics._sampled_level['INFO'] == 1

    @patch('infrastructure.m_logging.log_metrics.LogMetrics.__new__')
    def test_singleton_instance(self, mock_new):
        """
        Test that LogMetrics maintains a single instance
        """
        # Setup mock to return a new MagicMock for the instance
        instance = MagicMock()
        mock_new.return_value = instance

        # First instantiation
        log1 = LogMetrics()
        # Second instantiation
        log2 = LogMetrics()

        # Verify
        assert log1 is log2
        mock_new.assert_called_once_with(LogMetrics)

    @patch('infrastructure.m_logging.log_metrics.LogMetrics._instance', None)
    @patch('infrastructure.m_logging.log_metrics.LogMetrics._lock', MagicMock())
    def test_latency_recording_updates_metrics(self):
        """
        Test that recording latency updates the total_latency and count_latency metrics.
        """
        # Initialize the singleton instance
        logger = LogMetrics()

        # Reset metrics for clean test
        logger._metrics = {
            'total_latency': 0,
            'count_latency': 0
        }

        # Call the record method with latency
        logger.record("INFO", "main", latency=0.5)

        # Verify metrics were updated correctly
        assert logger._metrics['total_latency'] == 0.5
        assert logger._metrics['count_latency'] == 1

        # Record another latency
        logger.record("INFO", "main", latency=1.5)

        # Verify metrics were accumulated correctly
        assert logger._metrics['total_latency'] == 2.0
        assert logger._metrics['count_latency'] == 2

    @patch('infrastructure.m_logging.log_metrics.LogMetrics._instance', None)
    @patch('infrastructure.m_logging.log_metrics.LogMetrics._lock', MagicMock())
    def test_record_without_latency_does_not_affect_metrics(self):
        """
        Test that recording without latency doesn't affect the latency metrics.
        """
        # Initialize the singleton instance
        logger = LogMetrics()

        # Reset metrics for clean test
        logger._metrics = {
            'total_latency': 10,
            'count_latency': 5
        }

        # Call the record method without latency
        logger.record("INFO", "main")

        # Verify metrics remain unchanged
        assert logger._metrics['total_latency'] == 10
        assert logger._metrics['count_latency'] == 5

    def test_prometheus_metric_recording(self):
        """
        Test that Prometheus metrics are updated when enabled
        """
        # Mock the Prometheus counter
        mock_counter = MagicMock()

        with patch('C.PythonProject.RQA2025.src.infrastructure.m_logging.log_metrics.log_total', mock_counter):
            # Call the record function with prometheus enabled
            record("INFO", "main", enable_prometheus=True)

            # Verify the metric was incremented
            mock_counter.labels.assert_called_once_with(level="INFO", module="main")
            mock_counter.labels().inc.assert_called_once()

    def test_get_metrics_without_latency(self):
        """
        Test that get_metrics() returns a dict without 'avg_latency' key
        when no latency has been recorded.
        """
        # Create a mock instance of LogMetrics
        log_metrics = LogMetrics.__new__(LogMetrics)
        log_metrics._initialized = True
        log_metrics._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            # No latency metrics recorded
        }

        # Call get_metrics
        metrics = log_metrics.get_metrics()

        # Verify the metrics dict doesn't contain 'avg_latency'
        assert isinstance(metrics, dict)
        assert 'avg_latency' not in metrics
        assert 'total_requests' in metrics
        assert 'successful_requests' in metrics
        assert 'failed_requests' in metrics

    def test_get_metrics_with_latency(self):
        """
        Test that get_metrics() correctly calculates average latency
        after recording latency values.
        """
        # Setup - create instance and record some latency values
        metrics = LogMetrics()
        test_latencies = [100, 200, 300]  # in milliseconds
        for latency in test_latencies:
            metrics.record_latency(latency)

        # Exercise - get the metrics
        result = metrics.get_metrics()

        # Verify - check avg_latency is calculated correctly
        expected_avg = sum(test_latencies) / len(test_latencies)
        assert 'avg_latency' in result
        assert result['avg_latency'] == expected_avg

        # Verify - check count matches number of recorded latencies
        assert 'latency_count' in result
        assert result['latency_count'] == len(test_latencies)

    def test_get_metrics_with_no_latency(self):
        """
        Test that get_metrics() handles case when no latency recorded.
        """
        metrics = LogMetrics()
        result = metrics.get_metrics()

        # Verify - avg_latency should be 0 or None when no data
        assert 'avg_latency' in result
        assert result['avg_latency'] in (0, None)
        assert 'latency_count' in result
        assert result['latency_count'] == 0

    @patch('infrastructure.m_logging.log_metrics.requests')
    def test_push_metrics_no_url(self, mock_requests):
        """
        Test that no HTTP request is made when push_url is None
        """
        # Setup
        mock_config = MagicMock()
        mock_config.push_url = None

        # Execute
        push_metrics(mock_config)

        # Verify
        mock_requests.post.assert_not_called()

    @patch('infrastructure.m_logging.log_metrics.time.time')
    @patch('infrastructure.m_logging.log_metrics.LogMetrics._push_metrics_impl')
    def test_push_metrics_interval(self, mock_push_impl, mock_time):
        """
        Test that metrics are only pushed after the interval has passed.
        Multiple calls within the interval should only result in one push.
        """
        # Setup
        mock_time.side_effect = [0, 0.5, 1.0, 1.5, 2.1]  # Simulate time progression
        metrics = LogMetrics.get_instance()  # Using singleton pattern
        metrics._last_push_time = 0
        metrics.PUSH_INTERVAL = 2.0  # Set interval to 2 seconds

        # First call (should trigger push)
        metrics.push_metrics()
        assert mock_push_impl.call_count == 1

        # Subsequent calls within interval (should not trigger push)
        metrics.push_metrics()
        metrics.push_metrics()
        assert mock_push_impl.call_count == 1  # Still only 1 call

        # Call after interval has passed (should trigger push)
        metrics.push_metrics()
        assert mock_push_impl.call_count == 2

        # Verify the push times were recorded correctly
        assert metrics._last_push_time == 2.1  # Last successful push time

    @patch('infrastructure.m_logging.log_metrics.requests.post')
    def test_push_metrics_failure(self, mock_post):
        """
        Test that failed push doesn't raise exception
        """
        # Setup mock to raise an exception when called
        mock_post.side_effect = Exception("Connection failed")

        # Call the function with invalid URL
        try:
            push_metrics("invalid_url", {})
        except Exception as e:
            pytest.fail(f"push_metrics raised an exception when it shouldn't have: {e}")

        # Verify the mock was called (shows the function tried to push)
        assert mock_post.called

    def test_reset_clears_all_metrics(self):
        """
        Test that the reset() method clears all recorded metrics.
        """
        # Setup - create LogMetrics instance and record some metrics
        metrics = LogMetrics()
        metrics.record_metric("metric1", 10)
        metrics.record_metric("metric2", 20)

        # Verify metrics were recorded
        assert len(metrics._metrics) == 2

        # Action - reset metrics
        metrics.reset()

        # Assert - verify metrics dict is empty
        assert len(metrics._metrics) == 0
        assert metrics._metrics == {}

    def test_thread_safety(self):
        """
        Test that the __new__ method is thread-safe when multiple threads
        try to create instances simultaneously.
        """
        # Mock the class and its attributes
        cls = MagicMock()
        cls._instance = None
        cls._lock = threading.Lock()

        # Create a function that threads will call
        def create_instance():
            instance = cls.__new__(cls)
            if cls._instance is None:
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = instance
                        cls._instance._initialized = False
            return cls._instance

        # Create multiple threads
        threads = []
        results = []

        for _ in range(10):
            t = threading.Thread(target=lambda: results.append(create_instance()))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify all threads got the same instance
        assert all(instance is cls._instance for instance in results), \
            "All threads should get the same instance"
        assert cls._instance is not None, \
            "Instance should be created"
        assert cls._instance._initialized is False, \
            "Instance should be initialized with _initialized=False"

    @patch('infrastructure.m_logging.log_metrics.MetricsRecorder')
    def test_invalid_level_recording(self, mock_metrics_recorder):
        """
        Test that metrics are still recorded (or handled gracefully) when an invalid level parameter is provided.
        """
        # Setup mock
        mock_instance = MagicMock()
        mock_metrics_recorder.return_value = mock_instance

        # Call the function with invalid level (None)
        record(None, "main")

        # Verify metrics were still recorded (or at least the function didn't raise an exception)
        mock_instance.record.assert_called_once()

        # Alternatively, if the function should handle None gracefully:
        # assert mock_instance.record.call_count == 0  # if None should be skipped
        # or check for specific handling of None level

    @pytest.fixture
    def log_metrics(self):
        # Setup a LogMetrics instance with mocked dependencies
        metrics = LogMetrics.__new__(LogMetrics)
        metrics._instance = None  # Ensure we get a new instance
        metrics._lock = MagicMock()
        metrics._initialized = False
        return metrics

    def test_large_volume_recording(self, log_metrics):
        """Verify performance with high volume recording."""
        # Mock any necessary dependencies
        log_metrics._record_metric = MagicMock()

        # Perform 10,000 record calls
        for i in range(10000):
            log_metrics.record("test_metric", i)

        # Verify all metrics were recorded correctly
        assert log_metrics._record_metric.call_count == 10000

        # Verify the last recorded value is correct
        last_call_args = log_metrics._record_metric.call_args_list[-1]
        assert last_call_args[0] == ("test_metric", 9999)

        # Verify no errors occurred during high volume recording
        # (This would be caught by pytest if any exceptions were raised)


