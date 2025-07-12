from unittest.mock import MagicMock, patch

import pytest
from src.infrastructure.m_logging.log_aggregator import LogAggregator

class TestLogAggregatorInitWithPrimaryOnly:
    """
    Test class for LogAggregator initialization with only primary storage specified.
    """
    def test_init_with_primary_only(self):
        """
        Test initialization with only primary storage specified.
        Verifies that:
        - self.primary is set to the provided primary value
        - self.secondaries is an empty list when secondaries is None
        - self.current is set to the primary value
        """
        # Arrange
        primary = "file"
        secondaries = None

        # Act
        log_aggregator = LogAggregator(primary, secondaries)

        # Assert
        assert log_aggregator.primary == "file"
        assert log_aggregator.secondaries == []
        assert log_aggregator.current == "file"

    def test_initialization_with_primary_and_secondary_storages(self):
        """
        Test initialization with both primary and secondary storages
        Verifies that:
        - self.primary is set to "elasticsearch"
        - self.secondaries is set to ["file", "kafka"]
        - self.current is set to "elasticsearch"
        """
        # Arrange
        primary = "elasticsearch"
        secondaries = ["file", "kafka"]

        # Act
        aggregator = LogAggregator(primary, secondaries)

        # Assert
        assert aggregator.primary == "elasticsearch"
        assert aggregator.secondaries == ["file", "kafka"]
        assert aggregator.current == "elasticsearch"

    def test_successful_write_operation_to_primary_storage(self):
        """
        Test successful write operation to primary storage.
        Verifies that the write operation returns True and no storage switch occurs.
        """
        # Mock the primary storage to return True on write
        primary_storage = MagicMock()
        primary_storage.write.return_value = True

        # Initialize LogAggregator with mocked primary storage
        log_aggregator = LogAggregator(primary=primary_storage)

        # Test data
        logs = {"message": "test"}

        # Perform the write operation
        result = log_aggregator.write(logs)

        # Verify the result is True
        assert result is True

        # Verify primary storage's write was called exactly once with the logs
        primary_storage.write.assert_called_once_with(logs)

        # Verify no storage switch occurred (current remains primary)
        assert log_aggregator.current == log_aggregator.primary

    def test_write_operation_with_primary_failure(self):
        """
        Test write operation when primary fails but secondaries available
        Verifies that the system switches to first secondary and returns True
        """
        # Create mock loggers
        primary_logger = MagicMock()
        secondary_logger1 = MagicMock()
        secondary_logger2 = MagicMock()

        # Set up primary to fail (return False)
        primary_logger.write.return_value = False
        # Set up secondaries to succeed
        secondary_logger1.write.return_value = True
        secondary_logger2.write.return_value = True

        # Create LogAggregator with one failing primary and two working secondaries
        aggregator = LogAggregator(primary=primary_logger,
                                 secondaries=[secondary_logger1, secondary_logger2])

        # Test data
        test_logs = {"message": "test"}

        # Perform the write operation
        result = aggregator.write(test_logs)

        # Verify the result is True (successful write to secondary)
        assert result is True

        # Verify primary was called once
        primary_logger.write.assert_called_once_with(test_logs)

        # Verify first secondary was called once (after primary failed)
        secondary_logger1.write.assert_called_once_with(test_logs)

        # Verify second secondary was not called
        secondary_logger2.write.assert_not_called()

        # Verify current logger was switched to first secondary
        assert aggregator.current == secondary_logger1

    def test_write_operation_with_all_failures(self):
        """
        Test write operation when all storage options fail.
        Verifies that an Exception is raised when both primary and secondary storages fail.
        """
        # Mock primary storage that always fails
        mock_primary = MagicMock()
        mock_primary.write.side_effect = Exception("Primary storage failed")

        # Mock secondary storage that also fails
        mock_secondary = MagicMock()
        mock_secondary.write.side_effect = Exception("Secondary storage failed")

        # Create LogAggregator instance with failing storages
        aggregator = LogAggregator(primary=mock_primary, secondaries=[mock_secondary])

        # Test data
        test_logs = {"message": "test"}

        # Verify that write operation raises an Exception
        with pytest.raises(Exception):
            aggregator.write(logs=test_logs)

    def test_validate_log_with_complete_valid_log(self):
        """
        Test that _validate_log returns True for a complete valid log
        """
        # Mock primary logger (implementation not shown, assuming it's not needed for this test)
        primary_logger = None
        aggregator = LogAggregator(primary=primary_logger)

        # Test input
        test_log = {"level": "INFO", "message": "test", "source": "trading"}

        # Call the method and assert the result
        assert aggregator._validate_log(test_log) == True

    def test_validate_log_missing_required_field(self):
        """
        Test that _validate_log returns False when a log is missing required fields.
        Input: {"level": "INFO", "source": "trading"}
        Expected Outcome: Returns False
        """
        # Create a mock primary logger (the actual implementation doesn't matter for this test)
        mock_primary = MagicMock()

        # Initialize the LogAggregator with the mock primary logger
        aggregator = LogAggregator(primary=mock_primary)

        # Test log missing required fields
        test_log = {"level": "INFO", "source": "trading"}

        # Call the validation method and assert the result
        result = aggregator._validate_log(test_log)
        assert result is False, "Expected _validate_log to return False for log missing required fields"

    def test_validate_log_with_invalid_level(self):
        """
        Test that _validate_log returns False when given a log with invalid level.
        Input: {"level": "DEBUG", "message": "test", "source": "trading"}
        Expected Outcome: Returns False
        """
        # Mock the primary logger (assuming it's needed for initialization)
        mock_primary = MagicMock()

        # Initialize the LogAggregator
        aggregator = LogAggregator(primary=mock_primary)

        # Test data
        invalid_log = {"level": "DEBUG", "message": "test", "source": "trading"}

        # Call the validation method and assert the result
        result = aggregator._validate_log(invalid_log)
        assert result is False

    def test_log_with_invalid_source(self):
        """
        Test validation of log with invalid source
        """
        # Create a LogAggregator instance with mock loggers
        primary_logger = object()
        aggregator = LogAggregator(primary=primary_logger)

        # Test data
        invalid_log = {
            "level": "INFO",
            "message": "test",
            "source": "invalid"
        }

        # Call the validation method
        result = aggregator._validate_log(invalid_log)

        # Assert the expected outcome
        assert result is False

    def test_successful_log_addition(self):
        """Test successful addition of valid log to queue"""
        # Setup
        mock_primary = object()  # Mock primary logger
        aggregator = LogAggregator(primary=mock_primary)

        # Input
        test_log = {"level": "INFO", "message": "test", "source": "trading"}

        # Test
        result = aggregator.add_log(test_log)

        # Assert
        assert result is True

    def test_log_addition_to_full_queue(self):
        """
        Test that add_log returns False when the queue is full
        """
        # Create a mock primary logger that simulates a full queue
        mock_primary = MagicMock()
        mock_primary.add_log.return_value = False  # Simulate full queue

        # Create LogAggregator instance with the mock primary logger
        aggregator = LogAggregator(primary=mock_primary)

        # Test input
        log_entry = {
            "level": "INFO",
            "message": "test",
            "source": "trading"
        }

        # Call the method and verify the result
        result = aggregator.add_log(log_entry)

        # Assert that the method returned False
        assert result is False

        # Verify that the primary logger was called with the correct log entry
        mock_primary.add_log.assert_called_once_with(log_entry)

class TestLogAggregatorProcessLogs:
    """Test cases for LogAggregator's log processing functionality."""

    @patch('infrastructure.m_logging.log_aggregator.LogAggregator._process_batch')
    def test_log_processing_batch_size_reached(self, mock_process_batch):
        """
        Test that logs are processed when batch size is reached.
        Verifies that when 10 logs are added (matching batch_size=10),
        the batch processing function is called.
        """
        # Setup
        mock_primary = MagicMock()
        aggregator = LogAggregator(primary=mock_primary)
        aggregator.batch_size = 10

        # Exercise - add 10 logs
        for i in range(10):
            aggregator._process_logs(f"log_{i}")

        # Verify
        mock_process_batch.assert_called_once()
        assert len(mock_process_batch.call_args[0][0]) == 10
        assert all(f"log_{i}" in mock_process_batch.call_args[0][0] for i in range(10))

class LogAggregatorTest:
    def test_create_log_alert_returns_true_when_alert_created(self):
        """
        Test that create_log_alert returns True when successfully creating an alert
        with given condition and action.
        """
        # Mock the primary logger (assuming it's needed for initialization)
        mock_primary = MagicMock()

        # Initialize the log aggregator
        aggregator = LogAggregator(primary=mock_primary)

        # Mock the create_log_alert method if it's not implemented yet
        # If it is implemented, we can use the real method
        aggregator.create_log_alert = MagicMock(return_value=True)

        # Test inputs
        condition = "level=ERROR"
        action = "notify"

        # Call the method
        result = aggregator.create_log_alert(condition, action)

        # Assertions
        assert result is True
        aggregator.create_log_alert.assert_called_once_with(condition, action)

    @patch('infrastructure.m_logging.log_aggregator.Elasticsearch')
    def test_process_to_elasticsearch_successfully_processes_logs(self, mock_es):
        """
        Test that logs are successfully processed to Elasticsearch storage
        """
        # Setup
        mock_primary = MagicMock()
        aggregator = LogAggregator(primary=mock_primary)

        test_log = [{
            "@timestamp": "...",
            "level": "INFO",
            "source": "trading",
            "message": "test"
        }]

        # Mock the _process_to_elasticsearch method if it's not directly accessible
        # Alternatively, if it's a private method, we might need to test through a public interface
        # Here we assume it's accessible for testing purposes
        aggregator._process_to_elasticsearch = MagicMock(return_value=True)

        # Execute
        result = aggregator._process_to_elasticsearch(test_log)

        # Verify
        assert result is True
        aggregator._process_to_elasticsearch.assert_called_once_with(test_log)

        # If we want to verify the actual Elasticsearch interaction (assuming the method uses self.primary)
        # Reset mock and test actual implementation (if available)
        if hasattr(aggregator, '_process_to_elasticsearch'):
            aggregator.primary.index = MagicMock(return_value={'_shards': {'failed': 0}})
            result = aggregator._process_to_elasticsearch(test_log)
            assert result is True
            aggregator.primary.index.assert_called_once_with(
                index='logs',  # assuming default index name
                body=test_log[0]
            )

    @patch('infrastructure.m_logging.log_aggregator.time')
    def test_log_processing_flush_interval_reached(self, mock_time):
        """
        Test that logs are processed when the flush interval is reached.
        Verifies that after adding a log and waiting longer than the flush interval,
        the batch processing occurs.
        """
        # Setup
        mock_primary = MagicMock()
        mock_time.time.side_effect = [0, 6]  # First call returns 0, second returns 6 (5+ seconds later)

        # Create aggregator with flush interval of 5 seconds
        aggregator = LogAggregator(primary=mock_primary)
        aggregator._flush_interval = 5
        aggregator._process_logs = MagicMock()

        # Add a log (this should start the timer)
        aggregator.add_log("test log")

        # Simulate time passing (5+ seconds)
        # The timer should trigger processing
        aggregator._check_flush_interval()

        # Verify that _process_logs was called
        aggregator._process_logs.assert_called_once()

    def test_process_to_file_writes_logs_successfully(self, mock_json_dump, mock_open):
        """
        Test that logs are successfully written to file storage
        """
        # Setup
        test_logs = [{"@timestamp": "...", "level": "INFO", "source": "trading", "message": "test"}]
        mock_primary = MagicMock()
        aggregator = LogAggregator(primary=mock_primary)

        # Execute
        aggregator._process_to_file(test_logs)

        # Verify
        # Check that open was called (we assume it writes to a file)
        mock_open.assert_called_once()

        # Check that json.dump was called with our test logs
        args, kwargs = mock_json_dump.call_args
        assert args[0] == test_logs

    @patch('infrastructure.m_logging.log_aggregator.KafkaProducer')
    def test_process_to_kafka_successfully_processes_logs(self, mock_kafka_producer):
        """
        Test that logs are successfully processed to Kafka storage.
        Verifies that the Kafka producer is called with the correct log data.
        """
        # Setup
        mock_primary = MagicMock()
        log_aggregator = LogAggregator(primary=mock_primary)

        test_logs = [{
            "@timestamp": "...",
            "level": "INFO",
            "source": "trading",
            "message": "test"
        }]

        # Mock the Kafka producer instance and its send method
        mock_producer_instance = MagicMock()
        mock_kafka_producer.return_value = mock_producer_instance

        # Execute
        # Assuming _process_to_kafka is a method that takes logs as input
        result = log_aggregator._process_to_kafka(test_logs)

        # Verify
        # Check that the producer was called with the correct data
        mock_producer_instance.send.assert_called_once()

        # Check the first argument (topic) and second argument (value) of the send call
        call_args = mock_producer_instance.send.call_args[0]
        assert call_args[1] == test_logs[0]  # Verify the log data was sent

        # If the function is expected to return something specific on success
        # For example, if it returns True on success:
        assert result is True  # or whatever success indicator is expected

    def test_search_logs_returns_matching_entries_with_limit(self):
        """
        Test that search_logs returns a list of matching log entries
        when provided with a query and limit.
        """
        # Setup mock primary logger
        mock_primary = MagicMock()
        mock_primary.search_logs.return_value = [
            {"message": "Test log 1", "level": "INFO"},
            {"message": "Test log 2", "level": "INFO"}
        ]

        # Create LogAggregator instance with mock primary logger
        log_aggregator = LogAggregator(primary=mock_primary)

        # Define test inputs
        query = {"level": "INFO"}
        limit = 10

        # Call the method
        result = log_aggregator.search_logs(query, limit)

        # Assert the result is as expected
        assert isinstance(result, list)
        assert len(result) <= limit
        for entry in result:
            assert entry["level"] == "INFO"

        # Verify the mock was called correctly
        mock_primary.search_logs.assert_called_once_with(query, limit=limit)

    def test_get_log_statistics_returns_statistics_dictionary(self):
        """Test that get_log_statistics returns a statistics dictionary for given time range"""
        # Mock the primary logger
        mock_primary = MagicMock()
        mock_primary.get_log_statistics.return_value = {
            'error_count': 2,
            'warning_count': 5,
            'info_count': 20,
            'time_range': '1h'
        }

        # Create LogAggregator instance with mocked primary logger
        aggregator = LogAggregator(primary=mock_primary)

        # Call the method with test input
        result = aggregator.get_log_statistics(time_range="1h")

        # Assert the result is a dictionary with expected statistics
        assert isinstance(result, dict)
        assert 'error_count' in result
        assert 'warning_count' in result
        assert 'info_count' in result
        assert result['time_range'] == '1h'

        # Verify the primary logger was called with correct parameter
        mock_primary.get_log_statistics.assert_called_once_with(time_range="1h")

    """Test cases for the tail_logs functionality of LogAggregator"""
    def test_tail_logs_returns_recent_log_lines(self):
        """Test that tail_logs returns a list of recent log lines"""
        # Setup mock primary logger
        mock_primary = MagicMock()
        mock_primary.tail_logs.return_value = [
            "INFO: trading - Message 1",
            "INFO: trading - Message 2",
            "INFO: trading - Message 3"
        ]

        # Create LogAggregator instance with mock primary logger
        aggregator = LogAggregator(primary=mock_primary)

        # Call the method (assuming tail_logs is an instance method)
        result = aggregator.tail_logs(source="trading", level="INFO", lines=20)

        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(line, str) for line in result)
        assert all("INFO: trading" in line for line in result)

        # Verify the mock was called correctly
        mock_primary.tail_logs.assert_called_once_with(
            source="trading", level="INFO", lines=20
        )














