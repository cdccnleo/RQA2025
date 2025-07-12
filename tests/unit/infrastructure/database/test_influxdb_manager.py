from unittest.mock import MagicMock, patch

import pytest
from src.infrastructure.database.influxdb_manager import InfluxDBManager

class TestInfluxdbManager:
    def test_parse_empty_duration_string_raises_value_error(self):
        """
        Test that parsing an empty duration string raises ValueError
        """
        manager = InfluxDBManager()
        
        with pytest.raises(ValueError):
            manager._parse_duration("")

    @patch('infrastructure.database.influxdb_manager.InfluxDBClient')
    def test_create_continuous_query_with_complex_query_returns_true(self, mock_client):
        """
        Test creating continuous query with complex Flux query returns True
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = MagicMock()
        
        # Complex multi-line Flux query
        complex_query = """
        from(bucket: "example-bucket")
          |> range(start: -1h)
          |> filter(fn: (r) => r._measurement == "example_measurement")
          |> aggregateWindow(every: 1m, fn: mean)
          |> to(bucket: "downsampled", org: "example-org")
        """
        
        # Mock the buckets API response
        manager._buckets_api.create_cq.return_value = True
        
        # Test
        result = manager.create_continuous_query(
            name="complex_cq",
            query=complex_query,
            destination_bucket="downsampled",
            org="example-org"
        )
        
        # Assert
        assert result is True
        manager._buckets_api.create_cq.assert_called_once_with(
            name="complex_cq",
            query=complex_query.strip(),
            destination_bucket="downsampled",
            org="example-org"
        )

    def test_create_retention_policy_with_maximum_duration(self):
        """
        Test creating retention policy with maximum supported duration
        Verifies that the function returns True with correct duration in seconds
        when given a valid bucket and duration="1000w"
        """
        # Setup
        manager = InfluxDBManager()
        
        # Mock the retention API
        manager._retention_api = MagicMock()
        manager._retention_api.create_bucket_retention_policy = MagicMock(return_value=True)
        
        # Test data
        test_bucket = "test_bucket"
        test_duration = "1000w"
        expected_duration_seconds = 1000 * 7 * 24 * 60 * 60  # 1000 weeks in seconds
        
        # Execute
        result = manager.create_retention_policy(bucket=test_bucket, duration=test_duration)
        
        # Verify
        assert result is True
        manager._retention_api.create_bucket_retention_policy.assert_called_once_with(
            bucket=test_bucket,
            duration_seconds=expected_duration_seconds
        )

    @patch('infrastructure.database.influxdb_manager.InfluxDBManager._get_buckets_api')
    @patch('infrastructure.database.influxdb_manager.InfluxDBManager._get_retention_api')
    def test_optimize_for_high_frequency_with_invalid_bucket_name_raises_exception(
            self, mock_get_retention_api, mock_get_buckets_api):
        """
        Test that optimize_for_high_frequency raises an exception when given an invalid bucket name
        """
        # Setup
        manager = InfluxDBManager()
        
        # Configure mock APIs
        mock_buckets_api = MagicMock()
        mock_buckets_api.find_bucket_by_name.side_effect = Exception("Bucket not found")
        mock_get_buckets_api.return_value = mock_buckets_api
        
        mock_retention_api = MagicMock()
        mock_get_retention_api.return_value = mock_retention_api
        
        # Test & Assert
        with pytest.raises(Exception) as exc_info:
            manager.optimize_for_high_frequency("invalid_bucket_name")
        
        assert "Bucket not found" in str(exc_info.value)
        mock_buckets_api.find_bucket_by_name.assert_called_once_with("invalid_bucket_name")

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionApi')
    def test_optimize_for_high_frequency_creates_retention_policies_and_continuous_query_without_errors(self, mock_retention_api, mock_buckets_api):
        """
        Test that optimize_for_high_frequency successfully creates retention policies
        and continuous query without errors when given a valid bucket name.
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api.return_value
        manager._retention_api = mock_retention_api.return_value
        
        # Mock the bucket existence check
        mock_buckets_api.return_value.find_bucket_by_name.return_value = MagicMock()
        
        # Mock successful creation of retention policies and continuous queries
        mock_retention_api.return_value.create_retention_rule.return_value = None
        
        # Test input
        valid_bucket_name = "test_bucket"
        
        # Execute
        try:
            manager.optimize_for_high_frequency(valid_bucket_name)
            # If we get here without exceptions, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"optimize_for_high_frequency raised an exception: {str(e)}")
        
        # Verify that the necessary methods were called
        mock_buckets_api.return_value.find_bucket_by_name.assert_called_once_with(valid_bucket_name)
        assert mock_retention_api.return_value.create_retention_rule.call_count >= 1, \
            "Should create at least one retention rule"

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    def test_get_bucket_id_for_non_existent_bucket_raises_value_error(self, mock_buckets_api):
        """
        Test that _get_bucket_id raises ValueError when bucket doesn't exist
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api.return_value
        
        # Configure mock to return empty list (no buckets found)
        mock_buckets_api.return_value.find_buckets.return_value = []
        
        # Test & Assert
        with pytest.raises(ValueError) as excinfo:
            manager._get_bucket_id("non_existent_bucket")
        
        assert "Bucket 'non_existent_bucket' not found" in str(excinfo.value)

    @patch('src.infrastructure.database.influxdb_manager.InfluxDBManager._buckets_api')
    def test_get_bucket_id_for_existing_bucket(self, mock_buckets_api):
        """
        Test getting ID for existing bucket
        Input: Valid bucket name
        Expected Outcome: Returns bucket ID string
        """
        # Setup
        test_bucket_name = "existing_bucket"
        expected_bucket_id = "12345abcde"
        
        # Mock the buckets API response
        mock_bucket = MagicMock()
        mock_bucket.name = test_bucket_name
        mock_bucket.id = expected_bucket_id
        mock_buckets_api.find_bucket_by_name.return_value = mock_bucket
        
        # Create instance and test
        manager = InfluxDBManager()
        result = manager._get_bucket_id(test_bucket_name)
        
        # Verify
        mock_buckets_api.find_bucket_by_name.assert_called_once_with(test_bucket_name)
        assert result == expected_bucket_id

    def test_parse_duration_with_unknown_unit_returns_default_multiplier(self):
        """
        Test that parsing duration string with unknown unit returns the value
        with default multiplier of 1.
        """
        # Arrange
        manager = InfluxDBManager()
        duration_str = "10x"
        expected_result = 10
        
        # Act
        result = manager._parse_duration(duration_str)
        
        # Assert
        assert result == expected_result

    def test_parse_duration_with_minutes_unit(self):
        """Test parsing duration string with minutes unit"""
        # Arrange
        manager = InfluxDBManager()
        duration_str = "15m"
        expected_result = 900  # 15 * 60 seconds
        
        # Act
        result = manager._parse_duration(duration_str)
        
        # Assert
        assert result == expected_result

    def test_parse_duration_with_seconds_unit(self):
        """
        Test parsing duration string with seconds unit
        Input: "30s"
        Expected Outcome: Returns 30
        """
        # Create an instance of the manager
        manager = InfluxDBManager()
        
        # Test the _parse_duration method with seconds input
        result = manager._parse_duration("30s")
        
        # Assert the expected outcome
        assert result == 30

    def test_create_continuous_query_with_invalid_interval_format(self):
        """
        Test creating continuous query with malformed interval string
        Verifies that the function returns False when given an invalid interval unit ('1y')
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = MagicMock()
        manager._retention_api = MagicMock()
        
        # Test parameters
        bucket_name = "test_bucket"
        destination_bucket = "dest_bucket"
        query = "SELECT * FROM measurements"
        cq_name = "test_cq"
        invalid_interval = "1y"  # Invalid unit
        
        # Execute
        result = manager.create_continuous_query(
            bucket_name=bucket_name,
            destination_bucket=destination_bucket,
            query=query,
            cq_name=cq_name,
            interval=invalid_interval
        )
        
        # Verify
        assert result is False, "Expected False when interval format is invalid"

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionApi')
    def test_create_continuous_query_with_invalid_source_bucket(self, mock_retention_api, mock_buckets_api):
        """
        Test creating continuous query with non-existent source bucket
        Expected to return False due to bucket not found
        """
        # Setup mock for buckets_api
        mock_buckets_instance = MagicMock()
        mock_buckets_instance.find_bucket_by_name.return_value = None  # Simulate bucket not found
        mock_buckets_api.return_value = mock_buckets_instance
        
        # Initialize manager
        manager = InfluxDBManager()
        
        # Test parameters
        source_bucket = "non_existent_bucket"
        destination_bucket = "valid_bucket"
        query_name = "test_query"
        query = "SELECT * FROM measurement"
        
        # Call the method
        result = manager.create_continuous_query(
            source_bucket=source_bucket,
            destination_bucket=destination_bucket,
            query_name=query_name,
            query=query
        )
        
        # Verify the result
        assert result is False
        mock_buckets_instance.find_bucket_by_name.assert_called_once_with(source_bucket)

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.QueryApi')
    def test_create_continuous_query_successfully(self, mock_query_api, mock_buckets_api):
        """
        Test creating continuous query with valid parameters returns True
        """
        # Setup
        manager = InfluxDBManager()
        
        # Mock the query API to return a successful response
        mock_query_instance = MagicMock()
        mock_query_instance.create_query.return_value = MagicMock()
        mock_query_api.return_value = mock_query_instance
        
        # Mock the buckets API to confirm buckets exist
        mock_buckets_instance = MagicMock()
        mock_buckets_instance.find_bucket_by_name.side_effect = [MagicMock(), MagicMock()]
        mock_buckets_api.return_value = mock_buckets_instance
        
        # Test data
        test_name = "test_query"
        source_bucket = "source_bucket"
        target_bucket = "target_bucket"
        test_query = "from(bucket: \"source_bucket\") |> range(start: -1h) |> filter(fn: (r) => r._measurement == \"test\") |> to(bucket: \"target_bucket\")"
        interval = "1h"
        
        # Execute
        result = manager.create_continuous_query(
            name=test_name,
            source_bucket=source_bucket,
            target_bucket=target_bucket,
            query=test_query,
            interval=interval
        )
        
        # Assert
        assert result is True
        mock_buckets_instance.find_bucket_by_name.assert_any_call(source_bucket)
        mock_buckets_instance.find_bucket_by_name.assert_any_call(target_bucket)
        mock_query_instance.create_query.assert_called_once()

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionRulesApi')
    def test_create_retention_policy_without_shard_duration(self, mock_retention_api, mock_buckets_api):
        """
        Test creating retention policy without optional shard duration
        Verifies that the function returns True with basic retention rule when shard_duration is None
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api.return_value
        manager._retention_api = mock_retention_api.return_value
        
        # Mock the bucket object
        mock_bucket = MagicMock()
        mock_bucket.name = "test_bucket"
        
        # Configure the mock retention API to return success
        manager._retention_api.post_buckets_id_retention_rules.return_value = True
        
        # Test
        result = manager.create_retention_policy(
            bucket=mock_bucket,
            duration="1w",
            shard_duration=None
        )
        
        # Verify
        assert result is True
        manager._retention_api.post_buckets_id_retention_rules.assert_called_once()
        
        # Get the actual retention rule that was passed
        called_args = manager._retention_api.post_buckets_id_retention_rules.call_args[1]
        retention_rule = called_args['retention_rule']
        
        # Verify the retention rule has the expected properties
        assert retention_rule['everySeconds'] == 604800  # 1 week in seconds
        assert 'shardGroupDurationSeconds' not in retention_rule

    def test_create_retention_policy_with_invalid_duration_format(self):
        """
        Test creating retention policy with malformed duration string
        Verifies that the function returns False when given an invalid duration format
        """
        # Setup
        manager = InfluxDBManager()
        manager._retention_api = MagicMock()
        
        # Mock the bucket API response
        bucket = "valid_bucket"
        invalid_duration = "1x"  # Invalid unit format
        
        # Test
        result = manager.create_retention_policy(bucket, invalid_duration)
        
        # Verify
        assert result is False, "Should return False for invalid duration format"

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionRulesApi')
    def test_create_retention_policy_with_nonexistent_bucket(self, mock_retention_api, mock_buckets_api):
        """
        Test creating retention policy with non-existent bucket
        Verifies that ValueError is raised and returns False when trying to create
        a retention policy for a bucket that doesn't exist
        """
        # Setup mock to simulate non-existent bucket
        mock_buckets_api_instance = MagicMock()
        mock_buckets_api_instance.find_bucket_by_name.return_value = None
        mock_buckets_api.return_value = mock_buckets_api_instance
        
        # Initialize the manager
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api_instance
        manager._retention_api = MagicMock()  # Not needed for this test but initialized
        
        # Test with invalid bucket name
        invalid_bucket_name = "non_existent_bucket"
        duration = "1d"
        
        with pytest.raises(ValueError) as exc_info:
            manager.create_retention_policy(invalid_bucket_name, duration)
        
        # Verify the error message
        assert "Bucket does not exist" in str(exc_info.value)
        
        # Verify the bucket API was called to check existence
        mock_buckets_api_instance.find_bucket_by_name.assert_called_once_with(invalid_bucket_name)
        
        # Verify retention API was not called (since bucket doesn't exist)
        manager._retention_api.create.assert_not_called()

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionRulesApi')
    def test_create_retention_policy_with_valid_bucket(self, mock_retention_api, mock_buckets_api):
        """
        Test creating retention policy with existing bucket
        Verifies that:
        1. Function returns True when successful
        2. Bucket retention rules are updated
        """
        # Setup test data
        bucket_name = "test_bucket"
        duration = "1d"
        shard_duration = "1h"
        
        # Create mock instances
        mock_bucket = MagicMock()
        mock_bucket.retention_rules = []
        
        # Configure mock APIs
        mock_buckets_api_instance = MagicMock()
        mock_buckets_api_instance.find_bucket_by_name.return_value = mock_bucket
        mock_buckets_api.return_value = mock_buckets_api_instance
        
        mock_retention_api_instance = MagicMock()
        mock_retention_api.return_value = mock_retention_api_instance
        
        # Instantiate the manager
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api_instance
        manager._retention_api = mock_retention_api_instance
        
        # Call the method
        result = manager.create_retention_policy(bucket_name, duration, shard_duration)
        
        # Assertions
        assert result is True
        mock_buckets_api_instance.find_bucket_by_name.assert_called_once_with(bucket_name)
        assert len(mock_bucket.retention_rules) == 1
        assert mock_bucket.retention_rules[0].every_seconds == 86400  # 1d in seconds
        assert mock_bucket.retention_rules[0].shard_group_duration_seconds == 3600  # 1h in seconds
        mock_buckets_api_instance.update_bucket.assert_called_once_with(mock_bucket)

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionApi')
    @patch('infrastructure.database.influxdb_manager.InfluxDBClient')
    def test_connect_method_properly_initializes_apis(self, mock_client, mock_retention_api, mock_buckets_api):
        """
        Test that connect method properly initializes APIs
        """
        # Setup
        config = {
            'url': 'http://localhost:8086',
            'token': 'test-token',
            'org': 'test-org'
        }
        
        # Mock the client and APIs
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_buckets_api_instance = MagicMock()
        mock_buckets_api.return_value = mock_buckets_api_instance
        
        mock_retention_api_instance = MagicMock()
        mock_retention_api.return_value = mock_retention_api_instance
        
        # Test
        manager = InfluxDBManager()
        manager.connect(config)
        
        # Assert
        assert manager._buckets_api is not None
        assert manager._retention_api is not None
        assert isinstance(manager._buckets_api, MagicMock)
        assert isinstance(manager._retention_api, MagicMock)
        
        # Verify APIs were created with the client
        mock_buckets_api.assert_called_once_with(mock_client_instance)
        mock_retention_api.assert_called_once_with(mock_client_instance)

    def test_initialize_with_default_values(self):
        """
        Test that the class initializes with None values for _buckets_api and _retention_api
        """
        # Create an instance of InfluxDBManager
        manager = InfluxDBManager()
        
        # Assert that the attributes are initialized to None
        assert manager._buckets_api is None
        assert manager._retention_api is None

    @patch('infrastructure.database.influxdb_manager.BucketsApi')
    @patch('infrastructure.database.influxdb_manager.RetentionApi')
    def test_optimize_for_high_frequency_creates_retention_policies_and_continuous_query_without_errors(self, mock_retention_api, mock_buckets_api):
        """
        Test that optimize_for_high_frequency successfully creates retention policies
        and continuous query without errors when given a valid bucket name.
        """
        # Setup
        manager = InfluxDBManager()
        manager._buckets_api = mock_buckets_api.return_value
        manager._retention_api = mock_retention_api.return_value
        
        # Mock the bucket existence check
        mock_buckets_api.return_value.find_bucket_by_name.return_value = MagicMock()
        
        # Mock successful creation of retention policies and continuous queries
        mock_retention_api.return_value.create_retention_rule.return_value = None
        
        # Test input
        valid_bucket_name = "test_bucket"
        
        # Execute
        try:
            manager.optimize_for_high_frequency(valid_bucket_name)
            # If we get here without exceptions, the test passes
            assert True
        except Exception as e:
            pytest.fail(f"optimize_for_high_frequency raised an exception: {str(e)}")
        
        # Verify that the necessary methods were called
        mock_buckets_api.return_value.find_bucket_by_name.assert_called_once_with(valid_bucket_name)
        assert mock_retention_api.return_value.create_retention_rule.call_count >= 1, \
            "Should create at least one retention rule"

    @patch('infrastructure.database.influxdb_manager.InfluxDBManager._get_buckets_api')
    @patch('infrastructure.database.influxdb_manager.InfluxDBManager._get_retention_api')
    def test_optimize_for_high_frequency_with_invalid_bucket_name_raises_exception(
            self, mock_get_retention_api, mock_get_buckets_api):
        """
        Test that optimize_for_high_frequency raises an exception when given an invalid bucket name
        """
        # Setup
        manager = InfluxDBManager()
        
        # Configure mock APIs
        mock_buckets_api = MagicMock()
        mock_buckets_api.find_bucket_by_name.side_effect = Exception("Bucket not found")
        mock_get_buckets_api.return_value = mock_buckets_api
        
        mock_retention_api = MagicMock()
        mock_get_retention_api.return_value = mock_retention_api
        
        # Test & Assert
        with pytest.raises(Exception) as exc_info:
            manager.optimize_for_high_frequency("invalid_bucket_name")
        
        assert "Bucket not found" in str(exc_info.value)
        mock_buckets_api.find_bucket_by_name.assert_called_once_with("invalid_bucket_name")


        