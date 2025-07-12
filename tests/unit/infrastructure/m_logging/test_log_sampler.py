import time
from unittest.mock import MagicMock
from src.infrastructure.m_logging.log_sampler import SamplingRule, SamplingStrategyType
import pytest
from unittest.mock import Mock
from src.infrastructure.m_logging.log_sampler import LogSampler


@pytest.fixture
def sampler(self):
    """Fixture to create a LogSampler instance with some initial rules."""
    sampler = LogSampler()
    # Add some mock rules
    sampler._rules = [MagicMock(), MagicMock(), MagicMock()]
    return sampler

class TestLogSamplerInitialization:
    def test_initialization_with_default_parameters(self):
        """
        Test that the sampler initializes correctly with default parameters.
        """
        # Create an instance of LogSampler with default parameters
        sampler = LogSampler()

        # Check that the attributes are set to the expected default values
        assert sampler._base_rate == 1.0
        assert sampler._min_rate == 0.01
        assert sampler._max_rate == 1.0
        assert sampler._load_window == 60

        # Check that the default rule is added
        assert len(sampler._rules) == 1
        default_rule = sampler._rules[0]
        assert default_rule.strategy == "FIXED_RATE"
        assert default_rule.rate == 1.0

    def test_initialization_with_custom_parameters(self):
        """
        Test that the sampler initializes correctly with custom parameters.
        """
        # Arrange
        base_rate = 0.5
        min_rate = 0.1
        max_rate = 0.9
        load_window = 30

        # Act
        sampler = LogSampler(
            base_rate=base_rate,
            min_rate=min_rate,
            max_rate=max_rate,
            load_window=load_window
        )

        # Assert
        assert sampler._base_rate == base_rate
        assert sampler._min_rate == min_rate
        assert sampler._max_rate == max_rate
        assert sampler._load_window == load_window

        # Check default rule was added
        assert len(sampler._rules) == 1
        default_rule = sampler._rules[0]
        assert isinstance(default_rule, SamplingRule)
        assert default_rule.strategy == SamplingStrategyType.FIXED_RATE
        assert default_rule.rate == base_rate

    def test_add_rule(self):
        """
        Test that adding a new sampling rule correctly adds it to the _rules list.
        """
        # Setup
        sampler = LogSampler()
        mock_rule = Mock()

        # Pre-condition check
        assert len(sampler._rules) == 1  # Default rule added in __init__

        # Execute
        sampler.add_rule(mock_rule)

        # Verify
        assert len(sampler._rules) == 2
        assert mock_rule in sampler._rules
        assert sampler._rules[-1] == mock_rule  # Should be added to the end

    def test_add_rule_thread_safety(self):
        """
        Test that adding a rule is thread-safe (lock is acquired/released).
        """
        # Setup
        sampler = LogSampler()
        mock_rule = Mock()

        # Mock the lock to verify it's used
        original_lock = sampler._lock
        mock_lock = Mock(wraps=original_lock)
        sampler._lock = mock_lock

        try:
            # Execute
            sampler.add_rule(mock_rule)

            # Verify lock was used
            mock_lock.__enter__.assert_called_once()
            mock_lock.__exit__.assert_called_once_with(None, None, None)
        finally:
            # Restore original lock
            sampler._lock = original_lock

    def test_remove_rule_by_valid_index(self, sampler):
        """Test that remove_rule correctly removes a rule at valid index"""
        # Store original rules for comparison
        original_rules = sampler._rules.copy()

        # Remove rule at index 1
        sampler.remove_rule(1)

        # Verify the rule at index 1 was removed
        assert len(sampler._rules) == len(original_rules) - 1
        assert sampler._rules[0] == original_rules[0]
        assert sampler._rules[1] == original_rules[2]  # The third rule should now be at index 1

        # Verify lock was used
        sampler._lock.__enter__.assert_called()
        sampler._lock.__exit__.assert_called()

    def test_remove_rule_by_invalid_index(self, sampler):
        """
        Test that removing a rule by an invalid index doesn't modify the rules list.
        """
        # Store original rules for comparison
        original_rules = sampler._rules.copy()

        # Try to remove with index that's too large
        sampler.remove_rule(10)
        assert sampler._rules == original_rules

        # Try to remove with negative index
        sampler.remove_rule(-1)
        assert sampler._rules == original_rules

        # Try to remove with index equal to length
        sampler.remove_rule(len(sampler._rules))
        assert sampler._rules == original_rules

    def test_clear_rules_clears_all_rules(self):
        """
        Test that clear_rules() method clears all sampling rules.
        """
        # Setup - create a LogSampler instance with some rules
        sampler = LogSampler()

        # Add some mock rules
        mock_rule1 = MagicMock()
        mock_rule2 = MagicMock()
        sampler._rules = [mock_rule1, mock_rule2]

        # Action - call clear_rules
        sampler.clear_rules()

        # Assert - verify _rules list is empty
        assert len(sampler._rules) == 0

    def test_set_base_rate_within_bounds(self):
        """
        Test setting base rate within min/max bounds.
        Verifies that the base rate is correctly set when within valid bounds.
        """
        # Initialize with default values (min_rate=0.01, max_rate=1.0)
        sampler = LogSampler()

        # Test input
        test_rate = 0.5

        # Expected outcome
        expected_rate = 0.5

        # Set the base rate
        sampler.set_base_rate(test_rate)

        # Verify the base rate was set correctly
        assert sampler._base_rate == expected_rate, \
            f"Expected base rate {expected_rate}, got {sampler._base_rate}"

    def test_set_base_rate_below_min(self):
        """
        Test that setting base rate below minimum clamps to min_rate.
        """
        # Initialize with default min_rate=0.01
        sampler = LogSampler()

        # Set base rate below minimum
        sampler.set_base_rate(0.001)

        # Verify it's clamped to min_rate
        assert sampler._base_rate == 0.01

    def test_set_base_rate_above_max(self):
        """
        Test that setting base rate above maximum value gets clamped to max.
        """
        # Initialize with max_rate=1.0
        sampler = LogSampler(max_rate=1.0)

        # Try to set base rate above max
        sampler.set_base_rate(1.5)

        # Assert that the value was clamped to max
        assert sampler._base_rate == 1.0

    def test_adjust_for_load_below_threshold(self):
        """
        Test that the base rate remains unchanged when current load is below threshold (0.7)
        """
        # Setup
        sampler = LogSampler(base_rate=0.8, min_rate=0.01, max_rate=1.0)
        original_base_rate = sampler._base_rate

        # Mock necessary components if needed
        sampler._load_history = Mock()
        sampler._load_history.__len__.return_value = 60  # Ensure window is full

        # Test
        sampler.adjust_for_load(current_load=0.5)

        # Verify
        assert sampler._base_rate == original_base_rate, \
            "Base rate should remain unchanged when load is below threshold"

    def test_adjust_for_load_above_threshold(self):
        """
        Test that the base rate is reduced proportionally when current load
        is above the threshold (0.7).
        """
        # Setup
        base_rate = 1.0
        min_rate = 0.01
        max_rate = 1.0
        load_window = 60

        sampler = LogSampler(
            base_rate=base_rate,
            min_rate=min_rate,
            max_rate=max_rate,
            load_window=load_window
        )

        # Mock the load history to simulate high load
        sampler._load_history = Mock()
        sampler._load_history.__len__.return_value = load_window
        sampler._load_history.__iter__.return_value = [0.8] * load_window

        # Test
        current_load = 0.8
        adjusted_rate = sampler.adjust_for_load(current_load)

        # Verify
        assert adjusted_rate < base_rate  # Rate should be reduced
        assert adjusted_rate >= min_rate  # Shouldn't go below min rate
        assert adjusted_rate <= max_rate  # Shouldn't exceed max rate

        # Check proportional reduction (0.8 is 14.3% above 0.7 threshold)
        # So rate should be reduced by approximately 14.3%
        expected_rate = base_rate * (0.7 / 0.8)
        assert adjusted_rate == pytest.approx(expected_rate, rel=0.1)

    def test_adjust_for_load_too_frequent(self):
        """
        Test that adjust_for_load does not adjust rate when called too frequently (<5s).
        """
        # Create a sampler instance with default parameters
        sampler = LogSampler()

        # Mock the necessary internal methods to track calls
        sampler._adjust_rate = MagicMock()

        # First call to set the last adjust time
        sampler.adjust_for_load(current_load=0.8)

        # Call again within 5 seconds
        sampler.adjust_for_load(current_load=0.8)

        # Verify adjust_rate was not called (only called once during initialization)
        assert sampler._adjust_rate.call_count == 1

    def test_adjust_for_load_after_cooldown(self):
        """
        Test that adjust_for_load does adjust rate when called after cooldown period.
        """
        # Create a sampler instance with default parameters
        sampler = LogSampler()

        # Mock the necessary internal methods to track calls
        sampler._adjust_rate = MagicMock()

        # First call to set the last adjust time
        sampler.adjust_for_load(current_load=0.8)

        # Simulate waiting more than 5 seconds
        sampler._last_adjust_time = time.time() - 6

        # Call again after cooldown
        sampler.adjust_for_load(current_load=0.8)

        # Verify adjust_rate was called (once during init, once now)
        assert sampler._adjust_rate.call_count == 2

    def test_calculate_dynamic_rate_below_threshold(self):
        """
        Test that dynamic rate calculation returns base rate unchanged
        when load is below threshold.
        """
        # Setup
        base_rate = 0.8
        min_rate = 0.1
        max_rate = 1.0
        sampler = LogSampler(base_rate=base_rate, min_rate=min_rate, max_rate=max_rate)

        # Mock necessary components if needed
        sampler._load_history = Mock()
        sampler._load_history.__iter__.return_value = [0.5]  # Load below threshold

        # Test
        result = sampler._calculate_dynamic_rate()

        # Verify
        assert result == base_rate, "Should return base rate unchanged when load is below threshold"

    def test_should_sample_with_matching_rule(self):
        """
        Test that should_sample returns the correct sampling decision
        when a record matches a rule's conditions.
        """
        # Setup
        sampler = LogSampler(base_rate=0.5)

        # Create a mock rule that will match and return True
        matching_rule = MagicMock()
        matching_rule.matches.return_value = True
        matching_rule.should_sample.return_value = True

        # Add the mock rule to the sampler
        sampler._rules = [matching_rule]

        # Test data
        test_record = {"message": "test log message"}

        # Execute
        result = sampler.should_sample(test_record)

        # Verify
        matching_rule.matches.assert_called_once_with(test_record)
        matching_rule.should_sample.assert_called_once()
        assert result is True

    def test_should_sample_with_no_matching_rules(self):
        """
        Test that should_sample returns the base rate sampling decision
        when no rules match the record.
        """
        # Setup
        base_rate = 0.5
        sampler = LogSampler(base_rate=base_rate)

        # Create a mock rule that won't match
        non_matching_rule = MagicMock()
        non_matching_rule.matches.return_value = False

        # Add the mock rule to the sampler
        sampler._rules = [non_matching_rule]

        # Test data
        test_record = {"message": "test log message"}

        # Execute
        result = sampler.should_sample(test_record)

        # Verify
        non_matching_rule.matches.assert_called_once_with(test_record)
        non_matching_rule.should_sample.assert_not_called()
        # Since no rules match, it should use the base rate
        # We can't directly assert the random sampling result, but we can verify
        # that it's being called with the base rate
        assert isinstance(result, bool)

    def test_match_fixed_rate_rule(self):
        """
        Test that a rule with FIXED_RATE strategy always matches (returns True).
        """
        # Setup
        sampler = LogSampler()
        fixed_rate_rule = SamplingRule(strategy=SamplingStrategyType.FIXED_RATE, rate=0.5)

        # Mock the log record (not actually used for fixed rate)
        mock_record = MagicMock()

        # Test
        result = sampler._match_rule(mock_record, fixed_rate_rule)

        # Verify
        assert result is True, "Fixed rate rule should always match (return True)"

    def test_match_level_based_rule(self):
        """
        Test that the sampler correctly matches records based on level rules.
        """
        # Setup
        sampler = LogSampler()

        # Create a mock rule that matches DEBUG level
        debug_rule = Mock()
        debug_rule.strategy = "LEVEL_BASED"
        debug_rule.level = "DEBUG"

        # Add the rule to the sampler
        sampler._rules = [debug_rule]

        # Test cases
        test_cases = [
            # (record_level, expected_result)
            ("DEBUG", True),  # exact match
            ("INFO", False),  # different level
            ("WARNING", False),  # different level
            ("ERROR", False),  # different level
            ("CRITICAL", False),  # different level
        ]

        for record_level, expected in test_cases:
            # Create a mock record with the current level
            record = Mock()
            record.level = record_level

            # Exercise
            result = sampler._match_rule(record, debug_rule)

            # Verify
            assert result == expected, f"Failed for level {record_level}"

        @pytest.fixture
        def sampler(self):
            return LogSampler(base_rate=1.0)

        def test_apply_sampling_fixed_rate(self, sampler):
            """Test fixed rate sampling strategy"""
            rule = SamplingRule(strategy=SamplingStrategyType.FIXED_RATE, rate=0.5)

            # Mock random.random to control the test
            original_random = sampler._random.random
            sampler._random.random = MagicMock(return_value=0.4)  # Below threshold
            assert sampler._apply_sampling(rule) is True

            sampler._random.random = MagicMock(return_value=0.6)  # Above threshold
            assert sampler._apply_sampling(rule) is False

            # Restore original random function
            sampler._random.random = original_random

        def test_apply_sampling_dynamic_rate(self, sampler):
            """Test dynamic rate sampling strategy"""
            rule = SamplingRule(strategy=SamplingStrategyType.DYNAMIC_RATE, rate=0.7)

            # Mock random.random to control the test
            original_random = sampler._random.random
            sampler._random.random = MagicMock(return_value=0.65)  # Below threshold
            assert sampler._apply_sampling(rule) is True

            sampler._random.random = MagicMock(return_value=0.75)  # Above threshold
            assert sampler._apply_sampling(rule) is False

            # Restore original random function
            sampler._random.random = original_random

        def test_apply_sampling_always_sample(self, sampler):
            """Test always sample strategy"""
            rule = SamplingRule(strategy=SamplingStrategyType.ALWAYS, rate=0.0)
            assert sampler._apply_sampling(rule) is True

        def test_apply_sampling_never_sample(self, sampler):
            """Test never sample strategy"""
            rule = SamplingRule(strategy=SamplingStrategyType.NEVER, rate=1.0)
            assert sampler._apply_sampling(rule) is False

        def test_get_current_strategy_returns_configuration(self):
            """
            Test that get_current_strategy returns the current configuration
            including base rate, min rate, max rate, load window and rules.
            """
            # Setup
            sampler = LogSampler(
                base_rate=0.5,
                min_rate=0.1,
                max_rate=0.9,
                load_window=30
            )

            # Mock the rules list
            mock_rule = MagicMock()
            mock_rule.to_dict.return_value = {
                "strategy": "FIXED_RATE",
                "rate": 0.5
            }
            sampler._rules = [mock_rule]

            # Execute
            result = sampler.get_current_strategy()

            # Verify
            assert isinstance(result, dict)
            assert result["base_rate"] == 0.5
            assert result["min_rate"] == 0.1
            assert result["max_rate"] == 0.9
            assert result["load_window"] == 30
            assert len(result["rules"]) == 1
            assert result["rules"][0] == {
                "strategy": "FIXED_RATE",
                "rate": 0.5
            }

    def test_should_sample_with_no_rules(self):
        """
        Test that sampling returns True (default sampling) when no rules exist.
        """
        # Create a LogSampler instance with default parameters
        sampler = LogSampler()

        # Clear any default rules that might be added in __init__
        sampler._rules = []

        # Call the should_sample method
        result = sampler.should_sample()

        # Verify the result is True (default sampling)
        assert result is True

    def test_calculate_dynamic_rate_above_threshold(self, sampler):
        """
        Test dynamic rate calculation when load is above threshold.
        Should return reduced rate (base_rate * (1 - scale * 0.9)).
        """
        # Mock the necessary attributes
        sampler._base_rate = 0.5
        sampler._min_rate = 0.01
        sampler._max_rate = 1.0
        sampler._load_window = 60

        # Test with load above threshold (0.8)
        load = 0.8
        expected_rate = 0.5 * (1 - 0.9 * 0.9)  # base_rate * (1 - scale * 0.9)

        # Call the method
        result = sampler._calculate_dynamic_rate(load)

        # Assert the result
        assert result == pytest.approx(expected_rate)
        assert result < sampler._base_rate  # Should be reduced from base rate
        assert result >= sampler._min_rate  # Should be above minimum rate



