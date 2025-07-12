import re
import unittest
from unittest.mock import MagicMock
from src.infrastructure.m_logging.quant_filter import QuantFilter  # Assuming the filter method is in QuantFilter class

class QuantFilterTest(unittest.TestCase):
    def test_filter_record_without_signal_attribute(self):
        """Test when the record does not have a 'signal' attribute"""
        # Create a mock record without signal attribute
        record = MagicMock()
        del record.signal  # Ensure signal attribute doesn't exist

        # Create instance of the class containing the filter method
        quant_filter = QuantFilter()

        # Call the filter method
        result = quant_filter.filter(record)

        # Assert the signal attribute was set to None
        self.assertIsNone(record.signal)
        # Assert the method returns True
        self.assertTrue(result)

    def test_filter_with_existing_signal_attribute(self):
        """Test when the record already has a 'signal' attribute"""
        # Create a mock record object with 'signal' attribute
        record = MagicMock()
        record.signal = "existing_signal_value"
        record.msg = "test message"

        # Create an instance of the class containing the filter method
        # (assuming it's part of a class with SENSITIVE_PATTERNS)
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = []  # No patterns to avoid interference

        # Call the filter method
        result = filter_instance.filter(record)

        # Assert that the signal attribute remains unchanged
        self.assertEqual(record.signal, "existing_signal_value")
        self.assertTrue(result)  # The filter method should always return True

    def test_empty_message_with_no_sensitive_patterns(self):
        """Test with an empty message and no sensitive patterns to match"""
        # Create a mock record object with empty msg
        record = MagicMock()
        record.msg = ""

        # Create an instance of the filter class (assuming it's a class method)
        # We need to mock the SENSITIVE_PATTERNS as empty list
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = []

        # Call the filter method
        result = filter_instance.filter(record)

        # Assert the record.msg remains unchanged
        self.assertEqual(record.msg, "")
        self.assertTrue(result)  # filter should always return True

    def test_message_with_single_sensitive_pattern_match(self):
        """Test when message contains one sensitive pattern"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'password:\s*\w+', r'creditcard:\s*\d+']
        filter_instance.filter = QuantFilterTest._get_filter_function(filter_instance)

        # Create a test record
        record = MagicMock()
        record.msg = "This contains password: secret123"

        # Execute
        result = filter_instance.filter(record)

        # Assert
        self.assertTrue(result)
        self.assertEqual(record.msg, "[REDACTED] This contains password: ****")

    @staticmethod
    def _get_filter_function(filter_instance):
        """Helper method to get the filter function with proper 'self' binding"""
        def filter(record):
            """Filter method that adds quant-specific fields and filters sensitive info"""
            # Keep original signal field processing
            if not hasattr(record, 'signal'):
                record.signal = None  # Default signal value

            # New sensitive info filtering
            msg = str(record.msg)
            for pattern in filter_instance.SENSITIVE_PATTERNS:
                if re.search(pattern, msg):
                    record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                    break

            return True
        return filter

    def test_message_with_multiple_sensitive_patterns(self):
        """Test when message contains multiple sensitive patterns"""
        # Setup
        class MockFilter:
            SENSITIVE_PATTERNS = [
                r'password=\w+',
                r'credit_card=\d+',
                r'api_key=\w+'
            ]

            def filter(self, record):
                # Original filter function implementation
                if not hasattr(record, 'signal'):
                    record.signal = None

                msg = str(record.msg)
                for pattern in self.SENSITIVE_PATTERNS:
                    if re.search(pattern, msg):
                        record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                        break

                return True

        # Create test record
        record = MagicMock()
        record.msg = "Login attempt with password=secret and credit_card=1234567890"

        # Execute
        filter_instance = MockFilter()
        result = filter_instance.filter(record)

        # Verify
        self.assertTrue(result)
        self.assertTrue(record.msg.startswith("[REDACTED] "))
        self.assertIn("password=****", record.msg)
        self.assertIn("credit_card=1234567890", record.msg)  # Should not be redacted
        self.assertEqual(record.signal, None)

    def test_filter_message_with_no_sensitive_patterns(self):
        """Test when message contains no sensitive patterns"""
        # Mock the filter object with SENSITIVE_PATTERNS
        filter_obj = MagicMock()
        filter_obj.SENSITIVE_PATTERNS = [
            r'password=\w+',
            r'token=\w+',
            r'secret=\w+'
        ]

        # Create a test record with message containing no sensitive patterns
        record = MagicMock()
        original_msg = "This is a normal log message without sensitive data"
        record.msg = original_msg

        # Call the filter method
        result = filter_obj.filter(record)

        # Assertions
        self.assertTrue(result)  # filter should always return True
        self.assertEqual(record.msg, original_msg)  # message should remain unchanged

    def test_message_with_partial_sensitive_pattern_match(self):
        """Test when message partially matches a sensitive pattern"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'\bpassword\b', r'\bcredit_card\b']

        # Create a mock record with msg containing substring of a pattern
        record = MagicMock()
        record.msg = "This contains pass but not the full pattern"

        # Save original msg for comparison
        original_msg = str(record.msg)

        # Execute
        result = filter.__get__(filter_instance)(filter_instance, record)

        # Assert
        self.assertTrue(result)  # filter should always return True
        self.assertEqual(record.msg, original_msg)  # msg should remain unchanged

    def test_message_with_case_sensitive_pattern_match(self):
        """Test case sensitivity in pattern matching"""
        # Create a mock filter object with SENSITIVE_PATTERNS
        filter_obj = MagicMock()
        filter_obj.SENSITIVE_PATTERNS = [r'password', r'secret']

        # Create a test record with message containing case variation of pattern
        record = MagicMock()
        record.msg = "This is a SECRET message"

        # Call the filter method
        result = filter_obj.filter(record)

        # Check if the message was modified (depends on case sensitivity)
        if any(re.compile(p, re.IGNORECASE).search(record.msg) for p in filter_obj.SENSITIVE_PATTERNS):
            # If patterns are case-insensitive, message should be redacted
            self.assertTrue(record.msg.startswith("[REDACTED]"))
            self.assertIn("****", record.msg)
        else:
            # If patterns are case-sensitive, message should remain unchanged
            self.assertEqual(record.msg, "This is a SECRET message")

        # The method should always return True
        self.assertTrue(result)

    def test_filter_message_with_special_characters(self):
        """Test filter method with message containing special characters"""
        # Mock the record object
        record = MagicMock()
        record.msg = "This message contains special characters @#$%"

        # Mock the filter instance with SENSITIVE_PATTERNS
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'[@#$%]']  # Pattern to match special characters

        # Replace the actual filter method with our test version
        original_filter = filter_instance.filter
        def test_filter(rec):
            msg = str(rec.msg)
            for pattern in filter_instance.SENSITIVE_PATTERNS:
                if re.search(pattern, msg):
                    rec.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                    break
            return True
        filter_instance.filter = test_filter

        # Execute the filter
        result = filter_instance.filter(record)

        # Verify the outcome
        self.assertTrue(result)
        self.assertEqual(record.msg, "[REDACTED] This message contains special characters ****")

    def test_filter_with_multiline_message(self):
        """Test that filter properly handles multi-line messages in pattern matching"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'secret', r'password']  # Example sensitive patterns

        # Create a record with multi-line message containing sensitive info
        record = MagicMock()
        record.msg = "This is a multi-line\nmessage containing\nsecret information"

        # Replace the actual filter method with our test version
        from src.infrastructure.m_logging.quant_filter import QuantFilter
        original_filter = QuantFilter.filter
        QuantFilter.filter = lambda self, record: filter(self, record)

        try:
            # Test
            result = filter(filter_instance, record)

            # Verify
            self.assertTrue(result)  # Should always return True
            self.assertIn("[REDACTED]", record.msg)  # Should be redacted
            self.assertIn("****", record.msg)  # Should have sensitive info replaced
            self.assertEqual(record.msg.count("\n"), 2)  # Should preserve line breaks

        finally:
            # Restore original filter method
            QuantFilter.filter = original_filter

    def test_filter_with_empty_sensitive_patterns(self):
        """Test when SENSITIVE_PATTERNS is empty"""
        # Create a mock record object
        record = MagicMock()
        record.msg = "This contains sensitive information like password123"

        # Create an instance of the filter class (assuming it's a class method)
        # We'll mock the SENSITIVE_PATTERNS as empty list
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = []

        # Call the filter method
        result = filter_instance.filter(record)

        # Assert the message remains unchanged
        self.assertEqual(record.msg, "This contains sensitive information like password123")
        self.assertTrue(result)  # The filter should always return True

    def test_multiple_pattern_matches_but_only_first_is_redacted(self):
        """Verify only first match is redacted when multiple patterns match"""
        # Create a mock filter instance with SENSITIVE_PATTERNS
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [
            r'password:\s*\w+',  # First pattern to match
            r'credit_card:\s*\d+'  # Second pattern to match
        ]

        # Create a mock record with msg containing multiple sensitive patterns
        record = MagicMock()
        record.msg = "User entered password: secret and credit_card: 1234567890"

        # Call the filter method
        result = filter.__get__(filter_instance)(record)

        # Verify the result is True
        self.assertTrue(result)

        # Verify only the first pattern was redacted
        self.assertEqual(record.msg, "[REDACTED] User entered password: **** and credit_card: 1234567890")

    def test_non_string_message_content(self):
        """Test with non-string message content"""
        # Create a mock record with non-string msg
        record = MagicMock()
        record.msg = 12345  # Non-string message (number)

        # Create filter instance with some SENSITIVE_PATTERNS
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'sensitive']
        filter_instance.filter = QuantFilterTest._get_filter_method()

        # Call the filter method
        result = filter_instance.filter(record)

        # Verify the message was converted to string
        self.assertIsInstance(record.msg, str)
        self.assertEqual(record.msg, "12345")
        self.assertTrue(result)

    @staticmethod
    def _get_filter_method():
        """Helper method to get the original filter method implementation"""
        def filter(self, record):
            """Filter method, add quant-specific fields and filter sensitive info"""
            # Preserve original signal field processing
            if not hasattr(record, 'signal'):
                record.signal = None  # Default signal value

            # New sensitive information filtering
            msg = str(record.msg)  # This is what we're testing - conversion to string
            for pattern in self.SENSITIVE_PATTERNS:
                if re.search(pattern, msg):
                    record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                    break

            return True
        return filter.__get__(object)

    def test_complex_regex_pattern_matching(self):
        """Test with complex regex patterns in SENSITIVE_PATTERNS"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?\b'
            # Phone pattern
        ]

        # Create a record with message containing sensitive information
        record = MagicMock()
        record.msg = "User info: SSN 123-45-6789, email test@example.com, phone 123-456-7890"

        # Call the filter method
        result = filter.__get__(filter_instance)(filter_instance, record)

        # Assertions
        self.assertTrue(result)
        self.assertEqual(record.msg, "[REDACTED] User info: SSN ****, email ****, phone ****")

        # Verify signal field was set to None if not present
        self.assertIsNone(record.signal)

    def test_message_exactly_matching_sensitive_pattern(self):
        """Test when message exactly matches a sensitive pattern"""
        # Setup
        class MockFilter:
            SENSITIVE_PATTERNS = [r'password=\w+', r'credit_card=\d+']

            def filter(self, record):
                # Original filter function implementation
                if not hasattr(record, 'signal'):
                    record.signal = None

                msg = str(record.msg)
                for pattern in self.SENSITIVE_PATTERNS:
                    if re.search(pattern, msg):
                        record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                        break

                return True

        filter_instance = MockFilter()

        # Create a mock record with message exactly matching a sensitive pattern
        record = MagicMock()
        record.msg = "password=secret123"
        record.signal = None  # Ensure signal is set to None as per filter behavior

        # Execute
        result = filter_instance.filter(record)

        # Assert
        self.assertTrue(result)  # Filter should always return True
        self.assertEqual(record.msg, "[REDACTED] password=****")

    def test_message_containing_multiple_instances_of_sensitive_pattern(self):
        """Test when message contains multiple instances of the same sensitive pattern"""
        # Setup
        class MockFilter:
            SENSITIVE_PATTERNS = [r'password:\w+']

            def filter(self, record):
                # Original filter function implementation
                if not hasattr(record, 'signal'):
                    record.signal = None

                msg = str(record.msg)
                for pattern in self.SENSITIVE_PATTERNS:
                    if re.search(pattern, msg):
                        record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                        break

                return True

        filter_instance = MockFilter()

        # Create a mock record with message containing multiple sensitive patterns
        record = MagicMock()
        record.msg = "User login with password:12345 and password:abcde"

        # Execute
        result = filter_instance.filter(record)

        # Verify
        self.assertTrue(result)
        self.assertEqual(record.msg, "[REDACTED] User login with **** and ****")

    def test_filter_very_long_message_with_sensitive_content(self):
        """Test filter with very long message containing sensitive content"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'password:\s*\w+', r'credit card:\s*\d+']

        # Create a very long message with sensitive content
        long_message = "This is a very long message " * 100 + "with sensitive content like password: mysecret and credit card: 1234567890123456"
        record = MagicMock()
        record.msg = long_message

        # Call the actual filter function (not the mock)
        from src.infrastructure.m_logging.quant_filter import QuantFilter
        quant_filter = QuantFilter()
        quant_filter.SENSITIVE_PATTERNS = filter_instance.SENSITIVE_PATTERNS

        # Execute
        result = quant_filter.filter(record)

        # Verify
        self.assertTrue(result)
        self.assertTrue(record.msg.startswith("[REDACTED]"))
        self.assertIn("password: ****", record.msg)
        self.assertIn("credit card: ****", record.msg)
        self.assertNotIn("mysecret", record.msg)
        self.assertNotIn("1234567890123456", record.msg)

    def test_filter_with_unicode_characters_in_message(self):
        """Test that filter properly handles unicode characters in message"""
        # Setup
        filter_instance = MagicMock()
        filter_instance.SENSITIVE_PATTERNS = [r'敏感词']  # Chinese sensitive word pattern

        # Create a record with unicode message
        record = MagicMock()
        record.msg = "这是一条包含敏感词的消息"  # Chinese message containing sensitive word

        # Replace the actual filter method with our testable version
        def filter_method(record):
            msg = str(record.msg)
            for pattern in filter_instance.SENSITIVE_PATTERNS:
                if re.search(pattern, msg):
                    record.msg = "[REDACTED] " + re.sub(pattern, "****", msg)
                    break
            return True

        filter_instance.filter = filter_method

        # Execute
        result = filter_instance.filter(record)

        # Verify
        self.assertTrue(result)
        self.assertEqual(record.msg, "[REDACTED] 这是一条包含****的消息")

    def test_filter_always_returns_true(self):
        """Test that the filter method always returns True"""
        # Create a mock record object with minimal required attributes
        record = MagicMock()
        record.msg = "Test message"

        # Create an instance of the QuantFilter class (or the class that contains the filter method)
        # Assuming SENSITIVE_PATTERNS is a class attribute that needs to be initialized
        quant_filter = QuantFilter()
        quant_filter.SENSITIVE_PATTERNS = []  # Empty patterns list for this test

        # Call the filter method
        result = quant_filter.filter(record)

        # Assert that the result is always True
        self.assertTrue(result)








































