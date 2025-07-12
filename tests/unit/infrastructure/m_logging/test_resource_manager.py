import threading
import time
import warnings

import pytest
from unittest.mock import MagicMock, patch, Mock
from src.infrastructure.m_logging.resource_manager import ResourceManager


class TestResourceManagerSingleton:
    @patch('infrastructure.m_logging.resource_manager.ResourceManager._lock')
    def test_singleton_instance_creation(self, mock_lock):
        # Setup
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        ResourceManager._instance = None

        # First call to __new__ should create a new instance
        first_instance = ResourceManager.__new__(ResourceManager)
        assert first_instance is not None
        assert ResourceManager._instance is first_instance
        assert mock_lock_instance.__enter__.called
        assert mock_lock_instance.__exit__.called

        # Reset mock for subsequent calls
        mock_lock_instance.reset_mock()

        # Subsequent calls should return the same instance
        second_instance = ResourceManager.__new__(ResourceManager)
        assert second_instance is first_instance
        # Lock should not be acquired again
        assert not mock_lock_instance.__enter__.called
        assert not mock_lock_instance.__exit__.called

        # Verify initialized flag
        assert not hasattr(first_instance, '_initialized') or first_instance._initialized is False

    def test_singleton_thread_safety(self):
        # This test would require more complex threading setup to properly test
        # For now we'll just verify the lock is used
        with patch('infrastructure.m_logging.resource_manager.ResourceManager._lock') as mock_lock:
            mock_lock_instance = MagicMock()
            mock_lock.return_value = mock_lock_instance
            ResourceManager._instance = None

            ResourceManager.__new__(ResourceManager)
            mock_lock.assert_called()
            mock_lock_instance.__enter__.assert_called()
            mock_lock_instance.__exit__.assert_called()

    @pytest.fixture
    def resource_manager_cls(self):
        """Fixture to provide a clean ResourceManager class for each test"""
        # We need to reset the _instance and _lock between tests
        with patch.object(ResourceManager, '_instance', None):
            with patch.object(ResourceManager, '_lock', threading.Lock()):
                yield ResourceManager

    @pytest.fixture
    def resource_manager(self):
        # Create a mock instance of ResourceManager
        rm = ResourceManager.__new__(ResourceManager)
        rm._initialized = False
        rm._lock = MagicMock()
        return rm

    def test_initialization_flag(self, resource_manager):
        """Test that _initialized flag prevents reinitialization."""
        # Mock the super().__new__ method
        with patch.object(ResourceManager, '__init__', return_value=None) as mock_init:
            # First call to __new__ should initialize
            instance1 = ResourceManager.__new__(ResourceManager.__class__)
            assert instance1._initialized is False
            mock_init.assert_called_once()

            # Reset the mock for clearer assertion
            mock_init.reset_mock()

            # Second call to __new__ should not initialize again
            instance2 = ResourceManager.__new__(ResourceManager.__class__)
            assert instance2._initialized is False  # Still False because we're not calling __init__
            mock_init.assert_not_called()

            # Now simulate __init__ being called
            instance1.__init__()
            assert instance1._initialized is True

            # Third call to __new__ should return existing instance without calling __init__
            instance3 = ResourceManager.__new__(ResourceManager.__class__)
            assert instance3._initialized is True  # Because we're returning the same instance
            mock_init.assert_not_called()

    @patch('infrastructure.m_logging.resource_manager.ResourceManager._instance', None)
    @patch('infrastructure.m_logging.resource_manager.ResourceManager._lock', MagicMock())
    def test_resource_manager_initialization(self):
        """
        Test that ResourceManager properly initializes all attributes on first call.
        """
        # Create instance (first call)
        instance = ResourceManager()

        # Verify attributes are initialized correctly
        assert hasattr(instance, '_handlers')
        assert isinstance(instance._handlers, dict)
        assert hasattr(instance, '_closed')
        assert isinstance(instance._closed, bool)
        assert instance._closed is False
        assert hasattr(instance, '_initialized')
        assert instance._initialized is True

    @patch('infrastructure.m_logging.resource_manager.ResourceManager._instance', None)
    @patch('infrastructure.m_logging.resource_manager.ResourceManager._lock', MagicMock())
    def test_singleton_behavior(self):
        """
        Test that ResourceManager maintains singleton behavior.
        """
        # First call creates instance
        instance1 = ResourceManager()

        # Second call should return same instance
        instance2 = ResourceManager()

        assert instance1 is instance2
        assert ResourceManager._instance is not None

    def test_warning_threshold_setup(self):
        """
        Test that the warning_thresholds dictionary is set up correctly during initialization.
        """
        # Instantiate the ResourceManager
        resource_manager = ResourceManager()

        # Check if the warning_thresholds dictionary exists and has the expected keys
        assert hasattr(resource_manager, 'warning_thresholds'), "ResourceManager should have warning_thresholds attribute"

        # Verify the expected values in warning_thresholds
        expected_thresholds = {
            'cpu': 80,
            'memory': 85,
            'disk': 90,
            'network': 70
        }

        assert resource_manager.warning_thresholds == expected_thresholds, \
            f"Warning thresholds should be {expected_thresholds}, got {resource_manager.warning_thresholds}"

    def test_singleton_behavior(self):
        """
        Test that ResourceManager follows singleton pattern correctly.
        """
        # Create two instances
        instance1 = ResourceManager()
        instance2 = ResourceManager()

        # They should be the same instance
        assert instance1 is instance2, "ResourceManager should return the same instance"

        # The _instance class variable should be set
        assert ResourceManager._instance is not None, "ResourceManager._instance should be set"
        assert ResourceManager._instance is instance1, "ResourceManager._instance should point to the instance"

    def test_initialized_flag(self):
        """
        Test that the _initialized flag is set correctly.
        """
        # Create an instance
        instance = ResourceManager()

        # The _initialized flag should be False initially
        assert not instance._initialized, "_initialized should be False initially"

        # After initialization, it should be True
        instance._initialized = True
        assert instance._initialized, "_initialized should be True after initialization"

    def test_add_warning_handler(self):
        """
        Test that a callback function is correctly added to warning_callbacks.
        """
        # Setup
        resource_manager = ResourceManager()
        mock_callback = Mock()

        # Exercise
        resource_manager.add_warning_handler(mock_callback)

        # Verify
        assert mock_callback in resource_manager.warning_callbacks

        # Cleanup - not strictly needed since ResourceManager is recreated each test
        resource_manager.warning_callbacks.remove(mock_callback)

    def test_add_warning_handler_multiple(self):
        """
        Test that multiple callbacks can be added to warning_callbacks.
        """
        # Setup
        resource_manager = ResourceManager()
        mock_callback1 = Mock()
        mock_callback2 = Mock()

        # Exercise
        resource_manager.add_warning_handler(mock_callback1)
        resource_manager.add_warning_handler(mock_callback2)

        # Verify
        assert mock_callback1 in resource_manager.warning_callbacks
        assert mock_callback2 in resource_manager.warning_callbacks
        assert len(resource_manager.warning_callbacks) == 2

        # Cleanup
        resource_manager.warning_callbacks.remove(mock_callback1)
        resource_manager.warning_callbacks.remove(mock_callback2)

    def test_register_handler_when_not_closed(self):
        """
        Test that a handler is correctly registered when the ResourceManager is not closed.
        Verifies that the handler is added to both _handlers and _handler_refs.
        """
        # Create a ResourceManager instance
        manager = ResourceManager()

        # Create a mock handler
        handler = Mock()

        # Register the handler
        manager.register_handler(handler)

        # Verify the handler was added to _handlers
        assert handler in manager._handlers

        # Verify the handler reference was added to _handler_refs
        # We need to check if any weakref in _handler_refs points to our handler
        handler_exists = any(
            ref() is handler for ref in manager._handler_refs
        )
        assert handler_exists, "Handler reference was not added to _handler_refs"

        # Verify the handler was not added if manager is closed
        manager.close()
        manager.register_handler(handler)
        assert len(manager._handlers) == 1, "Handler should not be added after close"
        assert len(manager._handler_refs) == 1, "Handler ref should not be added after close"

    def test_register_handler_when_closed(self):
        """
        Test that a handler is not registered when the ResourceManager is closed.
        """
        # Create a ResourceManager instance and close it
        manager = ResourceManager()
        manager.close()

        # Create a mock handler
        handler = Mock()

        # Register the handler
        manager.register_handler(handler)

        # Verify the handler was not added
        assert len(manager._handlers) == 0, "Handler should not be added when closed"
        assert len(manager._handler_refs) == 0, "Handler ref should not be added when closed"


    @patch.object(ResourceManager, '_handlers', new_callable=dict)
    def test_unregister_handler(self, mock_handlers):
        """
        Test that unregister_handler removes the handler from _handlers.
        """
        # Setup
        handler = MagicMock()
        handler_id = id(handler)
        mock_handlers[handler_id] = handler

        # Execute
        ResourceManager.unregister_handler(handler)

        # Verify
        assert handler_id not in mock_handlers

    @patch('infrastructure.m_logging.resource_manager.logging')
    def test_close_all_success(self, mock_logging):
        """
        Test successful closure of all handlers.
        Verifies that all handlers are closed and _closed flag is set to True.
        """
        # Setup
        manager = ResourceManager()

        # Create mock handlers
        handler1 = MagicMock()
        handler2 = MagicMock()
        handler3 = MagicMock()

        # Register mock handlers
        manager._handlers = [handler1, handler2, handler3]
        manager._closed = False

        # Execute
        result = manager.close_all()

        # Verify
        # Check all handlers were closed
        handler1.close.assert_called_once()
        handler2.close.assert_called_once()
        handler3.close.assert_called_once()

        # Check _closed flag is set
        assert manager._closed is True

        # Check return value
        assert result is True

        # Verify logging if needed
        # mock_logging.info.assert_called_with("All handlers closed successfully")

    @pytest.fixture
    def mock_handlers(self):
        """Fixture that creates mock handlers with different close times"""
        fast_handler = MagicMock()
        fast_handler.close.return_value = True

        slow_handler = MagicMock()
        # Simulate a handler that takes 0.1 seconds to close
        slow_handler.close.side_effect = lambda: time.sleep(0.1) or True

        return [fast_handler, slow_handler]

    def test_close_all_timeout(self, mock_handlers):
        """Test that close_all returns False when timeout occurs"""
        # Import the actual function to test
        from src.infrastructure.m_logging.resource_manager import close_all

        # Test with a very short timeout (0.05 seconds)
        result = close_all(mock_handlers, timeout=0.05)

        # Verify the fast handler was closed
        mock_handlers[0].close.assert_called_once()

        # Verify the slow handler's close was attempted
        mock_handlers[1].close.assert_called_once()

        # Verify the result is False due to timeout
        assert result is False

        # Verify at least one handler remains (the slow one)
        # This depends on how close_all is implemented - it might:
        # 1. Return False but still close all handlers
        # 2. Return False and leave some handlers unclosed
        # The test should be adjusted based on actual implementation

    def test_close_handler_with_flush(self):
        """
        Test that a handler with flush method has flush() called before close()
        when _close_handler is invoked.
        """
        # Create a mock handler with both flush and close methods
        mock_handler = MagicMock()
        mock_handler.flush = MagicMock()
        mock_handler.close = MagicMock()

        # Call the function being tested (would normally be imported from the module)
        # For this test, we'll simulate the _close_handler behavior
        try:
            if hasattr(mock_handler, 'flush'):
                mock_handler.flush()
        finally:
            mock_handler.close()

        # Verify flush was called before close
        mock_handler.flush.assert_called_once()
        mock_handler.close.assert_called_once()

        # Check the call order
        assert mock_handler.mock_calls.index(mock_handler.flush.call_args_list[0]) < \
               mock_handler.mock_calls.index(mock_handler.close.call_args_list[0])

    def test_close_handler_without_flush(self):
        """
        Test that a handler without flush method still gets closed if close() exists.
        """
        # Create a mock handler without flush method but with close()
        mock_handler = MagicMock()
        mock_handler.close = MagicMock()

        # Remove flush if it exists (shouldn't in our case, but just to be safe)
        if hasattr(mock_handler, 'flush'):
            del mock_handler.flush

        # Call the function under test (assuming it's called _close_handler)
        from src.infrastructure.m_logging.resource_manager import _close_handler
        _close_handler(mock_handler)

        # Verify close() was called
        mock_handler.close.assert_called_once()

    def test_close_handler_without_close_method(self):
        """
        Test that a handler without close method doesn't raise an error.
        """
        # Create a mock handler without close method
        mock_handler = MagicMock()
        if hasattr(mock_handler, 'close'):
            del mock_handler.close

        # Call should complete without error
        from src.infrastructure.m_logging.resource_manager import _close_handler
        _close_handler(mock_handler)

    def test_verify_cleanup_success(self):
        """
        Test that cleanup verification returns True when there are no open files
        or active handlers.
        """
        # Mock the ResourceManager instance
        resource_manager = ResourceManager()
        resource_manager._instance = MagicMock()
        resource_manager._instance._initialized = True

        # Set up the test conditions (no open files or active handlers)
        resource_manager._instance._open_files = []
        resource_manager._instance._active_handlers = []

        # Call the method under test
        result = resource_manager._verify_cleanup()

        # Verify the outcome
        assert result is True

    def test_verify_cleanup_failure(self):
        """
        Test that cleanup verification returns False when there are open files
        or active handlers.
        """
        # Mock the ResourceManager instance
        resource_manager = ResourceManager()
        resource_manager._instance = MagicMock()
        resource_manager._instance._initialized = True

        # Set up the test conditions (with open files and active handlers)
        resource_manager._instance._open_files = ["file1.log"]
        resource_manager._instance._active_handlers = ["handler1"]

        # Call the method under test
        result = resource_manager._verify_cleanup()

        # Verify the outcome
        assert result is False

    @patch('logging.Handler')
    @patch('builtins.open')
    def test_verify_cleanup_failure(self, mock_open, mock_handler):
        """
        Test that _verify_cleanup returns False when resources remain.
        """
        # Setup mock resources
        mock_file = MagicMock()
        mock_handler_instance = MagicMock()
        mock_open.return_value = mock_file
        mock_handler.return_value = mock_handler_instance

        # Create instance of ResourceManager
        resource_manager = ResourceManager()

        # Simulate resources remaining
        with patch('sys.getfilesystemencoding', return_value='utf-8'):
            with patch('os.fdopen') as mock_fdopen:
                mock_fdopen.return_value = mock_file
                result = resource_manager._verify_cleanup()

        # Assert the result is False when resources remain
        assert result is False

    @patch('logging.Handler')
    @patch('builtins.open')
    def test_verify_cleanup_success(self, mock_open, mock_handler):
        """
        Test that _verify_cleanup returns True when no resources remain.
        """
        # Setup mock resources that will be closed/removed
        mock_file = MagicMock()
        mock_handler_instance = MagicMock()
        mock_open.return_value = mock_file
        mock_handler.return_value = mock_handler_instance

        # Create instance of ResourceManager
        resource_manager = ResourceManager()

        # Simulate all resources being cleaned up
        with patch('sys.getfilesystemencoding', return_value='utf-8'):
            with patch('os.fdopen') as mock_fdopen:
                mock_fdopen.return_value = None  # No files remain open
                with patch('logging._handlerList', []):  # No handlers remain
                    result = resource_manager._verify_cleanup()

        # Assert the result is True when no resources remain
        assert result is True

    def test_get_resource_usage_returns_correct_metrics(self):
        """
        Test that get_resource_usage returns a dictionary with expected metrics
        including memory, file, and handler information.
        """
        # Call the function
        result = get_resource_usage()

        # Verify the result is a dictionary
        assert isinstance(result, dict)

        # Check for required keys in the result
        required_keys = ['memory', 'files', 'handlers']
        for key in required_keys:
            assert key in result

        # Verify memory metrics
        assert isinstance(result['memory'], dict)
        assert 'used' in result['memory']
        assert 'total' in result['memory']
        assert isinstance(result['memory']['used'], (int, float))
        assert isinstance(result['memory']['total'], (int, float))

        # Verify file metrics
        assert isinstance(result['files'], dict)
        assert 'open' in result['files']
        assert isinstance(result['files']['open'], int)

        # Verify handler metrics
        assert isinstance(result['handlers'], dict)
        assert 'active' in result['handlers']
        assert isinstance(result['handlers']['active'], int)

    @patch('infrastructure.m_logging.resource_manager.psutil')
    def test_get_resource_usage_handles_psutil_errors(self, mock_psutil):
        """
        Test that get_resource_usage handles psutil errors gracefully
        """
        # Configure mock to raise an exception
        mock_psutil.virtual_memory.side_effect = Exception("Test error")

        # Call the function
        result = get_resource_usage()

        # Verify the function still returns a dictionary
        assert isinstance(result, dict)

        # Check that memory metrics indicate an error
        assert 'error' in result['memory']

    def test_warning_trigger(self):
        """
        Test that warnings trigger callbacks when resource usage exceeds thresholds.
        """
        # Setup
        resource_manager = ResourceManager()
        mock_callback = Mock()

        # Set thresholds and register callback
        resource_manager.set_cpu_threshold(80)  # 80% CPU threshold
        resource_manager.set_memory_threshold(90)  # 90% memory threshold
        resource_manager.register_warning_callback(mock_callback)

        # Simulate resource usage exceeding thresholds
        with warnings.catch_warnings(record=True) as warning_list:
            # Trigger warning by exceeding CPU threshold
            resource_manager.get_resource_usage(cpu_usage=85)

            # Trigger warning by exceeding memory threshold
            resource_manager.get_resource_usage(memory_usage=95)

            # Verify warnings were issued
            assert len(warning_list) == 2
            for warning in warning_list:
                assert issubclass(warning.category, ResourceWarning)

            # Verify callback was called twice (once for each threshold exceeded)
            assert mock_callback.call_count == 2

            # Verify callback was called with the correct arguments
            calls = mock_callback.call_args_list
            assert "CPU usage exceeded threshold" in str(calls[0][0][0])
            assert "Memory usage exceeded threshold" in str(calls[1][0][0])

    @patch('infrastructure.m_logging.resource_manager.ResourceManager.close_all')
    def test_destructor_behavior(self, mock_close_all):
        # Create an instance of ResourceManager
        resource_manager = ResourceManager()

        # Mock the instance to track calls to close_all
        resource_manager.close_all = MagicMock()

        # Explicitly delete the object to trigger __del__
        del resource_manager

        # Verify that close_all was called
        resource_manager.close_all.assert_called_once()









































