import pytest
from unittest.mock import patch
from src.infrastructure.utils.exception_utils import ExceptionHandler, is_timeout_error

class TestExceptionUtilsEnhanced:

    def test_handle_empty_context(self):
        handler = ExceptionHandler()
        handler.handle(ValueError("test"), context={})

    def test_identify_timeout_exception(self):
        assert is_timeout_error(TimeoutError())
        assert not is_timeout_error(ValueError())

    def test_log_context_handling(self):
        handler = ExceptionHandler()
        with patch('logging.Logger.error') as mock_error:
            handler.handle(
                RuntimeError("test"),
                context={'module': 'test'},
                log_level='INFO'
            )
            call_args = mock_error.call_args
            assert call_args is not None
            extra = call_args[1].get('extra') if call_args and call_args[1] else None
            assert extra is not None
            assert 'module' not in extra  # 'module'被过滤，不应出现在extra中
