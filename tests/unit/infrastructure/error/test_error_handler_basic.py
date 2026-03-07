from src.infrastructure.error.error_handler import ErrorHandler


def test_basic_error_handler_stores_errors():
    handler = ErrorHandler()
    err1 = ValueError("bad value")
    err2 = RuntimeError("boom")

    assert handler.handle_error(err1) is True
    assert handler.handle_error(err2) is True

    errors = handler.get_errors()
    assert errors == [err1, err2]

