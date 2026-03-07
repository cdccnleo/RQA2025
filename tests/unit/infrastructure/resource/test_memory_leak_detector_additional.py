import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.utils.memory_leak_detector import MemoryLeakDetector


@pytest.fixture
def detector():
    logger = MagicMock()
    error_handler = MagicMock()
    leak_detector = MemoryLeakDetector(logger=logger, error_handler=error_handler)
    return leak_detector, logger, error_handler


@pytest.fixture
def mock_psutil(monkeypatch):
    process = MagicMock()
    process.memory_info.return_value = MagicMock(rss=200 * 1024**2, vms=450 * 1024**2)
    process.memory_percent.return_value = 15.5

    virtual = MagicMock(
        total=4096 * 1024**2,
        available=2048 * 1024**2,
        used=2048 * 1024**2,
        percent=50.0,
    )
    monkeypatch.setattr("psutil.Process", MagicMock(return_value=process))
    monkeypatch.setattr("psutil.virtual_memory", MagicMock(return_value=virtual))
    return process, virtual


@pytest.fixture
def mock_gc(monkeypatch):
    monkeypatch.setattr("gc.collect", MagicMock(return_value=0))
    monkeypatch.setattr("gc.get_objects", MagicMock(return_value=[]))


def test_detect_memory_leaks_no_issues(detector, mock_psutil, mock_gc):
    leak_detector, logger, error_handler = detector
    issues = leak_detector.detect_memory_leaks()
    assert issues == []
    logger.log_info.assert_called_once()
    error_handler.handle_error.assert_not_called()


def test_detect_memory_leaks_trend(monkeypatch, detector, mock_psutil):
    leak_detector, logger, _ = detector

    leak_detector._memory_history = [
        {"timestamp": datetime.now() - timedelta(minutes=i), "memory_mb": value}
        for i, value in enumerate([80, 90, 100, 110], start=4)
    ]
    process, _ = mock_psutil
    process.memory_info.return_value = MagicMock(rss=180 * 1024**2, vms=180 * 1024**2)

    issues = leak_detector.detect_memory_leaks()
    assert any("内存使用快速增长" in issue for issue in issues)
    logger.log_warning.assert_called()


def test_detect_memory_leaks_object_references(monkeypatch, detector, mock_psutil):
    leak_detector, logger, _ = detector
    objects = [list(range(10)) for _ in range(10050)]
    monkeypatch.setattr("gc.get_objects", lambda: objects)
    issues = leak_detector.detect_memory_leaks()
    assert any("对象类型 'list' 数量异常" in issue for issue in issues)


def test_detect_memory_leaks_circular(monkeypatch, detector, mock_psutil):
    leak_detector, logger, _ = detector
    monkeypatch.setattr("gc.collect", MagicMock(return_value=7))
    issues = leak_detector.detect_memory_leaks()
    assert any("循环引用对象" in issue for issue in issues)


def test_detect_memory_leaks_large_objects(monkeypatch, detector, mock_psutil):
    leak_detector, logger, _ = detector

    class LargeObject:
        def __init__(self, size):
            self.data = bytearray(size)

    monkeypatch.setattr("gc.get_objects", lambda: [bytearray(2 * 1024**2)])
    issues = leak_detector.detect_memory_leaks()
    assert any("大对象检测" in issue for issue in issues)


def test_detect_memory_leaks_error(detector, mock_psutil, monkeypatch):
    leak_detector, logger, error_handler = detector
    monkeypatch.setattr(leak_detector, "_check_memory_trend", MagicMock(side_effect=RuntimeError("trend fail")))
    monkeypatch.setattr("gc.get_objects", lambda: [])
    issues = leak_detector.detect_memory_leaks()
    assert issues == ["检测失败: trend fail"]
    error_handler.handle_error.assert_called_once()


def test_get_memory_report(detector, mock_psutil, mock_gc):
    leak_detector, logger, _ = detector
    report = leak_detector.get_memory_report()
    assert report["process_memory"]["percent"] == 15.5
    assert isinstance(report["issues"], list)


def test_get_memory_report_error(detector, mock_psutil, monkeypatch):
    leak_detector, logger, error_handler = detector
    monkeypatch.setattr("psutil.Process", MagicMock(side_effect=RuntimeError("process fail")))
    report = leak_detector.get_memory_report()
    assert report["error"] == "process fail"
    error_handler.handle_error.assert_called_once()

