import types
import importlib
from unittest.mock import MagicMock
import pytest

from src.features.core.dependency_manager import (
    DependencyManager,
    get_transformers_pipeline,
    get_torch_device,
    is_gpu_available,
    get_gpu_count,
    dependency_manager
)


def _set_optional(monkeypatch, deps):
    monkeypatch.setattr(DependencyManager, "OPTIONAL_DEPENDENCIES", deps)


def test_dependency_available_when_version_ok(monkeypatch):
    _set_optional(
        monkeypatch,
        {"torch": {"min_version": "1.9.0", "purpose": "", "fallback": ""}},
    )
    torch_module = types.SimpleNamespace(__version__="1.10.0")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: torch_module,
    )

    manager = DependencyManager()
    info = manager.get_dependency_info()["torch"]

    assert info["available"] is True
    assert info["module"] is torch_module
    assert info["version"] == "1.10.0"


def test_dependency_rejected_when_version_low(monkeypatch):
    _set_optional(
        monkeypatch,
        {"transformers": {"min_version": "4.0.0", "purpose": "", "fallback": ""}},
    )
    transformers_module = types.SimpleNamespace(__version__="3.9.9")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: transformers_module,
    )

    manager = DependencyManager()
    info = manager.get_dependency_info()["transformers"]

    assert info["available"] is False
    assert "低于要求" in info["error"]
    assert info["module"] is None


def test_dependency_missing_sets_error(monkeypatch):
    _set_optional(monkeypatch, {"datasets": {"min_version": "2.0.0", "purpose": "", "fallback": ""}})
    def _raise(_):
        raise ImportError("No module named datasets")

    monkeypatch.setattr(
        importlib,
        "import_module",
        _raise,
    )

    manager = DependencyManager()
    info = manager.get_dependency_info()["datasets"]

    assert info["available"] is False
    assert "No module" in info["error"]


@pytest.mark.parametrize(
    "current,min_version,expected",
    [
        ("1.0.0", "1.0.0", True),
        ("1.2.0", "1.1.0", True),
        ("1.0.0", "1.1.0", False),
        ("unknown", "1.0.0", False),
        ("invalid", "1.0.0", False),
    ],
)
def test_version_satisfies_handles_edge_cases(monkeypatch, current, min_version, expected):
    _set_optional(monkeypatch, {"torch": {"min_version": min_version, "purpose": "", "fallback": ""}})

    module = types.SimpleNamespace()
    if current != "missing":
        module.__version__ = current

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module,
    )

    manager = DependencyManager()
    assert manager._version_satisfies(current, min_version) is expected


def test_safe_import_returns_fallback_when_provided(monkeypatch):
    _set_optional(monkeypatch, {"torch": {"min_version": "1.0.0", "purpose": "", "fallback": ""}})
    def _raise(_):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _raise)
    manager = DependencyManager()
    fallback = object()

    result = manager.safe_import("torch", fallback=fallback)

    assert result is fallback


def test_safe_import_returns_mock_when_missing(monkeypatch):
    _set_optional(monkeypatch, {"torch": {"min_version": "1.0.0", "purpose": "", "fallback": ""}})
    def _raise(_):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _raise)
    manager = DependencyManager()

    result = manager.safe_import("torch")

    assert isinstance(result, MagicMock)


def test_get_dependency_info_returns_copy(monkeypatch):
    _set_optional(monkeypatch, {"torch": {"min_version": "1.0.0", "purpose": "", "fallback": ""}})
    def _raise(_):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _raise)
    manager = DependencyManager()

    info = manager.get_dependency_info()
    info.pop("torch")

    assert "torch" in manager.get_dependency_info()


def test_log_dependency_status_outputs_to_logger(monkeypatch, caplog):
    _set_optional(monkeypatch, {"torch": {"min_version": "1.0.0", "purpose": "", "fallback": ""}})
    def _raise(_):
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _raise)
    manager = DependencyManager()

    with caplog.at_level("INFO"):
        manager.log_dependency_status()

    assert "特征层依赖状态" in caplog.text
    assert "❌" in caplog.text


def test_get_transformers_pipeline_uses_real_pipeline(monkeypatch):
    called = {}

    class StubManager:
        def safe_import(self, name):
            assert name == "transformers"

            class StubTransformers:
                @staticmethod
                def pipeline(task):
                    called["task"] = task
                    return f"pipeline:{task}"

            return StubTransformers()

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    pipeline = get_transformers_pipeline("sentiment-analysis")

    assert pipeline == "pipeline:sentiment-analysis"
    assert called["task"] == "sentiment-analysis"


def test_get_transformers_pipeline_fallback_when_missing(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return object()

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    pipeline = get_transformers_pipeline("sentiment-analysis")

    assert isinstance(pipeline(), list)
    assert pipeline()[0]["label"] == "POSITIVE"


def test_get_transformers_pipeline_fallback_on_exception(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            class StubTransformers:
                @staticmethod
                def pipeline(task):
                    raise RuntimeError("boom")

            return StubTransformers()

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    pipeline = get_transformers_pipeline("sentiment-analysis")

    assert isinstance(pipeline(), list)
    assert pipeline()[0]["score"] == 0.9


def _torch_stub(available=True, count=2):
    class CudaStub:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def device_count():
            return count

    return types.SimpleNamespace(cuda=CudaStub(), device=lambda name: f"device:{name}")


def test_get_torch_device_returns_cuda(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return _torch_stub(available=True)

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    device = get_torch_device()
    assert device == "device:cuda"


def test_get_torch_device_returns_none_without_cuda(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return _torch_stub(available=False)

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    assert get_torch_device() is None


def test_is_gpu_available(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return _torch_stub(available=True)

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    assert is_gpu_available() is True


def test_is_gpu_unavailable_when_no_cuda(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return _torch_stub(available=False)

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    assert is_gpu_available() is False


def test_get_gpu_count(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return _torch_stub(available=True, count=4)

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    assert get_gpu_count() == 4


def test_get_gpu_count_returns_zero_without_cuda(monkeypatch):
    class StubManager:
        def safe_import(self, name):
            return object()

    monkeypatch.setattr(dependency_manager, "safe_import", StubManager().safe_import)

    assert get_gpu_count() == 0

