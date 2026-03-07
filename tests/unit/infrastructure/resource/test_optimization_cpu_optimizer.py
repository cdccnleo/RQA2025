import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.optimization_cpu_optimizer import CpuOptimizer
from src.infrastructure.resource.core.optimization_config import CpuOptimizationConfig


@pytest.fixture
def optimizer():
    logger = MagicMock()
    error_handler = MagicMock()
    optimizer = CpuOptimizer(logger=logger, error_handler=error_handler)
    return optimizer, logger, error_handler


def test_optimize_cpu_defaults(optimizer):
    opt, _, _ = optimizer
    result = opt.optimize_cpu({}, {"cpu_usage": 30})
    assert result["status"] == "applied"
    assert result["actions"] == []


def test_optimize_cpu_affinity_and_power_saving(optimizer):
    opt, _, _ = optimizer
    config = {
        "cpu_affinity": {"enabled": True},
        "power_saving": True,
        "priority_threshold": 80,
    }
    resources = {"cpu_usage": 75}
    result = opt.optimize_cpu(config, resources)
    assert "配置CPU亲和性" in result["actions"]
    assert "启用CPU节能模式" in result["actions"]


def test_optimize_cpu_adjust_priority(optimizer):
    opt, _, _ = optimizer
    config = {"priority_threshold": 60}
    resources = {"cpu_usage": 75}
    result = opt.optimize_cpu(config, resources)
    assert "调整进程优先级" in result["actions"]


def test_optimize_cpu_from_config_dataclass(optimizer, monkeypatch):
    opt, _, _ = optimizer
    monkeypatch.setattr(opt, "optimize_cpu", MagicMock(return_value={"status": "applied"}))
    config = CpuOptimizationConfig(
        enabled=True,
        cpu_affinity={"enabled": True},
        power_saving=True,
        priority_threshold=70,
    )
    result = opt.optimize_cpu_from_config(config, {"cpu_usage": 75})
    opt.optimize_cpu.assert_called_once()
    assert result["status"] == "applied"


def test_optimize_cpu_error_path(optimizer):
    opt, _, error_handler = optimizer

    class BadDict(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("cpu fail")

    result = opt.optimize_cpu(BadDict(), {"cpu_usage": 80})
    error_handler.handle_error.assert_called_once()
    assert result["status"] == "failed"
    assert result["error"] == "cpu fail"

