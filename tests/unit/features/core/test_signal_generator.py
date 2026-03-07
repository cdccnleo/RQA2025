import time
from types import SimpleNamespace

import pytest

import sys
import types


class _StubFpgaManager:
    def get_accelerator(self, name):
        return None


fpga_module = types.ModuleType("src.acceleration.fpga")
fpga_module.FpgaManager = _StubFpgaManager
sys.modules.setdefault("src.acceleration", types.ModuleType("src.acceleration"))
sys.modules["src.acceleration.fpga"] = fpga_module

from src.features.core.signal_generator import (
    ChinaSignalGenerator,
    SignalConfig,
    SignalGenerator,
)

class DummyFeatureEngine:
    def process(self, data):
        return data


class DummyAccelerator:
    def __init__(self, healthy=True):
        self.health_monitor = SimpleNamespace(is_healthy=lambda: healthy)


class DummyFpgaManager:
    def __init__(self, accelerator=None, raises=False):
        self.accelerator = accelerator
        self.raises = raises

    def get_accelerator(self, name):
        if self.raises:
            raise RuntimeError("fpga unavailable")
        return self.accelerator


@pytest.fixture
def signal_generator(monkeypatch):
    config = SignalConfig(use_fpga=False, min_confidence=0.6)
    gen = SignalGenerator(DummyFeatureEngine(), config=config)
    return gen


def test_generate_respects_cooldown(signal_generator):
    features = {"price": 100}
    signal = signal_generator.generate("AAA", features)
    assert signal["signal"] == "BUY"
    assert signal_generator.generate("AAA", features) is None


def test_generate_without_a_share_rules(monkeypatch):
    config = SignalConfig(a_share_rules=False, use_fpga=False)
    gen = SignalGenerator(DummyFeatureEngine(), config=config)
    assert gen.generate("ST001", {"price": 10}) is not None


def test_generate_uses_fpga_when_available(monkeypatch):
    accelerator = DummyAccelerator()
    manager = DummyFpgaManager(accelerator)
    config = SignalConfig(use_fpga=True)
    gen = SignalGenerator(DummyFeatureEngine(), config=config)
    monkeypatch.setattr(gen, "fpga_manager", manager)
    features = {"price": 50, "momentum": 0.3}
    signal = gen.generate("AAA", features)
    assert signal["signal"] == "BUY"
    assert signal["confidence"] == 0.8


def test_generate_handles_fpga_failure(monkeypatch):
    manager = DummyFpgaManager(raises=True)
    config = SignalConfig(use_fpga=True, min_confidence=0.9)
    gen = SignalGenerator(DummyFeatureEngine(), config=config)
    monkeypatch.setattr(gen, "fpga_manager", manager)
    signal = gen.generate("AAA", {"price": 100})
    assert signal["signal"] == "HOLD"


def test_generate_checks_a_share_restrictions(monkeypatch):
    config = SignalConfig(use_fpga=False, cool_down_period=0)
    gen = SignalGenerator(DummyFeatureEngine(), config=config)
    assert gen._check_a_share_restrictions("ST0001", "SELL") is False


def test_batch_generate_handles_exceptions(monkeypatch):
    gen = SignalGenerator(DummyFeatureEngine(), config=SignalConfig(use_fpga=False))
    results = iter([{"signal": "BUY"}, RuntimeError("boom")])

    def fake_generate(symbol, features):
        outcome = next(results)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gen, "generate", fake_generate)
    outputs = gen.batch_generate(["AAA", "BBB"], [{"price": 1}, {"price": 2}])
    assert outputs[0]["signal"] == "BUY"
    assert outputs[1] is None


def test_china_signal_generator_adjusts_position():
    config = SignalConfig(use_fpga=False, max_position=0.2)
    gen = ChinaSignalGenerator(DummyFeatureEngine(), config=config)
    features = {
        "price": 100,
        "margin_ratio": 0.6,
        "short_balance": 0.1,
        "institutional_net_buy": 0.9,
        "hot_money_flow": 0.5,
    }
    signal = gen.generate("600001", features)
    assert signal is not None
    assert signal["position"] <= config.max_position


def test_china_signal_generator_adjusts_stocks():
    config = SignalConfig(use_fpga=False, max_position=0.3)
    gen = ChinaSignalGenerator(DummyFeatureEngine(), config=config)
    features = {"price": 10, "is_st": True}
    signal = gen.generate("ST0001", features)
    assert signal["position"] <= config.max_position * 0.5

