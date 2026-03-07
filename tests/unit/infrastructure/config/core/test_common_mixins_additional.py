import time
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.config.core.common_mixins import (
    BatchOperationsMixin,
    ComponentLifecycleMixin,
    ConfigComponentMixin,
    CRUDOperationsMixin,
    MonitoringMixin,
)


class SampleComponent(ConfigComponentMixin):
    def __init__(self, **kwargs):
        self._init_component_attributes(**kwargs)


class MonitoringComponent(MonitoringMixin):
    pass


class CrudComponent(CRUDOperationsMixin):
    pass


class LifecycleComponent(ComponentLifecycleMixin):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _do_initialize(self):
        self.calls.append("init")

    def _do_start(self):
        self.calls.append("start")

    def _do_stop(self):
        self.calls.append("stop")


class BatchComponent(BatchOperationsMixin):
    def __init__(self):
        super().__init__()
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


def test_config_component_mixin_attribute_flags():
    component = SampleComponent(
        enable_threading=True,
        enable_config=True,
        enable_metrics=True,
        enable_alerts=True,
        enable_history=True,
        enable_data=True,
        config={"foo": "bar"},
    )

    assert hasattr(component, "_lock")
    assert component._config == {"foo": "bar"}
    assert component._metrics == {}
    assert component._alerts == []
    assert component._history == []
    assert component._data == {}


def test_monitoring_mixin_records_and_trims_metrics(monkeypatch):
    component = MonitoringComponent()
    base_time = time.time()
    monkeypatch.setattr("time.time", lambda: base_time)

    for idx in range(1200):
        component.record_metric("latency", idx, timestamp=base_time + idx)

    latest = component.get_latest_metric("latency")
    assert latest["value"] == 1199
    assert len(component._metrics["latency"]) <= 1000
    assert component._metrics["latency"][0]["value"] >= 500


def test_crud_operations_mixin_record_history():
    component = CrudComponent()
    component.create("a", 1)
    assert component.read("a") == 1

    assert component.update("a", 2) is True
    assert component.update("missing", 3) is False

    assert component.delete("a") is True
    assert component.delete("missing") is False

    assert len(component._history) == 3
    assert component._history[0]["operation"] == "create"
    assert component._history[-1]["operation"] == "delete"


def test_component_lifecycle_mixin_flow():
    component = LifecycleComponent()
    component.start()

    assert component.is_initialized is True
    assert component.is_started is True

    component.stop()
    assert component.is_stopped is True

    component.restart()
    assert component.is_started is True
    assert component.calls.count("init") == 1
    assert component.calls.count("start") >= 2
    assert component.calls.count("stop") >= 1


def test_batch_operations_mixin_success_and_failure():
    component = BatchComponent()
    assert component.batch_get(["missing"]) == {"missing": None}

    assert component.batch_set({"a": 1, "b": 2}) is True
    assert component.batch_get(["a", "b"]) == {"a": 1, "b": 2}

    component._logger = MagicMock()

    class FailingBatch(BatchComponent):
        def set(self, key, value):
            raise RuntimeError("boom")

    failing = FailingBatch()
    failing._logger = MagicMock()
    assert failing.batch_set({"a": 1}) is False
    failing._logger.error.assert_called_once()

