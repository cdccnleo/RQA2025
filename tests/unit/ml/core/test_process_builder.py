"""ML流程构建器单元测试，覆盖模板构建、步骤配置与提交流程。"""

import pytest

import sys
from pathlib import Path

from src.ml.core.process_builder import (
    MLProcessBuilder,
    ProcessPriority,
)

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DummyOrchestrator:
    def __init__(self):
        self.callback_records = []
        self.submitted_processes = []

    def add_process_callback(self, process_id, event, callback):
        self.callback_records.append((process_id, event, callback))

    def submit_process(self, process):
        if not process.process_id:
            process.process_id = "generated-process-id"
        self.submitted_processes.append(process)
        return process.process_id


@pytest.fixture
def builder(monkeypatch):
    orchestrator = DummyOrchestrator()

    import src.ml.core.process_builder as process_builder

    monkeypatch.setattr(
        process_builder, "get_ml_process_orchestrator", lambda: orchestrator
    )
    monkeypatch.setattr(
        process_builder,
        "register_ml_step_executors",
        lambda *args, **kwargs: None,
    )

    builder_instance = process_builder.MLProcessBuilder()
    return builder_instance, orchestrator


def test_from_template_populates_steps_and_config(builder):
    process_builder, _ = builder
    process_builder.from_template("basic_training")

    assert set(process_builder.steps.keys()) == {
        "load_data",
        "feature_engineering",
        "train_model",
        "evaluate_model",
    }
    assert process_builder.process_config["name"] == "基础模型训练流程"
    assert process_builder.process_config["template"] == "basic_training"


def test_from_template_applies_custom_config(builder):
    process_builder, _ = builder
    custom_config = {
        "priority": ProcessPriority.HIGH,
        "config": {"artifact_path": "/tmp/model"},
        "metadata": {"owner": "unit-test"},
    }

    process_builder.from_template("batch_prediction", custom_config)

    assert process_builder.process_config["priority"] == ProcessPriority.HIGH
    assert process_builder.process_config["config"]["artifact_path"] == "/tmp/model"
    assert process_builder.process_config["metadata"]["owner"] == "unit-test"


def test_add_and_configure_step(builder):
    process_builder, _ = builder
    process_builder.add_step("s1", "Step 1", "custom")
    process_builder.configure_step("s1", {"param": 42})

    assert "s1" in process_builder.steps
    assert process_builder.steps["s1"].config["param"] == 42


def test_build_and_submit_registers_callbacks(builder):
    process_builder, orchestrator = builder
    process_builder.add_step("prepare", "Prepare data", "data_loading")
    process_builder.set_priority(ProcessPriority.CRITICAL)
    process_builder.set_timeout(180)

    callback = object()
    process_builder.add_callback("on_complete", callback)

    process_id = process_builder.build_and_submit()

    assert process_id == "generated-process-id"
    assert orchestrator.submitted_processes
    submitted_process = orchestrator.submitted_processes[0]
    assert submitted_process.steps.keys() == {"prepare"}
    assert ("", "on_complete", callback) in orchestrator.callback_records


def test_build_without_steps_raises(builder):
    process_builder, _ = builder
    process_builder.reset()
    with pytest.raises(ValueError):
        process_builder.build()

