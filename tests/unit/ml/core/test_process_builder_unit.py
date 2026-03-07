from typing import Callable, List, Tuple

import pytest

import sys
from pathlib import Path

from src.ml.core.process_builder import (
    MLProcessBuilder,
    ProcessPriority,
    register_ml_step_executors,
)

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from src.ml.core.process_orchestrator import MLProcessType


class StubOrchestrator:
    def __init__(self):
        self.callbacks: List[Tuple[str, str, Callable]] = []
        self.submitted_processes = []

    def add_process_callback(self, process_id: str, event: str, callback: Callable):
        self.callbacks.append((process_id, event, callback))

    def submit_process(self, process):
        self.submitted_processes.append(process)
        process.process_id = process.process_id or f"{process.process_type.value}-stub"
        return process.process_id


@pytest.fixture
def builder():
    orchestrator = StubOrchestrator()
    builder = MLProcessBuilder()
    builder.orchestrator = orchestrator  # 直接设置orchestrator
    return builder, orchestrator


def test_from_template_populates_steps_and_config(builder):
    process_builder, _ = builder

    process_builder.from_template(
        "basic_training",
        config={
            "config": {"batch_size": 64},
            "metadata": {"owner": "ml-team"},
            "priority": ProcessPriority.HIGH,
            "timeout": 600,
        },
    )

    assert set(process_builder.steps.keys()) == {
        "load_data",
        "feature_engineering",
        "train_model",
        "evaluate_model",
    }
    feature_step = process_builder.steps["feature_engineering"]
    assert feature_step.dependencies == ["load_data"]
    assert (
        process_builder.process_config["config"]["batch_size"] == 64
    ), "流程自定义配置应被合并"
    assert (
        process_builder.process_config["metadata"]["owner"] == "ml-team"
    ), "元数据应被保留"
    assert (
        process_builder.process_config["priority"] == ProcessPriority.HIGH
    ), "额外字段应直接写入流程配置"
    assert process_builder.process_config["timeout"] == 600


def test_from_template_unknown_raises(builder):
    process_builder, _ = builder
    with pytest.raises(ValueError, match="未找到流程模板"):
        process_builder.from_template("missing_template")


def test_build_requires_at_least_one_step(builder):
    process_builder, _ = builder
    with pytest.raises(ValueError):
        process_builder.build()


def test_configure_individual_step_and_process(builder):
    process_builder, _ = builder

    process_builder.add_step("prepare", "Prepare", "prep")
    with pytest.raises(ValueError):
        process_builder.configure_step("absent", {"foo": "bar"})

    process_builder.configure_step("prepare", {"foo": "bar"})
    process_builder.configure_process({"threshold": 0.9})
    process_builder.set_priority(ProcessPriority.CRITICAL)
    process_builder.set_timeout(240)

    process = process_builder.build()
    assert process.steps["prepare"].config["foo"] == "bar"
    assert process.config["threshold"] == 0.9
    assert process.priority == ProcessPriority.CRITICAL
    assert process.timeout == 240


def test_build_and_submit_registers_callbacks(builder):
    process_builder, orchestrator = builder

    process_builder.add_step("only_step", "Only", "single")
    process_builder.add_callback("on_start", lambda *args: None)
    process_builder.add_callback("on_complete", lambda *args: None)

    returned_id = process_builder.build_and_submit()

    assert len(orchestrator.callbacks) == 2
    recorded_events = [event for _, event, _ in orchestrator.callbacks]
    assert recorded_events == ["on_start", "on_complete"]
    assert orchestrator.submitted_processes, "流程应被提交到编排器"
    submitted = orchestrator.submitted_processes[0]
    assert submitted.steps["only_step"].step_name == "Only"
    assert submitted.process_type == MLProcessType.MODEL_TRAINING
    assert returned_id == submitted.process_id
    assert returned_id.endswith("-stub")


def test_register_ml_step_executors_noop():
    assert register_ml_step_executors("anything") is None

