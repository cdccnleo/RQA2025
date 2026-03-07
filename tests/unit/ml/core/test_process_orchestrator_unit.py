import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest

import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core.process_orchestrator import (
    MLProcess,
    MLProcessOrchestrator,
    MLProcessType,
    ProcessPriority,
    ProcessStatus,
    ProcessStep,
    cancel_ml_process,
    create_ml_process,
    get_ml_orchestrator_stats,
    get_ml_process_status,
    get_ml_process_orchestrator,
    submit_ml_process,
)


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("ml.core.process_orchestrator.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield


class RecordingExecutor:
    def __init__(self, label: str, record: List[Tuple[str, str]], exc: Exception = None):
        self.label = label
        self.record = record
        self.exc = exc

    def validate(self, step: ProcessStep) -> bool:
        self.record.append(("validate", self.label))
        return True

    def execute(self, step: ProcessStep, context: Dict[str, Dict]):
        self.record.append(("execute", self.label))
        if self.exc:
            raise self.exc
        return {"step": step.step_id, "label": self.label}


@pytest.fixture
def orchestrator(monkeypatch):
    captured: List[Tuple[Exception, Dict]] = []
    monkeypatch.setattr(
        "src.ml.core.process_orchestrator.handle_ml_error",
        lambda exc, ctx: captured.append((exc, ctx)),
    )
    orch = MLProcessOrchestrator(max_workers=1, queue_size=8)
    orch.stats = {
        "total_processes": 0,
        "completed_processes": 0,
        "failed_processes": 0,
        "avg_process_time": 0.0,
        "active_workers": 0,
    }
    return orch, captured


def _make_process(process_id: str, steps: Dict[str, ProcessStep]) -> MLProcess:
    return MLProcess(
        process_id=process_id,
        process_type=MLProcessType.MODEL_TRAINING,
        process_name="test-process",
        priority=ProcessPriority.NORMAL,
        steps=steps,
    )


def test_execute_process_success_updates_stats(orchestrator):
    orch, captured = orchestrator
    events: List[Tuple[str, str]] = []
    call_log: List[Tuple[str, str]] = []

    orch.register_step_executor("load", RecordingExecutor("load", call_log))
    orch.register_step_executor("train", RecordingExecutor("train", call_log))

    steps = {
        "load": ProcessStep("load", "Load", "load"),
        "train": ProcessStep("train", "Train", "train", dependencies=["load"]),
    }
    process = _make_process("success", steps)
    process.callbacks = {
        "on_start": [lambda proc: events.append(("start", proc.process_id))],
        "on_complete": [lambda proc: events.append(("complete", proc.process_id))],
    }
    orch.active_processes[process.process_id] = process

    orch._execute_process(process)

    assert captured == []
    assert process.status is ProcessStatus.COMPLETED
    assert process.progress == 1.0
    assert orch.stats["completed_processes"] == 1
    assert events == [("start", "success"), ("complete", "success")]
    assert call_log == [
        ("validate", "load"),
        ("execute", "load"),
        ("validate", "train"),
        ("execute", "train"),
    ]


def test_execute_process_failure_records_error(orchestrator):
    orch, captured = orchestrator
    orch.register_step_executor("fail", RecordingExecutor("fail", [], RuntimeError("boom")))

    steps = {"fail": ProcessStep("fail", "Fail step", "fail")}
    process = _make_process("failure", steps)
    process.callbacks = {
        "on_start": [lambda proc: None],
        "on_fail": [lambda proc: process.metadata.setdefault("callback", "hit")],
    }
    orch.active_processes[process.process_id] = process

    orch._execute_process(process)

    assert process.status is ProcessStatus.FAILED
    assert process.metadata.get("error") is None
    assert process.metadata["callback"] == "hit"
    assert orch.stats["failed_processes"] == 1
    assert process.process_id in orch.completed_processes
    assert captured == []
    assert process.steps["fail"].status is ProcessStatus.FAILED


def test_execute_process_steps_detects_dependency_cycle(orchestrator):
    orch, _ = orchestrator
    steps = {
        "a": ProcessStep("a", "A", "noop", dependencies=["b"]),
        "b": ProcessStep("b", "B", "noop", dependencies=["a"]),
    }
    process = _make_process("cycle", steps)
    assert orch._execute_process_steps(process) is False


def test_submit_process_enqueues_when_running(orchestrator):
    orch, _ = orchestrator
    recorded = []

    class DummyQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)
            recorded.append(item)

        def qsize(self):
            return len(self.items)

    orch.process_queue = DummyQueue()
    steps = {"s": ProcessStep("s", "Single", "noop")}
    proc = _make_process("", steps)

    orch.running = True
    process_id = orch.submit_process(proc)

    assert process_id in orch.active_processes
    assert orch.stats["total_processes"] == 1
    assert proc.status is ProcessStatus.QUEUED
    assert recorded == [(-ProcessPriority.NORMAL.value, process_id)]

    orch.running = False
    with pytest.raises(RuntimeError):
        orch.submit_process(proc)


def test_stop_noop_when_not_running(orchestrator):
    orch, _ = orchestrator
    orch.running = False
    called = {"invoked": False}
    orch.executor_pool = SimpleNamespace(
        shutdown=lambda wait, timeout: called.update({"invoked": True})
    )

    orch.stop(timeout=0)

    assert orch.running is False
    assert not called["invoked"]


def test_start_warns_when_already_running(orchestrator, caplog):
    orch, _ = orchestrator
    orch.running = True
    with caplog.at_level("WARNING"):
        orch.start()
    assert "编排器已在运行" in caplog.text


def test_start_initializes_workers_and_resets_event(monkeypatch):
    created = []
    started = []

    class FakeThread:
        def __init__(self, target, name, daemon):
            self._target = target
            self.name = name
            self.daemon = daemon
            created.append(name)

        def start(self):
            started.append(self.name)

    monkeypatch.setattr("src.ml.core.process_orchestrator.threading.Thread", FakeThread)
    orch = MLProcessOrchestrator(max_workers=2, queue_size=2)
    orch.running = False
    orch.shutdown_event.set()

    orch.start()

    assert orch.running is True
    assert orch.shutdown_event.is_set() is False
    assert created == ["MLProcessWorker-1", "MLProcessWorker-2"]
    assert started == ["MLProcessWorker-1", "MLProcessWorker-2"]


def test_stop_sets_shutdown_flag_and_invokes_executor(orchestrator):
    orch, _ = orchestrator
    orch.running = True
    called = {"wait_args": None}

    class DummyPool:
        def shutdown(self, wait=True):
            called["wait_args"] = wait

    orch.executor_pool = DummyPool()

    orch.stop(timeout=0)

    assert orch.running is False
    assert orch.shutdown_event.is_set() is True
    assert called["wait_args"] is True


def test_get_process_status_summary(orchestrator):
    orch, _ = orchestrator
    steps = {
        "done": ProcessStep("done", "Done", "noop"),
        "failed": ProcessStep("failed", "Failed", "noop"),
    }
    steps["done"].status = ProcessStatus.COMPLETED
    steps["failed"].status = ProcessStatus.FAILED
    proc = _make_process("status", steps)
    proc.status = ProcessStatus.RUNNING
    orch.active_processes[proc.process_id] = proc

    snapshot = orch.get_process_status("status")

    assert snapshot["process_id"] == "status"
    assert snapshot["completed_steps"] == 1
    assert snapshot["failed_steps"] == 1
    assert snapshot["status"] == ProcessStatus.RUNNING.value


def test_get_process_status_returns_none_when_missing(orchestrator):
    orch, _ = orchestrator
    assert orch.get_process_status("absent") is None


def test_register_default_executors_is_idempotent(orchestrator):
    orch, _ = orchestrator
    orch._register_default_executors()


def test_pause_and_resume_process(orchestrator):
    orch, _ = orchestrator
    steps = {"only": ProcessStep("only", "Only", "noop")}
    proc = _make_process("to_pause", steps)
    proc.status = ProcessStatus.RUNNING
    orch.active_processes[proc.process_id] = proc

    assert orch.pause_process(proc.process_id) is True
    assert proc.status == ProcessStatus.PAUSED
    assert orch.pause_process("missing") is False

    assert orch.resume_process(proc.process_id) is True
    assert proc.status == ProcessStatus.RUNNING
    assert orch.resume_process("missing") is False


def test_add_process_callback_registers_and_triggers(orchestrator):
    orch, _ = orchestrator
    steps = {"only": ProcessStep("only", "Only", "noop")}
    proc = _make_process("with_callback", steps)
    orch.active_processes[proc.process_id] = proc

    recorded = []
    orch.add_process_callback(proc.process_id, "on_complete", lambda p: recorded.append(p.process_id))

    orch._trigger_callbacks(proc, "on_complete")
    assert recorded == [proc.process_id]


def test_trigger_callbacks_logs_exceptions(orchestrator, caplog):
    orch, _ = orchestrator
    steps = {"only": ProcessStep("only", "Only", "noop")}
    proc = _make_process("err_cb", steps)
    proc.callbacks = {
        "on_complete": [
            lambda p: (_ for _ in ()).throw(RuntimeError("callback boom"))
        ]
    }

    with caplog.at_level("ERROR"):
        orch._trigger_callbacks(proc, "on_complete")

    assert "回调执行失败" in caplog.text


def test_process_worker_logs_exception_and_exits(orchestrator, caplog):
    orch, _ = orchestrator

    class ErrorQueue:
        def get(self, timeout):
            raise RuntimeError("queue boom")

        def task_done(self):
            pass

    orch.process_queue = ErrorQueue()

    with caplog.at_level("ERROR"):
        worker = threading.Thread(target=orch._process_worker, daemon=True)
        worker.start()
        time.sleep(0.05)
        orch.shutdown_event.set()
        worker.join(timeout=1)

    assert "流程处理线程异常" in caplog.text


def test_process_worker_executes_active_process(orchestrator):
    orch, _ = orchestrator
    proc = _make_process("queued", {})
    orch.active_processes[proc.process_id] = proc
    executed = []

    def fake_execute(process):
        executed.append(process.process_id)
        orch.shutdown_event.set()

    orch._execute_process = fake_execute

    class SingleQueue:
        def __init__(self):
            self._returned = False

        def get(self, timeout):
            if not self._returned:
                self._returned = True
                return (-ProcessPriority.NORMAL.value, proc.process_id)
            raise RuntimeError("should not fetch again")

        def task_done(self):
            executed.append("task_done")

    orch.process_queue = SingleQueue()
    worker = threading.Thread(target=orch._process_worker, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert executed == ["queued", "task_done"]


def test_execute_step_records_metrics_and_context(orchestrator):
    orch, _ = orchestrator
    call_log: List[Tuple[str, str]] = []

    class DummyExecutor:
        def validate(self, step):
            call_log.append(("validate", step.step_id))
            return True

        def execute(self, step, context):
            call_log.append(("execute", context["step"].step_id))
            assert context["process"].process_id == "exec"
            assert context["metadata"] is context["process"].metadata
            return {"ok": True}

    orch.register_step_executor("dummy", DummyExecutor())
    step = ProcessStep("dummy", "Dummy", "dummy")
    process = _make_process("exec", {"dummy": step})

    result = orch._execute_step(step, process)

    assert result == {"ok": True}
    assert step.status is ProcessStatus.COMPLETED
    assert "execution_time" in step.metrics
    assert call_log == [("validate", "dummy"), ("execute", "dummy")]


def test_execute_step_raises_when_executor_missing(orchestrator):
    orch, _ = orchestrator
    step = ProcessStep("missing", "Missing", "unknown")
    process = _make_process("proc", {"missing": step})

    with pytest.raises(ValueError) as exc:
        orch._execute_step(step, process)
    assert "未找到步骤执行器" in str(exc.value)


def test_execute_step_raises_when_validation_fails(orchestrator):
    orch, _ = orchestrator

    class InvalidExecutor:
        def validate(self, step):
            return False

        def execute(self, step, context):
            raise AssertionError("should not run")

    orch.register_step_executor("bad", InvalidExecutor())
    step = ProcessStep("bad", "Bad", "bad")
    process = _make_process("validator", {"bad": step})

    with pytest.raises(ValueError) as exc:
        orch._execute_step(step, process)
    assert "步骤配置验证失败" in str(exc.value)


def test_cancel_process_moves_to_completed(orchestrator):
    orch, _ = orchestrator
    proc = _make_process("cancel-me", {"only": ProcessStep("only", "Only", "noop")})
    orch.active_processes[proc.process_id] = proc

    assert orch.cancel_process(proc.process_id) is True
    assert proc.status is ProcessStatus.CANCELLED
    assert proc.process_id in orch.completed_processes
    assert proc.process_id not in orch.active_processes


def test_cancel_process_returns_false_when_missing(orchestrator):
    orch, _ = orchestrator
    assert orch.cancel_process("absent") is False


def test_get_statistics_reports_runtime(monkeypatch):
    orch = MLProcessOrchestrator(max_workers=1, queue_size=1)

    class DummyQueue:
        def qsize(self):
            return 3

    orch.process_queue = DummyQueue()
    proc = _make_process("active", {"only": ProcessStep("only", "Only", "noop")})
    orch.active_processes[proc.process_id] = proc
    orch.completed_processes["done"] = _make_process(
        "done", {"only": ProcessStep("only", "Only", "noop")}
    )
    orch.running = True

    stats = orch.get_statistics()

    assert stats["running"] is True
    assert stats["active_processes"] == 1
    assert stats["completed_processes"] == 1
    assert stats["queue_size"] == 3
    assert stats["executor_pool_info"]["total_threads"] == orch.max_workers


def test_execute_process_handles_unexpected_exception(orchestrator, monkeypatch):
    orch, captured = orchestrator
    proc = _make_process("boom", {})
    orch.active_processes[proc.process_id] = proc

    def raising(_process):
        raise RuntimeError("explode")

    monkeypatch.setattr(orch, "_execute_process_steps", raising)

    orch._execute_process(proc)

    assert proc.status is ProcessStatus.FAILED
    assert proc.metadata["error"] == "explode"
    assert proc.process_id in orch.completed_processes
    assert captured and captured[0][0].args[0] == "explode"


def test_global_helper_functions(monkeypatch):
    dummy = Mock()
    dummy.submit_process.return_value = "proc-id"
    dummy.get_process_status.return_value = {"process_id": "proc-id"}
    dummy.cancel_process.return_value = True
    dummy.get_statistics.return_value = {"running": False}

    monkeypatch.setattr("src.ml.core.process_orchestrator._ml_orchestrator", dummy)

    # 启动编排器
    dummy.start()

    process = create_ml_process(MLProcessType.MODEL_TRAINING, "name", {})
    assert process.process_name == "name"

    assert submit_ml_process(process) == "proc-id"
    assert get_ml_process_status("proc-id") == {"process_id": "proc-id"}
    assert cancel_ml_process("proc-id") is True
    assert get_ml_orchestrator_stats() == {"running": False}
    assert get_ml_process_orchestrator() is dummy
