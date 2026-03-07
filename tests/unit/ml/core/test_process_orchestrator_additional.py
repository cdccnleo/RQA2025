import threading
from types import SimpleNamespace

import pytest

import sys
from pathlib import Path

from src.ml.core.process_orchestrator import (
    MLProcessOrchestrator,
    MLProcess,
    ProcessStep,
    ProcessStatus,
    ProcessPriority,
    MLProcessType,
)

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DummyExecutor:
    def __init__(self, fail=False):
        self.fail = fail
        self.executed = []

    def validate(self, step):
        return True

    def execute(self, step, context):
        if self.fail:
            raise RuntimeError("boom")
        self.executed.append(step.step_id)
        return {"ok": True}


def _make_process(step_ids):
    steps = {
        sid: ProcessStep(step_id=sid, step_name=sid, step_type="dummy")
        for sid in step_ids
    }
    return MLProcess(
        process_id=f"process_{len(step_ids)}",
        process_type=MLProcessType.MODEL_TRAINING,
        process_name="test",
        steps=steps,
        priority=ProcessPriority.NORMAL,
    )


def test_submit_requires_running():
    orchestrator = MLProcessOrchestrator(max_workers=1)
    orchestrator.register_step_executor("dummy", DummyExecutor())

    proc = _make_process(["s1"])
    with pytest.raises(RuntimeError):
        orchestrator.submit_process(proc)


def test_start_and_stop_idempotent():
    orchestrator = MLProcessOrchestrator(max_workers=1)
    orchestrator.start()
    orchestrator.start()  # second start should log warning but not crash
    orchestrator.stop()
    orchestrator.stop()  # stopping twice is safe


def test_process_executes_steps_and_updates_stats(monkeypatch):
    orchestrator = MLProcessOrchestrator(max_workers=1)
    executor = DummyExecutor()
    orchestrator.register_step_executor("dummy", executor)
    orchestrator.start()

    proc = _make_process(["s1", "s2"])
    orchestrator.submit_process(proc)

    orchestrator.process_queue.join()
    orchestrator.stop()

    assert proc.status == ProcessStatus.COMPLETED
    assert orchestrator.stats["completed_processes"] == 1
    assert len(orchestrator.completed_processes) == 1


def test_process_failure_marks_failed(monkeypatch):
    orchestrator = MLProcessOrchestrator(max_workers=1)
    executor = DummyExecutor(fail=True)
    orchestrator.register_step_executor("dummy", executor)
    orchestrator.start()

    proc = _make_process(["s1"])
    orchestrator.submit_process(proc)

    orchestrator.process_queue.join()
    orchestrator.stop()

    assert proc.status == ProcessStatus.FAILED
    assert orchestrator.stats["failed_processes"] == 1


def test_callbacks_triggered(monkeypatch):
    orchestrator = MLProcessOrchestrator(max_workers=1)
    executor = DummyExecutor()
    orchestrator.register_step_executor("dummy", executor)
    orchestrator.start()

    proc = _make_process(["s1"])
    results = []

    proc.callbacks["on_start"] = [lambda p: results.append(("start", p.process_id))]
    proc.callbacks["on_complete"] = [lambda p: results.append(("complete", p.process_id))]
    proc.callbacks["on_fail"] = [lambda p: results.append(("fail", p.process_id))]

    orchestrator.submit_process(proc)
    orchestrator.process_queue.join()
    orchestrator.stop()

    assert ("start", proc.process_id) in results
    assert ("complete", proc.process_id) in results
    assert all(r[0] != "fail" for r in results)


def test_get_process_status_and_cancel():
    orchestrator = MLProcessOrchestrator(max_workers=1)
    orchestrator.register_step_executor("dummy", DummyExecutor())
    orchestrator.start()

    proc = _make_process(["s1"])
    orchestrator.submit_process(proc)
    orchestrator.process_queue.join()

    status = orchestrator.get_process_status(proc.process_id)
    assert status["status"] in {"completed", "failed"}

    # cancel after completion should return False
    assert orchestrator.cancel_process(proc.process_id) is False
    orchestrator.stop()

