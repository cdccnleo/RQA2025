import json
from unittest.mock import MagicMock

import pytest
import yaml

from src.infrastructure.resource.utils.optimization_report_generator import OptimizationReportGenerator


@pytest.fixture
def dependencies():
    system_analyzer = MagicMock()
    thread_analyzer = MagicMock()
    memory_detector = MagicMock()
    logger = MagicMock()
    error_handler = MagicMock()
    return system_analyzer, thread_analyzer, memory_detector, logger, error_handler


@pytest.fixture
def generator(dependencies):
    system_analyzer, thread_analyzer, memory_detector, logger, error_handler = dependencies
    gen = OptimizationReportGenerator(
        system_analyzer=system_analyzer,
        thread_analyzer=thread_analyzer,
        memory_detector=memory_detector,
        logger=logger,
        error_handler=error_handler,
    )
    return gen, dependencies


def test_generate_summary_report(generator):
    gen, (system_analyzer, thread_analyzer, memory_detector, _, _) = generator
    system_analyzer.get_resource_summary.return_value = {"cpu_usage": 95, "disk_read_bytes": 600 * 1024 * 1024}
    thread_analyzer.get_thread_summary.return_value = {"thread_count": 120}
    memory_detector.detect_memory_leaks.return_value = ["leak1", "leak2"]

    report = gen.generate_optimization_report("summary")
    assert report["report_type"] == "summary"
    assert len(report["recommendations"]) >= 4
    system_analyzer.get_resource_summary.assert_called_once()


def test_generate_detailed_report(generator):
    gen, (system_analyzer, thread_analyzer, memory_detector, _, _) = generator
    system_analyzer.get_system_resources.return_value = {"cpu": {"avg": 40}}
    thread_analyzer.analyze_threads.return_value = {"thread_count": 5}
    thread_analyzer.detect_thread_issues.return_value = {"problems": []}
    memory_detector.get_memory_report.return_value = {"issues": []}

    report = gen.generate_optimization_report("detailed")
    assert report["report_type"] == "detailed"
    assert "sections" in report and "optimization_suggestions" in report["sections"]
    thread_analyzer.analyze_threads.assert_called_once_with(include_stacks=True)


def test_generate_report_error(generator):
    gen, (system_analyzer, _, _, _, error_handler) = generator
    system_analyzer.get_resource_summary.side_effect = RuntimeError("system fail")
    report = gen.generate_optimization_report("summary")
    assert report["error"] == "system fail"
    error_handler.handle_error.assert_called_once()


def test_export_report_formats(generator):
    gen, _ = generator
    report = {"foo": "bar"}
    json_output = gen.export_report(report, "json")
    assert json.loads(json_output)["foo"] == "bar"

    yaml_output = gen.export_report(report, "yaml")
    assert yaml.safe_load(yaml_output)["foo"] == "bar"

    invalid = gen.export_report(report, "xml")
    assert invalid.startswith("导出失败")


def test_generate_detailed_report_error(generator):
    gen, (system_analyzer, _, _, _, _) = generator
    system_analyzer.get_system_resources.side_effect = RuntimeError("detailed crash")
    report = gen.generate_optimization_report("detailed")
    assert report["error"] == "detailed crash"

