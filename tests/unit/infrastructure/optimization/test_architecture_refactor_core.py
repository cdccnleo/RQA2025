import json
from pathlib import Path

import pytest

from src.infrastructure.optimization.architecture_refactor import ArchitectureRefactor


@pytest.fixture()
def temp_infrastructure(tmp_path: Path) -> Path:
    """构造一个最小化的 infrastructure 目录用于测试。"""
    (tmp_path / "core").mkdir()
    (tmp_path / "modules").mkdir()
    (tmp_path / "empty_dir").mkdir()

    file_with_relative_import = tmp_path / "modules" / "sample_module.py"
    file_with_relative_import.write_text(
        "from .common.core.base_components import BaseComponent\n",
        encoding="utf-8",
    )
    return tmp_path


def test_analyze_directory_compliance_reports_missing_dirs(temp_infrastructure: Path):
    refactor = ArchitectureRefactor(str(temp_infrastructure))
    compliance = refactor._analyze_directory_compliance()

    assert compliance["compliance_score"] <= 1.0
    assert compliance["compliance_percentage"] == pytest.approx(
        compliance["compliance_score"] * 100, rel=1e-3
    )
    assert "interfaces" in compliance["missing_dirs"]
    assert "utils" in compliance["missing_dirs"]


def test_create_refactor_plan_aggregates_actions():
    refactor = ArchitectureRefactor()
    issues = {
        "import_issues": ["a.py"],
        "large_files": [
            {"file": "big.py", "lines": 1500, "size_kb": 512.0},
        ],
        "empty_dirs": ["obsolete_dir"],
        "architecture_compliance": {
            "expected_dirs": ["core", "interfaces", "utils"],
            "actual_dirs": ["core"],
            "compliance_score": 0.33,
            "compliance_percentage": 33.0,
            "missing_dirs": ["interfaces", "utils"],
            "extra_dirs": [],
        },
    }

    plan = refactor.create_refactor_plan(issues)

    assert "refactor_actions" in plan
    assert "estimated_impact" in plan
    assert len(plan["refactor_actions"]) >= 3
    assert plan["estimated_impact"]["maintainability"] == "high"
    assert plan["risk_assessment"]["overall_risk"] == "medium"


def test_execute_import_fix_rewrites_relative_imports(temp_infrastructure: Path):
    refactor = ArchitectureRefactor(str(temp_infrastructure))
    target_file = temp_infrastructure / "modules" / "sample_module.py"

    action = {
        "action": "fix_relative_imports",
        "files": [str(target_file)],
        "description": "",
    }

    assert refactor._execute_import_fix(action, dry_run=False)
    rewritten = target_file.read_text(encoding="utf-8")
    assert "infrastructure.utils.common.core.base_components" in rewritten


def test_execute_directory_cleanup_removes_empty(temp_infrastructure: Path):
    refactor = ArchitectureRefactor(str(temp_infrastructure))
    empty_dir = temp_infrastructure / "empty_dir"

    action = {
        "action": "remove_empty_dirs",
        "dirs": [str(empty_dir)],
        "description": "",
    }

    assert refactor._execute_directory_cleanup(action, dry_run=False)
    assert not empty_dir.exists()


def test_execute_architecture_improvement_creates_dirs(temp_infrastructure: Path):
    refactor = ArchitectureRefactor(str(temp_infrastructure))

    action = {
        "action": "create_missing_dirs",
        "dirs": ["interfaces"],
        "description": "",
    }

    assert refactor._execute_architecture_improvement(action, dry_run=False)
    created_dir = temp_infrastructure / "interfaces"
    assert created_dir.exists()
    assert (created_dir / "__init__.py").exists()


def test_run_full_refactor_dry_run(temp_infrastructure: Path, tmp_path: Path):
    refactor = ArchitectureRefactor(str(temp_infrastructure))
    summary_log = tmp_path / "architecture_refactor_log.json"

    # 将日志输出目录重定向到 tmp_path，避免污染真实目录
    def fake_save(plan):
        summary_log.write_text(json.dumps(plan))

    refactor._save_refactor_log = fake_save  # type: ignore

    assert refactor.run_full_refactor(dry_run=True)

