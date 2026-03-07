"""测试用例配置对象"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class TestCaseConfig:
    """测试用例配置"""
    title: str
    description: str
    priority: str = "medium"  # high, medium, low
    category: str = "functional"  # functional, integration, performance, security
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    """测试场景配置"""
    name: str
    description: str
    endpoint: str
    method: str
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """导出配置"""
    format_type: str = "json"
    output_dir: Path = Path("docs/api/tests")
    include_timestamps: bool = True
    include_statistics: bool = True
    pretty_print: bool = True
