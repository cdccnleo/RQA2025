"""
API模块参数对象定义

用于解决长参数列表问题，提高代码可读性和可维护性
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime


# ============ 测试相关参数对象 ============

@dataclass
class TestCaseConfig:
    """测试用例配置参数对象"""
    title: str
    description: str
    priority: str = "medium"  # high, medium, low
    category: str = "functional"  # functional, integration, performance, security
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    environment: str = "test"


@dataclass
class TestScenarioConfig:
    """测试场景配置参数对象"""
    name: str
    description: str
    endpoint: str
    method: str
    setup_steps: List[str] = field(default_factory=list)
    teardown_steps: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteExportConfig:
    """测试套件导出配置"""
    format_type: str = "json"  # json, yaml, html, markdown
    output_dir: Path = Path("docs/api/tests")
    include_timestamps: bool = True
    include_statistics: bool = True
    include_metadata: bool = True
    pretty_print: bool = True
    compress: bool = False


@dataclass
class DocumentationExportConfig:
    """文档导出配置参数对象"""
    output_dir: str = "docs/api"
    include_examples: bool = True
    include_statistics: bool = True
    format_types: List[str] = field(default_factory=lambda: ["json", "yaml"])
    pretty_print: bool = True
    include_metadata: bool = True
    compress: bool = False
    theme: str = "default"


# ============ 文档相关参数对象 ============

@dataclass
class DocumentationConfig:
    """文档生成配置"""
    title: str
    version: str = "1.0.0"
    description: str = ""
    base_url: str = "/api/v1"
    include_authentication: bool = True
    include_validation: bool = True
    include_examples: bool = True
    include_error_codes: bool = True


@dataclass
class EndpointDocumentationConfig:
    """端点文档配置"""
    path: str
    method: str
    summary: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SearchConfig:
    """文档搜索配置"""
    query: str
    search_in_paths: bool = True
    search_in_methods: bool = True
    search_in_descriptions: bool = True
    search_in_parameters: bool = True
    search_in_responses: bool = True
    case_sensitive: bool = False
    max_results: int = 50
    min_relevance_score: float = 0.3


# ============ 流程图相关参数对象 ============

@dataclass
class FlowNodeConfig:
    """流程节点配置"""
    node_id: str
    node_type: str
    label: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, str] = field(default_factory=dict)


@dataclass
class FlowConnectionConfig:
    """流程连接配置"""
    from_node: str
    to_node: str
    label: str = ""
    connection_type: str = "default"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowDiagramConfig:
    """流程图配置"""
    diagram_id: str
    title: str
    description: str = ""
    diagram_type: str = "sequential"  # sequential, parallel, conditional
    orientation: str = "TB"  # TB, LR, BT, RL
    include_legend: bool = True
    include_metadata: bool = True


@dataclass
class FlowExportConfig:
    """流程图导出配置"""
    format_type: str = "mermaid"  # mermaid, json, svg, png
    output_path: Path = Path("docs/api/flows")
    include_statistics: bool = True
    theme: str = "default"
    scale: float = 1.0


# ============ OpenAPI相关参数对象 ============

@dataclass
class OpenAPIConfig:
    """OpenAPI文档生成配置"""
    title: str
    version: str
    description: str = ""
    terms_of_service: Optional[str] = None
    contact: Optional[Dict[str, str]] = None
    license: Optional[Dict[str, str]] = None
    servers: List[Dict[str, str]] = field(default_factory=list)
    security_schemes: Dict[str, Any] = field(default_factory=dict)
    tags: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SchemaGenerationConfig:
    """Schema生成配置"""
    include_examples: bool = True
    include_descriptions: bool = True
    include_constraints: bool = True
    strict_mode: bool = False
    additional_properties: bool = True


@dataclass
class EndpointGenerationConfig:
    """端点生成配置"""
    service_type: str  # data, feature, trading, monitoring
    include_crud: bool = True
    include_batch: bool = False
    include_async: bool = False
    authentication_required: bool = True


# ============ 版本管理相关参数对象 ============

@dataclass
class VersionCreationConfig:
    """版本创建配置"""
    version_string: str
    name: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    created_by: str = "system"


@dataclass
class VersionComparisonConfig:
    """版本比较配置"""
    version1: str
    version2: str
    include_metadata: bool = False
    detailed_diff: bool = False


# ============ 监控相关参数对象 ============

@dataclass
class MetricRecordConfig:
    """指标记录配置"""
    metric_name: str
    value: float
    node_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None
    unit: str = ""


@dataclass
class AlertConfig:
    """告警配置"""
    alert_id: str
    title: str
    message: str
    severity: str = "medium"  # info, low, medium, high, critical
    category: str = "system"
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    auto_resolve: bool = False
    cooldown: int = 300  # 秒


@dataclass
class DashboardConfig:
    """监控面板配置"""
    dashboard_id: str
    title: str
    refresh_interval: int = 30  # 秒
    layout: str = "grid"  # grid, list, custom
    widgets: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)


# ============ 资源管理相关参数对象 ============

@dataclass
class ResourceQuotaConfig:
    """资源配额配置"""
    resource_type: str
    max_value: float
    warning_threshold: float = 0.8
    critical_threshold: float = 0.9
    unit: str = ""


@dataclass
class ResourceAllocationConfig:
    """资源分配配置"""
    cpu_cores: int = 2
    memory_mb: int = 2048
    disk_gb: int = 100
    network_mbps: int = 100
    priority: str = "normal"


# ============ 缓存相关参数对象 ============

@dataclass
class CacheConfig:
    """缓存配置参数对象"""
    cache_type: str = "memory"  # memory, redis, disk
    max_size: int = 1024
    ttl: int = 3600
    eviction_policy: str = "lru"  # lru, lfu, fifo
    compression: bool = False
    serialization: str = "json"  # json, pickle, msgpack


# ============ 日志相关参数对象 ============

@dataclass
class LogConfig:
    """日志配置参数对象"""
    level: str = "INFO"
    format_type: str = "json"  # json, text, structured
    output_type: str = "file"  # file, console, both
    log_file: Optional[Path] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    include_context: bool = True


# ============ 使用示例 ============

"""
# 示例 1: 测试用例生成
config = TestCaseConfig(
    title="数据验证测试",
    description="验证数据格式和内容",
    priority="high",
    tags=["validation", "data"]
)

# 示例 2: 文档搜索
search_config = SearchConfig(
    query="user authentication",
    max_results=10,
    min_relevance_score=0.5
)

# 示例 3: 流程图生成
flow_config = FlowDiagramConfig(
    diagram_id="trading_flow",
    title="交易流程",
    orientation="LR"
)
export_config = FlowExportConfig(
    format_type="mermaid",
    output_path=Path("docs/flows")
)

# 示例 4: 缓存配置
cache_config = CacheConfig(
    cache_type="redis",
    max_size=10240,
    ttl=7200,
    eviction_policy="lru"
)
"""

