from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from src.infrastructure.api.openapi_generation.coordinator import RQAApiDocCoordinator
from src.infrastructure.api.openapi_generation.endpoint_builder import EndpointBuilder


@dataclass
class _StubEndpoint:
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    responses: Dict[str, Any] = field(default_factory=dict)
    request_body: Dict[str, Any] = None
    security: List[Dict[str, Any]] = field(default_factory=list)
    operation_id: str = "opId"


@dataclass
class _StubSchema:
    title: str = "Test API"
    version: str = "1.0.0"
    description: str = "测试描述"
    servers: List[Dict[str, str]] = field(default_factory=lambda: [{"url": "http://localhost"}])
    security_schemes: Dict[str, Any] = field(default_factory=lambda: {"bearerAuth": {"type": "http"}})
    endpoints: List[_StubEndpoint] = field(default_factory=list)
    schemas: Dict[str, Any] = field(default_factory=lambda: {"Example": {"type": "object"}})


class _CoordinatorUnderTest(RQAApiDocCoordinator):
    """通过注入 stub schema 来测试内部路径生成逻辑"""

    def __init__(self, schema: _StubSchema):
        self.api_schema = schema

    def _create_rqa_api_schema(self):
        return self.api_schema


def test_coordinator_generates_paths_and_tags():
    endpoint = _StubEndpoint(
        path="/api/example",
        method="GET",
        summary="示例接口",
        description="描述",
        tags=["Data Service"],
        parameters=[{"name": "symbol"}],
        responses={"200": {"description": "OK"}},
        request_body=None,
    )
    schema = _StubSchema(endpoints=[endpoint])
    coordinator = _CoordinatorUnderTest(schema)

    spec = coordinator._generate_openapi_spec()
    assert spec["info"]["title"] == "Test API"
    assert spec["paths"]["/api/example"]["get"]["summary"] == "示例接口"

    tags = spec["tags"]
    assert tags[0]["name"] == "Data Service"
    assert "Data Service" in coordinator.get_statistics()["services"]


def test_generate_paths_handles_security_and_request_body():
    endpoint = _StubEndpoint(
        path="/secure",
        method="POST",
        summary="安全端点",
        description="需要认证",
        tags=["Trading Service"],
        parameters=[],
        responses={"201": {"description": "Created"}},
        request_body={"content": {"application/json": {"schema": {"type": "object"}}}},
        security=[{"bearerAuth": []}],
    )
    schema = _StubSchema(endpoints=[endpoint])
    coordinator = _CoordinatorUnderTest(schema)

    paths = coordinator._generate_paths()
    method_spec = paths["/secure"]["post"]

    assert method_spec["requestBody"]["content"]["application/json"]["schema"]["type"] == "object"
    assert method_spec["security"] == [{"bearerAuth": []}]


def test_generate_tags_default_description():
    monitor_endpoint = _StubEndpoint(
        path="/monitor",
        method="GET",
        summary="监控",
        description="",
        tags=["Monitoring"],
    )
    unknown_endpoint = _StubEndpoint(
        path="/custom",
        method="GET",
        summary="自定义",
        description="",
        tags=["Unknown"],
    )
    none_tag_endpoint = _StubEndpoint(
        path="/untagged",
        method="GET",
        summary="无标签",
        description="",
        tags=[],
    )
    schema = _StubSchema(endpoints=[monitor_endpoint, unknown_endpoint, none_tag_endpoint])
    coordinator = _CoordinatorUnderTest(schema)

    tags = {tag["name"]: tag["description"] for tag in coordinator._generate_tags()}
    assert tags["Monitoring"] == "系统监控和健康检查"
    assert tags["Unknown"] == ""
    assert "Untagged" not in tags


def test_get_statistics_counts_security_schemes():
    endpoint = _StubEndpoint(
        path="/data",
        method="GET",
        summary="数据",
        description="",
        tags=["Data Service"],
    )
    schema = _StubSchema(
        endpoints=[endpoint],
        schemas={"Example": {"type": "object"}, "Another": {"type": "string"}},
        security_schemes={"bearerAuth": {"type": "http"}, "apiKey": {"type": "apiKey"}},
    )
    coordinator = _CoordinatorUnderTest(schema)

    stats = coordinator.get_statistics()
    assert stats["total_endpoints"] == 1
    assert stats["total_schemas"] == 2
    assert stats["security_schemes"] == 2


def test_create_rqa_api_schema_aggregates_generators(monkeypatch):
    coordinator = RQAApiDocCoordinator()
    builder: EndpointBuilder = coordinator.endpoint_builder

    def fake_data_endpoints():
        return [
            builder.create_endpoint(
                path="/api/custom",
                method="GET",
                summary="自定义端点",
                tags=["Data Service"],
                responses={"200": {"description": "OK"}},
            )
        ]

    monkeypatch.setattr(coordinator.data_gen, "generate_endpoints", fake_data_endpoints)
    monkeypatch.setattr(coordinator.feature_gen, "generate_endpoints", lambda: [])
    monkeypatch.setattr(coordinator.trading_gen, "generate_endpoints", lambda: [])
    monkeypatch.setattr(coordinator.monitoring_gen, "generate_endpoints", lambda: [])

    schema = coordinator._create_rqa_api_schema()

    assert len(schema.endpoints) == 1
    assert schema.endpoints[0].operation_id == "get_api_custom"
    assert schema.security_schemes["BearerAuth"]["type"] == "http"
    assert "Dataset" in schema.schemas


def test_generate_paths_merges_duplicate_path(monkeypatch):
    get_endpoint = _StubEndpoint(
        path="/conflict",
        method="GET",
        summary="查询",
        description="获取资源",
        tags=["Data Service"],
        responses={"200": {"description": "OK"}},
    )
    post_endpoint = _StubEndpoint(
        path="/conflict",
        method="POST",
        summary="创建",
        description="创建资源",
        tags=["Data Service"],
        request_body={"content": {"application/json": {"schema": {"type": "object"}}}},
        responses={"201": {"description": "Created"}},
    )
    schema = _StubSchema(endpoints=[get_endpoint, post_endpoint])
    coordinator = _CoordinatorUnderTest(schema)

    paths = coordinator._generate_paths()

    assert set(paths["/conflict"].keys()) == {"get", "post"}
    assert paths["/conflict"]["post"]["requestBody"]["content"]["application/json"]["schema"]["type"] == "object"

