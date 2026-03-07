import json
from pathlib import Path

import yaml

from src.infrastructure.api.openapi_generation.builders.documentation_assembler import (
    DocumentationAssembler,
)
from src.infrastructure.api.openapi_generation.builders.endpoint_builder import (
    APIEndpoint,
    EndpointBuilderCoordinator,
)
from src.infrastructure.api.openapi_generation.builders.schema_builder import (
    CommonResponseBuilder,
    SchemaBuilder,
)


class _StubEndpoint:
    """构造 DocumentationAssembler 所需的最小端点对象"""

    def __init__(
        self,
        path: str,
        method: str,
        summary: str = "",
        description: str = "",
        tags=None,
        parameters=None,
        responses=None,
        request_body=None,
    ):
        self.path = path
        self.method = method
        self.summary = summary
        self.description = description
        self.tags = tags or []
        self.parameters = parameters or []
        self.responses = responses or {}
        self.request_body = request_body


def test_schema_builder_creates_expected_schemas():
    builder = SchemaBuilder()
    schemas = builder.build_all_schemas()
    initial_count = builder.count_schemas()

    assert "BaseResponse" in schemas
    assert schemas["BaseResponse"]["properties"]["success"]["type"] == "boolean"

    custom_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    builder.add_custom_schema("Custom", custom_schema)

    assert builder.get_schema("Custom") == custom_schema
    assert builder.count_schemas() == initial_count + 1
    assert "TokenResponse" in builder.get_all_schemas()


def test_common_response_builder_returns_predefined_responses():
    responses = CommonResponseBuilder.build_all_common_responses()

    assert "200" in responses and responses["200"]["description"] == "请求成功"
    assert responses["400"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "ValidationErrorResponse"
    )
    assert set(responses.keys()) >= {"200", "201", "400", "401", "403", "404", "429", "500"}


def test_documentation_assembler_assemble_and_export(tmp_path: Path):
    assembler = DocumentationAssembler()
    assembler.set_api_info("Test API", "1.0.0", "描述")
    assembler.add_servers([{"url": "http://localhost"}])
    assembler.add_security_schemes({"bearerAuth": {"type": "http", "scheme": "bearer"}})
    assembler.add_tags([{"name": "Data Service", "description": "数据服务"}])

    endpoint = _StubEndpoint(
        path="/api/example",
        method="GET",
        summary="示例接口",
        description="返回示例数据",
        tags=["Data Service"],
        parameters=[{"name": "symbol", "in": "query"}],
        responses={"200": {"description": "OK"}},
    )
    assembler.add_endpoints([endpoint])
    assembler.add_schemas({"Example": {"type": "object"}})

    doc = assembler.assemble()
    assert doc["info"]["title"] == "Test API"
    assert doc["paths"]["/api/example"]["get"]["summary"] == "示例接口"
    assert doc["components"]["schemas"]["Example"]["type"] == "object"
    assert doc["security"] == [{"bearerAuth": []}]

    json_path = tmp_path / "openapi.json"
    assembler.export_to_json(str(json_path))
    dumped = json.loads(json_path.read_text(encoding="utf-8"))
    assert dumped["info"]["version"] == "1.0.0"

    yaml_path = tmp_path / "openapi.yaml"
    assembler.export_to_yaml(str(yaml_path))
    yaml_doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert yaml_doc["servers"][0]["url"] == "http://localhost"

    stats = assembler.get_statistics()
    assert stats["total_endpoints"] == 1
    assert stats["endpoints_by_method"]["GET"] == 1
    assert stats["endpoints_by_tag"]["Data Service"] == 1


def test_endpoint_builder_coordinator_counts_and_custom_service():
    coordinator = EndpointBuilderCoordinator()

    endpoints = coordinator.build_all_endpoints()
    assert len(endpoints) > 0
    data_endpoints = coordinator.get_endpoints_by_service("data_service")
    assert data_endpoints and all(isinstance(ep, APIEndpoint) for ep in data_endpoints)

    class CustomServiceBuilder:
        def build_endpoints(self):
            return [
                APIEndpoint(
                    path="/api/custom",
                    method="POST",
                    summary="自定义",
                    description="测试",
                    tags=["Custom"],
                )
            ]

    coordinator.add_service_builder("custom_service", CustomServiceBuilder())
    counts = coordinator.count_endpoints()
    assert counts["custom_service"] == 1
    assert counts["total"] >= len(endpoints) + 1

