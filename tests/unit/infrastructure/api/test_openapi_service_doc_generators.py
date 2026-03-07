from src.infrastructure.api.openapi_generation.endpoint_builder import EndpointBuilder
from src.infrastructure.api.openapi_generation.schema_builder import SchemaBuilder
from src.infrastructure.api.openapi_generation.service_doc_generators import (
    DataServiceDocGenerator,
    FeatureServiceDocGenerator,
    MonitoringServiceDocGenerator,
    TradingServiceDocGenerator,
)


def _setup_builders():
    endpoint_builder = EndpointBuilder()
    schema_builder = SchemaBuilder()
    schema_builder.add_common_data_schemas()
    schema_builder.add_common_response_schemas()
    return endpoint_builder, schema_builder


def test_data_service_doc_generator_creates_dataset_and_stock_endpoints():
    endpoint_builder, schema_builder = _setup_builders()
    generator = DataServiceDocGenerator(endpoint_builder, schema_builder)

    endpoints = generator.generate_endpoints()
    paths = {endpoint.path for endpoint in endpoints}

    assert "/api/v1/data/datasets" in paths
    assert "/api/v1/data/datasets/{dataset_id}" in paths

    dataset_endpoint = next(ep for ep in endpoints if ep.path == "/api/v1/data/datasets")
    response_schema = dataset_endpoint.responses["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"].endswith("PaginatedResponse")


def test_feature_service_doc_generator_includes_request_body():
    endpoint_builder, schema_builder = _setup_builders()
    generator = FeatureServiceDocGenerator(endpoint_builder, schema_builder)

    endpoints = generator.generate_endpoints()
    compute = next(ep for ep in endpoints if ep.path == "/api/v1/features/technical-indicators")

    assert compute.request_body["required"] is True
    schema = compute.request_body["content"]["application/json"]["schema"]
    assert "indicators" in schema["properties"]


def test_trading_service_doc_generator_sets_security():
    endpoint_builder, schema_builder = _setup_builders()
    generator = TradingServiceDocGenerator(endpoint_builder, schema_builder)

    endpoints = generator.generate_endpoints()
    secure_endpoint = next(ep for ep in endpoints if ep.path == "/api/v1/trading/backtest")

    assert secure_endpoint.security == [{"BearerAuth": []}]
    assert secure_endpoint.request_body["content"]["application/json"]["schema"]["type"] == "object"


def test_monitoring_service_doc_generator_has_metrics_endpoint():
    endpoint_builder, schema_builder = _setup_builders()
    generator = MonitoringServiceDocGenerator(endpoint_builder, schema_builder)

    endpoints = generator.generate_endpoints()
    health = next(ep for ep in endpoints if ep.path == "/api/v1/health")
    metrics = next(ep for ep in endpoints if ep.path == "/api/v1/metrics")

    assert health.responses["200"]["description"] == "服务健康"
    assert metrics.security == [{"ApiKeyAuth": []}]


def test_endpoint_builder_defaults_responses_when_missing():
    builder = EndpointBuilder()
    endpoint = builder.create_endpoint(
        path="/api/default",
        method="GET",
        summary="默认响应",
        responses=None,
    )

    assert "200" in endpoint.responses
    assert endpoint.responses["400"]["description"] == "请求错误"
    assert endpoint.operation_id == "get_api_default"

