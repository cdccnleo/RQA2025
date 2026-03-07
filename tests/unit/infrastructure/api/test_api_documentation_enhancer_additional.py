import json
from pathlib import Path

from src.infrastructure.api.api_documentation_enhancer_refactored import (
    APIDocumentationEnhancer,
    APIEndpointDocumentation,
)
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import (
    APIParameterDocumentation,
)
from src.infrastructure.api.documentation_enhancement.response_standardizer import (
    APIResponseDocumentation,
)


def _build_endpoint() -> APIEndpointDocumentation:
    parameters = [
        APIParameterDocumentation(
            name="symbol",
            type="string",
            required=True,
            description="交易对",
        ),
        APIParameterDocumentation(
            name="limit",
            type="integer",
            required=False,
            description="数量限制",
        ),
    ]

    responses = [
        APIResponseDocumentation(
            status_code=200,
            description="成功",
        )
    ]

    return APIEndpointDocumentation(
        path="/api/v1/data/market/{symbol}",
        method="GET",
        summary="获取市场数据",
        description="返回指定交易对的市场数据",
        parameters=parameters,
        responses=responses,
        authentication=[{"type": "bearer"}],
        error_codes=[],
    )


def test_api_documentation_enhancer_enhances_single_endpoint(tmp_path: Path):
    enhancer = APIDocumentationEnhancer()
    endpoint = _build_endpoint()

    key = f"{endpoint.method}_{endpoint.path}"
    enhancer.add_endpoint(endpoint)
    enhancer.enhance_endpoint_documentation(key)

    enhanced = enhancer.endpoints[key]

    # 参数应被增强并生成示例
    assert enhanced.parameters[0].example is not None
    assert "constraints" in enhanced.parameters[0].__dict__

    # 响应被标准化并添加错误代码
    statuses = {resp.status_code for resp in enhanced.responses}
    assert {200, 400, 401, 403, 404, 500}.issubset(statuses)
    assert any(code["category"] == "authentication" for code in enhanced.error_codes)

    # 生成示例
    assert "request" in enhanced.examples
    assert "success_response" in enhanced.examples
    assert enhanced.examples["error_responses"]["400"]["error"]["code"] == "E400"

    # 生成增强文档并验证内容
    output_path = tmp_path / "enhanced.json"
    enhancer.generate_enhanced_documentation(str(output_path))

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["endpoints"][0]["path"] == endpoint.path
    assert "common_responses" in saved
    assert saved["error_codes"]


def test_enhance_all_endpoints_returns_count():
    enhancer = APIDocumentationEnhancer()
    first = _build_endpoint()
    second = _build_endpoint()
    second.path = "/api/v1/data/kline"

    enhancer.add_endpoint(first)
    enhancer.add_endpoint(second)

    enhanced_count = enhancer.enhance_all_endpoints()
    assert enhanced_count == 2

    # 向后兼容属性应返回对应组件
    assert enhancer.parameter_enhancer is enhancer.get_parameter_enhancer()
    assert enhancer.response_standardizer is enhancer.get_response_standardizer()
    assert enhancer.example_generator is enhancer.get_example_generator()

