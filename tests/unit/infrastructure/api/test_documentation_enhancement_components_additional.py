from types import SimpleNamespace

import pytest

from src.infrastructure.api.documentation_enhancement.example_generator import ExampleGenerator
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import (
    APIParameterDocumentation,
    ParameterEnhancer,
)
from src.infrastructure.api.documentation_enhancement.response_standardizer import ResponseStandardizer


def test_example_generator_produces_examples_and_data_variations():
    generator = ExampleGenerator()

    parameters = [
        SimpleNamespace(name="symbol", example="ETH/USDT", default=None),
        SimpleNamespace(name="limit", example=None, default=50),
    ]
    endpoint = SimpleNamespace(parameters=parameters, path="/api/market/ticker")

    request_example = generator.generate_request_example(endpoint)
    assert request_example == {"symbol": "ETH/USDT", "limit": 50}

    success_example = generator.generate_success_response_example(endpoint)
    assert success_example["data"]["symbol"] == "BTC/USDT"

    error_example = generator.generate_error_response_example(404)
    assert error_example["error"]["code"] == "E404"
    assert error_example["message"] == "资源不存在"

    order_endpoint = SimpleNamespace(parameters=[], path="/api/order/create")
    order_data = generator._generate_data_example(order_endpoint)
    assert order_data["order_id"].startswith("ORD")

    feature_endpoint = SimpleNamespace(parameters=[], path="/api/feature/build")
    feature_data = generator._generate_data_example(feature_endpoint)
    assert "features" in feature_data


def test_parameter_enhancer_applies_constraints_and_cache_behaviour():
    enhancer = ParameterEnhancer()
    email_param = APIParameterDocumentation(
        name="user_email",
        type="string",
        required=True,
        description="用户邮箱",
    )

    enhancer.enhance_parameter(email_param)

    assert email_param.constraints["format"] == "email"
    assert any(rule.startswith("最小长度") for rule in email_param.validation_rules)

    cached_value = enhancer._generate_example_value(email_param)
    stats = enhancer.get_cache_stats()
    assert cached_value == email_param.example
    assert stats["cache_hit_count"] >= 1
    assert stats["cache_miss_count"] >= 1

    price_param = APIParameterDocumentation(
        name="price",
        type="number",
        required=False,
        description="价格",
    )
    enhancer.enhance_parameter(price_param)
    assert price_param.example == pytest.approx(100.50)

    enhancer.clear_cache()
    cleared_stats = enhancer.get_cache_stats()
    assert cleared_stats["cache_size"] == 0
    assert cleared_stats["cache_miss_count"] == 0


def test_response_standardizer_adds_defaults_and_error_codes():
    standardizer = ResponseStandardizer()

    class EndpointStub:
        def __init__(self):
            self.responses = []
            self.authentication = True
            self.method = "POST"
            self.error_codes = []

    endpoint = EndpointStub()
    standardizer.standardize_responses(endpoint)

    status_codes = {resp.status_code for resp in endpoint.responses}
    assert {200, 400, 401, 403, 404, 500}.issubset(status_codes)

    standardizer.add_error_codes_to_endpoint(endpoint)
    categories = {code["category"] for code in endpoint.error_codes}
    assert {"general", "authentication", "authorization", "validation"}.issubset(categories)


