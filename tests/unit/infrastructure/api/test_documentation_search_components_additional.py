import importlib
import json

from src.infrastructure.api.documentation_search.document_loader import DocumentLoader
from src.infrastructure.api.documentation_search.search_engine import SearchEngine


def test_document_loader_reads_endpoints(tmp_path):
    docs = {
        "endpoints": {
            "getUser": {
                "path": "/api/users/{user_id}",
                "summary": "获取用户信息",
                "description": "根据ID获取用户详情",
            }
        }
    }
    docs_path = tmp_path / "docs.json"
    docs_path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    loader = DocumentLoader()
    documents = loader.load_documents(str(docs_path))

    assert "getUser" in documents
    assert documents["getUser"]["path"] == "/api/users/{user_id}"


def test_search_engine_search_and_cache_behaviour():
    documents = {
        "data:get_prices": {
            "path": "/api/data/prices",
            "summary": "获取市场价格",
            "description": "返回指定交易对的市场价格数据",
            "tags": ["data", "market"],
            "parameters": [
                {"name": "symbol", "description": "交易对"},
                {"name": "limit", "description": "返回条数"},
            ],
            "responses": {
                "200": {"description": "成功返回价格"},
                "404": {"description": "资源不存在"},
            },
        }
    }

    engine = SearchEngine()
    results = engine.search("market price data", documents, limit=5)
    assert results
    first_result = results[0]
    assert first_result.endpoint_id == "data:get_prices"
    assert "path" in first_result.matched_fields
    assert first_result.match_type in {"exact", "partial"}

    cached_results = engine.search("market price data", documents, limit=5)
    assert cached_results == results

    parameter_results = engine.search("limit", documents, search_type="parameters")
    assert parameter_results

    stats = engine.get_statistics()
    assert stats["total_searches"] >= 3
    assert stats["cache_hit_rate"] > 0

    engine.clear_cache()
    assert engine.search_cache == {}


def test_lazy_import_from_package():
    module = importlib.import_module("src.infrastructure.api")
    reloaded = importlib.reload(module)

    # 确保触发 __getattr__ 延迟导入
    generator_cls = getattr(reloaded, "APIFlowDiagramGenerator")

    assert generator_cls.__name__ == "FlowDiagramGenerator"
    assert generator_cls.__module__.endswith("api_flow_diagram_generator_refactored")

