from __future__ import annotations

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True

mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager

import json
from typing import Any, Dict, List

import numpy as np
import pytest

from src.data.edge.edge_node import (
    EdgeDataProcessor,
    EdgeNetworkManager,
    EdgeNode,
    EdgeServices,
    NodeStatus,
    ProcessingCapability,
)


@pytest.fixture(autouse=True)
def _patch_numpy_secrets(monkeypatch):
    class _Secrets:
        def uniform(self, a, b):
            return (a + b) / 2.0

        def choice(self, seq: List[Any]):
            return seq[0]

    monkeypatch.setattr(np, "secrets", _Secrets(), raising=False)


def _make_node(node_id: str = "edge-1", status: NodeStatus = NodeStatus.OFFLINE) -> EdgeNode:
    node = EdgeNode(
        node_id=node_id,
        location="bj",
        capabilities=[ProcessingCapability.REAL_TIME_ANALYSIS.value],
    )
    node.status = status
    return node


class TestEdgeNode:
    def test_initialize_success_when_resources_sufficient(self):
        node = _make_node()
        node.resources["cpu_usage"] = 10.0
        node.resources["memory_usage"] = 15.0

        assert node.initialize() is True
        assert node.status == NodeStatus.ONLINE

    def test_initialize_fail_when_resources_overloaded(self):
        node = _make_node()
        node.resources["cpu_usage"] = 120.0

        assert node.initialize() is False
        assert node.status == NodeStatus.ERROR

    def test_process_data_restores_status_and_returns_capability_results(self):
        node = _make_node(status=NodeStatus.ONLINE)
        node.capabilities = [
            ProcessingCapability.REAL_TIME_ANALYSIS.value,
            ProcessingCapability.DATA_COMPRESSION.value,
        ]
        data = {"market_data": {"prices": [1, 2, 3]}}

        result = node.process_data(data)

        assert node.status == NodeStatus.ONLINE
        assert "real_time_analysis" in result
        assert "compressed_data" in result
        assert result["node_id"] == node.node_id

    def test_process_data_handles_risk_and_predictive_capabilities(self):
        node = _make_node(status=NodeStatus.ONLINE)
        node.capabilities = [
            ProcessingCapability.LOCAL_RISK_ASSESSMENT.value,
            ProcessingCapability.PREDICTIVE_ANALYTICS.value,
        ]
        payload = {
            "order_data": {"orders": [1]},
            "historical_data": {"prices": [1, 2]},
        }

        result = node.process_data(payload)

        assert "risk_assessment" in result
        assert "predictive_analytics" in result
        assert node.status == NodeStatus.ONLINE

    def test_process_data_error_path_sets_status_error(self, monkeypatch):
        node = _make_node(status=NodeStatus.ONLINE)

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(node, "_real_time_analysis", _boom)
        node.capabilities = [ProcessingCapability.REAL_TIME_ANALYSIS.value]

        result = node.process_data({"market_data": {}})

        assert node.status == NodeStatus.ERROR
        assert result["error"] == "boom"

    def test_sync_with_cloud_returns_true(self):
        node = _make_node(status=NodeStatus.ONLINE)
        assert node.sync_with_cloud({"payload": "cloud"}) is True


class TestEdgeNetworkManager:
    def test_add_node_success_and_duplicate_rejected(self):
        manager = EdgeNetworkManager()
        node = _make_node("n1", NodeStatus.ONLINE)
        assert manager.add_node(node) is True
        assert manager.add_node(node) is False

    def test_route_data_returns_node_when_online(self):
        manager = EdgeNetworkManager()
        node = _make_node("n1", NodeStatus.ONLINE)
        node.initialize()
        manager.add_node(node)
        node.status = NodeStatus.ONLINE

        target = manager.route_data({"payload": 1}, "bj")

        assert target == "n1"

    def test_route_data_falls_back_to_cloud_when_no_node(self):
        manager = EdgeNetworkManager()
        assert manager.route_data({}, "sh") == "cloud"

    def test_route_data_falls_back_when_nodes_offline(self):
        manager = EdgeNetworkManager()
        node = _make_node("n-offline", NodeStatus.ONLINE)
        node.initialize()
        manager.add_node(node)
        node.status = NodeStatus.ERROR  # simulate runtime degradation

        target = manager.route_data({"payload": 1}, "bj")
        assert target == "cloud"

    def test_optimize_network_updates_routing_table(self):
        manager = EdgeNetworkManager()
        node = _make_node("n2", NodeStatus.ONLINE)
        node.initialize()
        manager.add_node(node)
        node.status = NodeStatus.ONLINE

        result = manager.optimize_network()

        assert result["nodes_optimized"] == 1
        assert "bj" in manager.routing_table


class TestEdgeDataProcessor:
    def test_preprocess_data_removes_none_and_handles_arrays(self):
        node = _make_node("processor-node", NodeStatus.ONLINE)
        processor = EdgeDataProcessor(node)
        raw_data = {"prices": [1, 2], "volumes": [3, 4], "unused": None}

        processed = processor.preprocess_data(raw_data)

        assert "unused" not in processed
        assert processed["prices"].dtype == float
        assert "processed_at" in processed

    def test_run_local_ml_model_caches_results(self):
        node = _make_node("processor-node", NodeStatus.ONLINE)
        processor = EdgeDataProcessor(node)
        result = processor.run_local_ml_model({"foo": "bar"})

        assert "prediction" in result
        assert processor.local_cache  # cache populated

    def test_compress_data_success_and_error(self, monkeypatch):
        node = _make_node("processor-node", NodeStatus.ONLINE)
        processor = EdgeDataProcessor(node)
        payload = {"foo": "bar"}

        compressed = processor.compress_data(payload)
        assert isinstance(compressed, bytes)

        original_dumps = json.dumps

        def flaky_dumps(*args, **kwargs):
            if flaky_dumps.calls == 0:
                flaky_dumps.calls += 1
                raise ValueError("broken json")
            return original_dumps(*args, **kwargs)

        flaky_dumps.calls = 0
        monkeypatch.setattr("json.dumps", flaky_dumps)
        fallback = processor.compress_data(payload)
        assert b"broken json" in fallback


class TestEdgeServices:
    def test_real_time_market_analysis_returns_expected_keys(self):
        data = {"prices": [1, 2, 3]}
        result = EdgeServices.real_time_market_analysis(data)

        assert {"market_sentiment", "volatility_forecast", "trend_prediction"}.issubset(result)

    def test_data_compression_and_transmission_returns_metadata_bytes(self):
        payload = {"foo": "bar"}
        compressed = EdgeServices.data_compression_and_transmission(payload)
        decoded = json.loads(compressed.decode("utf - 8"))

        assert decoded["original_size"] >= decoded["compressed_size"]
        assert "timestamp" in decoded

    def test_local_risk_assessment_and_predictive_analytics(self):
        assessment = EdgeServices.local_risk_assessment({"orders": []})
        assert {"risk_score", "risk_level", "approval_status"}.issubset(assessment.keys())

        analytics = EdgeServices.predictive_analytics({"historical": []})
        assert {"price_forecast", "volume_prediction", "confidence_interval"}.issubset(analytics.keys())

