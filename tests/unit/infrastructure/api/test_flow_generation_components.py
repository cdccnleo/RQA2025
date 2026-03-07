from pathlib import Path

import json

import pytest

from src.infrastructure.api.flow_generation.coordinator import APIFlowCoordinator
from src.infrastructure.api.flow_generation.exporter import FlowExporter
from src.infrastructure.api.flow_generation.flow_generators import (
    DataServiceFlowGenerator,
    FeatureEngineeringFlowGenerator,
    TradingFlowGenerator,
)
from src.infrastructure.api.flow_generation.models import APIFlowDiagram, APIFlowEdge, APIFlowNode
from src.infrastructure.api.flow_generation.node_builder import FlowNodeBuilder


def build_sample_diagram(layout: str = "horizontal") -> APIFlowDiagram:
    nodes = [
        APIFlowNode(id="start", label="开始", node_type="start"),
        APIFlowNode(id="data", label="加载数据", node_type="process"),
        APIFlowNode(id="decision", label="是否命中", node_type="decision"),
        APIFlowNode(id="db", label="数据库", node_type="database"),
        APIFlowNode(id="cache", label="缓存", node_type="cache"),
        APIFlowNode(id="api", label="外部调用", node_type="api_call"),
        APIFlowNode(id="end", label="结束", node_type="end"),
    ]
    edges = [
        APIFlowEdge(source="start", target="data", edge_type="normal"),
        APIFlowEdge(source="data", target="decision", label="判断", edge_type="conditional"),
        APIFlowEdge(source="decision", target="cache", label="命中", edge_type="success"),
        APIFlowEdge(source="decision", target="db", label="未命中", edge_type="error"),
        APIFlowEdge(source="cache", target="api"),
        APIFlowEdge(source="db", target="api"),
        APIFlowEdge(source="api", target="end"),
    ]

    return APIFlowDiagram(
        id="sample_flow",
        title="示例流程",
        description="用于测试的示例流程图",
        nodes=nodes,
        edges=edges,
        layout=layout,
        theme="dark",
    )


def test_flow_node_builder_creates_and_clears_nodes_edges():
    builder = FlowNodeBuilder()

    start = builder.create_start_node(position={"x": 0, "y": 0})
    process = builder.create_process_node("process", "处理")
    decision = builder.create_decision_node("decision", "判断")
    api_call = builder.create_api_call_node("api_call", "外部调用")
    end = builder.create_end_node(position={"x": 100, "y": 0})

    edge = builder.create_edge("start", "process", label="开始", edge_type="success", condition="ok")

    nodes = builder.get_nodes()
    edges = builder.get_edges()

    assert {n.id for n in nodes} == {"start", "process", "decision", "api_call", "end"}
    assert edge.label == "开始"
    assert edge.edge_type == "success"
    assert edges[0].condition == "ok"

    nodes.append(APIFlowNode(id="extra", label="无效", node_type="process"))
    assert len(builder.get_nodes()) == 5

    builder.clear()
    assert builder.get_nodes() == []
    assert builder.get_edges() == []


def test_flow_exporter_mermaid_writes_expected_content(tmp_path: Path):
    diagram = build_sample_diagram(layout="vertical")
    exporter = FlowExporter()

    mermaid_path = tmp_path / "flow.md"
    mermaid_code = exporter.export_to_mermaid(diagram, str(mermaid_path))

    assert "graph TD" in mermaid_code
    assert "sample_flow" not in mermaid_code  # 不直接输出ID
    assert mermaid_path.read_text(encoding="utf-8") == mermaid_code
    assert "[(" in mermaid_code  # database 形状
    assert "[/" in mermaid_code  # cache 形状
    assert "|判断|" in mermaid_code


def test_flow_exporter_json_and_statistics(tmp_path: Path):
    diagram = build_sample_diagram()
    exporter = FlowExporter()

    json_path = tmp_path / "flow.json"
    data = exporter.export_to_json(diagram, str(json_path))

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["id"] == "sample_flow"
    assert len(saved["nodes"]) == len(diagram.nodes)
    assert len(saved["edges"]) == len(diagram.edges)

    stats = exporter.get_statistics(diagram)
    assert stats["total_nodes"] == len(diagram.nodes)
    assert stats["edge_types"]["error"] == 1
    assert stats["node_types"]["cache"] == 1
    assert stats["complexity"] == len(diagram.nodes) + len(diagram.edges)


def test_flow_generators_produce_expected_diagrams():
    builder = FlowNodeBuilder()

    data_diagram = DataServiceFlowGenerator(builder).generate()
    assert data_diagram.id == "data_service_flow"
    assert any(edge.edge_type == "success" for edge in data_diagram.edges)

    trading_diagram = TradingFlowGenerator(builder).generate()
    assert trading_diagram.id == "trading_flow"
    assert len(trading_diagram.nodes) > 5

    feature_diagram = FeatureEngineeringFlowGenerator(builder).generate()
    assert feature_diagram.id == "feature_engineering_flow"
    assert all(node.node_type != "" for node in feature_diagram.nodes)


def test_api_flow_coordinator_exports_and_statistics(tmp_path: Path):
    coordinator = APIFlowCoordinator()

    diagrams = coordinator.generate_all_flows()
    assert {"data_service_flow", "trading_flow", "feature_engineering_flow"} == set(diagrams.keys())

    mermaid_files = coordinator.export_to_mermaid(output_dir=str(tmp_path / "mermaid"))
    json_files = coordinator.export_to_json(output_dir=str(tmp_path / "json"))

    assert all(Path(path).exists() for path in mermaid_files.values())
    assert all(Path(path).exists() for path in json_files.values())

    stats = coordinator.get_flow_statistics()
    assert stats["total_flows"] == 3
    assert all(flow_id in stats["flows"] for flow_id in diagrams.keys())
    assert stats["flows"]["data_service_flow"]["total_nodes"] > 0

