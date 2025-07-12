import pytest
from unittest.mock import MagicMock, patch
from src.strategy_workspace.visual_editor import (
    VisualStrategyEditor,
    StrategyNode,
    NodeType
)

@pytest.fixture
def sample_nodes():
    """创建测试用策略节点"""
    return [
        StrategyNode(
            node_id="data_1",
            node_type=NodeType.DATA_SOURCE,
            name="Stock Data",
            params={"symbol": "600000", "frequency": "daily"},
            next_nodes=[]
        ),
        StrategyNode(
            node_id="feature_1",
            node_type=NodeType.FEATURE,
            name="Technical Features",
            params={"indicators": ["MA", "RSI"]},
            next_nodes=[]
        ),
        StrategyNode(
            node_id="model_1",
            node_type=NodeType.MODEL,
            name="LSTM Model",
            params={"lookback": 30, "units": 64},
            next_nodes=[]
        ),
        StrategyNode(
            node_id="trade_1",
            node_type=NodeType.TRADE,
            name="Trading Rules",
            params={"buy_threshold": 0.5, "sell_threshold": -0.3},
            next_nodes=[]
        )
    ]

@pytest.fixture
def editor_with_nodes(sample_nodes):
    """创建包含节点的编辑器实例"""
    editor = VisualStrategyEditor()
    for node in sample_nodes:
        editor.add_node(node)
    return editor

def test_add_and_remove_nodes(editor_with_nodes):
    """测试节点添加和移除"""
    # 验证初始节点
    assert len(editor_with_nodes.get_all_nodes()) == 4

    # 添加新节点
    new_node = StrategyNode(
        node_id="risk_1",
        node_type=NodeType.RISK,
        name="Risk Control",
        params={"max_drawdown": 0.1},
        next_nodes=[]
    )
    assert editor_with_nodes.add_node(new_node) is True
    assert len(editor_with_nodes.get_all_nodes()) == 5

    # 重复添加
    assert editor_with_nodes.add_node(new_node) is False

    # 移除节点
    assert editor_with_nodes.remove_node("risk_1") is True
    assert len(editor_with_nodes.get_all_nodes()) == 4

    # 移除不存在的节点
    assert editor_with_nodes.remove_node("nonexistent") is False

def test_node_connections(editor_with_nodes):
    """测试节点连接管理"""
    # 建立连接
    assert editor_with_nodes.connect_nodes("data_1", "feature_1") is True
    assert editor_with_nodes.connect_nodes("feature_1", "model_1") is True
    assert editor_with_nodes.connect_nodes("model_1", "trade_1") is True

    # 验证连接
    connections = editor_with_nodes.get_connections()
    assert len(connections) == 3
    assert {"source": "data_1", "target": "feature_1"} in connections

    # 验证下游节点
    data_node = editor_with_nodes.get_node("data_1")
    assert "feature_1" in data_node.next_nodes

    # 重复连接
    assert editor_with_nodes.connect_nodes("data_1", "feature_1") is False

    # 断开连接
    assert editor_with_nodes.disconnect_nodes("feature_1", "model_1") is True
    assert len(editor_with_nodes.get_connections()) == 2

    # 断开不存在的连接
    assert editor_with_nodes.disconnect_nodes("nonexistent", "nonexistent") is False

def test_strategy_validation(editor_with_nodes):
    """测试策略验证"""
    # 有效策略
    editor_with_nodes.connect_nodes("data_1", "feature_1")
    editor_with_nodes.connect_nodes("feature_1", "model_1")
    editor_with_nodes.connect_nodes("model_1", "trade_1")
    assert editor_with_nodes.validate_strategy() is True

    # 创建孤立节点
    isolated_node = StrategyNode(
        node_id="risk_1",
        node_type=NodeType.RISK,
        name="Risk Control",
        params={"max_drawdown": 0.1},
        next_nodes=["nonexistent"]  # 无效下游节点
    )
    editor_with_nodes.add_node(isolated_node)
    assert editor_with_nodes.validate_strategy() is False

    # 创建循环依赖
    editor_with_nodes.connect_nodes("trade_1", "data_1")  # 形成环
    assert editor_with_nodes.validate_strategy() is False

def test_import_export(editor_with_nodes):
    """测试策略导入导出"""
    # 建立连接
    editor_with_nodes.connect_nodes("data_1", "feature_1")
    editor_with_nodes.connect_nodes("feature_1", "model_1")

    # 导出策略
    exported = editor_with_nodes.export_strategy()
    assert isinstance(exported, dict)
    assert "nodes" in exported
    assert "connections" in exported
    assert len(exported["nodes"]) == 4
    assert len(exported["connections"]) == 2

    # 创建新编辑器并导入
    new_editor = VisualStrategyEditor()
    assert new_editor.import_strategy(exported) is True
    assert len(new_editor.get_all_nodes()) == 4
    assert len(new_editor.get_connections()) == 2

    # 验证节点参数
    imported_node = new_editor.get_node("data_1")
    assert imported_node.params["symbol"] == "600000"
    assert imported_node.node_type == NodeType.DATA_SOURCE

    # 测试无效导入
    assert new_editor.import_strategy({}) is False
    assert new_editor.import_strategy({"nodes": "invalid"}) is False

def test_update_node_params(editor_with_nodes):
    """测试节点参数更新"""
    # 更新参数
    assert editor_with_nodes.update_node_params(
        "data_1",
        {"symbol": "000001", "adjust": "post"}
    ) is True

    # 验证更新
    updated_node = editor_with_nodes.get_node("data_1")
    assert updated_node.params["symbol"] == "000001"
    assert updated_node.params["adjust"] == "post"
    assert updated_node.params["frequency"] == "daily"  # 保留原有参数

    # 更新不存在的节点
    assert editor_with_nodes.update_node_params("nonexistent", {}) is False

def test_visualization(editor_with_nodes, capsys):
    """测试可视化输出"""
    # 建立连接
    editor_with_nodes.connect_nodes("data_1", "feature_1")
    editor_with_nodes.connect_nodes("feature_1", "model_1")

    # 调用可视化方法
    editor_with_nodes.visualize()

    # 捕获输出
    captured = capsys.readouterr()
    output = captured.out

    # 验证输出内容
    assert "Node data_1 (data_source): Stock Data" in output
    assert "-> feature_1" in output
    assert "Node feature_1 (feature): Technical Features" in output
