"""
测试API流程图生成协调器

覆盖 coordinator.py 中的 APIFlowCoordinator 类
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.flow_generation.coordinator import APIFlowCoordinator


class TestAPIFlowCoordinator:
    """APIFlowCoordinator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        coordinator = APIFlowCoordinator()

        assert hasattr(coordinator, 'node_builder')
        assert hasattr(coordinator, 'exporter')
        assert hasattr(coordinator, 'data_gen')
        assert hasattr(coordinator, 'trading_gen')
        assert hasattr(coordinator, 'feature_gen')
        assert hasattr(coordinator, 'diagrams')
        assert isinstance(coordinator.diagrams, dict)

    @patch('src.infrastructure.api.flow_generation.coordinator.FlowNodeBuilder')
    @patch('src.infrastructure.api.flow_generation.coordinator.FlowExporter')
    @patch('src.infrastructure.api.flow_generation.coordinator.DataServiceFlowGenerator')
    @patch('src.infrastructure.api.flow_generation.coordinator.TradingFlowGenerator')
    @patch('src.infrastructure.api.flow_generation.coordinator.FeatureEngineeringFlowGenerator')
    def test_initialization_with_mocks(self, mock_feature_gen, mock_trading_gen, mock_data_gen, mock_exporter, mock_node_builder):
        """测试初始化（使用模拟对象）"""
        mock_node_builder.return_value = Mock()
        mock_exporter.return_value = Mock()
        mock_data_gen.return_value = Mock()
        mock_trading_gen.return_value = Mock()
        mock_feature_gen.return_value = Mock()

        coordinator = APIFlowCoordinator()

        mock_node_builder.assert_called_once()
        mock_exporter.assert_called_once()
        mock_data_gen.assert_called_once_with(mock_node_builder.return_value)
        mock_trading_gen.assert_called_once_with(mock_node_builder.return_value)
        mock_feature_gen.assert_called_once_with(mock_node_builder.return_value)

    def test_create_data_service_flow(self):
        """测试创建数据服务流程图"""
        coordinator = APIFlowCoordinator()

        result = coordinator.create_data_service_flow()

        # 验证返回结果是流程图对象
        assert result is not None
        assert hasattr(result, 'nodes')
        assert hasattr(result, 'edges')
        # 验证流程图已被添加到diagrams中
        assert 'data_service_flow' in coordinator.diagrams

    def test_create_trading_flow(self):
        """测试创建交易流程图"""
        coordinator = APIFlowCoordinator()

        result = coordinator.create_trading_flow()

        assert result is not None
        assert hasattr(result, 'nodes')
        assert hasattr(result, 'edges')
        assert 'trading_flow' in coordinator.diagrams

    def test_create_feature_engineering_flow(self):
        """测试创建特征工程流程图"""
        coordinator = APIFlowCoordinator()

        result = coordinator.create_feature_engineering_flow()

        assert result is not None
        assert hasattr(result, 'nodes')
        assert hasattr(result, 'edges')
        assert 'feature_engineering_flow' in coordinator.diagrams

    def test_generate_all_flows(self):
        """测试生成所有流程图"""
        coordinator = APIFlowCoordinator()

        # 模拟生成器方法
        mock_data_diagram = Mock()
        mock_data_diagram.id = "data_service"
        coordinator.data_gen.generate = Mock(return_value=mock_data_diagram)

        mock_trading_diagram = Mock()
        mock_trading_diagram.id = "trading"
        coordinator.trading_gen.generate = Mock(return_value=mock_trading_diagram)

        mock_feature_diagram = Mock()
        mock_feature_diagram.id = "feature_engineering"
        coordinator.feature_gen.generate = Mock(return_value=mock_feature_diagram)

        result = coordinator.generate_all_flows()

        assert "data_service" in result
        assert "trading" in result
        assert "feature_engineering" in result
        assert result["data_service"] == mock_data_diagram
        assert result["trading"] == mock_trading_diagram
        assert result["feature_engineering"] == mock_feature_diagram

    def test_export_to_mermaid(self):
        """测试导出为Mermaid格式"""
        coordinator = APIFlowCoordinator()

        # 模拟流程图
        mock_diagram = Mock()
        mock_diagram.nodes = []  # 设置为空列表以避免迭代错误
        mock_diagram.edges = []
        coordinator.diagrams = {"test_flow": mock_diagram}

        # Mock导出器
        mock_exporter = Mock()
        coordinator.exporter = mock_exporter

        with tempfile.TemporaryDirectory() as temp_dir:
            result = coordinator.export_to_mermaid(temp_dir)

            # 验证导出器被调用
            mock_exporter.export_to_mermaid.assert_called_once_with(mock_diagram, str(Path(temp_dir) / "test_flow.md"))

            # 验证返回的文件路径
            assert "test_flow" in result
            assert result["test_flow"] == str(Path(temp_dir) / "test_flow.md")

    def test_export_to_json(self):
        """测试导出为JSON格式"""
        coordinator = APIFlowCoordinator()

        # 模拟流程图
        mock_diagram = Mock()
        mock_diagram.nodes = []
        mock_diagram.edges = []
        coordinator.diagrams = {"test_flow": mock_diagram}

        # Mock导出器
        mock_exporter = Mock()
        coordinator.exporter = mock_exporter

        with tempfile.TemporaryDirectory() as temp_dir:
            result = coordinator.export_to_json(temp_dir)

            # 验证导出器被调用
            mock_exporter.export_to_json.assert_called_once_with(mock_diagram, str(Path(temp_dir) / "test_flow.json"))

            # 验证返回的文件路径
            assert "test_flow" in result
            assert result["test_flow"] == str(Path(temp_dir) / "test_flow.json")

    def test_get_flow_statistics(self):
        """测试获取流程统计信息"""
        coordinator = APIFlowCoordinator()

        # 模拟流程图
        mock_node1 = Mock()
        mock_node1.node_type = "input"
        mock_node2 = Mock()
        mock_node2.node_type = "process"

        mock_edge1 = Mock()
        mock_edge1.edge_type = "flow"

        mock_diagram1 = Mock()
        mock_diagram1.nodes = [mock_node1]
        mock_diagram1.edges = [mock_edge1]

        mock_node3 = Mock()
        mock_node3.node_type = "output"
        mock_node4 = Mock()
        mock_node4.node_type = "decision"

        mock_edge2 = Mock()
        mock_edge2.edge_type = "conditional"

        mock_diagram2 = Mock()
        mock_diagram2.nodes = [mock_node3, mock_node4]
        mock_diagram2.edges = [mock_edge2]

        coordinator.diagrams = {
            "flow1": mock_diagram1,
            "flow2": mock_diagram2
        }

        stats = coordinator.get_flow_statistics()

        assert isinstance(stats, dict)
        assert stats["total_flows"] == 2
        assert "flows" in stats
        assert "flow1" in stats["flows"]
        assert "flow2" in stats["flows"]


class TestAPIFlowCoordinatorIntegration:
    """APIFlowCoordinator 集成测试"""

    def test_complete_flow_generation_workflow(self):
        """测试完整的流程图生成工作流"""
        coordinator = APIFlowCoordinator()

        # 1. 生成数据服务流程图
        data_config = {
            "service_name": "UserDataService",
            "endpoints": ["GET /users", "POST /users"]
        }

        # 2. 生成交易服务流程图
        trading_config = {
            "symbol": "BTC/USDT",
            "strategy": "momentum"
        }

        # 3. 生成特征工程流程图
        feature_config = {
            "model": "xgboost",
            "features": ["price", "volume", "momentum"]
        }

        # 模拟生成器返回结果
        mock_data_node = Mock()
        mock_data_node.node_type = "input"
        mock_data_edge = Mock()
        mock_data_edge.edge_type = "flow"
        mock_data_diagram = Mock()
        mock_data_diagram.nodes = [mock_data_node]
        mock_data_diagram.edges = [mock_data_edge]
        mock_data_diagram.id = "data_service"
        coordinator.data_gen.generate = Mock(return_value=mock_data_diagram)

        mock_trading_node = Mock()
        mock_trading_node.node_type = "process"
        mock_trading_edge = Mock()
        mock_trading_edge.edge_type = "flow"
        mock_trading_diagram = Mock()
        mock_trading_diagram.nodes = [mock_trading_node]
        mock_trading_diagram.edges = [mock_trading_edge]
        mock_trading_diagram.id = "trading"
        coordinator.trading_gen.generate = Mock(return_value=mock_trading_diagram)

        mock_feature_node = Mock()
        mock_feature_node.node_type = "output"
        mock_feature_edge = Mock()
        mock_feature_edge.edge_type = "flow"
        mock_feature_diagram = Mock()
        mock_feature_diagram.nodes = [mock_feature_node]
        mock_feature_diagram.edges = [mock_feature_edge]
        mock_feature_diagram.id = "feature_engineering"
        coordinator.feature_gen.generate = Mock(return_value=mock_feature_diagram)

        # 生成所有流程图
        data_diagram = coordinator.create_data_service_flow()
        trading_diagram = coordinator.create_trading_flow()
        feature_diagram = coordinator.create_feature_engineering_flow()

        # 验证结果
        assert len(data_diagram.nodes) == 1
        assert len(trading_diagram.nodes) == 1
        assert len(feature_diagram.nodes) == 1
        assert data_diagram.id == "data_service"
        assert trading_diagram.id == "trading"
        assert feature_diagram.id == "feature_engineering"

        # 验证统计信息
        stats = coordinator.get_flow_statistics()
        assert "flows" in stats
        assert "data_service" in stats["flows"]
        assert "trading" in stats["flows"]
        assert "feature_engineering" in stats["flows"]
        assert stats["total_flows"] == 3

    def test_file_based_export_workflow(self):
        """测试基于文件的导出工作流"""
        coordinator = APIFlowCoordinator()

        # 模拟流程图
        mock_diagram = Mock()
        mock_diagram.nodes = [Mock()]
        mock_diagram.edges = [Mock()]
        coordinator.diagrams = {"test_flow": mock_diagram}

        # Mock导出器
        mock_exporter = Mock()
        coordinator.exporter = mock_exporter

        with tempfile.TemporaryDirectory() as temp_dir:
            # 导出为不同格式
            json_result = coordinator.export_to_json(temp_dir)
            mermaid_result = coordinator.export_to_mermaid(temp_dir)

            # 验证导出器被调用
            mock_exporter.export_to_json.assert_called_once()
            mock_exporter.export_to_mermaid.assert_called_once()

            # 验证返回结果
            assert "test_flow" in json_result
            assert "test_flow" in mermaid_result

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        coordinator = APIFlowCoordinator()

        # 测试生成所有流程图
        result = coordinator.generate_all_flows()
        assert isinstance(result, dict)
        assert len(result) == 3

        # 验证统计信息可用
        stats = coordinator.get_flow_statistics()
        assert isinstance(stats, dict)
        assert stats["total_flows"] == 3
        assert "flows" in stats

    def test_service_isolation(self):
        """测试服务隔离"""
        coordinator1 = APIFlowCoordinator()
        coordinator2 = APIFlowCoordinator()

        # 在第一个协调器中修改生成器
        original_data_gen = coordinator1.data_gen
        mock_gen = Mock()
        mock_gen.generate.return_value = Mock()
        coordinator1.data_gen = mock_gen

        # 生成流程图
        result1 = coordinator1.create_data_service_flow()
        result2 = coordinator2.create_data_service_flow()

        # 验证结果不同（一个是mock，一个是真实生成器）
        assert result1 != result2

        # 验证第二个协调器未受影响
        assert coordinator2.data_gen != mock_gen

    def test_configuration_persistence(self):
        """测试配置持久性"""
        coordinator = APIFlowCoordinator()

        # Mock生成器来验证调用
        mock_generate = Mock()
        coordinator.data_gen.generate = mock_generate

        # 多次生成流程图
        result1 = coordinator.create_data_service_flow()
        result2 = coordinator.create_data_service_flow()

        # 验证生成器被调用了两次
        assert mock_generate.call_count == 2
