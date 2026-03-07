"""
API流程图生成器重构版本测试
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.api.api_flow_diagram_generator_refactored import APIFlowDiagramGenerator


class TestAPIFlowDiagramGenerator:
    """测试API流程图生成器"""

    @pytest.fixture
    def generator(self):
        """创建生成器实例"""
        return APIFlowDiagramGenerator()

    @pytest.fixture
    def mock_flow_diagram(self):
        """创建模拟流程图"""
        diagram = Mock()
        diagram.title = "Test Flow"
        diagram.nodes = [
            Mock(node_id="start", label="Start", node_type="start", position=(0, 0)),
            Mock(node_id="process", label="Process", node_type="process", position=(100, 0)),
            Mock(node_id="end", label="End", node_type="end", position=(200, 0))
        ]
        diagram.edges = [
            Mock(from_node="start", to_node="process", label="flow1", condition=None),
            Mock(from_node="process", to_node="end", label="flow2", condition=None)
        ]
        return diagram

    def test_init(self, generator):
        """测试初始化"""
        assert hasattr(generator, 'strategies')
        assert 'data_service' in generator.strategies
        assert 'trading' in generator.strategies
        assert 'feature_engineering' in generator.strategies

    @patch('src.infrastructure.api.flow_generation.strategies.DataServiceFlowStrategy.generate_flow')
    def test_create_data_service_flow(self, mock_generate, generator, mock_flow_diagram):
        """测试创建数据服务流程"""
        mock_generate.return_value = mock_flow_diagram

        result = generator.create_data_service_flow()

        assert result == mock_flow_diagram
        mock_generate.assert_called_once()

    @patch('src.infrastructure.api.flow_generation.strategies.TradingFlowStrategy.generate_flow')
    def test_create_trading_flow(self, mock_generate, generator, mock_flow_diagram):
        """测试创建交易流程"""
        mock_generate.return_value = mock_flow_diagram

        result = generator.create_trading_flow()

        assert result == mock_flow_diagram
        mock_generate.assert_called_once()

    @patch('src.infrastructure.api.flow_generation.strategies.FeatureFlowStrategy.generate_flow')
    def test_create_feature_engineering_flow(self, mock_generate, generator, mock_flow_diagram):
        """测试创建特征工程流程"""
        mock_generate.return_value = mock_flow_diagram

        result = generator.create_feature_engineering_flow()

        assert result == mock_flow_diagram
        mock_generate.assert_called_once()

    @patch('src.infrastructure.api.flow_generation.strategies.DataServiceFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.TradingFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.FeatureFlowStrategy.generate_flow')
    def test_generate_all_flows(self, mock_feature, mock_trading, mock_data, generator, mock_flow_diagram):
        """测试生成所有流程"""
        mock_data.return_value = mock_flow_diagram
        mock_trading.return_value = mock_flow_diagram
        mock_feature.return_value = mock_flow_diagram

        result = generator.generate_all_flows()

        assert len(result) == 3
        assert 'data_service' in result
        assert 'trading' in result
        assert 'feature_engineering' in result

        mock_data.assert_called_once()
        mock_trading.assert_called_once()
        mock_feature.assert_called_once()

    @patch('src.infrastructure.api.flow_generation.strategies.DataServiceFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.TradingFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.FeatureFlowStrategy.generate_flow')
    def test_export_to_mermaid(self, mock_feature, mock_trading, mock_data, generator, mock_flow_diagram):
        """测试导出为Mermaid格式"""
        mock_data.return_value = mock_flow_diagram
        mock_trading.return_value = mock_flow_diagram
        mock_feature.return_value = mock_flow_diagram

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generator.export_to_mermaid(temp_dir)

            assert len(result) == 3
            assert all('.mmd' in path for path in result.values())

            # 检查文件是否创建
            for file_path in result.values():
                assert Path(file_path).exists()

    @patch('src.infrastructure.api.flow_generation.strategies.DataServiceFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.TradingFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.FeatureFlowStrategy.generate_flow')
    def test_export_to_json(self, mock_feature, mock_trading, mock_data, generator, mock_flow_diagram):
        """测试导出为JSON格式"""
        mock_data.return_value = mock_flow_diagram
        mock_trading.return_value = mock_flow_diagram
        mock_feature.return_value = mock_flow_diagram

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generator.export_to_json(temp_dir)

            assert len(result) == 3
            assert all('.json' in path for path in result.values())

            # 检查文件是否创建
            for file_path in result.values():
                assert Path(file_path).exists()

                # 检查JSON内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assert 'title' in data
                    assert 'nodes' in data
                    assert 'edges' in data

    def test_convert_diagram_to_dict(self, generator, mock_flow_diagram):
        """测试流程图转字典"""
        result = generator._convert_diagram_to_dict(mock_flow_diagram)

        assert result['title'] == 'Test Flow'
        assert 'nodes' in result
        assert 'edges' in result
        assert len(result['nodes']) == 3
        assert len(result['edges']) == 2

    def test_convert_nodes_to_dict(self, generator, mock_flow_diagram):
        """测试节点转字典"""
        result = generator._convert_nodes_to_dict(mock_flow_diagram.nodes)

        assert len(result) == 3
        assert result[0]['id'] == 'start'
        assert result[0]['label'] == 'Start'
        assert result[0]['type'] == 'start'
        assert result[0]['position'] == (0, 0)

    def test_convert_edges_to_dict(self, generator, mock_flow_diagram):
        """测试边转字典"""
        result = generator._convert_edges_to_dict(mock_flow_diagram.edges)

        assert len(result) == 2
        assert result[0]['source'] == 'start'
        assert result[0]['target'] == 'process'
        assert result[0]['label'] == 'flow1'

    def test_write_json_file(self, generator):
        """测试写入JSON文件"""
        test_data = {"test": "data"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_path = Path(f.name)

        try:
            generator._write_json_file(file_path, test_data)

            assert file_path.exists()
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data

        finally:
            file_path.unlink(missing_ok=True)

    @patch('src.infrastructure.api.flow_generation.strategies.DataServiceFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.TradingFlowStrategy.generate_flow')
    @patch('src.infrastructure.api.flow_generation.strategies.FeatureFlowStrategy.generate_flow')
    def test_get_flow_statistics(self, mock_feature, mock_trading, mock_data, generator, mock_flow_diagram):
        """测试获取流程统计"""
        mock_data.return_value = mock_flow_diagram
        mock_trading.return_value = mock_flow_diagram
        mock_feature.return_value = mock_flow_diagram

        result = generator.get_flow_statistics()

        assert 'total_flows' in result
        assert 'total_nodes' in result
        assert 'total_edges' in result
        assert 'flows' in result
        assert len(result['flows']) == 3

    def test_generate_mermaid(self, generator, mock_flow_diagram):
        """测试生成Mermaid图"""
        result = generator._generate_mermaid(mock_flow_diagram)

        assert isinstance(result, str)
        assert 'graph LR' in result
        assert 'start[' in result
        assert 'process[' in result
        assert 'end[' in result

    def test_get_node_shape(self, generator):
        """测试获取节点形状"""
        # 测试不同类型的节点形状
        start_shape = generator._get_node_shape('start')
        assert start_shape == ('([', '])')

        process_shape = generator._get_node_shape('process')
        assert process_shape == ('[', ']')

        decision_shape = generator._get_node_shape('decision')
        assert decision_shape == ('{', '}')

        end_shape = generator._get_node_shape('end')
        assert end_shape == ('([', '])')

        # 测试未知类型
        unknown_shape = generator._get_node_shape('unknown')
        assert unknown_shape == ('[', ']')

    def test_get_strategy(self, generator):
        """测试获取策略"""
        data_strategy = generator.get_strategy('data_service')
        assert data_strategy is not None

        trading_strategy = generator.get_strategy('trading')
        assert trading_strategy is not None

        feature_strategy = generator.get_strategy('feature_engineering')
        assert feature_strategy is not None

    def test_get_strategy_not_found(self, generator):
        """测试获取不存在的策略"""
        result = generator.get_strategy('nonexistent')
        assert result is None

    def test_add_strategy(self, generator):
        """测试添加策略"""
        mock_strategy = Mock()
        generator.add_strategy('custom', mock_strategy)

        assert 'custom' in generator.strategies
        assert generator.strategies['custom'] is mock_strategy


class TestAPIFlowDiagramGeneratorIntegration:
    """集成测试"""

    @pytest.fixture
    def generator(self):
        """创建生成器实例"""
        return APIFlowDiagramGenerator()

    def test_complete_workflow(self, generator):
        """测试完整工作流程"""
        # 生成所有流程
        flows = generator.generate_all_flows()
        assert len(flows) == 3

        # 导出为JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            json_files = generator.export_to_json(temp_dir)
            assert len(json_files) == 3

            # 检查每个JSON文件
            for flow_name, file_path in json_files.items():
                assert Path(file_path).exists()
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    assert 'title' in data
                    assert 'nodes' in data
                    assert 'edges' in data

        # 导出为Mermaid
        with tempfile.TemporaryDirectory() as temp_dir:
            mermaid_files = generator.export_to_mermaid(temp_dir)
            assert len(mermaid_files) == 3

            # 检查每个Mermaid文件
            for flow_name, file_path in mermaid_files.items():
                assert Path(file_path).exists()
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert 'graph LR' in content

        # 获取统计信息
        stats = generator.get_flow_statistics()
        assert stats['total_flows'] == 3
        assert 'total_nodes' in stats
        assert 'total_edges' in stats

    def test_strategy_management(self, generator):
        """测试策略管理"""
        # 获取现有策略
        data_strategy = generator.get_strategy('data_service')
        assert data_strategy is not None

        # 添加自定义策略
        custom_strategy = Mock()
        generator.add_strategy('test_strategy', custom_strategy)

        # 验证添加成功
        retrieved = generator.get_strategy('test_strategy')
        assert retrieved is custom_strategy

    def test_error_handling(self, generator):
        """测试错误处理"""
        # 测试获取不存在的策略
        result = generator.get_strategy('invalid_type')
        assert result is None

        # 测试添加无效策略
        generator.add_strategy('empty', None)
        result = generator.get_strategy('empty')
        assert result is None
