"""
流程图配置测试

测试覆盖:
- FlowNodeConfig: 流程节点配置
- FlowEdgeConfig: 流程边配置  
- FlowGenerationConfig: 流程生成配置
- FlowExportConfig: 流程导出配置
"""

import pytest

from src.infrastructure.api.configs.flow_configs import (
    FlowNodeConfig,
    FlowEdgeConfig,
    FlowGenerationConfig,
    FlowExportConfig,
)
from src.infrastructure.api.configs.base_config import ExportFormat


class TestFlowNodeConfig:
    """流程节点配置测试"""
    
    def test_create_valid_node(self):
        """测试创建有效节点"""
        node = FlowNodeConfig(
            node_id="node1",
            label="Start",
            node_type="start"
        )
        assert node.node_id == "node1"
        assert node.label == "Start"
        assert node.node_type == "start"
    
    def test_node_validation_empty_id(self):
        """测试空节点ID"""
        # 设置strict模式进行测试
        original_mode = FlowNodeConfig._validation_mode
        FlowNodeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowNodeConfig(
                    node_id="",
                    label="Node",
                    node_type="process"
                )
            assert "节点ID不能为空" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowNodeConfig.set_validation_mode(original_mode)

    def test_node_validation_invalid_type(self):
        """测试无效的节点类型"""
        # 设置strict模式进行测试
        original_mode = FlowNodeConfig._validation_mode
        FlowNodeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowNodeConfig(
                    node_id="node1",
                    label="Node",
                    node_type="invalid"
                )
            assert "节点类型必须是" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowNodeConfig.set_validation_mode(original_mode)
    
    def test_node_with_position(self):
        """测试带位置的节点"""
        node = FlowNodeConfig(
            node_id="node1",
            label="Process",
            node_type="process",
            position={"x": 100, "y": 200}
        )
        assert node.position == {"x": 100, "y": 200}
    
    def test_node_validation_invalid_position(self):
        """测试无效的位置"""
        # 设置strict模式进行测试
        original_mode = FlowNodeConfig._validation_mode
        FlowNodeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowNodeConfig(
                    node_id="node1",
                    label="Node",
                    node_type="process",
                    position={"x": 100}  # 缺少y坐标
                )
            assert "位置信息必须包含x和y坐标" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowNodeConfig.set_validation_mode(original_mode)


class TestFlowEdgeConfig:
    """流程边配置测试"""
    
    def test_create_valid_edge(self):
        """测试创建有效边"""
        edge = FlowEdgeConfig(
            edge_id="edge1",
            from_node="node1",
            to_node="node2"
        )
        assert edge.edge_id == "edge1"
        assert edge.from_node == "node1"
        assert edge.to_node == "node2"
        assert edge.arrow_type == "normal"
    
    def test_edge_validation_empty_id(self):
        """测试空边ID"""
        # 设置strict模式进行测试
        original_mode = FlowEdgeConfig._validation_mode
        FlowEdgeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowEdgeConfig(
                    edge_id="",
                    from_node="node1",
                    to_node="node2"
                )
            assert "边ID不能为空" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowEdgeConfig.set_validation_mode(original_mode)
    
    def test_edge_validation_empty_nodes(self):
        """测试空节点"""
        # 设置strict模式进行测试
        original_mode = FlowEdgeConfig._validation_mode
        FlowEdgeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowEdgeConfig(
                    edge_id="edge1",
                    from_node="",
                    to_node="node2"
                )
            assert "起始节点不能为空" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowEdgeConfig.set_validation_mode(original_mode)
    
    def test_edge_with_condition(self):
        """测试带条件的边"""
        edge = FlowEdgeConfig(
            edge_id="edge1",
            from_node="node1",
            to_node="node2",
            label="Yes",
            condition="value > 0"
        )
        assert edge.label == "Yes"
        assert edge.condition == "value > 0"
    
    def test_edge_validation_invalid_arrow_type(self):
        """测试无效的箭头类型"""
        # 设置strict模式进行测试
        original_mode = FlowEdgeConfig._validation_mode
        FlowEdgeConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowEdgeConfig(
                    edge_id="edge1",
                    from_node="node1",
                    to_node="node2",
                    arrow_type="invalid"
                )
            assert "箭头类型必须是" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowEdgeConfig.set_validation_mode(original_mode)


class TestFlowGenerationConfig:
    """流程生成配置测试"""
    
    def test_create_minimal_flow(self):
        """测试创建最小流程"""
        flow = FlowGenerationConfig(
            flow_id="flow1",
            title="Test Flow",
            description="Test flow description",
            flow_type="data_service"
        )
        assert flow.flow_id == "flow1"
        assert flow.title == "Test Flow"
        assert flow.flow_type == "data_service"
    
    def test_flow_validation_empty_id(self):
        """测试空流程ID"""
        # 设置strict模式进行测试
        original_mode = FlowGenerationConfig._validation_mode
        FlowGenerationConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowGenerationConfig(
                    flow_id="",
                    title="Flow",
                    description="Desc",
                    flow_type="trading"
                )
            assert "流程ID不能为空" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowGenerationConfig.set_validation_mode(original_mode)
    
    def test_flow_validation_invalid_type(self):
        """测试无效的流程类型"""
        # 设置strict模式进行测试
        original_mode = FlowGenerationConfig._validation_mode
        FlowGenerationConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowGenerationConfig(
                    flow_id="flow1",
                    title="Flow",
                    description="Desc",
                    flow_type="invalid"
                )
            assert "流程类型必须是" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowGenerationConfig.set_validation_mode(original_mode)
    
    def test_flow_add_node(self):
        """测试添加节点"""
        flow = FlowGenerationConfig(
            flow_id="flow1",
            title="Test Flow",
            description="Desc",
            flow_type="trading"
        )
        
        node = FlowNodeConfig(
            node_id="node1",
            label="Start",
            node_type="start"
        )
        
        flow.add_node(node)
        assert len(flow.nodes) == 1
        assert flow.nodes[0].node_id == "node1"
    
    def test_flow_add_edge(self):
        """测试添加边"""
        flow = FlowGenerationConfig(
            flow_id="flow1",
            title="Test Flow",
            description="Desc",
            flow_type="trading"
        )
        
        edge = FlowEdgeConfig(
            edge_id="edge1",
            from_node="node1",
            to_node="node2"
        )
        
        flow.add_edge(edge)
        assert len(flow.edges) == 1
        assert flow.edges[0].edge_id == "edge1"
    
    def test_flow_get_node(self):
        """测试获取节点"""
        node = FlowNodeConfig(
            node_id="node1",
            label="Start",
            node_type="start"
        )
        
        flow = FlowGenerationConfig(
            flow_id="flow1",
            title="Test Flow",
            description="Desc",
            flow_type="trading",
            nodes=[node]
        )
        
        found = flow.get_node("node1")
        assert found is not None
        assert found.node_id == "node1"
        
        not_found = flow.get_node("nonexistent")
        assert not_found is None
    
    def test_flow_validation_edge_reference_nonexistent_nodes(self):
        """测试边引用不存在的节点"""
        # 设置strict模式进行测试
        original_mode = FlowGenerationConfig._validation_mode
        FlowGenerationConfig.set_validation_mode("strict")

        try:
            node = FlowNodeConfig(
                node_id="node1",
                label="Start",
                node_type="start"
            )

            edge = FlowEdgeConfig(
                edge_id="edge1",
                from_node="node1",
                to_node="nonexistent"  # 不存在的节点
            )

            with pytest.raises(ValueError) as exc_info:
                FlowGenerationConfig(
                    flow_id="flow1",
                    title="Test Flow",
                    description="Desc",
                    flow_type="trading",
                    nodes=[node],
                    edges=[edge]
                )
            assert "目标节点" in str(exc_info.value)
            assert "不存在" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowGenerationConfig.set_validation_mode(original_mode)
    
    def test_flow_with_complete_structure(self):
        """测试完整的流程结构"""
        # 创建节点
        start_node = FlowNodeConfig(
            node_id="start",
            label="Start",
            node_type="start"
        )
        process_node = FlowNodeConfig(
            node_id="process1",
            label="Process Data",
            node_type="process"
        )
        end_node = FlowNodeConfig(
            node_id="end",
            label="End",
            node_type="end"
        )
        
        # 创建边
        edge1 = FlowEdgeConfig(
            edge_id="edge1",
            from_node="start",
            to_node="process1"
        )
        edge2 = FlowEdgeConfig(
            edge_id="edge2",
            from_node="process1",
            to_node="end"
        )
        
        # 创建流程
        flow = FlowGenerationConfig(
            flow_id="complete_flow",
            title="Complete Flow",
            description="A complete flow example",
            flow_type="data_service",
            nodes=[start_node, process_node, end_node],
            edges=[edge1, edge2]
        )
        
        assert len(flow.nodes) == 3
        assert len(flow.edges) == 2


class TestFlowExportConfig:
    """流程导出配置测试"""
    
    def test_create_valid_export_config(self):
        """测试创建有效导出配置"""
        export_config = FlowExportConfig(
            format=ExportFormat.JSON,
            output_path="output/flow.json"
        )
        assert export_config.format == ExportFormat.JSON
        assert export_config.output_path == "output/flow.json"
        assert export_config.encoding == "utf-8"
    
    def test_export_validation_empty_path(self):
        """测试空输出路径"""
        # 设置strict模式进行测试
        original_mode = FlowExportConfig._validation_mode
        FlowExportConfig.set_validation_mode("strict")

        try:
            with pytest.raises(ValueError) as exc_info:
                FlowExportConfig(
                    format=ExportFormat.MARKDOWN,
                    output_path=""
                )
            assert "输出路径不能为空" in str(exc_info.value)
        finally:
            # 恢复原始模式
            FlowExportConfig.set_validation_mode(original_mode)
    
    def test_export_with_options(self):
        """测试带选项的导出配置"""
        export_config = FlowExportConfig(
            format=ExportFormat.MARKDOWN,
            output_path="docs/flow.md",
            include_metadata=True,
            include_statistics=True,
            pretty_print=False
        )
        assert export_config.include_metadata is True
        assert export_config.include_statistics is True
        assert export_config.pretty_print is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

