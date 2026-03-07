"""
流程图配置测试
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.api.configs.flow_configs import (
    FlowNodeConfig,
    FlowEdgeConfig,
    FlowGenerationConfig,
    FlowExportConfig
)
from src.infrastructure.api.configs.base_config import ValidationResult


class TestFlowNodeConfig:
    """测试流程节点配置"""

    def setup_method(self):
        """测试前设置lenient验证模式"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        # 临时patch __post_init__ 以禁用strict验证
        original_post_init = BaseConfig.__post_init__
        def patched_post_init(self):
            if type(self) is BaseConfig:
                self._last_validation_result = ValidationResult()
                return
            validation = self.validate()
            self._last_validation_result = validation
            # 注释掉抛出异常的代码，改为lenient模式
            # if not validation.is_valid and self._validation_mode == "strict":
            #     raise ValueError(f"配置验证失败: {validation}")
        BaseConfig.__post_init__ = patched_post_init
        self._original_post_init = original_post_init

    def teardown_method(self):
        """测试后恢复原始方法"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        BaseConfig.__post_init__ = self._original_post_init

    def test_init_basic(self):
        """测试基本初始化"""
        config = FlowNodeConfig(
            node_id="start_1",
            label="Start",
            node_type="start"
        )

        assert config.node_id == "start_1"
        assert config.label == "Start"
        assert config.node_type == "start"
        assert config.description is None
        assert config.position is None
        assert config.metadata == {}
        assert config.style is None

    def test_init_complete(self):
        """测试完整初始化"""
        position = {"x": 100, "y": 50}
        metadata = {"service": "user", "version": "1.0"}
        style = {"color": "blue", "shape": "circle"}

        config = FlowNodeConfig(
            node_id="process_1",
            label="Process User",
            node_type="process",
            description="Process user request",
            position=position,
            metadata=metadata,
            style=style
        )

        assert config.node_id == "process_1"
        assert config.label == "Process User"
        assert config.node_type == "process"
        assert config.description == "Process user request"
        assert config.position == position
        assert config.metadata == metadata
        assert config.style == style

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = FlowNodeConfig(
            node_id="start_1",
            label="Start",
            node_type="start"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_id(self):
        """测试验证缺失ID"""
        config = FlowNodeConfig(
            node_id="",
            label="Start",
            node_type="start"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "节点ID不能为空" in result.errors[0]

    def test_validate_invalid_type(self):
        """测试验证无效类型"""
        config = FlowNodeConfig(
            node_id="node_1",
            label="Node",
            node_type="invalid"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "节点类型必须是" in result.errors[0]

    def test_validate_invalid_position(self):
        """测试验证无效位置"""
        config = FlowNodeConfig(
            node_id="node_1",
            label="Node",
            node_type="process",
            position={"x": 100}  # 缺少y坐标
        )

        result = config.validate()
        assert result.is_valid is False
        assert "位置信息必须包含x和y坐标" in result.errors[0]


class TestFlowEdgeConfig:
    """测试流程边配置"""

    def setup_method(self):
        """测试前设置lenient验证模式"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        # 临时patch __post_init__ 以禁用strict验证
        original_post_init = BaseConfig.__post_init__
        def patched_post_init(self):
            if type(self) is BaseConfig:
                self._last_validation_result = ValidationResult()
                return
            validation = self.validate()
            self._last_validation_result = validation
            # 注释掉抛出异常的代码，改为lenient模式
            # if not validation.is_valid and self._validation_mode == "strict":
            #     raise ValueError(f"配置验证失败: {validation}")
        BaseConfig.__post_init__ = patched_post_init
        self._original_post_init = original_post_init

    def teardown_method(self):
        """测试后恢复原始方法"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        BaseConfig.__post_init__ = self._original_post_init

    def test_init_basic(self):
        """测试基本初始化"""
        config = FlowEdgeConfig(
            edge_id="edge_1",
            from_node="node_1",
            to_node="node_2"
        )

        assert config.edge_id == "edge_1"
        assert config.from_node == "node_1"
        assert config.to_node == "node_2"
        assert config.label is None
        assert config.condition is None
        assert config.arrow_type == "normal"

    def test_init_complete(self):
        """测试完整初始化"""
        config = FlowEdgeConfig(
            edge_id="edge_1",
            from_node="node_1",
            to_node="node_2",
            label="success",
            condition="status == 200",
            arrow_type="thick"
        )

        assert config.edge_id == "edge_1"
        assert config.from_node == "node_1"
        assert config.to_node == "node_2"
        assert config.label == "success"
        assert config.condition == "status == 200"
        assert config.arrow_type == "thick"

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = FlowEdgeConfig(
            edge_id="edge_1",
            from_node="node_1",
            to_node="node_2"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_nodes(self):
        """测试验证缺失节点"""
        config = FlowEdgeConfig(
            edge_id="edge_1",
            from_node="",
            to_node="node_2"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "起始节点不能为空" in result.errors[0]


class TestFlowGenerationConfig:
    """测试流程生成配置"""

    def setup_method(self):
        """测试前设置lenient验证模式"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        # 临时patch __post_init__ 以禁用strict验证
        original_post_init = BaseConfig.__post_init__
        def patched_post_init(self):
            if type(self) is BaseConfig:
                self._last_validation_result = ValidationResult()
                return
            validation = self.validate()
            self._last_validation_result = validation
            # 注释掉抛出异常的代码，改为lenient模式
            # if not validation.is_valid and self._validation_mode == "strict":
            #     raise ValueError(f"配置验证失败: {validation}")
        BaseConfig.__post_init__ = patched_post_init
        self._original_post_init = original_post_init

    def teardown_method(self):
        """测试后恢复原始方法"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        BaseConfig.__post_init__ = self._original_post_init

    def test_init_basic(self):
        """测试基本初始化"""
        config = FlowGenerationConfig(
            title="User Flow",
            flow_type="data_service"
        )

        assert config.title == "User Flow"
        assert config.flow_type == "data_service"
        assert config.description is None
        assert config.nodes == []
        assert config.edges == []

    def test_init_complete(self):
        """测试完整初始化"""
        nodes = [
            FlowNodeConfig(
                node_id="start",
                label="Start",
                node_type="start"
            )
        ]

        edges = [
            FlowEdgeConfig(
                edge_id="edge_1",
                from_node="start",
                to_node="end"
            )
        ]

        config = FlowGenerationConfig(
            title="Complete Flow",
            flow_type="trading",
            description="Complete trading flow",
            nodes=nodes,
            edges=edges
        )

        assert config.title == "Complete Flow"
        assert config.flow_type == "trading"
        assert config.description == "Complete trading flow"
        assert len(config.nodes) == 1
        assert len(config.edges) == 1

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = FlowGenerationConfig(
            title="Test Flow",
            flow_type="data_service"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_title(self):
        """测试验证缺失标题"""
        config = FlowGenerationConfig(
            title="",
            flow_type="data_service"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "流程标题不能为空" in result.errors[0]


class TestFlowExportConfig:
    """测试流程导出配置"""

    def setup_method(self):
        """测试前设置lenient验证模式"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        # 临时patch __post_init__ 以禁用strict验证
        original_post_init = BaseConfig.__post_init__
        def patched_post_init(self):
            if type(self) is BaseConfig:
                self._last_validation_result = ValidationResult()
                return
            validation = self.validate()
            self._last_validation_result = validation
            # 注释掉抛出异常的代码，改为lenient模式
            # if not validation.is_valid and self._validation_mode == "strict":
            #     raise ValueError(f"配置验证失败: {validation}")
        BaseConfig.__post_init__ = patched_post_init
        self._original_post_init = original_post_init

    def teardown_method(self):
        """测试后恢复原始方法"""
        from src.infrastructure.api.configs.base_config import BaseConfig
        BaseConfig.__post_init__ = self._original_post_init

    def test_init_basic(self):
        """测试基本初始化"""
        config = FlowExportConfig(
            format="mermaid",
            output_dir="./docs"
        )

        assert config.format == "mermaid"
        assert config.output_dir == "./docs"
        assert config.include_metadata is True
        assert config.compress_output is False

    def test_init_complete(self):
        """测试完整初始化"""
        config = FlowExportConfig(
            format="json",
            output_dir="/tmp/flows",
            include_metadata=False,
            compress_output=True,
            theme="dark"
        )

        assert config.format == "json"
        assert config.output_dir == "/tmp/flows"
        assert config.include_metadata is False
        assert config.compress_output is True
        assert config.theme == "dark"

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = FlowExportConfig(
            format="mermaid",
            output_dir="./docs"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_invalid_format(self):
        """测试验证无效格式"""
        config = FlowExportConfig(
            format="invalid",
            output_dir="./docs"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "不支持的导出格式" in result.errors[0]
