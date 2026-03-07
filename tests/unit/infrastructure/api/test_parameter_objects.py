"""
测试API模块参数对象定义

覆盖 parameter_objects.py 中定义的所有数据类
"""

import pytest
from pathlib import Path
from datetime import datetime
from src.infrastructure.api.parameter_objects import (
    TestCaseConfig,
    TestScenarioConfig,
    TestSuiteExportConfig,
    DocumentationExportConfig,
    DocumentationConfig,
    EndpointDocumentationConfig,
    SearchConfig,
    FlowNodeConfig,
    FlowConnectionConfig,
    FlowDiagramConfig,
    FlowExportConfig,
    OpenAPIConfig,
    SchemaGenerationConfig,
    EndpointGenerationConfig,
    VersionCreationConfig,
    VersionComparisonConfig,
    MetricRecordConfig,
    AlertConfig,
    DashboardConfig,
    ResourceQuotaConfig,
    ResourceAllocationConfig,
    CacheConfig,
    LogConfig
)


class TestTestCaseConfig:
    """TestCaseConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = TestCaseConfig(
            title="Test Title",
            description="Test Description"
        )

        assert config.title == "Test Title"
        assert config.description == "Test Description"
        assert config.priority == "medium"
        assert config.category == "functional"
        assert config.preconditions == []
        assert config.test_steps == []
        assert config.expected_results == []
        assert config.tags == []
        assert config.environment == "test"

    def test_initialization_all_parameters(self):
        """测试所有参数初始化"""
        config = TestCaseConfig(
            title="Complete Test",
            description="Complete test description",
            priority="high",
            category="integration",
            preconditions=["Setup DB", "Login user"],
            test_steps=[{"action": "click", "target": "button"}],
            expected_results=["Button clicked", "Page updated"],
            tags=["smoke", "critical"],
            environment="production"
        )

        assert config.title == "Complete Test"
        assert config.description == "Complete test description"
        assert config.priority == "high"
        assert config.category == "integration"
        assert config.preconditions == ["Setup DB", "Login user"]
        assert config.test_steps == [{"action": "click", "target": "button"}]
        assert config.expected_results == ["Button clicked", "Page updated"]
        assert config.tags == ["smoke", "critical"]
        assert config.environment == "production"

    def test_default_lists_are_independent(self):
        """测试默认列表是独立的（避免共享引用问题）"""
        config1 = TestCaseConfig(title="Test1", description="Desc1")
        config2 = TestCaseConfig(title="Test2", description="Desc2")

        config1.preconditions.append("Precondition1")
        config1.tags.append("tag1")

        # config2的默认列表应该不受影响
        assert config2.preconditions == []
        assert config2.tags == []


class TestTestScenarioConfig:
    """TestScenarioConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = TestScenarioConfig(
            name="Login Scenario",
            description="User login test scenario",
            endpoint="/api/login",
            method="POST"
        )

        assert config.name == "Login Scenario"
        assert config.description == "User login test scenario"
        assert config.endpoint == "/api/login"
        assert config.method == "POST"
        assert config.setup_steps == []
        assert config.teardown_steps == []
        assert config.variables == {}

    def test_initialization_with_optional(self):
        """测试包含可选参数的初始化"""
        config = TestScenarioConfig(
            name="Complex Scenario",
            description="Complex test scenario",
            endpoint="/api/complex",
            method="PUT",
            setup_steps=["Initialize DB", "Create test user"],
            teardown_steps=["Clean up DB", "Logout user"],
            variables={"user_id": 123, "token": "abc123"}
        )

        assert config.name == "Complex Scenario"
        assert config.endpoint == "/api/complex"
        assert config.method == "PUT"
        assert config.setup_steps == ["Initialize DB", "Create test user"]
        assert config.teardown_steps == ["Clean up DB", "Logout user"]
        assert config.variables == {"user_id": 123, "token": "abc123"}


class TestTestSuiteExportConfig:
    """TestSuiteExportConfig 数据类测试"""

    def test_initialization_defaults(self):
        """测试默认初始化"""
        config = TestSuiteExportConfig()

        assert config.format_type == "json"
        assert config.output_dir == Path("docs/api/tests")
        assert config.include_timestamps == True
        assert config.include_statistics == True
        assert config.include_metadata == True
        assert config.pretty_print == True
        assert config.compress == False

    def test_initialization_custom(self):
        """测试自定义初始化"""
        output_dir = Path("/tmp/tests")
        config = TestSuiteExportConfig(
            format_type="yaml",
            output_dir=output_dir,
            include_timestamps=False,
            include_statistics=False,
            include_metadata=False,
            pretty_print=False,
            compress=True
        )

        assert config.format_type == "yaml"
        assert config.output_dir == output_dir
        assert config.include_timestamps == False
        assert config.include_statistics == False
        assert config.include_metadata == False
        assert config.pretty_print == False
        assert config.compress == True


class TestDocumentationExportConfig:
    """DocumentationExportConfig 数据类测试"""

    def test_initialization_defaults(self):
        """测试默认初始化"""
        config = DocumentationExportConfig()

        assert config.output_dir == "docs/api"
        assert config.include_examples == True
        assert config.include_statistics == True
        assert config.format_types == ["json", "yaml"]
        assert config.pretty_print == True
        assert config.include_metadata == True
        assert config.compress == False
        assert config.theme == "default"

    def test_initialization_custom(self):
        """测试自定义初始化"""
        config = DocumentationExportConfig(
            output_dir="/tmp/docs",
            include_examples=False,
            include_statistics=False,
            format_types=["html", "pdf"],
            pretty_print=False,
            include_metadata=False,
            compress=True,
            theme="dark"
        )

        assert config.output_dir == "/tmp/docs"
        assert config.include_examples == False
        assert config.include_statistics == False
        assert config.format_types == ["html", "pdf"]
        assert config.pretty_print == False
        assert config.include_metadata == False
        assert config.compress == True
        assert config.theme == "dark"


class TestDocumentationConfig:
    """DocumentationConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = DocumentationConfig(title="API Documentation")

        assert config.title == "API Documentation"
        assert config.version == "1.0.0"
        assert config.description == ""
        assert config.base_url == "/api/v1"

    def test_initialization_all_parameters(self):
        """测试所有参数初始化"""
        config = DocumentationConfig(
            title="Complete API Docs",
            version="2.1.0",
            description="Complete API documentation",
            base_url="/api/v2"
        )

        assert config.title == "Complete API Docs"
        assert config.version == "2.1.0"
        assert config.description == "Complete API documentation"
        assert config.base_url == "/api/v2"


class TestEndpointDocumentationConfig:
    """EndpointDocumentationConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = EndpointDocumentationConfig(
            path="/users",
            method="GET",
            summary="Get users list"
        )

        assert config.path == "/users"
        assert config.method == "GET"
        assert config.summary == "Get users list"
        assert config.description == ""
        assert config.parameters == []
        assert config.responses == {}
        assert config.tags == []

    def test_initialization_complete(self):
        """测试完整初始化"""
        config = EndpointDocumentationConfig(
            path="/users/{id}",
            method="PUT",
            summary="Update user",
            description="Update user information",
            parameters=[{"name": "id", "type": "integer"}],
            responses={"200": {"description": "Success"}},
            tags=["users", "update"]
        )

        assert config.path == "/users/{id}"
        assert config.method == "PUT"
        assert config.summary == "Update user"
        assert config.description == "Update user information"
        assert config.parameters == [{"name": "id", "type": "integer"}]
        assert config.responses == {"200": {"description": "Success"}}
        assert config.tags == ["users", "update"]


class TestFlowNodeConfig:
    """FlowNodeConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = FlowNodeConfig(
            node_id="node1",
            node_type="start",
            label="Start Node"
        )

        assert config.node_id == "node1"
        assert config.node_type == "start"
        assert config.label == "Start Node"
        assert config.description == ""
        assert config.properties == {}
        assert config.style == {}

    def test_initialization_complete(self):
        """测试完整初始化"""
        config = FlowNodeConfig(
            node_id="process_node",
            node_type="processor",
            label="Process Data",
            description="Data processing node",
            properties={"config": "value"},
            style={"color": "blue"}
        )

        assert config.node_id == "process_node"
        assert config.node_type == "processor"
        assert config.label == "Process Data"
        assert config.description == "Data processing node"
        assert config.properties == {"config": "value"}
        assert config.style == {"color": "blue"}


class TestFlowConnectionConfig:
    """FlowConnectionConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = FlowConnectionConfig(
            from_node="node1",
            to_node="node2"
        )

        assert config.from_node == "node1"
        assert config.to_node == "node2"
        assert config.label == ""
        assert config.connection_type == "default"
        assert config.properties == {}

    def test_initialization_complete(self):
        """测试完整初始化"""
        config = FlowConnectionConfig(
            from_node="processor",
            to_node="output",
            label="data flow",
            connection_type="data_flow",
            properties={"bandwidth": "100Mbps"}
        )

        assert config.from_node == "processor"
        assert config.to_node == "output"
        assert config.label == "data flow"
        assert config.connection_type == "data_flow"
        assert config.properties == {"bandwidth": "100Mbps"}


class TestSearchConfig:
    """SearchConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = SearchConfig(query="user authentication")

        assert config.query == "user authentication"
        assert config.search_in_paths == True
        assert config.search_in_methods == True
        assert config.search_in_descriptions == True
        assert config.search_in_parameters == True
        assert config.search_in_responses == True
        assert config.case_sensitive == False
        assert config.max_results == 50
        assert config.min_relevance_score == 0.3

    def test_initialization_complete(self):
        """测试完整初始化"""
        config = SearchConfig(
            query="user authentication",
            search_in_paths=False,
            search_in_methods=True,
            search_in_descriptions=True,
            search_in_parameters=False,
            search_in_responses=True,
            case_sensitive=True,
            max_results=100,
            min_relevance_score=0.5
        )

        assert config.query == "user authentication"
        assert config.search_in_paths == False
        assert config.search_in_methods == True
        assert config.search_in_descriptions == True
        assert config.search_in_parameters == False
        assert config.search_in_responses == True
        assert config.case_sensitive == True
        assert config.max_results == 100
        assert config.min_relevance_score == 0.5


class TestOpenAPIConfig:
    """OpenAPIConfig 数据类测试"""

    def test_initialization_required_only(self):
        """测试仅必需参数初始化"""
        config = OpenAPIConfig(
            title="API Documentation",
            version="1.0.0"
        )

        assert config.title == "API Documentation"
        assert config.version == "1.0.0"
        assert config.description == ""
        assert config.servers == []

    def test_initialization_complete(self):
        """测试完整初始化"""
        config = OpenAPIConfig(
            title="Complete API",
            version="2.1.0",
            description="Complete API documentation",
            servers=[{"url": "https://api.example.com"}]
        )

        assert config.title == "Complete API"
        assert config.version == "2.1.0"
        assert config.description == "Complete API documentation"
        assert config.servers == [{"url": "https://api.example.com"}]




class TestParameterObjectsIntegration:
    """参数对象集成测试"""

    def test_config_workflow(self):
        """测试配置工作流"""
        # 创建测试用例配置
        test_config = TestCaseConfig(
            title="User API Test",
            description="Test user API endpoints",
            priority="high",
            tags=["api", "user"]
        )

        # 创建文档配置
        doc_config = DocumentationConfig(
            title="User API Documentation",
            version="1.0.0",
            description="User management API"
        )

        # 创建导出配置
        export_config = DocumentationExportConfig(
            output_dir="docs/user-api",
            format_types=["json", "html"],
            theme="light"
        )

        # 验证配置关系
        assert test_config.title == "User API Test"
        assert doc_config.title == "User API Documentation"
        assert export_config.output_dir == "docs/user-api"
        assert export_config.theme == "light"

