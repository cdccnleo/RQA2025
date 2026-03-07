"""
Schema配置测试
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.api.configs.schema_configs import (
    SchemaPropertyConfig,
    SchemaDefinitionConfig,
    SchemaGenerationConfig
)


class TestSchemaPropertyConfig:
    """测试Schema属性配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = SchemaPropertyConfig(
            name="id",
            property_type="integer"
        )

        assert config.name == "id"
        assert config.property_type == "integer"
        assert config.description is None
        assert config.required is False
        assert config.nullable is False

    def test_init_string_property(self):
        """测试字符串属性初始化"""
        config = SchemaPropertyConfig(
            name="name",
            property_type="string",
            description="User name",
            min_length=1,
            max_length=100,
            pattern=r"^[a-zA-Z\s]+$",
            example="John Doe"
        )

        assert config.name == "name"
        assert config.property_type == "string"
        assert config.description == "User name"
        assert config.min_length == 1
        assert config.max_length == 100
        assert config.pattern == r"^[a-zA-Z\s]+$"
        assert config.example == "John Doe"

    def test_init_numeric_property(self):
        """测试数值属性初始化"""
        config = SchemaPropertyConfig(
            name="age",
            property_type="integer",
            minimum=0,
            maximum=150,
            default=18,
            example=25
        )

        assert config.name == "age"
        assert config.property_type == "integer"
        assert config.minimum == 0
        assert config.maximum == 150
        assert config.default == 18
        assert config.example == 25

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = SchemaPropertyConfig(
            name="email",
            property_type="string",
            format="email"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_name(self):
        """测试验证缺失名称"""
        config = SchemaPropertyConfig(
            name="",
            property_type="string"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "属性名称不能为空" in result.errors[0]

    def test_validate_invalid_type(self):
        """测试验证无效类型"""
        config = SchemaPropertyConfig(
            name="field",
            property_type="invalid"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "不支持的属性类型" in result.errors[0]


class TestSchemaDefinitionConfig:
    """测试Schema定义配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = SchemaDefinitionConfig(
            schema_name="User",
            schema_type="object"
        )

        assert config.schema_name == "User"
        assert config.schema_type == "object"
        assert config.description is None
        assert config.properties == []

    def test_init_complete(self):
        """测试完整初始化"""
        properties = [
            SchemaPropertyConfig(
                name="id",
                property_type="integer",
                required=True
            ),
            SchemaPropertyConfig(
                name="name",
                property_type="string",
                required=True,
                min_length=1,
                max_length=100
            )
        ]

        config = SchemaDefinitionConfig(
            schema_name="CompleteUser",
            schema_type="object",
            description="Complete user schema",
            properties=properties
        )

        assert config.schema_name == "CompleteUser"
        assert config.schema_type == "object"
        assert config.description == "Complete user schema"
        assert len(config.properties) == 2

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = SchemaDefinitionConfig(
            schema_name="User",
            schema_type="object"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_missing_name(self):
        """测试验证缺失名称"""
        config = SchemaDefinitionConfig(
            schema_name="",
            schema_type="object"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "Schema名称不能为空" in result.errors[0]

    def test_validate_invalid_type(self):
        """测试验证无效类型"""
        config = SchemaDefinitionConfig(
            schema_name="User",
            schema_type="invalid"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "不支持的Schema类型" in result.errors[0]


class TestSchemaGenerationConfig:
    """测试Schema生成配置"""

    def test_init_basic(self):
        """测试基本初始化"""
        config = SchemaGenerationConfig(
            output_format="json",
            output_dir="./schemas"
        )

        assert config.output_format == "json"
        assert config.output_dir == "./schemas"
        assert config.include_examples is True
        assert config.strict_validation is False

    def test_init_complete(self):
        """测试完整初始化"""
        config = SchemaGenerationConfig(
            output_format="yaml",
            output_dir="/tmp/schemas",
            include_examples=False,
            strict_validation=True,
            custom_validators=[]
        )

        assert config.output_format == "yaml"
        assert config.output_dir == "/tmp/schemas"
        assert config.include_examples is False
        assert config.strict_validation is True
        assert config.custom_validators == []

    def test_validate_valid_config(self):
        """测试验证有效配置"""
        config = SchemaGenerationConfig(
            output_format="json",
            output_dir="./schemas"
        )

        result = config.validate()
        assert result.is_valid is True

    def test_validate_invalid_format(self):
        """测试验证无效格式"""
        config = SchemaGenerationConfig(
            output_format="invalid",
            output_dir="./schemas"
        )

        result = config.validate()
        assert result.is_valid is False
        assert "不支持的输出格式" in result.errors[0]
