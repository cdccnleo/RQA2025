"""
单元测试 - OpenAPI生成器重构版本

测试openapi_generator_refactored.py中的重构功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch
from src.infrastructure.api.openapi_generator_refactored import RQAApiDocumentationGenerator
from src.infrastructure.api.parameter_objects import DocumentationExportConfig


class TestRQAApiDocumentationGenerator:
    """测试RQAApiDocumentationGenerator类"""

    @pytest.fixture
    def generator(self):
        """创建生成器实例"""
        return RQAApiDocumentationGenerator()

    def test_generate_documentation_with_config_object(self, generator):
        """测试使用配置对象生成文档"""
        config = DocumentationExportConfig(
            output_dir="/tmp/test_docs",
            format_types=["json"],
            include_statistics=False
        )

        with patch.object(generator, '_prepare_document_assembler'), \
             patch.object(generator, '_export_documentation_files_with_config') as mock_export, \
             patch.object(generator, '_display_documentation_stats'):

            mock_export.return_value = {"json": "/tmp/test_docs/openapi.json"}

            result = generator.generate_documentation(config)

            assert result == {"json": "/tmp/test_docs/openapi.json"}
            mock_export.assert_called_once()

    def test_generate_documentation_backward_compatibility(self, generator):
        """测试向后兼容性 - 使用字符串参数"""
        with patch.object(generator, '_prepare_document_assembler'), \
             patch.object(generator, '_export_documentation_files') as mock_export, \
             patch.object(generator, '_display_documentation_stats'):

            mock_export.return_value = {"json": "docs/api/openapi.json", "yaml": "docs/api/openapi.yaml"}

            result = generator.generate_documentation("/tmp/test_docs")

            assert "json" in result
            assert "yaml" in result
            mock_export.assert_called_once()

    def test_prepare_document_assembler(self, generator):
        """测试文档组装器准备"""
        with patch.object(generator._doc_assembler, 'set_api_info'), \
             patch.object(generator._doc_assembler, 'add_servers'), \
             patch.object(generator._doc_assembler, 'add_security_schemes'), \
             patch.object(generator._doc_assembler, 'add_endpoints'), \
             patch.object(generator._doc_assembler, 'add_schemas'), \
             patch.object(generator, '_add_standard_tags'):

            generator._prepare_document_assembler()

            # 验证所有设置方法都被调用
            generator._doc_assembler.set_api_info.assert_called_once()
            generator._doc_assembler.add_servers.assert_called_once()
            generator._doc_assembler.add_security_schemes.assert_called_once()
            generator._doc_assembler.add_endpoints.assert_called_once()
            generator._doc_assembler.add_schemas.assert_called_once()

    def test_add_standard_tags(self, generator):
        """测试添加标准标签"""
        with patch.object(generator._doc_assembler, 'add_tags') as mock_add_tags:
            generator._add_standard_tags()

            # 验证添加了标准的服务标签
            mock_add_tags.assert_called_once()
            tags = mock_add_tags.call_args[0][0]

            assert len(tags) == 4
            tag_names = [tag["name"] for tag in tags]
            assert "Data Service" in tag_names
            assert "Feature Engineering" in tag_names
            assert "Trading Service" in tag_names
            assert "Monitoring" in tag_names

    def test_export_documentation_files_with_config(self, generator):
        """测试根据配置导出文档文件"""
        config = DocumentationExportConfig(
            output_dir="/tmp/test",
            format_types=["json", "yaml"]
        )

        with patch.object(generator._doc_assembler, 'export_to_json'), \
             patch.object(generator._doc_assembler, 'export_to_yaml'):

            result = generator._export_documentation_files_with_config(Path("/tmp/test"), config)

            assert result == {
                "json": str(Path("/tmp/test") / "openapi.json"),
                "yaml": str(Path("/tmp/test") / "openapi.yaml")
            }

    def test_export_documentation_files_with_json_only(self, generator):
        """测试只导出JSON格式"""
        config = DocumentationExportConfig(
            output_dir="/tmp/test",
            format_types=["json"]
        )

        with patch.object(generator._doc_assembler, 'export_to_json'), \
             patch.object(generator._doc_assembler, 'export_to_yaml') as mock_yaml:

            result = generator._export_documentation_files_with_config(Path("/tmp/test"), config)

            assert result == {"json": str(Path("/tmp/test") / "openapi.json")}
            # 验证没有调用YAML导出
            mock_yaml.assert_not_called()


class TestDocumentationExportConfig:
    """测试DocumentationExportConfig与生成器的集成"""

    def test_config_creation(self):
        """测试配置对象的创建"""
        config = DocumentationExportConfig(
            output_dir="/custom/docs",
            format_types=["json"],
            include_statistics=False,
            theme="dark"
        )

        assert config.output_dir == "/custom/docs"
        assert config.format_types == ["json"]
        assert config.include_statistics is False
        assert config.theme == "dark"

    def test_default_config_values(self):
        """测试默认配置值"""
        config = DocumentationExportConfig()

        assert config.output_dir == "docs/api"
        assert config.include_examples is True
        assert config.include_statistics is True
        assert config.format_types == ["json", "yaml"]
        assert config.pretty_print is True
        assert config.include_metadata is True
        assert config.compress is False
        assert config.theme == "default"


if __name__ == "__main__":
    pytest.main([__file__])
