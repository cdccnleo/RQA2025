"""
测试API文档增强器

覆盖 api_documentation_enhancer.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.api_documentation_enhancer import APIDocumentationEnhancer


class TestAPIDocumentationEnhancer:
    """APIDocumentationEnhancer 类测试"""

    def test_initialization(self):
        """测试初始化"""
        enhancer = APIDocumentationEnhancer()

        assert enhancer.enhancements == {}
        assert enhancer.templates == {}

    def test_add_enhancement(self):
        """测试添加增强功能"""
        enhancer = APIDocumentationEnhancer()

        def mock_enhancement(docs):
            return docs

        enhancer.add_enhancement("test_enhancement", mock_enhancement)

        assert "test_enhancement" in enhancer.enhancements
        assert enhancer.enhancements["test_enhancement"] == mock_enhancement

    def test_add_multiple_enhancements(self):
        """测试添加多个增强功能"""
        enhancer = APIDocumentationEnhancer()

        def enhancement1(docs):
            docs["enhanced_by_1"] = True
            return docs

        def enhancement2(docs):
            docs["enhanced_by_2"] = True
            return docs

        enhancer.add_enhancement("enhancement1", enhancement1)
        enhancer.add_enhancement("enhancement2", enhancement2)

        assert len(enhancer.enhancements) == 2
        assert "enhancement1" in enhancer.enhancements
        assert "enhancement2" in enhancer.enhancements

    def test_enhance_documentation_no_enhancements(self):
        """测试增强文档（无增强功能）"""
        enhancer = APIDocumentationEnhancer()

        docs = {"title": "API Docs", "version": "1.0"}
        result = enhancer.enhance_documentation(docs)

        assert result == docs
        assert result is not docs  # 应该是副本

    def test_enhance_documentation_single_enhancement(self):
        """测试增强文档（单个增强功能）"""
        enhancer = APIDocumentationEnhancer()

        def uppercase_title(docs):
            docs_copy = docs.copy()
            docs_copy["title"] = docs_copy["title"].upper()
            return docs_copy

        enhancer.add_enhancement("uppercase", uppercase_title)

        docs = {"title": "api docs", "version": "1.0"}
        result = enhancer.enhance_documentation(docs)

        assert result["title"] == "API DOCS"
        assert result["version"] == "1.0"
        assert result is not docs

    def test_enhance_documentation_multiple_enhancements(self):
        """测试增强文档（多个增强功能）"""
        enhancer = APIDocumentationEnhancer()

        def add_metadata(docs):
            docs_copy = docs.copy()
            docs_copy["metadata"] = {"generated": True}
            return docs_copy

        def add_examples(docs):
            docs_copy = docs.copy()
            docs_copy["examples"] = ["example1", "example2"]
            return docs_copy

        def add_version_info(docs):
            docs_copy = docs.copy()
            docs_copy["version_info"] = f"v{docs_copy['version']}"
            return docs_copy

        enhancer.add_enhancement("metadata", add_metadata)
        enhancer.add_enhancement("examples", add_examples)
        enhancer.add_enhancement("version", add_version_info)

        docs = {"title": "API Docs", "version": "1.0"}
        result = enhancer.enhance_documentation(docs)

        assert result["title"] == "API Docs"
        assert result["version"] == "1.0"
        assert result["metadata"] == {"generated": True}
        assert result["examples"] == ["example1", "example2"]
        assert result["version_info"] == "v1.0"

    def test_enhance_documentation_with_exception_handling(self):
        """测试增强文档时的异常处理"""
        enhancer = APIDocumentationEnhancer()

        def good_enhancement(docs):
            docs_copy = docs.copy()
            docs_copy["good"] = True
            return docs_copy

        def bad_enhancement(docs):
            raise ValueError("Enhancement failed")

        def another_good_enhancement(docs):
            docs_copy = docs.copy()
            docs_copy["another_good"] = True
            return docs_copy

        enhancer.add_enhancement("good1", good_enhancement)
        enhancer.add_enhancement("bad", bad_enhancement)
        enhancer.add_enhancement("good2", another_good_enhancement)

        docs = {"title": "API Docs", "version": "1.0"}

        # 虽然有异常，但系统应该继续处理其他增强
        result = enhancer.enhance_documentation(docs)

        assert result["title"] == "API Docs"
        assert result["version"] == "1.0"
        assert result["good"] == True
        assert result["another_good"] == True

    def test_remove_enhancement(self):
        """测试移除增强功能"""
        enhancer = APIDocumentationEnhancer()

        def test_enhancement(docs):
            return docs

        enhancer.add_enhancement("test", test_enhancement)
        assert "test" in enhancer.enhancements

        result = enhancer.remove_enhancement("test")
        assert result == True
        assert "test" not in enhancer.enhancements

    def test_remove_enhancement_nonexistent(self):
        """测试移除不存在的增强功能"""
        enhancer = APIDocumentationEnhancer()

        result = enhancer.remove_enhancement("nonexistent")
        assert result == False

    def test_get_enhancement_count(self):
        """测试获取增强数量"""
        enhancer = APIDocumentationEnhancer()

        assert enhancer.get_enhancement_count() == 0

        enhancer.add_enhancement("enh1", lambda x: x)
        enhancer.add_enhancement("enh2", lambda x: x)
        enhancer.add_enhancement("enh3", lambda x: x)

        assert enhancer.get_enhancement_count() == 3

    def test_add_template(self):
        """测试添加模板"""
        enhancer = APIDocumentationEnhancer()

        template = {"base": {"title": "Template Title"}}
        enhancer.add_template("base_template", template)

        assert "base_template" in enhancer.templates
        assert enhancer.templates["base_template"] == template

    def test_get_template_count(self):
        """测试获取模板数量"""
        enhancer = APIDocumentationEnhancer()

        assert enhancer.get_template_count() == 0

        enhancer.add_template("temp1", lambda x: x)
        enhancer.add_template("temp2", lambda x: x)
        enhancer.add_template("temp3", lambda x: x)

        assert enhancer.get_template_count() == 3

    def test_apply_template(self):
        """测试应用模板"""
        enhancer = APIDocumentationEnhancer()

        template = {
            "info": {"title": "API Title", "version": "1.0"},
            "paths": {}
        }
        enhancer.add_template("api_base", template)

        docs = {"custom": "data"}
        result = enhancer.apply_template(docs, "api_base")

        assert result["info"]["title"] == "API Title"
        assert result["info"]["version"] == "1.0"
        assert result["paths"] == {}
        assert result["custom"] == "data"

    def test_apply_template_nonexistent(self):
        """测试应用不存在的模板"""
        enhancer = APIDocumentationEnhancer()

        docs = {"original": "data"}
        result = enhancer.apply_template(docs, "nonexistent")

        assert result == docs

    def test_get_enhancement_and_template_counts(self):
        """测试获取增强和模板数量"""
        enhancer = APIDocumentationEnhancer()

        # 初始状态
        assert enhancer.get_enhancement_count() == 0
        assert enhancer.get_template_count() == 0

        # 添加增强
        enhancer.add_enhancement("enh1", lambda x: x)
        enhancer.add_enhancement("enh2", lambda x: x)
        assert enhancer.get_enhancement_count() == 2

        # 添加模板
        enhancer.add_template("temp1", lambda x: x)
        enhancer.add_template("temp2", lambda x: x)
        assert enhancer.get_template_count() == 2


class TestAPIDocumentationEnhancerIntegration:
    """APIDocumentationEnhancer 集成测试"""

    def test_complete_documentation_enhancement_workflow(self):
        """测试完整的文档增强工作流"""
        enhancer = APIDocumentationEnhancer()

        # 1. 设置模板
        base_template = {
            "openapi": "3.0.0",
            "info": {
                "title": "Base API",
                "version": "1.0.0"
            },
            "paths": {},
            "components": {"schemas": {}}
        }
        enhancer.add_template("base_api", base_template)

        # 2. 添加增强功能
        def add_security_schemes(docs):
            docs_copy = docs.copy()
            if "components" not in docs_copy:
                docs_copy["components"] = {}
            docs_copy["components"]["securitySchemes"] = {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            }
            return docs_copy

        def add_response_examples(docs):
            docs_copy = docs.copy()
            # 为所有响应添加示例
            if "paths" in docs_copy:
                for path_item in docs_copy["paths"].values():
                    for operation in path_item.values():
                        if "responses" in operation:
                            for response in operation["responses"].values():
                                if isinstance(response, dict):
                                    response["examples"] = {
                                        "application/json": {"example": "sample data"}
                                    }
            return docs_copy

        def validate_openapi_format(docs):
            docs_copy = docs.copy()
            # 验证OpenAPI格式
            if "openapi" in docs_copy:
                # 这里可以添加更复杂的验证逻辑
                pass
            return docs_copy

        enhancer.add_enhancement("security", add_security_schemes)
        enhancer.add_enhancement("examples", add_response_examples)
        enhancer.add_enhancement("validation", validate_openapi_format)

        # 3. 创建基础文档
        base_docs = enhancer.apply_template({}, "base_api")

        # 4. 添加API路径
        api_docs = base_docs.copy()
        api_docs["paths"] = {
            "/users": {
                "get": {
                    "responses": {
                        "200": {"description": "Success"}
                    }
                },
                "post": {
                    "responses": {
                        "201": {"description": "Created"}
                    }
                }
            }
        }

        # 5. 应用所有增强
        enhanced_docs = enhancer.enhance_documentation(api_docs)

        # 6. 验证结果
        assert enhanced_docs["openapi"] == "3.0.0"
        assert enhanced_docs["info"]["title"] == "Base API"
        assert "securitySchemes" in enhanced_docs["components"]
        assert enhanced_docs["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"

        # 验证响应示例
        get_response = enhanced_docs["paths"]["/users"]["get"]["responses"]["200"]
        assert "examples" in get_response

        post_response = enhanced_docs["paths"]["/users"]["post"]["responses"]["201"]
        assert "examples" in post_response

    def test_enhancement_error_isolation(self):
        """测试增强功能错误隔离"""
        enhancer = APIDocumentationEnhancer()

        call_log = []

        def enhancement1(docs):
            call_log.append("enhancement1")
            docs_copy = docs.copy()
            docs_copy["step1"] = True
            return docs_copy

        def enhancement2(docs):
            call_log.append("enhancement2")
            raise RuntimeError("Enhancement 2 failed")

        def enhancement3(docs):
            call_log.append("enhancement3")
            docs_copy = docs.copy()
            docs_copy["step3"] = True
            return docs_copy

        enhancer.add_enhancement("step1", enhancement1)
        enhancer.add_enhancement("step2", enhancement2)
        enhancer.add_enhancement("step3", enhancement3)

        docs = {"original": True}
        result = enhancer.enhance_documentation(docs)

        # 验证调用顺序
        assert call_log == ["enhancement1", "enhancement2", "enhancement3"]

        # 验证结果包含成功步骤的结果
        assert result["original"] == True
        assert result["step1"] == True
        assert result["step3"] == True

    def test_template_merging_with_deep_structures(self):
        """测试模板合并（深度结构）"""
        enhancer = APIDocumentationEnhancer()

        template = {
            "info": {
                "title": "Template API",
                "version": "1.0.0",
                "contact": {"email": "api@example.com"}
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}}
                    }
                }
            }
        }

        enhancer.add_template("complex_template", template)

        custom_docs = {
            "info": {
                "description": "Custom API description",
                "contact": {"name": "API Team"}
            },
            "paths": {
                "/users": {"get": {"responses": {"200": {"description": "OK"}}}}
            }
        }

        result = enhancer.apply_template(custom_docs, "complex_template")

        # 验证模板合并
        assert result["info"]["title"] == "Template API"
        assert result["info"]["version"] == "1.0.0"
        assert result["info"]["description"] == "Custom API description"
        assert result["info"]["contact"]["email"] == "api@example.com"
        assert result["info"]["contact"]["name"] == "API Team"

        assert result["components"]["schemas"]["User"]["properties"]["id"]["type"] == "string"
        assert result["paths"]["/users"]["get"]["responses"]["200"]["description"] == "OK"

    def test_multiple_templates_and_enhancements_chain(self):
        """测试多模板和增强功能链"""
        enhancer = APIDocumentationEnhancer()

        # 设置多个模板
        base_template = {"openapi": "3.0.0", "info": {"version": "1.0.0"}}
        auth_template = {"components": {"securitySchemes": {"bearer": {}}}}
        docs_template = {"info": {"title": "API Docs", "description": "Auto-generated"}}

        enhancer.add_template("base", base_template)
        enhancer.add_template("auth", auth_template)
        enhancer.add_template("docs", docs_template)

        # 设置增强功能链
        enhancements = []

        def merge_auth_template(docs):
            return enhancer.apply_template(docs, "auth")

        def merge_docs_template(docs):
            return enhancer.apply_template(docs, "docs")

        def add_timestamp(docs):
            docs_copy = docs.copy()
            docs_copy["generated_at"] = "2024-01-01T12:00:00Z"
            return docs_copy

        enhancer.add_enhancement("merge_auth", merge_auth_template)
        enhancer.add_enhancement("merge_docs", merge_docs_template)
        enhancer.add_enhancement("add_timestamp", add_timestamp)

        # 从基础模板开始
        base_docs = enhancer.apply_template({}, "base")
        final_docs = enhancer.enhance_documentation(base_docs)

        # 验证最终结果
        assert final_docs["openapi"] == "3.0.0"
        assert final_docs["info"]["version"] == "1.0.0"
        assert final_docs["info"]["title"] == "API Docs"
        assert final_docs["info"]["description"] == "Auto-generated"
        assert "securitySchemes" in final_docs["components"]
        assert final_docs["generated_at"] == "2024-01-01T12:00:00Z"