"""
测试API流程图生成器

覆盖 api_flow_diagram_generator.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.api_flow_diagram_generator import APIFlowDiagramGenerator


class TestAPIFlowDiagramGenerator:
    """APIFlowDiagramGenerator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        generator = APIFlowDiagramGenerator()

        assert generator.diagrams == {}

    def test_generate_diagram(self):
        """测试生成流程图"""
        generator = APIFlowDiagramGenerator()

        api_spec = {
            "paths": {
                "/users": {
                    "get": {"responses": {"200": {"description": "Success"}}}
                }
            }
        }

        result = generator.generate_diagram(api_spec)

        expected = {
            "type": "flow_diagram",
            "data": api_spec
        }

        assert result == expected
        assert result["type"] == "flow_diagram"
        assert result["data"] == api_spec

    def test_generate_diagram_empty_spec(self):
        """测试生成流程图（空规格）"""
        generator = APIFlowDiagramGenerator()

        api_spec = {}
        result = generator.generate_diagram(api_spec)

        expected = {
            "type": "flow_diagram",
            "data": {}
        }

        assert result == expected

    def test_generate_diagram_complex_spec(self):
        """测试生成流程图（复杂规格）"""
        generator = APIFlowDiagramGenerator()

        api_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0"
            },
            "paths": {
                "/users": {
                    "get": {
                        "summary": "Get users",
                        "responses": {
                            "200": {"description": "Success"},
                            "404": {"description": "Not found"}
                        }
                    },
                    "post": {
                        "summary": "Create user",
                        "responses": {
                            "201": {"description": "Created"}
                        }
                    }
                },
                "/users/{id}": {
                    "put": {
                        "summary": "Update user",
                        "responses": {"200": {"description": "Updated"}}
                    },
                    "delete": {
                        "summary": "Delete user",
                        "responses": {"204": {"description": "Deleted"}}
                    }
                }
            }
        }

        result = generator.generate_diagram(api_spec)

        assert result["type"] == "flow_diagram"
        assert result["data"]["openapi"] == "3.0.0"
        assert result["data"]["info"]["title"] == "Test API"
        assert "/users" in result["data"]["paths"]
        assert "/users/{id}" in result["data"]["paths"]

    def test_add_diagram(self):
        """测试添加流程图"""
        generator = APIFlowDiagramGenerator()

        diagram = {"nodes": [], "edges": []}
        generator.add_diagram("test_diagram", diagram)

        assert "test_diagram" in generator.diagrams
        assert generator.diagrams["test_diagram"] == diagram

    def test_add_multiple_diagrams(self):
        """测试添加多个流程图"""
        generator = APIFlowDiagramGenerator()

        diagram1 = {"id": "diag1", "content": "content1"}
        diagram2 = {"id": "diag2", "content": "content2"}
        diagram3 = {"id": "diag3", "content": "content3"}

        generator.add_diagram("diagram1", diagram1)
        generator.add_diagram("diagram2", diagram2)
        generator.add_diagram("diagram3", diagram3)

        assert len(generator.diagrams) == 3
        assert generator.diagrams["diagram1"] == diagram1
        assert generator.diagrams["diagram2"] == diagram2
        assert generator.diagrams["diagram3"] == diagram3

    def test_get_diagram(self):
        """测试获取流程图"""
        generator = APIFlowDiagramGenerator()

        diagram = {"nodes": [{"id": "1"}], "edges": []}
        generator.add_diagram("test_diagram", diagram)

        result = generator.get_diagram("test_diagram")

        assert result == diagram

    def test_get_diagram_nonexistent(self):
        """测试获取不存在的流程图"""
        generator = APIFlowDiagramGenerator()

        result = generator.get_diagram("nonexistent")

        assert result is None

    def test_list_diagrams(self):
        """测试列出流程图"""
        generator = APIFlowDiagramGenerator()

        generator.add_diagram("diag1", {"id": 1})
        generator.add_diagram("diag2", {"id": 2})
        generator.add_diagram("diag3", {"id": 3})

        diagrams = generator.list_diagrams()

        assert len(diagrams) == 3
        assert "diag1" in diagrams
        assert "diag2" in diagrams
        assert "diag3" in diagrams

    def test_remove_diagram(self):
        """测试移除流程图"""
        generator = APIFlowDiagramGenerator()

        diagram = {"content": "test"}
        generator.add_diagram("test_diagram", diagram)
        assert "test_diagram" in generator.diagrams

        result = generator.remove_diagram("test_diagram")
        assert result == True
        assert "test_diagram" not in generator.diagrams

    def test_remove_diagram_nonexistent(self):
        """测试移除不存在的流程图"""
        generator = APIFlowDiagramGenerator()

        result = generator.remove_diagram("nonexistent")
        assert result == False

    def test_clear_diagrams(self):
        """测试清除所有流程图"""
        generator = APIFlowDiagramGenerator()

        generator.add_diagram("diag1", {"id": 1})
        generator.add_diagram("diag2", {"id": 2})
        assert len(generator.diagrams) == 2

        generator.clear_diagrams()
        assert len(generator.diagrams) == 0

    def test_get_stats(self):
        """测试获取统计信息"""
        generator = APIFlowDiagramGenerator()

        generator.add_diagram("diag1", {})
        generator.add_diagram("diag2", {})
        generator.add_diagram("diag3", {})

        stats = generator.get_stats()

        assert isinstance(stats, dict)
        assert stats["total_diagrams"] == 3


class TestAPIFlowDiagramGeneratorIntegration:
    """APIFlowDiagramGenerator 集成测试"""

    def test_complete_diagram_generation_workflow(self):
        """测试完整的流程图生成工作流"""
        generator = APIFlowDiagramGenerator()

        # 1. 创建API规格
        api_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "User Management API",
                "version": "1.0.0"
            },
            "paths": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "responses": {"200": {"description": "Success"}}
                    },
                    "post": {
                        "summary": "Create user",
                        "responses": {"201": {"description": "Created"}}
                    }
                },
                "/users/{id}": {
                    "get": {
                        "summary": "Get user",
                        "responses": {"200": {"description": "Success"}}
                    },
                    "put": {
                        "summary": "Update user",
                        "responses": {"200": {"description": "Updated"}}
                    },
                    "delete": {
                        "summary": "Delete user",
                        "responses": {"204": {"description": "Deleted"}}
                    }
                }
            }
        }

        # 2. 生成流程图
        diagram = generator.generate_diagram(api_spec)

        # 3. 保存流程图
        generator.add_diagram("user_management_flow", diagram)

        # 4. 验证流程图内容
        saved_diagram = generator.get_diagram("user_management_flow")
        assert saved_diagram is not None
        assert saved_diagram["type"] == "flow_diagram"
        assert saved_diagram["data"]["info"]["title"] == "User Management API"

        # 5. 检查统计信息
        stats = generator.get_stats()
        assert stats["total_diagrams"] == 1

    def test_diagram_management_operations(self):
        """测试流程图管理操作"""
        generator = APIFlowDiagramGenerator()

        # 添加多个流程图
        diagrams_data = {
            "auth_flow": {"nodes": ["login", "verify", "success"], "edges": []},
            "payment_flow": {"nodes": ["init", "process", "complete"], "edges": []},
            "notification_flow": {"nodes": ["trigger", "send", "acknowledge"], "edges": []}
        }

        for name, data in diagrams_data.items():
            generator.add_diagram(name, data)

        # 验证所有流程图都已添加
        assert len(generator.diagrams) == 3

        # 移除一个流程图
        generator.remove_diagram("payment_flow")
        assert len(generator.diagrams) == 2
        assert "payment_flow" not in generator.diagrams

        # 列出剩余流程图
        remaining = generator.list_diagrams()
        assert len(remaining) == 2
        assert "auth_flow" in remaining
        assert "notification_flow" in remaining

        # 清除所有流程图
        generator.clear_diagrams()
        assert len(generator.diagrams) == 0

    def test_diagram_generation_with_various_api_specs(self):
        """测试使用各种API规格生成流程图"""
        generator = APIFlowDiagramGenerator()

        test_cases = [
            # 简单API
            {
                "name": "simple_api",
                "spec": {"paths": {"/health": {"get": {"responses": {"200": {}}}}}}
            },
            # 复杂API
            {
                "name": "complex_api",
                "spec": {
                    "openapi": "3.0.0",
                    "info": {"title": "Complex API", "version": "2.0.0"},
                    "paths": {
                        "/api/v1/users": {"get": {}, "post": {}},
                        "/api/v1/users/{id}": {"get": {}, "put": {}, "delete": {}},
                        "/api/v1/orders": {"get": {}, "post": {}},
                        "/api/v1/orders/{id}": {"get": {}, "put": {}, "delete": {}}
                    },
                    "components": {"schemas": {}}
                }
            },
            # 微服务API
            {
                "name": "microservice_api",
                "spec": {
                    "services": ["auth", "user", "order", "payment"],
                    "endpoints": [
                        {"service": "auth", "path": "/auth/login"},
                        {"service": "user", "path": "/users"},
                        {"service": "order", "path": "/orders"},
                        {"service": "payment", "path": "/payments"}
                    ]
                }
            }
        ]

        for test_case in test_cases:
            # 生成流程图
            diagram = generator.generate_diagram(test_case["spec"])

            # 保存流程图
            generator.add_diagram(test_case["name"], diagram)

            # 验证流程图结构
            saved = generator.get_diagram(test_case["name"])
            assert saved["type"] == "flow_diagram"
            assert saved["data"] == test_case["spec"]

        # 验证统计信息
        stats = generator.get_stats()
        assert stats["total_diagrams"] == 3

    def test_error_handling_and_edge_cases(self):
        """测试错误处理和边界情况"""
        generator = APIFlowDiagramGenerator()

        # 测试空规格
        empty_diagram = generator.generate_diagram({})
        assert empty_diagram["type"] == "flow_diagram"
        assert empty_diagram["data"] == {}

        # 测试None规格
        none_diagram = generator.generate_diagram(None)
        assert none_diagram["type"] == "flow_diagram"
        assert none_diagram["data"] is None

        # 测试复杂嵌套规格
        complex_spec = {
            "nested": {
                "deeply": {
                    "nested": {
                        "structure": {
                            "with": ["multiple", "levels"],
                            "and": {"various": "data_types"}
                        }
                    }
                }
            },
            "array_data": [1, 2, {"key": "value"}],
            "boolean": True,
            "number": 42
        }

        complex_diagram = generator.generate_diagram(complex_spec)
        assert complex_diagram["type"] == "flow_diagram"
        assert complex_diagram["data"]["nested"]["deeply"]["nested"]["structure"]["with"] == ["multiple", "levels"]
        assert complex_diagram["data"]["boolean"] == True
        assert complex_diagram["data"]["number"] == 42

    def test_diagram_operations_isolation(self):
        """测试流程图操作隔离性"""
        generator1 = APIFlowDiagramGenerator()
        generator2 = APIFlowDiagramGenerator()

        # 在第一个生成器中添加流程图
        generator1.add_diagram("shared_name", {"data": "generator1"})

        # 在第二个生成器中添加同名但不同内容的流程图
        generator2.add_diagram("shared_name", {"data": "generator2"})

        # 验证两个生成器互相隔离
        assert generator1.get_diagram("shared_name")["data"] == "generator1"
        assert generator2.get_diagram("shared_name")["data"] == "generator2"

        # 验证统计信息独立
        assert generator1.get_stats()["total_diagrams"] == 1
        assert generator2.get_stats()["total_diagrams"] == 1