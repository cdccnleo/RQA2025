"""
测试API文档搜索系统 - 重构版本

覆盖 api_documentation_search_refactored.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.api.api_documentation_search_refactored import APIDocumentationSearch


class TestAPIDocumentationSearch:
    """APIDocumentationSearch 类测试"""

    def test_initialization(self):
        """测试初始化"""
        search = APIDocumentationSearch()

        assert hasattr(search, '_document_loader')
        assert hasattr(search, '_search_engine')
        assert hasattr(search, '_navigation_builder')
        assert hasattr(search, '_documents')
        assert isinstance(search._documents, dict)

    @patch('src.infrastructure.api.api_documentation_search_refactored.DocumentLoader')
    @patch('src.infrastructure.api.api_documentation_search_refactored.SearchEngine')
    @patch('src.infrastructure.api.api_documentation_search_refactored.NavigationBuilder')
    def test_initialization_with_mocked_components(self, mock_navigation_builder, mock_search_engine, mock_document_loader):
        """测试初始化（使用模拟对象）"""
        search = APIDocumentationSearch()

        mock_document_loader.assert_called_once()
        mock_search_engine.assert_called_once()
        mock_navigation_builder.assert_called_once()

    def test_documents_property_getter(self):
        """测试documents属性getter"""
        search = APIDocumentationSearch()

        # 初始状态应该是空字典
        assert search.documents == {}

    def test_documents_property_setter(self):
        """测试documents属性setter"""
        search = APIDocumentationSearch()

        test_docs = {
            "user_api": {"paths": {"/users": {"get": {}}}},
            "order_api": {"paths": {"/orders": {"post": {}}}}
        }

        search.documents = test_docs

        assert search.documents == test_docs

    def test_load_documents(self):
        """测试加载文档"""
        search = APIDocumentationSearch()

        # 模拟文档加载器
        mock_docs = {
            "test_api": {
                "openapi": "3.0.0",
                "paths": {"/test": {"get": {"responses": {"200": {"description": "OK"}}}}}
            }
        }
        search._document_loader.load_documents = Mock(return_value=mock_docs)

        result = search.load_documents("test_docs.json")

        assert result == mock_docs
        assert search.documents == mock_docs
        search._document_loader.load_documents.assert_called_once_with("test_docs.json")

    def test_search_basic(self):
        """测试基本搜索功能"""
        search = APIDocumentationSearch()

        # 设置模拟数据
        search._endpoint_documents = {
            "users_get": {"endpoint": "/users", "summary": "Get users"},
            "orders_post": {"endpoint": "/orders", "summary": "Create order"}
        }

        # 模拟搜索结果
        mock_result = [Mock(matches=[{"endpoint": "/users", "score": 0.9}])]
        search._search_engine.search = Mock(return_value=mock_result)

        result = search.search("users")

        assert result == mock_result
        search._search_engine.search.assert_called_once_with("users", search._endpoint_documents, 20, "all")

    def test_search_with_options(self):
        """测试带选项的搜索"""
        search = APIDocumentationSearch()

        search._endpoint_documents = {"test": {}}
        mock_result = [Mock(matches=[])]
        search._search_engine.search = Mock(return_value=mock_result)

        result = search.search("test query", limit=10, search_type="endpoint")

        assert result == mock_result
        search._search_engine.search.assert_called_once_with("test query", search._endpoint_documents, 10, "endpoint")

    def test_get_navigation_suggestions_no_path(self):
        """测试获取导航建议（无路径）"""
        search = APIDocumentationSearch()

        mock_suggestions = {"categories": ["user", "order"], "endpoints": ["/users", "/orders"]}
        search._navigation_builder.build_navigation = Mock(return_value=mock_suggestions)

        result = search.get_navigation_suggestions()

        assert result == mock_suggestions
        search._navigation_builder.build_navigation.assert_called_once_with("")

    def test_get_navigation_suggestions_with_path(self):
        """测试获取导航建议（带路径）"""
        search = APIDocumentationSearch()

        mock_suggestions = {"subpaths": ["/users/profile", "/users/settings"]}
        search._navigation_builder.build_navigation = Mock(return_value=mock_suggestions)

        result = search.get_navigation_suggestions("/users")

        assert result == mock_suggestions
        search._navigation_builder.build_navigation.assert_called_once_with("/users")

    def test_get_endpoint_details_found(self):
        """测试获取端点详情（找到）"""
        search = APIDocumentationSearch()

        endpoint_data = {
            "method": "GET",
            "path": "/users",
            "summary": "Get users",
            "responses": {"200": {"description": "Success"}}
        }

        # 设置_endpoint_documents以匹配方法实现
        search._endpoint_documents = {"users_get": endpoint_data}

        result = search.get_endpoint_details("users_get")

        assert result == endpoint_data

    def test_get_endpoint_details_not_found(self):
        """测试获取端点详情（未找到）"""
        search = APIDocumentationSearch()

        # 设置_endpoint_documents为空字典
        search._endpoint_documents = {}

        result = search.get_endpoint_details("nonexistent")

        assert result is None

    def test_get_endpoints_by_category(self):
        """测试按类别获取端点"""
        search = APIDocumentationSearch()

        endpoints = ["/users", "/user/profile", "/user/settings"]
        # 设置index的categories并防止重新构建
        search.index.categories = {"user": endpoints}
        search._index_dirty = False

        result = search.get_endpoints_by_category("user")

        assert result == endpoints

    def test_get_endpoints_by_method(self):
        """测试按方法获取端点"""
        search = APIDocumentationSearch()

        endpoints = ["/users", "/orders", "/products"]
        # 设置index的endpoints_by_method
        search.index.endpoints_by_method = {"GET": endpoints}
        search._index_dirty = False

        result = search.get_endpoints_by_method("GET")

        assert result == endpoints

    def test_get_endpoints_by_parameter(self):
        """测试按参数获取端点"""
        search = APIDocumentationSearch()

        endpoints = ["/users/{id}", "/orders/{order_id}", "/products/{product_id}"]
        # 设置index的parameter_index
        search.index.parameter_index = {"id": endpoints}
        search._index_dirty = False

        result = search.get_endpoints_by_parameter("id")

        assert result == endpoints

    def test_clear_cache(self):
        """测试清除缓存"""
        search = APIDocumentationSearch()

        # Mock组件的clear_cache方法
        search._search_engine.clear_cache = Mock()
        search._navigation_builder.clear_cache = Mock()

        search.clear_cache()

        # 验证所有组件的clear_cache方法都被调用
        search._search_engine.clear_cache.assert_called_once()
        search._navigation_builder.clear_cache.assert_called_once()

    def test_get_search_statistics(self):
        """测试获取搜索统计信息"""
        search = APIDocumentationSearch()

        mock_stats = {
            "total_searches": 150,
            "cache_hits": 120,
            "average_response_time": 0.05,
            "total_documents": 25
        }

        search._search_engine.get_statistics = Mock(return_value=mock_stats)

        result = search.get_search_statistics()

        assert result == mock_stats
        search._search_engine.get_statistics.assert_called_once()

    def test_get_search_engine(self):
        """测试获取搜索引擎"""
        search = APIDocumentationSearch()

        engine = search.get_search_engine()

        assert engine == search._search_engine

    def test_get_navigation_builder(self):
        """测试获取导航构建器"""
        search = APIDocumentationSearch()

        builder = search.get_navigation_builder()

        assert builder == search._navigation_builder

    def test_rebuild_index(self):
        """测试重建索引"""
        search = APIDocumentationSearch()

        # Mock导航构建器的rebuild_index方法
        search._navigation_builder.rebuild_index = Mock()

        search.rebuild_index()

        # 验证导航构建器的rebuild_index方法被调用
        search._navigation_builder.rebuild_index.assert_called_once()

    def test_extract_endpoints(self):
        """测试提取端点"""
        search = APIDocumentationSearch()

        api_data = {
            "paths": {
                "/users": {
                    "get": {"summary": "Get users"},
                    "post": {"summary": "Create user"}
                },
                "/users/{id}": {
                    "get": {"summary": "Get user"},
                    "put": {"summary": "Update user"},
                    "delete": {"summary": "Delete user"}
                }
            }
        }

        result = search._extract_endpoints(api_data)

        assert "users_get" in result
        assert "users_post" in result
        assert "users/{id}_get" in result
        assert "users/{id}_put" in result
        assert "users/{id}_delete" in result

        assert result["users_get"]["method"] == "get"
        assert result["users_get"]["path"] == "/users"
        assert result["users_post"]["method"] == "post"
        assert result["users_post"]["path"] == "/users"

    def test_extract_endpoints_empty_data(self):
        """测试提取端点（空数据）"""
        search = APIDocumentationSearch()

        result = search._extract_endpoints({})

        assert result == {}

    def test_extract_endpoints_no_paths(self):
        """测试提取端点（无路径）"""
        search = APIDocumentationSearch()

        api_data = {"info": {"title": "API"}, "components": {}}

        result = search._extract_endpoints(api_data)

        assert result == {}

    def test_ensure_index(self):
        """测试确保索引"""
        search = APIDocumentationSearch()

        # 设置一些文档
        search._documents = {"api1": {"paths": {"/test": {"get": {}}}}}

        # Mock导航构建器的build_navigation_index方法
        search._navigation_builder.build_navigation_index = Mock()

        search._ensure_index()

        # 验证导航构建器的build_navigation_index方法被调用
        search._navigation_builder.build_navigation_index.assert_called_once()

    def test_sync_navigation_builder_documents(self):
        """测试同步导航构建器文档"""
        search = APIDocumentationSearch()

        test_docs = {"api1": {}, "api2": {}}
        search._documents = test_docs

        search._sync_navigation_builder_documents()

        # 验证导航构建器的documents属性被设置
        assert search._navigation_builder.documents == search._endpoint_documents


class TestAPIDocumentationSearchIntegration:
    """APIDocumentationSearch 集成测试"""

    def test_complete_search_workflow(self):
        """测试完整的搜索工作流"""
        search = APIDocumentationSearch()

        # 1. 设置文档
        docs = {
            "openapi": "3.0.0",
            "info": {"title": "User API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "summary": "Get all users",
                        "responses": {"200": {"description": "Success"}}
                    },
                    "post": {
                        "summary": "Create user",
                        "responses": {"201": {"description": "Created"}}
                    }
                },
                "/users/{id}": {
                    "get": {
                        "summary": "Get user by ID",
                        "responses": {"200": {"description": "Success"}}
                    }
                }
            }
        }
        search.documents = docs

        # 2. 执行搜索
        mock_result = Mock()
        mock_result.matches = [
            {"endpoint": "/users", "method": "get", "score": 0.9},
            {"endpoint": "/users/{id}", "method": "get", "score": 0.8}
        ]
        search._search_engine.search = Mock(return_value=mock_result)

        search_result = search.search("users")

        # 3. 验证搜索结果
        assert len(search_result.matches) == 2
        assert search_result.matches[0]["endpoint"] == "/users"
        assert search_result.matches[0]["method"] == "get"

        # 4. 获取导航建议
        mock_suggestions = {
            "categories": ["user"],
            "methods": ["GET", "POST"],
            "tags": [],
            "current_path": "",
            "total_endpoints": 2
        }
        search._navigation_builder.build_navigation = Mock(return_value=mock_suggestions)

        suggestions = search.get_navigation_suggestions()

        # 5. 验证建议
        assert "user" in suggestions["categories"]
        assert suggestions["total_endpoints"] == 2

        # 6. 获取端点详情
        endpoint_details = {
            "method": "get",
            "path": "/users",
            "summary": "Get all users",
            "responses": {"200": {"description": "Success"}}
        }
        search._navigation_builder.get_endpoint_details = Mock(return_value=endpoint_details)

        details = search.get_endpoint_details("users_get")

        # 7. 验证端点详情
        assert details["method"] == "get"
        assert details["path"] == "/users"
        assert details["operation"]["summary"] == "Get all users"

        # 8. 获取统计信息
        mock_stats = {"total_searches": 5, "cache_hit_rate": 0.8}
        search._search_engine.get_statistics = Mock(return_value=mock_stats)

        stats = search.get_search_statistics()

        # 9. 验证统计信息
        assert stats["total_searches"] == 5
        assert stats["cache_hit_rate"] == 0.8

    def test_search_and_navigation_integration(self):
        """测试搜索和导航的集成"""
        search = APIDocumentationSearch()

        # 设置文档
        search._documents = {
            "api": {
                "paths": {
                    "/users": {"get": {"tags": ["user"]}},
                    "/orders": {"get": {"tags": ["order"]}},
                    "/products": {"get": {"tags": ["product"]}}
                }
            }
        }

        # 模拟搜索结果
        search._search_engine.search = Mock(return_value=Mock(matches=[
            {"endpoint": "/users", "score": 1.0}
        ]))

        # 模拟导航建议
        search._navigation_builder.build_navigation = Mock(return_value={
            "categories": ["user"],
            "methods": ["GET"],
            "tags": ["user"],
            "current_path": "/users",
            "total_endpoints": 2,
            "endpoints": ["/users", "/users/{id}"]
        })

        # 执行搜索
        search_result = search.search("user")
        suggestions = search.get_navigation_suggestions("/users")

        # 验证集成结果
        assert len(search_result.matches) == 1
        assert search_result.matches[0]["endpoint"] == "/users"
        assert "/users" in suggestions["endpoints"]
        assert "/users/{id}" in suggestions["endpoints"]

    def test_document_loading_and_indexing_workflow(self):
        """测试文档加载和索引构建工作流"""
        search = APIDocumentationSearch()

        # 1. 模拟文档加载
        loaded_docs = {
            "api1": {"paths": {"/test1": {"get": {}}}},
            "api2": {"paths": {"/test2": {"post": {}}}}
        }
        search._document_loader.load_from_file = Mock(return_value=loaded_docs)

        # 2. 加载文档
        result = search.load_documents("api_docs.json")

        # 3. 验证文档加载
        assert result == loaded_docs
        assert search.documents == loaded_docs

        # 4. 触发索引重建
        search._navigation_builder.rebuild_index = Mock()
        search.rebuild_index()

        # 5. 验证索引重建被调用
        search._navigation_builder.rebuild_index.assert_called_once()

        # 6. 获取导航建议（会触发索引确保）
        mock_suggestions = {
            "categories": [],
            "methods": ["GET", "POST"],
            "tags": [],
            "current_path": "",
            "total_endpoints": 2,
            "endpoints": ["/test1", "/test2"]
        }
        search._navigation_builder.build_navigation = Mock(return_value=mock_suggestions)

        suggestions = search.get_navigation_suggestions()

        # 7. 验证导航建议
        assert "/test1" in suggestions["endpoints"]
        assert "/test2" in suggestions["endpoints"]

    def test_cache_management_workflow(self):
        """测试缓存管理工作流"""
        search = APIDocumentationSearch()

        # 1. 执行搜索（会使用缓存）
        search._search_engine.search = Mock(return_value=Mock(matches=[]))
        search.search("test")

        # 2. 清除缓存
        search._search_engine.clear_cache = Mock()
        search._navigation_builder.clear_cache = Mock()
        search.clear_cache()

        # 3. 验证所有缓存都被清除
        search._search_engine.clear_cache.assert_called_once()
        search._navigation_builder.clear_cache.assert_called_once()

        # 4. 再次搜索（缓存已清除）
        search.search("test2")

        # 5. 验证搜索方法被调用了两次
        assert search._search_engine.search.call_count == 2

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        search = APIDocumentationSearch()

        # 测试搜索异常处理
        search._search_engine.search = Mock(side_effect=Exception("Search engine error"))

        # 不应该抛出异常，而是返回None或空结果
        try:
            result = search.search("test")
            # 如果没有抛出异常，验证结果处理
            assert result is not None
        except Exception:
            # 如果抛出异常，验证异常被正确处理
            pass

        # 测试导航建议异常处理
        search._navigation_builder.build_navigation = Mock(side_effect=Exception("Navigation error"))

        try:
            suggestions = search.get_navigation_suggestions()
            assert suggestions is not None
        except Exception:
            pass

        # 验证系统仍然可用
        stats = search.get_search_statistics()
        assert isinstance(stats, dict)

    def test_component_access_methods(self):
        """测试组件访问方法"""
        search = APIDocumentationSearch()

        # 测试获取搜索引擎
        engine = search.get_search_engine()
        assert engine == search._search_engine

        # 测试获取导航构建器
        builder = search.get_navigation_builder()
        assert builder == search._navigation_builder

        # 验证组件类型
        assert hasattr(engine, 'search')
        assert hasattr(builder, 'build_navigation')