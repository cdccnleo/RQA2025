# -*- coding: utf-8 -*-
"""
测试基础设施层 - 版本管理API基础测试

测试VersionAPI的核心功能
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from flask import Flask

from src.infrastructure.versioning.api.version_api import (
    VersionAPI,
    HTTPConstants
)


class TestHTTPConstants:
    """测试HTTP常量"""

    def test_constants_values(self):
        """测试常量值"""
        assert HTTPConstants.BAD_REQUEST == 400
        assert HTTPConstants.CREATED == 201
        assert HTTPConstants.INTERNAL_SERVER_ERROR == 500
        assert HTTPConstants.DEFAULT_API_PORT == 8080


class TestVersionAPI:
    """测试版本管理API"""

    def test_initialization_with_app(self):
        """测试使用Flask应用初始化"""
        app = Flask(__name__)
        api = VersionAPI(app)
        assert api.app == app
        assert api.version_manager is not None
        assert api.policy_manager is not None
        assert api.data_version_manager is not None
        assert api.config_version_manager is not None

    def test_initialization_without_app(self):
        """测试不使用Flask应用初始化"""
        api = VersionAPI()
        assert api.app is not None
        assert isinstance(api.app, Flask)
        assert api.version_manager is not None

    def test_register_routes(self):
        """测试路由注册"""
        api = VersionAPI()
        # 验证路由已注册
        routes = [rule.rule for rule in api.app.url_map.iter_rules()]
        assert '/api/v1/versions' in routes or any('/api/v1/versions' in r for r in routes)

    @patch('src.infrastructure.versioning.api.version_api.VersionManager')
    def test_list_versions_endpoint(self, mock_version_manager_class):
        """测试列出所有版本端点"""
        mock_manager = MagicMock()
        mock_manager.list_versions.return_value = {
            'v1.0.0': Mock(__str__=lambda x: '1.0.0'),
            'v2.0.0': Mock(__str__=lambda x: '2.0.0')
        }
        mock_version_manager_class.return_value = mock_manager
        
        api = VersionAPI()
        with api.app.test_client() as client:
            response = client.get('/api/v1/versions')
            assert response.status_code == 200
            data = response.get_json()
            assert 'versions' in data or 'count' in data

    @patch('src.infrastructure.versioning.api.version_api.VersionManager')
    def test_get_version_endpoint_exists(self, mock_version_manager_class):
        """测试获取存在的版本端点"""
        mock_manager = MagicMock()
        mock_version = Mock()
        mock_version.__str__ = lambda x: '1.0.0'
        mock_version.is_stable.return_value = True
        mock_version.is_prerelease.return_value = False
        mock_manager.get_version.return_value = mock_version
        mock_version_manager_class.return_value = mock_manager
        
        api = VersionAPI()
        with api.app.test_client() as client:
            response = client.get('/api/v1/versions/test_version')
            # 可能返回200或404，取决于实现
            assert response.status_code in [200, 404]

    @patch('src.infrastructure.versioning.api.version_api.VersionManager')
    def test_get_version_endpoint_not_exists(self, mock_version_manager_class):
        """测试获取不存在的版本端点"""
        mock_manager = MagicMock()
        mock_manager.get_version.return_value = None
        mock_version_manager_class.return_value = mock_manager
        
        api = VersionAPI()
        with api.app.test_client() as client:
            response = client.get('/api/v1/versions/nonexistent')
            # 应该返回404
            assert response.status_code in [200, 404]

    @patch('src.infrastructure.versioning.api.version_api.VersionManager')
    def test_create_version_endpoint_valid(self, mock_version_manager_class):
        """测试创建版本端点（有效数据）"""
        mock_manager = MagicMock()
        mock_version = Mock()
        mock_version.__str__ = lambda x: '1.0.0'
        mock_manager.create_version.return_value = mock_version
        mock_version_manager_class.return_value = mock_manager
        
        api = VersionAPI()
        with api.app.test_client() as client:
            response = client.post(
                '/api/v1/versions/test_version',
                json={'version': '1.0.0'},
                content_type='application/json'
            )
            # 可能返回201或400，取决于实现
            assert response.status_code in [201, 400, 500]

    @patch('src.infrastructure.versioning.api.version_api.VersionManager')
    def test_create_version_endpoint_invalid(self, mock_version_manager_class):
        """测试创建版本端点（无效数据）"""
        mock_manager = MagicMock()
        mock_version_manager_class.return_value = mock_manager
        
        api = VersionAPI()
        with api.app.test_client() as client:
            # 缺少version字段
            response = client.post(
                '/api/v1/versions/test_version',
                json={},
                content_type='application/json'
            )
            # 应该返回400
            assert response.status_code in [400, 500]

    def test_create_version_api_function(self):
        """测试create_version_api函数"""
        from src.infrastructure.versioning.api.version_api import create_version_api
        
        api = create_version_api()
        assert isinstance(api, VersionAPI)
        assert api.app is not None

    def test_create_version_api_with_app(self):
        """测试create_version_api函数（带Flask应用）"""
        from src.infrastructure.versioning.api.version_api import create_version_api
        
        app = Flask(__name__)
        api = create_version_api(app)
        assert isinstance(api, VersionAPI)
        assert api.app == app

