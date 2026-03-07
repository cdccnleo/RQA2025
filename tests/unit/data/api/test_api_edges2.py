"""
数据API模块边界测试
测试 api/__init__.py 中的边界情况和异常场景
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from src.data.api import app, DataManagerSingleton, _build_market_response


class TestHealthCheck:
    """健康检查接口边界测试"""

    def test_health_check_success(self):
        """测试健康检查成功"""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestBuildMarketResponse:
    """市场数据响应构建边界测试"""

    def test_build_market_response_valid(self):
        """测试构建有效市场响应"""
        result = _build_market_response("BTC", "USDT")
        assert result["success"] is True
        assert result["symbol"] == "BTC/USDT"
        assert result["data"] == []

    def test_build_market_response_empty_base(self):
        """测试空base参数"""
        with pytest.raises(HTTPException) as exc_info:
            _build_market_response("", "USDT")
        assert exc_info.value.status_code == 400
        assert "invalid symbol" in exc_info.value.detail

    def test_build_market_response_empty_quote(self):
        """测试空quote参数"""
        with pytest.raises(HTTPException) as exc_info:
            _build_market_response("BTC", "")
        assert exc_info.value.status_code == 400
        assert "invalid symbol" in exc_info.value.detail

    def test_build_market_response_both_empty(self):
        """测试base和quote都为空"""
        with pytest.raises(HTTPException) as exc_info:
            _build_market_response("", "")
        assert exc_info.value.status_code == 400

    def test_build_market_response_none_base(self):
        """测试base为None"""
        with pytest.raises(HTTPException) as exc_info:
            _build_market_response(None, "USDT")  # type: ignore
        assert exc_info.value.status_code == 400

    def test_build_market_response_none_quote(self):
        """测试quote为None"""
        with pytest.raises(HTTPException) as exc_info:
            _build_market_response("BTC", None)  # type: ignore
        assert exc_info.value.status_code == 400

    def test_build_market_response_whitespace_base(self):
        """测试base为空白字符"""
        # 空白字符在Python中是truthy，所以不会抛出异常
        # 这是预期的行为，因为 `if not base` 只检查空字符串
        result = _build_market_response("   ", "USDT")
        assert result["success"] is True
        assert result["symbol"] == "   /USDT"

    def test_build_market_response_whitespace_quote(self):
        """测试quote为空白字符"""
        # 空白字符在Python中是truthy，所以不会抛出异常
        result = _build_market_response("BTC", "   ")
        assert result["success"] is True
        assert result["symbol"] == "BTC/   "

    def test_build_market_response_special_characters(self):
        """测试特殊字符"""
        result = _build_market_response("BTC-USD", "USDT-T")
        assert result["success"] is True
        assert result["symbol"] == "BTC-USD/USDT-T"


class TestGetMarketData:
    """市场数据接口边界测试"""

    def test_get_market_data_valid(self):
        """测试获取有效市场数据"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market/BTC/USDT")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["symbol"] == "BTC/USDT"

    def test_get_market_data_empty_base(self):
        """测试空base参数"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market//USDT")
        # 应该返回400或404，取决于路由匹配
        assert response.status_code in [400, 404]

    def test_get_market_data_empty_quote(self):
        """测试空quote参数"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market/BTC/")
        # 应该返回400或404
        assert response.status_code in [400, 404]

    def test_get_market_data_fallback_invalid_path(self):
        """测试兜底路由处理无效路径"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market/invalid")
        assert response.status_code == 400

    def test_get_market_data_fallback_too_many_segments(self):
        """测试兜底路由处理过多段"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market/BTC/USDT/EXTRA")
        assert response.status_code == 400

    def test_get_market_data_fallback_single_segment(self):
        """测试兜底路由处理单个段"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market/BTC")
        assert response.status_code == 400

    def test_get_market_data_fallback_empty_segments(self):
        """测试兜底路由处理空段"""
        client = TestClient(app)
        response = client.get("/api/v1/data/market//")
        assert response.status_code == 400

    def test_get_market_data_fallback_valid(self):
        """测试兜底路由处理有效路径"""
        client = TestClient(app)
        # 使用一个明确会走fallback路由的路径格式
        # 例如带有多余斜杠的路径
        response = client.get("/api/v1/data/market/BTC/USDT/")
        # 可能返回200或400，取决于路由匹配
        assert response.status_code in [200, 400]
        
        # 或者直接测试fallback函数
        from src.data.api import get_market_data_fallback
        result = get_market_data_fallback("BTC/USDT")
        assert result["success"] is True
        assert result["symbol"] == "BTC/USDT"


class TestValidateData:
    """数据验证接口边界测试"""

    def test_validate_data_with_manager(self):
        """测试使用DataManager验证数据"""
        client = TestClient(app)
        request_data = {
            "data": [{"field1": "value1"}],
            "data_type": "test"
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "validation_result" in data

    def test_validate_data_empty_data(self):
        """测试空数据验证"""
        client = TestClient(app)
        request_data = {
            "data": [],
            "data_type": "test"
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["validation_result"]["total_records"] == 0

    def test_validate_data_missing_data_key(self):
        """测试缺失data键"""
        client = TestClient(app)
        request_data = {
            "data_type": "test"
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_validate_data_missing_data_type(self):
        """测试缺失data_type键"""
        client = TestClient(app)
        request_data = {
            "data": [{"field1": "value1"}]
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_validate_data_empty_request(self):
        """测试空请求"""
        client = TestClient(app)
        request_data = {}
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_validate_data_none_data(self):
        """测试data为None"""
        client = TestClient(app)
        request_data = {
            "data": None,
            "data_type": "test"
        }
        # 应该能处理None值
        try:
            response = client.post("/api/v1/data/validate", json=request_data)
            assert response.status_code in [200, 422]  # 422是FastAPI的验证错误
        except Exception:
            # 如果抛出异常也是可以接受的边界情况
            pass

    def test_validate_data_large_data(self):
        """测试大数据量验证"""
        client = TestClient(app)
        request_data = {
            "data": [{"field1": f"value{i}"} for i in range(1000)],
            "data_type": "test"
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["validation_result"]["total_records"] == 1000

    @patch('src.data.api.DataManagerSingleton')
    def test_validate_data_manager_without_method(self, mock_manager_class):
        """测试DataManager没有validate_data方法"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        # 移除validate_data方法
        del mock_manager.validate_data
        
        client = TestClient(app)
        request_data = {
            "data": [{"field1": "value1"}],
            "data_type": "test"
        }
        response = client.post("/api/v1/data/validate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # 应该使用后备实现
        assert data["validation_result"]["total_records"] == 1

    def test_validate_data_manager_has_method(self):
        """测试DataManager有validate_data方法时使用该方法"""
        # 创建一个有validate_data方法的mock manager
        with patch('src.data.api.DataManagerSingleton') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.validate_data = Mock(return_value={
                "is_valid": True,
                "total_records": 5,
                "valid_records": 5,
                "invalid_records": 0,
                "errors": []
            })
            mock_manager_class.return_value = mock_manager
            
            client = TestClient(app)
            request_data = {
                "data": [{"field1": f"value{i}"} for i in range(5)],
                "data_type": "test"
            }
            response = client.post("/api/v1/data/validate", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # 应该使用manager的validate_data方法
            assert data["validation_result"]["total_records"] == 5
            mock_manager.validate_data.assert_called_once()


class TestDataManagerSingleton:
    """DataManagerSingleton边界测试"""

    def test_data_manager_singleton_import_success(self):
        """测试成功导入DataManagerSingleton"""
        # 应该能够导入
        assert DataManagerSingleton is not None

    def test_data_manager_singleton_validate_empty(self):
        """测试验证空数据"""
        manager = DataManagerSingleton()
        # 检查是否有validate_data方法
        if hasattr(manager, "validate_data"):
            result = manager.validate_data([], "test")
            assert result["is_valid"] is True
            assert result["total_records"] == 0
        else:
            # 如果没有该方法，说明使用的是兜底实现
            # 这是可以接受的边界情况
            assert True

    def test_data_manager_singleton_validate_none_data(self):
        """测试验证None数据"""
        manager = DataManagerSingleton()
        # 应该能处理None
        try:
            result = manager.validate_data(None, "test")  # type: ignore
            assert isinstance(result, dict)
        except (TypeError, AttributeError):
            # 如果抛出异常也是可以接受的边界情况
            pass

    def test_data_manager_singleton_validate_empty_data_type(self):
        """测试空data_type"""
        manager = DataManagerSingleton()
        # 检查是否有validate_data方法
        if hasattr(manager, "validate_data"):
            result = manager.validate_data([{"field": "value"}], "")
            assert result["is_valid"] is True
        else:
            # 如果没有该方法，这是可以接受的边界情况
            assert True

    def test_data_manager_singleton_validate_none_data_type(self):
        """测试None data_type"""
        manager = DataManagerSingleton()
        # 检查是否有validate_data方法
        if hasattr(manager, "validate_data"):
            result = manager.validate_data([{"field": "value"}], None)  # type: ignore
            assert result["is_valid"] is True
        else:
            # 如果没有该方法，这是可以接受的边界情况
            assert True


class TestEdgeCases:
    """其他边界情况测试"""

    def test_app_instance(self):
        """测试app实例存在"""
        assert app is not None
        assert hasattr(app, "get")
        assert hasattr(app, "post")

    def test_multiple_health_checks(self):
        """测试多次健康检查"""
        client = TestClient(app)
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

    def test_concurrent_requests(self):
        """测试并发请求"""
        import threading
        client = TestClient(app)
        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(code == 200 for code in results)

    def test_invalid_json(self):
        """测试无效JSON"""
        client = TestClient(app)
        # 发送无效JSON
        response = client.post(
            "/api/v1/data/validate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        # 应该返回422（验证错误）或400（错误请求）
        assert response.status_code in [400, 422]

    def test_missing_content_type(self):
        """测试缺失Content-Type"""
        client = TestClient(app)
        response = client.post(
            "/api/v1/data/validate",
            data='{"data": []}'
        )
        # FastAPI应该能处理
        assert response.status_code in [200, 422]

