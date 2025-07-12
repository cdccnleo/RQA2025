import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.config.services.config_loader_service import ConfigLoaderService
from src.infrastructure.config.exceptions import ConfigLoadError

class TestConfigLoader:
    @pytest.fixture
    def mock_loader(self):
        return MagicMock()

    @pytest.fixture
    def mock_validator(self):
        return MagicMock()

    @pytest.mark.unit
    def test_load_single_config(self, mock_loader, mock_validator):
        """测试配置加载器返回值"""
        # 准备测试数据
        expected_config = {"key": "value"}
        loader_meta = {"load_time": 0.1}  # 加载器返回的元数据

        # 配置mock返回最内层数据结构
        mock_loader.load.return_value = (expected_config, loader_meta)
        mock_validator.validate.return_value = (True, [])

        # 初始化服务
        service = ConfigLoaderService(
            loader=mock_loader,
            validator=mock_validator,
            sources=["test.json"]
        )

        # 执行测试
        config, meta = service.load("test.json")

        # 验证配置数据
        assert isinstance(config, dict)
        assert config == expected_config

        # 验证元数据结构
        assert isinstance(meta, dict)

        # 验证服务添加的元数据字段
        assert meta.get('env') == "test.json"
        assert meta.get('sources') == ["test.json"]
        assert isinstance(meta.get('timestamp'), float)

        # 验证加载器返回的元数据是否被正确保留
        assert 'load_time' in meta
        assert meta['load_time'] == 0.1

        # 验证所有元数据字段，包括加载器返回的和服务添加的
        expected_fields = {
            'env': "test.json",
            'sources': ["test.json"],
            'timestamp': float,
            'load_time': 0.1
        }

        # 验证每个字段的存在和类型
        for field, expected_type in expected_fields.items():
            assert field in meta, f"元数据中缺少必需的字段: {field}"
            if isinstance(expected_type, type):
                assert isinstance(meta[field], expected_type), \
                    f"{field}字段类型不正确, 期望: {expected_type}, 实际: {type(meta[field])}"
            else:
                assert meta[field] == expected_type, \
                    f"{field}字段值不正确, 期望: {expected_type}, 实际: {meta[field]}"

    @pytest.mark.unit
    def test_load_invalid_config(self, mock_loader, mock_validator):
        """测试加载无效配置"""
        # 设置加载器返回有效配置
        mock_loader.load.return_value = ({"key": "value"}, {})

        # 设置验证器返回验证失败
        mock_validator.validate.return_value = (False, ["Invalid config"])

        # 初始化服务
        service = ConfigLoaderService(
            loader=mock_loader,
            validator=mock_validator,
            sources=["test.json"]
        )

        # 验证加载无效配置时抛出异常
        with pytest.raises(ConfigLoadError):
            service.load("test.json")

    @pytest.mark.unit
    def test_batch_load_configs(self, mock_loader, mock_validator):
        """测试批量加载配置文件"""
        # 设置mock返回值 - 返回 (config, metadata) 元组
        mock_loader.load.side_effect = [
            ({"key": "value1"}, {"load_time": 0.1, "size": 100}),
            ({"key": "value2"}, {"load_time": 0.2, "size": 200})
        ]
        mock_validator.validate.return_value = (True, [])

        loader = ConfigLoaderService(
            mock_loader,
            mock_validator,
            sources=["{env}.json"]  # 使用格式化字符串
        )
        test_files = ["config1.json", "config2.json"]

        # 执行测试
        results = loader.batch_load(test_files)

        # 验证结果结构
        assert len(results) == 2
        assert "config1.json" in results
        assert "config2.json" in results

        # 验证返回的是 (config, metadata) 元组
        result1 = results["config1.json"]
        result2 = results["config2.json"]

        # 解包元组
        config1, meta1 = result1
        config2, meta2 = result2

        # 验证配置数据
        assert config1 == {"key": "value1"}
        assert config2 == {"key": "value2"}

        # 验证元数据
        assert meta1["size"] == 100
        assert meta2["size"] == 200

    @pytest.mark.performance
    def test_concurrent_access(self, mock_loader, mock_validator):
        """测试并发访问配置"""
        service = ConfigLoaderService(mock_loader, mock_validator)
        test_config = {"key": "value"}
        mock_loader.load.return_value = (test_config, {})

        import threading
        results = []

        def worker():
            results.append(service.load("config.json"))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == test_config for r in results)
