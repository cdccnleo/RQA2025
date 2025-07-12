from unittest.mock import patch, mock_open

import pytest
from datetime import datetime, time
from src.infrastructure.config.services import ConfigLoaderService
from src.infrastructure.config.strategies import JSONLoader
from src.infrastructure.config.services import validate_trading_hours

class TestAShareFeatures:
    """A股特定功能测试"""

    @pytest.fixture
    def json_loader(self):
        return JSONLoader()

    @pytest.fixture
    def trading_hours_config(self):
        """A股交易时段配置"""
        return {
            "trading_hours": {
                "morning": ["09:30", "11:30"],
                "afternoon": ["13:00", "15:00"],
                "night": ["21:00", "23:00"]
            },
            "market": "ASHARE"
        }

    @pytest.fixture
    def mock_validator(self):
        class MockValidator:
            def validate(self, config, schema=None):
                # 验证器始终返回验证通过
                return True, []
        return MockValidator()

    @pytest.mark.ashare
    def test_trading_hours_loading(self, json_loader, trading_hours_config, mock_validator):
        """测试交易时段配置加载"""
        import json
        from pathlib import Path
        from unittest.mock import PropertyMock

        # 确保mock数据包含完整的配置结构
        config_str = json.dumps(trading_hours_config)
        print(f"原始配置JSON: {config_str}")  # 调试输出

        # 创建模拟文件状态
        mock_stat = PropertyMock()
        mock_stat.st_size = 1024  # 模拟文件大小

        with patch("builtins.open", mock_open(read_data=config_str)), \
                patch("pathlib.Path.exists", return_value=True), \
                patch("pathlib.Path.stat", return_value=mock_stat):
            # 初始化加载服务
            loader = ConfigLoaderService(
                loader=json_loader,
                validator=mock_validator,
                sources=["trading_hours.json"]
            )

            # 加载配置
            config, meta = loader.load("trading_hours.json")
            print(f"加载后的配置: {config}")  # 调试输出

            # 验证配置内容
            assert config["market"] == "ASHARE"
            assert config["trading_hours"]["morning"] == ["09:30", "11:30"]
            assert config["trading_hours"]["afternoon"] == ["13:00", "15:00"]
            assert config["trading_hours"]["night"] == ["21:00", "23:00"]

    @pytest.mark.ashare
    @pytest.mark.trading_hours
    def test_trading_hours_validation(self, json_loader):
        """测试交易时段验证"""
        # 有效配置
        valid_config = {
            "trading_hours": {
                "morning": ["09:30", "11:30"],
                "afternoon": ["13:00", "15:00"]
            }
        }
        assert validate_trading_hours(valid_config)

        # 无效配置(时间重叠)
        invalid_config = {
            "trading_hours": {
                "morning": ["09:30", "15:00"],  # 覆盖下午时段
                "afternoon": ["13:00", "15:00"]
            }
        }
        assert not validate_trading_hours(invalid_config)

    @pytest.mark.ashare
    def test_limit_status_handling(self, json_loader, mock_validator):
        """测试涨跌停状态处理"""
        test_config = {
            "stocks": {
                "600519": {
                    "limit_status": "U"  # 涨停
                },
                "601318": {
                    "limit_status": "D"  # 跌停
                }
            }
        }

        # 直接模拟加载器行为，而不是模拟文件系统
        def mock_load(source):
            return test_config, {}

        # 替换加载器的load方法
        with patch.object(json_loader, 'load', side_effect=mock_load):
            # 初始化加载服务
            loader = ConfigLoaderService(
                loader=json_loader,
                validator=mock_validator,
                sources=["stocks.json"]
            )

            # 加载配置
            config, _ = loader.load("stocks.json")

            # 验证配置内容
            assert config["stocks"]["600519"]["limit_status"] == "U"
            assert config["stocks"]["601318"]["limit_status"] == "D"

    @pytest.mark.ashare
    @pytest.mark.performance
    def test_trading_hours_performance(self, json_loader, trading_hours_config, mock_validator):
        """测试交易时段配置的性能影响"""
        import json
        import timeit

        # 准备测试数据
        config_data = trading_hours_config
        loader_meta = {"load_time": 0.01}  # 模拟加载器元数据

        # 直接模拟加载器行为
        def mock_load(source):
            return config_data, loader_meta

        # 替换加载器的load方法
        with patch.object(json_loader, 'load', side_effect=mock_load):
            # 初始化加载服务
            loader = ConfigLoaderService(
                loader=json_loader,
                validator=mock_validator,
                sources=["trading_hours.json"]
            )

            # 定义加载函数
            def load_config():
                return loader.load("trading_hours.json")

            # 测量加载时间
            time_taken = timeit.timeit(load_config, number=100)

            # 验证性能指标
            print(f"100次配置加载耗时: {time_taken:.4f}秒")
            assert time_taken < 1.0, "配置加载性能不符合要求"

