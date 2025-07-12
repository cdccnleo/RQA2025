# tests/features/test_feature_metadata.py
import logging
import time
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch
import copy

import pandas as pd
import pytest
import numpy as np
from src.features.feature_manager import FeatureManager
from src.features.feature_metadata import FeatureMetadata
from src.models.model_manager import ModelManager


def test_update_features_metadata_not_initialized():
    model_manager = Mock(spec=ModelManager)
    manager = FeatureManager(
        model_path="/tmp",
        stock_code="TEST",
        model_manager=model_manager
    )
    manager.metadata = None

    with pytest.raises(RuntimeError) as excinfo:
        manager.update_features(pd.DataFrame())
    assert "特征元数据未初始化" in str(excinfo.value)


def test_feature_metadata_lifecycle():
    meta = FeatureMetadata(
        feature_params={"window": 20},
        data_source_version="v2.0",
        feature_list=["f1", "f2"]
    )
    original_created_at = meta.created_at
    time.sleep(0.001)  # 添加微小延迟
    meta.update(["f1", "f2", "f3"])
    assert meta.last_updated > original_created_at  # 与初始时间比较


def test_metadata_lifecycle(tmp_path):
    # 初始化测试
    meta = FeatureMetadata(
        feature_params={"window": 20},
        data_source_version="v2.1",
        feature_list=["close", "volume"],
        scaler_path=tmp_path / "scaler.pkl"
    )

    # 更新测试
    meta.update(["close", "volume", "rsi"])
    assert len(meta.feature_list) == 3
    assert meta.last_updated > meta.created_at

    # 序列化测试
    save_path = tmp_path / "meta.pkl"
    meta.save(save_path)

    # 反序列化测试
    loaded = FeatureMetadata.load(save_path)
    assert loaded.data_source_version == "v2.1"

    # 兼容性验证
    new_meta = FeatureMetadata({}, "v2.2", [])
    assert not meta.validate_compatibility(new_meta)


def test_feature_metadata_initialization():
    metadata = FeatureMetadata({}, "v1.0", [])
    assert metadata.feature_params == {}
    assert metadata.data_source_version == "v1.0"


def test_feature_metadata_save_load(tmp_path):
    metadata = FeatureMetadata({}, "v1.0", ["f1", "f2"])
    save_path = tmp_path / "metadata.pkl"
    metadata.save(save_path)
    loaded = FeatureMetadata.load(save_path)
    assert loaded.feature_list == ["f1", "f2"]


def test_metadata_timestamp_update():
    meta = FeatureMetadata(
        feature_params={},
        data_source_version="v1.0",
        feature_list=["close"]
    )
    original_time = meta.last_updated
    meta.update(["new_feature"])
    assert meta.last_updated > original_time  # 时间戳严格递增验证


def test_metadata_serialization(tmp_path):
    """测试元数据序列化完整性"""
    original = FeatureMetadata(
        feature_params={},
        data_source_version="v1.0",
        feature_list=["close"]
    )
    path = tmp_path / "metadata.pkl"
    original.save(path)
    loaded = FeatureMetadata.load(path)
    assert loaded.validate_compatibility(original)


def test_metadata_update_centralized():
    # 初始化时提供所有必需参数
    metadata = FeatureMetadata(
        feature_params={"technical_indicators": ["MA", "RSI"]},  # 示例参数
        data_source_version="v1.0",
        feature_list=["close", "volume"],  # 初始特征列表
        scaler_path=Path("/fake/scaler.pkl"),
        selector_path=Path("/fake/selector.pkl")
    )

    # 更新特征列表
    new_features = ["close", "volume", "MA_20", "RSI_14"]
    metadata.update_feature_columns(new_features)

    # 验证更新后的特征列表和版本时间戳
    assert metadata.feature_list == new_features
    assert metadata.last_updated > metadata.created_at


def test_metadata_version_conflict():
    """测试数据源版本冲突检测"""
    meta1 = FeatureMetadata(
        feature_params={"technical_indicators": ["MA", "RSI"]},  # 示例参数
        data_source_version="v1.0",
        feature_list=["close", "volume"],  # 初始特征列表
        scaler_path=Path("/fake/scaler.pkl"),
        selector_path=Path("/fake/selector.pkl")
    )
    meta2 = FeatureMetadata(
        feature_params={"technical_indicators": ["MA", "RSI"]},  # 示例参数
        data_source_version="v2.0",
        feature_list=["close", "volume"],  # 初始特征列表
        scaler_path=Path("/fake/scaler.pkl"),
        selector_path=Path("/fake/selector.pkl")
    )
    assert not meta1.validate_compatibility(meta2)


def test_update_feature_columns_timestamp_increment():
    """测试更新特征列时时间戳严格递增"""
    meta = FeatureMetadata({}, "v1.0", [])
    original_time = meta.last_updated
    time.sleep(0.001)
    meta.update_feature_columns(["new_feature"])
    assert meta.last_updated > original_time
    assert meta.last_updated > meta.created_at


def test_timestamp_strict_increase(caplog):
    """测试时间戳严格递增验证"""
    caplog.set_level(logging.ERROR)  # 设置日志级别为 ERROR
    meta = FeatureMetadata(
        feature_params={},
        data_source_version="v1.0",
        feature_list=[]
    )

    # 创建时间戳严格递增的数据
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3)
    }).set_index('date')

    # 验证时间戳严格递增
    try:
        meta._validate_alignment(data)
    except ValueError:
        pytest.fail("不应抛出异常")

    # 创建时间戳不严格递增的数据（包含重复索引）
    non_increasing_data = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
    }).set_index('date')

    # 验证时间戳不严格递增时抛出异常
    with pytest.raises(ValueError, match="索引存在重复日期") as excinfo:
        meta._validate_alignment(non_increasing_data)
    assert "索引存在重复日期" in str(excinfo.value)

    # 验证日志输出
    assert "索引存在重复日期" in caplog.text


def test_metadata_init_type_errors():
    """测试初始化类型错误"""
    with pytest.raises(TypeError):
        FeatureMetadata("not_dict", "v1.0", [])

    with pytest.raises(TypeError):
        FeatureMetadata({}, 123, [])

    with pytest.raises(TypeError):
        FeatureMetadata({}, "v1.0", "not_list")


def test_validate_alignment_errors():
    """测试时间戳验证错误场景"""
    meta = FeatureMetadata({}, "v1.0", [])

    # 非时间戳索引
    with pytest.raises(ValueError, match="索引不是时间戳类型"):
        data = pd.DataFrame(index=[1, 2, 3])
        meta._validate_alignment(data)

    # 未来日期
    future_dates = pd.date_range(pd.Timestamp.now() + pd.Timedelta(days=1), periods=3)
    with pytest.raises(ValueError, match="检测到未来日期数据"):
        meta._validate_alignment(pd.DataFrame(index=future_dates))

    # 重复索引
    dup_dates = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="索引存在重复日期"):
        meta._validate_alignment(pd.DataFrame(index=dup_dates))

    # 非单调递增
    non_mono = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="索引非单调递增"):
        meta._validate_alignment(pd.DataFrame(index=non_mono))


def test_update_feature_columns_timestamp():
    """测试时间戳严格递增逻辑"""
    metadata = FeatureMetadata(feature_params={}, data_source_version="v1.0", feature_list=[])
    original_time = metadata.last_updated

    metadata.update_feature_columns(["close"])
    assert metadata.last_updated > original_time

    # 模拟相同时间戳场景
    with mock.patch('time.time', return_value=metadata.last_updated):
        metadata.update_feature_columns(["open"])
    assert metadata.last_updated > original_time + 1e-6


def test_validate_alignment_future_dates():
    """测试未来日期数据验证"""
    # 定义必需的参数
    feature_params = {}  # 特征工程参数
    data_source_version = "v1.0"  # 数据源版本标识
    feature_list = []  # 特征列表

    # 正确构造 FeatureMetadata 实例
    metadata = FeatureMetadata(
        feature_params=feature_params,
        data_source_version=data_source_version,
        feature_list=feature_list,
        scaler_path=None,  # 可选参数
        selector_path=None  # 可选参数
    )
    future_date = pd.Timestamp.now() + pd.Timedelta(days=1)
    data = pd.DataFrame(index=[future_date])
    with pytest.raises(ValueError):
        metadata._validate_alignment(data)


def test_path_validation(tmp_path):
    """测试路径验证"""
    # 测试有效路径
    valid_path = tmp_path / "valid.pkl"
    meta = FeatureMetadata(
        feature_params={},
        data_source_version="v1.0",
        feature_list=[],
        scaler_path=valid_path,
        selector_path=valid_path
    )
    assert meta.scaler_path == valid_path
    assert meta.selector_path == valid_path

    # 测试无效路径
    with pytest.raises(TypeError):
        FeatureMetadata(
            feature_params={},
            data_source_version="v1.0",
            feature_list=[],
            scaler_path="invalid_path_string"  # 应该是Path对象
        )


def test_deep_feature_params_comparison():
    """测试特征参数深度比较"""
    params1 = {
        "technical": {
            "ma": {"window": 20},
            "rsi": {"period": 14}
        }
    }
    params2 = copy.deepcopy(params1)
    params2["technical"]["ma"]["window"] = 30

    meta1 = FeatureMetadata(params1, "v1.0", [])
    meta2 = FeatureMetadata(params2, "v1.0", [])

    assert not meta1.validate_compatibility(meta2)


def test_large_feature_list():
    """测试大规模特征列表处理"""
    # 创建1000个特征名
    large_feature_list = [f"feature_{i}" for i in range(1000)]

    start_time = time.time()
    meta = FeatureMetadata({}, "v1.0", large_feature_list)
    meta.update(large_feature_list + ["new_feature"])
    end_time = time.time()

    # 验证性能（应该在1秒内完成）
    assert end_time - start_time < 1
    assert len(meta.feature_list) == 1001


def test_metadata_version_upgrade():
    """测试元数据版本升级"""
    old_meta = FeatureMetadata({}, "v1.0", [])
    old_meta.version = "0.9"  # 模拟旧版本

    # 序列化和反序列化，应该自动升级到新版本
    with patch('joblib.dump') as mock_dump:
        old_meta.save(Path("dummy.pkl"))
        # 验证保存的是新版本
        assert mock_dump.call_args[0][0].version == "1.0"


def test_logging_completeness(caplog):
    """测试日志记录完整性"""
    caplog.set_level(logging.INFO)
    meta = FeatureMetadata({}, "v1.0", ["feature1"])

    # 测试特征更新日志
    meta.update(["feature1", "feature2"])
    assert any("更新特征列表" in record.message for record in caplog.records)

    # 测试验证错误日志
    caplog.clear()
    with pytest.raises(ValueError):
        meta._validate_alignment(pd.DataFrame(index=[1, 2, 3]))
    assert any("索引不是时间戳类型" in record.message for record in caplog.records)


def test_edge_cases():
    """测试边界条件"""
    # 空特征参数
    meta = FeatureMetadata({}, "v1.0", [])
    assert meta.feature_params == {}

    # 空特征列表
    assert meta.feature_list == []

    # 特殊字符版本号
    meta = FeatureMetadata({}, "v1.0-beta.1", [])
    assert meta.data_source_version == "v1.0-beta.1"

    # 极长特征名
    long_feature_name = "x" * 1000
    meta = FeatureMetadata({}, "v1.0", [long_feature_name])
    assert meta.feature_list[0] == long_feature_name


def test_performance():
    """测试性能"""
    # 准备大量数据
    large_params = {"param" + str(i): i for i in range(1000)}
    large_features = ["feature" + str(i) for i in range(1000)]

    # 测试初始化性能
    start_time = time.time()
    meta = FeatureMetadata(large_params, "v1.0", large_features)
    init_time = time.time() - start_time
    assert init_time < 1  # 初始化应该在1秒内完成

    # 测试更新性能
    start_time = time.time()
    meta.update(large_features + ["new_feature"])
    update_time = time.time() - start_time
    assert update_time < 1  # 更新应该在1秒内完成


def test_metadata_immutability():
    """测试元数据不可变性"""
    params = {"window": 20}
    features = ["f1", "f2"]
    meta = FeatureMetadata(params, "v1.0", features)

    # 修改原始参数和特征列表
    params["window"] = 30
    features.append("f3")

    # 验证元数据中的值没有改变
    assert meta.feature_params["window"] == 20
    assert len(meta.feature_list) == 2


def test_feature_list_validation():
    """测试特征列表验证"""
    # 测试重复特征名
    with pytest.raises(ValueError):
        FeatureMetadata({}, "v1.0", ["f1", "f1"])

    # 测试无效特征名
    with pytest.raises(ValueError):
        FeatureMetadata({}, "v1.0", ["", "f1"])

    # 测试特殊字符
    with pytest.raises(ValueError):
        FeatureMetadata({}, "v1.0", ["f1!", "f2"])


def test_metadata_copy():
    """测试元数据复制"""
    original = FeatureMetadata({}, "v1.0", ["f1"])
    copied = copy.deepcopy(original)

    # 修改复制的元数据
    copied.update(["f1", "f2"])

    # 验证原始元数据未改变
    assert len(original.feature_list) == 1
    assert len(copied.feature_list) == 2


def test_concurrent_updates():
    """测试并发更新场景"""
    meta = FeatureMetadata({}, "v1.0", ["f1"])

    # 模拟并发更新
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        meta.update(["f1", "f2"])
        first_update = meta.last_updated

        # 模拟同时更新
        mock_time.return_value = 1000.0
        meta.update(["f1", "f2", "f3"])
        second_update = meta.last_updated

        # 验证时间戳严格递增
        assert second_update > first_update


def test_metadata_memory_usage():
    """测试元数据内存使用"""
    # 创建大量特征
    large_features = [f"feature_{i}" for i in range(10000)]
    meta = FeatureMetadata({}, "v1.0", large_features)

    # 获取内存使用
    import sys
    memory_size = sys.getsizeof(meta.feature_list)

    # 验证内存使用在合理范围内（小于1MB）
    assert memory_size < 1024 * 1024
