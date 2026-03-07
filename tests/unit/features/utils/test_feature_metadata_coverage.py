#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureMetadata测试覆盖
测试utils/feature_metadata.py
"""

import pytest
import tempfile
import os
from src.features.utils.feature_metadata import FeatureMetadata


class TestFeatureMetadata:
    """FeatureMetadata测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        metadata = FeatureMetadata()
        assert metadata.feature_params == {}
        assert metadata.data_source_version == "1.0"
        assert metadata.feature_list == []
        assert metadata.scaler_path is None
        assert metadata.selector_path is None
        assert metadata.metadata == {}
        assert metadata.created_at is not None
        assert metadata.last_updated is not None

    def test_initialization_with_params(self):
        """测试带参数初始化"""
        feature_params = {"param1": "value1", "param2": 123}
        feature_list = ["feature1", "feature2", "feature3"]
        metadata = FeatureMetadata(
            feature_params=feature_params,
            data_source_version="2.0",
            feature_list=feature_list,
            scaler_path="/path/to/scaler",
            selector_path="/path/to/selector"
        )
        assert metadata.feature_params == feature_params
        assert metadata.data_source_version == "2.0"
        assert metadata.feature_list == feature_list
        assert metadata.scaler_path == "/path/to/scaler"
        assert metadata.selector_path == "/path/to/selector"

    def test_initialization_with_version_alias(self):
        """测试使用version参数别名"""
        metadata = FeatureMetadata(version="3.0")
        assert metadata.data_source_version == "3.0"

    def test_initialization_invalid_feature_params_type(self):
        """测试无效的feature_params类型"""
        with pytest.raises(TypeError, match="feature_params must be a dict"):
            FeatureMetadata(feature_params="not a dict")

    def test_initialization_duplicate_features(self):
        """测试重复特征列表"""
        with pytest.raises(ValueError, match="feature_list contains duplicate features"):
            FeatureMetadata(feature_list=["feature1", "feature2", "feature1"])

    def test_initialization_empty_feature_name(self):
        """测试空特征名"""
        with pytest.raises(ValueError, match="feature_list contains empty feature name"):
            FeatureMetadata(feature_list=["feature1", "", "feature2"])

    def test_initialization_deep_copy(self):
        """测试深拷贝"""
        original_params = {"param": "value"}
        original_list = ["feature1", "feature2"]
        metadata = FeatureMetadata(
            feature_params=original_params,
            feature_list=original_list
        )
        # 修改原始对象不应影响metadata
        original_params["new_param"] = "new_value"
        original_list.append("feature3")
        assert "new_param" not in metadata.feature_params
        assert "feature3" not in metadata.feature_list

    def test_update(self):
        """测试update方法"""
        import time
        metadata = FeatureMetadata(feature_list=["old1", "old2"])
        original_updated = metadata.last_updated
        time.sleep(0.01)  # 确保时间戳不同
        new_list = ["new1", "new2", "new3"]
        metadata.update(new_list)
        assert metadata.feature_list == new_list
        assert metadata.last_updated >= original_updated

    def test_update_feature_columns(self):
        """测试update_feature_columns方法"""
        metadata = FeatureMetadata(feature_list=["old1", "old2"])
        original_updated = metadata.last_updated
        import time
        time.sleep(0.01)  # 确保时间戳不同
        new_list = ["new1", "new2"]
        metadata.update_feature_columns(new_list)
        assert metadata.feature_list == new_list
        assert metadata.last_updated > original_updated

    def test_validate_compatibility(self):
        """测试validate_compatibility方法"""
        metadata1 = FeatureMetadata()
        metadata2 = FeatureMetadata()
        # 当前实现总是返回True
        assert metadata1.validate_compatibility(metadata2) is True

    def test_save(self):
        """测试save方法"""
        metadata = FeatureMetadata()
        # 当前实现是pass，不会抛出异常
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metadata.pkl")
            metadata.save(path)  # 应该不抛出异常

    def test_save_metadata(self):
        """测试save_metadata方法"""
        metadata = FeatureMetadata()
        # 当前实现是pass，不会抛出异常
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metadata.json")
            metadata.save_metadata(path)  # 应该不抛出异常

