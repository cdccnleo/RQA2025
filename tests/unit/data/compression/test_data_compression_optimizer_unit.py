#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


import os
import zlib
import bz2
import lzma
import gzip
import pytest

from src.data.compression.data_compression_optimizer import (
    DataCompressionOptimizer,
    CompressionStrategy,
)


def test_small_data_skips_compression():
    opt = DataCompressionOptimizer()
    data = b"x" * 10  # 小于 1KB
    res = opt.compress_data(data, data_type="text")
    assert res["algorithm"] == "none"
    assert res["compressed_size"] == len(data)
    # 解压 none 返回原数据
    out = opt.decompress_data(res["compressed_data"], res["algorithm"])
    assert out == data


def test_gzip_roundtrip_and_metrics_growth():
    opt = DataCompressionOptimizer()
    data = (b"abc123" * 5000)  # 可压缩
    # 强制策略命中 gzip
    res = opt.compress_data(data, data_type="text", strategy_name="text_gzip_balanced")
    assert res["algorithm"] == "gzip"
    assert res["compressed_size"] < res["original_size"]
    back = opt.decompress_data(res["compressed_data"], "gzip")
    assert back == data
    # 指标入库
    assert opt.get_optimizer_status()["metrics_history_count"] >= 1


def test_strategy_applicability_thresholds():
    s = CompressionStrategy(
        name="test",
        algorithm="zlib",
        compression_level=6,
        min_size_threshold=100,
        max_size_threshold=200,
    )
    assert s.is_applicable(150, "x") is True
    assert s.is_applicable(50, "x") is False
    assert s.is_applicable(250, "x") is False
    s.enabled = False
    assert s.is_applicable(150, "x") is False


def test_add_enable_disable_remove_and_priority_update():
    opt = DataCompressionOptimizer()
    new_s = CompressionStrategy(name="tmp_rule", algorithm="zlib", compression_level=3)
    opt.add_compression_strategy(new_s)
    opt.enable_strategy("tmp_rule")
    opt.update_strategy_priority("tmp_rule", 9)
    opt.disable_strategy("tmp_rule")
    # 不抛异常即视为通过，再移除
    opt.remove_compression_strategy("tmp_rule")
    with pytest.raises(ValueError):
        opt.update_strategy_priority("tmp_rule", 1)


def test_get_compression_report_and_algorithm_performance():
    opt = DataCompressionOptimizer()
    data = (b"ab" * 2000)
    opt.compress_data(data, data_type="text", strategy_name="text_gzip_fast")
    report = opt.get_compression_report(time_range_hours=24)
    assert "summary" in report
    perf = opt.get_algorithm_performance("gzip")
    assert "algorithm" in perf


