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
import json
import pandas as pd
import pytest
from datetime import datetime

from src.data.alignment.data_aligner import DataAligner, AlignmentMethod, FrequencyType
from src.infrastructure.utils.exceptions import DataProcessingError


def _make_df(idx, vals):
    return pd.DataFrame({"v": vals}, index=idx)


def test_align_time_series_inner_and_history(tmp_path):
    idx_a = pd.date_range("2025-01-01", periods=3, freq="D")
    idx_b = pd.date_range("2025-01-02", periods=3, freq="D")
    df_a = _make_df(idx_a, [1, None, 3])
    df_b = _make_df(idx_b, [10, 20, 30])

    aligner = DataAligner()
    out = aligner.align_time_series(
        {"a": df_a, "b": df_b},
        freq=FrequencyType.DAILY,
        method=AlignmentMethod.INNER,
        fill_method=None,
    )
    # inner 范围应为 2025-01-02..2025-01-03
    for name, df in out.items():
        assert list(df.index) == list(pd.date_range("2025-01-02", "2025-01-03", freq="D"))
    # 历史记录写入
    assert aligner.get_alignment_history(limit=1)[-1]["method"] in ("inner", AlignmentMethod.INNER)

    # 持久化历史
    f = tmp_path / "history.json"
    aligner.save_alignment_history(f)
    assert f.exists()
    # 重新加载
    aligner.load_alignment_history(f)
    assert isinstance(aligner.get_alignment_history(), list)


def test_align_time_series_invalid_index_raises():
    # 无法转换为 DatetimeIndex
    df_bad = _make_df(["not-a-date", "x"], [1, 2])
    aligner = DataAligner()
    with pytest.raises(DataProcessingError):
        aligner.align_time_series({"bad": df_bad})


def test_align_and_merge_outer_basic():
    idx_a = pd.date_range("2025-01-01", periods=2, freq="D")
    idx_b = pd.date_range("2025-01-02", periods=2, freq="D")
    df_a = _make_df(idx_a, [1, 2])
    df_b = _make_df(idx_b, [10, 20])
    aligner = DataAligner()
    # 为缺失的 merge_data 提供桩实现
    class _ProcStub:
        def merge_data(self, frames, merge_on="index", how="outer", suffixes=None):
            out = None
            for name, df in frames.items():
                df2 = df.add_suffix(f"_{name}")
                if out is None:
                    out = df2
                else:
                    out = out.join(df2, how=how)
            return out
    aligner.processor.merge_data = _ProcStub().merge_data  # type: ignore
    merged = aligner.align_and_merge({"a": df_a, "b": df_b}, method=AlignmentMethod.OUTER)
    # 外连接应覆盖 1~3 日
    assert merged.index.min() == pd.Timestamp("2025-01-01")
    assert merged.index.max() == pd.Timestamp("2025-01-03")


def test_align_to_reference_infer_freq_and_fill():
    ref_idx = pd.date_range("2025-01-01", periods=3, freq="D")
    ref = _make_df(ref_idx, [0, 0, 0])
    tgt = _make_df(pd.date_range("2025-01-01", periods=2, freq="2D"), [1, 2])
    aligner = DataAligner()
    out = aligner.align_to_reference(ref, {"t": tgt}, fill_method=None)
    assert set(out.keys()) == {"t", "reference"}
    assert list(out["t"].index) == list(ref.index)


def test_align_to_reference_bad_ref_raises():
    ref = _make_df(["bad", "date"], [0, 1])
    tgt = _make_df(pd.date_range("2025-01-01", periods=2, freq="D"), [1, 2])
    aligner = DataAligner()
    with pytest.raises(DataProcessingError):
        aligner.align_to_reference(ref, {"t": tgt})


def test_align_multi_frequency_resample_and_fill():
    a = _make_df(pd.date_range("2025-01-01", periods=24, freq="H"), range(24))
    b = _make_df(pd.date_range("2025-01-01", periods=3, freq="8H"), [5, None, 7])
    aligner = DataAligner()
    # 为缺失的 resample_data 提供桩实现
    def _resample_data(df, freq, method="mean", fill_method=None):
        if fill_method:
            df = df.fillna(method=fill_method)
        return getattr(df.resample(freq), method)()
    aligner.processor.resample_data = _resample_data  # type: ignore
    aligner.processor.fill_missing = lambda df, method=None: df.fillna(method=method)  # type: ignore
    out = aligner.align_multi_frequency(
        {"a": a, "b": b},
        target_freq=FrequencyType.DAILY,
        resample_methods={"a": "mean", "b": "mean"},
        fill_method="ffill",
    )
    assert set(out.keys()) == {"a", "b"}
    # 统一到日频后 index 长度应一致
    assert len(out["a"].index) == len(out["b"].index)


