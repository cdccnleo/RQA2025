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


import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from src.data.alignment.data_aligner import (
    AlignmentMethod,
    DataAligner,
    FrequencyType,
)
from src.infrastructure.utils.exceptions import DataProcessingError


class DummyProcessor:
    """简化版处理器，用于隔离 DataAligner 逻辑。"""

    def fill_missing(self, df: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        if not method:
            return df
        if method in {"forward", "ffill"}:
            return df.ffill()
        if method in {"backward", "bfill"}:
            return df.bfill()
        if method == "zero":
            return df.fillna(0)
        return df.fillna(0)

    def merge_data(
        self,
        data_frames: dict[str, pd.DataFrame],
        merge_on: Optional[str] = None,
        how: str = "outer",
        suffixes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        renamed_frames: list[pd.DataFrame] = []
        for idx, (name, frame) in enumerate(data_frames.items()):
            suffix = (
                suffixes[idx]
                if suffixes and idx < len(suffixes)
                else f"_{name}"
            )
            renamed_frames.append(frame.add_suffix(suffix))
        return pd.concat(renamed_frames, axis=1, join=how)

    def resample_data(
        self,
        df: pd.DataFrame,
        freq: str,
        method: str = "mean",
        fill_method: Optional[str] = None,
    ) -> pd.DataFrame:
        agg_method = method if method in {"mean", "sum", "max", "min"} else "mean"
        resampled = getattr(df.resample(freq), agg_method)()
        if fill_method:
            resampled = self.fill_missing(resampled, fill_method)
        return resampled


@pytest.fixture
def data_aligner() -> DataAligner:
    aligner = DataAligner()
    aligner.processor = DummyProcessor()
    return aligner


def _make_series(start: str, periods: int, freq: str, values: list[int]) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq)
    return pd.DataFrame({"value": values}, index=index)


def test_align_time_series_enum_and_fill(data_aligner: DataAligner):
    frames = {
        "source_a": _make_series("2025-01-01", 2, "D", [1, 2]),
        "source_b": _make_series("2025-01-02", 2, "D", [5, 6]),
    }
    result = data_aligner.align_time_series(
        frames,
        freq=FrequencyType.DAILY,
        method=AlignmentMethod.OUTER,
        fill_method={"source_a": "forward", "source_b": "zero"},
    )

    expected_index = pd.date_range("2025-01-01", "2025-01-03", freq="D")
    pd.testing.assert_index_equal(result["source_a"].index, expected_index)
    assert result["source_a"].loc["2025-01-03", "value"] == 2
    assert result["source_b"].loc["2025-01-01", "value"] == 0
    assert len(data_aligner.get_alignment_history()) == 1


def test_align_time_series_invalid_index_raises(data_aligner: DataAligner):
    broken = pd.DataFrame({"value": [1]}, index=["invalid-date"])
    with pytest.raises(DataProcessingError) as exc:
        data_aligner.align_time_series({"broken": broken})
    assert "无法转换为DatetimeIndex" in str(exc.value)


def test_align_and_merge_applies_suffixes(data_aligner: DataAligner):
    frames = {
        "alpha": _make_series("2025-01-01", 2, "D", [10, 11]),
        "beta": _make_series("2025-01-01", 2, "D", [20, 21]),
    }
    merged = data_aligner.align_and_merge(
        frames,
        freq="D",
        method="outer",
        suffixes=["_a", "_b"],
    )

    assert {"value_a", "value_b"} <= set(merged.columns)
    assert merged.loc["2025-01-01", "value_a"] == 10
    assert merged.loc["2025-01-01", "value_b"] == 20


def test_align_to_reference_with_frequency_fallback(data_aligner: DataAligner):
    reference = pd.DataFrame(
        {"ref": [1, 2, 3]},
        index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04"]),
    )
    target = {
        "alpha": pd.DataFrame(
            {"value": [5]},
            index=pd.to_datetime(["2025-01-03"]),
        )
    }

    aligned = data_aligner.align_to_reference(
        reference_df=reference,
        target_dfs=target,
        freq=None,
        fill_method={"alpha": "zero"},
    )

    pd.testing.assert_index_equal(aligned["alpha"].index, reference.index)
    assert aligned["alpha"].loc["2025-01-01", "value"] == 0
    assert "reference" in aligned


def test_align_multi_frequency_resamples_and_fills(data_aligner: DataAligner):
    frames = {
        "fast": _make_series("2025-01-01 00:00", 4, "H", [1, 2, 3, 4]),
        "slow": _make_series("2025-01-01 00:00", 3, "2H", [10, 20, 30]),
    }

    aligned = data_aligner.align_multi_frequency(
        frames,
        target_freq=FrequencyType.HOURLY,
        resample_methods={"fast": "mean", "slow": "sum"},
        fill_method={"fast": "forward", "slow": "zero"},
    )

    expected_index = pd.date_range("2025-01-01 00:00", "2025-01-01 04:00", freq="H")
    pd.testing.assert_index_equal(aligned["fast"].index, expected_index)
    assert aligned["slow"].loc["2025-01-01 01:00", "value"] == 0


def test_alignment_history_persistence_roundtrip(tmp_path: Path, data_aligner: DataAligner):
    frames = {
        "source_a": _make_series("2025-01-01", 1, "D", [1]),
        "source_b": _make_series("2025-01-02", 1, "D", [2]),
    }
    data_aligner.align_time_series(frames)

    history_file = tmp_path / "history.json"
    data_aligner.save_alignment_history(history_file)

    reloaded = DataAligner()
    reloaded.load_alignment_history(history_file)

    assert history_file.exists()
    assert len(reloaded.get_alignment_history()) == 1
    with history_file.open() as f:
        persisted = json.load(f)
    assert persisted[0]["input_sources"] == ["source_a", "source_b"]

