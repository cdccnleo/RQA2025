#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""General feature processor behaviour tests."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.features.processors.general_processor import FeatureProcessor as GeneralFeatureProcessor

pytestmark = pytest.mark.features


@pytest.fixture()
def raw_features():
    return pd.DataFrame(
        {
            "alpha": [1.0, 3.0, 3.0, None, 5.0],
            "beta": ["x", "y", "y", "z", None],
            "gamma": [10.0, 11.0, 11.0, 12.0, None],
        }
    )


def test_process_features_removes_duplicates_and_fills_na(raw_features: pd.DataFrame):
    processor = GeneralFeatureProcessor()
    config = SimpleNamespace(handle_missing_values=True)

    processed = processor.process_features(raw_features, config=config)
    # duplicates removed → expect fewer rows
    assert len(processed) < len(raw_features)
    # numeric/ categorical nulls filled
    assert processed["alpha"].isna().sum() == 0
    assert processed["beta"].isna().sum() == 0


def test_handle_missing_values_respects_column_types(raw_features: pd.DataFrame):
    processor = GeneralFeatureProcessor()
    result = processor._handle_missing_values(raw_features)
    assert result["gamma"].isna().sum() == 0
    assert result["beta"].isna().sum() == 0


def test_process_features_returns_empty_for_empty_input():
    processor = GeneralFeatureProcessor()
    empty = pd.DataFrame()
    processed = processor.process_features(empty)
    assert processed.empty

