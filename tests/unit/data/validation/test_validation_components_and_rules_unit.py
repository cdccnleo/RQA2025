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


import pandas as pd
import pytest

from src.data.validation.validator import DataValidator, ValidationResult, QualityReport, OutlierReport, ConsistencyReport
from src.data.validation.validator_components import ValidatorComponentFactory, ValidatorComponent
from src.data.validation.assertion_components import AssertionComponentFactory, AssertionComponent


def test_validator_dataframe_empty_and_all_null():
    v = DataValidator()
    df_empty = pd.DataFrame()
    r1 = v.validate_data(df_empty, data_type="stock")
    assert isinstance(r1, ValidationResult)
    assert r1.is_valid is False
    assert "数据为空" in r1.errors

    df_all_null = pd.DataFrame({"a": [None, None]})
    r2 = v.validate_data(df_all_null, data_type="stock")
    assert isinstance(r2, ValidationResult)
    assert r2.is_valid is False
    assert any("空值" in e or "全为空" in e for e in r2.errors)


def test_validator_dict_required_fields_and_schema():
    v = DataValidator()
    # missing required fields
    r = v.validate_data({"date": "2025-01-01"}, data_type="stock")
    assert r.is_valid is False
    assert "缺少必需字段" in r.errors[0]
    # schema positive（使用内部实现以适配当前接口）
    assert v._validate_dict_schema({"date": "2025-01-01", "symbol": "AAA"}, {"date": str, "symbol": str}) is True


def test_validator_quality_and_outliers_and_consistency_reports():
    v = DataValidator()
    df = pd.DataFrame({"a": [1, 2, 3, 100]})
    q = v.validate_quality(df)
    assert isinstance(q, QualityReport)
    o = v.detect_outliers(df["a"], threshold=1.5)
    assert isinstance(o, OutlierReport)
    c = v.validate_data_consistency(df)
    assert isinstance(c, ConsistencyReport)


def test_validator_date_range_numeric_and_no_duplicates():
    v = DataValidator()
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "x": [1.0, 2.0]}
    )
    assert bool(v.validate_date_range(df, "2025-01-01", "2025-12-31"))
    assert bool(v.validate_numeric_columns(df, ["x"]))
    assert bool(v.validate_no_missing_values(df, ["x"]))
    assert bool(v.validate_no_duplicates(df))


def test_validator_components_factory_and_instances():
    # Validator components
    ids = ValidatorComponentFactory.get_available_validators()
    assert isinstance(ids, list) and len(ids) > 0
    comp = ValidatorComponentFactory.create_component(ids[0])
    assert isinstance(comp, ValidatorComponent)
    info = comp.get_info()
    assert "validator_id" in info and "component_type" in info
    out = comp.process({"k": "v"})
    assert out["status"] == "success"
    status = comp.get_status()
    assert status["status"] == "active"

    # Assertion components
    aids = AssertionComponentFactory.get_available_assertions()
    assert isinstance(aids, list) and len(aids) > 0
    a = AssertionComponentFactory.create_component(aids[0])
    assert isinstance(a, AssertionComponent)
    ainfo = a.get_info()
    assert "assertion_id" in ainfo and "component_type" in ainfo
    aout = a.process({"k": "v"})
    assert aout["status"] == "success"
    astatus = a.get_status()
    assert astatus["status"] == "active"


def test_validator_components_factory_invalid_id_raises():
    with pytest.raises(ValueError):
        ValidatorComponentFactory.create_component(-1)
    with pytest.raises(ValueError):
        AssertionComponentFactory.create_component(-1)


