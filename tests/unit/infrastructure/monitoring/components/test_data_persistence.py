#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据持久化组件
"""

import importlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def data_persistence_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.data_persistence"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def persistence(data_persistence_module, tmp_path):
    """创建DataPersistence实例"""
    data_file = tmp_path / "test_monitoring_data.json"
    return data_persistence_module.DataPersistence(
        max_history_items=100,
        data_file=str(data_file)
    )


def test_initialization(persistence):
    """测试初始化"""
    assert persistence.max_history_items == 100
    assert persistence.metrics_history == []
    assert persistence.data_file.endswith("test_monitoring_data.json")


def test_initialization_default_values(data_persistence_module):
    """测试初始化（默认值）"""
    persistence = data_persistence_module.DataPersistence()
    assert persistence.max_history_items == 1000
    assert persistence.data_file == 'monitoring_data.json'


def test_save_monitoring_data(persistence):
    """测试保存监控数据"""
    timestamp = datetime.now()
    data = {'cpu_usage': 50.0, 'memory_usage': 60.0}
    
    persistence.save_monitoring_data(timestamp, data)
    
    assert len(persistence.metrics_history) == 1
    record = persistence.metrics_history[0]
    assert record['timestamp'] == timestamp.isoformat()
    assert record['data'] == data


def test_save_monitoring_data_max_history(persistence):
    """测试保存监控数据（最大历史限制）"""
    persistence.max_history_items = 3
    
    for i in range(5):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    assert len(persistence.metrics_history) == 3
    assert persistence.metrics_history[0]['data']['index'] == 2  # 保留最后3个


def test_persist_monitoring_data_success(persistence):
    """测试持久化监控数据（成功）"""
    # 先添加一些历史数据
    timestamp = datetime.now()
    persistence.save_monitoring_data(timestamp, {
        'coverage': {'coverage_percent': 80.0},
        'performance': {'memory_usage_mb': 100.0, 'cpu_usage_percent': 50.0},
        'health': {'overall_status': 'healthy'}
    })
    
    config = {'interval_seconds': 60}
    alerts_history = [{'alert_id': '1'}]
    optimization_suggestions = [{'suggestion': 'test'}]
    
    persistence.persist_monitoring_data(config, alerts_history, optimization_suggestions)
    
    # 验证文件已创建
    assert Path(persistence.data_file).exists()
    
    # 验证文件内容
    with open(persistence.data_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        assert saved_data['config'] == config
        assert len(saved_data['metrics_history']) == 1
        assert len(saved_data['alerts_history']) == 1
        assert len(saved_data['optimization_suggestions']) == 1


def test_persist_monitoring_data_limit_alerts(persistence):
    """测试持久化监控数据（限制告警数量）"""
    # 添加历史数据
    persistence.save_monitoring_data(datetime.now(), {})
    
    alerts_history = [{'alert_id': str(i)} for i in range(60)]  # 60条告警
    
    persistence.persist_monitoring_data({}, alerts_history, [])
    
    with open(persistence.data_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        assert len(saved_data['alerts_history']) == 50  # 只保存最近50条


def test_persist_monitoring_data_limit_suggestions(persistence):
    """测试持久化监控数据（限制建议数量）"""
    persistence.save_monitoring_data(datetime.now(), {})
    
    optimization_suggestions = [{'suggestion': str(i)} for i in range(25)]  # 25条建议
    
    persistence.persist_monitoring_data({}, [], optimization_suggestions)
    
    with open(persistence.data_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        assert len(saved_data['optimization_suggestions']) == 20  # 只保存最近20条


def test_persist_monitoring_data_exception(persistence, monkeypatch, capsys):
    """测试持久化监控数据（异常）"""
    persistence.save_monitoring_data(datetime.now(), {})
    
    # 模拟文件写入失败
    def failing_open(*args, **kwargs):
        raise IOError("Write error")
    
    monkeypatch.setattr('builtins.open', failing_open)
    
    persistence.persist_monitoring_data({}, [], [])
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert '保存监控数据失败' in captured.out or 'Write error' in captured.out


def test_format_metrics_history(persistence):
    """测试格式化指标历史数据"""
    timestamp = datetime.now()
    persistence.save_monitoring_data(timestamp, {
        'coverage': {'coverage_percent': 80.0},
        'performance': {'memory_usage_mb': 100.0, 'cpu_usage_percent': 50.0},
        'health': {'overall_status': 'healthy'}
    })
    
    formatted = persistence._format_metrics_history()
    
    assert len(formatted) == 1
    assert formatted[0]['coverage_percent'] == 80.0
    assert formatted[0]['memory_usage_mb'] == 100.0
    assert formatted[0]['cpu_usage_percent'] == 50.0
    assert formatted[0]['overall_health'] == 'healthy'


def test_format_metrics_history_limit(persistence):
    """测试格式化指标历史数据（限制数量）"""
    # 添加超过100条记录
    for i in range(120):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    formatted = persistence._format_metrics_history()
    
    assert len(formatted) == 100  # 只格式化最近100条


def test_format_metrics_history_missing_fields(persistence, capsys):
    """测试格式化指标历史数据（缺少字段）"""
    # 添加不完整的数据
    persistence.save_monitoring_data(datetime.now(), {})  # 空数据
    
    formatted = persistence._format_metrics_history()
    
    # 应该使用默认值
    assert len(formatted) == 1
    assert formatted[0]['coverage_percent'] == 0
    assert formatted[0]['overall_health'] == 'unknown'


def test_format_metrics_history_exception(persistence, capsys):
    """测试格式化指标历史数据（异常）"""
    # 添加会导致异常的数据
    persistence.metrics_history.append({
        'timestamp': 'invalid',
        'data': object()  # 无法访问的对象
    })
    
    formatted = persistence._format_metrics_history()
    
    # 应该跳过有问题的记录
    assert isinstance(formatted, list)


def test_load_monitoring_data_success(persistence):
    """测试加载监控数据（成功）"""
    # 先保存一些数据
    persistence.save_monitoring_data(datetime.now(), {'test': 'data'})
    persistence.persist_monitoring_data({}, [], [])
    
    # 加载数据
    loaded_data = persistence.load_monitoring_data()
    
    assert loaded_data is not None
    assert 'config' in loaded_data
    assert 'metrics_history' in loaded_data


def test_load_monitoring_data_file_not_exists(persistence):
    """测试加载监控数据（文件不存在）"""
    # 确保文件不存在
    if Path(persistence.data_file).exists():
        Path(persistence.data_file).unlink()
    
    loaded_data = persistence.load_monitoring_data()
    
    assert loaded_data is None


def test_load_monitoring_data_invalid_json(persistence, capsys):
    """测试加载监控数据（无效JSON）"""
    # 创建无效的JSON文件
    Path(persistence.data_file).write_text("invalid json", encoding='utf-8')
    
    loaded_data = persistence.load_monitoring_data()
    
    assert loaded_data is None
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert '加载监控数据失败' in captured.out


def test_load_monitoring_data_exception(persistence, monkeypatch, capsys):
    """测试加载监控数据（异常）"""
    # 先创建一个文件
    Path(persistence.data_file).write_text('{"test": "data"}', encoding='utf-8')
    
    # 模拟文件读取失败
    import builtins
    original_open = builtins.open
    
    def failing_open(*args, **kwargs):
        if 'r' in kwargs.get('mode', '') or len(args) > 1 and 'r' in args[1]:
            raise IOError("Read error")
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr(builtins, 'open', failing_open)
    
    loaded_data = persistence.load_monitoring_data()
    
    assert loaded_data is None
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert '加载监控数据失败' in captured.out or 'Read error' in captured.out


def test_export_data_default_filename(persistence):
    """测试导出数据（默认文件名）"""
    persistence.save_monitoring_data(datetime.now(), {'test': 'data'})
    
    export_file = persistence.export_data()
    
    assert export_file.startswith('monitoring_export_')
    assert export_file.endswith('.json')
    assert Path(export_file).exists()


def test_export_data_custom_filename(persistence, tmp_path):
    """测试导出数据（自定义文件名）"""
    persistence.save_monitoring_data(datetime.now(), {'test': 'data'})
    
    custom_file = tmp_path / "custom_export.json"
    export_file = persistence.export_data(str(custom_file))
    
    assert export_file == str(custom_file)
    assert custom_file.exists()
    
    # 验证导出内容
    with open(custom_file, 'r', encoding='utf-8') as f:
        exported = json.load(f)
        assert 'export_timestamp' in exported
        assert 'metrics_history' in exported
        assert 'export_info' in exported


def test_export_data_exception(persistence, monkeypatch, capsys):
    """测试导出数据（异常）"""
    persistence.save_monitoring_data(datetime.now(), {'test': 'data'})
    
    # 模拟文件写入失败
    def failing_open(*args, **kwargs):
        raise IOError("Export error")
    
    monkeypatch.setattr('builtins.open', failing_open)
    
    export_file = persistence.export_data()
    
    assert export_file == ""
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert '导出监控数据失败' in captured.out or 'Export error' in captured.out


def test_get_metrics_history_no_limit(persistence):
    """测试获取指标历史（无限制）"""
    for i in range(5):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    history = persistence.get_metrics_history()
    
    assert len(history) == 5
    assert history == persistence.metrics_history.copy()


def test_get_metrics_history_with_limit(persistence):
    """测试获取指标历史（有限制）"""
    for i in range(10):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    history = persistence.get_metrics_history(limit=5)
    
    assert len(history) == 5
    assert history == persistence.metrics_history[-5:]


def test_get_metrics_history_zero_limit(persistence):
    """测试获取指标历史（限制为0）"""
    persistence.save_monitoring_data(datetime.now(), {'test': 'data'})
    
    history = persistence.get_metrics_history(limit=0)
    
    assert history == []


def test_clear_history(persistence):
    """测试清空历史数据"""
    for i in range(5):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    assert len(persistence.metrics_history) == 5
    
    persistence.clear_history()
    
    assert len(persistence.metrics_history) == 0


def test_get_latest_metrics(persistence):
    """测试获取最新的指标数据"""
    # 没有数据时应该返回None
    assert persistence.get_latest_metrics() is None
    
    # 添加数据后应该返回最新的
    timestamp1 = datetime.now()
    persistence.save_monitoring_data(timestamp1, {'first': 1})
    
    timestamp2 = datetime.now()
    persistence.save_monitoring_data(timestamp2, {'second': 2})
    
    latest = persistence.get_latest_metrics()
    assert latest is not None
    assert latest['data']['second'] == 2


def test_get_metrics_count(persistence):
    """测试获取指标记录数量"""
    assert persistence.get_metrics_count() == 0
    
    for i in range(5):
        persistence.save_monitoring_data(datetime.now(), {'index': i})
    
    assert persistence.get_metrics_count() == 5

