#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullLinkMonitor导出指标测试
补充full_link_monitor.py中export_metrics方法的全面测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    engine_full_link_monitor_module = importlib.import_module('src.monitoring.engine.full_link_monitor')
    FullLinkMonitor = getattr(engine_full_link_monitor_module, 'FullLinkMonitor', None)
    AlertLevel = getattr(engine_full_link_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(engine_full_link_monitor_module, 'MonitorType', None)
    MetricData = getattr(engine_full_link_monitor_module, 'MetricData', None)
    AlertRule = getattr(engine_full_link_monitor_module, 'AlertRule', None)
    Alert = getattr(engine_full_link_monitor_module, 'Alert', None)
    
    if FullLinkMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestFullLinkMonitorExportMetrics:
    """测试FullLinkMonitor导出指标功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return FullLinkMonitor()

    @pytest.fixture
    def monitor_with_data(self, monitor):
        """准备有数据的monitor"""
        # 添加指标数据
        for i in range(10):
            metric = MetricData(
                name='cpu_usage',
                value=50.0 + i,
                timestamp=datetime.now() - timedelta(hours=i),
                tags={'host': 'test'},
                monitor_type=MonitorType.SYSTEM,
                source='test'
            )
            monitor.metrics_history['cpu_usage'].append(metric)
        
        # 添加告警数据
        alert = Alert(
            id='alert1',
            rule_name='high_cpu',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now() - timedelta(hours=5),
            message='High CPU usage'
        )
        monitor.alert_history.append(alert)
        
        return monitor

    def test_export_metrics_basic(self, monitor_with_data):
        """测试基本导出功能"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=24)
            
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0
            
            # 验证文件内容
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'export_time' in data
            assert 'period' in data
            assert 'metrics' in data
            assert 'alerts' in data
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_empty_data(self, monitor):
        """测试导出空数据"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file, hours=24)
            
            assert os.path.exists(temp_file)
            
            # 验证文件内容
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'metrics' in data
            assert 'alerts' in data
            assert len(data['metrics']) == 0
            assert len(data['alerts']) == 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_custom_hours(self, monitor_with_data):
        """测试自定义时间窗口"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=5)
            
            assert os.path.exists(temp_file)
            
            # 验证文件内容
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['period']['hours'] == 5
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_time_filtering(self, monitor):
        """测试时间过滤"""
        now = datetime.now()
        # 添加不同时间的指标
        old_metric = MetricData(
            name='cpu_usage',
            value=50.0,
            timestamp=now - timedelta(hours=30),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        recent_metric = MetricData(
            name='cpu_usage',
            value=60.0,
            timestamp=now - timedelta(hours=10),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        
        monitor.metrics_history['cpu_usage'].append(old_metric)
        monitor.metrics_history['cpu_usage'].append(recent_metric)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 应该只包含10小时前的指标（在24小时内）
            exported_metrics = data['metrics'].get('cpu_usage', [])
            assert len(exported_metrics) == 1
            assert exported_metrics[0]['value'] == 60.0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_alerts_filtering(self, monitor):
        """测试告警时间过滤"""
        now = datetime.now()
        old_alert = Alert(
            id='old_alert',
            rule_name='rule1',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=now - timedelta(hours=30),
            message='Old alert'
        )
        recent_alert = Alert(
            id='recent_alert',
            rule_name='rule2',
            metric_name='memory_usage',
            current_value=95.0,
            threshold='> 85',
            level=AlertLevel.ERROR,
            timestamp=now - timedelta(hours=10),
            message='Recent alert'
        )
        
        monitor.alert_history.append(old_alert)
        monitor.alert_history.append(recent_alert)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 应该只包含10小时前的告警（在24小时内）
            exported_alerts = data['alerts']
            assert len(exported_alerts) == 1
            assert exported_alerts[0]['id'] == 'recent_alert'
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_multiple_metrics(self, monitor):
        """测试导出多个指标"""
        now = datetime.now()
        metrics = {
            'cpu_usage': [50.0, 60.0, 70.0],
            'memory_usage': [65.0, 75.0, 85.0],
            'disk_usage': [70.0, 80.0, 90.0]
        }
        
        for metric_name, values in metrics.items():
            for i, value in enumerate(values):
                metric = MetricData(
                    name=metric_name,
                    value=value,
                    timestamp=now - timedelta(hours=i),
                    tags={},
                    monitor_type=MonitorType.SYSTEM,
                    source='test'
                )
                monitor.metrics_history[metric_name].append(metric)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert len(data['metrics']) == 3
            assert 'cpu_usage' in data['metrics']
            assert 'memory_usage' in data['metrics']
            assert 'disk_usage' in data['metrics']
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_data_structure(self, monitor_with_data):
        """测试导出数据结构"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            assert 'export_time' in data
            assert isinstance(data['export_time'], str)
            
            assert 'period' in data
            assert 'start' in data['period']
            assert 'end' in data['period']
            assert 'hours' in data['period']
            assert data['period']['hours'] == 24
            
            assert 'metrics' in data
            assert isinstance(data['metrics'], dict)
            
            assert 'alerts' in data
            assert isinstance(data['alerts'], list)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_metric_data_structure(self, monitor_with_data):
        """测试导出指标数据结构"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证指标数据结构
            if data['metrics']:
                metric_name = list(data['metrics'].keys())[0]
                metrics = data['metrics'][metric_name]
                
                if metrics:
                    metric = metrics[0]
                    assert 'name' in metric
                    assert 'value' in metric
                    assert 'timestamp' in metric
                    assert 'tags' in metric
                    assert 'monitor_type' in metric
                    assert 'source' in metric
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_alert_data_structure(self, monitor_with_data):
        """测试导出告警数据结构"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=24)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证告警数据结构
            if data['alerts']:
                alert = data['alerts'][0]
                assert 'id' in alert
                assert 'rule_name' in alert
                assert 'metric_name' in alert
                assert 'current_value' in alert
                assert 'threshold' in alert
                assert 'level' in alert
                assert 'timestamp' in alert
                assert 'message' in alert
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_zero_hours(self, monitor_with_data):
        """测试0小时窗口（边界情况）"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=0)
            
            assert os.path.exists(temp_file)
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['period']['hours'] == 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_export_metrics_file_error(self, mock_open, monitor_with_data):
        """测试文件写入错误处理"""
        with pytest.raises(IOError):
            monitor_with_data.export_metrics('/invalid/path/file.json', hours=24)

    def test_export_metrics_json_serialization(self, monitor_with_data):
        """测试JSON序列化"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor_with_data.export_metrics(temp_file, hours=24)
            
            # 验证文件可以正常解析为JSON
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证可以重新序列化
            json_string = json.dumps(data)
            assert isinstance(json_string, str)
            assert len(json_string) > 0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_metrics_utf8_encoding(self, monitor):
        """测试UTF-8编码支持"""
        # 添加包含中文的指标
        metric = MetricData(
            name='测试指标',
            value=50.0,
            timestamp=datetime.now(),
            tags={'描述': '中文标签'},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        monitor.metrics_history['测试指标'].append(metric)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            monitor.export_metrics(temp_file, hours=24)
            
            # 验证文件可以正常读取
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert '测试指标' in str(data)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)



