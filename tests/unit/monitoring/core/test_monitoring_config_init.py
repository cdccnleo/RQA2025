#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig初始化测试
补充MonitoringSystem类的__init__方法和初始化逻辑测试
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入监控配置模块
try:
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("MonitoringSystem不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestMonitoringSystemInit:
    """测试MonitoringSystem初始化"""

    def test_init_default_state(self):
        """测试默认初始化状态"""
        monitoring_system = MonitoringSystem()
        
        # 验证所有属性都被正确初始化
        assert hasattr(monitoring_system, 'metrics')
        assert hasattr(monitoring_system, 'traces')
        assert hasattr(monitoring_system, 'alerts')
        
        # 验证初始值为空
        assert monitoring_system.metrics == {}
        assert monitoring_system.traces == []
        assert monitoring_system.alerts == []

    def test_init_metrics_type(self):
        """测试metrics属性类型"""
        monitoring_system = MonitoringSystem()
        
        assert isinstance(monitoring_system.metrics, dict)

    def test_init_traces_type(self):
        """测试traces属性类型"""
        monitoring_system = MonitoringSystem()
        
        assert isinstance(monitoring_system.traces, list)

    def test_init_alerts_type(self):
        """测试alerts属性类型"""
        monitoring_system = MonitoringSystem()
        
        assert isinstance(monitoring_system.alerts, list)

    def test_init_multiple_instances_independent(self):
        """测试多个实例之间相互独立"""
        monitoring_system1 = MonitoringSystem()
        monitoring_system2 = MonitoringSystem()
        
        # 为第一个实例添加数据
        monitoring_system1.record_metric('cpu_usage', 50.0)
        span_id = monitoring_system1.start_trace('trace_1', 'op1')
        monitoring_system1.end_trace(span_id)
        monitoring_system1.check_alerts()
        
        # 验证第二个实例不受影响
        assert len(monitoring_system2.metrics) == 0
        assert len(monitoring_system2.traces) == 0
        assert len(monitoring_system2.alerts) == 0
        
        # 验证第一个实例有数据
        assert len(monitoring_system1.metrics) > 0
        assert len(monitoring_system1.traces) > 0

    def test_init_empty_dict_mutable(self):
        """测试metrics字典是可变的"""
        monitoring_system = MonitoringSystem()
        
        # 验证可以直接修改
        monitoring_system.metrics['test'] = []
        assert 'test' in monitoring_system.metrics

    def test_init_empty_list_mutable(self):
        """测试traces和alerts列表是可变的"""
        monitoring_system = MonitoringSystem()
        
        # 验证可以直接修改
        monitoring_system.traces.append({'test': 'value'})
        monitoring_system.alerts.append({'test': 'value'})
        
        assert len(monitoring_system.traces) == 1
        assert len(monitoring_system.alerts) == 1

    def test_init_no_side_effects(self):
        """测试初始化不会产生副作用"""
        monitoring_system1 = MonitoringSystem()
        monitoring_system2 = MonitoringSystem()
        
        # 验证两个实例的初始状态相同
        assert monitoring_system1.metrics == monitoring_system2.metrics
        assert monitoring_system1.traces == monitoring_system2.traces
        assert monitoring_system1.alerts == monitoring_system2.alerts
        
        # 但它们不是同一个对象
        assert monitoring_system1.metrics is not monitoring_system2.metrics
        assert monitoring_system1.traces is not monitoring_system2.traces
        assert monitoring_system1.alerts is not monitoring_system2.alerts

    def test_init_after_clear(self):
        """测试清空后重新使用"""
        monitoring_system = MonitoringSystem()
        
        # 添加一些数据
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        # 清空
        monitoring_system.metrics = {}
        monitoring_system.traces = []
        monitoring_system.alerts = []
        
        # 验证可以重新使用
        assert len(monitoring_system.metrics) == 0
        assert len(monitoring_system.traces) == 0
        assert len(monitoring_system.alerts) == 0
        
        # 可以重新添加数据
        monitoring_system.record_metric('memory_usage', 60.0)
        assert len(monitoring_system.metrics) == 1

    def test_init_state_ready_for_operations(self):
        """测试初始化后可以立即执行操作"""
        monitoring_system = MonitoringSystem()
        
        # 验证可以立即执行所有操作
        monitoring_system.record_metric('test_metric', 42.0)
        span_id = monitoring_system.start_trace('test_trace', 'test_op')
        monitoring_system.end_trace(span_id)
        monitoring_system.add_trace_event(span_id, 'test_event')
        monitoring_system.check_alerts()
        report = monitoring_system.generate_report()
        
        # 验证所有操作都成功
        assert len(monitoring_system.metrics) > 0
        assert len(monitoring_system.traces) > 0
        assert isinstance(report, dict)

    def test_init_attributes_not_none(self):
        """测试初始化后属性不为None"""
        monitoring_system = MonitoringSystem()
        
        assert monitoring_system.metrics is not None
        assert monitoring_system.traces is not None
        assert monitoring_system.alerts is not None

    def test_init_no_exception(self):
        """测试初始化不会抛出异常"""
        try:
            monitoring_system = MonitoringSystem()
            assert monitoring_system is not None
        except Exception as e:
            pytest.fail(f"初始化不应抛出异常: {e}")

    def test_init_can_be_called_multiple_times(self):
        """测试可以创建多个实例"""
        instances = []
        
        for i in range(10):
            instance = MonitoringSystem()
            instances.append(instance)
        
        # 验证所有实例都正确初始化
        assert len(instances) == 10
        for instance in instances:
            assert instance.metrics == {}
            assert instance.traces == []
            assert instance.alerts == []

    def test_init_thread_safety_basic(self):
        """测试基本线程安全性（初始化时）"""
        import threading
        
        instances = []
        
        def create_instance():
            instance = MonitoringSystem()
            instances.append(instance)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证所有实例都正确初始化
        assert len(instances) == 5
        for instance in instances:
            assert instance.metrics == {}
            assert instance.traces == []
            assert instance.alerts == []


