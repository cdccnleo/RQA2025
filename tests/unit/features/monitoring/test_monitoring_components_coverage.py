#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoring组件测试覆盖
测试analyzer_components, profiler_components, tracker_components
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.features.monitoring.analyzer_components import (
    IAnalyzerComponent,
    AnalyzerComponent,
    AnalyzerComponentFactory,
    create_analyzer_analyzer_component_3,
    create_analyzer_analyzer_component_8,
    create_analyzer_analyzer_component_13,
    create_analyzer_analyzer_component_18,
    create_analyzer_analyzer_component_23,
)

from src.features.monitoring.profiler_components import (
    IProfilerComponent,
    ProfilerComponent,
    ProfilerComponentFactory,
    create_profiler_profiler_component_4,
    create_profiler_profiler_component_9,
    create_profiler_profiler_component_14,
    create_profiler_profiler_component_19,
    create_profiler_profiler_component_24,
)

from src.features.monitoring.tracker_components import (
    ITrackerComponent,
    TrackerComponent,
    TrackerComponentFactory,
    create_tracker_tracker_component_2,
    create_tracker_tracker_component_7,
    create_tracker_tracker_component_12,
    create_tracker_tracker_component_17,
    create_tracker_tracker_component_22,
)


class TestAnalyzerComponent:
    """Analyzer组件测试"""

    def test_analyzer_component_initialization(self):
        """测试Analyzer组件初始化"""
        component = AnalyzerComponent(analyzer_id=3)
        assert component.analyzer_id == 3
        assert component.component_type == "Analyzer"
        assert component.component_name == "Analyzer_Component_3"
        assert isinstance(component.creation_time, datetime)

    def test_analyzer_component_get_analyzer_id(self):
        """测试获取analyzer ID"""
        component = AnalyzerComponent(analyzer_id=8)
        assert component.get_analyzer_id() == 8

    def test_analyzer_component_get_info(self):
        """测试获取组件信息"""
        component = AnalyzerComponent(analyzer_id=13)
        info = component.get_info()
        assert info["analyzer_id"] == 13
        assert info["component_name"] == "Analyzer_Component_13"
        assert info["component_type"] == "Analyzer"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
        assert info["type"] == "unified_features_monitoring_component"

    def test_analyzer_component_process_success(self):
        """测试处理数据成功"""
        component = AnalyzerComponent(analyzer_id=18)
        data = {"key": "value", "number": 123}
        result = component.process(data)
        assert result["analyzer_id"] == 18
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert result["processing_type"] == "unified_analyzer_processing"

    def test_analyzer_component_process_with_exception(self):
        """测试处理数据时异常处理"""
        component = AnalyzerComponent(analyzer_id=23)
        data = {"key": "value"}
        # 模拟datetime.now()在try块中抛出异常
        mock_datetime_obj = MagicMock()
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("模拟异常")
            else:
                return datetime.now()
        mock_datetime_obj.now.side_effect = side_effect
        mock_datetime_obj.now.return_value.isoformat = lambda: datetime.now().isoformat()
        with patch('src.features.monitoring.analyzer_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["analyzer_id"] == 23
            assert result["status"] == "error"
            assert "error" in result
            assert "error_type" in result

    def test_analyzer_component_get_status(self):
        """测试获取组件状态"""
        component = AnalyzerComponent(analyzer_id=3)
        status = component.get_status()
        assert status["analyzer_id"] == 3
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status

    def test_analyzer_component_factory_create_component(self):
        """测试工厂创建组件"""
        component = AnalyzerComponentFactory.create_component(3)
        assert isinstance(component, AnalyzerComponent)
        assert component.analyzer_id == 3

    def test_analyzer_component_factory_create_invalid_id(self):
        """测试工厂创建无效ID组件"""
        with pytest.raises(ValueError, match="不支持的analyzer ID"):
            AnalyzerComponentFactory.create_component(99)

    def test_analyzer_component_factory_get_available_analyzers(self):
        """测试获取所有可用的analyzer ID"""
        available = AnalyzerComponentFactory.get_available_analyzers()
        assert isinstance(available, list)
        assert len(available) == 5
        assert 3 in available
        assert 8 in available
        assert 13 in available
        assert 18 in available
        assert 23 in available

    def test_analyzer_component_factory_create_all_analyzers(self):
        """测试创建所有可用analyzer"""
        all_analyzers = AnalyzerComponentFactory.create_all_analyzers()
        assert isinstance(all_analyzers, dict)
        assert len(all_analyzers) == 5
        for analyzer_id, component in all_analyzers.items():
            assert isinstance(component, AnalyzerComponent)
            assert component.analyzer_id == analyzer_id

    def test_analyzer_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = AnalyzerComponentFactory.get_factory_info()
        assert info["factory_name"] == "AnalyzerComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_analyzers"] == 5
        assert len(info["supported_ids"]) == 5

    def test_analyzer_component_backward_compatibility_functions(self):
        """测试向后兼容函数"""
        comp3 = create_analyzer_analyzer_component_3()
        assert comp3.analyzer_id == 3

        comp8 = create_analyzer_analyzer_component_8()
        assert comp8.analyzer_id == 8

        comp13 = create_analyzer_analyzer_component_13()
        assert comp13.analyzer_id == 13

        comp18 = create_analyzer_analyzer_component_18()
        assert comp18.analyzer_id == 18

        comp23 = create_analyzer_analyzer_component_23()
        assert comp23.analyzer_id == 23

    def test_analyzer_component_implements_interface(self):
        """测试AnalyzerComponent实现接口"""
        component = AnalyzerComponent(analyzer_id=3)
        assert isinstance(component, IAnalyzerComponent)
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_analyzer_id')


class TestProfilerComponent:
    """Profiler组件测试"""

    def test_profiler_component_initialization(self):
        """测试Profiler组件初始化"""
        component = ProfilerComponent(profiler_id=4)
        assert component.profiler_id == 4
        assert component.component_type == "Profiler"
        assert component.component_name == "Profiler_Component_4"
        assert isinstance(component.creation_time, datetime)

    def test_profiler_component_get_profiler_id(self):
        """测试获取profiler ID"""
        component = ProfilerComponent(profiler_id=9)
        assert component.get_profiler_id() == 9

    def test_profiler_component_get_info(self):
        """测试获取组件信息"""
        component = ProfilerComponent(profiler_id=14)
        info = component.get_info()
        assert info["profiler_id"] == 14
        assert info["component_name"] == "Profiler_Component_14"
        assert info["component_type"] == "Profiler"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"

    def test_profiler_component_process_success(self):
        """测试处理数据成功"""
        component = ProfilerComponent(profiler_id=19)
        data = {"key": "value", "number": 456}
        result = component.process(data)
        assert result["profiler_id"] == 19
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert result["processing_type"] == "unified_profiler_processing"

    def test_profiler_component_process_with_exception(self):
        """测试处理数据时异常处理"""
        component = ProfilerComponent(profiler_id=24)
        data = {"key": "value"}
        mock_datetime_obj = MagicMock()
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("模拟异常")
            else:
                return datetime.now()
        mock_datetime_obj.now.side_effect = side_effect
        mock_datetime_obj.now.return_value.isoformat = lambda: datetime.now().isoformat()
        with patch('src.features.monitoring.profiler_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["profiler_id"] == 24
            assert result["status"] == "error"
            assert "error" in result

    def test_profiler_component_get_status(self):
        """测试获取组件状态"""
        component = ProfilerComponent(profiler_id=4)
        status = component.get_status()
        assert status["profiler_id"] == 4
        assert status["status"] == "active"
        assert status["health"] == "good"

    def test_profiler_component_factory_create_component(self):
        """测试工厂创建组件"""
        component = ProfilerComponentFactory.create_component(4)
        assert isinstance(component, ProfilerComponent)
        assert component.profiler_id == 4

    def test_profiler_component_factory_create_invalid_id(self):
        """测试工厂创建无效ID组件"""
        with pytest.raises(ValueError, match="不支持的profiler ID"):
            ProfilerComponentFactory.create_component(99)

    def test_profiler_component_factory_get_available_profilers(self):
        """测试获取所有可用的profiler ID"""
        available = ProfilerComponentFactory.get_available_profilers()
        assert isinstance(available, list)
        assert len(available) == 5
        assert 4 in available
        assert 9 in available
        assert 14 in available
        assert 19 in available
        assert 24 in available

    def test_profiler_component_factory_create_all_profilers(self):
        """测试创建所有可用profiler"""
        all_profilers = ProfilerComponentFactory.create_all_profilers()
        assert isinstance(all_profilers, dict)
        assert len(all_profilers) == 5
        for profiler_id, component in all_profilers.items():
            assert isinstance(component, ProfilerComponent)
            assert component.profiler_id == profiler_id

    def test_profiler_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = ProfilerComponentFactory.get_factory_info()
        assert info["factory_name"] == "ProfilerComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_profilers"] == 5
        assert len(info["supported_ids"]) == 5

    def test_profiler_component_backward_compatibility_functions(self):
        """测试向后兼容函数"""
        comp4 = create_profiler_profiler_component_4()
        assert comp4.profiler_id == 4

        comp9 = create_profiler_profiler_component_9()
        assert comp9.profiler_id == 9

        comp14 = create_profiler_profiler_component_14()
        assert comp14.profiler_id == 14

        comp19 = create_profiler_profiler_component_19()
        assert comp19.profiler_id == 19

        comp24 = create_profiler_profiler_component_24()
        assert comp24.profiler_id == 24

    def test_profiler_component_implements_interface(self):
        """测试ProfilerComponent实现接口"""
        component = ProfilerComponent(profiler_id=4)
        assert isinstance(component, IProfilerComponent)
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_profiler_id')


class TestTrackerComponent:
    """Tracker组件测试"""

    def test_tracker_component_initialization(self):
        """测试Tracker组件初始化"""
        component = TrackerComponent(tracker_id=2)
        assert component.tracker_id == 2
        assert component.component_type == "Tracker"
        assert component.component_name == "Tracker_Component_2"
        assert isinstance(component.creation_time, datetime)

    def test_tracker_component_get_tracker_id(self):
        """测试获取tracker ID"""
        component = TrackerComponent(tracker_id=7)
        assert component.get_tracker_id() == 7

    def test_tracker_component_get_info(self):
        """测试获取组件信息"""
        component = TrackerComponent(tracker_id=12)
        info = component.get_info()
        assert info["tracker_id"] == 12
        assert info["component_name"] == "Tracker_Component_12"
        assert info["component_type"] == "Tracker"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"

    def test_tracker_component_process_success(self):
        """测试处理数据成功"""
        component = TrackerComponent(tracker_id=17)
        data = {"key": "value", "number": 789}
        result = component.process(data)
        assert result["tracker_id"] == 17
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert result["processing_type"] == "unified_tracker_processing"

    def test_tracker_component_process_with_exception(self):
        """测试处理数据时异常处理"""
        component = TrackerComponent(tracker_id=22)
        data = {"key": "value"}
        mock_datetime_obj = MagicMock()
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("模拟异常")
            else:
                return datetime.now()
        mock_datetime_obj.now.side_effect = side_effect
        mock_datetime_obj.now.return_value.isoformat = lambda: datetime.now().isoformat()
        with patch('src.features.monitoring.tracker_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["tracker_id"] == 22
            assert result["status"] == "error"
            assert "error" in result

    def test_tracker_component_get_status(self):
        """测试获取组件状态"""
        component = TrackerComponent(tracker_id=2)
        status = component.get_status()
        assert status["tracker_id"] == 2
        assert status["status"] == "active"
        assert status["health"] == "good"

    def test_tracker_component_factory_create_component(self):
        """测试工厂创建组件"""
        component = TrackerComponentFactory.create_component(2)
        assert isinstance(component, TrackerComponent)
        assert component.tracker_id == 2

    def test_tracker_component_factory_create_invalid_id(self):
        """测试工厂创建无效ID组件"""
        with pytest.raises(ValueError, match="不支持的tracker ID"):
            TrackerComponentFactory.create_component(99)

    def test_tracker_component_factory_get_available_trackers(self):
        """测试获取所有可用的tracker ID"""
        available = TrackerComponentFactory.get_available_trackers()
        assert isinstance(available, list)
        assert len(available) == 5
        assert 2 in available
        assert 7 in available
        assert 12 in available
        assert 17 in available
        assert 22 in available

    def test_tracker_component_factory_create_all_trackers(self):
        """测试创建所有可用tracker"""
        all_trackers = TrackerComponentFactory.create_all_trackers()
        assert isinstance(all_trackers, dict)
        assert len(all_trackers) == 5
        for tracker_id, component in all_trackers.items():
            assert isinstance(component, TrackerComponent)
            assert component.tracker_id == tracker_id

    def test_tracker_component_factory_get_factory_info(self):
        """测试获取工厂信息"""
        info = TrackerComponentFactory.get_factory_info()
        assert info["factory_name"] == "TrackerComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_trackers"] == 5
        assert len(info["supported_ids"]) == 5

    def test_tracker_component_backward_compatibility_functions(self):
        """测试向后兼容函数"""
        comp2 = create_tracker_tracker_component_2()
        assert comp2.tracker_id == 2

        comp7 = create_tracker_tracker_component_7()
        assert comp7.tracker_id == 7

        comp12 = create_tracker_tracker_component_12()
        assert comp12.tracker_id == 12

        comp17 = create_tracker_tracker_component_17()
        assert comp17.tracker_id == 17

        comp22 = create_tracker_tracker_component_22()
        assert comp22.tracker_id == 22

    def test_tracker_component_implements_interface(self):
        """测试TrackerComponent实现接口"""
        component = TrackerComponent(tracker_id=2)
        assert isinstance(component, ITrackerComponent)
        assert hasattr(component, 'get_info')
        assert hasattr(component, 'process')
        assert hasattr(component, 'get_status')
        assert hasattr(component, 'get_tracker_id')

