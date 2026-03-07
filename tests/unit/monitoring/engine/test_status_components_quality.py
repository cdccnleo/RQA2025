#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / 'src')

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    engine_status_components_module = importlib.import_module('src.monitoring.engine.status_components')
    StatusComponent = getattr(engine_status_components_module, 'StatusComponent', None)
    StatusComponentFactory = getattr(engine_status_components_module, 'StatusComponentFactory', None)
    IStatusComponent = getattr(engine_status_components_module, 'IStatusComponent', None)

    if StatusComponent is None:
        pytest.skip('监控模块导入失败', allow_module_level=True)
except ImportError:
    pytest.skip('监控模块导入失败', allow_module_level=True)


@pytest.fixture
def status_component():
    '''创建状态组件实例'''
    return StatusComponent(status_id=1, component_type='Status')


class TestStatusComponent:
    '''StatusComponent测试类'''

    def test_initialization(self, status_component):
        '''测试初始化'''
        assert status_component.status_id == 1
        assert status_component.component_type == 'Status'
        assert 'Status_Component_1' in status_component.component_name
        assert isinstance(status_component.creation_time, datetime)

    def test_get_status_id(self, status_component):
        '''测试获取status ID'''
        assert status_component.get_status_id() == 1

    def test_get_info(self, status_component):
        '''测试获取组件信息'''
        info = status_component.get_info()
        assert isinstance(info, dict)
        assert 'status_id' in info
        assert 'component_type' in info
        assert 'creation_time' in info

    def test_component_name_generation(self, status_component):
        '''测试组件名称生成'''
        assert status_component.component_name.startswith('Status_Component_')

    def test_creation_time_type(self, status_component):
        '''测试创建时间类型'''
        assert isinstance(status_component.creation_time, datetime)


class TestStatusComponentFactory:
    '''StatusComponentFactory测试类'''

    def test_create_component_valid_id(self):
        '''测试创建有效ID的组件'''
        component = StatusComponentFactory.create_component(1)
        assert isinstance(component, StatusComponent)
        assert component.get_status_id() == 1

    def test_create_component_invalid_id(self):
        '''测试创建无效ID的组件'''
        with pytest.raises(ValueError):
            StatusComponentFactory.create_component(999)

    def test_get_available_status_ids(self):
        '''测试获取可用status ID'''
        available_ids = StatusComponentFactory.get_available_status_ids()
        assert isinstance(available_ids, list)
        assert len(available_ids) > 0
        assert 1 in available_ids

    def test_factory_creates_correct_type(self):
        '''测试工厂创建正确类型'''
        component = StatusComponentFactory.create_component(2)
        assert isinstance(component, StatusComponent)
        assert component.get_status_id() == 2
