#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceleration组件测试覆盖
测试acceleration模块的组件工厂模式
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

from src.features.acceleration.accelerator_components import (
    ComponentFactory,
    IAcceleratorComponent,
    AcceleratorComponent,
    AcceleratorComponentFactory,
    create_accelerator_accelerator_component_2,
    create_accelerator_accelerator_component_7,
    create_accelerator_accelerator_component_12,
    create_accelerator_accelerator_component_17,
    create_accelerator_accelerator_component_22,
    create_accelerator_accelerator_component_27
)

from src.features.acceleration.distributed_components import (
    DistributedComponent,
    DistributedComponentFactory,
    IDistributedComponent
)

from src.features.acceleration.gpu_components import (
    GpuComponent,
    GpuComponentFactory,
    IGpuComponent
)

from src.features.acceleration.parallel_components import (
    ParallelComponent,
    ParallelComponentFactory,
    IParallelComponent
)


class TestComponentFactory:
    """ComponentFactory测试"""

    def test_component_factory_initialization(self):
        """测试组件工厂初始化"""
        factory = ComponentFactory()
        assert factory._components == {}

    def test_create_component_returns_none(self):
        """测试创建组件（默认返回None）"""
        factory = ComponentFactory()
        component = factory.create_component("test_type", {})
        assert component is None

    def test_create_component_handles_exception(self):
        """测试创建组件时异常处理"""
        factory = ComponentFactory()
        # _create_component_instance返回None，不会触发异常
        component = factory.create_component("test_type", {})
        assert component is None


class TestAcceleratorComponent:
    """AcceleratorComponent测试"""

    def test_accelerator_component_initialization(self):
        """测试Accelerator组件初始化"""
        component = AcceleratorComponent(accelerator_id=2)
        assert component.accelerator_id == 2
        assert component.component_type == "Accelerator"
        assert "Accelerator_Component_2" in component.component_name
        assert isinstance(component.creation_time, datetime)

    def test_accelerator_component_custom_type(self):
        """测试自定义组件类型"""
        component = AcceleratorComponent(accelerator_id=7, component_type="Custom")
        assert component.component_type == "Custom"
        assert "Custom_Component_7" in component.component_name

    def test_get_accelerator_id(self):
        """测试获取accelerator ID"""
        component = AcceleratorComponent(accelerator_id=12)
        assert component.get_accelerator_id() == 12

    def test_get_info(self):
        """测试获取组件信息"""
        component = AcceleratorComponent(accelerator_id=17)
        info = component.get_info()
        assert info["accelerator_id"] == 17
        assert "component_name" in info
        assert "component_type" in info
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
        assert info["type"] == "unified_features_acceleration_component"

    def test_process_success(self):
        """测试处理数据（成功）"""
        component = AcceleratorComponent(accelerator_id=22)
        data = {"key": "value"}
        result = component.process(data)
        assert result["accelerator_id"] == 22
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
        assert "result" in result

    def test_process_with_exception(self):
        """测试处理数据时异常处理"""
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        component = AcceleratorComponent(accelerator_id=27)
        data = {"key": "value"}
        # 模拟datetime.now()在try块中抛出异常
        mock_datetime_obj = MagicMock()
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:  # 第一次调用（try块中）
                raise Exception("模拟异常")
            else:  # 第二次调用（except块中）
                return datetime.now()
        mock_datetime_obj.now.side_effect = side_effect
        mock_datetime_obj.now.return_value.isoformat = lambda: datetime.now().isoformat()
        with patch('src.features.acceleration.accelerator_components.datetime', mock_datetime_obj):
            result = component.process(data)
            assert result["accelerator_id"] == 27
            assert result["status"] == "error"
            assert "error" in result
            assert "error_type" in result

    def test_get_status(self):
        """测试获取组件状态"""
        component = AcceleratorComponent(accelerator_id=2)
        status = component.get_status()
        assert status["accelerator_id"] == 2
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status


class TestAcceleratorComponentFactory:
    """AcceleratorComponentFactory测试"""

    def test_supported_accelerator_ids(self):
        """测试支持的accelerator ID列表"""
        assert AcceleratorComponentFactory.SUPPORTED_ACCELERATOR_IDS == [2, 7, 12, 17, 22, 27]

    def test_create_component_valid_id(self):
        """测试创建组件（有效ID）"""
        component = AcceleratorComponentFactory.create_component(2)
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 2

    def test_create_component_invalid_id(self):
        """测试创建组件（无效ID）"""
        with pytest.raises(ValueError, match="不支持的accelerator ID"):
            AcceleratorComponentFactory.create_component(99)

    def test_get_available_accelerators(self):
        """测试获取所有可用的accelerator ID"""
        ids = AcceleratorComponentFactory.get_available_accelerators()
        assert ids == [2, 7, 12, 17, 22, 27]

    def test_create_all_accelerators(self):
        """测试创建所有accelerator"""
        accelerators = AcceleratorComponentFactory.create_all_accelerators()
        assert len(accelerators) == 6
        for accelerator_id in [2, 7, 12, 17, 22, 27]:
            assert accelerator_id in accelerators
            assert isinstance(accelerators[accelerator_id], AcceleratorComponent)

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = AcceleratorComponentFactory.get_factory_info()
        assert info["factory_name"] == "AcceleratorComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_accelerators"] == 6
        assert info["supported_ids"] == [2, 7, 12, 17, 22, 27]
        assert "created_at" in info


class TestAcceleratorBackwardCompatibility:
    """向后兼容函数测试"""

    def test_create_accelerator_component_2(self):
        """测试创建accelerator组件2"""
        component = create_accelerator_accelerator_component_2()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 2

    def test_create_accelerator_component_7(self):
        """测试创建accelerator组件7"""
        component = create_accelerator_accelerator_component_7()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 7

    def test_create_accelerator_component_12(self):
        """测试创建accelerator组件12"""
        component = create_accelerator_accelerator_component_12()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 12

    def test_create_accelerator_component_17(self):
        """测试创建accelerator组件17"""
        component = create_accelerator_accelerator_component_17()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 17

    def test_create_accelerator_component_22(self):
        """测试创建accelerator组件22"""
        component = create_accelerator_accelerator_component_22()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 22

    def test_create_accelerator_component_27(self):
        """测试创建accelerator组件27"""
        component = create_accelerator_accelerator_component_27()
        assert isinstance(component, AcceleratorComponent)
        assert component.accelerator_id == 27


class TestDistributedComponent:
    """DistributedComponent测试"""

    def test_distributed_component_initialization(self):
        """测试Distributed组件初始化"""
        component = DistributedComponent(distributed_id=3)
        assert component.distributed_id == 3
        assert component.component_type == "Distributed"

    def test_get_distributed_id(self):
        """测试获取distributed ID"""
        component = DistributedComponent(distributed_id=5)
        assert component.get_distributed_id() == 5

    def test_distributed_component_process(self):
        """测试Distributed组件处理数据"""
        component = DistributedComponent(distributed_id=8)
        data = {"test": "data"}
        result = component.process(data)
        assert result["distributed_id"] == 8
        assert result["status"] == "success"


class TestGpuComponent:
    """GpuComponent测试"""

    def test_gpu_component_initialization(self):
        """测试Gpu组件初始化"""
        component = GpuComponent(gpu_id=1)
        assert component.gpu_id == 1
        assert component.component_type == "Gpu"

    def test_get_gpu_id(self):
        """测试获取gpu ID"""
        component = GpuComponent(gpu_id=2)
        assert component.get_gpu_id() == 2

    def test_gpu_component_process(self):
        """测试Gpu组件处理数据"""
        component = GpuComponent(gpu_id=1)
        data = {"test": "data"}
        result = component.process(data)
        assert result["gpu_id"] == 1
        assert result["status"] == "success"


class TestParallelComponent:
    """ParallelComponent测试"""

    def test_parallel_component_initialization(self):
        """测试Parallel组件初始化"""
        component = ParallelComponent(parallel_id=4)
        assert component.parallel_id == 4
        assert component.component_type == "Parallel"

    def test_get_parallel_id(self):
        """测试获取parallel ID"""
        component = ParallelComponent(parallel_id=6)
        assert component.get_parallel_id() == 6

    def test_parallel_component_process(self):
        """测试Parallel组件处理数据"""
        component = ParallelComponent(parallel_id=4)
        data = {"test": "data"}
        result = component.process(data)
        assert result["parallel_id"] == 4
        assert result["status"] == "success"


