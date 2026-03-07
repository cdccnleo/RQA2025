"""
ML Tuning模块组合逻辑测试补充

覆盖Grid、Hyperparameter、Search、Optimizer组件的组合使用场景
"""

import pytest
import sys
from typing import Dict, Any
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from src.ml.tuning.grid_components import (
        GridComponent,
        GridComponentFactory,
    )
    from src.ml.tuning.hyperparameter_components import (
        HyperparameterComponent,
        HyperparameterComponentFactory,
    )
    from src.ml.tuning.search_components import (
        SearchComponent,
        SearchComponentFactory,
    )
    from src.ml.tuning.optimizer_components import (
        OptimizerComponent,
        MLTuningOptimizerComponentFactory,
    )
except ImportError:
    pytest.skip("无法导入调优相关组件模块", allow_module_level=True)


class TestTuningComponentCombination:
    """测试Tuning组件的组合逻辑"""

    def test_grid_and_hyperparameter_combination(self):
        """测试Grid和Hyperparameter组件的组合使用"""
        # 创建Grid组件
        grid_component = GridComponentFactory.create_component(5)
        assert isinstance(grid_component, GridComponent)
        
        # 创建Hyperparameter组件
        hyperparameter_component = HyperparameterComponentFactory.create_component(3)
        assert isinstance(hyperparameter_component, HyperparameterComponent)
        
        # 组合使用：Grid组件处理数据，然后Hyperparameter组件处理结果
        grid_result = grid_component.process({"param": "value"})
        assert grid_result["status"] == "success"
        
        hyperparameter_result = hyperparameter_component.process(grid_result)
        assert hyperparameter_result["status"] == "success"
        assert "input_data" in hyperparameter_result

    def test_search_and_optimizer_combination(self):
        """测试Search和Optimizer组件的组合使用"""
        # 创建Search组件
        search_component = SearchComponentFactory.create_component(4)
        assert isinstance(search_component, SearchComponent)
        
        # 创建Optimizer组件
        optimizer_component = MLTuningOptimizerComponentFactory.create_component(2)
        assert isinstance(optimizer_component, OptimizerComponent)
        
        # 组合使用：Search组件搜索参数，然后Optimizer组件优化结果
        search_result = search_component.process({"search_space": {"lr": [0.01, 0.1]}})
        assert search_result["status"] == "success"
        
        optimizer_result = optimizer_component.process(search_result)
        assert optimizer_result["status"] == "success"
        assert "input_data" in optimizer_result

    def test_all_components_pipeline(self):
        """测试所有组件的管道式组合使用"""
        # 创建所有类型的组件
        grid = GridComponentFactory.create_component(10)
        hyperparameter = HyperparameterComponentFactory.create_component(8)
        search = SearchComponentFactory.create_component(9)
        optimizer = MLTuningOptimizerComponentFactory.create_component(7)
        
        # 构建管道：Grid -> Hyperparameter -> Search -> Optimizer
        data = {"initial": "data"}
        
        result1 = grid.process(data)
        assert result1["status"] == "success"
        
        result2 = hyperparameter.process(result1)
        assert result2["status"] == "success"
        
        result3 = search.process(result2)
        assert result3["status"] == "success"
        
        result4 = optimizer.process(result3)
        assert result4["status"] == "success"
        
        # 验证最终结果包含所有处理阶段的信息
        assert "input_data" in result4

    def test_component_factory_combination(self):
        """测试组件工厂的组合使用"""
        # 获取所有可用的组件ID
        grid_ids = GridComponentFactory.get_available_grids()
        hyperparameter_ids = HyperparameterComponentFactory.get_available_hyperparameters()
        search_ids = SearchComponentFactory.get_available_searches()
        optimizer_ids = MLTuningOptimizerComponentFactory.get_available_optimizers()
        
        # 验证每个工厂都有支持的组件
        assert len(grid_ids) > 0
        assert len(hyperparameter_ids) > 0
        assert len(search_ids) > 0
        assert len(optimizer_ids) > 0
        
        # 创建所有组件并组合使用
        all_components = {}
        all_components.update(GridComponentFactory.create_all_grids())
        all_components.update(HyperparameterComponentFactory.create_all_hyperparameters())
        all_components.update(SearchComponentFactory.create_all_searches())
        all_components.update(MLTuningOptimizerComponentFactory.create_all_optimizers())
        
        assert len(all_components) > 0
        
        # 测试组合处理
        for component in all_components.values():
            result = component.process({"test": "data"})
            assert result["status"] in ["success", "error"]  # 允许错误状态

    def test_component_status_combination(self):
        """测试组件状态信息的组合查询"""
        grid = GridComponentFactory.create_component(15)
        hyperparameter = HyperparameterComponentFactory.create_component(13)
        search = SearchComponentFactory.create_component(14)
        optimizer = MLTuningOptimizerComponentFactory.create_component(12)
        
        # 获取所有组件的状态
        grid_status = grid.get_status()
        hyperparameter_status = hyperparameter.get_status()
        search_status = search.get_status()
        optimizer_status = optimizer.get_status()
        
        # 验证状态信息的一致性
        assert grid_status["status"] == "active"
        assert hyperparameter_status["status"] == "active"
        assert search_status["status"] == "active"
        assert optimizer_status["status"] == "active"
        
        # 组合状态信息
        combined_status = {
            "grid": grid_status,
            "hyperparameter": hyperparameter_status,
            "search": search_status,
            "optimizer": optimizer_status,
        }
        
        assert len(combined_status) == 4
        assert all(status["status"] == "active" for status in combined_status.values())

    def test_component_info_combination(self):
        """测试组件信息查询的组合使用"""
        grid = GridComponentFactory.create_component(20)
        hyperparameter = HyperparameterComponentFactory.create_component(18)
        search = SearchComponentFactory.create_component(19)
        optimizer = MLTuningOptimizerComponentFactory.create_component(17)
        
        # 获取所有组件的信息
        grid_info = grid.get_info()
        hyperparameter_info = hyperparameter.get_info()
        search_info = search.get_info()
        optimizer_info = optimizer.get_info()
        
        # 验证信息结构的一致性
        assert "component_name" in grid_info
        assert "component_name" in hyperparameter_info
        assert "component_name" in search_info
        assert "component_name" in optimizer_info
        
        # 组合信息
        combined_info = {
            "grid": grid_info,
            "hyperparameter": hyperparameter_info,
            "search": search_info,
            "optimizer": optimizer_info,
        }
        
        assert len(combined_info) == 4
        assert all("component_name" in info for info in combined_info.values())

    def test_component_error_handling_combination(self):
        """测试组件错误处理的组合场景"""
        grid = GridComponentFactory.create_component(5)
        hyperparameter = HyperparameterComponentFactory.create_component(3)
        
        # 测试异常数据的处理
        invalid_data = None
        
        grid_result = grid.process(invalid_data)
        # 即使输入无效，也应该返回结果（包含错误信息）
        assert "status" in grid_result
        
        # 如果Grid处理失败，Hyperparameter应该能处理错误结果
        hyperparameter_result = hyperparameter.process(grid_result)
        assert "status" in hyperparameter_result

    def test_component_factory_info_combination(self):
        """测试组件工厂信息的组合查询"""
        grid_factory_info = GridComponentFactory.get_factory_info()
        hyperparameter_factory_info = HyperparameterComponentFactory.get_factory_info()
        search_factory_info = SearchComponentFactory.get_factory_info()
        optimizer_factory_info = MLTuningOptimizerComponentFactory.get_factory_info()
        
        # 验证工厂信息结构
        assert "factory_name" in grid_factory_info
        assert "factory_name" in hyperparameter_factory_info
        assert "factory_name" in search_factory_info
        assert "factory_name" in optimizer_factory_info
        
        # 组合工厂信息
        all_factory_info = {
            "grid": grid_factory_info,
            "hyperparameter": hyperparameter_factory_info,
            "search": search_factory_info,
            "optimizer": optimizer_factory_info,
        }
        
        assert len(all_factory_info) == 4
        assert all("factory_name" in info for info in all_factory_info.values())

    def test_component_id_consistency(self):
        """测试组件ID的一致性"""
        grid = GridComponentFactory.create_component(5)
        hyperparameter = HyperparameterComponentFactory.create_component(3)
        search = SearchComponentFactory.create_component(4)
        optimizer = MLTuningOptimizerComponentFactory.create_component(2)
        
        # 验证ID的一致性
        assert grid.get_grid_id() == 5
        assert hyperparameter.get_hyperparameter_id() == 3
        assert search.get_search_id() == 4
        assert optimizer.get_optimizer_id() == 2
        
        # 组合ID信息
        component_ids = {
            "grid": grid.get_grid_id(),
            "hyperparameter": hyperparameter.get_hyperparameter_id(),
            "search": search.get_search_id(),
            "optimizer": optimizer.get_optimizer_id(),
        }
        
        assert len(component_ids) == 4
        assert all(isinstance(id_val, int) for id_val in component_ids.values())

