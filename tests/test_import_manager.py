#!/usr/bin/env python3
"""
测试导入管理器 - 统一管理测试文件的模块导入

提供统一的路径设置和模块导入逻辑，避免各测试文件重复实现。
支持模块导入失败时的降级处理和错误报告。
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from unittest.mock import Mock


class TestImportManager:
    """
    测试导入管理器

    负责统一管理测试文件的模块导入，包括：
    - 路径设置
    - 模块导入
    - Mock对象管理
    - 错误处理
    """

    def __init__(self):
        self.project_root = self._find_project_root()
        self.src_path = self.project_root / "src"
        self._setup_paths()
        self._mock_modules: Dict[str, Any] = {}

    def _find_project_root(self) -> Path:
        """查找项目根目录"""
        current = Path(__file__).resolve()

        # 从tests目录向上查找
        for parent in current.parents:
            if (parent / "src").exists() and (parent / "tests").exists():
                return parent

        # 如果没找到，返回当前目录的父目录
        return current.parent.parent

    def _setup_paths(self):
        """设置Python路径"""
        paths_to_add = [
            str(self.src_path),
            str(self.project_root)
        ]

        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

    def create_mock_data_manager(self) -> Mock:
        """创建Mock数据管理器"""
        mock_data_manager = Mock()
        mock_data_manager.DataManager = Mock()
        mock_data_manager.DataLoaderError = Exception

        # 配置实例方法
        mock_instance = Mock()
        mock_instance.validate_all_configs.return_value = True
        mock_instance.health_check.return_value = {"status": "healthy"}
        mock_instance.store_data.return_value = True
        mock_instance.has_data.return_value = True

        mock_data_manager.instance = mock_instance
        return mock_data_manager

    def mock_module(self, module_name: str, **kwargs) -> Optional[Mock]:
        """
        Mock指定的模块

        Args:
            module_name: 模块名称
            **kwargs: Mock属性

        Returns:
            Mock对象或None
        """
        if module_name in sys.modules:
            return None  # 模块已存在，不需要Mock

        mock_module = Mock()
        for attr_name, attr_value in kwargs.items():
            setattr(mock_module, attr_name, attr_value)

        sys.modules[module_name] = mock_module
        self._mock_modules[module_name] = mock_module

        return mock_module

    def mock_data_module(self, module_name: str) -> Optional[Mock]:
        """Mock数据层模块"""
        return self.mock_module(
            module_name,
            DataModel=Mock(),
            DEFAULT_FREQUENCY="1d",
            _UNSET=object(),
            EnhancedDataModel=Mock()
        )

    def mock_adapter_module(self, module_name: str) -> Optional[Mock]:
        """Mock适配器模块"""
        return self.mock_module(
            module_name,
            BaseAdapter=Mock(),
            DataAdapter=Mock(),
            AdapterFactory=Mock()
        )

    def mock_loader_module(self, module_name: str) -> Optional[Mock]:
        """Mock加载器模块"""
        return self.mock_module(
            module_name,
            BaseDataLoader=Mock(),
            LoaderConfig=Mock(),
            DataLoader=Mock(),
            MockDataLoader=Mock(),
            FileDataLoader=Mock(),
            DatabaseDataLoader=Mock(),
            APIDataLoader=Mock(),
            create_data_loader=Mock(return_value=Mock()),
            get_data_loader=Mock(return_value=Mock())
        )

    def safe_import(self, module_name: str, fallback: Any = None) -> Any:
        """
        安全导入模块

        Args:
            module_name: 模块名称
            fallback: 导入失败时的默认值

        Returns:
            导入的模块或fallback
        """
        try:
            __import__(module_name)
            return sys.modules[module_name]
        except ImportError:
            return fallback

    def get_import_errors(self) -> List[str]:
        """获取导入错误列表"""
        errors = []
        for module_name in self._mock_modules:
            try:
                # 尝试重新导入，看是否还有问题
                if module_name in sys.modules:
                    del sys.modules[module_name]
                __import__(module_name)
            except ImportError as e:
                errors.append(f"{module_name}: {e}")
        return errors

    def setup_data_layer_mocks(self):
        """设置数据层相关Mock"""
        # Mock数据管理器
        mock_data_manager = self.create_mock_data_manager()
        sys.modules["src.data.data_manager"] = mock_data_manager

        # Mock数据模型
        self.mock_data_module("src.data.data_model")

        # Mock基础加载器
        self.mock_loader_module("src.data.base_loader")

        # Mock数据加载器
        self.mock_loader_module("src.data.data_loader")

    def setup_adapter_layer_mocks(self):
        """设置适配器层相关Mock"""
        # Mock基础适配器
        self.mock_adapter_module("src.data.adapters")

        # Mock适配器组件
        if "src.data.adapters.adapter_components" not in sys.modules:
            mock_module = Mock()
            # Mock AdapterComponent
            mock_adapter_component_class = Mock()
            mock_adapter_component_instance = Mock()
            mock_adapter_component_instance.get_info.return_value = {
                "adapter_id": 1,
                "component_type": "Unit",
                "description": "统一Unit组件实现"
            }
            mock_adapter_component_instance.get_status.return_value = {
                "status": "active",
                "component_name": "Unit_Component_1"
            }
            mock_adapter_component_instance.process.return_value = {
                "status": "success",
                "input_data": {"payload": 1}
            }
            mock_adapter_component_instance.get_adapter_id.return_value = 1
            mock_adapter_component_class.side_effect = lambda *args, **kwargs: mock_adapter_component_instance
            mock_module.AdapterComponent = mock_adapter_component_class

            sys.modules["src.data.adapters.adapter_components"] = mock_module

    def cleanup(self):
        """清理Mock模块"""
        for module_name in self._mock_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        self._mock_modules.clear()


# 全局导入管理器实例
import_manager = TestImportManager()


def setup_test_environment():
    """
    设置测试环境

    这是一个便捷函数，可以在测试文件的开头调用
    """
    import_manager.setup_data_layer_mocks()
    import_manager.setup_adapter_layer_mocks()


def get_import_manager() -> TestImportManager:
    """获取导入管理器实例"""
    return import_manager


if __name__ == "__main__":
    # 测试导入管理器
    print("Testing Import Manager...")

    # 设置测试环境
    setup_test_environment()

    # 检查导入错误
    errors = import_manager.get_import_errors()
    if errors:
        print(f"Import errors: {len(errors)}")
        for error in errors[:5]:
            print(f"  {error}")
    else:
        print("No import errors detected")

    print("Import Manager test completed")
