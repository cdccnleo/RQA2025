"""
统一导入策略示例测试文件

展示如何使用TestImportManager来统一管理模块导入，
避免每个测试文件重复实现导入逻辑。
"""

import pytest
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_import_manager import setup_test_environment, get_import_manager

# 在测试文件开头统一设置测试环境
setup_test_environment()

# 获取导入管理器（如果需要进一步配置）
import_manager = get_import_manager()


class TestUnifiedImportExample:
    """统一导入策略示例测试"""

    def test_data_manager_mock(self):
        """测试数据管理器Mock"""
        # 数据管理器应该已经被Mock
        import sys
        assert "src.data.data_manager" in sys.modules

        data_manager = sys.modules["src.data.data_manager"]
        assert hasattr(data_manager, 'DataManager')
        assert hasattr(data_manager, 'DataLoaderError')

        # 测试实例方法
        instance = data_manager.instance
        assert instance.validate_all_configs() is True
        assert instance.health_check() == {"status": "healthy"}

    def test_adapter_module_mock(self):
        """测试适配器模块Mock"""
        import sys
        assert "src.data.adapters" in sys.modules

        adapters = sys.modules["src.data.adapters"]
        assert hasattr(adapters, 'BaseAdapter')
        assert hasattr(adapters, 'DataAdapter')
        assert hasattr(adapters, 'AdapterFactory')

    def test_adapter_components_mock(self):
        """测试适配器组件Mock"""
        import sys
        assert "src.data.adapters.adapter_components" in sys.modules

        components = sys.modules["src.data.adapters.adapter_components"]
        assert hasattr(components, 'AdapterComponent')

        # 测试组件实例
        adapter_component = components.AdapterComponent()
        assert adapter_component.get_adapter_id() == 1
        assert adapter_component.get_status() == {
            "status": "active",
            "component_name": "Unit_Component_1"
        }

    def test_import_manager_functionality(self):
        """测试导入管理器功能"""
        # 测试安全导入
        result = import_manager.safe_import("sys")
        assert result is not None

        # 测试不存在模块的安全导入
        result = import_manager.safe_import("non_existent_module", fallback="default")
        assert result == "default"

    def test_path_setup(self):
        """测试路径设置"""
        import sys
        from pathlib import Path

        project_root = import_manager.project_root
        src_path = import_manager.src_path

        assert src_path.exists()
        assert str(src_path) in sys.path
        assert str(project_root) in sys.path


if __name__ == "__main__":
    # 直接运行测试
    test_instance = TestUnifiedImportExample()

    print("Running unified import example tests...")

    try:
        test_instance.test_data_manager_mock()
        print("✓ Data manager mock test passed")

        test_instance.test_adapter_module_mock()
        print("✓ Adapter module mock test passed")

        test_instance.test_adapter_components_mock()
        print("✓ Adapter components mock test passed")

        test_instance.test_import_manager_functionality()
        print("✓ Import manager functionality test passed")

        test_instance.test_path_setup()
        print("✓ Path setup test passed")

        print("\n🎉 All tests passed! Unified import strategy is working.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
