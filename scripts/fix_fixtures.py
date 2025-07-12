import os
import re
import sys
from pathlib import Path

def get_project_root():
    """获取项目根目录"""
    current = Path(__file__).parent
    while not (current / 'pyproject.toml').exists():
        if current.parent == current:
            raise FileNotFoundError("无法定位项目根目录")
        current = current.parent
    return current

def migrate_duplicate_fixtures(project_root):
    """迁移重复夹具到conftest.py"""
    try:
        data_conftest = project_root / "tests/unit/data/conftest.py"
        if not data_conftest.exists():
            data_conftest.parent.mkdir(parents=True, exist_ok=True)
            data_conftest.write_text('''import pytest
from src.data.adapters import DataSourceMock

@pytest.fixture
def mock_data_source():
    """标准化的模拟数据源夹具"""
    return DataSourceMock(autospec=True)
''')
            print(f"✅ 创建 {data_conftest}")

        # 更新引用文件
        for file in (project_root / "tests/unit/data").rglob("test_*.py"):
            content = file.read_text(encoding='utf-8')
            if "@pytest.fixture" not in content and "mock_data_source" in content:
                print(f"🔍 更新 {file}")
                file.write_text(content, encoding='utf-8')
    except Exception as e:
        print(f"❌ 迁移夹具失败: {str(e)}")
        raise

def main():
    print("="*50)
    print(" RQA2025 测试夹具规范修复工具")
    print("="*50)

    try:
        project_root = get_project_root()
        print(f"项目根目录: {project_root}")

        print("\n▶ 开始迁移重复夹具...")
        migrate_duplicate_fixtures(project_root)

        print("\n✅ 所有修复已完成")
        return 0
    except Exception as e:
        print(f"\n❌ 修复过程中发生错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
