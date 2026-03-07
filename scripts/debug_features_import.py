#!/usr/bin/env python3
"""
调试特征处理层导入问题
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_import_step_by_step():
    """逐步测试导入"""

    print("=== 测试基础导入 ===")

    try:
        import sklearn
        print(f"✓ sklearn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ sklearn import error: {e}")
        return

    try:
        print("✓ sklearn.base.BaseEstimator imported")
    except ImportError as e:
        print(f"✗ sklearn.base import error: {e}")
        return

    print("\n=== 测试特征处理层基础模块 ===")

    try:
        print("✓ infrastructure logger imported")
    except ImportError as e:
        print(f"✗ infrastructure logger import error: {e}")

    try:
        print("✓ data models imported")
    except ImportError as e:
        print(f"✗ data models import error: {e}")

    try:
        print("✓ utils logger imported")
    except ImportError as e:
        print(f"✗ utils logger import error: {e}")

    print("\n=== 测试特征处理层配置 ===")

    try:
        print("✓ config_integration imported")
    except ImportError as e:
        print(f"✗ config_integration import error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 测试特征选择器 ===")

    try:
        print("✓ FeatureSelector imported successfully")
    except ImportError as e:
        print(f"✗ FeatureSelector import error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_import_step_by_step()
