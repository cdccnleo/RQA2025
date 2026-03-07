#!/usr/bin/env python3
import sys
import os
import pytest

def test_paths():
    """测试路径设置"""
    with open('debug_paths.log', 'w') as f:
        f.write("Current sys.path:\n")
        for i, path in enumerate(sys.path):
            f.write(f"{i}: {path}\n")

        f.write(f"\nCurrent working directory: {os.getcwd()}\n")

        # 尝试导入
        f.write("\nTrying imports...\n")
        try:
            import risk
            f.write("✅ import risk - SUCCESS\n")
        except ImportError as e:
            f.write(f"❌ import risk - FAILED: {e}\n")

        try:
            import src
            f.write("✅ import src - SUCCESS\n")
        except ImportError as e:
            f.write(f"❌ import src - FAILED: {e}\n")

        try:
            import src.risk
            f.write("✅ import src.risk - SUCCESS\n")
        except ImportError as e:
            f.write(f"❌ import src.risk - FAILED: {e}\n")

        try:
            from risk.models.risk_manager import RiskManager
            f.write("✅ from risk.models.risk_manager import RiskManager - SUCCESS\n")
        except ImportError as e:
            f.write(f"❌ from risk.models.risk_manager import RiskManager - FAILED: {e}\n")

    # 总是通过
    assert True
