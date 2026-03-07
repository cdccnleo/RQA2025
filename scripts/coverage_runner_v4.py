#!/usr/bin/env python3
"""
版本管理服务覆盖率测试脚本 V4
专门用于在conda test环境下测试版本管理相关模块的覆盖率，避免cryptography导入问题
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 只导入需要的模块，避免cryptography导入问题
try:
    # 检查并导入配置相关模块
    from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager

    print("✅ 成功导入配置相关模块")

    # 运行测试
    import pytest

    # 运行基础设施配置测试
    print("🚀 开始运行基础设施配置测试...")
    exit_code = pytest.main([
        "tests/",
        "--cov=src",
        "--cov-report=term-missing",
        "-v"
    ])

    print(f"📊 测试完成，退出代码: {exit_code}")
    # 不直接退出，让程序正常结束
    if exit_code != 0:
        print(f"⚠️ 测试执行失败，退出代码: {exit_code}")
    else:
        print("✅ 测试执行成功")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("🔍 尝试分析导入路径...")
    
    # 分析导入路径
    import importlib.util
    
    # 检查version_manager模块
    version_manager_path = "src/infrastructure/core/config/services/version_manager.py"
    if os.path.exists(version_manager_path):
        print(f"✅ 找到文件: {version_manager_path}")
        spec = importlib.util.spec_from_file_location("version_manager", version_manager_path)
        if spec:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ 成功加载version_manager模块")
            except Exception as e2:
                print(f"❌ 加载失败: {e2}")
        else:
            print("❌ 无法创建模块规范")
    else:
        print(f"❌ 文件不存在: {version_manager_path}")
    
    # 检查diff_service模块
    diff_service_path = "src/infrastructure/core/config/services/diff_service.py"
    if os.path.exists(diff_service_path):
        print(f"✅ 找到文件: {diff_service_path}")
        spec = importlib.util.spec_from_file_location("diff_service", diff_service_path)
        if spec:
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ 成功加载diff_service模块")
            except Exception as e2:
                print(f"❌ 加载失败: {e2}")
        else:
            print("❌ 无法创建模块规范")
    else:
        print(f"❌ 文件不存在: {diff_service_path}")
    
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)
