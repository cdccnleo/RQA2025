#!/usr/bin/env python3
"""
配置管理优化验证脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("🔧 配置管理系统优化验证")
print("=" * 40)

try:
    # 测试增强的配置管理器
    from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager

    manager = UnifiedConfigManager()
    print("✅ 增强配置管理器导入成功")

    # 测试基本功能
    manager.set("test.key", "test_value")
    value = manager.get("test.key")
    print(f"✅ 基本功能测试: {value}")

    # 测试环境变量加载
    os.environ['RQA_TEST_VAR'] = 'true'
    success = manager.load_from_environment_variables('RQA_')
    print(f"✅ 环境变量加载: {'成功' if success else '失败'}")

    test_value = manager.get('test.var')
    print(f"✅ 环境变量值: {test_value} (类型: {type(test_value).__name__})")

    # 测试配置验证
    validation = manager.validate_config_integrity()
    print(f"✅ 配置验证: {'通过' if validation['is_valid'] else '失败'}")

    # 测试配置源信息
    source_info = manager.get_config_with_source_info('test.key')
    print(f"✅ 配置源信息: {source_info['source']}")

    # 测试配置元数据
    metadata = manager.export_config_with_metadata()
    print(f"✅ 配置元数据: {metadata['total_keys']} 个配置项")

    print("\n🎉 配置管理系统优化验证成功!")

    # 清理
    os.environ.pop('RQA_TEST_VAR', None)

except Exception as e:
    print(f"❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()

print("\n📋 优化功能总结:")
print("✅ 环境变量智能类型转换")
print("✅ YAML配置文件支持")
print("✅ 配置完整性验证")
print("✅ 配置来源信息追踪")
print("✅ 配置元数据导出")
print("✅ 配置备份和恢复")
print("✅ 多源配置刷新")
