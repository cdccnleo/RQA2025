#!/usr/bin/env python3
"""
配置管理优化演示

展示增强后的配置管理功能
"""

from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def demo_enhanced_config_manager():
    """演示增强的配置管理器功能"""
    print("🚀 配置管理器增强功能演示")
    print("=" * 50)

    # 创建配置管理器
    manager = UnifiedConfigManager()

    # 1. 从环境变量加载配置
    print("\n1. 从环境变量加载配置")
    os.environ['RQA_DATABASE_HOST'] = 'localhost'
    os.environ['RQA_DATABASE_PORT'] = '5432'
    os.environ['RQA_DEBUG'] = 'true'
    os.environ['RQA_FEATURES'] = 'auth,cache,monitoring'

    if hasattr(manager, 'load_from_env_with_priority'):
        success = manager.load_from_env_with_priority('RQA_')
        print(f"环境变量加载: {'✅ 成功' if success else '❌ 失败'}")
    else:
        print("❌ 环境变量优先级加载功能未启用")

    # 2. 获取配置及来源信息
    print("\n2. 获取配置及来源信息")
    if hasattr(manager, 'get_config_with_source'):
        source_info = manager.get_config_with_source('database.host')
        print(f"database.host: {source_info}")
    else:
        print("❌ 配置来源追踪功能未启用")

    # 3. 配置验证
    print("\n3. 配置验证")
    required_keys = ['database.host', 'database.port']
    if hasattr(manager, 'validate_required_config'):
        validation = manager.validate_required_config(required_keys)
        print(f"配置验证结果: {validation}")
    else:
        print("❌ 配置验证功能未启用")

    # 4. 配置报告
    print("\n4. 配置报告")
    if hasattr(manager, 'export_config_report'):
        report = manager.export_config_report()
        print(f"配置摘要: {report['total_keys']} 个配置项")
        print(f"配置来源: {list(report['sources'].keys())}")
    else:
        print("❌ 配置报告功能未启用")

    # 5. 配置来源列表
    print("\n5. 配置来源")
    if hasattr(manager, 'list_config_sources'):
        sources = manager.list_config_sources()
        for source_type, info in sources.items():
            print(f"  {source_type}: {info['count']} 个配置项")
    else:
        print("❌ 配置来源列表功能未启用")

    print("\n🎉 演示完成!")


if __name__ == "__main__":
    demo_enhanced_config_manager()
