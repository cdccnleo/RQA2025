#!/usr/bin/env python3
"""
统一配置管理器演示脚本
验证新的配置管理功能
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_unified_config_manager():
    """演示统一配置管理器功能"""
    print("🔧 统一配置管理器演示")
    print("=" * 50)

    try:
        # 导入新的统一配置管理器
        from src.infrastructure.config.unified_manager import (
            UnifiedConfigManager,
            get_unified_config_manager,
            get_config,
            set_config,
            ConfigScope
        )

        print("✅ 成功导入统一配置管理器")

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"📁 临时目录: {temp_dir}")

        # 1. 测试基本功能
        print("\n1️⃣ 测试基本功能")
        config_manager = UnifiedConfigManager(config_dir=temp_dir)

        # 设置和获取配置
        config_manager.set('database.url', 'sqlite:///demo.db')
        config_manager.set('logging.level', 'INFO')

        db_url = config_manager.get('database.url')
        log_level = config_manager.get('logging.level')

        print(f"  数据库URL: {db_url}")
        print(f"  日志级别: {log_level}")

        # 2. 测试作用域配置
        print("\n2️⃣ 测试作用域配置")

        # 特征工程作用域
        config_manager.set('feature.enabled', True, ConfigScope.FEATURES)
        config_manager.set('feature.batch_size', 1000, ConfigScope.FEATURES)

        # 数据层作用域
        config_manager.set('data.source', 'tushare', ConfigScope.DATA)
        config_manager.set('data.cache_enabled', True, ConfigScope.DATA)

        feature_enabled = config_manager.get('feature.enabled', ConfigScope.FEATURES)
        data_source = config_manager.get('data.source', ConfigScope.DATA)

        print(f"  特征工程启用: {feature_enabled}")
        print(f"  数据源: {data_source}")

        # 3. 测试配置验证
        print("\n3️⃣ 测试配置验证")

        # 有效配置
        is_valid, errors = config_manager.validate({
            'database.url': 'sqlite:///test.db',
            'logging.level': 'INFO'
        })
        print(f"  有效配置验证: {'✅' if is_valid else '❌'}")

        # 无效配置
        is_valid, errors = config_manager.validate({
            'database.url': None,
            'logging.level': 'INFO'
        })
        print(f"  无效配置验证: {'✅' if is_valid else '❌'}")
        if errors:
            print(f"  错误信息: {errors}")

        # 4. 测试配置持久化
        print("\n4️⃣ 测试配置持久化")

        config_file = Path(temp_dir) / "demo_config.json"
        success = config_manager.save(str(config_file))
        print(f"  配置保存: {'✅' if success else '❌'}")

        # 创建新的管理器并加载配置
        new_manager = UnifiedConfigManager(config_dir=temp_dir)
        success = new_manager.load(str(config_file))
        print(f"  配置加载: {'✅' if success else '❌'}")

        loaded_db_url = new_manager.get('database.url')
        print(f"  加载的数据库URL: {loaded_db_url}")

        # 5. 测试监听器功能
        print("\n5️⃣ 测试监听器功能")

        changes = []

        def watcher_callback(key, old_value, new_value):
            changes.append((key, old_value, new_value))
            print(f"    🔔 配置变更: {key} = {new_value}")

        watcher_id = config_manager.add_watcher('watched.key', watcher_callback)
        config_manager.set('watched.key', 'new_value')

        print(f"  监听器调用次数: {len(changes)}")

        # 6. 测试导出导入
        print("\n6️⃣ 测试导出导入")

        exported = config_manager.export_config()
        print(f"  导出配置键数量: {len(exported.get('global_config', {}))}")

        new_manager2 = UnifiedConfigManager(config_dir=temp_dir)
        success = new_manager2.import_config(exported)
        print(f"  配置导入: {'✅' if success else '❌'}")

        # 7. 测试全局函数
        print("\n7️⃣ 测试全局函数")

        set_config('global.key', 'global_value')
        global_value = get_config('global.key')
        print(f"  全局配置: {global_value}")

        set_config('scope.key', 'scope_value', ConfigScope.FEATURES)
        scope_value = get_config('scope.key', ConfigScope.FEATURES)
        print(f"  作用域配置: {scope_value}")

        # 8. 测试单例模式
        print("\n8️⃣ 测试单例模式")

        manager1 = get_unified_config_manager()
        manager2 = get_unified_config_manager()

        is_same = manager1 is manager2
        print(f"  单例模式: {'✅' if is_same else '❌'}")

        # 9. 测试作用域配置获取
        print("\n9️⃣ 测试作用域配置获取")

        infra_config = config_manager.get_scope_config(ConfigScope.INFRASTRUCTURE)
        data_config = config_manager.get_scope_config(ConfigScope.DATA)

        print(f"  基础设施配置项数: {len(infra_config)}")
        print(f"  数据层配置项数: {len(data_config)}")

        # 10. 测试配置项详情
        print("\n🔟 测试配置项详情")

        from src.infrastructure.config.interfaces.unified_interface import ConfigItem

        config_item = ConfigItem(
            key="demo.key",
            value="demo_value",
            scope=ConfigScope.GLOBAL,
            description="演示配置项",
            version="1.0",
            required=True
        )

        print(f"  配置项: {config_item.key} = {config_item.value}")
        print(f"  作用域: {config_item.scope.value}")
        print(f"  描述: {config_item.description}")
        print(f"  必需: {config_item.required}")

        print("\n✅ 所有测试通过!")

        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🧹 已清理临时目录: {temp_dir}")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """主函数"""
    print("🚀 开始统一配置管理器演示")
    print("=" * 50)

    success = demo_unified_config_manager()

    if success:
        print("\n🎉 演示完成!")
        print("📖 迁移指南: docs/migration/config_management_migration.md")
        print("🧪 测试文件: tests/unit/infrastructure/config/test_unified_config_manager_simple.py")
    else:
        print("\n💥 演示失败!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
