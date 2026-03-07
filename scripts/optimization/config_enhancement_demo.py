#!/usr/bin/env python3
"""
配置管理系统优化演示

展示增强后的配置管理功能：
1. 环境变量智能加载
2. YAML文件支持
3. 配置验证增强
4. 配置来源追踪
5. 配置元数据导出
"""

from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager
import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def demo_basic_config_features():
    """演示基本配置功能"""
    print("🔧 基本配置功能演示")
    print("-" * 40)

    manager = UnifiedConfigManager()

    # 设置基本配置
    manager.set("app.name", "RQA2025")
    manager.set("app.version", "2.0.0")
    manager.set("database.host", "localhost")
    manager.set("database.port", 5432)

    print(f"✅ 应用名称: {manager.get('app.name')}")
    print(f"✅ 数据库主机: {manager.get('database.host')}")
    print(f"✅ 数据库端口: {manager.get('database.port')}")

    # 获取配置摘要
    summary = manager.get_config_summary()
    print(f"✅ 配置摘要: {summary['total_sections']} 个节, {summary['total_keys']} 个配置项")

    return manager


def demo_environment_variables(manager):
    """演示环境变量功能"""
    print("\n🌍 环境变量配置演示")
    print("-" * 40)

    # 设置测试环境变量
    test_env_vars = {
        'RQA_LOG_LEVEL': 'INFO',
        'RQA_DEBUG_MODE': 'true',
        'RQA_MAX_CONNECTIONS': '100',
        'RQA_TIMEOUT': '30.5',
        'RQA_FEATURES': 'auth,cache,monitoring',
        'RQA_CONFIG_JSON': '{"key": "value"}'
    }

    # 临时设置环境变量
    for key, value in test_env_vars.items():
        os.environ[key] = value

    # 加载环境变量
    success = manager.load_from_environment_variables('RQA_')
    print(f"✅ 环境变量加载: {'成功' if success else '失败'}")

    # 检查转换结果
    print(f"✅ 日志级别 (字符串): {manager.get('log.level')} ({type(manager.get('log.level')).__name__})")
    print(f"✅ 调试模式 (布尔): {manager.get('debug.mode')} ({type(manager.get('debug.mode')).__name__})")
    print(
        f"✅ 最大连接 (整数): {manager.get('max.connections')} ({type(manager.get('max.connections')).__name__})")
    print(f"✅ 超时时间 (浮点): {manager.get('timeout')} ({type(manager.get('timeout')).__name__})")
    print(f"✅ 功能列表 (列表): {manager.get('features')} ({type(manager.get('features')).__name__})")

    # 清理环境变量
    for key in test_env_vars.keys():
        os.environ.pop(key, None)


def demo_yaml_config(manager):
    """演示YAML配置文件功能"""
    print("\n📄 YAML配置文件演示")
    print("-" * 40)

    # 创建临时YAML文件
    yaml_content = """
app:
  name: "RQA2025-Enhanced"
  version: "2.1.0"
  
database:
  host: "prod-db.example.com"
  port: 5432
  ssl: true
  
logging:
  level: "INFO"
  file: "/var/log/rqa2025.log"
  
features:
  - authentication
  - caching
  - monitoring
  - analytics
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_file = f.name

    try:
        # 加载YAML文件
        success = manager.load_from_yaml_file(yaml_file)
        print(f"✅ YAML文件加载: {'成功' if success else '失败'}")

        if success:
            print(f"✅ 应用名称: {manager.get('app.name')}")
            print(f"✅ 数据库SSL: {manager.get('database.ssl')}")
            print(f"✅ 日志文件: {manager.get('logging.file')}")
            print(f"✅ 功能列表: {manager.get('features')}")
    finally:
        # 清理临时文件
        os.unlink(yaml_file)


def demo_config_validation(manager):
    """演示配置验证功能"""
    print("\n🔍 配置验证演示")
    print("-" * 40)

    # 验证配置完整性
    validation_result = manager.validate_config_integrity()
    print(f"✅ 配置验证: {'通过' if validation_result['is_valid'] else '失败'}")

    if validation_result['missing_keys']:
        print(f"⚠️  缺失配置项: {validation_result['missing_keys']}")

    if validation_result['recommendations']:
        print(f"💡 建议: {validation_result['recommendations']}")

    # 添加缺失配置项
    manager.set('logging.level', 'INFO')
    manager.set('system.debug', False)

    # 重新验证
    validation_result = manager.validate_config_integrity()
    print(f"✅ 修复后验证: {'通过' if validation_result['is_valid'] else '失败'}")


def demo_config_metadata(manager):
    """演示配置元数据功能"""
    print("\n📊 配置元数据演示")
    print("-" * 40)

    # 获取配置源信息
    source_info = manager.get_config_with_source_info('app.name')
    print(f"✅ 配置项 'app.name' 信息:")
    print(f"   值: {source_info['value']}")
    print(f"   类型: {source_info['type']}")
    print(f"   来源: {source_info['source']}")

    # 导出配置元数据
    metadata = manager.export_config_with_metadata()
    print(f"✅ 配置元数据:")
    print(f"   时间戳: {metadata['timestamp']}")
    print(f"   配置节数: {metadata['sections_count']}")
    print(f"   总配置项: {metadata['total_keys']}")
    print(f"   格式版本: {metadata['format_version']}")


def demo_config_operations(manager):
    """演示配置操作功能"""
    print("\n⚙️ 配置操作演示")
    print("-" * 40)

    # 备份配置
    with tempfile.TemporaryDirectory() as temp_dir:
        backup_success = manager.backup_config(temp_dir)
        print(f"✅ 配置备份: {'成功' if backup_success else '失败'}")

        # 清空配置
        original_sections = manager.get_all_sections().copy()
        manager.clear_all()
        print(f"✅ 清空配置: {len(manager.get_all_sections())} 个节")

        # 从备份恢复
        import glob
        backup_files = glob.glob(os.path.join(temp_dir, "config_backup_*.json"))
        if backup_files:
            restore_success = manager.restore_from_backup(backup_files[0])
            print(f"✅ 配置恢复: {'成功' if restore_success else '失败'}")
            print(f"✅ 恢复后节数: {len(manager.get_all_sections())}")


def demo_config_refresh(manager):
    """演示配置刷新功能"""
    print("\n🔄 配置刷新演示")
    print("-" * 40)

    # 设置新的环境变量
    os.environ['RQA_REFRESH_TEST'] = 'success'

    # 刷新配置
    refresh_success = manager.refresh_from_sources()
    print(f"✅ 配置刷新: {'成功' if refresh_success else '失败'}")

    # 检查新配置
    refresh_value = manager.get('refresh.test')
    print(f"✅ 新配置项: {refresh_value}")

    # 清理
    os.environ.pop('RQA_REFRESH_TEST', None)


def main():
    """主演示函数"""
    print("🚀 配置管理系统增强功能演示")
    print("=" * 50)

    # 1. 基本配置功能
    manager = demo_basic_config_features()

    # 2. 环境变量功能
    demo_environment_variables(manager)

    # 3. YAML配置文件
    demo_yaml_config(manager)

    # 4. 配置验证
    demo_config_validation(manager)

    # 5. 配置元数据
    demo_config_metadata(manager)

    # 6. 配置操作
    demo_config_operations(manager)

    # 7. 配置刷新
    demo_config_refresh(manager)

    print("\n🎉 配置管理系统演示完成!")
    print("\n增强功能包括:")
    print("✅ 智能环境变量类型转换")
    print("✅ YAML配置文件支持")
    print("✅ 配置完整性验证")
    print("✅ 配置来源信息追踪")
    print("✅ 配置元数据导出")
    print("✅ 配置备份和恢复")
    print("✅ 多源配置刷新")


if __name__ == "__main__":
    main()
