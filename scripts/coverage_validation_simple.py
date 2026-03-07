#!/usr/bin/env python3
"""
简单测试覆盖率验证脚本
"""

import sys
import os
import subprocess

# 确保项目根目录在Python路径中
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_infrastructure_coverage():
    """测试基础设施层覆盖率"""

    print("🔬 开始基础设施层覆盖率测试...")

    # 测试配置管理
    try:
        # 直接导入具体的模块
        import src.infrastructure.config.unified_manager as config_module
        UnifiedConfigManager = config_module.UnifiedConfigManager
        config = UnifiedConfigManager()

        # 测试各种配置操作
        config.set('app', 'name', 'RQA2025')
        config.set('app', 'version', '1.0.0')
        config.set('cache', 'enabled', True)
        config.set('logging', 'level', 'INFO')

        # 测试读取配置
        name = config.get('app', 'name')
        version = config.get('app', 'version')
        cache_enabled = config.get('cache', 'enabled')
        log_level = config.get('logging', 'level')

        # 测试默认值
        default_value = config.get('nonexistent', 'key', 'default')

        assert name == 'RQA2025'
        assert version == '1.0.0'
        assert cache_enabled == True
        assert log_level == 'INFO'
        assert default_value == 'default'

        print("✅ 配置管理覆盖率测试通过")

    except Exception as e:
        print(f"❌ 配置管理测试失败: {e}")
        return False

    # 测试缓存管理
    try:
        import src.infrastructure.cache.cache_service as cache_module
        CacheService = cache_module.CacheService
        cache = CacheService(maxsize=50, ttl=600)

        # 测试缓存操作
        cache.set('user_1', {'name': 'Alice', 'age': 30})
        cache.set('user_2', {'name': 'Bob', 'age': 25})
        cache.set('settings', {'theme': 'dark', 'lang': 'zh'})

        # 测试缓存读取
        user1 = cache.get('user_1')
        user2 = cache.get('user_2')
        settings = cache.get('settings')
        nonexistent = cache.get('nonexistent')

        assert user1['name'] == 'Alice'
        assert user2['age'] == 25
        assert settings['theme'] == 'dark'
        assert nonexistent is None

        print("✅ 缓存管理覆盖率测试通过")

    except Exception as e:
        print(f"❌ 缓存管理测试失败: {e}")
        return False

    # 测试日志管理
    try:
        from src.infrastructure.logging import Logger
        logger = Logger('test_logger')

        # 测试日志记录
        logger.logger.info('测试信息日志')
        logger.logger.warning('测试警告日志')
        logger.logger.error('测试错误日志')

        print("✅ 日志管理覆盖率测试通过")

    except Exception as e:
        print(f"❌ 日志管理测试失败: {e}")
        return False

    return True


def generate_coverage_report():
    """生成覆盖率报告"""
    print("\n📊 生成覆盖率报告...")

    try:
        # 生成覆盖率报告
        cmd = [sys.executable, '-m', 'coverage', 'report', '--show-missing']
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\n📋 覆盖率报告:")
        print(result.stdout)

        if result.stderr:
            print("\n⚠️  错误信息:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ 生成覆盖率报告失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 开始测试覆盖率验证...")
    print("=" * 50)

    # 测试基础设施层覆盖率
    if test_infrastructure_coverage():
        print("\n✅ 基础设施层覆盖率测试全部通过")

        # 生成覆盖率报告
        if generate_coverage_report():
            print("✅ 覆盖率报告生成成功")
        else:
            print("❌ 覆盖率报告生成失败")

    else:
        print("\n❌ 基础设施层覆盖率测试失败")

    print("\n🎉 覆盖率验证完成！")


if __name__ == "__main__":
    main()
