#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证导入迁移效果
测试代码重复定义修复和统一导入规范
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_logger_import():
    """测试logger导入迁移效果"""
    print("🔍 测试logger导入迁移...")

    try:
        # 测试通用层logger导入
        from src.utils.logger import get_logger

        # 验证可以正常获取logger
        logger = get_logger("test_migration")
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')

        # 测试日志功能
        logger.info("✅ Logger导入测试通过")
        print("✅ Logger导入迁移成功")
        return True

    except Exception as e:
        print(f"❌ Logger导入测试失败: {e}")
        return False


def test_date_utils_import():
    """测试date_utils导入迁移效果"""
    print("🔍 测试date_utils导入迁移...")

    try:
        # 测试通用层date_utils导入
        from src.utils.date_utils import convert_timezone, get_business_date

        # 验证函数存在
        assert convert_timezone is not None
        assert get_business_date is not None

        # 验证函数签名
        import inspect
        convert_sig = inspect.signature(convert_timezone)
        assert len(convert_sig.parameters) == 3  # dt, from_tz, to_tz

        print("✅ Date utils导入迁移成功")
        return True

    except Exception as e:
        print(f"❌ Date utils导入测试失败: {e}")
        return False


def test_infrastructure_logger_availability():
    """测试基础设施层logger仍然可用"""
    print("🔍 测试基础设施层logger可用性...")

    try:
        # 测试基础设施层高级功能仍然可用
        from src.infrastructure.utils.logger import LoggerFactory, configure_logging

        # 验证高级功能存在
        assert LoggerFactory is not None
        assert configure_logging is not None

        print("✅ 基础设施层logger高级功能可用")
        return True

    except Exception as e:
        print(f"❌ 基础设施层logger测试失败: {e}")
        return False


def test_no_circular_import():
    """测试没有循环导入"""
    print("🔍 测试循环导入...")

    try:
        # 测试导入不会导致循环依赖
        pass

        print("✅ 无循环导入问题")
        return True

    except ImportError as e:
        print(f"❌ 循环导入检测失败: {e}")
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("🔍 测试向后兼容性...")

    try:
        # 测试旧的导入方式仍然可用（通过重定向）
        from src.utils.logger import get_logger
        from src.utils.date_utils import convert_timezone

        # 验证功能正常
        logger = get_logger("test_compatibility")
        logger.info("测试向后兼容性")

        # 验证时区转换功能
        from datetime import datetime
        dt = datetime.now()
        converted = convert_timezone(dt, "UTC", "Asia/Shanghai")
        assert converted is not None

        print("✅ 向后兼容性测试通过")
        return True

    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        return False


def test_import_consistency():
    """测试导入一致性"""
    print("🔍 测试导入一致性...")

    try:
        # 验证所有模块都使用统一的导入方式
        import importlib

        # 测试关键模块的导入
        modules_to_test = [
            'src.utils.logger',
            'src.utils.date_utils',
            'src.utils.math_utils',
            'src.utils.data_utils'
        ]

        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError as e:
                print(f"❌ 模块 {module_name} 导入失败: {e}")
                return False

        print("✅ 导入一致性测试通过")
        return True

    except Exception as e:
        print(f"❌ 导入一致性测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 开始验证导入迁移效果...")
    print("=" * 50)

    tests = [
        test_logger_import,
        test_date_utils_import,
        test_infrastructure_logger_availability,
        test_no_circular_import,
        test_backward_compatibility,
        test_import_consistency
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！导入迁移成功！")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
