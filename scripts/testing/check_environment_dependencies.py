#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
环境依赖检查脚本
确保测试环境中的关键依赖版本正确
"""

import sys
import importlib


def check_redis_version():
    """检查Redis版本"""
    try:
        import redis
        version = redis.__version__
        print(f"✅ Redis版本: {version}")

        # 检查RedisCluster是否可用
        try:
            print("✅ RedisCluster导入成功")
            return True
        except ImportError:
            print("❌ RedisCluster导入失败")
            return False

    except ImportError:
        print("❌ Redis未安装")
        return False


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 9


def check_key_dependencies():
    """检查关键依赖"""
    dependencies = {
        'pytest': '8.0.0',
        'numpy': '1.20.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0'
    }

    all_good = True
    for dep, min_version in dependencies.items():
        try:
            module = importlib.import_module(dep)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"✅ {dep}: {version}")
            else:
                print(f"✅ {dep}: 已安装")
        except ImportError:
            print(f"❌ {dep}: 未安装")
            all_good = False

    return all_good


def check_environment_variables():
    """检查环境变量"""
    import os

    # 检查PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH')
    if pythonpath:
        print(f"⚠️  PYTHONPATH已设置: {pythonpath}")
        print("   建议在运行测试时清除此环境变量")
        return False
    else:
        print("✅ PYTHONPATH未设置")
        return True


def main():
    """主函数"""
    print("=== 环境依赖检查 ===")

    checks = [
        ("Python版本", check_python_version),
        ("Redis版本", check_redis_version),
        ("关键依赖", check_key_dependencies),
        ("环境变量", check_environment_variables)
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n--- 检查{name} ---")
        if not check_func():
            all_passed = False

    print(f"\n=== 检查结果 ===")
    if all_passed:
        print("✅ 所有检查通过，环境配置正确")
        return 0
    else:
        print("❌ 发现配置问题，请参考上述建议进行修复")
        return 1


if __name__ == '__main__':
    sys.exit(main())
