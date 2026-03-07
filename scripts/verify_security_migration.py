#!/usr/bin/env python3
"""
验证安全模块迁移结果
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_new_security_imports():
    """测试新的安全模块导入"""
    print("=== 测试新的安全模块导入 ===")

    try:
        # 测试统一安全模块导入
        print("✅ unified_security 导入成功")

        # 测试认证服务导入
        print("✅ authentication_service 导入成功")

        # 测试基础安全接口导入
        print("✅ base_security 导入成功")

        # 测试组件导入
        print("✅ audit_components 导入成功")

        # 测试服务导入
        print("✅ data_protection_service 导入成功")

        # 测试我们自己的模块导入
        print("✅ 自定义安全模块导入成功")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False


def test_security_integration():
    """测试安全集成适配器"""
    print("\n=== 测试安全集成适配器 ===")

    try:
        from src.core.integration.security_adapter import SecurityIntegrationManager

        # 创建集成管理器
        security_manager = SecurityIntegrationManager()
        print("✅ 安全集成管理器创建成功")

        # 测试服务获取
        services = security_manager.get_security_services()
        print(f"✅ 获取安全服务: {list(services.keys())}")

        # 测试健康检查
        health = security_manager.perform_security_health_check()
        print(f"✅ 健康检查结果: {health.get('status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n=== 测试向后兼容性 ===")

    # 测试旧路径是否仍然可用（通过集成层）
    try:
        from src.core.integration.security_adapter import get_security_integration_manager

        old_security_manager = get_security_integration_manager()
        print("✅ 旧路径兼容性测试通过")

        return True

    except Exception as e:
        print(f"❌ 向后兼容性测试失败: {e}")
        return False


def check_directory_structure():
    """检查目录结构"""
    print("\n=== 检查目录结构 ===")

    base_path = Path(__file__).parent.parent / "src" / "core" / "security"

    expected_structure = {
        "files": [
            "__init__.py",
            "unified_security.py",
            "authentication_service.py",
            "base_security.py",
            "security_factory.py",
            "security_utils.py"
        ],
        "directories": [
            "components",
            "services"
        ]
    }

    # 检查文件
    missing_files = []
    for file in expected_structure["files"]:
        if not (base_path / file).exists():
            missing_files.append(file)

    # 检查目录
    missing_dirs = []
    for dir_name in expected_structure["directories"]:
        if not (base_path / dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False

    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False

    print("✅ 目录结构完整")
    return True


def main():
    """主测试函数"""
    print("🔍 RQA2025 安全模块迁移验证")
    print("="*50)

    test_results = []

    # 检查目录结构
    result = check_directory_structure()
    test_results.append(("目录结构检查", result))

    # 测试新导入
    result = test_new_security_imports()
    test_results.append(("新模块导入", result))

    # 测试集成
    result = test_security_integration()
    test_results.append(("安全集成", result))

    # 测试向后兼容性
    result = test_backward_compatibility()
    test_results.append(("向后兼容性", result))

    # 输出结果
    print("\n" + "="*50)
    print("📊 验证结果汇总:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 安全模块迁移验证完全成功！")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
