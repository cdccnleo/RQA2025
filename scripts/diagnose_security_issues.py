#!/usr/bin/env python3
"""
诊断安全模块迁移问题
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_imports():
    """测试基础导入"""
    print("=== 测试基础导入 ===")

    tests = [
        ("src.core.security.unified_security", "UnifiedSecurity"),
        ("src.core.security.authentication_service", "MultiFactorAuthenticationService"),
        ("src.core.security.base_security", "ISecurityComponent"),
        ("src.core.security.audit_system", "get_audit_system"),
        ("src.core.security.access_control", "get_access_control_system"),
        ("src.core.security.encryption_service", "get_encryption_service"),
    ]

    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} 导入失败: {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} 属性不存在: {e}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} 其他错误: {e}")


def test_component_imports():
    """测试组件导入"""
    print("\n=== 测试组件导入 ===")

    component_tests = [
        ("src.core.security.components.audit_components", "AuditComponent"),
        ("src.core.security.components.auth_components", "AuthComponent"),
        ("src.core.security.components.encrypt_components", "EncryptComponent"),
    ]

    for module_name, class_name in component_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} 导入失败: {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} 属性不存在: {e}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} 其他错误: {e}")


def test_service_imports():
    """测试服务导入"""
    print("\n=== 测试服务导入 ===")

    service_tests = [
        ("src.core.security.services.data_protection_service", "DataProtectionService"),
        ("src.core.security.services.web_management_service", "WebManagementService"),
    ]

    for module_name, class_name in service_tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} 导入失败: {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name} 属性不存在: {e}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} 其他错误: {e}")


def test_integration_adapter():
    """测试集成适配器"""
    print("\n=== 测试集成适配器 ===")

    try:
        from src.core.integration.security_adapter import SecurityIntegrationManager
        print("✅ SecurityIntegrationManager 导入成功")

        manager = SecurityIntegrationManager()
        print("✅ SecurityIntegrationManager 创建成功")

        services = manager.get_security_services()
        print(f"✅ 获取安全服务成功: {list(services.keys())}")

    except Exception as e:
        print(f"❌ 集成适配器测试失败: {e}")


def check_file_contents():
    """检查文件内容问题"""
    print("\n=== 检查文件内容问题 ===")

    files_to_check = [
        "src/core/security/__init__.py",
        "src/core/security/unified_security.py",
        "src/core/security/authentication_service.py",
        "src/core/security/audit_system.py",
        "src/core/security/access_control.py",
        "src/core/security/encryption_service.py"
    ]

    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否有明显的问题
                if 'open(' in content and 'with open(' not in content:
                    print(f"⚠️  {file_path} 可能存在 open() 函数使用问题")

                if 'from .' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'from .' in line and 'from .base_security' in line:
                            print(f"⚠️  {file_path}:{i} 相对导入问题: {line.strip()}")

                print(f"✅ {file_path} 文件检查完成")

            except Exception as e:
                print(f"❌ {file_path} 文件检查失败: {e}")
        else:
            print(f"❌ {file_path} 文件不存在")


def main():
    """主函数"""
    print("🔍 RQA2025 安全模块问题诊断")
    print("="*50)

    test_basic_imports()
    test_component_imports()
    test_service_imports()
    test_integration_adapter()
    check_file_contents()

    print("\n" + "="*50)
    print("📋 诊断完成")
    print("如果发现问题，请根据上述输出修复相应的文件")


if __name__ == "__main__":
    main()
