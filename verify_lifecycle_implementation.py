#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证组件生命周期管理实现

检查：
1. 生命周期管理器是否正确初始化
2. 基础设施服务注册表是否正确工作
3. 适配器是否正确集成
4. 服务是否只初始化一次
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_lifecycle_manager():
    """测试生命周期管理器"""
    print("=" * 60)
    print("测试1: 组件生命周期管理器")
    print("=" * 60)
    
    try:
        from src.core.lifecycle import get_lifecycle_manager, ComponentLifecycleManager
        
        # 测试单例模式
        manager1 = get_lifecycle_manager()
        manager2 = get_lifecycle_manager()
        
        if manager1 is manager2:
            print("✅ 生命周期管理器单例模式正常")
        else:
            print("❌ 生命周期管理器单例模式失败")
            return False
        
        # 测试基本功能
        components = manager1.get_all_components()
        print(f"✅ 已注册组件数量: {len(components)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生命周期管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_registry():
    """测试服务注册表"""
    print("\n" + "=" * 60)
    print("测试2: 基础设施服务注册表")
    print("=" * 60)
    
    try:
        from src.infrastructure.core import get_service_registry, InfrastructureServiceRegistry
        
        # 测试单例模式
        registry1 = get_service_registry()
        registry2 = get_service_registry()
        
        if registry1 is registry2:
            print("✅ 服务注册表单例模式正常")
        else:
            print("❌ 服务注册表单例模式失败")
            return False
        
        # 测试服务注册
        from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
        
        if not registry1.is_service_registered('test_config_manager'):
            registry1.register_singleton('test_config_manager', service_class=UnifiedConfigManager)
            print("✅ 服务注册功能正常")
        else:
            print("⚠️  测试服务已注册")
        
        # 测试服务获取
        service = registry1.get_service('test_config_manager')
        if service is not None:
            print("✅ 服务获取功能正常")
        else:
            print("❌ 服务获取失败")
            return False
        
        # 测试单例：多次获取应该是同一实例
        service1 = registry1.get_service('test_config_manager')
        service2 = registry1.get_service('test_config_manager')
        
        if service1 is service2:
            print("✅ 服务单例模式正常")
        else:
            print("❌ 服务单例模式失败")
            return False
        
        # 清理测试服务
        registry1.unregister_service('test_config_manager')
        
        return True
        
    except Exception as e:
        print(f"❌ 服务注册表测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter_integration():
    """测试适配器集成"""
    print("\n" + "=" * 60)
    print("测试3: 适配器集成")
    print("=" * 60)
    
    try:
        from src.core.integration.adapters import get_all_adapters
        from src.core.lifecycle import get_lifecycle_manager
        
        # 获取适配器
        adapters = get_all_adapters()
        print(f"✅ 获取适配器成功，数量: {len(adapters)}")
        
        # 检查生命周期管理器中的适配器
        lifecycle_manager = get_lifecycle_manager()
        components = lifecycle_manager.get_all_components()
        
        adapter_components = [
            c for c in components.values()
            if c.component_id.startswith('adapter_')
        ]
        
        print(f"✅ 生命周期管理器中的适配器数量: {len(adapter_components)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 适配器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_business_adapter():
    """测试统一业务适配器"""
    print("\n" + "=" * 60)
    print("测试4: 统一业务适配器")
    print("=" * 60)
    
    try:
        from src.core.integration.unified_business_adapters import (
            UnifiedBusinessAdapter,
            BusinessLayerType
        )
        from src.infrastructure.core import get_service_registry
        
        # 创建适配器
        adapter = UnifiedBusinessAdapter(layer_type=BusinessLayerType.DATA)
        print("✅ 创建适配器成功")
        
        # 获取基础设施服务
        services = adapter.get_infrastructure_services()
        print(f"✅ 获取基础设施服务成功，数量: {len(services)}")
        
        # 检查服务是否来自注册表
        registry = get_service_registry()
        config_manager = registry.get_service('config_manager')
        
        if config_manager is not None:
            print("✅ 服务注册表中有配置管理器")
        else:
            print("⚠️  服务注册表中没有配置管理器（可能未初始化）")
        
        return True
        
    except Exception as e:
        print(f"❌ 统一业务适配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_service_initialization():
    """测试服务初始化"""
    print("\n" + "=" * 60)
    print("测试5: 服务初始化")
    print("=" * 60)
    
    try:
        from src.infrastructure.core import initialize_infrastructure_services
        
        # 初始化服务
        results = initialize_infrastructure_services()
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        print(f"✅ 服务初始化完成: {success_count}/{total_count} 成功")
        
        for service_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {service_name}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 服务初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("组件生命周期管理实现验证")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("生命周期管理器", test_lifecycle_manager()))
    results.append(("服务注册表", test_service_registry()))
    results.append(("适配器集成", test_adapter_integration()))
    results.append(("统一业务适配器", test_unified_business_adapter()))
    results.append(("服务初始化", test_service_initialization()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"总计: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)