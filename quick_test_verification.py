#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试验证脚本

验证缓存测试文件是否能正确导入和基本运行
"""

import sys
import os
import traceback

# 添加项目根目录到路径
sys.path.insert(0, '.')

def test_module_import_and_basic_functionality(module_name, test_class_name):
    """测试模块导入和基本功能"""
    try:
        # 导入模块
        module = __import__(module_name, fromlist=[test_class_name])
        print(f"✅ {module_name} 导入成功")
        
        # 获取测试类
        test_class = getattr(module, test_class_name)
        print(f"✅ {test_class_name} 类获取成功")
        
        # 创建实例
        test_instance = test_class()
        print(f"✅ {test_class_name} 实例创建成功")
        
        # 调用setup_method（如果存在）
        if hasattr(test_instance, 'setup_method'):
            test_instance.setup_method(None)
            print(f"✅ {test_class_name} setup_method 执行成功")
        
        # 调用teardown_method（如果存在）
        if hasattr(test_instance, 'teardown_method'):
            test_instance.teardown_method(None)
            print(f"✅ {test_class_name} teardown_method 执行成功")
        
        return True
    except Exception as e:
        print(f"❌ {module_name} 测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始快速测试验证...")
    
    # 测试配置
    test_modules = [
        ("tests.unit.infrastructure.cache.test_cache_manager_coverage_enhancement", "TestCacheManagerCoverageEnhancement"),
        ("tests.unit.infrastructure.cache.test_multi_level_cache_enhancement", "TestMultiLevelCacheEnhancement"),
        ("tests.unit.infrastructure.cache.test_cache_comprehensive_coverage", "TestCacheComprehensiveCoverage")
    ]
    
    success_count = 0
    total_count = len(test_modules)
    
    for module_name, test_class_name in test_modules:
        print(f"\n--- 测试 {module_name} ---")
        if test_module_import_and_basic_functionality(module_name, test_class_name):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"测试结果: {success_count}/{total_count} 个模块测试成功")
    
    if success_count == total_count:
        print("🎉 所有测试模块都能正确工作！")
        return 0
    else:
        print("⚠️  部分测试模块存在问题，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())