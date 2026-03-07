#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证缓存测试文件是否能正确导入和运行
"""

import sys
import os
import traceback

# 添加项目根目录到路径
sys.path.insert(0, '.')

def test_import(module_name):
    """测试模块导入"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} 导入成功")
        return True
    except Exception as e:
        print(f"❌ {module_name} 导入失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始验证缓存测试文件...")
    
    # 测试文件列表
    test_files = [
        "tests.unit.infrastructure.cache.test_cache_manager_coverage_enhancement",
        "tests.unit.infrastructure.cache.test_multi_level_cache_enhancement",
        "tests.unit.infrastructure.cache.test_cache_comprehensive_coverage"
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        if test_import(test_file):
            success_count += 1
    
    print(f"\n测试结果: {success_count}/{total_count} 个测试文件导入成功")
    
    if success_count == total_count:
        print("🎉 所有测试文件都能正确导入！")
        return 0
    else:
        print("⚠️  部分测试文件导入失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())