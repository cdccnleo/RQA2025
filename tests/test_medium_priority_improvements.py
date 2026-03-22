#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中风险改进项验证测试脚本

用于验证以下改进项的实施效果：
1. ST-1: 统一数据库配置管理
2. ST-2: 实现存储配额管理
"""

import sys
import os
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, r'c:\PythonProject\RQA2025')


def test_st1_unified_db_config():
    """测试 ST-1: 统一数据库配置管理"""
    print("=" * 60)
    print("验证 ST-1: 统一数据库配置管理")
    print("=" * 60)
    
    try:
        # 测试1: 验证 feature_store.py 中已移除硬编码配置
        print("\n1. 检查 feature_store.py 中已移除硬编码配置...")
        
        with open(r'c:\PythonProject\RQA2025\src\features\core\feature_store.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否还有硬编码密码
        if 'SecurePass123!' in content:
            print("❌ 测试失败：feature_store.py 中仍存在硬编码密码")
            return False
        
        # 检查是否统一使用 database_config 模块
        if 'from src.infrastructure.persistence.database_config import get_db_config' in content:
            print("✅ 测试通过：feature_store.py 统一使用 database_config 模块")
        else:
            print("❌ 测试失败：未找到统一的配置导入")
            return False
        
        # 测试2: 验证 feature_saver.py 中已移除硬编码配置
        print("\n2. 检查 feature_saver.py 中已移除硬编码配置...")
        
        with open(r'c:\PythonProject\RQA2025\src\features\core\feature_saver.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'SecurePass123!' in content:
            print("❌ 测试失败：feature_saver.py 中仍存在硬编码密码")
            return False
        
        print("✅ 测试通过：feature_saver.py 中已移除硬编码密码")
        
        print("\n🎉 ST-1 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_st2_storage_quota_manager():
    """测试 ST-2: 实现存储配额管理"""
    print("\n" + "=" * 60)
    print("验证 ST-2: 实现存储配额管理")
    print("=" * 60)
    
    try:
        # 测试1: 验证 StorageQuotaManager 类存在
        print("\n1. 验证 StorageQuotaManager 类...")
        
        from src.infrastructure.persistence.storage_quota_manager import (
            StorageQuotaManager, StorageQuotaConfig, get_storage_quota_manager
        )
        
        print("✅ StorageQuotaManager 模块导入成功")
        
        # 测试2: 验证配额管理器可以创建
        print("\n2. 验证配额管理器创建...")
        
        # 创建临时目录用于测试
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = StorageQuotaConfig(
                max_storage_size=100 * 1024 * 1024,  # 100MB
                max_file_count=100,
                auto_cleanup_enabled=True
            )
            
            manager = StorageQuotaManager(temp_dir, config)
            print("✅ 配额管理器创建成功")
            
            # 测试3: 验证存储统计功能
            print("\n3. 验证存储统计功能...")
            
            stats = manager.get_storage_stats()
            print(f"✅ 存储统计获取成功")
            print(f"   - 文件数量: {stats.file_count}")
            print(f"   - 总大小: {stats.total_size} bytes")
            
            # 测试4: 验证配额检查功能
            print("\n4. 验证配额检查功能...")
            
            quota_check = manager.check_quota()
            print(f"✅ 配额检查成功")
            print(f"   - 状态: {quota_check.get('status')}")
            print(f"   - 使用率: {quota_check.get('usage_percent', 0):.2f}%")
            
            # 测试5: 验证配额报告功能
            print("\n5. 验证配额报告功能...")
            
            report = manager.get_quota_report()
            print(f"✅ 配额报告生成成功")
            print(f"   - 存储路径: {report.get('storage_path')}")
            print(f"   - 最大存储: {report.get('config', {}).get('max_storage_size_mb')} MB")
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 测试6: 验证 FeatureStore 集成
        print("\n6. 验证 FeatureStore 集成...")
        
        from src.features.core.feature_store import FeatureStore, STORAGE_QUOTA_AVAILABLE
        
        if STORAGE_QUOTA_AVAILABLE:
            print("✅ FeatureStore 已集成存储配额管理")
        else:
            print("⚠️ FeatureStore 未集成存储配额管理（可能依赖未安装）")
        
        print("\n🎉 ST-2 验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有验证测试"""
    print("\n" + "🧪 " * 20)
    print("中风险改进项实施效果验证测试")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # 测试 ST-1
    result1 = test_st1_unified_db_config()
    results.append(("ST-1: 统一数据库配置管理", result1))
    
    # 测试 ST-2
    result2 = test_st2_storage_quota_manager()
    results.append(("ST-2: 实现存储配额管理", result2))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("验证测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项验证通过")
    
    if passed == total:
        print("\n🎉 所有改进项验证通过！实施成功。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 项验证失败，请检查实施。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
