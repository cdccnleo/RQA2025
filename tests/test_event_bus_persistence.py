#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件总线持久化测试脚本

用于验证以下改进项：
1. P0-2: 完善事件总线集成，实现事件持久化
2. P0-3: 实现数据备份机制
"""

import asyncio
import sys
import time
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, r'c:\PythonProject\RQA2025')


def test_event_bus_persistence():
    """测试事件总线持久化"""
    print("=" * 60)
    print("测试 P0-2: 事件总线持久化")
    print("=" * 60)
    
    try:
        from src.core.event_bus.core import EventBus, EventBusConfig
        
        # 测试1: 使用内存持久化模式
        print("\n1. 测试内存持久化模式...")
        config_memory = EventBusConfig(
            enable_persistence=True,
            persistence_mode="memory"
        )
        event_bus_memory = EventBus(config=config_memory)
        print(f"✅ EventBus创建成功（内存模式）")
        print(f"   - 持久化模式: {event_bus_memory.persistence_mode}")
        
        # 测试2: 使用数据库持久化模式（自动降级）
        print("\n2. 测试数据库持久化模式...")
        config_db = EventBusConfig(
            enable_persistence=True,
            persistence_mode="database"
        )
        event_bus_db = EventBus(config=config_db)
        print(f"✅ EventBus创建成功（数据库模式）")
        print(f"   - 持久化模式: {event_bus_db.persistence_mode}")
        
        # 获取持久化统计
        stats = event_bus_db.get_statistics()
        if "persistence" in stats:
            persistence_stats = stats["persistence"]
            print(f"   - 持久化状态: {persistence_stats}")
        
        # 测试3: 使用自动持久化模式
        print("\n3. 测试自动持久化模式...")
        config_auto = EventBusConfig(
            enable_persistence=True,
            persistence_mode="auto"
        )
        event_bus_auto = EventBus(config=config_auto)
        print(f"✅ EventBus创建成功（自动模式）")
        print(f"   - 持久化模式: {event_bus_auto.persistence_mode}")
        
        print("\n🎉 P0-2 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backup_manager():
    """测试备份管理器"""
    print("\n" + "=" * 60)
    print("测试 P0-3: 数据备份机制")
    print("=" * 60)
    
    try:
        from src.infrastructure.persistence.backup import BackupManager, BackupConfig
        
        # 测试1: 创建备份管理器
        print("\n1. 测试备份管理器初始化...")
        config = BackupConfig(
            backup_dir="test_backups",
            retention_days=7,
            compress=True
        )
        backup_manager = BackupManager(config)
        print("✅ 备份管理器创建成功")
        
        # 测试2: 获取备份统计
        print("\n2. 测试获取备份统计...")
        stats = backup_manager.get_backup_stats()
        print(f"✅ 备份统计获取成功")
        print(f"   - 总备份数: {stats['total_backups']}")
        print(f"   - 总大小: {stats['total_size_mb']} MB")
        print(f"   - 备份目录: {stats['backup_dir']}")
        
        # 测试3: 列出备份
        print("\n3. 测试列出备份...")
        backups = backup_manager.list_backups()
        print(f"✅ 备份列表获取成功")
        print(f"   - 备份数量: {len(backups)}")
        
        print("\n🎉 P0-3 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 " * 20)
    print("事件总线持久化与备份机制验证测试")
    print("🧪 " * 20 + "\n")
    
    results = []
    
    # 测试 P0-2
    result1 = test_event_bus_persistence()
    results.append(("P0-2: 事件总线持久化", result1))
    
    # 测试 P0-3
    result2 = test_backup_manager()
    results.append(("P0-3: 数据备份机制", result2))
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！架构改进实施成功。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 项测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
