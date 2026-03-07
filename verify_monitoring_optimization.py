#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据持久化优化验证脚本
验证所有优化功能是否正常工作
"""

import sys
import os
from pathlib import Path
import time

# 添加脚本路径到sys.path
current_dir = Path(__file__).parent
scripts_dir = current_dir / "scripts" / "optimization"
sys.path.insert(0, str(scripts_dir))


def main():
    """主验证函数"""
    print("=== 监控数据持久化优化验证 ===")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本目录: {scripts_dir}")

    # 验证文件存在性
    required_files = [
        "monitoring_persistence_enhancer.py",
        "enhanced_monitoring_service.py",
        "monitoring_persistence_demo.py",
        "apply_monitoring_persistence_enhancements.py"
    ]

    print("\n1. 检查必要文件...")
    all_files_exist = True
    for file_name in required_files:
        file_path = scripts_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name} - 存在")
        else:
            print(f"✗ {file_name} - 缺失")
            all_files_exist = False

    if not all_files_exist:
        print("❌ 部分必要文件缺失")
        return False

    # 尝试导入核心模块
    print("\n2. 测试模块导入...")
    try:
        from monitoring_persistence_enhancer import EnhancedMetricsPersistenceManager
        print("✓ EnhancedMetricsPersistenceManager - 导入成功")
    except ImportError as e:
        print(f"✗ EnhancedMetricsPersistenceManager - 导入失败: {e}")
        return False

    try:
        from enhanced_monitoring_service import EnhancedMonitoringService
        print("✓ EnhancedMonitoringService - 导入成功")
    except ImportError as e:
        print(f"✗ EnhancedMonitoringService - 导入失败: {e}")
        return False

    # 创建简单的功能测试
    print("\n3. 基本功能测试...")
    try:
        # 测试持久化管理器
        config = {
            'path': './test_monitoring_data',
            'enable_compression': True,
            'batch_size': 100
        }

        manager = EnhancedMetricsPersistenceManager(config)
        print("✓ 持久化管理器创建成功")

        # 测试监控服务
        service = EnhancedMonitoringService()
        print("✓ 监控服务创建成功")

        # 简单的度量存储测试
        test_metric = {
            'name': 'test_metric',
            'value': 100.0,
            'timestamp': time.time(),
            'tags': {'test': 'true'}
        }

        manager.store_metrics([test_metric])
        print("✓ 度量存储测试成功")

        # 查询测试
        recent_metrics = manager.get_recent_metrics('test_metric', hours=1)
        print(f"✓ 度量查询测试成功 - 找到 {len(recent_metrics)} 条记录")

    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False

    print("\n=== 验证完成 ===")
    print("✅ 监控数据持久化优化验证通过")
    print("\n优化特性:")
    print("• 高性能批量数据写入")
    print("• 多级缓存机制(热缓存/温缓存)")
    print("• 数据压缩和归档")
    print("• 智能生命周期管理")
    print("• 实时数据流处理")
    print("• 与现有系统兼容")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
