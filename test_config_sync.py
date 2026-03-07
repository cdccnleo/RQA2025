#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置同步和状态监控功能
"""

from src.gateway.web.data_source_config_manager import get_data_source_config_manager
from src.core.integration.data_source_manager import get_data_source_manager, reset_data_source_manager


def test_config_sync():
    """测试配置同步功能"""
    print("=== 测试配置同步功能 ===")
    
    # 重置数据源管理器实例
    reset_data_source_manager()
    print("✅ 数据源管理器已重置")
    
    # 获取配置管理器实例
    config_manager = get_data_source_config_manager()
    print("✅ 配置管理器初始化成功")
    
    # 获取所有数据源配置
    data_sources = config_manager.get_data_sources()
    print(f"✅ 数据源数量: {len(data_sources)}")
    
    # 查找Baostock数据源
    baostock_source = None
    for source in data_sources:
        if source.get('id') == 'baostock_stock_a':
            baostock_source = source
            break
    
    if baostock_source:
        print(f"✅ 找到Baostock数据源: {baostock_source.get('name')}")
        print(f"  当前状态: {'启用' if baostock_source.get('enabled') else '禁用'}")
        
        # 切换Baostock数据源的启用状态
        new_enabled = not baostock_source.get('enabled', False)
        print(f"\n=== 切换Baostock数据源状态 ===")
        print(f"  新状态: {'启用' if new_enabled else '禁用'}")
        
        # 更新数据源配置
        update_result = config_manager.update_data_source('baostock_stock_a', {'enabled': new_enabled})
        print(f"✅ 数据源配置更新: {'成功' if update_result else '失败'}")
        
        # 重新获取数据源配置
        updated_sources = config_manager.get_data_sources()
        updated_baostock = None
        for source in updated_sources:
            if source.get('id') == 'baostock_stock_a':
                updated_baostock = source
                break
        
        if updated_baostock:
            print(f"✅ 验证更新结果: {'启用' if updated_baostock.get('enabled') else '禁用'}")
            print(f"  更新成功: {updated_baostock.get('enabled') == new_enabled}")
    else:
        print("❌ 未找到Baostock数据源")
    
    # 测试数据源管理器配置同步
    print("\n=== 测试数据源管理器配置同步 ===")
    data_source_manager = get_data_source_manager()
    print("✅ 数据源管理器初始化成功")
    
    # 获取数据源统计信息
    stats = data_source_manager.get_data_source_stats()
    print("✅ 数据源统计信息获取成功")
    for source_name, source_stats in stats.items():
        print(f"- {source_name}")
        print(f"  状态: {source_stats['status']}")
        print(f"  失败次数: {source_stats['failure_count']}")
        print(f"  成功率: {source_stats['success_rate']:.2f}")
    
    print("\n✅ 配置同步和状态监控功能测试完成")


if __name__ == "__main__":
    test_config_sync()
