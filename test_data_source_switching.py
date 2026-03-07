#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据源切换和数据采集功能
"""

import asyncio
from src.core.integration.data_source_manager import get_data_source_manager


def test_data_source_switching():
    """测试数据源切换和数据采集功能"""
    print("=== 测试数据源切换和数据采集功能 ===")
    
    # 获取数据源管理器实例
    data_source_manager = get_data_source_manager()
    print("✅ 数据源管理器初始化成功")
    
    # 获取数据源统计信息
    stats = data_source_manager.get_data_source_stats()
    print("\n=== 数据源统计信息 ===")
    for source_name, source_stats in stats.items():
        print(f"- {source_name}")
        print(f"  状态: {source_stats['status']}")
        print(f"  失败次数: {source_stats['failure_count']}")
        print(f"  成功率: {source_stats['success_rate']:.2f}")
        print(f"  平均响应时间: {source_stats['avg_response_time']:.2f}秒")
    
    # 获取缓存统计信息
    cache_stats = data_source_manager.get_cache_stats()
    print("\n=== 缓存统计信息 ===")
    print(f"  缓存命中: {cache_stats['hits']}")
    print(f"  缓存未命中: {cache_stats['misses']}")
    print(f"  缓存大小: {cache_stats['size']}")
    print(f"  缓存命中率: {cache_stats['hit_rate']:.2f}")
    
    print("\n✅ 数据源切换和数据采集功能测试完成")


async def test_data_collection():
    """测试数据采集功能"""
    print("\n=== 测试数据采集功能 ===")
    
    # 获取数据源管理器实例
    data_source_manager = get_data_source_manager()
    print("✅ 数据源管理器初始化成功")
    
    # 测试获取股票数据
    try:
        print("\n=== 测试获取股票数据 ===")
        symbol = "002837"
        start_date = "20260101"
        end_date = "20260131"
        
        print(f"测试获取股票数据: {symbol}, 日期范围: {start_date}~{end_date}")
        stock_data = await data_source_manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if stock_data:
            print(f"✅ 成功获取股票数据: {len(stock_data)} 条记录")
            if stock_data:
                print(f"  第一条记录: {stock_data[0]}")
        else:
            print("❌ 未能获取股票数据")
    except Exception as e:
        print(f"❌ 获取股票数据失败: {e}")
    
    # 测试获取股票信息
    try:
        print("\n=== 测试获取股票信息 ===")
        symbol = "002837"
        stock_info = await data_source_manager.get_stock_info(symbol=symbol)
        
        if stock_info:
            print(f"✅ 成功获取股票信息: {symbol}")
            print(f"  股票名称: {stock_info.get('股票名称')}")
            print(f"  所属行业: {stock_info.get('所属行业')}")
        else:
            print(f"❌ 未能获取股票信息: {symbol}")
    except Exception as e:
        print(f"❌ 获取股票信息失败: {e}")
    
    print("\n✅ 数据采集功能测试完成")


if __name__ == "__main__":
    # 测试数据源切换
    test_data_source_switching()
    
    # 测试数据采集
    asyncio.run(test_data_collection())
