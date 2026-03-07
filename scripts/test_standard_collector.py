#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试标准数据采集器功能

验证标准数据采集器的各项功能是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestration.standard_data_collector import get_standard_data_collector

def main():
    """
    主函数
    """
    print("🚀 启动标准数据采集器测试...")
    print("=" * 60)
    
    try:
        # 获取标准数据采集器实例
        collector = get_standard_data_collector()
        print("✅ 标准数据采集器初始化成功")
        print(f"📊 AKShare服务可用性: {collector.is_available}")
        
        # 1. 获取标准配置
        print("\n📋 获取标准配置...")
        config = collector.get_standard_config()
        print(f"✅ 标准配置获取成功")
        print(f"📋 标准source_id: {config['standard_source_id']}")
        print(f"📋 支持的数据类型: {config['standard_data_types']}")
        print(f"📋 最大并发任务: {config['max_concurrent_tasks']}")
        print(f"📋 批次大小: {config['batch_size']}")
        
        # 2. 测试单个股票数据采集
        print("\n📈 测试单个股票数据采集...")
        print("目标: 002837, 日期范围: 20260101 ~ 20260131")
        
        import asyncio
        result = asyncio.run(collector.collect_stock_data(
            symbol="002837",
            start_date="20260101",
            end_date="20260131",
            data_type="daily",
            adjust="qfq"
        ))
        
        print(f"\n📊 采集结果:")
        print(f"✅ 成功: {result.get('success', False)}")
        if result.get('success'):
            print(f"📋 记录数: {result.get('records_count', 0)}")
            print(f"📋 数据源ID: {result.get('source_id')}")
            print(f"📋 数据类型: {result.get('data_type')}")
            
            # 检查数据质量
            validation = result.get('validation', {})
            quality_score = validation.get('overall_quality_score', 0)
            print(f"📊 数据质量得分: {quality_score:.2f}")
            
            # 检查第一条记录
            data = result.get('data', [])
            if data:
                print(f"💡 第一条记录: {data[0] if len(data) > 0 else '无'}")
        else:
            print(f"❌ 错误: {result.get('error')}")
        
        # 3. 测试批量数据采集
        print("\n📊 测试批量数据采集...")
        print("目标: ['002837', '600519'], 日期范围: 20260101 ~ 20260131")
        
        batch_result = asyncio.run(collector.batch_collect_stock_data(
            symbols=["002837", "600519"],
            start_date="20260101",
            end_date="20260131",
            data_type="daily",
            adjust="qfq"
        ))
        
        print(f"\n📊 批量采集结果:")
        print(f"📋 总任务数: {len(batch_result)}")
        
        success_count = sum(1 for r in batch_result if r.get("success"))
        fail_count = len(batch_result) - success_count
        
        print(f"✅ 成功: {success_count}, ❌ 失败: {fail_count}")
        
        # 4. 测试增量数据采集
        print("\n🔄 测试增量数据采集...")
        print("目标: ['002837'], 最近7天")
        
        incremental_result = asyncio.run(collector.incremental_collect(
            symbols=["002837"],
            days=7,
            data_type="daily",
            adjust="qfq"
        ))
        
        print(f"\n📊 增量采集结果:")
        print(f"📋 任务数: {len(incremental_result)}")
        
        incremental_success = sum(1 for r in incremental_result if r.get("success"))
        incremental_fail = len(incremental_result) - incremental_success
        
        print(f"✅ 成功: {incremental_success}, ❌ 失败: {incremental_fail}")
        
        # 5. 测试市场数据采集
        print("\n🌐 测试市场数据采集...")
        
        market_result = asyncio.run(collector.collect_market_data())
        
        print(f"\n📊 市场数据采集结果:")
        print(f"✅ 成功: {market_result.get('success', False)}")
        if market_result.get('success'):
            print(f"📋 记录数: {market_result.get('records_count', 0)}")
            print(f"📋 数据源ID: {market_result.get('source_id')}")
        else:
            print(f"❌ 错误: {market_result.get('error')}")
        
        print("\n" + "=" * 60)
        print("🎉 标准数据采集器测试完成！")
        print("✅ 所有测试项目执行完毕")
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    """
    运行测试
    """
    sys.exit(main())
