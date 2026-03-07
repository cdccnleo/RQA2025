#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行数据清理操作

删除不符合要求的历史数据，确保数据质量
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.maintenance.data_cleanup import get_data_cleanup_service

def main():
    """
    主函数
    """
    print("🚀 启动数据清理操作...")
    print("=" * 60)
    
    try:
        # 获取数据清理服务实例
        cleanup_service = get_data_cleanup_service()
        print("✅ 数据清理服务初始化成功")
        
        # 1. 生成清理前的报告
        print("\n📊 生成清理前的数据报告...")
        before_report = cleanup_service.generate_cleanup_report()
        print(f"📋 清理前 - 记录数: {before_report['table_stats'].get('akshare_stock_data', {}).get('record_count', 0)}")
        print(f"📦 清理前 - 表大小: {before_report['table_stats'].get('akshare_stock_data', {}).get('size', {}).get('total', 'N/A')}")
        
        # 2. 执行数据清理
        print("\n🧹 执行数据清理...")
        print("目标: 删除 source_id='historical_collection_688702' 的数据")
        
        cleanup_result = cleanup_service.clean_invalid_data(
            source_id="historical_collection_688702"
        )
        
        print(f"\n📊 清理结果:")
        print(f"✅ 成功: {cleanup_result['success']}")
        print(f"🗑️  删除记录数: {cleanup_result['deleted_count']}")
        print(f"⏱️  执行时间: {cleanup_result['duration']:.2f}秒")
        
        if not cleanup_result['success']:
            print(f"❌ 错误: {cleanup_result['error']}")
            return 1
        
        # 3. 生成清理后的报告
        print("\n📊 生成清理后的数据报告...")
        after_report = cleanup_service.generate_cleanup_report()
        print(f"📋 清理后 - 记录数: {after_report['table_stats'].get('akshare_stock_data', {}).get('record_count', 0)}")
        print(f"📦 清理后 - 表大小: {after_report['table_stats'].get('akshare_stock_data', {}).get('size', {}).get('total', 'N/A')}")
        
        # 4. 验证数据完整性
        print("\n🔍 验证数据完整性...")
        integrity_result = cleanup_service.validate_data_integrity()
        
        if integrity_result['success']:
            validation = integrity_result['validation_result']
            print(f"✅ 数据完整性验证通过")
            print(f"📋 总记录数: {validation.get('total_records', 0)}")
            print(f"⚠️  无效记录数: {validation.get('invalid_records', 0)}")
            
            # 显示 source_id 分布
            print("\n📊 Source ID 分布:")
            source_dist = validation.get('source_id_distribution', {})
            for source_id, count in source_dist.items():
                print(f"  - {source_id}: {count} 条")
        else:
            print(f"❌ 数据完整性验证失败: {integrity_result['error']}")
            return 1
        
        # 5. 优化表空间
        print("\n⚡ 优化表空间...")
        optimize_result = cleanup_service.optimize_table_space()
        print(f"✅ 优化成功: {optimize_result['success']}")
        print(f"📋 优化的表: {optimize_result['optimized_tables']}")
        print(f"⏱️  执行时间: {optimize_result['duration']:.2f}秒")
        
        print("\n" + "=" * 60)
        print("🎉 数据清理操作完成！")
        print(f"✅ 成功删除 {cleanup_result['deleted_count']} 条不符合要求的记录")
        print("✅ 数据质量和完整性已得到保障")
        print("✅ 表空间已优化")
        return 0
        
    except Exception as e:
        print(f"\n❌ 数据清理操作失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    """
    运行数据清理操作
    """
    sys.exit(main())
