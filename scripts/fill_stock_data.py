#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据填充脚本

功能：
- 填充akshare_stock_data表
- 支持默认股票和自定义股票列表
- 支持增量采集

作者: AI Assistant
创建日期: 2026-02-21
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.collectors.enhanced_akshare_collector import get_enhanced_akshare_collector
from src.data.collectors.data_collection_orchestrator import get_data_collection_orchestrator


def fill_default_stock_data(days_back: int = 30):
    """
    填充默认股票数据（000001 平安银行）
    
    Args:
        days_back: 采集多少天的历史数据
    """
    print("=" * 60)
    print("开始填充默认股票数据")
    print("=" * 60)
    
    # 获取增强版采集器
    collector = get_enhanced_akshare_collector()
    
    # 默认股票代码
    default_symbol = "000001"
    
    print(f"\n正在采集股票 {default_symbol} 的数据...")
    
    # 采集并保存数据（使用增量采集）
    success = collector.collect_and_save(
        symbol=default_symbol,
        use_incremental=True
    )
    
    if success:
        print(f"✓ 股票 {default_symbol} 数据采集成功")
    else:
        print(f"✗ 股票 {default_symbol} 数据采集失败")
    
    print("\n" + "=" * 60)
    print("数据填充完成")
    print("=" * 60)
    
    return success


def fill_multiple_stocks(symbols: list, days_back: int = 30):
    """
    填充多只股票数据
    
    Args:
        symbols: 股票代码列表
        days_back: 采集多少天的历史数据
    """
    print("=" * 60)
    print(f"开始填充 {len(symbols)} 只股票的数据")
    print("=" * 60)
    
    # 获取采集协调器
    orchestrator = get_data_collection_orchestrator()
    
    # 获取增强版采集器
    collector = get_enhanced_akshare_collector()
    
    # 注册采集器
    orchestrator.register_collector("enhanced_akshare", collector)
    
    # 调度采集任务
    task_ids = orchestrator.schedule_collection(
        symbols=symbols,
        collector_name="enhanced_akshare",
        frequency="daily"
    )
    
    print(f"\n已调度 {len(task_ids)} 个采集任务")
    
    # 执行所有任务
    results = orchestrator.execute_all_pending_tasks()
    
    # 统计结果
    success_count = sum(1 for result in results.values() if result)
    failed_count = len(results) - success_count
    
    print("\n" + "=" * 60)
    print("采集结果统计")
    print("=" * 60)
    print(f"总任务数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    
    # 显示详细状态
    status = orchestrator.get_status()
    print("\n采集器状态:")
    for name, collector_status in status["collectors"].items():
        print(f"  {name}:")
        print(f"    - 可用: {collector_status['is_available']}")
        print(f"    - 总采集次数: {collector_status['total_collections']}")
        print(f"    - 成功次数: {collector_status['successful_collections']}")
        print(f"    - 成功率: {collector_status['success_rate']:.2f}%")
    
    print("\n" + "=" * 60)
    
    return success_count == len(results)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="股票数据填充脚本")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="股票代码列表（如：000001 000002）"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="采集多少天的历史数据（默认：30）"
    )
    parser.add_argument(
        "--default",
        action="store_true",
        help="只填充默认股票（000001）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("RQA2025 股票数据填充工具")
    print("=" * 60 + "\n")
    
    if args.default:
        # 只填充默认股票
        success = fill_default_stock_data(args.days)
        sys.exit(0 if success else 1)
    
    elif args.symbols:
        # 填充指定股票
        success = fill_multiple_stocks(args.symbols, args.days)
        sys.exit(0 if success else 1)
    
    else:
        # 默认填充默认股票
        print("未指定股票代码，将填充默认股票（000001）")
        print("使用 --symbols 参数指定股票代码")
        print("使用 --default 参数明确填充默认股票\n")
        
        success = fill_default_stock_data(args.days)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
