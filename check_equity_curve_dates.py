#!/usr/bin/env python3
"""
检查资金曲线数据与交易记录日期是否匹配
"""
import sys
sys.path.insert(0, '/app')

from src.gateway.web.backtest_persistence import load_backtest_result
from datetime import datetime

def check_equity_curve_dates():
    """检查资金曲线日期"""
    print("=" * 60)
    print("📊 资金曲线日期检查报告")
    print("=" * 60)
    print()
    
    # 获取最新的回测记录
    backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771321359'
    result = load_backtest_result(backtest_id)
    
    if not result:
        print("❌ 回测记录未找到")
        return
    
    print(f"回测ID: {backtest_id}")
    print()
    
    # 基本信息
    print("【基本信息】")
    print(f"  开始日期: {result.get('start_date')}")
    print(f"  结束日期: {result.get('end_date')}")
    print()
    
    # 交易记录日期
    print("【交易记录日期】")
    trades = result.get('trades', [])
    if trades:
        trade_dates = []
        for trade in trades:
            timestamp = trade.get('timestamp', '')
            if timestamp:
                # 提取日期部分
                date_str = timestamp[:10] if isinstance(timestamp, str) else str(timestamp)[:10]
                trade_dates.append(date_str)
                print(f"  {date_str} - {trade.get('symbol')} - {trade.get('type')}")
        
        if trade_dates:
            print()
            print(f"  最早交易日期: {min(trade_dates)}")
            print(f"  最晚交易日期: {max(trade_dates)}")
    else:
        print("  无交易记录")
    
    print()
    
    # 资金曲线数据
    print("【资金曲线数据】")
    equity_curve = result.get('equity_curve', [])
    print(f"  数据点数量: {len(equity_curve)}")
    print(f"  起始值: {equity_curve[0] if equity_curve else 'N/A'}")
    print(f"  结束值: {equity_curve[-1] if equity_curve else 'N/A'}")
    
    if len(equity_curve) > 0:
        print()
        print("  资金曲线所有值:")
        for i, val in enumerate(equity_curve):
            print(f"    点{i}: {val}")
    
    print()
    
    # 问题分析
    print("【问题分析】")
    start_date = result.get('start_date', '')
    end_date = result.get('end_date', '')
    
    # 计算预期交易日数量
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end - start).days + 1
            
            print(f"  回测日期范围: {start_date} 至 {end_date}")
            print(f"  总天数: {days_diff}")
            print(f"  资金曲线数据点: {len(equity_curve)}")
            
            # 检查是否有强制平仓
            forced_trades = [t for t in trades if t.get('type') == 'sell_forced']
            if forced_trades:
                print()
                print(f"  强制平仓交易:")
                for t in forced_trades:
                    print(f"    {t.get('timestamp')} - {t.get('symbol')} - 盈亏: {t.get('pnl')}")
            
            # 检查数据点是否过多
            if len(equity_curve) > days_diff:
                print()
                print(f"  ⚠️ 资金曲线数据点({len(equity_curve)})多于总天数({days_diff})")
                print(f"     可能原因: 每个交易日生成了多个数据点")
            
        except Exception as e:
            print(f"  日期解析错误: {e}")

if __name__ == "__main__":
    check_equity_curve_dates()
