#!/usr/bin/env python3
"""
检查前端接收的数据结构
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result
from datetime import datetime, timedelta

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771323648'
result = load_backtest_result(backtest_id)

if result:
    print('=== 前端数据流向检查 ===')
    print()
    
    # 检查关键字段
    print('【关键字段检查】')
    print(f"  backtest_id: {result.get('backtest_id')}")
    print(f"  start_date: {result.get('start_date')}")
    print(f"  end_date: {result.get('end_date')}")
    print(f"  initial_capital: {result.get('initial_capital')}")
    print(f"  final_capital: {result.get('final_capital')}")
    print()
    
    # 检查资金曲线
    equity_curve = result.get('equity_curve', [])
    print(f"【资金曲线】")
    print(f"  数据点数量: {len(equity_curve)}")
    print(f"  起始值: {equity_curve[0] if equity_curve else 'N/A'}")
    print(f"  结束值: {equity_curve[-1] if equity_curve else 'N/A'}")
    print()
    
    # 检查交易记录
    trades = result.get('trades', [])
    print(f"【交易记录】")
    print(f"  交易数量: {len(trades)}")
    
    trade_dates = []
    for t in trades:
        ts = t.get('timestamp', '')
        if ts and len(str(ts)) >= 10:
            trade_dates.append(str(ts)[:10])
    
    if trade_dates:
        print(f"  交易日期范围: {min(trade_dates)} 至 {max(trade_dates)}")
        print()
        
        # 模拟前端标签生成逻辑
        start_date = result.get('start_date', '')
        end_date = result.get('end_date', '')
        data_length = len(equity_curve)
        
        print('【前端标签生成模拟】')
        print(f"  使用的start_date: {start_date}")
        print(f"  使用的end_date: {end_date}")
        print(f"  数据点数量: {data_length}")
        print()
        
        # 模拟前端的标签生成算法
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days_diff = max(1, (end - start).days + 1)
        trading_days = int(days_diff * 0.7)
        step = max(1, trading_days // data_length)
        
        print(f"  计算的总天数: {days_diff}")
        print(f"  估算的交易日: {trading_days}")
        print(f"  计算的步长: {step}")
        print()
        
        # 生成标签
        labels = []
        current_date = start
        for i in range(data_length):
            # 跳过周末
            while current_date.weekday() >= 5:  # 5=周六, 6=周日
                current_date += timedelta(days=1)
            
            labels.append(current_date.strftime('%m-%d'))
            current_date += timedelta(days=step)
        
        print(f"  生成的标签数量: {len(labels)}")
        print(f"  第一个标签: {labels[0]}")
        print(f"  最后一个标签: {labels[-1]}")
        print()
        
        # 对比实际交易日期
        print('【对比分析】')
        print(f"  实际交易开始日期: {min(trade_dates)}")
        print(f"  实际交易结束日期: {max(trade_dates)}")
        print(f"  标签起始日期: {start_date[:4]}-{labels[0]}")
        print(f"  标签结束日期: {end_date[:4]}-{labels[-1]}")
        print()
        
        # 检查问题
        if labels[-1] != max(trade_dates)[5:]:
            print(f"  ❌ 问题发现：标签结束日期 ({labels[-1]}) 与实际交易结束日期 ({max(trade_dates)[5:]}) 不匹配")
        else:
            print(f"  ✅ 标签日期与实际交易日期匹配")
else:
    print('回测记录未找到')
