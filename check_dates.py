#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771321359'
result = load_backtest_result(backtest_id)

if result:
    print('=== 回测日期检查 ===')
    print(f"回测ID: {result.get('backtest_id')}")
    print(f"开始日期: {result.get('start_date')}")
    print(f"结束日期: {result.get('end_date')}")
    print()
    
    trades = result.get('trades', [])
    if trades:
        trade_dates = []
        for t in trades:
            ts = t.get('timestamp', '')
            if ts and len(str(ts)) >= 10:
                trade_dates.append(str(ts)[:10])
        
        if trade_dates:
            print(f"交易日期范围: {min(trade_dates)} 至 {max(trade_dates)}")
        print(f"交易记录数: {len(trades)}")
        print()
        print('交易明细:')
        for t in trades:
            ts = t.get('timestamp', '')
            symbol = t.get('symbol', '')
            ttype = t.get('type', '')
            if ts and len(str(ts)) >= 10:
                print(f"  {str(ts)[:10]} - {symbol} - {ttype}")
    
    equity_curve = result.get('equity_curve', [])
    print()
    print(f"资金曲线数据点: {len(equity_curve)}")
    print(f"资金曲线值: {equity_curve}")
else:
    print('回测记录未找到')
