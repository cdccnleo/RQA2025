#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771323648'
result = load_backtest_result(backtest_id)

if result:
    print('=== 修复验证报告 ===')
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
            print()
            
            # 验证 end_date 是否匹配最后交易日期
            end_date = result.get('end_date', '')
            last_trade_date = max(trade_dates)
            
            print(f"end_date 值: '{end_date}'")
            print(f"最后交易日期值: '{last_trade_date}'")
            print(f"类型: end_date={type(end_date)}, last_trade_date={type(last_trade_date)}")
            print(f"长度: end_date={len(end_date)}, last_trade_date={len(last_trade_date)}")
            print(f"是否相等: {end_date == last_trade_date}")
            print()
            
            if end_date == last_trade_date:
                print(f"✅ 修复成功！end_date ({end_date}) 与最后交易日期 ({last_trade_date}) 匹配")
            else:
                print(f"❌ 修复失败！end_date ({end_date}) 与最后交易日期 ({last_trade_date}) 不匹配")
        
        print(f"\n交易记录数: {len(trades)}")
        print('\n交易明细:')
        for t in trades:
            ts = t.get('timestamp', '')
            symbol = t.get('symbol', '')
            ttype = t.get('type', '')
            if ts and len(str(ts)) >= 10:
                print(f"  {str(ts)[:10]} - {symbol} - {ttype}")
    
    equity_curve = result.get('equity_curve', [])
    print(f"\n资金曲线数据点: {len(equity_curve)}")
else:
    print('回测记录未找到')
