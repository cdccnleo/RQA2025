#!/usr/bin/env python3
"""
深度分析累计收益率走势准确性
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771323986'
result = load_backtest_result(backtest_id)

if not result:
    print('回测记录未找到')
    sys.exit(1)

print('=' * 80)
print('📊 累计收益率走势准确性深度分析报告')
print('=' * 80)
print()

# 基本信息
print('【基本信息】')
print(f"  回测ID: {result.get('backtest_id')}")
print(f"  日期范围: {result.get('start_date')} 至 {result.get('end_date')}")
print(f"  初始资金: ¥{result.get('initial_capital', 0):,.2f}")
print(f"  最终资金: ¥{result.get('final_capital', 0):,.2f}")
print(f"  总收益率: {result.get('total_return', 0)*100:.2f}%")
print()

# 资金曲线分析
print('【资金曲线分析】')
equity_curve = result.get('equity_curve', [])
print(f"  数据点数量: {len(equity_curve)}")
if equity_curve:
    print(f"  起始值: ¥{equity_curve[0]:,.2f}")
    print(f"  结束值: ¥{equity_curve[-1]:,.2f}")
    print(f"  资金变化: ¥{equity_curve[-1] - equity_curve[0]:,.2f}")
    
    # 分析资金曲线变化
    print()
    print('  资金曲线详细变化:')
    for i, value in enumerate(equity_curve):
        change = 0
        if i > 0:
            change = value - equity_curve[i-1]
        print(f"    点{i}: ¥{value:,.2f} (变化: {change:+,.2f})")
print()

# 交易记录分析
print('【交易记录分析】')
trades = result.get('trades', [])
print(f"  交易数量: {len(trades)}")
print()

# 按日期分组交易
trades_by_date = {}
for trade in trades:
    ts = trade.get('timestamp', '')
    if ts and len(str(ts)) >= 10:
        date = str(ts)[:10]
        if date not in trades_by_date:
            trades_by_date[date] = []
        trades_by_date[date].append(trade)

# 分析每日交易
print('  每日交易明细:')
total_pnl = 0
for date in sorted(trades_by_date.keys()):
    day_trades = trades_by_date[date]
    day_pnl = 0
    print(f"    {date}:")
    for trade in day_trades:
        symbol = trade.get('symbol', '')
        ttype = trade.get('type', '')
        price = trade.get('price', 0)
        quantity = trade.get('quantity', 0)
        pnl = trade.get('pnl', 0)
        
        if ttype in ['sell', 'sell_forced']:
            day_pnl += pnl
            print(f"      {symbol} {ttype}: 价格¥{price}, 数量{quantity}, 盈亏¥{pnl:,.2f}")
        else:
            print(f"      {symbol} {ttype}: 价格¥{price}, 数量{quantity}")
    
    total_pnl += day_pnl
    if day_pnl != 0:
        print(f"      当日盈亏: ¥{day_pnl:,.2f}")
    print()

print(f"  总盈亏: ¥{total_pnl:,.2f}")
print()

# 对比分析
print('【对比分析】')
if equity_curve:
    equity_pnl = equity_curve[-1] - equity_curve[0]
    print(f"  资金曲线盈亏: ¥{equity_pnl:,.2f}")
    print(f"  交易记录盈亏: ¥{total_pnl:,.2f}")
    
    if abs(equity_pnl - total_pnl) < 0.01:
        print(f"  ✅ 盈亏数据匹配")
    else:
        print(f"  ❌ 盈亏数据不匹配！差异: ¥{abs(equity_pnl - total_pnl):,.2f}")
        print(f"     可能原因: 手续费、滑点或其他费用未正确计算")
print()

# 收益率验证
print('【收益率验证】')
initial = result.get('initial_capital', 0)
final = result.get('final_capital', 0)
recorded_return = result.get('total_return', 0)

if initial > 0:
    calculated_return = (final - initial) / initial
    print(f"  计算的收益率: {calculated_return*100:.2f}%")
    print(f"  记录的收益率: {recorded_return*100:.2f}%")
    
    if abs(calculated_return - recorded_return) < 0.0001:
        print(f"  ✅ 收益率数据匹配")
    else:
        print(f"  ❌ 收益率数据不匹配！差异: {abs(calculated_return - recorded_return)*100:.4f}%")
print()

# 问题诊断
print('【问题诊断】')
issues = []

if equity_curve and len(equity_curve) > 0:
    # 检查资金曲线是否单调变化
    has_decrease = False
    has_increase = False
    for i in range(1, len(equity_curve)):
        if equity_curve[i] < equity_curve[i-1]:
            has_decrease = True
        if equity_curve[i] > equity_curve[i-1]:
            has_increase = True
    
    if not has_decrease:
        issues.append("资金曲线没有下降，可能未正确反映买入成本")
    if not has_increase:
        issues.append("资金曲线没有上升，可能未正确反映卖出盈亏")

if abs(equity_pnl - total_pnl) > 0.01:
    issues.append("资金曲线盈亏与交易记录盈亏不匹配")

if issues:
    print("  发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
else:
    print("  ✅ 未发现明显问题")

print()
print('=' * 80)
