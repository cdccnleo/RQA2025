#!/usr/bin/env python3
"""
检查资金曲线和交易记录中的负数问题
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result

backtest_id = 'backtest_model_model_job_1771237605_20260216_182646_1771328506'
result = load_backtest_result(backtest_id)

if not result:
    print('回测记录未找到')
    sys.exit(1)

print('=' * 80)
print('🔍 可用资金负数问题深度检查报告')
print('=' * 80)
print()

# 基本信息
print('【回测基本信息】')
print(f"  回测ID: {result.get('backtest_id')}")
print(f"  初始资金: ¥{float(result.get('initial_capital', 0)):,.2f}")
print(f"  最终资金: ¥{float(result.get('final_capital', 0)):,.2f}")
print(f"  总收益率: {float(result.get('total_return', 0))*100:.2f}%")
print()

# 检查回测配置（杠杆、融资等）
print('【回测配置检查】')
leverage = result.get('leverage', 1)
margin_enabled = result.get('margin_enabled', False)
initial_capital = float(result.get('initial_capital', 100000))

print(f"  杠杆倍数: {leverage}")
print(f"  融资功能: {'启用' if margin_enabled else '未启用'}")
print(f"  初始资金: ¥{initial_capital:,.2f}")

# 如果使用了杠杆，计算最大可用资金
if leverage > 1:
    max_position = initial_capital * leverage
    print(f"  杠杆后最大仓位: ¥{max_position:,.2f}")
    print(f"  说明: 使用了{leverage}倍杠杆，允许资金为负")
print()

# 检查资金曲线数据
equity_curve = result.get('equity_curve', [])
print('【资金曲线检查】')
print(f"  数据点数量: {len(equity_curve)}")

negative_points = []
for i, value in enumerate(equity_curve):
    if float(value) < 0:
        negative_points.append((i, float(value)))

if negative_points:
    print(f"  ⚠️ 发现 {len(negative_points)} 个负数点:")
    for idx, val in negative_points[:5]:  # 只显示前5个
        print(f"    点{idx}: ¥{val:,.2f}")
else:
    print(f"  ✅ 资金曲线无负数")
print()

# 检查交易记录
trades = result.get('trades', [])
print('【交易记录检查】')
print(f"  交易数量: {len(trades)}")
print()

# 按日期分组检查
from collections import defaultdict
trades_by_date = defaultdict(list)

for trade in trades:
    ts = trade.get('timestamp', '')
    if ts and len(str(ts)) >= 10:
        date = str(ts)[:10]
        trades_by_date[date].append(trade)

print('  每日交易明细:')
for date in sorted(trades_by_date.keys()):
    day_trades = trades_by_date[date]
    print(f"    {date}: {len(day_trades)}笔交易")
    
    # 按时间排序
    day_trades_sorted = sorted(day_trades, key=lambda x: x.get('timestamp', ''))
    
    for i, trade in enumerate(day_trades_sorted):
        ts = trade.get('timestamp', '')
        time_part = ts[11:19] if len(str(ts)) >= 19 else '未知时间'
        print(f"      {time_part} - {trade.get('symbol', '')} - {trade.get('type', '')}")
print()

# 模拟资金计算，检查负数出现的位置
print('【资金计算模拟】')
available_capital = initial_capital
print(f"  初始资金: ¥{available_capital:,.2f}")
print()

# 按时间排序所有交易
all_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))

for i, trade in enumerate(all_trades):
    trade_type = trade.get('type', '')
    price = float(trade.get('price', 0))
    quantity = float(trade.get('quantity', 0))
    cost = float(trade.get('cost') or 0)
    
    capital_before = available_capital
    trade_amount = price * quantity
    
    # 判断cost字段是手续费还是买入总金额
    is_cost_total_amount = abs(cost - trade_amount) < trade_amount * 0.1
    
    if trade_type in ['buy', 'buy_forced']:
        if is_cost_total_amount and cost > 0:
            available_capital -= cost
        else:
            available_capital -= trade_amount
    elif trade_type in ['sell', 'sell_forced']:
        fee = cost if cost < trade_amount * 0.5 else 0
        available_capital += (trade_amount - fee)
    
    # 检查是否为负数
    is_negative = available_capital < 0
    warning = " ⚠️ 负数！" if is_negative else ""
    
    print(f"  交易{i+1}: {trade.get('timestamp', '')[:19]} {trade.get('symbol', '')} {trade_type}")
    print(f"    交易前: ¥{capital_before:,.2f} → 交易后: ¥{available_capital:,.2f}{warning}")
    
    if is_negative and leverage <= 1:
        print(f"    ❌ 错误：普通交易不应出现负数资金！")
        print(f"       可能原因：交易顺序错误或数据问题")

print()
print('【结论】')
if leverage > 1:
    print(f"✅ 使用了{leverage}倍杠杆，允许资金为负，这是正常的")
elif negative_points:
    print(f"❌ 发现资金负数问题！")
    print(f"   建议：检查交易顺序或回测逻辑")
else:
    print(f"✅ 未发现资金负数问题")
