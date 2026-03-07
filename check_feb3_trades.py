#!/usr/bin/env python3
"""
检查2026/2/3两笔买入交易的资金情况
"""
import sys
sys.path.insert(0, '/app')
from src.gateway.web.backtest_persistence import load_backtest_result, list_backtest_results

# 获取最新的回测记录
results = list_backtest_results(limit=3)
if not results:
    print('没有回测记录')
    sys.exit(1)

backtest_id = results[0].get('backtest_id')
print(f"检查回测记录: {backtest_id}")
print()

result = load_backtest_result(backtest_id)
if not result:
    print('回测记录未找到')
    sys.exit(1)

# 基本信息
initial_capital = float(result.get('initial_capital', 100000))
print(f"初始资金: ¥{initial_capital:,.2f}")
print()

# 获取所有交易记录
trades = result.get('trades', [])

# 筛选2026/2/3的交易
feb3_trades = []
for trade in trades:
    ts = trade.get('timestamp', '')
    if ts and '2026-02-03' in str(ts):
        feb3_trades.append(trade)

print(f"2026/2/3交易数量: {len(feb3_trades)}")
print()

# 按时间排序
feb3_trades_sorted = sorted(feb3_trades, key=lambda x: x.get('timestamp', ''))

# 模拟资金计算
available_capital = initial_capital
print('【2026/2/3交易资金计算】')
print()

for i, trade in enumerate(feb3_trades_sorted):
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
            change = -cost
        else:
            available_capital -= trade_amount
            change = -trade_amount
    else:
        change = 0
    
    is_negative = available_capital < 0
    warning = " ⚠️ 资金不足！" if is_negative else ""
    
    print(f"交易{i+1}: {trade.get('timestamp', '')} {trade.get('symbol', '')} {trade_type}")
    print(f"  价格: ¥{price:,.2f}, 数量: {quantity}")
    print(f"  交易金额: ¥{trade_amount:,.2f}")
    print(f"  cost字段: ¥{cost:,.2f} (是否为总金额: {is_cost_total_amount})")
    print(f"  资金变化: ¥{change:+,.2f}")
    print(f"  交易前: ¥{capital_before:,.2f} → 交易后: ¥{available_capital:,.2f}{warning}")
    print()

print('【问题分析】')
print(f"初始资金: ¥{initial_capital:,.2f}")
print(f"两笔买入总金额: ¥{feb3_trades_sorted[0].get('cost', 0) + feb3_trades_sorted[1].get('cost', 0):,.2f}")
print()

if available_capital < 0:
    print("❌ 问题确认：2026/2/3的两笔买入交易总金额超过初始资金！")
    print()
    print("可能的原因：")
    print("1. 回测使用了杠杆或融资功能")
    print("2. 回测引擎允许透支买入")
    print("3. 交易记录数据有误")
    print()
    print("建议：")
    print("- 检查回测配置是否使用了杠杆")
    print("- 检查回测引擎的资金管理逻辑")
    print("- 如果允许透支，前端需要添加相应标识")
else:
    print("✅ 资金充足")
