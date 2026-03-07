#!/usr/bin/env python3
"""
分析回测引擎资金问题
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
final_capital = float(result.get('final_capital', 0))

print(f"初始资金: ¥{initial_capital:,.2f}")
print(f"最终资金: ¥{final_capital:,.2f}")
print(f"总收益率: {float(result.get('total_return', 0))*100:.2f}%")
print()

# 获取交易记录
trades = result.get('trades', [])

# 筛选2026/2/3的交易
feb3_trades = []
for trade in trades:
    ts = trade.get('timestamp', '')
    if ts and '2026-02-03' in str(ts):
        feb3_trades.append(trade)

print(f"2026/2/3交易数量: {len(feb3_trades)}")
print()

# 分析买入交易
buy_trades = [t for t in feb3_trades if t.get('type') in ['buy', 'buy_forced']]
sell_trades = [t for t in feb3_trades if t.get('type') in ['sell', 'sell_forced']]

print(f"买入交易: {len(buy_trades)}笔")
print(f"卖出交易: {len(sell_trades)}笔")
print()

# 计算买入总金额
buy_total = 0
for trade in buy_trades:
    cost = float(trade.get('cost') or 0)
    price = float(trade.get('price', 0))
    quantity = float(trade.get('quantity', 0))
    trade_amount = price * quantity
    
    # 判断cost字段
    is_cost_total = abs(cost - trade_amount) < trade_amount * 0.1
    
    if is_cost_total and cost > 0:
        buy_total += cost
    else:
        buy_total += trade_amount
    
    print(f"  买入: {trade.get('symbol')} - 价格¥{price:,.2f} × 数量{quantity} = ¥{trade_amount:,.2f}")
    print(f"         cost字段: ¥{cost:,.2f} (是否为总金额: {is_cost_total})")

print()
print(f"买入总金额: ¥{buy_total:,.2f}")
print(f"初始资金: ¥{initial_capital:,.2f}")
print(f"差额: ¥{buy_total - initial_capital:,.2f}")
print()

if buy_total > initial_capital:
    print("❌ 问题确认：买入总金额超过初始资金！")
    print()
    print("问题分析：")
    print("1. 回测引擎在2026/2/3同时发出了两个买入信号")
    print("2. 第一笔买入使用了大部分资金")
    print("3. 第二笔买入时，资金应该不足，但回测引擎还是执行了")
    print()
    print("可能的原因：")
    print("- 回测引擎同时处理多个股票的买入信号，没有考虑资金限制")
    print("- 回测引擎的资金检查逻辑有缺陷")
    print("- 策略配置允许透支或使用了杠杆")
    print()
    print("建议修复：")
    print("- 在回测引擎中添加全局资金限制检查")
    print("- 确保所有买入请求的总额不超过可用资金")
else:
    print("✅ 买入总金额在初始资金范围内")
