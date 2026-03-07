#!/usr/bin/env python3
"""
最终验证资金计算逻辑
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
print('💰 账号可用资金计算最终验证报告')
print('=' * 80)
print()

# 基本信息
initial_capital = float(result.get('initial_capital', 100000))
final_capital = float(result.get('final_capital', 0))

print(f"初始资金: ¥{initial_capital:,.2f}")
print(f"记录的最终资金: ¥{final_capital:,.2f}")
print()

# 获取交易记录
trades = result.get('trades', [])
print(f"交易记录数: {len(trades)}")
print()

# 按时间排序
trades_sorted = sorted(trades, key=lambda x: x.get('timestamp', ''))

# 模拟前端资金计算逻辑
available_capital = initial_capital
print('【逐笔交易资金计算（使用修复后的逻辑）】')
print()

for i, trade in enumerate(trades_sorted):
    trade_type = trade.get('type', '')
    price = float(trade.get('price', 0))
    quantity = float(trade.get('quantity', 0))
    cost = float(trade.get('cost') or 0)
    pnl = float(trade.get('pnl', 0))
    
    capital_before = available_capital
    trade_amount = price * quantity
    
    # 判断cost字段是手续费还是买入总金额
    is_cost_total_amount = abs(cost - trade_amount) < trade_amount * 0.1
    
    if trade_type in ['buy', 'buy_forced']:
        # 买入交易
        if is_cost_total_amount and cost > 0:
            # cost是买入总金额
            available_capital -= cost
            change = -cost
            calculation_method = f"cost字段(买入总金额) ¥{cost:,.2f}"
        else:
            # cost是手续费或为空
            available_capital -= trade_amount
            change = -trade_amount
            calculation_method = f"交易金额 ¥{trade_amount:,.2f}"
    elif trade_type in ['sell', 'sell_forced']:
        # 卖出交易
        fee = cost if cost < trade_amount * 0.5 else 0
        available_capital += (trade_amount - fee)
        change = trade_amount - fee
        calculation_method = f"交易金额 ¥{trade_amount:,.2f} - 手续费 ¥{fee:,.2f}"
    else:
        change = 0
        calculation_method = "未知交易类型"
    
    print(f"交易{i+1}: {trade.get('timestamp', '')[:10]} {trade.get('symbol', '')} {trade_type}")
    print(f"  价格: ¥{price:,.2f}, 数量: {quantity}, 交易金额: ¥{trade_amount:,.2f}")
    print(f"  cost字段: ¥{cost:,.2f}, 是否识别为总金额: {is_cost_total_amount}")
    print(f"  计算方式: {calculation_method}")
    print(f"  资金变化: ¥{change:+,.2f}")
    print(f"  交易前: ¥{capital_before:,.2f} → 交易后: ¥{available_capital:,.2f}")
    print()

print('【验证结果】')
print(f"计算的最终资金: ¥{available_capital:,.2f}")
print(f"记录的最终资金: ¥{final_capital:,.2f}")

difference = abs(available_capital - final_capital)
if difference < 0.01:
    print(f"✅ 资金计算正确！差异: ¥{difference:,.2f}")
    print()
    print('结论: 账号可用资金计算逻辑正确，修复成功！')
else:
    print(f"❌ 资金计算有误！差异: ¥{difference:,.2f}")
    print()
    print('可能的原因:')
    print('1. cost字段识别逻辑仍需调整')
    print('2. 卖出交易的手续费处理不正确')
    print('3. 交易记录数据不完整')
