#!/usr/bin/env python3
"""
验证回测资金曲线数据准确性
"""
import sys
sys.path.insert(0, '/app')

from src.gateway.web.backtest_persistence import load_backtest_result

def verify_equity_curve(backtest_id):
    """验证资金曲线数据"""
    result = load_backtest_result(backtest_id)
    
    if not result:
        print("❌ 回测记录未找到")
        return
    
    print("=" * 60)
    print("📊 回测资金曲线数据验证报告")
    print("=" * 60)
    print()
    
    # 基本信息
    print("【基本信息】")
    print(f"  回测ID: {result.get('backtest_id')}")
    initial_capital = result.get('initial_capital', 0)
    final_capital = result.get('final_capital', 0)
    print(f"  初始资金: ¥{initial_capital:,.2f}")
    print(f"  最终资金: ¥{final_capital:,.2f}")
    print(f"  记录的总收益率: {result.get('total_return', 0)*100:.2f}%")
    print()
    
    # 资金曲线
    print("【资金曲线】")
    equity_curve = result.get('equity_curve', [])
    print(f"  数据点数: {len(equity_curve)}")
    
    if equity_curve:
        print(f"  起始: ¥{equity_curve[0]:,.2f}")
        print(f"  结束: ¥{equity_curve[-1]:,.2f}")
        calculated_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve[0] > 0 else 0
        print(f"  计算收益率: {calculated_return*100:.2f}%")
    print()
    
    # 交易记录
    print("【交易记录】")
    trades = result.get('trades', [])
    print(f"  交易次数: {len(trades)}")
    
    # 计算实际盈亏
    total_pnl = 0
    sell_count = 0
    
    for trade in trades:
        trade_type = trade.get('type')
        if trade_type in ['sell', 'sell_forced']:
            pnl = trade.get('pnl', 0)
            total_pnl += pnl
            sell_count += 1
            print(f"  {trade.get('timestamp')} {trade.get('symbol')} {trade_type}: 盈亏=¥{pnl:,.2f}")
    
    print()
    print("【盈亏核对】")
    print(f"  卖出交易次数: {sell_count}")
    print(f"  交易盈亏总和: ¥{total_pnl:,.2f}")
    
    if equity_curve and len(equity_curve) > 1:
        equity_pnl = equity_curve[-1] - equity_curve[0]
        print(f"  资金曲线盈亏: ¥{equity_pnl:,.2f}")
        
        # 检查是否匹配
        if abs(total_pnl - equity_pnl) < 0.01:
            print("  ✅ 盈亏数据匹配")
        else:
            print(f"  ❌ 盈亏数据不匹配！差异: ¥{abs(total_pnl - equity_pnl):,.2f}")
            print(f"     可能原因: 手续费、滑点或其他费用未正确计算")
    
    # 检查总收益率计算
    if initial_capital > 0:
        calculated_total_return = (final_capital - initial_capital) / initial_capital
        recorded_return = result.get('total_return', 0)
        print()
        print("【收益率核对】")
        print(f"  计算的收益率: {calculated_total_return*100:.2f}%")
        print(f"  记录的收益率: {recorded_return*100:.2f}%")
        
        if abs(calculated_total_return - recorded_return) < 0.0001:
            print("  ✅ 收益率数据匹配")
        else:
            print(f"  ❌ 收益率数据不匹配！差异: {abs(calculated_total_return - recorded_return)*100:.4f}%")

if __name__ == "__main__":
    backtest_id = "backtest_model_model_job_1771237605_20260216_182646_1771253231"
    verify_equity_curve(backtest_id)
