"""
RQA2025基础使用示例
"""
from src.utils.convert import DataConverter
from src.utils.performance import PerformanceUtils
from src.utils.exception import ExceptionHandler
import pandas as pd

def main():
    """主示例函数"""
    print("=== RQA2025量化系统基础使用示例 ===")

    # 1. 数据转换示例
    print("\n1. 数据转换工具演示:")
    prev_close = 10.0
    limits = DataConverter.calculate_limit_prices(prev_close)
    print(f"前收盘价: {prev_close}, 涨停价: {limits['upper_limit']}, 跌停价: {limits['lower_limit']}")

    # 2. 性能分析示例
    print("\n2. 性能分析工具演示:")
    trades = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'symbol': ['600000.SH'] * 5,
        'quantity': [1000, -500, -500, 800, -800],
        'price': [10.0, 11.0, 12.0, 13.0, 14.0],
        'action': ['buy', 'sell', 'sell', 'buy', 'sell']
    })
    performance = PerformanceUtils.calculate_returns(trades, 100000)
    print("累计收益率:", performance['cum_return'].iloc[-1])

    # 3. 异常处理示例
    print("\n3. 异常处理工具演示:")
    try:
        raise LimitViolation(
            symbol='600000.SH',
            price=11.1,
            limit=11.0,
            is_upper=True
        )
    except LimitViolation as e:
        print(f"捕获异常: {e}")
        adjusted = ExceptionHandler.handle(e, {
            'order': {
                'symbol': '600000.SH',
                'price': 11.1,
                'quantity': 1000
            }
        })
        print("调整后订单价格:", adjusted['price'])

if __name__ == "__main__":
    main()
