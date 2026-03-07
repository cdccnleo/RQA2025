#!/usr/bin/env python3
"""
测试AKShare函数调用问题
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_akshare_daily():
    """测试AKShare日线数据获取"""
    try:
        import akshare as ak
        print(f"AKShare版本: {ak.__version__}")

        # 测试股票日线数据
        symbol = "688702"
        start_date = "20241221"  # 2024年12月21日
        end_date = "20250120"    # 2025年1月20日

        print(f"测试参数: symbol={symbol}, start_date={start_date}, end_date={end_date}")

        # 调用函数
        df = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        print(f"调用成功，返回类型: {type(df)}")
        if df is not None:
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            if not df.empty:
                print(f"前5行数据:")
                print(df.head())
            else:
                print("数据为空")
        else:
            print("返回None")

        return True

    except Exception as e:
        print(f"❌ 调用失败: {e}")
        traceback.print_exc()
        return False

def test_date_format():
    """测试日期格式问题"""
    from datetime import datetime, timedelta

    print("\n🔍 测试日期格式:")

    # 测试UTC时间
    end_date_obj = datetime.utcnow()
    start_date_obj = end_date_obj - timedelta(days=30)

    print(f"UTC end_date: {end_date_obj}")
    print(f"UTC start_date: {start_date_obj}")

    start_date_str = start_date_obj.strftime("%Y%m%d")
    end_date_str = end_date_obj.strftime("%Y%m%d")

    print(f"格式化后: {start_date_str} 到 {end_date_str}")

    # 检查年份
    current_year = end_date_obj.year
    print(f"当前年份: {current_year}")

    if current_year >= 2026:
        print("⚠️  警告: 系统时间可能不正确!")
    elif current_year < 2025:
        print("⚠️  警告: 系统时间可能过期!")

def main():
    """主函数"""
    print("🔧 测试AKShare函数调用")
    print("=" * 50)

    # 1. 测试日期格式
    test_date_format()

    # 2. 测试AKShare调用
    print("\n" + "=" * 30)
    print("测试AKShare stock_zh_a_daily:")
    success = test_akshare_daily()

    if success:
        print("\n✅ AKShare函数调用正常")
    else:
        print("\n❌ AKShare函数调用失败")

if __name__ == "__main__":
    main()