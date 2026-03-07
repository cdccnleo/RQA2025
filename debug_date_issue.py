#!/usr/bin/env python3
"""
调试数据采集日期问题
"""

import sys
import os
from datetime import datetime, timedelta

def debug_date_calculation():
    """调试日期计算问题"""
    print("🔍 调试日期计算问题")
    print("=" * 50)

    # 1. 检查当前时间
    print("\n1. 当前时间检查:")
    now = datetime.now()
    print(f"datetime.now(): {now}")
    print(f"datetime.now() strftime('%Y%m%d'): {now.strftime('%Y%m%d')}")

    # 2. 模拟数据采集的日期计算逻辑
    print("\n2. 模拟数据采集日期计算:")
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - timedelta(days=30)

    print(f"end_date_obj: {end_date_obj}")
    print(f"start_date_obj: {start_date_obj}")

    start_date_dt = start_date_obj
    end_date_dt = end_date_obj

    print(f"start_date_dt: {start_date_dt}")
    print(f"end_date_dt: {end_date_dt}")

    # 3. 格式化日期字符串
    print("\n3. 日期格式化:")
    start_date_str = start_date_dt.strftime("%Y%m%d")
    end_date_str = end_date_dt.strftime("%Y%m%d")

    print(f"start_date_str: {start_date_str}")
    print(f"end_date_str: {end_date_str}")

    # 4. 检查时区
    print("\n4. 时区检查:")
    try:
        import time
        print(f"time.tzname: {time.tzname}")
        print(f"time.timezone: {time.timezone}")
    except:
        print("时区信息不可用")

    # 5. 检查系统时间
    print("\n5. 系统时间:")
    import time
    print(f"time.time(): {time.time()}")
    print(f"time.ctime(): {time.ctime()}")

    # 6. 手动设置正确的日期范围
    print("\n6. 手动设置正确的日期范围:")
    # 应该是2024年12月21日到2025年1月20日
    correct_start = datetime(2024, 12, 21)
    correct_end = datetime(2025, 1, 20)

    print(f"正确start_date: {correct_start.strftime('%Y%m%d')}")
    print(f"正确end_date: {correct_end.strftime('%Y%m%d')}")

    # 7. 分析问题
    print("\n7. 问题分析:")
    current_year = now.year
    if current_year == 2026:
        print("⚠️  检测到系统年份为2026年，这可能是问题所在！")
        print("容器中的系统时间可能设置不正确。")
    else:
        print(f"系统年份为{current_year}，可能不是时区问题。")

    print("\n从日志看，日期范围是 20251221 到 20260120")
    print("这意味着：")
    print("- start_date 被设置为 2025年12月21日")
    print("- end_date 被设置为 2026年01月20日")
    print("这不符合30天的数据采集预期。")

if __name__ == "__main__":
    debug_date_calculation()