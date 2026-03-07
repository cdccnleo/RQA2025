#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统快速测试
"""

import sqlite3
import time
from pathlib import Path

def test_monitoring():
    print("🔍 测试监控系统...")
    
    # 测试数据库
    try:
        with sqlite3.connect("data/monitoring.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM monitoring_data")
            count = cursor.fetchone()[0]
            print(f"✅ 监控数据: {count} 条记录")
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")
    
    # 测试脚本文件
    scripts = [
        "scripts/monitoring/enhanced_deployment_monitor.py",
        "scripts/monitoring/advanced_web_dashboard.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script}: 存在")
        else:
            print(f"❌ {script}: 不存在")
    
    print("✅ 测试完成")

if __name__ == "__main__":
    test_monitoring() 