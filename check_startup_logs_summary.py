#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查启动日志总结

分析最新的启动日志，统计重复初始化情况
"""

import subprocess
import sys
from collections import Counter

def analyze_startup_logs():
    """分析启动日志"""
    print("=" * 60)
    print("启动日志重复初始化检查")
    print("=" * 60)
    
    try:
        # 获取最新的启动日志（从LIFESPAN开始）
        result = subprocess.run(
            ['docker', 'logs', 'rqa2025-rqa2025-app-1'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ 获取日志失败: {result.stderr}")
            return False
        
        logs = result.stdout
        
        # 查找最新的LIFESPAN开始位置
        lines = logs.split('\n')
        lifespan_indices = [i for i, line in enumerate(lines) if 'LIFESPAN 函数开始执行' in line]
        
        if not lifespan_indices:
            print("⚠️  未找到LIFESPAN函数执行记录")
            return False
        
        # 获取最后一次启动的日志
        last_startup_index = lifespan_indices[-1]
        startup_logs = '\n'.join(lines[last_startup_index:last_startup_index+200])
        
        # 统计关键消息
        patterns = {
            'features层基础设施服务初始化完成': 'features层基础设施服务初始化完成',
            'UnifiedCacheManager initialized': 'UnifiedCacheManager initialized',
            '基础设施服务初始化完成': '基础设施服务初始化完成',
            '组件生命周期管理器初始化完成': '组件生命周期管理器初始化完成',
            '启动RQA2025基础设施层连续监控': '启动RQA2025基础设施层连续监控',
            '开始初始化EventBus': '开始初始化EventBus'
        }
        
        counts = {}
        for key, pattern in patterns.items():
            count = startup_logs.count(pattern)
            counts[key] = count
        
        # 输出结果
        print("\n📊 最新启动日志统计（从LIFESPAN开始）:")
        print("-" * 60)
        
        all_good = True
        for key, count in counts.items():
            if count == 0:
                status = "✅"
            elif count == 1:
                status = "✅"
            else:
                status = "❌"
                all_good = False
            
            print(f"{status} {key}: {count} 次")
        
        print("\n" + "=" * 60)
        
        if all_good:
            print("✅ 所有关键初始化消息都在合理范围内（0-1次）")
            return True
        else:
            print("❌ 发现重复初始化问题")
            return False
        
    except subprocess.TimeoutExpired:
        print("❌ 日志分析超时")
        return False
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_startup_logs()
    sys.exit(0 if success else 1)