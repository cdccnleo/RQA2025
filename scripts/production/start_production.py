#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境启动脚本
启动所有优化功能
"""


def start_production_system():
    """启动生产系统"""
    import os
    import sys

    # 设置Python路径
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/scripts')
    sys.path.insert(0, '/app/src')

    print("🚀 启动生产环境系统...")

    # 启动缓存优化
    print("📦 启动缓存优化...")
    try:
        from scripts.optimization.cache_optimization import main as cache_main
        cache_main()
        print("✅ 缓存优化启动成功")
    except Exception as e:
        print(f"❌ 缓存优化启动失败: {e}")

    # 启动监控告警
    print("📊 启动监控告警...")
    try:
        from scripts.optimization.monitoring_alert_system import main as monitoring_main
        monitoring_main()
        print("✅ 监控告警启动成功")
    except Exception as e:
        print(f"❌ 监控告警启动失败: {e}")

    # 启动性能基准测试
    print("⚡ 启动性能基准测试...")
    try:
        from scripts.optimization.performance_benchmark import main as benchmark_main
        benchmark_main()
        print("✅ 性能基准测试启动成功")
    except Exception as e:
        print(f"❌ 性能基准测试启动失败: {e}")

    print("🎉 生产环境系统启动完成!")


if __name__ == "__main__":
    start_production_system()
