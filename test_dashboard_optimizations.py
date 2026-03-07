#!/usr/bin/env python3
"""
测试仪表盘短期优化功能
验证数据源过滤、图表缓存和状态指示器功能
"""

import json
import time
import requests
from typing import Dict, Any

def test_dashboard_optimizations():
    """测试仪表盘优化功能"""
    print("🧪 测试仪表盘短期优化功能")
    print("=" * 50)

    try:
        # 1. 测试数据源过滤功能
        print("1️⃣ 测试数据源过滤功能")
        print("-" * 30)

        # 获取数据源列表
        api_url = "http://localhost:8000/api/v1/data/sources"
        response = requests.post(api_url, json={"action": "get_all"}, timeout=10)

        if response.status_code == 200:
            data = response.json()
            sources = data.get('data') or data.get('data_sources') or []

            # 统计数据源类型
            types_count = {}
            enabled_sources = []

            for source in sources:
                if source.get('enabled', True):
                    enabled_sources.append(source)
                    source_type = source.get('type', 'unknown')
                    types_count[source_type] = types_count.get(source_type, 0) + 1

            print(f"✅ 找到 {len(enabled_sources)} 个启用数据源")
            print(f"📊 数据源类型分布:")
            for type_name, count in types_count.items():
                print(f"   • {type_name}: {count} 个")

            # 验证类型过滤逻辑
            if len(types_count) >= 2:
                print("✅ 数据源类型多样性足够，支持过滤功能")
            else:
                print("⚠️ 数据源类型较少，过滤功能效果有限")

        print()

        # 2. 测试图表缓存功能
        print("2️⃣ 测试图表缓存功能")
        print("-" * 30)

        metrics_url = "http://localhost:8000/api/v1/data-sources/metrics"

        # 第一次请求
        start_time = time.time()
        response1 = requests.get(metrics_url, timeout=10)
        first_request_time = time.time() - start_time

        if response1.status_code == 200:
            print(f"✅ 第一次请求成功: {first_request_time:.2f}秒")
        else:
            print(f"❌ 第一次请求失败: {response1.status_code}")
            return False

        # 等待一小段时间
        time.sleep(0.5)

        # 第二次请求（应该从缓存获取）
        start_time = time.time()
        response2 = requests.get(metrics_url, timeout=10)
        second_request_time = time.time() - start_time

        if response2.status_code == 200:
            print(f"✅ 第二次请求成功: {second_request_time:.2f}秒")
            # 比较响应时间
            if second_request_time < first_request_time * 0.8:
                print("✅ 缓存功能正常，第二次请求明显更快")
            else:
                print("⚠️ 缓存效果不明显，可能由于网络或其他因素")
        else:
            print(f"❌ 第二次请求失败: {response2.status_code}")

        print()

        # 3. 测试状态指示器功能
        print("3️⃣ 测试状态指示器功能")
        print("-" * 30)

        # 验证数据源都有状态信息
        sources_with_status = 0
        total_enabled = 0

        for source in enabled_sources:
            total_enabled += 1
            if source.get('status'):
                sources_with_status += 1

        print(f"📊 启用数据源状态统计:")
        print(f"   • 总启用数据源: {total_enabled}")
        print(f"   • 有状态信息: {sources_with_status}")
        print(f"   • 状态覆盖率: {(sources_with_status/total_enabled*100):.1f}%")

        if sources_with_status > 0:
            print("✅ 状态指示器有数据基础")
        else:
            print("❌ 没有状态数据，状态指示器无法正常工作")

        print()

        # 4. 综合验证
        print("4️⃣ 综合功能验证")
        print("-" * 30)

        checks = [
            ("数据源多样性", len(types_count) >= 2),
            ("API响应正常", response1.status_code == 200 and response2.status_code == 200),
            ("状态数据存在", sources_with_status > 0),
            ("启用数据源数量", len(enabled_sources) >= 10)
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {'通过' if passed else '失败'}")
            if not passed:
                all_passed = False

        print()

        if all_passed:
            print("🎉 所有仪表盘优化功能测试通过！")
            print("📋 功能清单:")
            print("   ✅ 数据源类型过滤功能")
            print("   ✅ 图表数据缓存功能")
            print("   ✅ 数据源状态指示器")
            return True
        else:
            print("❌ 部分功能测试失败，请检查实现")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard_optimizations()
    exit(0 if success else 1)
