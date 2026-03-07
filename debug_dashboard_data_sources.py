#!/usr/bin/env python3
"""
调试仪表盘数据源显示问题
检查配置中启用数据源数量与仪表盘显示数据源数量的差异
"""

import json
import requests
from typing import List, Dict, Any

def check_data_sources_consistency():
    """检查数据源配置一致性"""
    print("🔍 检查数据源配置一致性")
    print("=" * 50)

    # 1. 检查配置文件中的数据源
    try:
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            config_sources = json.load(f)

        enabled_config_sources = [s for s in config_sources if s.get('enabled', True)]
        print(f"📄 配置文件中启用数据源: {len(enabled_config_sources)}")
        for s in enabled_config_sources:
            print(f"   • {s['id']}: {s.get('name', 'unnamed')}")

    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return

    print()

    # 2. 检查API返回的数据源
    try:
        # 尝试获取API数据
        api_url = "http://localhost:8000/api/v1/data/sources"
        print(f"🌐 尝试连接API: {api_url}")

        response = requests.post(api_url, json={"action": "get_all"}, timeout=10)

        if response.status_code == 200:
            api_data = response.json()
            api_sources = api_data.get('data') or api_data.get('data_sources') or []

            enabled_api_sources = [s for s in api_sources if s.get('enabled', True)]
            print(f"🔌 API返回启用数据源: {len(enabled_api_sources)}")

            for s in enabled_api_sources:
                print(f"   • {s['id']}: {s.get('name', 'unnamed')}")
        else:
            print(f"❌ API请求失败: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ API连接失败: {e}")

    print()

    # 3. 检查metrics API返回的数据
    try:
        metrics_url = "http://localhost:8000/api/v1/data-sources/metrics"
        print(f"📊 检查metrics API: {metrics_url}")

        response = requests.get(metrics_url, timeout=10)

        if response.status_code == 200:
            metrics_data = response.json()
            print(f"📈 Metrics API返回数据源指标:")

            # 检查各个指标的数据源数量
            latency_sources = list(metrics_data.get('latency_data', {}).keys())
            throughput_sources = list(metrics_data.get('throughput_data', {}).keys())
            error_sources = list(metrics_data.get('error_rates', {}).keys())
            availability_sources = list(metrics_data.get('availability', {}).keys())
            health_sources = list(metrics_data.get('health_scores', {}).keys())

            print(f"   • 延迟数据: {len(latency_sources)} 个数据源")
            print(f"   • 吞吐量数据: {len(throughput_sources)} 个数据源")
            print(f"   • 错误率数据: {len(error_sources)} 个数据源")
            print(f"   • 可用性数据: {len(availability_sources)} 个数据源")
            print(f"   • 健康评分: {len(health_sources)} 个数据源")

            # 检查是否有数据源不一致
            all_metric_sources = set(latency_sources + throughput_sources + error_sources + availability_sources + health_sources)
            print(f"   • 所有指标涉及数据源: {len(all_metric_sources)} 个")

            # 找出缺失的数据源
            config_ids = set(s['id'] for s in enabled_config_sources)
            metric_ids = all_metric_sources

            missing_in_metrics = config_ids - metric_ids
            extra_in_metrics = metric_ids - config_ids

            if missing_in_metrics:
                print(f"   ⚠️ 配置文件中有但metrics中缺失: {missing_in_metrics}")

            if extra_in_metrics:
                print(f"   ⚠️ metrics中有但配置文件中没有: {extra_in_metrics}")

        else:
            print(f"❌ Metrics API请求失败: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Metrics API连接失败: {e}")

    print()
    print("💡 分析:")
    print("   • 如果配置文件中有14个启用数据源，但仪表盘只显示2个")
    print("   • 可能是前端DOM查询只找到了部分页面元素")
    print("   • 或者存在过滤/隐藏逻辑导致部分数据源不显示")

if __name__ == "__main__":
    check_data_sources_consistency()
