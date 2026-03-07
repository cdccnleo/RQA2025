#!/usr/bin/env python3
"""
测试仪表盘修复效果
验证所有14个启用数据源都能在仪表盘中正确显示
"""

import json
import time
import requests
from typing import Dict, Any

def test_dashboard_data_consistency():
    """测试仪表盘数据一致性"""
    print("🧪 测试仪表盘数据一致性修复")
    print("=" * 50)

    try:
        # 1. 获取配置文件数据
        with open('data/data_sources_config.json', 'r', encoding='utf-8') as f:
            config_sources = json.load(f)

        enabled_config = [s for s in config_sources if s.get('enabled', True)]
        config_ids = set(s['id'] for s in enabled_config)
        print(f"📄 配置文件启用数据源: {len(config_ids)} 个")

        # 2. 获取API数据源列表
        api_url = "http://localhost:8000/api/v1/data/sources"
        api_response = requests.post(api_url, json={"action": "get_all"}, timeout=10)
        api_data = api_response.json()
        api_sources = api_data.get('data') or api_data.get('data_sources') or []
        enabled_api = [s for s in api_sources if s.get('enabled', True)]
        api_ids = set(s['id'] for s in enabled_api)
        print(f"🔌 API返回启用数据源: {len(api_ids)} 个")

        # 3. 获取Metrics数据
        metrics_url = "http://localhost:8000/api/v1/data-sources/metrics"
        metrics_response = requests.get(metrics_url, timeout=10)
        metrics = metrics_response.json()

        latency_ids = set(metrics.get('latency_data', {}).keys())
        throughput_ids = set(metrics.get('throughput_data', {}).keys())
        error_ids = set(metrics.get('error_rates', {}).keys())
        availability_ids = set(metrics.get('availability', {}).keys())
        health_ids = set(metrics.get('health_scores', {}).keys())

        print(f"📊 Metrics延迟数据源: {len(latency_ids)} 个")
        print(f"📊 Metrics吞吐量数据源: {len(throughput_ids)} 个")
        print(f"📊 Metrics错误率数据源: {len(error_ids)} 个")
        print(f"📊 Metrics可用性数据源: {len(availability_ids)} 个")
        print(f"📊 Metrics健康评分: {len(health_ids)} 个")

        # 4. 检查一致性
        all_consistent = True

        if config_ids != api_ids:
            print(f"❌ 配置与API不一致!")
            print(f"   配置有但API没有: {config_ids - api_ids}")
            print(f"   API有但配置没有: {api_ids - config_ids}")
            all_consistent = False

        if config_ids != latency_ids:
            print(f"❌ 配置与延迟数据不一致!")
            print(f"   配置有但延迟数据没有: {config_ids - latency_ids}")
            print(f"   延迟数据有但配置没有: {latency_ids - config_ids}")
            all_consistent = False

        if config_ids != throughput_ids:
            print(f"❌ 配置与吞吐量数据不一致!")
            print(f"   配置有但吞吐量数据没有: {config_ids - throughput_ids}")
            print(f"   吞吐量数据有但配置没有: {throughput_ids - config_ids}")
            all_consistent = False

        # 5. 检查数据值合理性
        print("\n🔍 检查数据合理性:")
        for source_id in config_ids:
            latency = metrics.get('latency_data', {}).get(source_id, 0)
            throughput = metrics.get('throughput_data', {}).get(source_id, 0)
            error_rate = metrics.get('error_rates', {}).get(source_id, 0)
            availability = metrics.get('availability', {}).get(source_id, 0)

            if latency <= 0 or throughput <= 0:
                print(f"⚠️ {source_id}: 延迟={latency}, 吞吐量={throughput} - 可能是估算数据")
            else:
                print(f"✅ {source_id}: 延迟={latency:.1f}ms, 吞吐量={throughput:.1f}")

        if all_consistent and len(config_ids) == 14:
            print("\n🎉 修复成功! 所有14个启用数据源都应该能在仪表盘中正确显示")
            return True
        else:
            print(f"\n❌ 仍有问题需要解决")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard_data_consistency()
    exit(0 if success else 1)
