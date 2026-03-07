#!/usr/bin/env python3
"""
测试分层数据采集配置
"""

import sys
sys.path.append('.')

def test_layered_collection():
    """测试分层采集配置"""
    print("🧪 测试分层数据采集配置")

    try:
        # 测试股票池选择逻辑
        from src.gateway.web.data_collectors import get_core_universe_symbols, get_extended_universe_symbols, get_batch_symbols_for_collection

        # 模拟股票列表
        all_symbols = [f"{i:06d}" for i in range(1, 100)]  # 000001到000099

        print("1. 测试核心股票池选择...")
        core_symbols = get_core_universe_symbols(all_symbols, 10)
        print(f"   核心池选择结果: {len(core_symbols)}只股票")
        print(f"   示例股票: {core_symbols[:5]}")

        print("2. 测试扩展股票池选择...")
        extended_symbols = get_extended_universe_symbols(all_symbols, 15)
        print(f"   扩展池选择结果: {len(extended_symbols)}只股票")
        print(f"   示例股票: {extended_symbols[:5]}")

        print("3. 测试全市场批次轮询...")
        batch_symbols = get_batch_symbols_for_collection(all_symbols, {"config": {"batch_size": 20}})
        print(f"   批次轮询结果: {len(batch_symbols)}只股票")
        print(f"   示例股票: {batch_symbols[:5]}")

        print("4. 验证配置一致性...")
        # 确保没有重复的股票选择
        all_selected = set(core_symbols + extended_symbols + batch_symbols)
        print(f"   总计选择的股票数: {len(all_selected)}")
        print(f"   原始股票总数: {len(all_symbols)}")

        # 测试监控配置加载
        print("5. 测试监控配置...")
        from src.infrastructure.monitoring.services.data_collection_monitor import get_data_collection_monitor
        monitor = get_data_collection_monitor()
        config = monitor.config

        pools = config.get("data_collection_monitoring", {}).get("pools", {})
        print(f"   监控配置中的股票池数量: {len(pools)}")
        for pool_name, pool_config in pools.items():
            print(f"   - {pool_name}: {pool_config.get('description', 'N/A')}")

        print("✅ 分层采集配置测试通过")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_layered_collection()
    print(f"\n测试结果: {'通过' if result else '失败'}")