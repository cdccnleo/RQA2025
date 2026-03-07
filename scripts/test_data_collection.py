#!/usr/bin/env python3
"""
测试数据采集和样本生成脚本
用于解决数据源样本显示问题
"""

import asyncio
import sys
import os
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

async def test_wallstreet_collection():
    """测试华尔街见闻数据采集"""
    try:
        print("🔍 开始测试华尔街见闻数据采集...")

        # 导入必要的模块
        from src.gateway.web.api import load_data_sources
        from src.gateway.web.data_collectors import collect_data_via_data_layer
        from src.gateway.web.api_utils import persist_collected_data

        # 加载数据源配置
        sources = load_data_sources()
        wallstreet_source = None

        for source in sources:
            if source.get('id') == 'akshare_news_wallstreet':
                wallstreet_source = source
                break

        if not wallstreet_source:
            print("❌ 未找到华尔街见闻数据源配置")
            return False

        print("✅ 找到数据源配置:")
        print(f"   ID: {wallstreet_source.get('id')}")
        print(f"   名称: {wallstreet_source.get('name')}")
        print(f"   类型: {wallstreet_source.get('type')}")
        print(f"   AKShare函数: {wallstreet_source.get('config', {}).get('akshare_function')}")

        # 采集数据
        print("📊 开始采集数据...")
        collected_data = await collect_data_via_data_layer(wallstreet_source)

        if collected_data and len(collected_data) > 0:
            print(f"✅ 成功采集 {len(collected_data)} 条数据")

            # 显示前3条数据样例
            print("📋 数据样例:")
            for i, item in enumerate(collected_data[:3]):
                print(f"   {i+1}. {item}")

            # 持久化数据
            print("💾 持久化数据...")
            import time
            metadata = {
                "collection_timestamp": time.time(),
                "test_collection": True,
                "data_count": len(collected_data)
            }

            persist_result = await persist_collected_data(
                wallstreet_source['id'],
                collected_data,
                metadata,
                wallstreet_source
            )

            print("✅ 数据持久化完成")
            return True
        else:
            print("❌ 未采集到数据")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    print("🚀 开始数据采集测试...")
    success = await test_wallstreet_collection()

    if success:
        print("🎉 测试完成！请检查前端是否能显示数据样本。")
    else:
        print("❌ 测试失败！")

if __name__ == "__main__":
    asyncio.run(main())
