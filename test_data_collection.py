#!/usr/bin/env python3
"""
测试数据采集修复
"""
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_data_collection():
    """测试数据采集功能"""
    try:
        from src.gateway.web.data_collectors import collect_data_via_data_layer

        # 模拟数据源配置
        source_config = {
            'id': 'akshare_stock_a',
            'name': 'AKShare A股数据',
            'type': '股票数据',
            'config': {
                'stock_pool_type': 'custom',
                'custom_stocks': ['002837', '688702'],
                'data_types': ['daily'],
                'batch_size': 50
            }
        }

        print("🧪 开始测试自选股数据采集...")
        print(f"配置: {source_config['config']}")

        result = await collect_data_via_data_layer(source_config, {})

        if result and result.get('success'):
            print("✅ 数据采集成功!")
            data_len = len(result.get('data', []))
            print(f"采集到 {data_len} 条数据")

            if data_len > 0:
                first_record = result['data'][0]
                print(f"示例数据: 股票={first_record.get('symbol')}, 日期={first_record.get('date')}")
            return True
        else:
            error_msg = result.get('error', '未知错误') if result else '无结果'
            print(f"❌ 数据采集失败: {error_msg}")
            return False

    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_data_collection())
    sys.exit(0 if success else 1)