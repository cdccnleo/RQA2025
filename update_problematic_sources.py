#!/usr/bin/env python3
"""
更新问题数据源状态脚本
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

async def update_problematic_sources():
    """更新问题数据源状态"""
    from gateway.web.api import test_data_source_override

    print('🔧 更新问题数据源状态...')
    print('=' * 40)

    # 测试港股数据
    print('\n📊 测试港股数据源...')
    try:
        result1 = await test_data_source_override('akshare_stock_hk')
        success1 = result1.get('success', False)
        status1 = result1.get('status', '未知')
        print(f'✅ 港股数据测试结果: {success1} - {status1}')
    except Exception as e:
        print(f'❌ 港股数据测试失败: {e}')

    # 测试金十新闻
    print('\n📰 测试金十新闻数据源...')
    try:
        result2 = await test_data_source_override('akshare_news_js')
        success2 = result2.get('success', False)
        status2 = result2.get('status', '未知')
        print(f'✅ 金十新闻测试结果: {success2} - {status2}')
    except Exception as e:
        print(f'❌ 金十新闻测试失败: {e}')

    print('\n🎉 状态更新完成！')
    print('建议重新运行验证脚本来确认修复效果。')

if __name__ == "__main__":
    asyncio.run(update_problematic_sources())
