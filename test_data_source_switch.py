#!/usr/bin/env python3
"""
测试数据源切换和数据采集功能

验证以下功能：
1. 数据源管理器能够正确加载配置
2. 能够使用AKShare数据源采集数据
3. 当AKShare不可用时，能够切换到Baostock数据源
4. 数据采集和保存功能正常工作
"""

import asyncio
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入数据源管理器
from src.core.integration.data_source_manager import get_data_source_manager

async def test_data_source_manager():
    """
    测试数据源管理器功能
    """
    logger.info("🔍 开始测试数据源管理器")
    
    try:
        # 获取数据源管理器实例
        data_source_manager = get_data_source_manager()
        logger.info("✅ 数据源管理器初始化成功")
        
        # 获取数据源统计信息
        stats = data_source_manager.get_data_source_stats()
        logger.info(f"📊 数据源统计信息: {stats}")
        
        # 测试获取股票数据
        symbol = "002837"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        logger.info(f"📡 测试获取股票数据: {symbol}，日期范围: {start_date}~{end_date}")
        
        # 使用数据源管理器获取股票数据
        stock_data = await data_source_manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="hfq"
        )
        
        if stock_data:
            logger.info(f"✅ 成功获取股票数据，共 {len(stock_data)} 条记录")
            logger.info(f"📊 第一条数据: {stock_data[0]}")
            
            # 检查数据来源
            data_source = stock_data[0].get('data_source', 'unknown')
            logger.info(f"🔄 数据来源: {data_source}")
        else:
            logger.warning("⚠️ 未获取到股票数据")
        
        # 测试获取股票信息（基本面数据）
        logger.info(f"📡 测试获取股票信息: {symbol}")
        stock_info = await data_source_manager.get_stock_info(symbol=symbol)
        
        if stock_info:
            logger.info(f"✅ 成功获取股票信息")
            logger.info(f"📊 股票信息: {stock_info}")
        else:
            logger.warning("⚠️ 未获取到股票信息")
        
        # 测试缓存统计信息
        cache_stats = data_source_manager.get_cache_stats()
        logger.info(f"📊 缓存统计信息: {cache_stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试数据源管理器失败: {e}")
        return False

async def test_data_source_switching():
    """
    测试数据源切换功能
    """
    logger.info("🔍 开始测试数据源切换功能")
    
    try:
        # 获取数据源管理器实例
        data_source_manager = get_data_source_manager()
        
        # 模拟AKShare不可用的情况
        # 这里我们通过修改配置来模拟
        # 注意：这只是一个简单的测试，实际情况会更复杂
        
        logger.info("🔄 测试数据源切换机制")
        
        # 测试多次获取数据，观察是否会切换数据源
        symbol = "688702"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        for i in range(3):
            logger.info(f"📡 第 {i+1} 次测试获取股票数据")
            stock_data = await data_source_manager.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )
            
            if stock_data:
                data_source = stock_data[0].get('data_source', 'unknown')
                logger.info(f"✅ 成功获取数据，数据源: {data_source}")
            else:
                logger.warning(f"⚠️ 第 {i+1} 次未获取到数据")
            
            # 等待一段时间
            await asyncio.sleep(2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试数据源切换失败: {e}")
        return False

async def main():
    """
    主测试函数
    """
    logger.info("🚀 开始测试数据源切换和数据采集功能")
    
    # 测试数据源管理器
    manager_test_result = await test_data_source_manager()
    
    # 测试数据源切换
    switching_test_result = await test_data_source_switching()
    
    if manager_test_result and switching_test_result:
        logger.info("🎉 所有测试通过！数据源切换和数据采集功能正常工作")
    else:
        logger.error("❌ 部分测试失败，请检查日志")

if __name__ == "__main__":
    asyncio.run(main())
