#!/usr/bin/env python3
"""
测试数据源管理器

验证BaoStock服务和数据源切换机制的功能
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

async def test_bao_stock_service():
    """
    测试BaoStock服务是否正常工作
    """
    logger.info("=== 开始测试BaoStock服务 ===")
    
    data_source_manager = get_data_source_manager()
    
    try:
        # 测试获取股票数据
        symbol = "000001"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        logger.info(f"测试获取股票 {symbol} 的数据，日期范围: {start_date} ~ {end_date}")
        
        # 测试获取股票数据
        stock_data = await data_source_manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="hfq"
        )
        
        if stock_data is not None and not stock_data.empty:
            logger.info(f"✅ BaoStock服务测试成功！获取到 {len(stock_data)} 条数据")
            logger.info(f"数据示例:\n{stock_data.head()}")
        else:
            logger.error("❌ BaoStock服务测试失败：未获取到数据")
            
    except Exception as e:
        logger.error(f"❌ BaoStock服务测试异常: {e}")

async def test_data_source_switching():
    """
    测试数据源切换机制
    """
    logger.info("=== 开始测试数据源切换机制 ===")
    
    data_source_manager = get_data_source_manager()
    
    try:
        # 测试默认数据源（应该是AKShare）
        symbol = "000001"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        logger.info(f"测试默认数据源获取股票 {symbol} 的数据")
        
        # 不指定数据源，让管理器自动选择
        stock_data = await data_source_manager.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="hfq"
        )
        
        if stock_data is not None and not stock_data.empty:
            logger.info(f"✅ 数据源切换测试成功！获取到 {len(stock_data)} 条数据")
        else:
            logger.error("❌ 数据源切换测试失败：未获取到数据")
            
    except Exception as e:
        logger.error(f"❌ 数据源切换测试异常: {e}")

async def test_market_data():
    """
    测试获取市场数据
    """
    logger.info("=== 开始测试市场数据获取 ===")
    
    data_source_manager = get_data_source_manager()
    
    try:
        logger.info("测试获取市场数据")
        
        market_data = await data_source_manager.get_market_data()
        
        if market_data is not None and not market_data.empty:
            logger.info(f"✅ 市场数据获取测试成功！获取到 {len(market_data)} 条数据")
            logger.info(f"数据示例:\n{market_data.head()}")
        else:
            logger.error("❌ 市场数据获取测试失败：未获取到数据")
            
    except Exception as e:
        logger.error(f"❌ 市场数据获取测试异常: {e}")

async def test_stock_info():
    """
    测试获取股票基本信息
    """
    logger.info("=== 开始测试股票基本信息获取 ===")
    
    data_source_manager = get_data_source_manager()
    
    try:
        symbol = "000001"
        logger.info(f"测试获取股票 {symbol} 的基本信息")
        
        stock_info = await data_source_manager.get_stock_info(symbol=symbol)
        
        if stock_info:
            logger.info(f"✅ 股票基本信息获取测试成功！")
            logger.info(f"股票信息: {stock_info}")
        else:
            logger.error("❌ 股票基本信息获取测试失败：未获取到数据")
            
    except Exception as e:
        logger.error(f"❌ 股票基本信息获取测试异常: {e}")

async def main():
    """
    运行所有测试
    """
    logger.info("开始测试数据源管理器...")
    
    # 运行各项测试
    await test_bao_stock_service()
    await asyncio.sleep(2)  # 避免请求过于频繁
    
    await test_data_source_switching()
    await asyncio.sleep(2)
    
    await test_market_data()
    await asyncio.sleep(2)
    
    await test_stock_info()
    
    logger.info("所有测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
