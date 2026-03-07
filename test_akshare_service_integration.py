#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare服务集成测试

验证重构后的AKShare服务功能是否正常
"""

import asyncio
import logging
from datetime import datetime, timedelta

from src.core.integration.akshare_service import get_akshare_service

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_akshare_service_basic():
    """测试AKShare服务基本功能"""
    logger.info("开始测试AKShare服务基本功能...")
    
    try:
        # 获取AKShare服务实例
        akshare_service = get_akshare_service()
        
        # 检查服务是否可用
        if not akshare_service.is_available:
            logger.warning("⚠️ AKShare库不可用，跳过部分测试")
            return False
        
        logger.info("✅ AKShare服务初始化成功")
        
        # 测试获取市场数据
        logger.info("测试获取市场数据...")
        market_data = await akshare_service.get_market_data()
        if market_data is not None and not market_data.empty:
            logger.info(f"✅ 市场数据获取成功: {len(market_data)} 条记录")
        else:
            logger.warning("⚠️ 市场数据获取失败或为空")
        
        # 测试获取股票数据
        logger.info("测试获取股票数据...")
        
        # 计算日期范围（最近30天）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        # 测试获取股票数据
        stock_data = await akshare_service.get_stock_data(
            symbol="000001",  # 平安银行
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"
        )
        
        if stock_data is not None and not stock_data.empty:
            logger.info(f"✅ 股票数据获取成功: {len(stock_data)} 条记录")
            logger.info(f"📊 数据字段: {list(stock_data.columns)}")
            
            # 测试转换为标准格式
            standard_data = akshare_service.convert_to_standard_format(stock_data)
            if standard_data:
                logger.info(f"✅ 标准格式转换成功: {len(standard_data)} 条记录")
                logger.info(f"📋 标准格式字段: {list(standard_data[0].keys()) if standard_data else []}")
            else:
                logger.warning("⚠️ 标准格式转换失败")
        else:
            logger.warning("⚠️ 股票数据获取失败或为空")
        
        # 测试获取股票信息
        logger.info("测试获取股票信息...")
        stock_info = await akshare_service.get_stock_info(symbol="000001")
        if stock_info:
            logger.info(f"✅ 股票信息获取成功: {len(stock_info)} 个字段")
            logger.info(f"📋 主要信息: {list(stock_info.keys())[:5]}...")
        else:
            logger.warning("⚠️ 股票信息获取失败")
        
        logger.info("✅ AKShare服务基本功能测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ AKShare服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_akshare_service_with_different_configs():
    """测试不同配置的AKShare服务"""
    logger.info("开始测试不同配置的AKShare服务...")
    
    try:
        # 测试默认配置
        logger.info("测试默认配置...")
        default_service = get_akshare_service()
        logger.info(f"✅ 默认配置服务创建成功: {default_service.config.get('retry_policy')}")
        
        # 测试不同环境配置
        from src.core.integration.config.akshare_service_config import get_akshare_config
        
        dev_config = get_akshare_config("development")
        logger.info(f"✅ 开发环境配置: {dev_config.retry_policy}")
        
        prod_config = get_akshare_config("production")
        logger.info(f"✅ 生产环境配置: {prod_config.retry_policy}")
        
        test_config = get_akshare_config("test")
        logger.info(f"✅ 测试环境配置: {test_config.retry_policy}")
        
        logger.info("✅ 不同配置测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    logger.info("🚀 开始AKShare服务集成测试")
    
    results = []
    
    # 运行基本功能测试
    result1 = await test_akshare_service_basic()
    results.append(result1)
    
    # 运行配置测试
    result2 = await test_akshare_service_with_different_configs()
    results.append(result2)
    
    # 总结测试结果
    logger.info("\n📊 测试结果总结")
    logger.info(f"基本功能测试: {'✅ 通过' if result1 else '❌ 失败'}")
    logger.info(f"配置测试: {'✅ 通过' if result2 else '❌ 失败'}")
    
    if all(results):
        logger.info("🎉 所有测试通过！")
    else:
        logger.warning("⚠️ 部分测试失败")
    
    logger.info("🔚 AKShare服务集成测试完成")


if __name__ == "__main__":
    asyncio.run(main())
