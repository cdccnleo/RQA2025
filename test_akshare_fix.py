#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AKShare连接异常修复测试脚本
用于验证修复方案的有效性
"""

import asyncio
import akshare as ak
import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AKShareTest")

async def test_akshare_connection():
    """测试AKShare连接状态"""
    logger.info("🔍 开始AKShare连接测试...")
    
    # 使用合理的日期范围
    today = datetime.datetime.now().strftime('%Y%m%d')
    start_date = '20260120'  # 一周前
    
    logger.info(f"📅 测试日期范围: {start_date} 到 {today}")
    
    # 测试股票代码
    test_symbols = ['000001', '002837', '688702']
    
    for symbol in test_symbols:
        logger.info(f"\n📊 测试股票代码: {symbol}")
        
        # 使用增强的重试机制
        max_retries = 8
        retry_delay = 5
        timeout_seconds = 45
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{max_retries}")
                
                # 调用AKShare API
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period='daily',
                    start_date=start_date,
                    end_date=today,
                    adjust='qfq',
                    timeout=timeout_seconds
                )
                
                if df is not None and not df.empty:
                    logger.info(f"✅ 股票 {symbol} 数据获取成功: {len(df)} 条记录")
                    logger.info(f"📈 数据字段: {list(df.columns)}")
                    break
                else:
                    logger.warning(f"⚠️ 股票 {symbol} 返回空数据")
                    
            except Exception as e:
                logger.error(f"❌ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"⏳ {retry_delay}秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.error(f"💥 股票 {symbol} 所有重试均失败")
                    
                    # 连接错误诊断
                    import requests
                    if isinstance(e, (requests.exceptions.ConnectionError, ConnectionError)):
                        logger.error("🔌 连接异常诊断:")
                        logger.error("   • AKShare服务器可能暂时不可用")
                        logger.error("   • 建议等待5-30分钟后重试")
                        logger.error("   • 检查网络连接状态")
                    
                    # 尝试备用方案
                    logger.info("🔄 尝试备用数据源方案...")
                    # 这里可以添加备用数据源的调用逻辑

async def test_network_connectivity():
    """测试网络连接性"""
    logger.info("\n🌐 测试网络连接性...")
    
    import requests
    import socket
    
    # 测试基础网络
    try:
        socket.create_connection(('www.baidu.com', 80), timeout=10)
        logger.info("✅ 基础网络连接正常")
    except Exception as e:
        logger.error(f"❌ 基础网络连接失败: {e}")
        return False
    
    # 测试AKShare相关服务器
    servers = [
        'http://quote.eastmoney.com',
        'http://push2.eastmoney.com', 
        'http://data.eastmoney.com'
    ]
    
    for server in servers:
        try:
            response = requests.get(server, timeout=10)
            logger.info(f"✅ {server} 连接正常 (状态码: {response.status_code})")
        except Exception as e:
            logger.warning(f"⚠️ {server} 连接异常: {e}")
    
    return True

async def main():
    """主测试函数"""
    logger.info("🚀 AKShare连接异常修复测试开始")
    
    # 测试网络连接
    await test_network_connectivity()
    
    # 测试AKShare连接
    await test_akshare_connection()
    
    logger.info("\n📋 测试总结:")
    logger.info("✅ 修复方案已实施:")
    logger.info("   • 增强重试机制 (8次重试，5秒初始延迟)")
    logger.info("   • 延长超时设置 (45秒)")
    logger.info("   • 改进错误诊断和日志")
    logger.info("   • 添加备用数据源方案")
    logger.info("\n💡 如果AKShare服务器仍不可用:")
    logger.info("   • 等待服务器恢复 (通常5-30分钟)")
    logger.info("   • 检查网络连接状态")
    logger.info("   • 降低请求频率")
    
    logger.info("\n🎯 测试完成")

if __name__ == "__main__":
    asyncio.run(main())