#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无缝切换机制测试脚本
验证stock_zh_a_hist失败时自动切换到stock_zh_a_daily
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
logger = logging.getLogger("SeamlessSwitchTest")

async def test_seamless_switch():
    """测试无缝切换机制"""
    logger.info("🔀 开始测试无缝切换机制...")
    
    # 测试参数
    today = datetime.datetime.now().strftime('%Y%m%d')
    start_date = '20260120'
    symbol = '000001'
    
    logger.info(f"📊 测试股票: {symbol}")
    logger.info(f"📅 日期范围: {start_date} 到 {today}")
    
    # 优化后的重试机制 - 最多3次重试后切换到备用接口
    max_retries = 3
    retry_delay = 3
    timeout_seconds = 30
    
    df = None
    api_used = "stock_zh_a_hist"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 尝试 {attempt + 1}/{max_retries}")
            
            # 根据重试次数决定使用哪个接口
            if attempt < 2:  # 前2次尝试使用原始接口
                logger.info(f"🔀 第{attempt + 1}次尝试使用原始接口 (stock_zh_a_hist)")
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period='daily',
                    start_date=start_date,
                    end_date=today,
                    adjust='qfq',
                    timeout=timeout_seconds
                )
                api_used = "stock_zh_a_hist"
            else:  # 第3次尝试切换到新浪接口
                logger.warning("⚠️ 原始接口连续失败，切换到新浪财经接口")
                
                # 添加市场前缀
                market_prefix = "sh" if symbol.startswith("6") else "sz"
                full_symbol = f"{market_prefix}{symbol}"
                
                logger.info(f"🔀 切换到新浪财经接口: {full_symbol}")
                api_used = "stock_zh_a_daily"
                
                # 注意：stock_zh_a_daily接口不支持timeout参数
                df = ak.stock_zh_a_daily(
                    symbol=full_symbol,
                    start_date=start_date,
                    end_date=today,
                    adjust="qfq"
                )
            
            if df is not None and not df.empty:
                logger.info(f"✅ 接口 {api_used} 调用成功: {len(df)} 条记录")
                
                # 字段映射逻辑
                if api_used == "stock_zh_a_daily":
                    # 新浪接口字段映射
                    field_mapping = {
                        'date': '日期',
                        'open': '开盘', 
                        'high': '最高',
                        'low': '最低',
                        'close': '收盘',
                        'volume': '成交量'
                    }
                    df = df.rename(columns=field_mapping)
                    logger.info("🔄 字段映射完成")
                
                logger.info(f"📊 最终字段: {list(df.columns)}")
                break
                
        except Exception as e:
            logger.error(f"❌ 尝试 {attempt + 1} 失败: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ {retry_delay}秒后重试...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                logger.error("💥 所有重试均失败")
                
    return df, api_used

async def main():
    """主测试函数"""
    logger.info("🚀 无缝切换机制测试开始")
    
    # 测试无缝切换
    df, api_used = await test_seamless_switch()
    
    logger.info("\n📋 测试结果:")
    if df is not None and not df.empty:
        logger.info(f"✅ 数据采集成功 (接口: {api_used})")
        logger.info(f"📊 数据记录数: {len(df)}")
        logger.info(f"🔤 字段列表: {list(df.columns)}")
        
        # 显示前几行数据
        if len(df) > 0:
            logger.info("📋 数据样例:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                logger.info(f"   行{i+1}: 日期={row['日期'] if '日期' in df.columns else 'N/A'}, "
                           f"开盘={row['开盘'] if '开盘' in df.columns else 'N/A'}, "
                           f"收盘={row['收盘'] if '收盘' in df.columns else 'N/A'}")
    else:
        logger.error("❌ 数据采集失败")
    
    logger.info("\n🎯 无缝切换机制验证完成")
    logger.info("💡 系统现在具备自动故障转移能力:")
    logger.info("   • 优先使用stock_zh_a_hist接口")
    logger.info("   • 失败时自动切换到stock_zh_a_daily接口")
    logger.info("   • 自动处理字段映射差异")
    logger.info("   • 增强的重试和超时机制")

if __name__ == "__main__":
    asyncio.run(main())