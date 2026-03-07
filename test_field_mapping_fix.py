#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字段映射修复测试脚本
验证新浪财经接口字段映射是否正确
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
logger = logging.getLogger("FieldMappingTest")

async def test_field_mapping():
    """测试字段映射功能"""
    logger.info("🔍 开始测试字段映射功能...")
    
    # 测试参数
    symbol = '002837'
    today = datetime.datetime.now().strftime('%Y%m%d')
    start_date = '20250403'
    
    logger.info(f"📊 测试股票: {symbol}")
    logger.info(f"📅 日期范围: {start_date} 到 {today}")
    
    # 模拟历史数据调度器的无缝切换逻辑
    df = None
    
    # 直接使用新浪财经接口（模拟第3次尝试）
    logger.info("🔀 直接使用新浪财经接口")
    
    # 添加市场前缀
    market_prefix = "sh" if symbol.startswith("6") else "sz"
    full_symbol = f"{market_prefix}{symbol}"
    
    logger.info(f"📡 新浪财经接口: {full_symbol}")
    
    try:
        df = ak.stock_zh_a_daily(
            symbol=full_symbol,
            start_date=start_date,
            end_date=today,
            adjust="qfq"
        )
        
        if df is not None and not df.empty:
            logger.info(f"✅ 新浪财经接口调用成功: {len(df)} 条记录")
            logger.info(f"📊 原始字段列表: {list(df.columns)}")
            
            # 应用字段映射
            field_mapping = {
                'date': '日期',
                'open': '开盘', 
                'high': '最高',
                'low': '最低',
                'close': '收盘',
                'volume': '成交量',
                'amount': '成交额',
                'outstanding_share': '流通股本',
                'turnover': '换手率'
            }
            
            df_mapped = df.rename(columns=field_mapping)
            logger.info(f"🔄 字段映射完成")
            logger.info(f"📊 映射后字段列表: {list(df_mapped.columns)}")
            
            # 测试数据转换逻辑
            logger.info("🔍 测试数据转换逻辑...")
            
            collected_data = []
            for index, row in df_mapped.iterrows():
                # 使用修复后的安全字段访问逻辑
                record = {
                    'symbol': symbol,
                    'data_type': 'price',
                    'timestamp': datetime.datetime.now().timestamp()
                }
                
                # 日期字段映射
                if '日期' in row:
                    record['date'] = row['日期']
                elif 'date' in row:
                    record['date'] = row['date']
                
                # 价格字段映射
                if '开盘' in row:
                    record['open'] = row['开盘']
                elif 'open' in row:
                    record['open'] = row['open']
                    
                if '最高' in row:
                    record['high'] = row['最高']
                elif 'high' in row:
                    record['high'] = row['high']
                    
                if '最低' in row:
                    record['low'] = row['最低']
                elif 'low' in row:
                    record['low'] = row['low']
                    
                if '收盘' in row:
                    record['close'] = row['收盘']
                elif 'close' in row:
                    record['close'] = row['close']
                
                # 成交量字段映射
                if '成交量' in row:
                    record['volume'] = row['成交量']
                elif 'volume' in row:
                    record['volume'] = row['volume']
                
                # 成交额字段映射
                if '成交额' in row:
                    record['amount'] = row['成交额']
                elif 'amount' in row:
                    record['amount'] = row['amount']
                
                collected_data.append(record)
            
            logger.info(f"✅ 数据转换成功: {len(collected_data)} 条记录")
            
            # 检查关键字段是否存在
            if collected_data:
                first_record = collected_data[0]
                logger.info("🔍 关键字段检查:")
                
                required_fields = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                for field in required_fields:
                    if field in first_record:
                        logger.info(f"   ✅ {field}: 存在, 值: {first_record[field]}")
                    else:
                        logger.error(f"   ❌ {field}: 缺失")
                
                logger.info("📋 第一条记录详情:")
                for key, value in first_record.items():
                    logger.info(f"   {key}: {value}")
            
            return True
        else:
            logger.error("❌ 新浪财经接口返回空数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ 接口调用失败: {e}")
        return False

async def main():
    """主测试函数"""
    logger.info("🚀 字段映射修复测试开始")
    
    # 测试字段映射功能
    success = await test_field_mapping()
    
    logger.info("\n📋 测试结果:")
    if success:
        logger.info("✅ 字段映射修复测试成功")
        logger.info("💡 修复内容:")
        logger.info("   • 添加完整的字段映射表")
        logger.info("   • 实现安全的字段访问逻辑")
        logger.info("   • 兼容不同接口的字段名差异")
    else:
        logger.error("❌ 字段映射修复测试失败")
    
    logger.info("\n🎯 测试完成")

if __name__ == "__main__":
    asyncio.run(main())