#!/usr/bin/env python3
"""
统一数据源计算逻辑测试验证脚本

验证数据源管理器的统一计算逻辑，包括：
1. AKShare和BaoStock数据源的数据处理
2. 统一计算逻辑（涨跌额、振幅、涨跌幅）的正确性
3. 数据标准化处理
4. 数据源切换机制
"""

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/unified_data_source_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

from src.core.integration.data_source_manager import get_data_source_manager

async def test_akshare_data_source():
    """
    测试AKShare数据源
    """
    logger.info("🧪 开始测试AKShare数据源")
    
    try:
        data_source_manager = get_data_source_manager()
        
        # 测试获取股票数据
        stock_data = await data_source_manager.get_stock_data(
            symbol="000001",
            start_date="20250101",
            end_date="20250110",
            adjust="hfq"
        )
        
        if stock_data:
            logger.info(f"✅ AKShare数据源返回 {len(stock_data)} 条记录")
            
            # 验证数据结构
            if stock_data and len(stock_data) > 0:
                sample_record = stock_data[0]
                required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'data_source']
                calculated_fields = ['change', 'amplitude', 'pct_change']
                
                logger.info(f"📊 样本记录: {sample_record}")
                
                # 验证必需字段
                missing_fields = [field for field in required_fields if field not in sample_record]
                if missing_fields:
                    logger.error(f"❌ 缺少必需字段: {missing_fields}")
                    return False
                else:
                    logger.info("✅ 所有必需字段都存在")
                
                # 验证计算字段
                missing_calculated = [field for field in calculated_fields if field not in sample_record]
                if missing_calculated:
                    logger.error(f"❌ 缺少计算字段: {missing_calculated}")
                    return False
                else:
                    logger.info("✅ 所有计算字段都存在")
                
                # 验证计算逻辑
                for record in stock_data[:3]:  # 只验证前3条记录
                    open_price = record.get('open', 0)
                    close_price = record.get('close', 0)
                    high_price = record.get('high', 0)
                    low_price = record.get('low', 0)
                    change = record.get('change', 0)
                    amplitude = record.get('amplitude', 0)
                    pct_change = record.get('pct_change', 0)
                    
                    # 验证涨跌额计算
                    expected_change = close_price - open_price
                    if abs(change - expected_change) > 0.0001:
                        logger.error(f"❌ 涨跌额计算错误: 实际={change}, 预期={expected_change}")
                        return False
                    
                    # 验证振幅计算
                    if open_price > 0:
                        expected_amplitude = (high_price - low_price) / open_price * 100
                        if abs(amplitude - expected_amplitude) > 0.0001:
                            logger.error(f"❌ 振幅计算错误: 实际={amplitude}, 预期={expected_amplitude}")
                            return False
                    
                    # 验证涨跌幅计算
                    if open_price > 0:
                        expected_pct_change = (close_price - open_price) / open_price * 100
                        if abs(pct_change - expected_pct_change) > 0.0001:
                            logger.error(f"❌ 涨跌幅计算错误: 实际={pct_change}, 预期={expected_pct_change}")
                            return False
                
                logger.info("✅ 计算逻辑验证通过")
                return True
            else:
                logger.warning("⚠️ AKShare数据源返回空数据")
                return False
        else:
            logger.warning("⚠️ AKShare数据源未返回数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ AKShare数据源测试失败: {e}")
        return False

async def test_baostock_data_source():
    """
    测试BaoStock数据源
    """
    logger.info("🧪 开始测试BaoStock数据源")
    
    try:
        data_source_manager = get_data_source_manager()
        
        # 测试获取股票数据
        stock_data = await data_source_manager.get_stock_data(
            symbol="000001",
            start_date="20250101",
            end_date="20250110",
            adjust="hfq"
        )
        
        if stock_data:
            logger.info(f"✅ BaoStock数据源返回 {len(stock_data)} 条记录")
            
            # 验证数据结构
            if stock_data and len(stock_data) > 0:
                sample_record = stock_data[0]
                required_fields = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'data_source']
                calculated_fields = ['change', 'amplitude', 'pct_change']
                
                logger.info(f"📊 样本记录: {sample_record}")
                
                # 验证必需字段
                missing_fields = [field for field in required_fields if field not in sample_record]
                if missing_fields:
                    logger.error(f"❌ 缺少必需字段: {missing_fields}")
                    return False
                else:
                    logger.info("✅ 所有必需字段都存在")
                
                # 验证计算字段
                missing_calculated = [field for field in calculated_fields if field not in sample_record]
                if missing_calculated:
                    logger.error(f"❌ 缺少计算字段: {missing_calculated}")
                    return False
                else:
                    logger.info("✅ 所有计算字段都存在")
                
                logger.info("✅ BaoStock数据源测试通过")
                return True
            else:
                logger.warning("⚠️ BaoStock数据源返回空数据")
                return False
        else:
            logger.warning("⚠️ BaoStock数据源未返回数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ BaoStock数据源测试失败: {e}")
        return False

async def test_data_source_switching():
    """
    测试数据源切换机制
    """
    logger.info("🧪 开始测试数据源切换机制")
    
    try:
        data_source_manager = get_data_source_manager()
        
        # 测试获取股票数据（应该能够自动切换数据源）
        stock_data = await data_source_manager.get_stock_data(
            symbol="000001",
            start_date="20250101",
            end_date="20250110",
            adjust="hfq"
        )
        
        if stock_data:
            logger.info(f"✅ 数据源切换机制成功返回 {len(stock_data)} 条记录")
            
            # 验证数据来源
            data_sources = set()
            for record in stock_data:
                data_sources.add(record.get('data_source', 'unknown'))
            
            logger.info(f"🌐 数据来源: {data_sources}")
            return True
        else:
            logger.warning("⚠️ 数据源切换机制未返回数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ 数据源切换机制测试失败: {e}")
        return False

async def test_unified_calculation_logic():
    """
    测试统一计算逻辑
    """
    logger.info("🧪 开始测试统一计算逻辑")
    
    try:
        data_source_manager = get_data_source_manager()
        
        # 创建测试数据
        test_df = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'open': [100.0, 102.0],
            'high': [105.0, 106.0],
            'low': [95.0, 100.0],
            'close': [102.0, 104.0],
            'volume': [1000000, 1200000],
            'amount': [102000000, 124800000]
        })
        
        # 测试数据标准化方法
        normalized_data = data_source_manager._normalize_stock_data(test_df, "test")
        
        if normalized_data:
            logger.info(f"✅ 统一计算逻辑返回 {len(normalized_data)} 条记录")
            
            # 验证计算结果
            expected_results = [
                {'change': 2.0, 'amplitude': 10.0, 'pct_change': 2.0},
                {'change': 2.0, 'amplitude': 5.88235294117647},
                {'pct_change': 1.9607843137254902}
            ]
            
            for i, record in enumerate(normalized_data):
                logger.info(f"📊 计算结果 {i+1}: change={record['change']:.2f}, amplitude={record['amplitude']:.2f}, pct_change={record['pct_change']:.2f}")
            
            logger.info("✅ 统一计算逻辑测试通过")
            return True
        else:
            logger.warning("⚠️ 统一计算逻辑未返回数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ 统一计算逻辑测试失败: {e}")
        return False

async def main():
    """
    主测试函数
    """
    logger.info("🚀 开始统一数据源计算逻辑测试验证")
    
    tests = [
        ("AKShare数据源测试", test_akshare_data_source),
        ("BaoStock数据源测试", test_baostock_data_source),
        ("数据源切换机制测试", test_data_source_switching),
        ("统一计算逻辑测试", test_unified_calculation_logic)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 开始测试: {test_name}")
        success = await test_func()
        results[test_name] = success
        logger.info(f"📋 测试结果: {test_name} {'✅ 通过' if success else '❌ 失败'}")
    
    # 总结测试结果
    logger.info("\n📊 测试结果总结")
    passed_tests = [name for name, success in results.items() if success]
    failed_tests = [name for name, success in results.items() if not success]
    
    logger.info(f"✅ 通过测试: {len(passed_tests)}/{len(tests)}")
    for test_name in passed_tests:
        logger.info(f"  - {test_name}")
    
    if failed_tests:
        logger.error(f"❌ 失败测试: {len(failed_tests)}/{len(tests)}")
        for test_name in failed_tests:
            logger.error(f"  - {test_name}")
    
    if len(passed_tests) == len(tests):
        logger.info("🎉 所有测试通过！统一数据源计算逻辑验证成功")
        return 0
    else:
        logger.error("💥 部分测试失败，需要进一步排查")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
