#!/usr/bin/env python3
"""
测试AKShare stock_zh_a_daily接口
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stock_zh_a_daily():
    """测试stock_zh_a_daily接口"""

    try:
        import akshare as ak

        # 测试股票代码（需要市场前缀）
        test_symbol = "sz002837"  # 英维克，深圳股票
        start_date = "20250101"
        end_date = "20250120"

        logger.info("🐳 测试AKShare stock_zh_a_daily接口")
        logger.info(f"📈 测试参数: 股票={test_symbol}, 日期={start_date}-{end_date}")

        # 调用stock_zh_a_daily接口
        logger.info("🚀 调用ak.stock_zh_a_daily...")
        df = ak.stock_zh_a_daily(
            symbol=test_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        logger.info(f"📊 API调用结果: {'成功' if df is not None else '失败'}")

        if df is None:
            logger.error("❌ API返回None")
            return False

        logger.info(f"📏 DataFrame状态: empty={df.empty}, shape={df.shape}")

        if df.empty:
            logger.error("❌ DataFrame为空")
            return False

        logger.info(f"📋 字段列表: {list(df.columns)}")

        # 检查价格字段是否存在
        price_fields = ['open', 'close', 'high', 'low']
        existing_price_fields = [f for f in price_fields if f in df.columns]
        logger.info(f"💰 价格字段存在性: {existing_price_fields}")

        # 检查数据质量
        logger.info("📈 数据质量检查:")

        for field in price_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                total_count = len(df)
                null_count = total_count - non_null_count
                logger.info(f"   {field}: 非空={non_null_count}, 空值={null_count}, 总计={total_count}")

                if non_null_count > 0:
                    sample_values = df[field].dropna().head(3).tolist()
                    logger.info(f"   {field}样例值: {sample_values}")
                else:
                    logger.warning(f"   {field}全部为空值!")
            else:
                logger.error(f"   {field}字段不存在!")

        # 显示前几行数据
        if len(df) > 0:
            logger.info("📋 前3行原始数据:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                price_data = {f: row.get(f) for f in price_fields if f in df.columns}
                logger.info(f"   行{i+1}: {price_data}")

        # 转换为字典列表并检查
        records = df.to_dict('records')
        logger.info(f"🔄 转换为{len(records)}条记录")

        if records:
            sample_record = records[0]
            logger.info(f"📊 第一条记录字段: {list(sample_record.keys())}")

            # 检查价格字段
            price_check = {f: sample_record.get(f) for f in ['open', 'close', 'high', 'low']}
            logger.info(f"💰 价格字段: {price_check}")

        return True

    except Exception as e:
        logger.error(f"💥 测试异常: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 开始stock_zh_a_daily测试")
    success = test_stock_zh_a_daily()
    logger.info(f"🧪 测试完成: {'成功' if success else '失败'}")