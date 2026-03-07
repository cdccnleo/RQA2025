#!/usr/bin/env python3
"""
测试AKShare不同adjust参数的行为
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

def test_akshare_adjust():
    """测试不同adjust参数"""

    try:
        import akshare as ak

        test_symbol = "002837"  # 英维克
        test_period = "daily"
        start_date = "20250101"
        end_date = "20250120"

        logger.info("🐳 测试AKShare不同adjust参数")

        # 测试1: 不传adjust参数
        try:
            logger.info("测试1: 不传adjust参数")
            df1 = ak.stock_zh_a_hist(
                symbol=test_symbol,
                period=test_period,
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"结果: {'成功' if df1 is not None and not df1.empty else '失败'}")
            if df1 is not None and not df1.empty:
                logger.info(f"字段: {list(df1.columns)}")
                if len(df1) > 0:
                    price_cols = ['开盘', '收盘', '最高', '最低']
                    sample = df1.iloc[0][price_cols].to_dict() if all(col in df1.columns for col in price_cols) else {}
                    logger.info(f"价格样例: {sample}")
        except Exception as e:
            logger.error(f"测试1失败: {e}")

        # 测试2: adjust=""
        try:
            logger.info("测试2: adjust=''")
            df2 = ak.stock_zh_a_hist(
                symbol=test_symbol,
                period=test_period,
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            logger.info(f"结果: {'成功' if df2 is not None and not df2.empty else '失败'}")
            if df2 is not None and not df2.empty:
                logger.info(f"字段: {list(df2.columns)}")
                if len(df2) > 0:
                    price_cols = ['开盘', '收盘', '最高', '最低']
                    sample = df2.iloc[0][price_cols].to_dict() if all(col in df2.columns for col in price_cols) else {}
                    logger.info(f"价格样例: {sample}")
        except Exception as e:
            logger.error(f"测试2失败: {e}")

        # 测试3: adjust="qfq"
        try:
            logger.info("测试3: adjust='qfq'")
            df3 = ak.stock_zh_a_hist(
                symbol=test_symbol,
                period=test_period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            logger.info(f"结果: {'成功' if df3 is not None and not df3.empty else '失败'}")
            if df3 is not None and not df3.empty:
                logger.info(f"字段: {list(df3.columns)}")
                if len(df3) > 0:
                    price_cols = ['开盘', '收盘', '最高', '最低']
                    sample = df3.iloc[0][price_cols].to_dict() if all(col in df3.columns for col in price_cols) else {}
                    logger.info(f"价格样例: {sample}")
        except Exception as e:
            logger.error(f"测试3失败: {e}")

        return True

    except Exception as e:
        logger.error(f"💥 测试异常: {e}")
        import traceback
        logger.error(f"异常详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 开始AKShare adjust参数测试")
    success = test_akshare_adjust()
    logger.info(f"🧪 测试完成: {'成功' if success else '失败'}")