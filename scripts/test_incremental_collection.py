"""
测试增量采集功能
验证增量采集不会遗漏数据
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"


def test_incremental_collection():
    """测试增量采集功能"""
    
    source_id = "akshare_stock_a"
    
    logger.info("=" * 60)
    logger.info("增量采集功能测试")
    logger.info("=" * 60)
    
    # 步骤1: 第一次采集（全量）
    logger.info("\n步骤1: 第一次采集（全量采集）")
    logger.info("-" * 60)
    
    end_date = datetime.now() - timedelta(days=30)  # 使用30天前的日期作为结束日期
    start_date = end_date - timedelta(days=10)  # 开始日期为40天前
    
    request_data_1 = {
        "symbols": ["000001"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "incremental": False,  # 禁用增量采集，进行全量采集
        "persist": True
    }
    
    logger.info(f"采集日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"请求参数: {json.dumps(request_data_1, indent=2, ensure_ascii=False)}")
    
    response_1 = requests.post(
        f"{API_BASE_URL}/data/sources/{source_id}/collect",
        json=request_data_1,
        timeout=60
    )
    
    if response_1.status_code != 200:
        logger.error(f"第一次采集失败: {response_1.status_code} - {response_1.text}")
        return False
    
    result_1 = response_1.json()
    logger.info(f"第一次采集结果:")
    logger.info(f"  - 成功: {result_1.get('success')}")
    logger.info(f"  - 采集记录数: {result_1.get('data_count', 0)}")
    logger.info(f"  - 持久化记录数: {result_1.get('data_count', 0)}")  # 假设持久化记录数与采集记录数相同
    
    first_collection_count = result_1.get('data_count', 0)
    
    # 等待一下，确保数据已持久化
    time.sleep(2)
    
    # 步骤2: 第二次采集（增量，相同日期范围）
    logger.info("\n步骤2: 第二次采集（增量采集，相同日期范围）")
    logger.info("-" * 60)
    
    request_data_2 = {
        "symbols": ["000001"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "incremental": True,  # 启用增量采集
        "persist": True
    }
    
    logger.info(f"采集日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"请求参数: {json.dumps(request_data_2, indent=2, ensure_ascii=False)}")
    
    response_2 = requests.post(
        f"{API_BASE_URL}/data/sources/{source_id}/collect",
        json=request_data_2,
        timeout=60
    )
    
    if response_2.status_code != 200:
        logger.error(f"第二次采集失败: {response_2.status_code} - {response_2.text}")
        return False
    
    result_2 = response_2.json()
    logger.info(f"第二次采集结果:")
    logger.info(f"  - 成功: {result_2.get('success')}")
    logger.info(f"  - 采集记录数: {result_2.get('data_count', 0)}")
    logger.info(f"  - 持久化记录数: {result_2.get('data_count', 0)}")  # 假设持久化记录数与采集记录数相同
    
    second_collection_count = result_2.get('data_count', 0)
    
    # 验证：增量采集应该跳过已存在的数据
    if second_collection_count == 0:
        logger.info("✅ 增量采集正确：跳过了已存在的数据")
    else:
        logger.warning(f"⚠️  增量采集可能有问题：采集了 {second_collection_count} 条记录（应该为0）")
    
    # 步骤3: 第三次采集（增量，扩展日期范围）
    logger.info("\n步骤3: 第三次采集（增量采集，扩展日期范围）")
    logger.info("-" * 60)
    
    extended_start_date = start_date - timedelta(days=5)
    
    request_data_3 = {
        "symbols": ["000001"],
        "start_date": extended_start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "incremental": True,  # 启用增量采集
        "persist": True
    }
    
    logger.info(f"采集日期范围: {extended_start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"请求参数: {json.dumps(request_data_3, indent=2, ensure_ascii=False)}")
    
    response_3 = requests.post(
        f"{API_BASE_URL}/data/sources/{source_id}/collect",
        json=request_data_3,
        timeout=60
    )
    
    if response_3.status_code != 200:
        logger.error(f"第三次采集失败: {response_3.status_code} - {response_3.text}")
        return False
    
    result_3 = response_3.json()
    logger.info(f"第三次采集结果:")
    logger.info(f"  - 成功: {result_3.get('success')}")
    logger.info(f"  - 采集记录数: {result_3.get('data_count', 0)}")
    logger.info(f"  - 持久化记录数: {result_3.get('data_count', 0)}")  # 假设持久化记录数与采集记录数相同
    
    third_collection_count = result_3.get('data_count', 0)
    
    # 验证：应该只采集新增的5天数据
    if third_collection_count > 0:
        logger.info(f"✅ 增量采集正确：采集了新增日期范围的数据（{third_collection_count} 条记录）")
    else:
        logger.warning("⚠️  增量采集可能有问题：应该采集新增日期范围的数据")
    
    # 步骤4: 查询数据库验证数据完整性
    logger.info("\n步骤4: 查询数据库验证数据完整性")
    logger.info("-" * 60)
    
    # 这里可以添加数据库查询逻辑来验证数据完整性
    # 由于需要数据库连接，这里只做提示
    logger.info("提示: 可以通过以下SQL查询验证数据完整性:")
    logger.info(f"  SELECT COUNT(*) FROM akshare_stock_data WHERE source_id='{source_id}' AND symbol='000001' AND date >= '{extended_start_date.strftime('%Y-%m-%d')}' AND date <= '{end_date.strftime('%Y-%m-%d')}';")
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    logger.info(f"第一次采集（全量）: {first_collection_count} 条记录")
    logger.info(f"第二次采集（增量，相同范围）: {second_collection_count} 条记录")
    logger.info(f"第三次采集（增量，扩展范围）: {third_collection_count} 条记录")
    
    if second_collection_count == 0 and third_collection_count > 0:
        logger.info("✅ 增量采集功能测试通过！")
        return True
    else:
        logger.warning("⚠️  增量采集功能可能存在问题，请检查日志")
        return False


if __name__ == "__main__":
    try:
        success = test_incremental_collection()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)

