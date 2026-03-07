"""
InfluxDB时序数据库持久化模块
用于存储和查询时间序列性能指标数据
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# 使用统一日志系统
logger = logging.getLogger(__name__)

# 延迟导入InfluxDB客户端
influxdb_available = False
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    influxdb_available = True
    logger.info("InfluxDB客户端导入成功")
except ImportError as e:
    logger.warning(f"InfluxDB客户端导入失败: {e}")

# 全局InfluxDB客户端
_influxdb_client = None


def get_influxdb_client() -> Optional[Any]:
    """
    获取InfluxDB客户端实例
    
    Returns:
        InfluxDB客户端实例，如果初始化失败返回None
    """
    global _influxdb_client
    
    if not influxdb_available:
        return None
    
    if _influxdb_client:
        return _influxdb_client
    
    try:
        # 从环境变量获取配置
        url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        token = os.getenv('INFLUXDB_TOKEN', 'your-token')
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        
        # 创建客户端
        _influxdb_client = InfluxDBClient(
            url=url,
            token=token,
            org=org,
            timeout=30000  # 30秒超时
        )
        
        # 测试连接
        health = _influxdb_client.health()
        if health.status == 'pass':
            logger.info(f"InfluxDB连接成功: {url}")
            return _influxdb_client
        else:
            logger.warning(f"InfluxDB连接失败: {health.message}")
            return None
            
    except Exception as e:
        logger.error(f"初始化InfluxDB客户端失败: {e}")
        return None


def ensure_influxdb_bucket() -> bool:
    """
    确保InfluxDB桶存在，不存在则创建
    
    Returns:
        是否成功确保桶存在
    """
    client = get_influxdb_client()
    if not client:
        return False
    
    try:
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        
        # 检查桶是否存在
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        
        for b in buckets:
            if b.name == bucket:
                logger.debug(f"InfluxDB桶 {bucket} 已存在")
                return True
        
        # 创建桶
        buckets_api.create_bucket(
            bucket_name=bucket,
            org=org
        )
        logger.info(f"InfluxDB桶 {bucket} 创建成功")
        return True
        
    except Exception as e:
        logger.error(f"确保InfluxDB桶存在失败: {e}")
        return False


def save_performance_metrics(strategy_id: str, metrics: Dict[str, Any]) -> bool:
    """
    保存策略性能指标到InfluxDB
    
    Args:
        strategy_id: 策略ID
        metrics: 性能指标字典
    
    Returns:
        是否成功保存
    """
    client = get_influxdb_client()
    if not client:
        return False
    
    try:
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        
        # 确保桶存在
        if not ensure_influxdb_bucket():
            return False
        
        # 创建数据点
        point = Point("strategy_performance")
        point.tag("strategy_id", strategy_id)
        
        # 添加字段
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                point.field(key, value)
        
        # 添加时间戳
        point.time(datetime.utcnow())
        
        # 写入数据
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=bucket, org=org, record=point)
        write_api.close()
        
        logger.debug(f"性能指标已保存到InfluxDB: {strategy_id}")
        return True
        
    except Exception as e:
        logger.error(f"保存性能指标到InfluxDB失败: {e}")
        return False


def batch_save_performance_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    批量保存性能指标到InfluxDB
    
    Args:
        metrics_list: 性能指标列表，每个元素必须包含strategy_id字段
    
    Returns:
        批量操作结果，包含成功和失败的数量
    """
    client = get_influxdb_client()
    if not client:
        return {
            "success": False,
            "total_processed": 0,
            "success_count": 0,
            "failed_count": len(metrics_list)
        }
    
    try:
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        
        # 确保桶存在
        if not ensure_influxdb_bucket():
            return {
                "success": False,
                "total_processed": 0,
                "success_count": 0,
                "failed_count": len(metrics_list)
            }
        
        # 准备数据点
        points = []
        for metrics in metrics_list:
            strategy_id = metrics.get('strategy_id')
            if not strategy_id:
                continue
            
            point = Point("strategy_performance")
            point.tag("strategy_id", strategy_id)
            
            # 添加字段
            for key, value in metrics.items():
                if key != 'strategy_id' and isinstance(value, (int, float)):
                    point.field(key, value)
            
            # 添加时间戳
            point.time(datetime.utcnow())
            points.append(point)
        
        # 批量写入
        total_processed = len(metrics_list)
        total_success = 0
        total_failed = 0
        
        if points:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            try:
                write_api.write(bucket=bucket, org=org, record=points)
                total_success = len(points)
                total_failed = total_processed - total_success
            finally:
                write_api.close()
        else:
            total_failed = total_processed
        
        logger.info(f"批量保存性能指标到InfluxDB完成: 处理 {total_processed} 条，成功 {total_success} 条，失败 {total_failed} 条")
        return {
            "success": total_failed == 0,
            "total_processed": total_processed,
            "success_count": total_success,
            "failed_count": total_failed
        }
        
    except Exception as e:
        logger.error(f"批量保存性能指标到InfluxDB失败: {e}")
        return {
            "success": False,
            "total_processed": 0,
            "success_count": 0,
            "failed_count": len(metrics_list)
        }


def query_performance_metrics(strategy_id: str, start_time: str = None, end_time: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    查询策略性能指标
    
    Args:
        strategy_id: 策略ID
        start_time: 开始时间，格式为RFC3339
        end_time: 结束时间，格式为RFC3339
        limit: 返回的最大结果数
    
    Returns:
        性能指标列表
    """
    client = get_influxdb_client()
    if not client:
        return []
    
    try:
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        
        # 构建查询语句
        query = f"""
            from(bucket: "{bucket}")
              |> range(start: {start_time or '-30d'}, stop: {end_time or 'now()'})
              |> filter(fn: (r) => r["_measurement"] == "strategy_performance")
              |> filter(fn: (r) => r["strategy_id"] == "{strategy_id}")
              |> limit(n: {limit})
        """
        
        # 执行查询
        query_api = client.query_api()
        result = query_api.query(org=org, query=query)
        
        # 处理查询结果
        metrics = []
        for table in result:
            for record in table.records:
                metric = {
                    "time": record.get_time().isoformat(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "strategy_id": record.get_tag("strategy_id")
                }
                metrics.append(metric)
        
        logger.debug(f"从InfluxDB查询性能指标: {strategy_id}, 找到 {len(metrics)} 条记录")
        return metrics
        
    except Exception as e:
        logger.error(f"查询性能指标失败: {e}")
        return []


def query_aggregate_metrics(start_time: str = None, end_time: str = None) -> Dict[str, Any]:
    """
    查询聚合性能指标
    
    Args:
        start_time: 开始时间，格式为RFC3339
        end_time: 结束时间，格式为RFC3339
    
    Returns:
        聚合性能指标
    """
    client = get_influxdb_client()
    if not client:
        return {}
    
    try:
        bucket = os.getenv('INFLUXDB_BUCKET', 'strategy_performance')
        org = os.getenv('INFLUXDB_ORG', 'rqa2025')
        
        # 构建查询语句
        query = f"""
            from(bucket: "{bucket}")
              |> range(start: {start_time or '-30d'}, stop: {end_time or 'now()'})
              |> filter(fn: (r) => r["_measurement"] == "strategy_performance")
              |> aggregateWindow(every: 1d, fn: mean, createEmpty: false)
              |> yield(name: "mean")
        """
        
        # 执行查询
        query_api = client.query_api()
        result = query_api.query(org=org, query=query)
        
        # 处理查询结果
        aggregate_metrics = {}
        for table in result:
            for record in table.records:
                field = record.get_field()
                if field not in aggregate_metrics:
                    aggregate_metrics[field] = []
                
                aggregate_metrics[field].append({
                    "time": record.get_time().isoformat(),
                    "value": record.get_value()
                })
        
        logger.debug(f"从InfluxDB查询聚合性能指标: 找到 {len(aggregate_metrics)} 个指标")
        return aggregate_metrics
        
    except Exception as e:
        logger.error(f"查询聚合性能指标失败: {e}")
        return {}


def close_influxdb_client():
    """
    关闭InfluxDB客户端
    """
    global _influxdb_client
    
    if _influxdb_client:
        try:
            _influxdb_client.close()
            logger.info("InfluxDB客户端已关闭")
        except Exception as e:
            logger.error(f"关闭InfluxDB客户端失败: {e}")
        finally:
            _influxdb_client = None


# 注册退出处理
try:
    import atexit
    atexit.register(close_influxdb_client)
except Exception:
    pass


# 测试InfluxDB连接
def test_influxdb_connection() -> bool:
    """
    测试InfluxDB连接
    
    Returns:
        是否连接成功
    """
    client = get_influxdb_client()
    if not client:
        return False
    
    try:
        health = client.health()
        return health.status == 'pass'
    except Exception:
        return False
