"""
AKShare Index Data Collector
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def collect_index_data() -> Optional[List[Dict[str, Any]]]:
    """
    采集A股指数实时行情数据（新浪）
    返回所有指数的当前价格快照，日期标记为今天
    
    Returns:
        指数数据列表
    """
    try:
        import akshare as ak
        import pandas as pd
        
        logger.info("开始采集A股指数数据...")
        df = ak.stock_zh_index_spot_sina()
        
        if df.empty:
            logger.warning("未获取到指数数据")
            return None
        
        logger.info(f"成功获取指数数据: {len(df)} 条")
        
        data = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for _, row in df.iterrows():
            code = row.get('代码', '')
            name = row.get('名称', '')
            latest_price = row.get('最新价')
            change_amt = row.get('涨跌额')
            change_ratio = row.get('涨跌幅')
            prev_close = row.get('昨收')
            today_open = row.get('今开')
            high = row.get('最高')
            low = row.get('最低')
            volume = row.get('成交量')
            amount = row.get('成交额')
            
            data.append({
                'index_code': str(code) if code is not None else None,
                'index_name': str(name) if name is not None else None,
                'date': today,
                'open_price': float(today_open) if today_open is not None else None,
                'high_price': float(high) if high is not None else None,
                'low_price': float(low) if low is not None else None,
                'close_price': float(latest_price) if latest_price is not None else None,
                'change_ratio': float(change_ratio) if change_ratio is not None else None,
                'volume': int(volume) if volume is not None else None,
                'amount': float(amount) if amount is not None else None,
                'data_source': 'akshare',
                'source_id': 'akshare_index',
            })
        
        logger.info(f"指数数据转换完成: {len(data)} 条")
        return data
        
    except Exception as e:
        logger.error(f"采集指数数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_index_data_to_database(data: List[Dict[str, Any]]) -> bool:
    """
    保存指数数据到数据库
    
    Args:
        data: 指数数据列表
        
    Returns:
        是否成功
    """
    if not data:
        logger.warning("没有指数数据需要保存")
        return False
    
    conn = None
    cursor = None
    
    try:
        from src.gateway.web.postgresql_persistence import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO akshare_index_data 
                (source_id, index_code, index_name, date, open_price, high_price, 
                 low_price, close_price, change_ratio, volume, amount, data_source, collected_at)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_id, index_code, date) DO UPDATE SET
                index_name = EXCLUDED.index_name,
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                change_ratio = EXCLUDED.change_ratio,
                volume = EXCLUDED.volume,
                amount = EXCLUDED.amount,
                collected_at = EXCLUDED.collected_at
        """
        
        records_to_insert = []
        now = datetime.now()
        for record in data:
            values = (
                record.get('source_id', 'akshare_index'),
                record.get('index_code'),
                record.get('index_name'),
                record.get('date'),
                record.get('open_price'),
                record.get('high_price'),
                record.get('low_price'),
                record.get('close_price'),
                record.get('change_ratio'),
                record.get('volume'),
                record.get('amount'),
                record.get('data_source', 'akshare'),
                now
            )
            records_to_insert.append(values)
        
        cursor.executemany(insert_query, records_to_insert)
        conn.commit()
        
        logger.info(f"成功保存 {len(data)} 条指数数据到数据库")
        return True
        
    except Exception as e:
        logger.error(f"保存指数数据到数据库失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            from src.gateway.web.postgresql_persistence import return_db_connection
            return_db_connection(conn)
