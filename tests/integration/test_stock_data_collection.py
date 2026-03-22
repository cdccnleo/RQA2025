#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据采集集成测试脚本

执行实际数据采集测试，验证端到端流程。
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def test_akshare_connection():
    """
    测试 AKShare API 连接
    
    Returns:
        测试结果字典
    """
    result = {
        'test_name': 'AKShare API 连接测试',
        'status': 'pending',
        'details': {}
    }
    
    try:
        from src.gateway.web.enhanced_akshare_collector import get_enhanced_collector
        
        collector = get_enhanced_collector(timeout=30, max_retries=3)
        conn_result = collector.test_connection()
        
        result['details'] = conn_result
        
        if conn_result.get('api_accessible'):
            result['status'] = 'passed'
            result['message'] = f"✅ AKShare 连接成功，获取到 {conn_result.get('sample_data_count', 0)} 只股票"
        else:
            result['status'] = 'failed'
            result['message'] = f"❌ AKShare 连接失败: {conn_result.get('error', 'Unknown error')}"
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = f"❌ 测试执行失败: {e}"
    
    return result


def test_single_stock_collection():
    """
    测试单只股票数据采集
    
    Returns:
        测试结果字典
    """
    result = {
        'test_name': '单只股票数据采集测试',
        'status': 'pending',
        'details': {}
    }
    
    try:
        from src.gateway.web.enhanced_akshare_collector import get_enhanced_collector
        
        collector = get_enhanced_collector(timeout=30, max_retries=3)
        
        # 测试采集平安银行(000001)最近30天数据
        symbol = '000001'
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        logger.info(f"开始采集 {symbol} 数据: {start_date} ~ {end_date}")
        
        start_time = time.time()
        data = collector.get_stock_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period='daily',
            adjust='qfq'
        )
        elapsed_time = time.time() - start_time
        
        result['details'] = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'records_count': len(data),
            'elapsed_time': round(elapsed_time, 2)
        }
        
        if len(data) > 0:
            result['status'] = 'passed'
            result['message'] = f"✅ 成功采集 {len(data)} 条记录，耗时 {elapsed_time:.2f}秒"
        else:
            result['status'] = 'warning'
            result['message'] = "⚠️ 采集成功但无数据返回"
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = f"❌ 单只股票采集失败: {e}"
    
    return result


def test_batch_collection():
    """
    测试批量股票数据采集
    
    Returns:
        测试结果字典
    """
    result = {
        'test_name': '批量股票数据采集测试',
        'status': 'pending',
        'details': {}
    }
    
    try:
        from src.gateway.web.enhanced_akshare_collector import get_enhanced_collector
        
        collector = get_enhanced_collector(timeout=30, max_retries=3)
        
        # 测试采集5只股票
        symbols = ['000001', '000002', '600000', '600036', '000858']
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        logger.info(f"开始批量采集 {len(symbols)} 只股票数据")
        
        start_time = time.time()
        results = collector.collect_batch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            progress_callback=lambda cur, total, sym, success, err: 
                logger.info(f"进度: {cur}/{total} - {sym}: {'✅' if success else '❌'}")
        )
        elapsed_time = time.time() - start_time
        
        success_count = sum(1 for v in results.values() if v)
        total_records = sum(len(v) for v in results.values())
        
        result['details'] = {
            'symbols': symbols,
            'success_count': success_count,
            'total_count': len(symbols),
            'total_records': total_records,
            'elapsed_time': round(elapsed_time, 2),
            'success_rate': round(success_count / len(symbols) * 100, 2) if symbols else 0
        }
        
        if success_count == len(symbols):
            result['status'] = 'passed'
            result['message'] = f"✅ 批量采集全部成功: {success_count}/{len(symbols)}，共 {total_records} 条记录"
        elif success_count > 0:
            result['status'] = 'warning'
            result['message'] = f"⚠️ 部分成功: {success_count}/{len(symbols)}，共 {total_records} 条记录"
        else:
            result['status'] = 'failed'
            result['message'] = "❌ 批量采集全部失败"
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = f"❌ 批量采集测试失败: {e}"
    
    return result


def test_database_persistence():
    """
    测试数据库持久化
    
    Returns:
        测试结果字典
    """
    result = {
        'test_name': '数据库持久化测试',
        'status': 'pending',
        'details': {}
    }
    
    try:
        import psycopg2
        
        # 连接数据库
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
            user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
            password=os.getenv('POSTGRES_PASSWORD', '')
        )
        cur = conn.cursor()
        
        # 检查表是否存在
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'akshare_stock_data'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        # 检查视图是否存在
        cur.execute("""
            SELECT table_name FROM information_schema.views 
            WHERE table_schema = 'public' AND table_name LIKE 'v_stock%'
        """)
        views = [row[0] for row in cur.fetchall()]
        
        # 检查数据量
        record_count = 0
        if table_exists:
            cur.execute("SELECT COUNT(*) FROM akshare_stock_data")
            record_count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        result['details'] = {
            'table_exists': table_exists,
            'views': views,
            'record_count': record_count
        }
        
        if table_exists and len(views) >= 3:
            result['status'] = 'passed'
            result['message'] = f"✅ 数据库结构完整: 表存在, {len(views)} 个视图, {record_count} 条记录"
        elif table_exists:
            result['status'] = 'warning'
            result['message'] = f"⚠️ 表存在但视图不完整: {len(views)} 个视图"
        else:
            result['status'] = 'failed'
            result['message'] = "❌ 数据库表不存在"
        
    except Exception as e:
        result['status'] = 'failed'
        result['message'] = f"❌ 数据库测试失败: {e}"
    
    return result


def run_all_tests():
    """
    运行所有测试
    
    Returns:
        测试结果列表
    """
    print("=" * 70)
    print("股票数据采集集成测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    tests = [
        test_akshare_connection,
        test_single_stock_collection,
        test_batch_collection,
        test_database_persistence
    ]
    
    results = []
    passed = 0
    failed = 0
    warnings = 0
    
    for test_func in tests:
        print(f"\n执行: {test_func.__doc__.strip()}")
        print("-" * 50)
        
        try:
            result = test_func()
            results.append(result)
            
            status = result.get('status', 'unknown')
            message = result.get('message', 'No message')
            
            print(f"状态: {status.upper()}")
            print(f"结果: {message}")
            
            if status == 'passed':
                passed += 1
            elif status == 'failed':
                failed += 1
            elif status == 'warning':
                warnings += 1
            
            # 打印详情
            details = result.get('details', {})
            if details:
                print("详情:")
                for key, value in details.items():
                    if key not in ['data']:  # 不打印大数据
                        print(f"  - {key}: {value}")
        
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")
            results.append({
                'test_name': test_func.__name__,
                'status': 'error',
                'message': str(e)
            })
            failed += 1
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    print(f"总计: {len(tests)} 个测试")
    print(f"  ✅ 通过: {passed}")
    print(f"  ⚠️ 警告: {warnings}")
    print(f"  ❌ 失败: {failed}")
    print(f"成功率: {passed / len(tests) * 100:.1f}%")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault('POSTGRES_HOST', 'localhost')
    os.environ.setdefault('POSTGRES_PASSWORD', 'SecurePass123!')
    
    # 运行测试
    results = run_all_tests()
    
    # 返回退出码
    failed_count = sum(1 for r in results if r.get('status') == 'failed')
    sys.exit(0 if failed_count == 0 else 1)
