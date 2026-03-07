#!/usr/bin/env python3
"""
直接测试PostgreSQL持久化功能（不依赖API）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'
os.environ['DB_NAME'] = 'rqa2025'
os.environ['DB_USER'] = 'rqa2025'
os.environ['DB_PASSWORD'] = 'rqa2025pass'

import psycopg2
from datetime import datetime

def test_postgresql_persistence():
    """直接测试PostgreSQL持久化"""
    print("🧪 直接测试PostgreSQL持久化功能")
    print("=" * 60)
    
    # 1. 测试连接
    print("\n1. 测试PostgreSQL连接...")
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='rqa2025',
            user='rqa2025',
            password='rqa2025pass',
            connect_timeout=5
        )
        print("   ✅ 连接成功")
        
        cursor = conn.cursor()
        
        # 2. 检查表是否存在
        print("\n2. 检查表结构...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'akshare_stock_data'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("   ✅ akshare_stock_data表存在")
            
            # 检查索引
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'akshare_stock_data';
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            print(f"   ✅ 索引数量: {len(indexes)}")
            for idx in indexes:
                print(f"      - {idx}")
        else:
            print("   ❌ 表不存在")
            return False
        
        # 3. 测试插入数据
        print("\n3. 测试插入数据...")
        test_data = {
            'source_id': 'akshare_stock',
            'symbol': '000001',
            'date': '2024-12-20',
            'open_price': 10.99,
            'high_price': 11.1,
            'low_price': 10.98,
            'close_price': 11.02,
            'volume': 714646,
            'amount': 831437460.3,
            'pct_change': 0.27,
            'change': 0.03,
            'turnover_rate': 0.37,
            'amplitude': 1.09
        }
        
        insert_query = """
            INSERT INTO akshare_stock_data (
                source_id, symbol, date, open_price, high_price, low_price,
                close_price, volume, amount, pct_change, change,
                turnover_rate, amplitude, data_source
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (source_id, symbol, date) 
            DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                amount = EXCLUDED.amount,
                pct_change = EXCLUDED.pct_change,
                change = EXCLUDED.change,
                turnover_rate = EXCLUDED.turnover_rate,
                amplitude = EXCLUDED.amplitude,
                persistence_timestamp = CURRENT_TIMESTAMP
        """
        
        cursor.execute(insert_query, (
            test_data['source_id'],
            test_data['symbol'],
            test_data['date'],
            test_data['open_price'],
            test_data['high_price'],
            test_data['low_price'],
            test_data['close_price'],
            test_data['volume'],
            test_data['amount'],
            test_data['pct_change'],
            test_data['change'],
            test_data['turnover_rate'],
            test_data['amplitude'],
            'akshare'
        ))
        
        conn.commit()
        print("   ✅ 数据插入成功")
        
        # 4. 测试查询数据
        print("\n4. 测试查询数据...")
        cursor.execute("""
            SELECT symbol, date, close_price, volume, pct_change
            FROM akshare_stock_data
            WHERE symbol = %s AND date = %s
        """, ('000001', '2024-12-20'))
        
        result = cursor.fetchone()
        if result:
            print("   ✅ 数据查询成功")
            print(f"      股票代码: {result[0]}")
            print(f"      日期: {result[1]}")
            print(f"      收盘价: {result[2]}")
            print(f"      成交量: {result[3]}")
            print(f"      涨跌幅: {result[4]}%")
        else:
            print("   ❌ 未找到数据")
            return False
        
        # 5. 测试去重（重复插入）
        print("\n5. 测试数据去重...")
        cursor.execute(insert_query, (
            test_data['source_id'],
            test_data['symbol'],
            test_data['date'],
            11.05,  # 更新价格
            test_data['high_price'],
            test_data['low_price'],
            11.05,
            test_data['volume'],
            test_data['amount'],
            test_data['pct_change'],
            test_data['change'],
            test_data['turnover_rate'],
            test_data['amplitude'],
            'akshare'
        ))
        conn.commit()
        
        cursor.execute("""
            SELECT COUNT(*) FROM akshare_stock_data
            WHERE symbol = %s AND date = %s
        """, ('000001', '2024-12-20'))
        
        count = cursor.fetchone()[0]
        if count == 1:
            print("   ✅ 去重功能正常（只有1条记录）")
            
            # 检查价格是否更新
            cursor.execute("""
                SELECT close_price FROM akshare_stock_data
                WHERE symbol = %s AND date = %s
            """, ('000001', '2024-12-20'))
            updated_price = cursor.fetchone()[0]
            if updated_price == 11.05:
                print("   ✅ 数据更新功能正常（价格已更新）")
            else:
                print(f"   ⚠️  价格未更新（当前: {updated_price}）")
        else:
            print(f"   ❌ 去重失败（有{count}条记录）")
        
        # 6. 统计信息
        print("\n6. 统计信息...")
        cursor.execute("SELECT COUNT(*) FROM akshare_stock_data")
        total_count = cursor.fetchone()[0]
        print(f"   总记录数: {total_count}")
        
        cursor.execute("""
            SELECT COUNT(DISTINCT symbol) FROM akshare_stock_data
        """)
        unique_symbols = cursor.fetchone()[0]
        print(f"   唯一股票数: {unique_symbols}")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ PostgreSQL持久化功能测试通过！")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"   ❌ 连接失败: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_postgresql_persistence()
    sys.exit(0 if success else 1)

