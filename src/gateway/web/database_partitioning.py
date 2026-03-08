"""
数据库分区策略模块
用于实施数据分区策略，提高查询性能
"""

import logging
from typing import Dict, List, Any, Optional

# 使用统一日志系统
logger = logging.getLogger(__name__)

# 延迟导入数据库连接模块
db_available = False
try:
    from .postgresql_persistence import get_db_connection, return_db_connection
    db_available = True
except ImportError:
    logger.warning("PostgreSQL持久化模块不可用，无法实施数据分区策略")


class DatabasePartitioner:
    """
    数据库分区器
    负责实施数据分区策略
    """
    
    @staticmethod
    def ensure_backtest_results_partitioning() -> bool:
        """
        确保backtest_results表使用分区策略
        
        Returns:
            是否成功实施分区策略
        """
        if not db_available:
            logger.warning("数据库连接不可用，跳过分区策略实施")
            return False
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 检查backtest_results表是否存在
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'backtest_results'
                );
            """)
            
            if not cursor.fetchone()[0]:
                logger.warning("backtest_results表不存在，跳过分区策略实施")
                cursor.close()
                return False
            
            # 检查表是否已经分区
            cursor.execute("""
                SELECT relkind 
                FROM pg_class 
                WHERE relname = 'backtest_results';
            """)
            
            result = cursor.fetchone()
            # relkind = 'p' 表示分区表, 'r' 表示普通表
            if result and result[0] == 'p':
                logger.info("backtest_results表已经分区，跳过分区策略实施")
                cursor.close()
                return True
            
            # 实施分区策略
            # 1. 创建分区表
            logger.info("开始实施backtest_results表分区策略...")
            
            # 步骤1：创建新的分区表
            cursor.execute("""
                -- 创建主分区表（分区根表）
                CREATE TABLE IF NOT EXISTS backtest_results_partitioned (
                    backtest_id VARCHAR(100) PRIMARY KEY,
                    strategy_id VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital DECIMAL(18, 2) NOT NULL,
                    final_capital DECIMAL(18, 2),
                    total_return DECIMAL(10, 4),
                    annualized_return DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    max_drawdown DECIMAL(10, 4),
                    win_rate DECIMAL(10, 4),
                    total_trades INTEGER,
                    equity_curve JSONB,
                    trades JSONB,
                    metrics JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                ) PARTITION BY RANGE (created_at);
            """)
            
            # 步骤2：创建分区
            # 创建未来5年的分区
            import datetime
            current_year = datetime.datetime.now().year
            
            for year in range(current_year, current_year + 5):
                partition_name = f"backtest_results_y{year}"
                start_date = f"{year}-01-01"
                end_date = f"{year + 1}-01-01"
                
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {partition_name} 
                    PARTITION OF backtest_results_partitioned
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """)
                logger.debug(f"创建分区 {partition_name} 成功")
            
            # 步骤3：为分区表创建索引
            cursor.execute("""
                -- 为分区表创建索引
                CREATE INDEX IF NOT EXISTS idx_backtest_partitioned_strategy ON backtest_results_partitioned(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_backtest_partitioned_created ON backtest_results_partitioned(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_backtest_partitioned_status ON backtest_results_partitioned(status);
                CREATE INDEX IF NOT EXISTS idx_backtest_partitioned_strategy_created ON backtest_results_partitioned(strategy_id, created_at DESC);
            """)
            
            # 步骤4：迁移数据（如果backtest_results表有数据）
            cursor.execute("SELECT COUNT(*) FROM backtest_results;")
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"迁移 {count} 条数据到分区表...")
                cursor.execute("""
                    INSERT INTO backtest_results_partitioned
                    SELECT * FROM backtest_results
                    ON CONFLICT (backtest_id) DO NOTHING;
                """)
                migrated_count = cursor.rowcount
                logger.info(f"成功迁移 {migrated_count} 条数据到分区表")
            
            # 步骤5：重命名表
            # 注意：这一步需要小心执行，确保没有正在进行的操作
            # 首先重命名原表为backtest_results_old
            cursor.execute("ALTER TABLE IF EXISTS backtest_results RENAME TO backtest_results_old;")
            # 然后重命名分区表为backtest_results
            cursor.execute("ALTER TABLE backtest_results_partitioned RENAME TO backtest_results;")
            
            # 步骤6：创建视图，确保向后兼容
            cursor.execute("""
                CREATE OR REPLACE VIEW backtest_results_old_view AS
                SELECT * FROM backtest_results;
            """)
            
            conn.commit()
            cursor.close()
            
            logger.info("backtest_results表分区策略实施成功")
            return True
            
        except Exception as e:
            logger.error(f"实施backtest_results表分区策略失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    @staticmethod
    def create_partition_maintenance_job() -> bool:
        """
        创建分区维护作业
        用于自动创建未来的分区
        
        Returns:
            是否成功创建维护作业
        """
        if not db_available:
            return False
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            # 创建分区维护函数
            cursor.execute("""
                CREATE OR REPLACE FUNCTION create_future_backtest_partitions()
                RETURNS void AS $$
                DECLARE
                    current_year INTEGER;
                    future_year INTEGER;
                    partition_name TEXT;
                    start_date TEXT;
                    end_date TEXT;
                BEGIN
                    -- 获取当前年份
                    SELECT EXTRACT(YEAR FROM CURRENT_DATE) INTO current_year;
                    
                    -- 创建未来3年的分区
                    FOR i IN 1..3 LOOP
                        future_year := current_year + i;
                        partition_name := 'backtest_results_y' || future_year;
                        start_date := future_year || '-01-01';
                        end_date := (future_year + 1) || '-01-01';
                        
                        -- 检查分区是否已存在
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class 
                            WHERE relname = partition_name
                        ) THEN
                            -- 创建分区
                            EXECUTE format('''
                                CREATE TABLE IF NOT EXISTS %I 
                                PARTITION OF backtest_results
                                FOR VALUES FROM (''%s'') TO (''%s'');
                            ''', partition_name, start_date, end_date);
                        END IF;
                    END LOOP;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # 创建分区维护作业
            # 注意：在PostgreSQL中，可以使用pg_cron扩展来定期执行函数
            # 这里创建一个SQL函数，供外部调度系统调用
            
            conn.commit()
            cursor.close()
            
            logger.info("分区维护作业创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建分区维护作业失败: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    @staticmethod
    def analyze_partitioning_effectiveness() -> Dict[str, Any]:
        """
        分析分区策略的有效性
        
        Returns:
            分区策略有效性分析结果
        """
        if not db_available:
            return {"available": False, "error": "数据库连接不可用"}
        
        conn = None
        try:
            conn = get_db_connection()
            if not conn:
                return {"available": False, "error": "无法获取数据库连接"}
            
            cursor = conn.cursor()
            
            # 检查backtest_results表是否使用分区
            cursor.execute("""
                SELECT partitioned 
                FROM pg_class 
                WHERE relname = 'backtest_results';
            """)
            
            partitioned = cursor.fetchone()
            if not partitioned or not partitioned[0]:
                cursor.close()
                return {
                    "available": True,
                    "partitioned": False,
                    "message": "backtest_results表未使用分区策略"
                }
            
            # 分析分区情况
            cursor.execute("""
                SELECT 
                    relname as partition_name,
                    pg_size_pretty(pg_total_relation_size(relname::regclass)) as size,
                    (SELECT COUNT(*) FROM (TABLE relname::regclass) as t) as row_count
                FROM pg_class
                WHERE relname LIKE 'backtest_results_y%'
                ORDER BY relname;
            """)
            
            partitions = []
            total_size = 0
            total_rows = 0
            
            for row in cursor.fetchall():
                partition_name = row[0]
                size = row[1]
                row_count = row[2]
                
                partitions.append({
                    "name": partition_name,
                    "size": size,
                    "row_count": row_count
                })
                
                # 计算总大小和总行数
                total_rows += row_count
            
            # 获取整体表大小
            cursor.execute("""
                SELECT pg_size_pretty(pg_total_relation_size('backtest_results'));
            """)
            total_size = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                "available": True,
                "partitioned": True,
                "total_size": total_size,
                "total_rows": total_rows,
                "partitions": partitions,
                "message": f"backtest_results表使用分区策略，共 {len(partitions)} 个分区"
            }
            
        except Exception as e:
            logger.error(f"分析分区策略有效性失败: {e}")
            return {
                "available": True,
                "error": str(e)
            }
        finally:
            if conn:
                return_db_connection(conn)
    
    @staticmethod
    def optimize_query_for_partitioning() -> bool:
        """
        优化查询以利用分区策略
        
        Returns:
            是否成功优化查询
        """
        # 这里可以添加查询优化建议或创建特定的索引
        # 由于分区策略已经实施，查询优化主要依赖于应用程序代码
        # 确保查询包含分区键（created_at）以利用分区剪枝
        
        logger.info("查询优化建议：确保查询包含created_at字段以利用分区剪枝")
        return True


# 全局数据库分区器实例
database_partitioner = DatabasePartitioner()


# 工具函数
def ensure_partitioning() -> bool:
    """
    确保所有必要的表都使用分区策略
    
    Returns:
        是否成功实施分区策略
    """
    success = True
    
    # 实施backtest_results表分区策略
    if not DatabasePartitioner.ensure_backtest_results_partitioning():
        success = False
    
    # 创建分区维护作业
    if not DatabasePartitioner.create_partition_maintenance_job():
        success = False
    
    # 优化查询
    if not DatabasePartitioner.optimize_query_for_partitioning():
        success = False
    
    return success


def analyze_partitioning() -> Dict[str, Any]:
    """
    分析分区策略的有效性
    
    Returns:
        分区策略有效性分析结果
    """
    return DatabasePartitioner.analyze_partitioning_effectiveness()


def run_partition_maintenance() -> bool:
    """
    运行分区维护作业
    
    Returns:
        是否成功运行维护作业
    """
    if not db_available:
        return False
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 执行分区维护函数
        cursor.execute("SELECT create_future_backtest_partitions();")
        conn.commit()
        cursor.close()
        
        logger.info("分区维护作业运行成功")
        return True
        
    except Exception as e:
        logger.error(f"运行分区维护作业失败: {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_connection(conn)


# 测试函数
def test_partitioning():
    """
    测试分区功能
    """
    print("测试分区策略实施...")
    result = ensure_partitioning()
    print(f"分区策略实施结果: {result}")
    
    print("\n分析分区策略有效性...")
    analysis = analyze_partitioning()
    print(f"分区策略分析结果: {analysis}")
    
    print("\n运行分区维护作业...")
    maintenance_result = run_partition_maintenance()
    print(f"分区维护作业运行结果: {maintenance_result}")


if __name__ == "__main__":
    test_partitioning()
