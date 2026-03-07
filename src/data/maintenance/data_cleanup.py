#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清理和维护服务

负责定期清理无效数据、维护数据质量和优化表空间
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DataCleanupService:
    """
    数据清理服务
    
    提供数据清理、维护和优化功能
    """
    
    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """
        初始化数据清理服务
        
        Args:
            db_config: 数据库配置
        """
        # 尝试从环境变量获取数据库配置
        import os
        env_db_config = {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "database": os.environ.get("DB_NAME", "rqa2025_prod"),
            "user": os.environ.get("DB_USER", "rqa2025_admin"),
            "password": os.environ.get("DB_PASSWORD", "SecurePass123!")
        }
        
        # 使用传入的配置或环境变量配置
        self.db_config = db_config or env_db_config
        
        self.conn = None
        self.cursor = None
        
        logger.info(f"✅ 数据清理服务初始化完成")
        logger.info(f"📊 数据库配置: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
    
    def _connect(self):
        """
        连接数据库
        """
        try:
            self.conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("✅ 数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"❌ 数据库连接失败: {e}")
            return False
    
    def _disconnect(self):
        """
        断开数据库连接
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("✅ 数据库连接已关闭")
        except Exception as e:
            logger.error(f"❌ 断开数据库连接失败: {e}")
    
    def clean_invalid_data(self, source_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        清理无效数据
        
        Args:
            source_id: 数据源ID
            **kwargs: 其他过滤条件
            
        Returns:
            清理结果
        """
        start_time = time.time()
        result = {
            "success": False,
            "deleted_count": 0,
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                result["error"] = "数据库连接失败"
                return result
            
            # 构建删除SQL
            delete_conditions = []
            params = []
            
            if source_id:
                delete_conditions.append("source_id = %s")
                params.append(source_id)
            
            # 添加其他过滤条件
            for key, value in kwargs.items():
                delete_conditions.append(f"{key} = %s")
                params.append(value)
            
            if not delete_conditions:
                result["error"] = "至少需要一个过滤条件"
                return result
            
            where_clause = " AND ".join(delete_conditions)
            sql = f"DELETE FROM akshare_stock_data WHERE {where_clause}"
            
            logger.info(f"🔄 执行数据清理: {sql}")
            logger.info(f"参数: {params}")
            
            self.cursor.execute(sql, params)
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            
            result["success"] = True
            result["deleted_count"] = deleted_count
            result["duration"] = time.time() - start_time
            
            logger.info(f"✅ 数据清理完成，删除 {deleted_count} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 数据清理失败: {e}")
            result["error"] = str(e)
            if self.conn:
                self.conn.rollback()
        finally:
            self._disconnect()
        
        return result
    
    def clean_historical_data(self, days: int = 365) -> Dict[str, Any]:
        """
        清理历史数据
        
        Args:
            days: 保留天数
            
        Returns:
            清理结果
        """
        start_time = time.time()
        result = {
            "success": False,
            "deleted_count": 0,
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                result["error"] = "数据库连接失败"
                return result
            
            # 计算截止日期
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
            
            sql = "DELETE FROM akshare_stock_data WHERE date < %s"
            params = [cutoff_date_str]
            
            logger.info(f"🔄 清理 {days} 天前的历史数据")
            logger.info(f"截止日期: {cutoff_date_str}")
            
            self.cursor.execute(sql, params)
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            
            result["success"] = True
            result["deleted_count"] = deleted_count
            result["duration"] = time.time() - start_time
            
            logger.info(f"✅ 历史数据清理完成，删除 {deleted_count} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 历史数据清理失败: {e}")
            result["error"] = str(e)
            if self.conn:
                self.conn.rollback()
        finally:
            self._disconnect()
        
        return result
    
    def optimize_table_space(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        优化表空间
        
        Args:
            tables: 表名列表
            
        Returns:
            优化结果
        """
        start_time = time.time()
        result = {
            "success": False,
            "optimized_tables": [],
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                result["error"] = "数据库连接失败"
                return result
            
            # 默认优化的表
            default_tables = ["akshare_stock_data"]
            target_tables = tables or default_tables
            
            for table in target_tables:
                try:
                    # 执行 VACUUM 优化
                    sql = f"VACUUM ANALYZE {table}"
                    logger.info(f"🔄 优化表空间: {table}")
                    
                    self.cursor.execute(sql)
                    self.conn.commit()
                    
                    result["optimized_tables"].append(table)
                    logger.info(f"✅ 表 {table} 优化完成")
                    
                except Exception as e:
                    logger.error(f"❌ 优化表 {table} 失败: {e}")
            
            result["success"] = len(result["optimized_tables"]) > 0
            result["duration"] = time.time() - start_time
            
            logger.info(f"✅ 表空间优化完成，成功优化 {len(result['optimized_tables'])} 个表")
            
        except Exception as e:
            logger.error(f"❌ 表空间优化失败: {e}")
            result["error"] = str(e)
        finally:
            self._disconnect()
        
        return result
    
    def generate_cleanup_report(self) -> Dict[str, Any]:
        """
        生成清理报告
        
        Returns:
            清理报告
        """
        start_time = time.time()
        report = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "table_stats": {},
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                report["error"] = "数据库连接失败"
                return report
            
            # 获取表统计信息
            tables = ["akshare_stock_data"]
            
            for table in tables:
                try:
                    # 获取记录数
                    count_sql = f"SELECT COUNT(*) as count FROM {table}"
                    self.cursor.execute(count_sql)
                    count_result = self.cursor.fetchone()
                    
                    # 获取表大小
                    size_sql = f"""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('{table}')) as total_size,
                        pg_size_pretty(pg_indexes_size('{table}')) as index_size,
                        pg_size_pretty(pg_table_size('{table}')) as table_size
                    """
                    self.cursor.execute(size_sql)
                    size_result = self.cursor.fetchone()
                    
                    # 获取数据分布
                    source_sql = f"""
                    SELECT 
                        source_id, 
                        COUNT(*) as count, 
                        MIN(date) as min_date, 
                        MAX(date) as max_date
                    FROM {table}
                    GROUP BY source_id
                    ORDER BY count DESC
                    LIMIT 10
                    """
                    self.cursor.execute(source_sql)
                    source_stats = self.cursor.fetchall()
                    
                    report["table_stats"][table] = {
                        "record_count": count_result["count"],
                        "size": {
                            "total": size_result["total_size"],
                            "index": size_result["index_size"],
                            "table": size_result["table_size"]
                        },
                        "top_sources": source_stats
                    }
                    
                except Exception as e:
                    logger.error(f"❌ 获取表 {table} 统计信息失败: {e}")
            
            report["success"] = True
            report["duration"] = time.time() - start_time
            
            logger.info(f"✅ 清理报告生成完成")
            
        except Exception as e:
            logger.error(f"❌ 生成清理报告失败: {e}")
            report["error"] = str(e)
        finally:
            self._disconnect()
        
        return report
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        验证数据完整性
        
        Returns:
            验证结果
        """
        start_time = time.time()
        result = {
            "success": False,
            "validation_result": {},
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                result["error"] = "数据库连接失败"
                return result
            
            # 验证 akshare_stock_data 表
            validation = {
                "total_records": 0,
                "invalid_records": 0,
                "source_id_distribution": {},
                "data_type_distribution": {},
                "date_range": {}
            }
            
            # 获取总记录数
            count_sql = "SELECT COUNT(*) as count FROM akshare_stock_data"
            self.cursor.execute(count_sql)
            count_result = self.cursor.fetchone()
            validation["total_records"] = count_result["count"]
            
            # 获取无效记录数（例如 source_id 为空）
            invalid_sql = "SELECT COUNT(*) as count FROM akshare_stock_data WHERE source_id IS NULL"
            self.cursor.execute(invalid_sql)
            invalid_result = self.cursor.fetchone()
            validation["invalid_records"] = invalid_result["count"]
            
            # 获取 source_id 分布
            source_sql = """
            SELECT source_id, COUNT(*) as count
            FROM akshare_stock_data
            GROUP BY source_id
            ORDER BY count DESC
            """
            self.cursor.execute(source_sql)
            source_stats = self.cursor.fetchall()
            validation["source_id_distribution"] = {item["source_id"]: item["count"] for item in source_stats}
            
            # 获取 data_type 分布
            type_sql = """
            SELECT data_type, COUNT(*) as count
            FROM akshare_stock_data
            GROUP BY data_type
            ORDER BY count DESC
            """
            self.cursor.execute(type_sql)
            type_stats = self.cursor.fetchall()
            validation["data_type_distribution"] = {item["data_type"]: item["count"] for item in type_stats}
            
            # 获取日期范围
            date_sql = """
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM akshare_stock_data
            """
            self.cursor.execute(date_sql)
            date_result = self.cursor.fetchone()
            validation["date_range"] = {
                "min": date_result["min_date"],
                "max": date_result["max_date"]
            }
            
            result["validation_result"] = validation
            result["success"] = True
            result["duration"] = time.time() - start_time
            
            logger.info(f"✅ 数据完整性验证完成")
            
        except Exception as e:
            logger.error(f"❌ 数据完整性验证失败: {e}")
            result["error"] = str(e)
        finally:
            self._disconnect()
        
        return result
    
    def clean_non_standard_data(self) -> Dict[str, Any]:
        """
        清理不符合标准数据格式的数据
        
        标准字段包括：
        - data_type  数据类型
        - open_price 开盘价
        - high_price 最高价
        - low_price  最低价
        - close_price 收盘价
        - volume  成交量
        - amount  成交额
        - pct_change  涨跌幅
        - change  涨跌额
        - turnover_rate 换手率
        - amplitude 振幅
        - data_source 数据源
        
        Returns:
            清理结果
        """
        start_time = time.time()
        result = {
            "success": False,
            "deleted_count": 0,
            "error": None,
            "duration": 0
        }
        
        try:
            if not self._connect():
                result["error"] = "数据库连接失败"
                return result
            
            # 构建删除SQL，删除缺少标准字段的数据
            # 这里使用具体的字段检查逻辑
            delete_sql = """
            DELETE FROM akshare_stock_data 
            WHERE 
                data_type IS NULL OR 
                open_price IS NULL OR 
                high_price IS NULL OR 
                low_price IS NULL OR 
                close_price IS NULL OR 
                volume IS NULL OR 
                amount IS NULL OR 
                pct_change IS NULL OR 
                change IS NULL OR 
                turnover_rate IS NULL OR 
                amplitude IS NULL OR 
                data_source IS NULL
            """
            
            logger.info("🔄 执行不符合标准数据清理")
            logger.info(f"SQL: {delete_sql}")
            
            self.cursor.execute(delete_sql)
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            
            result["success"] = True
            result["deleted_count"] = deleted_count
            result["duration"] = time.time() - start_time
            
            logger.info(f"✅ 不符合标准数据清理完成，删除 {deleted_count} 条记录")
            
        except Exception as e:
            logger.error(f"❌ 清理不符合标准数据失败: {e}")
            result["error"] = str(e)
            if self.conn:
                self.conn.rollback()
        finally:
            self._disconnect()
        
        return result


def get_data_cleanup_service(db_config: Optional[Dict[str, Any]] = None) -> DataCleanupService:
    """
    获取数据清理服务实例
    
    Args:
        db_config: 数据库配置
        
    Returns:
        DataCleanupService实例
    """
    return DataCleanupService(db_config)
