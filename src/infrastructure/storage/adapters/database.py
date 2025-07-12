#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional
from ..core import StorageAdapter
from ...db import DatabaseConnectionPool

class DatabaseAdapter(StorageAdapter):
    """数据库存储适配器基类"""

    def __init__(self, connection_pool: DatabaseConnectionPool):
        self.pool = connection_pool
        self._init_schema()

    def _init_schema(self):
        """初始化数据库表结构"""
        raise NotImplementedError

    def write(self, path: str, data: Dict) -> bool:
        """写入数据到数据库"""
        with self.pool.get_connection() as conn:
            try:
                return self._execute_write(conn, path, data)
            except Exception as e:
                conn.rollback()
                raise

    def _execute_write(self, conn, path: str, data: Dict) -> bool:
        """具体数据库写入实现"""
        raise NotImplementedError

    def read(self, path: str) -> Optional[Dict]:
        """从数据库读取数据"""
        with self.pool.get_connection() as conn:
            return self._execute_read(conn, path)

    def _execute_read(self, conn, path: str) -> Optional[Dict]:
        """具体数据库读取实现"""
        raise NotImplementedError


class AShareDatabaseAdapter(DatabaseAdapter):
    """A股专用数据库适配器"""

    def _init_schema(self):
        """初始化A股行情数据表"""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ashare_quotes (
                    symbol VARCHAR(6) NOT NULL,
                    trade_date DATE NOT NULL,
                    quote_time TIME NOT NULL,
                    price DECIMAL(10,2),
                    volume INTEGER,
                    limit_status VARCHAR(4),
                    PRIMARY KEY (symbol, trade_date, quote_time)
                )
            ''')
            conn.commit()

    def _execute_write(self, conn, path: str, data: Dict) -> bool:
        """写入A股行情数据"""
        symbol, date = self._parse_path(path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ashare_quotes 
            (symbol, trade_date, quote_time, price, volume, limit_status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        ''', (
            symbol,
            date,
            data['time'],
            data['price'],
            data['volume'],
            data.get('limit_status', '')
        ))

        conn.commit()
        return cursor.rowcount > 0

    def _parse_path(self, path: str) -> tuple:
        """解析路径格式: quotes/<symbol>/<date>"""
        parts = path.split('/')
        if len(parts) != 3 or parts[0] != 'quotes':
            raise ValueError("Invalid path format")
        return parts[1], parts[2]
