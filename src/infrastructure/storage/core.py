#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import Dict, Optional
from datetime import datetime
from ..monitoring import StorageMonitor

class QuoteStorage:
    """A股行情数据存储核心类"""

    def __init__(self, adapter: 'StorageAdapter'):
        self._adapter = adapter
        self._lock = threading.RLock()
        self._monitor = StorageMonitor()
        self._trading_hours = {
            'morning': (9*60+30, 11*60+30),  # 9:30-11:30
            'afternoon': (13*60, 15*60)      # 13:00-15:00
        }

    def save_quote(self, symbol: str, data: Dict) -> bool:
        """
        存储单支股票行情数据(线程安全)

        Args:
            symbol: 股票代码(如600519)
            data: {
                "time": "09:30:00",  # 行情时间
                "price": 1720.5,     # 当前价格
                "volume": 1500,      # 成交量(手)
                "limit_status": ""   # 涨跌停状态(up/down)
            }

        Returns:
            bool: 是否存储成功
        """
        with self._lock:
            try:
                path = f"quotes/{symbol}/{datetime.now().date()}"
                success = self._adapter.write(path, data)
                self._monitor.record_write(
                    symbol=symbol,
                    size=len(str(data)),
                    status=success
                )

                # 标记涨跌停状态
                if data.get('limit_status'):
                    self._set_limit_status(symbol, data['limit_status'])

                return success
            except Exception as e:
                self._monitor.record_error(symbol)
                raise

    def _set_limit_status(self, symbol: str, status: str):
        """内部方法：记录涨跌停状态"""
        status_path = f"limit_status/{symbol}"
        self._adapter.write(status_path, {
            "status": status,
            "timestamp": datetime.now().timestamp()
        })

    def get_quote(self, symbol: str, date: str) -> Optional[Dict]:
        """
        读取行情数据
        """
        path = f"quotes/{symbol}/{date}"
        return self._adapter.read(path)


class StorageAdapter:
    """存储适配器抽象基类"""

    def write(self, path: str, data: Dict) -> bool:
        raise NotImplementedError

    def read(self, path: str) -> Optional[Dict]:
        raise NotImplementedError
