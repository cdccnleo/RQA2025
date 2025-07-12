#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import Dict, Optional
from ..core import StorageAdapter

class FileSystemAdapter(StorageAdapter):
    """本地文件系统存储适配器"""

    def __init__(self, base_path: str = "data/storage"):
        self.base_path = Path(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def write(self, path: str, data: Dict) -> bool:
        """写入数据到文件系统"""
        full_path = self.base_path / path
        os.makedirs(full_path.parent, exist_ok=True)

        try:
            with open(full_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (IOError, TypeError):
            return False

    def read(self, path: str) -> Optional[Dict]:
        """从文件系统读取数据"""
        full_path = self.base_path / path
        full_path = full_path.with_suffix('.json')

        if not full_path.exists():
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            return None


class AShareFileSystemAdapter(FileSystemAdapter):
    """A股专用文件存储适配器"""

    def save_quote(self, symbol: str, data: Dict) -> bool:
        """存储A股行情数据（带涨跌停标记）"""
        if not symbol.startswith(('6', '3', '0')):
            raise ValueError("非A股股票代码")

        # 添加市场标记
        data['market'] = 'SH' if symbol.startswith(('6', '3')) else 'SZ'
        return self.write(f"quotes/{symbol}", data)
