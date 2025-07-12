#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 合规报告接口定义
定义合规报告生成的抽象接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional

class IReportGenerator(ABC):
    """合规报告生成器抽象接口"""

    @abstractmethod
    def generate_daily_report(self) -> str:
        """生成每日合规报告"""
        pass

    @abstractmethod
    def get_today_trades(self) -> List[Dict]:
        """获取当日交易记录"""
        pass

    @abstractmethod
    def run_compliance_checks(self, trades: List[Dict]) -> List[Dict]:
        """执行合规检查"""
        pass
