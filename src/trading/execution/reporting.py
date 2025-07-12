#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 交易执行报告生成器
实现合规报告生成的具体逻辑
"""

from datetime import datetime, timedelta
from typing import List, Dict
import logging
from dataclasses import asdict
from fpdf import FPDF
import pandas as pd

from src.infrastructure.compliance import IReportGenerator
from .execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)

class TradingReportGenerator(IReportGenerator):
    """交易执行报告生成器具体实现"""

    def __init__(self, execution_engine: ExecutionEngine):
        self.engine = execution_engine
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.today = datetime.now().date()

    def generate_daily_report(self) -> str:
        """生成每日合规报告"""
        logger.info("开始生成交易执行合规报告")

        # 获取交易数据
        trades = self.get_today_trades()

        # 执行合规检查
        violations = self.run_compliance_checks(trades)

        # 生成报告内容
        report_content = self._generate_report_content(trades, violations)

        # 生成PDF文件
        report_path = f"reports/trading_report_{self.today.strftime('%Y%m%d')}.pdf"
        self._generate_pdf(report_content, report_path)

        logger.info(f"交易执行报告生成完成: {report_path}")
        return report_path

    def get_today_trades(self) -> List[Dict]:
        """从执行引擎获取当日交易记录"""
        return [asdict(trade) for trade in self.engine.get_trades(datetime.now().date())]

    def run_compliance_checks(self, trades: List[Dict]) -> List[Dict]:
        """执行交易合规检查"""
        return self.engine.check_compliance(trades)

    def _generate_report_content(self, trades: List[Dict], violations: List[Dict]) -> Dict:
        """生成报告内容(与原有逻辑保持一致)"""
        trade_df = pd.DataFrame(trades)
        violation_df = pd.DataFrame(violations)

        return {
            "report_date": self.today.strftime("%Y-%m-%d"),
            "trade_summary": {
                "total_trades": len(trades),
                "buy_count": sum(1 for t in trades if t['is_buy']),
                "sell_count": sum(1 for t in trades if not t['is_buy']),
                "total_volume": sum(t['quantity'] for t in trades),
                "total_value": sum(t['price'] * t['quantity'] for t in trades)
            },
            "violation_summary": {
                "total_violations": len(violations),
                "critical_count": sum(1 for v in violations if v['severity'] == "critical"),
                "major_count": sum(1 for v in violations if v['severity'] == "major"),
                "minor_count": sum(1 for v in violations if v['severity'] == "minor")
            },
            "trade_details": trade_df.to_dict(orient='records'),
            "violation_details": violation_df.to_dict(orient='records')
        }

    def _generate_pdf(self, content: Dict, output_path: str) -> None:
        """生成PDF报告(与原有逻辑保持一致)"""
        try:
            self.pdf.add_page()
            self.pdf.set_font("Arial", size=12)

            # 添加标题
            self.pdf.cell(200, 10, txt=f"RQA2025交易执行报告 - {content['report_date']}", ln=1, align='C')

            # 添加交易概览
            self.pdf.cell(200, 10, txt="交易概览", ln=1)
            self.pdf.cell(200, 10, txt=f"总交易笔数: {content['trade_summary']['total_trades']}", ln=1)

            # 保存文件
            self.pdf.output(output_path)
        except Exception as e:
            logger.error(f"生成PDF报告失败: {str(e)}")
            raise
