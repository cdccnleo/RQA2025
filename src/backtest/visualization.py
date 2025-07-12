#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测结果可视化模块
提供回测结果的图表展示功能
"""

from typing import Dict
import matplotlib.pyplot as plt
from .backtest_engine import BacktestResult

class BacktestVisualizer:
    @staticmethod
    def plot_performance(results: Dict[str, BacktestResult], save_path: str = None):
        """
        绘制策略绩效对比图
        :param results: 回测结果字典
        :param save_path: 图片保存路径(可选)
        """
        plt.figure(figsize=(12, 6))

        for strategy_name, result in results.items():
            dates = result.performance_chart["dates"]
            values = result.performance_chart["values"]
            plt.plot(dates, values, label=strategy_name)

        plt.title("Strategy Performance Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_drawdown(results: Dict[str, BacktestResult], save_path: str = None):
        """
        绘制回撤曲线
        :param results: 回测结果字典
        :param save_path: 图片保存路径(可选)
        """
        plt.figure(figsize=(12, 6))

        for strategy_name, result in results.items():
            dates = result.performance_chart["dates"]
            values = result.performance_chart["values"]

            # 计算回撤
            max_values = []
            drawdowns = []
            max_so_far = -float('inf')

            for value in values:
                if value > max_so_far:
                    max_so_far = value
                max_values.append(max_so_far)
                drawdowns.append((max_so_far - value) / max_so_far * 100)

            plt.plot(dates, drawdowns, label=strategy_name)

        plt.title("Strategy Drawdown Analysis")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_monthly_returns(results: Dict[str, BacktestResult], save_path: str = None):
        """
        绘制月度收益率热力图
        :param results: 回测结果字典
        :param save_path: 图片保存路径(可选)
        """
        # 这里简化实现，实际需要更复杂的月度收益率计算
        plt.figure(figsize=(12, 6))

        for strategy_name, result in results.items():
            dates = result.performance_chart["dates"]
            values = result.performance_chart["values"]

            # 计算月度收益率
            monthly_returns = {}
            prev_value = values[0]
            prev_month = dates[0][:7]  # 假设日期格式为YYYY-MM-DD

            for date, value in zip(dates[1:], values[1:]):
                current_month = date[:7]
                if current_month != prev_month:
                    monthly_returns[prev_month] = (value - prev_value) / prev_value * 100
                    prev_month = current_month
                    prev_value = value

            # 绘制热力图
            months = list(monthly_returns.keys())
            returns = list(monthly_returns.values())

            plt.bar(months, returns, label=strategy_name)

        plt.title("Monthly Returns")
        plt.xlabel("Month")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
