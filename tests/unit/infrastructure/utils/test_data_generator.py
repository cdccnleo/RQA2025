#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQA2025 测试数据生成工具
用于生成单元测试和集成测试所需的模拟数据
"""

import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

fake = Faker('zh_CN')

class TestDataGenerator:
    @staticmethod
    def generate_market_data(symbol, days=30, anomalies=False):
        """
        生成模拟行情数据
        :param symbol: 股票代码
        :param days: 生成天数
        :param anomalies: 是否包含异常数据
        :return: 行情数据列表
        """
        base_price = random.uniform(10, 500)
        data = []

        for i in range(days):
            # 基础数据
            record = {
                "symbol": symbol,
                "date": (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d"),
                "open": round(base_price * random.uniform(0.95, 1.05), 2),
                "high": 0,
                "low": 0,
                "close": 0,
                "volume": random.randint(100000, 1000000)
            }

            # 计算高低收
            record["high"] = round(record["open"] * random.uniform(1.0, 1.1), 2)
            record["low"] = round(record["open"] * random.uniform(0.9, 1.0), 2)
            record["close"] = round(random.uniform(record["low"], record["high"]), 2)

            # 处理异常场景
            if anomalies and i % 7 == 0:
                if random.choice([True, False]):
                    # 暴涨
                    record["high"] = round(record["open"] * 1.2, 2)
                    record["close"] = record["high"]
                else:
                    # 暴跌
                    record["low"] = round(record["open"] * 0.8, 2)
                    record["close"] = record["low"]

            data.append(record)

            # 更新基础价格
            base_price = record["close"]

        return data

    @staticmethod
    def generate_order_book(symbol, levels=10, anomalies=False):
        """
        生成模拟Level2订单簿数据
        :param symbol: 股票代码
        :param levels: 买卖档位
        :param anomalies: 是否包含异常数据
        :return: 订单簿字典
        """
        mid_price = random.uniform(10, 500)
        spread = random.uniform(0.01, 0.05)

        order_book = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "bids": [],
            "asks": []
        }

        # 生成买盘
        for i in range(levels):
            price = mid_price * (1 - spread * (i+1))
            size = random.randint(10, 100)
            order_book["bids"].append({
                "price": round(price, 2),
                "size": size
            })

        # 生成卖盘
        for i in range(levels):
            price = mid_price * (1 + spread * (i+1))
            size = random.randint(10, 100)
            order_book["asks"].append({
                "price": round(price, 2),
                "size": size
            })

        # 添加异常情况
        if anomalies:
            if random.choice([True, False]):
                # 买卖盘失衡
                order_book["bids"] = order_book["bids"][:2]  # 仅保留2档买盘
            else:
                # 隐藏大单
                order_book["bids"][3]["size"] *= 10

        return order_book

    @staticmethod
    def generate_news(count=100, positive_ratio=0.7):
        """
        生成模拟新闻数据
        :param count: 生成数量
        :param positive_ratio: 正面新闻比例
        :return: 新闻列表
        """
        news = []
        sectors = ["科技", "金融", "消费", "医药", "能源"]

        for _ in range(count):
            is_positive = random.random() < positive_ratio
            sector = random.choice(sectors)

            if is_positive:
                title = f"{sector}行业迎来重大利好：{fake.sentence()}"
            else:
                title = f"{sector}板块面临调整：{fake.sentence()}"

            news.append({
                "title": title,
                "content": fake.text(),
                "publish_time": fake.date_time_this_month().isoformat(),
                "sector": sector,
                "sentiment": 1 if is_positive else -1
            })

        return news

    @staticmethod
    def generate_crash_scenario():
        """生成股灾模拟场景"""
        symbols = ["600519.SH", "000001.SZ", "601318.SH"]
        start_price = 1800
        crash_data = []

        for minute in range(30):  # 30分钟暴跌
            timestamp = (datetime.now() - timedelta(minutes=30-minute)).isoformat()
            drop_rate = min(0.3, minute * 0.01)  # 每分钟下跌1%，最大30%

            for symbol in symbols:
                crash_data.append({
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": round(start_price * (1 - drop_rate), 2),
                    "volume": int(100000 * (1 + minute * 0.5))  # 成交量逐步放大
                })

        return crash_data
