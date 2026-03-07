import logging
#!/usr/bin/env python3
"""
RQA2025 数据层客户端 SDK

from src.engine.logging.unified_logger import get_unified_logger
提供简单易用的Python客户端SDK
支持数据加载、质量监控、性能指标等功能
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import websockets

# 配置日志
logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


class RQA2025DataClient:

    """RQA2025 数据客户端"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        初始化客户端

        Args:
            base_url: API基础URL
            api_key: API密钥（可选）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.websocket = None

        # 请求头
        self.headers = {
            'Content - Type': 'application / json',
            'User - Agent': 'RQA2025 - Client / 1.0'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / health") as response:
            return await response.json()

    async def list_data_sources(self) -> Dict[str, Any]:
        """获取数据源列表"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / sources") as response:
            return await response.json()

    async def get_data_source_info(self, source_type: str) -> Dict[str, Any]:
        """获取数据源信息"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / sources/{source_type}") as response:
            return await response.json()

    async def load_data(
        self,
        source_type: str,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "1d"
    ) -> Dict[str, Any]:
        """
        加载数据

        Args:
            source_type: 数据源类型
            symbol: 数据符号
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率
        """
        payload = {
            "source_type": source_type,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "frequency": frequency
        }

        async with self.session.post(
            f"{self.base_url}/api / v1 / data / load",
            json=payload
        ) as response:
            return await response.json()

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / performance") as response:
            return await response.json()

    async def check_data_quality(
        self,
        source_type: str,
        symbol: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检查数据质量

        Args:
            source_type: 数据源类型
            symbol: 数据符号
            metrics: 质量指标列表
        """
        payload = {
            "source_type": source_type,
            "symbol": symbol,
            "metrics": metrics or []
        }

        async with self.session.post(
            f"{self.base_url}/api / v1 / data / quality",
            json=payload
        ) as response:
            return await response.json()

    async def generate_quality_report(
        self,
        days: int = 7,
        source_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成质量报告

        Args:
            days: 报告天数
            source_type: 数据源类型
        """
        params = {"days": days}
        if source_type:
            params["source_type"] = source_type

        async with self.session.get(
            f"{self.base_url}/api / v1 / data / quality / report",
            params=params
        ) as response:
            return await response.json()

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / cache / stats") as response:
            return await response.json()

    async def clear_cache(self) -> Dict[str, Any]:
        """清除缓存"""
        async with self.session.post(f"{self.base_url}/api / v1 / data / cache / clear") as response:
            return await response.json()

    async def get_alerts(self) -> Dict[str, Any]:
        """获取告警信息"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / alerts") as response:
            return await response.json()

    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """获取仪表板指标"""
        async with self.session.get(f"{self.base_url}/api / v1 / data / metrics / dashboard") as response:
            return await response.json()

    async def connect_websocket(self, channel: str):
        """连接WebSocket"""
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws/{channel}"
        self.websocket = await websockets.connect(ws_url)
        logger.info(f"WebSocket连接建立: {channel}")
        return self.websocket

    async def subscribe_market_data(self, callback):
        """订阅市场数据"""
        ws = await self.connect_websocket("market_data")

        try:
            async for message in ws:
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
        finally:
            await ws.close()

    async def subscribe_quality_monitor(self, callback):
        """订阅质量监控数据"""
        ws = await self.connect_websocket("quality_monitor")

        try:
            async for message in ws:
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
        finally:
            await ws.close()

    async def subscribe_performance_monitor(self, callback):
        """订阅性能监控数据"""
        ws = await self.connect_websocket("performance_monitor")

        try:
            async for message in ws:
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
        finally:
            await ws.close()

    async def subscribe_alerts(self, callback):
        """订阅告警数据"""
        ws = await self.connect_websocket("alerts")

        try:
            async for message in ws:
                data = json.loads(message)
                await callback(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
        finally:
            await ws.close()


class DataQualityAnalyzer:

    """数据质量分析器"""

    def __init__(self, client: RQA2025DataClient):

        self.client = client

    async def analyze_data_quality(
        self,
        source_type: str,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        分析数据质量

        Args:
            source_type: 数据源类型
            symbol: 数据符号
            start_date: 开始日期
            end_date: 结束日期
        """
        # 加载数据
        data_result = await self.client.load_data(
            source_type=source_type,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        # 检查数据质量
        quality_result = await self.client.check_data_quality(
            source_type=source_type,
            symbol=symbol
        )

        # 生成质量报告
        report_result = await self.client.generate_quality_report(
            days=30,
            source_type=source_type
        )

        return {
            "data": data_result,
            "quality": quality_result,
            "report": report_result,
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def get_quality_trends(
        self,
        source_type: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        获取质量趋势

        Args:
            source_type: 数据源类型
            days: 天数
        """
        report = await self.client.generate_quality_report(
            days=days,
            source_type=source_type
        )

        return {
            "source_type": source_type,
            "trends": report,
            "analysis_period": f"最近{days}天"
        }


class PerformanceAnalyzer:

    """性能分析器"""

    def __init__(self, client: RQA2025DataClient):

        self.client = client

    async def analyze_performance(self) -> Dict[str, Any]:
        """分析系统性能"""
        # 获取性能指标
        performance = await self.client.get_performance_metrics()

        # 获取缓存统计
        cache_stats = await self.client.get_cache_statistics()

        # 获取仪表板指标
        dashboard = await self.client.get_dashboard_metrics()

        return {
            "performance": performance,
            "cache": cache_stats,
            "dashboard": dashboard,
            "analysis_timestamp": datetime.now().isoformat()
        }

    async def monitor_performance_trends(self, duration_minutes: int = 60):
        """
        监控性能趋势

        Args:
            duration_minutes: 监控时长（分钟）
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        trends = []

        while datetime.now() < end_time:
            try:
                performance = await self.client.get_performance_metrics()
                trends.append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": performance
                })

                await asyncio.sleep(60)  # 每分钟记录一次

            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(60)

        return {
            "monitoring_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "trends": trends,
            "total_records": len(trends)
        }


# 使用示例
async def example_usage():
    """使用示例"""
    async with RQA2025DataClient() as client:
        # 健康检查
        health = await client.health_check()
        print(f"系统健康状态: {health}")

        # 获取数据源列表
        sources = await client.list_data_sources()
        print(f"可用数据源: {sources}")

        # 加载加密货币数据
        crypto_data = await client.load_data(
            source_type="crypto",
            symbol="BTC",
            start_date="2024 - 01 - 01",
            end_date="2024 - 01 - 31"
        )
        print(f"加密货币数据: {crypto_data}")

        # 检查数据质量
        quality = await client.check_data_quality(
            source_type="crypto",
            symbol="BTC"
        )
        print(f"数据质量: {quality}")

        # 获取性能指标
        performance = await client.get_performance_metrics()
        print(f"性能指标: {performance}")


async def websocket_example():
    """WebSocket使用示例"""
    client = RQA2025DataClient()

    async def market_data_callback(data):
        """市场数据回调"""
        print(f"收到市场数据: {data}")

    async def quality_callback(data):
        """质量数据回调"""
        print(f"收到质量数据: {data}")

    # 订阅市场数据
    await client.subscribe_market_data(market_data_callback)

    # 订阅质量监控
    await client.subscribe_quality_monitor(quality_callback)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
