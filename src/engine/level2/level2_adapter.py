"""Level2行情处理器模块"""
import asyncio
from typing import Dict, List, Optional
import logging
from .real_time_engine import RealTimeEngine

logger = logging.getLogger(__name__)

class Level2Processor:
    """A股Level2行情适配器"""

    def __init__(self, engine: RealTimeEngine):
        self.engine = engine
        self.symbol_map = {}  # 股票代码映射表
        self.last_prices = {}  # 最新价格缓存

        # A股特有状态
        self.limit_status = {
            'up': set(),
            'down': set(),
            'normal': set()
        }

    async def connect(self, source: str):
        """连接数据源"""
        logger.info(f"Connecting to Level2 source: {source}")
        # TODO: 实现具体连接逻辑
        await asyncio.sleep(0.1)  # 模拟连接延迟

    async def subscribe(self, symbols: List[str]):
        """订阅股票行情"""
        for symbol in symbols:
            # 标准化A股代码
            normalized = self._normalize_symbol(symbol)
            self.symbol_map[normalized] = symbol
            logger.debug(f"Subscribed to {symbol} (normalized: {normalized})")

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化A股代码格式"""
        if symbol.startswith(('6', '9')):
            return f"{symbol}.SH"
        elif symbol.startswith(('0', '3')):
            return f"{symbol}.SZ"
        return symbol

    async def start(self):
        """启动数据处理"""
        logger.info("Starting Level2 data processing")
        # TODO: 实现具体数据接收和处理逻辑
        while True:
            await asyncio.sleep(0.01)  # 模拟数据接收间隔

    def _process_raw_data(self, raw_data: Dict):
        """处理原始Level2数据"""
        # 标准化数据格式
        normalized = {
            'type': 'order_book',
            'symbol': self.symbol_map.get(raw_data['code'], raw_data['code']),
            'timestamp': raw_data['timestamp'],
            'bids': self._parse_level2_bids(raw_data['bid']),
            'asks': self._parse_level2_asks(raw_data['ask']),
            'status': self._parse_trading_status(raw_data)
        }

        # 更新最新价格
        if normalized['bids']:
            self.last_prices[normalized['symbol']] = normalized['bids'][0][0]

        # 推送处理后的数据
        self.engine.feed_data(normalized)

        # 更新涨跌停状态
        self._update_limit_status(normalized['symbol'], normalized['status'])

    def _parse_level2_bids(self, raw_bids: List) -> List[tuple]:
        """解析买档数据"""
        return [
            (float(price)/10000.0, int(volume))
            for price, volume in raw_bids[:10]  # 取前10档
        ]

    def _parse_level2_asks(self, raw_asks: List) -> List[tuple]:
        """解析卖档数据"""
        return [
            (float(price)/10000.0, int(volume))
            for price, volume in raw_asks[:10]  # 取前10档
        ]

    def _parse_trading_status(self, raw_data: Dict) -> str:
        """解析交易状态"""
        if raw_data.get('limit_up'):
            return 'up'
        elif raw_data.get('limit_down'):
            return 'down'
        return 'normal'

    def _update_limit_status(self, symbol: str, status: str):
        """更新涨跌停状态"""
        # 从原有状态集合中移除
        for key in self.limit_status:
            if symbol in self.limit_status[key]:
                self.limit_status[key].remove(symbol)

        # 添加到新状态集合
        self.limit_status[status].add(symbol)

    def get_limit_status(self, symbol: str) -> Optional[str]:
        """获取股票的涨跌停状态"""
        for status, symbols in self.limit_status.items():
            if symbol in symbols:
                return status
        return None

    def get_last_price(self, symbol: str) -> Optional[float]:
        """获取最新成交价"""
        return self.last_prices.get(symbol)

async def create_and_start_adapter(engine: RealTimeEngine) -> Level2Adapter:
    """创建并启动适配器示例"""
    adapter = Level2Adapter(engine)
    await adapter.connect("xtp")  # 使用XTP行情源
    await adapter.subscribe(["600519", "000001"])  # 订阅茅台和平安
    return adapter
