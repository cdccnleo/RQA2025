import time

from xtquant import xtdata
from core.adapters.base_data_adapter import BaseDataAdapter
from core.exceptions import DataFetchError
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class MiniQMTDataAdapter(BaseDataAdapter):
    """MiniQMT行情数据适配器"""

    adapter_type = "miniqmt"

    def __init__(self, config: Dict):
        self.config = config
        self.xt = None
        self._connect()

    def _connect(self):
        """初始化MiniQMT连接"""
        try:
            self.xt = xtdata
            self.xt.init()
            logger.info("MiniQMT数据接口初始化成功")
        except Exception as e:
            logger.error(f"MiniQMT数据接口初始化失败: {str(e)}")
            raise ConnectionError("MiniQMT连接失败")

    def get_realtime_data(self, symbols: List[str]) -> Dict:
        """
        获取实时行情数据
        :param symbols: 股票代码列表
        :return: 行情数据字典
        """
        if not self.xt:
            self._reconnect()

        try:
            tick_data = self.xt.get_full_tick(symbols)
            return self._format_data(tick_data)
        except Exception as e:
            logger.error(f"获取实时行情失败: {str(e)}")
            raise DataFetchError("MiniQMT实时数据获取失败")

    def _format_data(self, raw_data: Dict) -> Dict:
        """格式化原始数据为统一格式"""
        formatted = {}
        for symbol, data in raw_data.items():
            formatted[symbol] = {
                'price': data['price'],
                'volume': data['volume'],
                'bid': data['bid'],
                'ask': data['ask'],
                'timestamp': data['timestamp']
            }
        return formatted

    def _reconnect(self):
        """断线重连机制"""
        max_retries = 3
        for i in range(max_retries):
            try:
                self._connect()
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise
                time.sleep(1)

    def validate(self, data: Dict) -> bool:
        """验证数据有效性"""
        # 实现MiniQMT特有的数据验证逻辑
        return True
