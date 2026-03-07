import time
import logging
from typing import Dict, List

# 条件导入，支持测试环境
try:
    from xtquant import xtdata
    XTQUANT_AVAILABLE = True
except ImportError:
    XTQUANT_AVAILABLE = False
    # 创建mock对象用于测试

    class MockXtData:

        def __init__(self):

            pass

        def init(self):

            pass

        def get_full_tick(self, symbols):

            return {}
    xtdata = MockXtData()

from ...base_adapter import BaseDataAdapter
from src.infrastructure.error.exceptions import DataFetchError

logger = logging.getLogger(__name__)


class MiniQMTDataAdapter(BaseDataAdapter):

    """MiniQMT行情数据适配器"""

    adapter_type = "miniqmt"

    def __init__(self, config: Dict):

        self.config = config
        self.xt = None
        if XTQUANT_AVAILABLE:
            self._connect()

    def _connect(self):
        """初始化MiniQMT连接"""
        if not XTQUANT_AVAILABLE:
            logger.warning("xtquant模块不可用，使用mock模式")
            return

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
        if not XTQUANT_AVAILABLE:
            logger.warning("xtquant模块不可用，返回mock数据")
            return self._get_mock_data(symbols)

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
                'price': data.get('price', 0),
                'volume': data.get('volume', 0),
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0),
                'timestamp': data.get('timestamp', time.time())
            }
        return formatted

    def _get_mock_data(self, symbols: List[str]) -> Dict:
        """获取mock数据用于测试"""
        mock_data = {}
        for symbol in symbols:
            mock_data[symbol] = {
                'price': 10.0,
                'volume': 1000,
                'bid': 9.99,
                'ask': 10.01,
                'timestamp': time.time()
            }
        return mock_data

    def _reconnect(self):
        """断线重连机制"""
        if XTQUANT_AVAILABLE:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"重连失败: {str(e)}")
                raise ConnectionError("MiniQMT重连失败")

    def validate(self, data: Dict) -> bool:
        """验证数据有效性"""
        # 实现MiniQMT特有的数据验证逻辑
        return True
