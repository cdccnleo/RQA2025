from typing import Dict, List, Tuple
import logging
from core.adapters.base_data_adapter import BaseDataAdapter
from core.exceptions import DataValidationError

logger = logging.getLogger(__name__)

class DataValidationService:
    """
    多源数据验证服务
    用于对比MiniQMT与其他数据源的数据一致性
    """

    def __init__(self, primary_adapter: BaseDataAdapter, secondary_adapters: List[BaseDataAdapter]):
        """
        初始化验证服务
        :param primary_adapter: 主数据源适配器(MiniQMT)
        :param secondary_adapters: 次数据源适配器列表(Wind/Tushare等)
        """
        self.primary = primary_adapter
        self.secondaries = secondary_adapters
        self.discrepancy_thresholds = {
            'price': 0.001,  # 0.1%
            'volume': 0.05,  # 5%
            'timestamp': 5  # 5秒
        }

    def validate_realtime_data(self, symbol: str) -> Tuple[bool, Dict]:
        """
        验证实时行情数据
        :param symbol: 股票代码
        :return: (是否验证通过, 差异详情)
        """
        primary_data = self.primary.get_realtime_data([symbol])[symbol]
        discrepancies = []

        for adapter in self.secondaries:
            try:
                secondary_data = adapter.get_realtime_data([symbol])[symbol]
                discrepancies.extend(self._compare_data(primary_data, secondary_data, adapter))
            except Exception as e:
                logger.warning(f"数据验证失败[{adapter.__class__.__name__}]: {str(e)}")

        if discrepancies:
            logger.warning(f"数据验证异常: {symbol} {discrepancies}")
            return False, {'symbol': symbol, 'discrepancies': discrepancies}

        return True, {}

    def _compare_data(self, primary: Dict, secondary: Dict, adapter: BaseDataAdapter) -> List[Dict]:
        """比较主次数据源的数据差异"""
        discrepancies = []

        # 价格验证
        price_diff = abs(primary['price'] - secondary['price'])
        if price_diff > primary['price'] * self.discrepancy_thresholds['price']:
            discrepancies.append({
                'type': 'price',
                'source': adapter.__class__.__name__,
                'primary': primary['price'],
                'secondary': secondary['price'],
                'diff': price_diff,
                'threshold': self.discrepancy_thresholds['price']
            })

        # 成交量验证
        volume_diff = abs(primary['volume'] - secondary['volume'])
        if volume_diff > primary['volume'] * self.discrepancy_thresholds['volume']:
            discrepancies.append({
                'type': 'volume',
                'source': adapter.__class__.__name__,
                'primary': primary['volume'],
                'secondary': secondary['volume'],
                'diff': volume_diff,
                'threshold': self.discrepancy_thresholds['volume']
            })

        # 时间戳验证
        time_diff = abs(primary['timestamp'] - secondary['timestamp'])
        if time_diff > self.discrepancy_thresholds['timestamp']:
            discrepancies.append({
                'type': 'timestamp',
                'source': adapter.__class__.__name__,
                'primary': primary['timestamp'],
                'secondary': secondary['timestamp'],
                'diff': time_diff,
                'threshold': self.discrepancy_thresholds['timestamp']
            })

        return discrepancies

    def batch_validate(self, symbols: List[str]) -> Dict:
        """
        批量验证多个标的
        :param symbols: 股票代码列表
        :return: 验证结果汇总
        """
        results = {
            'passed': [],
            'failed': [],
            'total': len(symbols),
            'discrepancies': []
        }

        for symbol in symbols:
            is_valid, detail = self.validate_realtime_data(symbol)
            if is_valid:
                results['passed'].append(symbol)
            else:
                results['failed'].append(symbol)
                results['discrepancies'].append(detail)

        return results
