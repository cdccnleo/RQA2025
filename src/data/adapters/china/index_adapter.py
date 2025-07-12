import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from ...adapters.base_data_adapter import BaseDataAdapter
from ...core.data_model import DataModel
from ....cache.data_cache import DataCache
from ....loader.parallel_loader import ParallelDataLoader
from ....monitoring.quality.checker import DataQualityChecker

logger = logging.getLogger(__name__)

class IndexDataAdapter(BaseDataAdapter):
    
    @property
    def adapter_type(self) -> str:
        return "china_index"
    """A股指数数据适配器"""

    # 主要A股指数代码
    MAIN_INDEXES = {
        '000001.SH': '上证指数',
        '399001.SZ': '深证成指',
        '399006.SZ': '创业板指',
        '000016.SH': '上证50',
        '000300.SH': '沪深300',
        '000905.SH': '中证500'
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_checker = DataQualityChecker()
        self._init_data_sources()

    def _init_data_sources(self):
        """初始化数据源连接"""
        self.data_sources = {
            'jqdata': self._init_jqdata(),
            'tushare': self._init_tushare()
        }

    def load(
        self,
        index_codes: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str = 'daily',
        data_source: str = 'jqdata',
        **kwargs
    ) -> DataModel:
        """
        加载A股指数数据

        Args:
            index_codes: 指数代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型(daily/constituents/weights)
            data_source: 数据源(jqdata/tushare)
            **kwargs: 其他参数

        Returns:
            DataModel: 包含指数数据和元数据的对象
        """
        if isinstance(index_codes, str):
            index_codes = [index_codes]

        # 验证指数代码
        for code in index_codes:
            if code not in self.MAIN_INDEXES:
                logger.warning(f"未知的指数代码: {code}")

        tasks = [{
            'func': self._load_single_index,
            'kwargs': {
                'index_code': code,
                'start_date': start_date,
                'end_date': end_date,
                'data_type': data_type,
                'data_source': data_source,
                **kwargs
            }
        } for code in index_codes]

        results = ParallelDataLoader().load(tasks)
        data = pd.concat(results)

        metadata = {
            'index_codes': index_codes,
            'start_date': start_date,
            'end_date': end_date,
            'data_type': data_type,
            'data_source': data_source
        }

        return DataModel(data, metadata)

    def _load_single_index(
        self,
        index_code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str,
        data_source: str,
        **kwargs
    ) -> pd.DataFrame:
        """加载单个指数数据"""
        cache_key = self._generate_cache_key(index_code, start_date, end_date, data_type)
        cached_data = DataCache().get(cache_key)
        if cached_data is not None:
            return cached_data

        source = self.data_sources.get(data_source)
        if source is None:
            raise ValueError(f"无效的数据源: {data_source}")

        if data_type == 'daily':
            raw_data = source.load_index_daily(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
        elif data_type == 'constituents':
            raw_data = source.load_index_constituents(
                index_code=index_code,
                date=end_date,
                **kwargs
            )
        elif data_type == 'weights':
            raw_data = source.load_index_weights(
                index_code=index_code,
                date=end_date,
                **kwargs
            )
        else:
            raise ValueError(f"未知的指数数据类型: {data_type}")

        # 数据验证
        if not self._validate_index_data(raw_data, data_type):
            raise ValueError(f"指数数据验证失败: {index_code}")

        processed_data = self._process_index_data(raw_data, data_type)
        DataCache().set(cache_key, processed_data)

        return processed_data

    def validate(self, data: DataModel) -> bool:
        """验证指数数据完整性"""
        if not hasattr(data, 'raw_data') or not isinstance(data.raw_data, pd.DataFrame):
            return False
            
        data_type = data.metadata.get('data_type')
        if not data_type:
            return False
            
        if data_type == 'daily':
            required = ['open', 'high', 'low', 'close', 'volume']
        elif data_type in ['constituents', 'weights']:
            required = ['symbol', 'name']
        else:
            return False

        return all(col in data.raw_data.columns for col in required)

    def _process_index_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """处理指数特有字段"""
        if data_type == 'daily':
            # 添加指数涨跌幅
            if 'pct_change' not in data.columns:
                data['pct_change'] = data['close'].pct_change()

        elif data_type == 'constituents':
            # 标准化成分股代码
            if 'symbol' in data.columns:
                data['symbol'] = data['symbol'].str.upper()

        return data

    def _generate_cache_key(
        self,
        index_code: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_type: str
    ) -> str:
        """生成缓存键"""
        return f"index_{index_code}_{start_date}_{end_date}_{data_type}"
