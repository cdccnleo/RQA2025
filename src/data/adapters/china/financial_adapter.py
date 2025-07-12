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

class FinancialDataAdapter(BaseDataAdapter):
    
    @property
    def adapter_type(self) -> str:
        return "china_financial"
    """A股财务数据适配器"""

    # 财务报表类型
    REPORT_TYPES = {
        'balance': '资产负债表',
        'income': '利润表',
        'cashflow': '现金流量表'
    }

    # 报告频率
    FREQUENCIES = {
        'q1': '一季度',
        'q2': '半年度',
        'q3': '三季度',
        'annual': '年度'
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
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        report_type: str = 'income',
        frequency: str = 'annual',
        data_source: str = 'jqdata',
        **kwargs
    ) -> DataModel:
        """
        加载A股财务数据

        Args:
            symbols: 股票代码或代码列表
            start_date: 开始日期
            end_date: 结束日期
            report_type: 报表类型(balance/income/cashflow)
            frequency: 报告频率(q1/q2/q3/annual)
            data_source: 数据源(jqdata/tushare)
            **kwargs: 其他参数

        Returns:
            DataModel: 包含财务数据和元数据的对象
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # 验证报表类型和频率
        if report_type not in self.REPORT_TYPES:
            raise ValueError(f"无效的报表类型: {report_type}")
        if frequency not in self.FREQUENCIES:
            raise ValueError(f"无效的报告频率: {frequency}")

        tasks = [{
            'func': self._load_single_stock_financial,
            'kwargs': {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'report_type': report_type,
                'frequency': frequency,
                'data_source': data_source,
                **kwargs
            }
        } for symbol in symbols]

        results = ParallelDataLoader().load(tasks)
        data = pd.concat(results)

        metadata = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'report_type': report_type,
            'frequency': frequency,
            'data_source': data_source
        }

        return DataModel(data, metadata)

    def _load_single_stock_financial(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        report_type: str,
        frequency: str,
        data_source: str,
        **kwargs
    ) -> pd.DataFrame:
        """加载单只股票财务数据"""
        cache_key = self._generate_cache_key(symbol, start_date, end_date, report_type, frequency)
        cached_data = DataCache().get(cache_key)
        if cached_data is not None:
            return cached_data

        source = self.data_sources.get(data_source)
        if source is None:
            raise ValueError(f"无效的数据源: {data_source}")

        # 根据报表类型加载数据
        if report_type == 'balance':
            raw_data = source.load_balance_sheet(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                **kwargs
            )
        elif report_type == 'income':
            raw_data = source.load_income_statement(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                **kwargs
            )
        elif report_type == 'cashflow':
            raw_data = source.load_cashflow_statement(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                **kwargs
            )
        else:
            raise ValueError(f"未知的报表类型: {report_type}")

        # 数据验证
        if not self._validate_financial_data(raw_data, report_type):
            raise ValueError(f"财务数据验证失败: {symbol}")

        # 处理A股特有财务指标
        processed_data = self._process_financial_data(raw_data, report_type)
        DataCache().set(cache_key, processed_data)

        return processed_data

    def validate(self, data: DataModel) -> bool:
        """验证财务数据完整性"""
        if not hasattr(data, 'raw_data') or not isinstance(data.raw_data, pd.DataFrame):
            return False
            
        report_type = data.metadata.get('report_type')
        if not report_type:
            return False
            
        if report_type == 'balance':
            required = ['total_assets', 'total_liabilities', 'equity']
        elif report_type == 'income':
            required = ['revenue', 'net_income', 'eps']
        elif report_type == 'cashflow':
            required = ['operating_cashflow', 'investing_cashflow', 'financing_cashflow']
        else:
            return False

        return all(col in data.raw_data.columns for col in required)

    def _process_financial_data(self, data: pd.DataFrame, report_type: str) -> pd.DataFrame:
        """处理A股特有财务指标"""
        # 添加报告日期和频率标记
        if 'report_date' not in data.columns:
            data['report_date'] = data.index

        # 计算常用财务比率
        if report_type == 'balance':
            if 'debt_to_equity' not in data.columns:
                data['debt_to_equity'] = data['total_liabilities'] / data['equity']

        elif report_type == 'income':
            if 'gross_margin' not in data.columns:
                data['gross_margin'] = (data['revenue'] - data['cogs']) / data['revenue']

        return data

    def _generate_cache_key(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        report_type: str,
        frequency: str
    ) -> str:
        """生成缓存键"""
        return f"financial_{symbol}_{start_date}_{end_date}_{report_type}_{frequency}"
