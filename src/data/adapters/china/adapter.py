"""
综合中国数据适配器，整合原src/data/china/adapter.py中的核心功能
实现A股特有数据格式的加载和转换
"""
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
import logging
from ..base_data_adapter import BaseDataAdapter
from .financial_adapter import FinancialDataAdapter
from .stock_adapter import StockDataAdapter

logger = logging.getLogger(__name__)

class ChinaComprehensiveAdapter(BaseDataAdapter):
    """处理中国市场的综合数据适配器"""
    
    @property
    def adapter_type(self) -> str:
        return "china_comprehensive"
        
    def __init__(self, config=None):
        super().__init__(config)
        self.financial_adapter = FinancialDataAdapter(config)
        self.stock_adapter = StockDataAdapter(config)
        self.data_dir = Path(config.get('data_dir', './data')) if config else Path('./data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def validate(self, data: DataModel) -> bool:
        """验证综合数据适配器的数据"""
        if not hasattr(data, 'raw_data'):
            return False
        return isinstance(data.raw_data, (pd.DataFrame, dict, list))
        
    def load_margin_data(self, symbol: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        加载融资融券数据
        
        Args:
            symbol: 股票代码(可选)
            start_date: 开始日期(可选)
            end_date: 结束日期(可选)
            
        Returns:
            DataFrame包含以下列:
            - date: 日期
            - symbol: 股票代码
            - margin_balance: 融资余额
            - short_balance: 融券余额
        """
        file_path = self.data_dir / "margin_data.csv"
        try:
            df = pd.read_csv(file_path)
            # 数据验证
            required_cols = ["date", "symbol", "margin_balance", "short_balance"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("融资融券数据缺少必要列")
                
            # 应用过滤条件
            if symbol:
                df = df[df['symbol'] == symbol]
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
                
            return df
        except Exception as e:
            logger.error(f"加载融资融券数据失败: {e}")
            raise
            
    def load_dragon_board_data(self, date: str = None) -> pd.DataFrame:
        """
        加载龙虎榜数据
        
        Args:
            date: 日期(可选)
            
        Returns:
            DataFrame包含以下列:
            - date: 日期
            - symbol: 股票代码  
            - buy_seat: 买入营业部
            - sell_seat: 卖出营业部
            - net_amount: 净买入额
        """
        file_path = self.data_dir / "dragon_board.csv"
        try:
            df = pd.read_csv(file_path)
            # 数据验证
            required_cols = ["date", "symbol", "buy_seat", "sell_seat", "net_amount"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError("龙虎榜数据缺少必要列")
                
            if date:
                df = df[df['date'] == date]
                
            return df
        except Exception as e:
            logger.error(f"加载龙虎榜数据失败: {e}")
            raise
            
    def load_level2_data(self, symbol: str, date: str) -> pd.DataFrame:
        """
        加载Level2行情数据
        
        Args:
            symbol: 股票代码
            date: 日期
            
        Returns:
            处理后的DataFrame，包含标准化的Level2数据
        """
        try:
            # 从数据源获取原始数据
            source = self.get_data_source('level2')
            raw_data = source.get_level2_data(symbol, date)
            
            # 实现A股特有的Level2数据处理逻辑
            df = pd.DataFrame(raw_data)
            # 添加必要的验证和转换
            return df
        except Exception as e:
            logger.error(f"处理Level2数据失败: {e}")
            raise
            
    def validate_t1_settlement(self, trades: pd.DataFrame) -> bool:
        """
        验证T+1结算规则
        
        Args:
            trades: 交易数据，包含symbol, date, action等列
            
        Returns:
            bool: 是否所有交易都符合T+1规则
        """
        try:
            # 检查买入后次日才能卖出
            buy_dates = trades[trades['action'] == 'BUY']['date']
            sell_dates = trades[trades['action'] == 'SELL']['date']
            return all(sd > bd for sd in sell_dates for bd in buy_dates)
        except Exception as e:
            logger.error(f"验证T+1结算失败: {e}")
            raise

    def get_adjusted_factors(self, symbol: str) -> Dict[str, float]:
        """
        获取复权因子
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含前后复权因子的字典:
            - 'forward': 前复权因子
            - 'backward': 后复权因子
        """
        try:
            return self.stock_adapter.get_adjusted_factors(symbol)
        except Exception as e:
            logger.error(f"获取复权因子失败: {e}")
            raise

    def get_price_limits(self, symbol: str) -> Dict[str, float]:
        """
        获取涨跌停价格
        
        Args:
            symbol: 股票代码
            
        Returns:
            包含价格限制的字典:
            - 'upper_limit': 涨停价
            - 'lower_limit': 跌停价
        """
        try:
            return self.stock_adapter.get_price_limits(symbol)
        except Exception as e:
            logger.error(f"获取涨跌停价格失败: {e}")
            raise
