"""
中国市场数据适配器实现层

职责定位：
1. 实现中国市场特定的数据适配器
2. 继承自 adapters/china/BaseChinaAdapter
3. 包含A股、科创板等市场特定功能
"""

from typing import Dict, Optional
import pandas as pd
from abc import ABC, abstractmethod
import logging

# 定义BaseChinaAdapter基类
class BaseChinaAdapter(ABC):
    """
    中国市场数据适配器基类

    提供中国市场数据适配器的通用功能和接口定义。
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化适配器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        连接数据源

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            bool: 断开连接是否成功
        """
        pass

    @abstractmethod
    def get_data(self, symbol: str, **kwargs) -> Dict:
        """
        获取数据

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict: 数据字典
        """
        pass

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._is_connected

    def ping(self) -> bool:
        """ping测试连接"""
        try:
            return self._is_connected
        except Exception as e:
            self.logger.error(f"Ping测试失败: {e}")
            return False


class AStockAdapter(BaseChinaAdapter):
    """
    A股基础数据适配器
    
    实现中国A股市场的基础数据获取功能。
    推荐使用此适配器替代传统的 ChinaStockDataAdapter。
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化适配器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self._is_connected = False

    def connect(self) -> bool:
        """连接数据源"""
        try:
            self.logger.info("连接A股数据源")
            self._is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self.logger.info("断开A股数据源连接")
            self._is_connected = False
            return True
        except Exception as e:
            self.logger.error(f"断开连接失败: {e}")
            return False

    def get_data(self, symbol: str, **kwargs) -> Dict:
        """
        获取数据

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict: 数据字典
        """
        return {
            'symbol': symbol,
            'market': 'A股',
            'data_type': kwargs.get('data_type', 'basic')
        }

    def get_stock_basic(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票基础信息

        Args:
            symbol: 股票代码(可选)

        Returns:
            pd.DataFrame: 股票基础信息DataFrame
        """
        try:
            import akshare as ak
            
            if symbol:
                # 获取单只股票信息
                try:
                    stock_info = ak.stock_individual_info_em(symbol=symbol)
                    self.logger.info(f"成功获取股票 {symbol} 基础信息")
                    return stock_info
                except Exception as e:
                    self.logger.warning(f"stock_individual_info_em失败，尝试备用方法: {e}")
                    # 备用方法：从股票列表中查找
                    stock_list = ak.stock_info_a_code_name()
                    if symbol in stock_list['code'].values:
                        return stock_list[stock_list['code'] == symbol]
                    return pd.DataFrame()
            else:
                # 获取所有A股股票列表
                stock_list = ak.stock_info_a_code_name()
                self.logger.info(f"成功获取A股股票列表，共 {len(stock_list)} 只股票")
                return stock_list
        except ImportError:
            self.logger.error("akshare未安装，无法获取股票基础信息")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取股票基础信息失败: {e}")
            return pd.DataFrame()

    def get_daily_quotes(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线行情数据

        Args:
            symbol: 股票代码
            start_date: 开始日期(YYYY-MM-DD)
            end_date: 结束日期(YYYY-MM-DD)

        Returns:
            pd.DataFrame: 日线行情DataFrame
        """
        try:
            import akshare as ak
            
            # 转换日期格式
            start_fmt = start_date.replace("-", "")
            end_fmt = end_date.replace("-", "")
            
            # 尝试使用主要接口
            try:
                df = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start_fmt,
                    end_date=end_fmt,
                    adjust="qfq"  # 前复权
                )
                if df is not None and not df.empty:
                    self.logger.info(f"成功获取股票 {symbol} 日线数据，共 {len(df)} 条")
                    return df
            except Exception as e:
                self.logger.debug(f"stock_zh_a_daily失败，尝试备用接口: {e}")
            
            # 备用接口
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                self.logger.info(f"成功获取股票 {symbol} 日线数据（备用接口），共 {len(df)} 条")
                return df
            
            self.logger.warning(f"股票 {symbol} 日线数据为空")
            return pd.DataFrame()
            
        except ImportError:
            self.logger.error("akshare未安装，无法获取日线行情")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取日线行情失败: {e}")
            return pd.DataFrame()

    def get_financial_data(self, symbol: str, report_type: str = 'annual') -> pd.DataFrame:
        """
        获取财务数据

        Args:
            symbol: 股票代码
            report_type: 报告类型(annual/quarterly)

        Returns:
            pd.DataFrame: 财务数据DataFrame
        """
        # 实现获取财务数据的逻辑
        return pd.DataFrame()

    def get_stock_basic_info(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票基本信息

        Args:
            symbol: 股票代码(可选)

        Returns:
            pd.DataFrame: 股票基本信息DataFrame
        """
        try:
            import akshare as ak
            
            if symbol:
                # 获取单只股票信息
                try:
                    stock_info = ak.stock_individual_info_em(symbol=symbol)
                    self.logger.info(f"成功获取股票 {symbol} 基础信息")
                    return self._normalize_stock_basic_info(stock_info, symbol)
                except Exception as e:
                    self.logger.warning(f"stock_individual_info_em失败，尝试备用方法: {e}")
                    # 备用方法：从股票列表中查找
                    stock_list = ak.stock_info_a_code_name()
                    if symbol in stock_list['code'].values:
                        return self._normalize_stock_basic_info(stock_list[stock_list['code'] == symbol], symbol)
                    return pd.DataFrame()
            else:
                # 获取所有A股股票列表
                stock_list = ak.stock_info_a_code_name()
                self.logger.info(f"成功获取A股股票列表，共 {len(stock_list)} 只股票")
                # 对每只股票获取详细信息
                all_stock_info = []
                batch_size = 100
                for i in range(0, len(stock_list), batch_size):
                    batch_symbols = stock_list['code'].iloc[i:i+batch_size].tolist()
                    for sym in batch_symbols:
                        try:
                            stock_info = self.get_stock_basic_info(sym)
                            if not stock_info.empty:
                                all_stock_info.append(stock_info)
                        except Exception as e:
                            self.logger.warning(f"获取股票 {sym} 基本信息失败: {e}")
                if all_stock_info:
                    return pd.concat(all_stock_info, ignore_index=True)
                return stock_list
        except ImportError:
            self.logger.error("akshare未安装，无法获取股票基础信息")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取股票基础信息失败: {e}")
            return pd.DataFrame()

    def _normalize_stock_basic_info(self, raw_info: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        标准化股票基本信息格式

        Args:
            raw_info: 原始股票信息
            symbol: 股票代码

        Returns:
            pd.DataFrame: 标准化后的股票信息
        """
        try:
            import akshare as ak
            
            # 创建标准化的DataFrame
            normalized_data = {
                'symbol': [symbol],
                'name': [''],
                'ipo_date': [None],
                'industry': [''],
                'market': [''],
                'total_share': [None],
                'float_share': [None],
                'pe': [None],
                'pb': [None],
                'roe': [None]
            }
            
            # 处理不同格式的原始数据
            if 'item' in raw_info.columns and 'value' in raw_info.columns:
                # 处理stock_individual_info_em返回的格式
                info_dict = dict(zip(raw_info['item'], raw_info['value']))
                
                # 提取股票名称
                if '股票简称' in info_dict:
                    normalized_data['name'] = [info_dict['股票简称']]
                elif '公司名称' in info_dict:
                    normalized_data['name'] = [info_dict['公司名称']]
                
                # 提取上市日期
                if '上市日期' in info_dict:
                    normalized_data['ipo_date'] = [info_dict['上市日期']]
                
                # 提取行业
                if '所属行业' in info_dict:
                    normalized_data['industry'] = [info_dict['所属行业']]
                
                # 提取市场
                if symbol.startswith('6'):
                    normalized_data['market'] = ['沪市']
                elif symbol.startswith('0') or symbol.startswith('3'):
                    normalized_data['market'] = ['深市']
                elif symbol.startswith('688'):
                    normalized_data['market'] = ['科创板']
                elif symbol.startswith('300'):
                    normalized_data['market'] = ['创业板']
                
                # 尝试获取其他财务指标
                try:
                    # 获取估值指标
                    valuation = ak.stock_a_estimates_em(symbol=symbol)
                    if not valuation.empty:
                        if '市盈率(TTM)' in valuation.columns:
                            normalized_data['pe'] = [valuation['市盈率(TTM)'].iloc[0] if not valuation['市盈率(TTM)'].isna().iloc[0] else None]
                        if '市净率' in valuation.columns:
                            normalized_data['pb'] = [valuation['市净率'].iloc[0] if not valuation['市净率'].isna().iloc[0] else None]
                except Exception as e:
                    self.logger.warning(f"获取估值指标失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"标准化股票基本信息失败: {e}")
        
        return pd.DataFrame(normalized_data)


class STARMarketAdapter(AStockAdapter):
    """
    科创板数据适配器
    
    实现科创板市场的特有功能，包括盘后固定价格交易等。
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化科创板适配器"""
        super().__init__(config)
        self.market_type = 'STAR'

    def connect(self) -> bool:
        """连接科创板数据源"""
        try:
            self.logger.info("连接科创板数据源")
            self._is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self._is_connected = False
            return False

    def get_data(self, symbol: str, **kwargs) -> Dict:
        """
        获取数据（科创板特定）

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict: 数据字典
        """
        return {
            'symbol': symbol,
            'market': '科创板',
            'market_type': self.market_type,
            'data_type': kwargs.get('data_type', 'basic')
        }

    def get_star_market_data(self, symbol: str) -> Dict:
        """
        获取科创板特有数据

        Args:
            symbol: 股票代码

        Returns:
            Dict: 科创板特有数据字典
        """
        return {
            'symbol': symbol,
            'market_type': 'STAR',
            'has_after_hours_trading': True
        }

    def get_after_hours_trading(self, symbol: str) -> pd.DataFrame:
        """
        获取盘后固定价格交易数据

        Args:
            symbol: 股票代码

        Returns:
            pd.DataFrame: 盘后交易数据DataFrame
        """
        # 实现获取盘后交易数据的逻辑
        return pd.DataFrame()

    def get_red_chip_info(self, symbol: str) -> Dict:
        """
        获取红筹企业特有信息

        Args:
            symbol: 股票代码

        Returns:
            Dict: 红筹企业信息字典
        """
        return {
            'symbol': symbol,
            'is_red_chip': True,
            'red_chip_type': 'VCDR'  # Variable Interest Entity (VIE)
        }


# 向后兼容的别名
ChinaStockAdapter = AStockAdapter


__all__ = [
    'AStockAdapter',
    'STARMarketAdapter',
    'ChinaStockAdapter',  # 向后兼容
]

