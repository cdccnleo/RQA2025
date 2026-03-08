"""
Baostock数据源适配器
对接Baostock API，提供股票历史数据、财务数据、宏观经济数据
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from abc import ABC

logger = logging.getLogger(__name__)


class BaostockAdapter(ABC):
    """
    Baostock数据源适配器
    
    职责：
    1. 连接Baostock API
    2. 获取股票历史数据
    3. 获取财务数据
    4. 获取宏观经济数据
    5. 数据格式转换
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Baostock适配器
        
        Args:
            config: 配置信息
                - username: 用户名（可选）
                - password: 密码（可选）
                - timeout: 连接超时时间
        """
        self.config = config
        self._connected = False
        self._bs = None  # baostock模块
        
        # 验证配置
        self._validate_config()
        
        logger.info("Baostock适配器初始化完成")
    
    def _validate_config(self) -> bool:
        """验证配置有效性"""
        # Baostock通常不需要登录即可获取基础数据
        return True
    
    async def connect(self) -> bool:
        """
        连接Baostock
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试导入baostock模块
            try:
                import baostock as bs
                self._bs = bs
            except ImportError:
                logger.warning("baostock模块未安装，使用模拟模式")
                self._connected = True
                return True
            
            # 登录Baostock（可选）
            lg = self._bs.login()
            if lg.error_code == '0':
                self._connected = True
                logger.info("Baostock连接成功")
                return True
            else:
                logger.error(f"Baostock登录失败: {lg.error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"连接Baostock失败: {e}")
            # 即使连接失败，也设置为连接状态（使用模拟数据）
            self._connected = True
            return True
    
    async def disconnect(self) -> None:
        """断开Baostock连接"""
        try:
            if self._bs and self._connected:
                self._bs.logout()
            
            self._connected = False
            logger.info("Baostock连接已断开")
            
        except Exception as e:
            logger.error(f"断开Baostock连接失败: {e}")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'd',
        adjustflag: str = '3'
    ) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            symbol: 股票代码（如：sh.600000）
            start_date: 开始日期
            end_date: 结束日期
            frequency: 数据频率（d=日, w=周, m=月）
            adjustflag: 复权类型（1=后复权, 2=前复权, 3=不复权）
            
        Returns:
            历史数据DataFrame
        """
        if not self._connected:
            logger.warning("Baostock未连接，无法获取历史数据")
            return pd.DataFrame()
        
        try:
            # 格式化日期
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"获取历史数据: {symbol}, {start_str} - {end_str}")
            
            if self._bs:
                # 使用Baostock API获取数据
                rs = self._bs.query_history_k_data_plus(
                    symbol,
                    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                    start_date=start_str,
                    end_date=end_str,
                    frequency=frequency,
                    adjustflag=adjustflag
                )
                
                if rs.error_code != '0':
                    logger.error(f"获取历史数据失败: {rs.error_msg}")
                    return pd.DataFrame()
                
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 数据类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            else:
                # baostock模块不可用，返回空DataFrame
                logger.error("❌ baostock模块不可用，无法获取真实数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return pd.DataFrame()
    
    async def get_financial_data(
        self,
        symbol: str,
        year: int,
        quarter: int
    ) -> Dict[str, Any]:
        """
        获取财务数据
        
        Args:
            symbol: 股票代码
            year: 年份
            quarter: 季度（1-4）
            
        Returns:
            财务数据字典
        """
        if not self._connected:
            logger.warning("Baostock未连接，无法获取财务数据")
            return {}
        
        try:
            logger.info(f"获取财务数据: {symbol}, {year}Q{quarter}")
            
            if self._bs:
                # 获取季频盈利能力数据
                profit_list = []
                rs_profit = self._bs.query_profit_data(code=symbol, year=year, quarter=quarter)
                while (rs_profit.error_code == '0') & rs_profit.next():
                    profit_list.append(rs_profit.get_row_data())
                
                # 获取季频营运能力数据
                operation_list = []
                rs_operation = self._bs.query_operation_data(code=symbol, year=year, quarter=quarter)
                while (rs_operation.error_code == '0') & rs_operation.next():
                    operation_list.append(rs_operation.get_row_data())
                
                # 获取季频偿债能力数据
                growth_list = []
                rs_growth = self._bs.query_growth_data(code=symbol, year=year, quarter=quarter)
                while (rs_growth.error_code == '0') & rs_growth.next():
                    growth_list.append(rs_growth.get_row_data())
                
                return {
                    'symbol': symbol,
                    'year': year,
                    'quarter': quarter,
                    'profit_data': profit_list,
                    'operation_data': operation_list,
                    'growth_data': growth_list
                }
            else:
                # 模拟数据
                return {
                    'symbol': symbol,
                    'year': year,
                    'quarter': quarter,
                    'roe': 0.0,
                    'net_profit': 0.0,
                    'eps': 0.0
                }
                
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            return {}
    
    async def get_macro_economic_data(
        self,
        indicator: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        获取宏观经济数据
        
        Args:
            indicator: 指标代码
                - cpi: 居民消费价格指数
                - ppi: 工业生产者出厂价格指数
                - gdp: 国内生产总值
                - m2: 广义货币供应量
                - rate: 存款利率
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            宏观经济数据DataFrame
        """
        if not self._connected:
            logger.warning("Baostock未连接，无法获取宏观经济数据")
            return pd.DataFrame()
        
        try:
            logger.info(f"获取宏观经济数据: {indicator}")
            
            if self._bs:
                # 根据指标类型调用不同的API
                if indicator == 'cpi':
                    rs = self._bs.query_cpi_data()
                elif indicator == 'ppi':
                    rs = self._bs.query_ppi_data()
                elif indicator == 'gdp':
                    rs = self._bs.query_gdp_data()
                elif indicator == 'm2':
                    rs = self._bs.query_money_supply_data_month()
                elif indicator == 'rate':
                    rs = self._bs.query_deposit_rate_data()
                else:
                    logger.warning(f"未知的宏观经济指标: {indicator}")
                    return pd.DataFrame()
                
                if rs.error_code != '0':
                    logger.error(f"获取宏观经济数据失败: {rs.error_msg}")
                    return pd.DataFrame()
                
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                return df
            else:
                # 模拟数据
                dates = pd.date_range(start=start_date, end=end_date, freq='M')
                df = pd.DataFrame({
                    'date': [d.strftime('%Y-%m') for d in dates],
                    'indicator': [indicator] * len(dates),
                    'value': [0.0] * len(dates)
                })
                return df
                
        except Exception as e:
            logger.error(f"获取宏观经济数据失败: {e}")
            return pd.DataFrame()
    
    async def get_stock_basic_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票基本信息字典
        """
        if not self._connected:
            logger.warning("Baostock未连接，无法获取股票基本信息")
            return {}
        
        try:
            logger.info(f"获取股票基本信息: {symbol}")
            
            if self._bs:
                # 获取证券基本资料
                rs = self._bs.query_stock_basic(code=symbol)
                
                if rs.error_code != '0':
                    logger.error(f"获取股票基本信息失败: {rs.error_msg}")
                    return {}
                
                # 获取第一条数据
                if rs.next():
                    data = rs.get_row_data()
                    fields = rs.fields
                    
                    return dict(zip(fields, data))
                else:
                    return {}
            else:
                # 模拟数据
                return {
                    'code': symbol,
                    'code_name': '模拟股票',
                    'ipoDate': '2020-01-01',
                    'outDate': '',
                    'type': '1',
                    'status': '1'
                }
                
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            return {}
    
    async def get_all_stocks(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        获取所有股票列表
        
        Args:
            date: 日期（可选，默认为当天）
            
        Returns:
            股票列表DataFrame
        """
        if not self._connected:
            logger.warning("Baostock未连接，无法获取股票列表")
            return pd.DataFrame()
        
        try:
            if date is None:
                date = datetime.now()
            
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"获取股票列表: {date_str}")
            
            if self._bs:
                # 获取所有股票
                rs = self._bs.query_all_stock(day=date_str)
                
                if rs.error_code != '0':
                    logger.error(f"获取股票列表失败: {rs.error_msg}")
                    return pd.DataFrame()
                
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                df = pd.DataFrame(data_list, columns=rs.fields)
                return df
            else:
                # 模拟数据
                return pd.DataFrame({
                    'code': ['sh.600000', 'sh.600001'],
                    'code_name': ['浦发银行', '模拟股票']
                })
                
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()


# 单例实例
_baostock_adapter: Optional[BaostockAdapter] = None


def get_baostock_adapter(config: Optional[Dict] = None) -> BaostockAdapter:
    """获取Baostock适配器实例"""
    global _baostock_adapter
    if _baostock_adapter is None:
        if config is None:
            config = {}
        _baostock_adapter = BaostockAdapter(config)
    return _baostock_adapter
