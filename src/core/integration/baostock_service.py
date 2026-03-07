#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaoStock服务模块

负责管理BaoStock库调用，提供统一的接口、错误处理和重试机制，
作为AKShare的备用数据源。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd

# 导入配置管理
from .config.baostock_service_config import get_baostock_config, BaoStockServiceConfig

# 延迟导入BaoStock，避免启动时依赖
bs = None
baostock_available = False
try:
    import baostock as bs
    baostock_available = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BaoStockService:
    """BaoStock服务"""

    def __init__(self, config: Optional[Union[Dict[str, Any], BaoStockServiceConfig]] = None, env: str = "default"):
        """
        初始化BaoStock服务
        
        Args:
            config: 配置参数或配置实例
            env: 环境名称 (default, production, development, test)
        """
        # 处理配置参数
        if isinstance(config, BaoStockServiceConfig):
            self.config_instance = config
        elif isinstance(config, dict):
            self.config_instance = BaoStockServiceConfig(config)
        else:
            self.config_instance = get_baostock_config(env)
        
        # 获取配置字典
        self.config = self.config_instance.config
        
        self._initialize_config()
        self._validate_baostock_availability()
        
        # 连接状态
        self.connected = False
        self.login_attempts = 0
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """递归合并配置"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _initialize_config(self):
        """初始化配置"""
        self.max_retries = self.config["retry_policy"]["max_retries"]
        self.initial_delay = self.config["retry_policy"]["initial_delay"]
        self.backoff_factor = self.config["retry_policy"]["backoff_factor"]
        self.timeout_config = self.config["timeout"]
        self.field_mapping = self.config["field_mapping"]
        self.performance_config = self.config.get("performance", {})
    
    def _validate_baostock_availability(self):
        """验证BaoStock可用性"""
        global bs, baostock_available
        if not baostock_available:
            try:
                import baostock as bs
                baostock_available = True
                logger.info("✅ BaoStock库加载成功")
            except ImportError:
                logger.warning("⚠️ BaoStock库不可用，部分功能将受限")
    
    @property
    def is_available(self) -> bool:
        """检查BaoStock是否可用"""
        return baostock_available
    
    async def _ensure_connected(self) -> bool:
        """确保BaoStock已连接"""
        if not self.is_available:
            logger.error("❌ BaoStock库不可用")
            return False
        
        # 即使 self.connected 为 True，也尝试重新连接以确保连接有效
        # BaoStock 的 login() 是幂等的，重复调用不会出错
        try:
            logger.info("🔄 正在连接BaoStock...")
            lg = bs.login()
            
            if lg.error_code == '0':
                self.connected = True
                logger.info("✅ BaoStock连接成功")
                return True
            else:
                logger.warning(f"⚠️ BaoStock连接失败: {lg.error_msg}")
                self.connected = False
                return False
        except Exception as e:
            logger.error(f"❌ BaoStock连接异常: {e}")
            self.connected = False
            return False
    
    async def _disconnect(self):
        """断开BaoStock连接"""
        if self.connected:
            try:
                bs.logout()
                self.connected = False
                logger.info("🔌 BaoStock连接已断开")
            except Exception as e:
                logger.warning(f"⚠️ 断开BaoStock连接失败: {e}")
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3"
    ) -> Optional[pd.DataFrame]:
        """
        获取股票历史K线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            frequency: 频率 (d=日线, w=周线, m=月线)
            adjustflag: 复权方式 (3=后复权, 2=前复权, 1=不复权)
            
        Returns:
            股票数据DataFrame
        """
        if not await self._ensure_connected():
            logger.error("❌ BaoStock未连接，无法获取数据")
            return None
        
        logger.info(f"📊 获取股票数据: {symbol}, 日期: {start_date}~{end_date}")
        
        # 格式化股票代码
        formatted_symbol = self._format_symbol(symbol)
        
        # K线数据字段
        fields = "date,code,open,high,low,close,volume,amount,turn,tradestatus,pctChg,isST"
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - 获取K线数据")
                
                # 调用BaoStock API
                rs = bs.query_history_k_data_plus(
                    code=formatted_symbol,
                    fields=fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjustflag
                )
                
                if rs.error_code != '0':
                    logger.warning(f"⚠️ BaoStock API错误: {rs.error_msg}")
                    
                    # 检查是否是网络错误，如果是则尝试重新连接
                    if "网络" in rs.error_msg or "连接" in rs.error_msg or "socket" in rs.error_msg.lower():
                        logger.info("🔄 检测到网络错误，尝试重新连接...")
                        self.connected = False
                        await self._ensure_connected()
                    
                    if attempt < self.max_retries - 1:
                        delay = self.initial_delay * (self.backoff_factor ** attempt)
                        logger.info(f"⏳ {delay}秒后重试...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return None
                
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    logger.warning(f"⚠️ 未获取到数据: {formatted_symbol}")
                    return None
                
                # 创建DataFrame
                df = pd.DataFrame(data_list, columns=fields.split(','))
                
                # 转换数据类型
                numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                for field in numeric_fields:
                    if field in df.columns:
                        df[field] = pd.to_numeric(df[field], errors='coerce')
                
                logger.info(f"✅ 成功获取数据: {formatted_symbol}, {len(df)} 条记录")
                
                # 应用字段映射
                df_mapped = self._map_fields(df, "kline")
                return df_mapped
                
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                # 检查是否是网络/连接相关异常
                error_str = str(e).lower()
                if "socket" in error_str or "network" in error_str or "connection" in error_str or "10038" in error_str:
                    logger.info("🔄 检测到连接异常，尝试重新连接...")
                    self.connected = False
                    await self._ensure_connected()
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ 所有尝试都失败: {e}")
                    return None
    
    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票基本信息字典
        """
        if not await self._ensure_connected():
            logger.error("❌ BaoStock未连接，无法获取信息")
            return None
        
        formatted_symbol = self._format_symbol(symbol)
        logger.info(f"📋 获取股票基本信息: {formatted_symbol}")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - 获取基本信息")
                
                # 调用BaoStock API
                rs = bs.query_stock_basic(code=formatted_symbol)
                
                if rs.error_code != '0':
                    logger.warning(f"⚠️ BaoStock API错误: {rs.error_msg}")
                    if attempt < self.max_retries - 1:
                        delay = self.initial_delay * (self.backoff_factor ** attempt)
                        logger.info(f"⏳ {delay}秒后重试...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return None
                
                # 转换为字典
                info_dict = {}
                while (rs.error_code == '0') & rs.next():
                    row = rs.get_row_data()
                    # 字段映射
                    info_dict['股票代码'] = row[0]
                    info_dict['股票名称'] = row[1]
                    info_dict['上市日期'] = row[2]
                    info_dict['退市日期'] = row[3]
                    info_dict['股票类型'] = row[4]
                    break
                
                if not info_dict:
                    logger.warning(f"⚠️ 未获取到基本信息: {formatted_symbol}")
                    return None
                
                logger.info(f"✅ 成功获取基本信息: {formatted_symbol}")
                return info_dict
                
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ 所有尝试都失败: {e}")
                    return None
    
    async def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        获取市场数据
        
        Returns:
            市场数据DataFrame
        """
        if not await self._ensure_connected():
            logger.error("❌ BaoStock未连接，无法获取市场数据")
            return None
        
        logger.info("📊 获取市场数据")
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - 获取市场数据")
                
                # 获取所有股票列表
                rs = bs.query_all_stock(datetime.now().strftime('%Y-%m-%d'))
                
                if rs.error_code != '0':
                    logger.warning(f"⚠️ BaoStock API错误: {rs.error_msg}")
                    if attempt < self.max_retries - 1:
                        delay = self.initial_delay * (self.backoff_factor ** attempt)
                        logger.info(f"⏳ {delay}秒后重试...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        return None
                
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if not data_list:
                    logger.warning("⚠️ 未获取到市场数据")
                    return None
                
                # 创建DataFrame
                df = pd.DataFrame(data_list, columns=['code', 'tradeStatus', 'code_name'])
                
                logger.info(f"✅ 成功获取市场数据: {len(df)} 条记录")
                return df
                
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ 所有尝试都失败: {e}")
                    return None
    
    def _format_symbol(self, symbol: str) -> str:
        """
        格式化股票代码
        
        Args:
            symbol: 股票代码
            
        Returns:
            格式化后的股票代码
        """
        # 移除可能的后缀
        symbol = symbol.replace('.SH', '').replace('.SZ', '')
        
        # 添加市场前缀
        if symbol.startswith('6'):
            return f'sh.{symbol}'
        else:
            return f'sz.{symbol}'
    
    def _map_fields(self, df: pd.DataFrame, field_type: str) -> pd.DataFrame:
        """
        映射字段名
        
        Args:
            df: 原始数据DataFrame
            field_type: 字段类型
            
        Returns:
            映射后的DataFrame
        """
        if field_type in self.field_mapping:
            mapping = self.field_mapping[field_type]
            # 只映射存在的字段
            mapping = {k: v for k, v in mapping.items() if k in df.columns}
            if mapping:
                df_mapped = df.rename(columns=mapping)
                logger.info(f"🔄 字段映射完成: {len(mapping)} 个字段")
                return df_mapped
        return df
    
    def convert_to_standard_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        转换为标准格式
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            标准格式的字典列表
        """
        if df is None or df.empty:
            return []
        
        records = []
        for _, row in df.iterrows():
            record = {
                "symbol": "",
                "date": "",
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
                "amount": None,
                "data_source": "baostock",
                "timestamp": datetime.now().timestamp()
            }
            
            # 填充字段
            if "日期" in row:
                record["date"] = row["日期"]
            elif "date" in row:
                record["date"] = row["date"]
            
            if "股票代码" in row:
                record["symbol"] = row["股票代码"].replace('sh.', '').replace('sz.', '')
            elif "code" in row:
                record["symbol"] = row["code"].replace('sh.', '').replace('sz.', '')
            
            if "开盘" in row:
                record["open"] = row["开盘"]
            elif "open" in row:
                record["open"] = row["open"]
                
            if "最高" in row:
                record["high"] = row["最高"]
            elif "high" in row:
                record["high"] = row["high"]
                
            if "最低" in row:
                record["low"] = row["最低"]
            elif "low" in row:
                record["low"] = row["low"]
                
            if "收盘" in row:
                record["close"] = row["收盘"]
            elif "close" in row:
                record["close"] = row["close"]
            
            if "成交量" in row:
                record["volume"] = row["成交量"]
            elif "volume" in row:
                record["volume"] = row["volume"]
            
            if "成交额" in row:
                record["amount"] = row["成交额"]
            elif "amount" in row:
                record["amount"] = row["amount"]
            
            records.append(record)
        
        logger.info(f"🔄 数据转换完成: {len(records)} 条记录")
        return records
    
    def __del__(self):
        """析构函数，断开连接"""
        # 注意：_disconnect 是异步方法，在析构函数中需要同步关闭连接
        if self.connected and bs is not None:
            try:
                bs.logout()
                self.connected = False
            except:
                pass


# 全局BaoStock服务实例
_baostock_service_instance = None


def get_baostock_service(config: Optional[Dict[str, Any]] = None) -> BaoStockService:
    """
    获取全局BaoStock服务实例
    
    Args:
        config: 配置参数
        
    Returns:
        BaoStockService实例
    """
    global _baostock_service_instance
    if _baostock_service_instance is None:
        _baostock_service_instance = BaoStockService(config)
    elif config:
        # 更新配置
        _merge_config(_baostock_service_instance.config, config)
        _baostock_service_instance._initialize_config()
    return _baostock_service_instance


def _merge_config(target: Dict[str, Any], source: Dict[str, Any]):
    """递归合并配置"""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_config(target[key], value)
        else:
            target[key] = value


def reset_baostock_service():
    """
    重置BaoStock服务实例
    """
    global _baostock_service_instance
    if _baostock_service_instance:
        try:
            # 同步断开连接
            if _baostock_service_instance.connected and bs is not None:
                bs.logout()
                _baostock_service_instance.connected = False
        except:
            pass
    _baostock_service_instance = None
    logger.info("🔄 BaoStock服务实例已重置")


def get_baostock_service_with_env(env: str) -> BaoStockService:
    """
    获取指定环境的BaoStock服务实例
    
    Args:
        env: 环境名称
        
    Returns:
        BaoStockService实例
    """
    # 可以根据环境返回不同配置的实例
    return get_baostock_service()