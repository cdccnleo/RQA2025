#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的AKShare服务

负责集中管理所有AKShare库调用，提供统一的接口、智能无缝切换、
字段映射和错误处理机制，消除代码重复，提高可维护性。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd

# 导入配置管理
from .config.akshare_service_config import get_akshare_config, AKShareServiceConfig

# 延迟导入AKShare，避免启动时依赖
ak = None
akshare_available = False
try:
    import akshare as ak
    akshare_available = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class AKShareService:
    """统一的AKShare服务"""

    def __init__(self, config: Optional[Union[Dict[str, Any], AKShareServiceConfig]] = None, env: str = "default"):
        """初始化AKShare服务"""
        # 处理配置参数
        if isinstance(config, AKShareServiceConfig):
            self.config_instance = config
        elif isinstance(config, dict):
            self.config_instance = AKShareServiceConfig(config)
        else:
            self.config_instance = get_akshare_config(env)
        
        # 获取配置字典
        self.config = self.config_instance.config
        
        self._initialize_config()
        self._validate_akshare_availability()
    
    def _initialize_config(self):
        """初始化配置"""
        self.max_retries = self.config["retry_policy"]["max_retries"]
        self.initial_delay = self.config["retry_policy"]["initial_delay"]
        self.backoff_factor = self.config["retry_policy"]["backoff_factor"]
        self.timeout_config = self.config["timeout"]
        self.field_mapping = self.config["field_mapping"]
        self.api_preference = self.config["api_preference"]
        self.cache_config = self.config.get("cache", {})
        self.performance_config = self.config.get("performance", {})
    
    def _validate_akshare_availability(self):
        """验证AKShare可用性"""
        global ak, akshare_available
        if not akshare_available:
            try:
                import akshare as ak
                akshare_available = True
                logger.info("✅ AKShare库加载成功")
            except ImportError:
                logger.warning("⚠️ AKShare库不可用，部分功能将受限")
    
    @property
    def is_available(self) -> bool:
        """检查AKShare是否可用"""
        return akshare_available
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        data_type: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """
        获取股票数据，支持智能无缝切换
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权方式 (qfq/hfq)
            data_type: 数据类型 (daily/minute)
            
        Returns:
            股票数据DataFrame
        """
        if not self.is_available:
            logger.error("❌ AKShare库不可用")
            return None
        
        logger.info(f"📊 获取股票数据: {symbol}, 日期: {start_date}~{end_date}")
        
        # 智能无缝切换机制
        apis_to_try = self.api_preference.get("stock_daily", ["stock_zh_a_hist"])
        
        for api_name in apis_to_try:
            try:
                logger.info(f"🔀 尝试接口: {api_name}")
                
                if api_name == "stock_zh_a_hist":
                    # 使用原始接口
                    df = await self._call_stock_zh_a_hist(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust
                    )
                else:
                    logger.warning(f"⚠️ 未知接口: {api_name}")
                    continue
                
                if df is not None and not df.empty:
                    logger.info(f"✅ 接口 {api_name} 调用成功: {len(df)} 条记录")
                    # 应用字段映射
                    df_mapped = self._map_fields(df, api_name)
                    return df_mapped
                    
            except Exception as e:
                logger.warning(f"⚠️ 接口 {api_name} 调用失败: {e}")
                continue
        
        logger.error(f"❌ 所有接口调用失败: {symbol}")
        return None
    
    async def _call_stock_zh_a_hist(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """调用stock_zh_a_hist接口"""
        timeout = self.timeout_config.get("stock_data", 30)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - stock_zh_a_hist")
                
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    timeout=timeout
                )
                
                return df
                
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    raise
    

    
    async def get_market_data(self) -> Optional[pd.DataFrame]:
        """获取市场数据"""
        if not self.is_available:
            logger.error("❌ AKShare库不可用")
            return None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - 获取市场数据")
                
                # 使用更可靠的 stock_zh_a_spot 函数
                df = ak.stock_zh_a_spot()
                logger.info(f"✅ 市场数据获取成功: {len(df)} 条记录")
                return df
                
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("❌ 市场数据获取失败")
                    return None
    
    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取股票基础信息"""
        if not self.is_available:
            logger.error("❌ AKShare库不可用")
            return None
        
        timeout = self.timeout_config.get("basic_info", 20)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries} - 获取股票信息: {symbol}")
                
                df = ak.stock_individual_info_em(symbol=symbol)
                if df is not None and not df.empty:
                    # 转换为字典
                    info_dict = {}
                    for _, row in df.iterrows():
                        if len(row) >= 2:
                            # 使用 iloc 避免 FutureWarning
                            key = row.iloc[0]
                            value = row.iloc[1]
                            info_dict[key] = value
                    logger.info(f"✅ 股票信息获取成功: {symbol}")
                    return info_dict
                else:
                    logger.warning(f"⚠️ 股票信息为空: {symbol}")
                    return None
                    
            except Exception as e:
                logger.warning(f"⚠️ 尝试 {attempt + 1} 失败: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (self.backoff_factor ** attempt)
                    logger.info(f"⏳ {delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ 股票信息获取失败: {symbol}")
                    return None
    
    def _map_fields(self, df: pd.DataFrame, api_name: str) -> pd.DataFrame:
        """映射字段名"""
        if api_name in self.field_mapping:
            mapping = self.field_mapping[api_name]
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
                "data_source": "akshare",
                "timestamp": datetime.now().timestamp()
            }
            
            # 填充字段
            if "日期" in row:
                record["date"] = row["日期"]
            elif "date" in row:
                record["date"] = row["date"]
            
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


# 全局AKShare服务实例
_akshare_service_instance = None


def get_akshare_service(config: Optional[Union[Dict[str, Any], AKShareServiceConfig]] = None, env: str = "default") -> AKShareService:
    """
    获取全局AKShare服务实例
    
    Args:
        config: 配置参数或配置实例
        env: 环境名称 (default, production, development, test)
        
    Returns:
        AKShareService实例
    """
    global _akshare_service_instance
    if _akshare_service_instance is None:
        _akshare_service_instance = AKShareService(config, env)
    elif config:
        # 更新配置
        if isinstance(config, AKShareServiceConfig):
            _akshare_service_instance.config_instance = config
            _akshare_service_instance.config = config.config
        elif isinstance(config, dict):
            _akshare_service_instance.config.update(config)
        _akshare_service_instance._initialize_config()
    return _akshare_service_instance


def reset_akshare_service():
    """重置AKShare服务实例"""
    global _akshare_service_instance
    _akshare_service_instance = None
    logger.info("🔄 AKShare服务实例已重置")


def get_akshare_service_with_env(env: str) -> AKShareService:
    """
    获取指定环境的AKShare服务实例
    
    Args:
        env: 环境名称
        
    Returns:
        AKShareService实例
    """
    return get_akshare_service(env=env)
