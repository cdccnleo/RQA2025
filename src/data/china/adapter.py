"""中国A股数据适配器

提供A股市场数据适配功能，包括融资融券、T+1结算验证、涨跌停价格计算等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import redis
from abc import ABC, abstractmethod


class ChinaDataAdapter:
    """中国A股数据适配器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化适配器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """初始化Redis连接"""
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
        except Exception as e:
            self.logger.warning(f"Redis连接失败: {e}")
            self.redis_client = None
    
    def load_margin_data(self) -> pd.DataFrame:
        """加载融资融券数据
        
        Returns:
            融资融券数据DataFrame
        """
        try:
            # 模拟从数据源加载融资融券数据
            data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'symbol': ['600519'] * 10,
                'margin_balance': np.random.uniform(1e8, 2e8, 10),
                'short_balance': np.random.uniform(1e7, 5e7, 10)
            })
            
            if data.empty:
                raise ValueError("融资融券数据为空")
            
            return data
        except Exception as e:
            self.logger.error(f"加载融资融券数据失败: {e}")
            raise ValueError(f"加载融资融券数据失败: {e}")
    
    def validate_t1_settlement(self, trades: pd.DataFrame) -> bool:
        """验证T+1结算规则
        
        Args:
            trades: 交易记录DataFrame
            
        Returns:
            是否符合T+1结算规则
        """
        if trades.empty:
            return True
        
        # 按日期分组检查
        for date, group in trades.groupby('date'):
            # 检查同一天是否有买入后卖出的情况
            buy_trades = group[group['action'] == 'BUY']
            sell_trades = group[group['action'] == 'SELL']
            
            if not buy_trades.empty and not sell_trades.empty:
                return False
        
        return True
    
    def get_price_limits(self, symbol: str) -> Dict[str, float]:
        """获取涨跌停价格限制
        
        Args:
            symbol: 股票代码
            
        Returns:
            涨跌停限制字典
        """
        # 主板股票
        if symbol.startswith(('60', '00', '300')):
            return {
                'upper_limit': 0.1,  # 涨停10%
                'lower_limit': -0.1,  # 跌停10%
                'market': 'main_board'
            }
        # 科创板股票
        elif symbol.startswith('688'):
            return {
                'upper_limit': 0.2,  # 涨停20%
                'lower_limit': -0.2,  # 跌停20%
                'market': 'star_board'
            }
        # 创业板股票
        elif symbol.startswith('300'):
            return {
                'upper_limit': 0.2,  # 涨停20%
                'lower_limit': -0.2,  # 跌停20%
                'market': 'gem_board'
            }
        else:
            return {
                'upper_limit': 0.1,
                'lower_limit': -0.1,
                'market': 'unknown'
            }
    
    def cache_data(self, key: str, data: pd.DataFrame) -> bool:
        """缓存数据到Redis
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            
        Returns:
            是否缓存成功
        """
        if self.redis_client is None:
            return False
        
        try:
            # 将DataFrame序列化为JSON
            data_json = data.to_json(orient='records')
            self.redis_client.set(key, data_json, ex=3600)  # 1小时过期
            return True
        except Exception as e:
            self.logger.error(f"缓存数据失败: {e}")
            return False
    
    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """从Redis获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据DataFrame，如果不存在返回None
        """
        if self.redis_client is None:
            return None
        
        try:
            data_json = self.redis_client.get(key)
            if data_json:
                return pd.read_json(data_json, orient='records')
            return None
        except Exception as e:
            self.logger.error(f"获取缓存数据失败: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票信息字典
        """
        # 模拟股票信息
        return {
            'symbol': symbol,
            'name': f'股票{symbol}',
            'market': 'SH' if symbol.startswith('6') else 'SZ',
            'industry': '金融',
            'list_date': '2020-01-01'
        }
    
    def get_market_status(self) -> Dict[str, Any]:
        """获取市场状态
        
        Returns:
            市场状态字典
        """
        now = datetime.now()
        return {
            'market_open': True,
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'trading_day': now.weekday() < 5,  # 周一到周五
            'session': 'morning' if 9 <= now.hour < 11 else 'afternoon' if 13 <= now.hour < 15 else 'closed'
        } 