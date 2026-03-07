"""
中国A股数据适配器实现

职责定位：
1. 实现中国市场数据适配器的具体功能
2. 包含Redis缓存、T+1验证等中国市场特定功能
3. 继承自 adapters/china/BaseChinaAdapter 或使用组合模式
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from .adapters import BaseChinaAdapter

# 使用基础设施层日志
logger = logging.getLogger(__name__)


class ChinaDataAdapter(BaseChinaAdapter):
    """
    中国A股数据适配器
    
    实现包含Redis缓存、T+1验证等中国市场特定功能的完整适配器。
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化适配器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.redis_client = None
        self._is_connected = False
        self._init_redis()

    def _init_redis(self):
        """初始化Redis连接"""
        try:
            import redis
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}")
            self.redis_client = None

    def load_margin_data(self) -> pd.DataFrame:
        """加载融资融券数据

        Returns:
            融资融券数据DataFrame
        """
        try:
            # 模拟从数据源加载融资融券数据
            data = pd.DataFrame({
                'date': pd.date_range('2023 - 01 - 01', periods=10),
                'symbol': ['600519'] * 10,
                'margin_balance': np.random.uniform(1e8, 2e8, 10),
                'short_balance': np.random.uniform(1e7, 5e7, 10)
            })

            if data.empty:
                raise ValueError("融资融券数据为空")

            return data
        except Exception as e:
            logger.error(f"加载融资融券数据失败: {e}")
            raise ValueError(f"加载融资融券数据失败: {e}")

    def validate_t1_settlement(self, trades: pd.DataFrame) -> bool:
        """验证T + 结算规则

        Args:
            trades: 交易记录DataFrame

        Returns:
            是否符合T + 结算规则
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
        # 处理None或空值
        if not symbol:
            return {
                'upper_limit': 0.1,
                'lower_limit': -0.1,
                'market': 'unknown'
            }

        # 创业板股票 (需要先判断，因为300开头)
        if symbol.startswith('300'):
            return {
                'upper_limit': 0.2,  # 涨停20%
                'lower_limit': -0.2,  # 跌停20%
                'market': 'gem_board'
            }
        # 科创板股票
        elif symbol.startswith('688'):
            return {
                'upper_limit': 0.2,  # 涨停20%
                'lower_limit': -0.2,  # 跌停20%
                'market': 'star_board'
            }
        # 主板股票
        elif symbol.startswith(('60', '00')):
            return {
                'upper_limit': 0.1,  # 涨停10%
                'lower_limit': -0.1,  # 跌停10%
                'market': 'main_board'
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
            logger.error(f"缓存数据失败: {e}")
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
            logger.error(f"获取缓存数据失败: {e}")
            return None

    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存数据 (兼容性方法)

        Args:
            key: 缓存键

        Returns:
            缓存的数据字典
        """
        df = self._get_cached(key)
        if df is not None and not df.empty:
            return df.to_dict('records')[0]
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
            'market': 'SH' if symbol and symbol.startswith('6') else 'SZ',
            'industry': '金融',
            'list_date': '2020 - 01 - 01'
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

    def connect(self) -> bool:
        """
        连接数据源

        Returns:
            bool: 连接是否成功
        """
        try:
            self.logger.info("连接中国A股数据源")
            self._is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            bool: 断开是否成功
        """
        try:
            self.logger.info("断开中国A股数据源连接")
            self._is_connected = False
            if self.redis_client:
                self.redis_client.close()
            return True
        except Exception as e:
            self.logger.error(f"断开连接失败: {e}")
            return False

    def get_data(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        获取数据（实现基类抽象方法）

        Args:
            symbol: 股票代码
            **kwargs: 其他参数

        Returns:
            Dict[str, Any]: 数据字典
        """
        # 优先从缓存获取
        cache_key = f"stock:{symbol}:{kwargs.get('data_type', 'basic')}"
        cached_data = self._get_cached(cache_key)
        if cached_data is not None:
            return {'symbol': symbol, 'data': cached_data, 'from_cache': True}

        # 获取股票基本信息
        stock_info = self.get_stock_info(symbol)
        return {
            'symbol': symbol,
            **stock_info,
            'from_cache': False
        }

    def validate_t_plus_one(self, trade_data) -> Dict[str, Any]:
        """
        验证T+1交易规则

        Args:
            trade_data: 交易数据 (DataFrame或dict)

        Returns:
            验证结果字典
        """
        if not self.config.get('t_plus_one_check', True):
            return {
                'is_valid': True,
                'message': 'T+1 validation disabled'
            }

        try:
            # 处理dict输入
            if isinstance(trade_data, dict):
                trade_date = trade_data.get('trade_date')
                settlement_date = trade_data.get('settlement_date')
                if not trade_date or not settlement_date:
                    return {
                        'is_valid': False,
                        'message': 'Missing trade_date or settlement_date'
                    }

                # 简单的日期比较
                trade_dt = pd.to_datetime(trade_date)
                settlement_dt = pd.to_datetime(settlement_date)
                is_valid = (settlement_dt - trade_dt).days == 1

                return {
                    'is_valid': is_valid,
                    'message': 'T+1 rule validated' if is_valid else 'T+1 rule violation'
                }

            # 处理DataFrame输入
            elif hasattr(trade_data, 'columns'):
                if 'trade_date' not in trade_data.columns or 'settlement_date' not in trade_data.columns:
                    return {
                        'is_valid': False,
                        'message': 'Missing trade_date or settlement_date columns'
                    }

                # 检查T+1规则：交易日后一天为结算日
                trade_dates = pd.to_datetime(trade_data['trade_date'])
                settlement_dates = pd.to_datetime(trade_data['settlement_date'])

                # 计算日期差异
                date_diff = (settlement_dates - trade_dates).dt.days

                # T+1规则：结算日应该是交易日后1个交易日
                is_valid = (date_diff == 1).all()

                return {
                    'is_valid': is_valid,
                    'message': 'T+1 rule validated' if is_valid else 'T+1 rule violation'
                }

            return {
                'is_valid': False,
                'message': 'Invalid data format'
            }

        except Exception as e:
            return {
                'is_valid': False,
                'message': f'Validation error: {str(e)}'
            }

    def get_market_data(self, symbol, **kwargs) -> Dict[str, Any]:
        """
        获取市场数据

        Args:
            symbol: 股票代码或股票代码列表
            **kwargs: 其他参数

        Returns:
            市场数据字典
        """
        if isinstance(symbol, list):
            # 如果是列表，批量获取
            return self.fetch_market_data(symbol)
        else:
            # 单个股票代码
            try:
                # 先尝试获取原始数据（用于测试错误处理）
                raw_data = self._fetch_raw_data(symbol)
                data = self.get_data(symbol, **kwargs)
                return {
                    'data': data,
                    'status': 'success',
                    'symbol': symbol
                }
            except Exception as e:
                return {
                    'data': None,
                    'status': 'error',
                    'error': str(e),
                    'symbol': symbol
                }

    def fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        批量获取市场数据

        Args:
            symbols: 股票代码列表

        Returns:
            市场数据字典
        """
        result = {}
        for symbol in symbols:
            try:
                # 确保symbol是字符串
                if isinstance(symbol, str):
                    result[symbol] = self.get_market_data(symbol)
                else:
                    result[str(symbol)] = None
            except Exception as e:
                self.logger.error(f"获取{symbol}市场数据失败: {e}")
                result[str(symbol)] = None
        return result

    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换数据格式

        Args:
            raw_data: 原始数据

        Returns:
            转换后的数据
        """
        # 简单的数据转换逻辑
        if 'price' in raw_data:
            # 确保价格为浮点数
            raw_data['price'] = float(raw_data['price'])
        if 'volume' in raw_data:
            # 确保成交量为整数
            raw_data['volume'] = int(raw_data['volume'])

        # 添加转换标记
        raw_data['transformed'] = True
        raw_data['transformed_at'] = datetime.now().isoformat()

        return raw_data

    def check_health(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            健康检查结果
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }

        # 检查Redis连接
        try:
            if self.redis_client:
                self.redis_client.ping()
                health_status['components']['redis'] = 'connected'
            else:
                health_status['components']['redis'] = 'not_configured'
        except Exception as e:
            health_status['components']['redis'] = f'error: {str(e)}'
            health_status['status'] = 'degraded'

        # 检查配置
        health_status['components']['config'] = 'valid' if self.config else 'missing'

        return health_status

    def _fetch_raw_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取原始数据 (内部方法，用于测试)

        Args:
            symbol: 股票代码

        Returns:
            原始数据字典
        """
        # 这个方法主要用于测试中的错误模拟
        return {
            'symbol': symbol,
            'price': 100.0,
            'volume': 10000
        }

    def adapt_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        适配市场数据格式

        Args:
            data: 原始市场数据DataFrame

        Returns:
            适配后的数据DataFrame
        """
        if data is None or data.empty:
            return pd.DataFrame()

        # 检查缓存
        cache_key = f"adapted_market_data_{hash(str(data.values.tobytes()) if hasattr(data, 'values') else str(data))}"

        # 尝试从缓存获取
        cached_result = self._get_cached(cache_key)
        if cached_result is not None:
            self.logger.info("从缓存获取适配数据")
            return cached_result

        # 复制数据避免修改原始数据
        adapted_data = data.copy()

        # 添加中国市场特定的字段
        adapted_data['market'] = 'china'
        adapted_data['data_source'] = 'ChinaDataAdapter'
        adapted_data['adapted_at'] = datetime.now()

        # 确保价格字段为float类型
        price_columns = ['price', 'open', 'high', 'low', 'close']
        for col in price_columns:
            if col in adapted_data.columns:
                adapted_data[col] = pd.to_numeric(adapted_data[col], errors='coerce')

        # 确保成交量为整数类型
        if 'volume' in adapted_data.columns:
            adapted_data['volume'] = pd.to_numeric(adapted_data['volume'], errors='coerce').astype('Int64')

        # 缓存结果
        self.cache_data(cache_key, adapted_data)

        return adapted_data