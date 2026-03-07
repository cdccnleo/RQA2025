#!/usr/bin/env python3
"""
RQA2025 宏观经济数据加载器

支持从多个宏观经济数据源获取数据：
- FRED API: 美国联邦储备银行经济数据
- World Bank API: 世界银行发展指标
- 本地缓存: 减少API调用频率
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
from dataclasses import dataclass

from ..core.base_loader import BaseDataLoader
from ..cache.cache_manager import CacheManager, CacheConfig
from src.infrastructure.logging import get_infrastructure_logger

# 配置日志
logging.basicConfig(level=logging.INFO)

logger = get_infrastructure_logger('__name__')


class MacroIndicator:

    """宏观经济指标数据结构"""
    indicator_id: str
    name: str
    value: float
    unit: str
    frequency: str
    date: datetime
    country: str
    source: str


@dataclass
class MacroSeries:

    """宏观经济时间序列数据结构"""
    series_id: str
    name: str
    data: List[Dict[str, Any]]
    frequency: str
    units: str
    country: str
    source: str


class FREDLoader(BaseDataLoader):

    """FRED API数据加载器"""

    def __init__(self, api_key: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key
        }
        super().__init__(config)
        self.base_url = "https://api.stlouisfed.org / fred"
        self.api_key = api_key
        self.session = None
        # 为CacheManager提供配置
        cache_config = CacheConfig(
            max_size=1000,
            ttl=3600,
            enable_disk_cache=True,
            disk_cache_dir="cache",
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        return self._validate_config()

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "fred",
            "version": "1.0.0",
            "description": "FRED API数据加载器",
            "supported_sources": ["fred"],
            "supported_frequencies": ["1d", "1m", "1q", "1y"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口

        Args:
            start_date: 开始日期，格式：YYYY - MM - DD
            end_date: 结束日期，格式：YYYY - MM - DD
            frequency: 数据频率，如 "1d", "1h", "5min"

        Returns:
            Any: 加载的数据模型对象
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use async methods for FRED data loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - FREDLoader / 1.0',
                'Accept': 'application / json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_series(self, series_id: str, start_date: str = None, end_date: str = None) -> Optional[Dict[str, Any]]:
        """获取FRED时间序列数据"""
        cache_key = f"fred_series_{series_id}_{start_date}_{end_date}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/series / observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }

            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（1小时）
                    await self.cache_manager.set(cache_key, data, ttl=3600)

                    logger.info(f"成功获取FRED数据: {series_id}")
                    return data
                else:
                    logger.error(f"FRED API请求失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取FRED数据时发生错误: {e}")
            return None

    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """获取系列信息"""
        cache_key = f"fred_series_info_{series_id}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（24小时）
                    await self.cache_manager.set(cache_key, data, ttl=86400)

                    return data
                else:
                    logger.error(f"获取系列信息失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取系列信息时发生错误: {e}")
            return None

    async def search_series(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """搜索数据系列"""
        cache_key = f"fred_search_{search_text}_{limit}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/series / search"
            params = {
                'search_text': search_text,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（1小时）
                    await self.cache_manager.set(cache_key, data, ttl=3600)

                    return data.get('seriess', [])
                else:
                    logger.error(f"搜索系列失败: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"搜索系列时发生错误: {e}")
            return []


class WorldBankLoader(BaseDataLoader):

    """World Bank API数据加载器"""

    def __init__(self, api_key: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key
        }
        super().__init__(config)
        self.base_url = "https://api.worldbank.org / v2"
        self.api_key = api_key
        self.session = None
        # 为CacheManager提供配置
        cache_config = CacheConfig(
            max_size=1000,
            ttl=3600,
            enable_disk_cache=True,
            disk_cache_dir="cache",
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        return self._validate_config()

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "worldbank",
            "version": "1.0.0",
            "description": "World Bank API数据加载器",
            "supported_sources": ["worldbank"],
            "supported_frequencies": ["1d", "1m", "1q", "1y"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口

        Args:
            start_date: 开始日期，格式：YYYY - MM - DD
            end_date: 结束日期，格式：YYYY - MM - DD
            frequency: 数据频率，如 "1d", "1h", "5min"

        Returns:
            Any: 加载的数据模型对象
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use async methods for World Bank data loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - WorldBankLoader / 1.0',
                'Accept': 'application / json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_indicator(self, indicator_id: str, country: str = "US", start_year: int = None, end_year: int = None) -> Optional[Dict[str, Any]]:
        """获取World Bank指标数据"""
        cache_key = f"worldbank_indicator_{indicator_id}_{country}_{start_year}_{end_year}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取World Bank数据: {indicator_id}")
            return cached_data

        try:
            url = f"{self.base_url}/country/{country}/indicator/{indicator_id}"
            params = {
                'format': 'json',
                'per_page': 1000
            }

            if start_year:
                params['date'] = f"{start_year}:{end_year or datetime.now().year}"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（1小时）
                    await self.cache_manager.set(cache_key, data, ttl=3600)

                    logger.info(f"成功获取World Bank数据: {indicator_id}")
                    return data
                else:
                    logger.error(f"World Bank API请求失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取World Bank数据时发生错误: {e}")
            return None

    async def get_countries(self) -> List[Dict[str, Any]]:
        """获取国家列表"""
        cache_key = "worldbank_countries"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/country"
            params = {
                'format': 'json',
                'per_page': 1000
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（24小时）
                    await self.cache_manager.set(cache_key, data, ttl=86400)

                    return data[1] if len(data) > 1 else []
                else:
                    logger.error(f"获取国家列表失败: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"获取国家列表时发生错误: {e}")
            return []

    async def get_indicators(self, topic: str = None) -> List[Dict[str, Any]]:
        """获取指标列表"""
        cache_key = f"worldbank_indicators_{topic or 'all'}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/indicator"
            params = {
                'format': 'json',
                'per_page': 1000
            }

            if topic:
                params['topic'] = topic

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（24小时）
                    await self.cache_manager.set(cache_key, data, ttl=86400)

                    return data[1] if len(data) > 1 else []
                else:
                    logger.error(f"获取指标列表失败: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"获取指标列表时发生错误: {e}")
            return []


class MacroDataLoader(BaseDataLoader):

    """统一的宏观经济数据加载器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        config = config or {}
        super().__init__(config)
        self.config = config
        self.fred_loader = None
        self.worldbank_loader = None
        # 为CacheManager提供配置
        cache_config = CacheConfig(
            max_size=1000,
            ttl=3600,
            enable_disk_cache=True,
            disk_cache_dir="cache",
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        return self._validate_config()

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "macro",
            "version": "1.0.0",
            "description": "宏观经济数据加载器",
            "supported_sources": ["fred", "worldbank"],
            "supported_frequencies": ["1d", "1m", "1q", "1y"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口

        Args:
            start_date: 开始日期，格式：YYYY - MM - DD
            end_date: 结束日期，格式：YYYY - MM - DD
            frequency: 数据频率，如 "1d", "1h", "5min"

        Returns:
            Any: 加载的数据模型对象
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def initialize(self):
        """初始化数据加载器"""
        logger.info("初始化宏观经济数据加载器...")

        # 初始化FRED加载器
        fred_api_key = self.config.get('fred_api_key')
        self.fred_loader = FREDLoader(fred_api_key)

        # 初始化World Bank加载器
        worldbank_api_key = self.config.get('worldbank_api_key')
        self.worldbank_loader = WorldBankLoader(worldbank_api_key)

    async def get_gdp_data(self, country: str = "US", years: int = 10) -> List[MacroIndicator]:
        """获取GDP数据"""
        end_year = datetime.now().year
        start_year = end_year - years

        # 从FRED获取美国GDP数据
        if country == "US":
            async with self.fred_loader as loader:
                gdp_data = await loader.get_series("GDP", f"{start_year}-01 - 01", f"{end_year}-12 - 31")
                if gdp_data:
                    indicators = []
                    for observation in gdp_data.get('observations', []):
                        if observation.get('value') != '.':
                            indicators.append(MacroIndicator(
                                indicator_id="GDP",
                                name="Gross Domestic Product",
                                value=float(observation['value']),
                                unit="Billions of Dollars",
                                frequency="Quarterly",
                                date=datetime.strptime(observation['date'], '%Y-%m-%d'),
                                country="US",
                                source="FRED"
                            ))
                    return indicators

        # 从World Bank获取其他国家GDP数据
        else:
            async with self.worldbank_loader as loader:
                gdp_data = await loader.get_indicator("NY.GDP.MKTP.CD", country, start_year, end_year)
                if gdp_data and len(gdp_data) > 1:
                    indicators = []
                    for item in gdp_data[1]:
                        if item.get('value'):
                            indicators.append(MacroIndicator(
                                indicator_id="NY.GDP.MKTP.CD",
                                name="GDP (current US$)",
                                value=float(item['value']),
                                unit="US Dollars",
                                frequency="Annual",
                                date=datetime.strptime(item['date'], '%Y'),
                                country=country,
                                source="World Bank"
                            ))
                    return indicators

        return []

    async def get_inflation_data(self, country: str = "US", years: int = 10) -> List[MacroIndicator]:
        """获取通胀率数据"""
        end_year = datetime.now().year
        start_year = end_year - years

        # 从FRED获取美国通胀数据
        if country == "US":
            async with self.fred_loader as loader:
                cpi_data = await loader.get_series("CPIAUCSL", f"{start_year}-01 - 01", f"{end_year}-12 - 31")
                if cpi_data:
                    indicators = []
                    for observation in cpi_data.get('observations', []):
                        if observation.get('value') != '.':
                            indicators.append(MacroIndicator(
                                indicator_id="CPIAUCSL",
                                name="Consumer Price Index",
                                value=float(observation['value']),
                                unit="Index 1982 - 84=100",
                                frequency="Monthly",
                                date=datetime.strptime(observation['date'], '%Y-%m-%d'),
                                country="US",
                                source="FRED"
                            ))
                    return indicators

        # 从World Bank获取其他国家通胀数据
        else:
            async with self.worldbank_loader as loader:
                inflation_data = await loader.get_indicator("FP.CPI.TOTL.ZG", country, start_year, end_year)
                if inflation_data and len(inflation_data) > 1:
                    indicators = []
                    for item in inflation_data[1]:
                        if item.get('value'):
                            indicators.append(MacroIndicator(
                                indicator_id="FP.CPI.TOTL.ZG",
                                name="Inflation, consumer prices (annual %)",
                                value=float(item['value']),
                                unit="Percent",
                                frequency="Annual",
                                date=datetime.strptime(item['date'], '%Y'),
                                country=country,
                                source="World Bank"
                            ))
                    return indicators

        return []

    async def get_interest_rate_data(self, country: str = "US", years: int = 10) -> List[MacroIndicator]:
        """获取利率数据"""
        end_year = datetime.now().year
        start_year = end_year - years

        # 从FRED获取美国利率数据
        if country == "US":
            async with self.fred_loader as loader:
                fed_rate_data = await loader.get_series("FEDFUNDS", f"{start_year}-01 - 01", f"{end_year}-12 - 31")
                if fed_rate_data:
                    indicators = []
                    for observation in fed_rate_data.get('observations', []):
                        if observation.get('value') != '.':
                            indicators.append(MacroIndicator(
                                indicator_id="FEDFUNDS",
                                name="Federal Funds Rate",
                                value=float(observation['value']),
                                unit="Percent",
                                frequency="Monthly",
                                date=datetime.strptime(observation['date'], '%Y-%m-%d'),
                                country="US",
                                source="FRED"
                            ))
                    return indicators

        return []

    async def get_employment_data(self, country: str = "US", years: int = 10) -> List[MacroIndicator]:
        """获取就业数据"""
        end_year = datetime.now().year
        start_year = end_year - years

        # 从FRED获取美国就业数据
        if country == "US":
            async with self.fred_loader as loader:
                employment_data = await loader.get_series("PAYEMS", f"{start_year}-01 - 01", f"{end_year}-12 - 31")
                if employment_data:
                    indicators = []
                    for observation in employment_data.get('observations', []):
                        if observation.get('value') != '.':
                            indicators.append(MacroIndicator(
                                indicator_id="PAYEMS",
                                name="Total Nonfarm Payrolls",
                                value=float(observation['value']),
                                unit="Thousands of Persons",
                                frequency="Monthly",
                                date=datetime.strptime(observation['date'], '%Y-%m-%d'),
                                country="US",
                                source="FRED"
                            ))
                    return indicators

        return []

    async def validate_data(self, data: List[MacroIndicator]) -> Dict[str, Any]:
        """验证宏观经济数据"""
        validation_result = {
            'valid': True,
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': []
        }

        for indicator in data:
            try:
                # 基本数据验证
                if not indicator.indicator_id or not indicator.name:
                    validation_result['errors'].append(f"{indicator.indicator_id}: 缺少必要字段")
                    validation_result['invalid_records'] += 1
                    continue

                # 数值合理性检查
                if indicator.value is None or indicator.value < 0:
                    validation_result['errors'].append(f"{indicator.indicator_id}: 数值异常")
                    validation_result['invalid_records'] += 1
                    continue

                # 日期合理性检查
                if indicator.date > datetime.now():
                    validation_result['errors'].append(f"{indicator.indicator_id}: 日期异常")
                    validation_result['invalid_records'] += 1
                    continue

                validation_result['valid_records'] += 1

            except Exception as e:
                validation_result['errors'].append(f"{indicator.indicator_id}: 验证异常 - {str(e)}")
                validation_result['invalid_records'] += 1

        validation_result['valid'] = validation_result['invalid_records'] == 0

        return validation_result

    async def load_data(self, **kwargs) -> Dict[str, Any]:
        """实现IDataLoader接口的load_data方法"""
        try:
            await self.initialize()

            indicator_type = kwargs.get('indicator_type', 'gdp')
            country = kwargs.get('country', 'US')
            years = kwargs.get('years', 10)

            # 根据指标类型获取数据
            if indicator_type == 'gdp':
                macro_data = await self.get_gdp_data(country, years)
            elif indicator_type == 'inflation':
                macro_data = await self.get_inflation_data(country, years)
            elif indicator_type == 'interest_rate':
                macro_data = await self.get_interest_rate_data(country, years)
            elif indicator_type == 'employment':
                macro_data = await self.get_employment_data(country, years)
            else:
                logger.error(f"不支持的指标类型: {indicator_type}")
                return {
                    'data': pd.DataFrame(),
                    'metadata': {
                        'error': f"不支持的指标类型: {indicator_type}",
                        'timestamp': datetime.now().isoformat()
                    }
                }

            # 验证数据
            validation_result = await self.validate_data(macro_data)

            # 转换为DataFrame
            df = pd.DataFrame([{
                'indicator_id': indicator.indicator_id,
                'name': indicator.name,
                'value': indicator.value,
                'unit': indicator.unit,
                'frequency': indicator.frequency,
                'date': indicator.date,
                'country': indicator.country,
                'source': indicator.source
            } for indicator in macro_data])

            return {
                'data': df,
                'metadata': {
                    'indicator_type': indicator_type,
                    'country': country,
                    'years': years,
                    'total_records': len(macro_data),
                    'timestamp': datetime.now().isoformat(),
                    'validation_result': validation_result
                }
            }

        except Exception as e:
            logger.error(f"加载宏观经济数据时发生错误: {e}")
            return {
                'data': pd.DataFrame(),
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }


# 便捷函数
async def get_macro_data(indicator_type: str = "gdp", country: str = "US", years: int = 10) -> Dict[str, Any]:
    """获取宏观经济数据的便捷函数"""
    loader = MacroDataLoader()
    return await loader.load_data(indicator_type=indicator_type, country=country, years=years)


if __name__ == "__main__":
    # 测试代码
    async def test_macro_loader():
        """测试宏观经济数据加载器"""
        print("测试宏观经济数据加载器...")

        loader = MacroDataLoader()
        result = await loader.load_data(indicator_type="gdp", country="US", years=5)

        print(f"加载结果: {len(result['data'])} 条记录")
        print(f"元数据: {result['metadata']}")

        if not result['data'].empty:
            print("\n数据预览:")
            print(result['data'].head())

    asyncio.run(test_macro_loader())
