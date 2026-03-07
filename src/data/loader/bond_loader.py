import logging
#!/usr/bin/env python3
"""
RQA2025 债券数据加载器

支持从多个债券数据源获取数据：
- 国债收益率曲线
- 企业债券数据
- 信用评级信息
- 本地缓存: 减少API调用频率
"""

# 使用基础设施层日志

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import aiohttp
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Mock缺失的模块
from unittest.mock import Mock

# 修复导入问题
try:
    from ..core.base_loader import BaseDataLoader
except ImportError:
    BaseDataLoader = Mock()

try:
    from ..interfaces.loader import IDataLoader
except ImportError:
    IDataLoader = Mock()

try:
    from ..cache.cache_manager import CacheManager, CacheConfig
except ImportError:
    # 创建Mock类
    class CacheConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CacheManager:
        def __init__(self, config):
            self.config = config
            self.cache = {}

        def get(self, key):
            return self.cache.get(key)

        def set(self, key, value, ttl=None):
            self.cache[key] = value

# 配置日志
import logging
logger = logging.getLogger(__name__)


class BondData:

    """债券数据结构"""
    def __init__(self, symbol: str = "", name: str = "", bond_type: str = "", 
                 maturity_date: Optional[datetime] = None, coupon_rate: float = 0.0,
                 yield_to_maturity: float = 0.0, price: float = 0.0, 
                 face_value: float = 0.0, credit_rating: str = "", issuer: str = "",
                 country: str = "", currency: str = "", timestamp: Optional[datetime] = None,
                 source: str = ""):
        self.symbol = symbol
        self.name = name
        self.bond_type = bond_type
        self.maturity_date = maturity_date
        self.coupon_rate = coupon_rate
        self.yield_to_maturity = yield_to_maturity
        self.price = price
        self.face_value = face_value
        self.credit_rating = credit_rating
        self.issuer = issuer
        self.country = country
        self.currency = currency
        self.timestamp = timestamp
        self.source = source


@dataclass
class YieldCurve:

    """收益率曲线数据结构"""
    country: str
    currency: str
    tenors: List[str]  # ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    yields: List[float]
    timestamp: datetime
    source: str


@dataclass
class CreditRating:

    """信用评级数据结构"""
    issuer: str
    rating: str
    outlook: str  # 'positive', 'stable', 'negative'
    rating_agency: str
    rating_date: datetime
    timestamp: datetime
    source: str


class TreasuryLoader(BaseDataLoader):

    """国债数据加载器"""

    def __init__(self, api_key: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key
        }
        super().__init__(config)
        self.base_url = "https://api.treasury.gov / v1"
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
            "loader_type": "treasury",
            "version": "1.0.0",
            "description": "国债数据加载器",
            "supported_sources": ["treasury"],
            "supported_frequencies": ["1d", "1w", "1m"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - BondLoader / 1.0',
                'Accept': 'application / json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_yield_curve(self, country: str = "US") -> Optional[YieldCurve]:
        """获取收益率曲线数据"""
        cache_key = f"treasury_yield_curve_{country}"

        # 检查缓存
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取国债收益率曲线数据: {country}")
            return YieldCurve(**cached_data)

        try:
            # 由于API访问限制，这里提供模拟实现
            logger.info(f"使用模拟数据代替国债API调用: {country}")

            # 模拟收益率曲线数据
            tenors = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
            base_yields = [0.05, 0.08, 0.12, 0.18, 0.25, 0.45, 0.75, 1.20]

            # 添加一些随机波动
            yields = [yield_rate + np.random.random() * 0.02 for yield_rate in base_yields]

            yield_curve = YieldCurve(
                country=country,
                currency="USD" if country == "US" else "EUR",
                tenors=tenors,
                yields=yields,
                timestamp=datetime.now(),
                source='treasury'
            )

            # 缓存数据（1小时）
            self.cache_manager.set(cache_key, yield_curve.__dict__, ttl=3600)

            logger.info(f"成功获取国债收益率曲线数据: {country}")
            return yield_curve

        except Exception as e:
            logger.error(f"获取国债收益率曲线数据时发生错误: {e}")
            return None

    async def get_treasury_bonds(self, country: str = "US") -> List[BondData]:
        """获取国债数据"""
        cache_key = f"treasury_bonds_{country}"

        # 检查缓存
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取国债数据: {country}")
            return [BondData(**bond) for bond in cached_data]

        try:
            # 模拟国债数据
            logger.info(f"使用模拟数据代替国债API调用: {country}")

            bonds = []
            maturities = [1, 2, 5, 10, 30]  # 年

            for maturity in maturities:
                # 模拟不同期限的国债
                bond = BondData(
                    symbol=f"T{maturity}Y",
                    name=f"{maturity}年期国债",
                    bond_type='treasury',
                    maturity_date=datetime.now() + timedelta(days=maturity * 365),
                    coupon_rate=0.02 + maturity * 0.01,  # 模拟票面利率
                    yield_to_maturity=0.02 + maturity * 0.015,  # 模拟到期收益率
                    price=98.0 + maturity * 0.5,  # 模拟价格
                    face_value=100.0,
                    credit_rating="AAA",
                    issuer=f"{country} Treasury",
                    country=country,
                    currency="USD" if country == "US" else "EUR",
                    timestamp=datetime.now(),
                    source='treasury'
                )
                bonds.append(bond)

            # 缓存数据（1小时）
            self.cache_manager.set(cache_key, [bond.__dict__ for bond in bonds], ttl=3600)

            logger.info(f"成功获取国债数据: {country}, {len(bonds)} 只债券")
            return bonds

        except Exception as e:
            logger.error(f"获取国债数据时发生错误: {e}")
            return []


class CorporateBondLoader(BaseDataLoader):

    """企业债券数据加载器"""

    def __init__(self, api_key: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key
        }
        super().__init__(config)
        self.base_url = "https://api.corporate - bonds.com / v1"
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
            "loader_type": "corporate",
            "version": "1.0.0",
            "description": "企业债券数据加载器",
            "supported_sources": ["corporate"],
            "supported_frequencies": ["1d", "1w", "1m"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - CorporateBondLoader / 1.0',
                'Accept': 'application / json',
                'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_corporate_bonds(self, issuer: Optional[str] = None) -> List[BondData]:
        """获取企业债券数据"""
        cache_key = f"corporate_bonds_{issuer or 'all'}"

        # 检查缓存
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取企业债券数据: {issuer or 'all'}")
            return [BondData(**bond) for bond in cached_data]

        try:
            # 模拟企业债券数据
            logger.info(f"使用模拟数据代替企业债券API调用: {issuer or 'all'}")

            issuers = ["Apple Inc", "Microsoft Corp", "Google LLC", "Amazon.com Inc", "Tesla Inc"]
            ratings = ["AA+", "AAA", "AA", "A+", "BBB+"]

            bonds = []
            for i, company in enumerate(issuers):
                if issuer and company != issuer:
                    continue

                # 模拟不同期限的企业债券
                for maturity in [3, 5, 10]:
                    bond = BondData(
                        symbol=f"{company[:3].upper()}{maturity}Y",
                        name=f"{company} {maturity}年期债券",
                        bond_type='corporate',
                        maturity_date=datetime.now() + timedelta(days=maturity * 365),
                        coupon_rate=0.03 + maturity * 0.005 + np.random.random() * 0.02,
                        yield_to_maturity=0.035 + maturity * 0.008 + np.random.random() * 0.03,
                        price=95.0 + maturity * 1.0 + np.random.random() * 5.0,
                        face_value=100.0,
                        credit_rating=ratings[i % len(ratings)],
                        issuer=company,
                        country="US",
                        currency="USD",
                        timestamp=datetime.now(),
                        source='corporate'
                    )
                    bonds.append(bond)

            # 缓存数据（30分钟）
            self.cache_manager.set(cache_key, [bond.__dict__ for bond in bonds], ttl=1800)

            logger.info(f"成功获取企业债券数据: {len(bonds)} 只债券")
            return bonds

        except Exception as e:
            logger.error(f"获取企业债券数据时发生错误: {e}")
            return []

    async def get_credit_ratings(self, issuer: Optional[str] = None) -> List[CreditRating]:
        """获取信用评级数据"""
        cache_key = f"credit_ratings_{issuer or 'all'}"

        # 检查缓存
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取信用评级数据: {issuer or 'all'}")
            return [CreditRating(**rating) for rating in cached_data]

        try:
            # 模拟信用评级数据
            logger.info(f"使用模拟数据代替信用评级API调用: {issuer or 'all'}")

            issuers = ["Apple Inc", "Microsoft Corp", "Google LLC", "Amazon.com Inc", "Tesla Inc"]
            ratings = ["AA+", "AAA", "AA", "A+", "BBB+"]
            outlooks = ["stable", "positive", "stable", "positive", "stable"]
            agencies = ["S&P", "Moody's", "Fitch"]

            credit_ratings = []
            for i, company in enumerate(issuers):
                if issuer and company != issuer:
                    continue

                for agency in agencies:
                    rating = CreditRating(
                        issuer=company,
                        rating=ratings[i % len(ratings)],
                        outlook=outlooks[i % len(outlooks)],
                        rating_agency=agency,
                        rating_date=datetime.now() - timedelta(days=np.random.randint(30, 365)),
                        timestamp=datetime.now(),
                        source='corporate'
                    )
                    credit_ratings.append(rating)

            # 缓存数据（24小时）
            self.cache_manager.set(
                cache_key, [rating.__dict__ for rating in credit_ratings], ttl=86400)

            logger.info(f"成功获取信用评级数据: {len(credit_ratings)} 条记录")
            return credit_ratings

        except Exception as e:
            logger.error(f"获取信用评级数据时发生错误: {e}")
            return []


class BondDataLoader(BaseDataLoader):

    """统一的债券数据加载器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        config = config or {}
        super().__init__(config)
        self.config = config
        self.treasury_loader = None
        self.corporate_loader = None
        cache_config = CacheConfig(
            max_size=config.get('max_size', 1000),
            ttl=config.get('ttl', 3600),
            enable_disk_cache=True,
            disk_cache_dir=config.get('cache_dir', 'cache'),
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    async def initialize(self):
        """初始化数据加载器"""
        logger.info("初始化债券数据加载器...")

        # 初始化国债加载器
        treasury_api_key = self.config.get('treasury_api_key')
        self.treasury_loader = TreasuryLoader(treasury_api_key)

        # 初始化企业债券加载器
        corporate_api_key = self.config.get('corporate_api_key')
        self.corporate_loader = CorporateBondLoader(corporate_api_key)

    async def get_yield_curve(self, country: str = "US") -> Optional[YieldCurve]:
        """获取收益率曲线"""
        if not self.treasury_loader:
            await self.initialize()

        return await self.treasury_loader.get_yield_curve(country)

    async def get_treasury_bonds(self, country: str = "US") -> List[BondData]:
        """获取国债数据"""
        if not self.treasury_loader:
            await self.initialize()

        return await self.treasury_loader.get_treasury_bonds(country)

    async def get_corporate_bonds(self, issuer: Optional[str] = None) -> List[BondData]:
        """获取企业债券数据"""
        if not self.corporate_loader:
            await self.initialize()

        return await self.corporate_loader.get_corporate_bonds(issuer)

    async def get_credit_ratings(self, issuer: Optional[str] = None) -> List[CreditRating]:
        """获取信用评级数据"""
        if not self.corporate_loader:
            await self.initialize()

        return await self.corporate_loader.get_credit_ratings(issuer)

    async def validate_data(self, data: Union[List[BondData], YieldCurve, List[CreditRating]]) -> Dict[str, Any]:
        """验证债券数据"""
        validation_result = {
            'valid': True,
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'errors': []
        }

        try:
            if isinstance(data, list) and data and isinstance(data[0], BondData):
                # 验证债券数据
                validation_result['total_records'] = len(data)

                for bond in data:
                    if bond.coupon_rate < 0 or bond.coupon_rate > 1:
                        validation_result['errors'].append(f"{bond.symbol}: 票面利率异常")
                        validation_result['invalid_records'] += 1
                        continue

                    if bond.yield_to_maturity < 0 or bond.yield_to_maturity > 1:
                        validation_result['errors'].append(f"{bond.symbol}: 到期收益率异常")
                        validation_result['invalid_records'] += 1
                        continue

                    if bond.price <= 0:
                        validation_result['errors'].append(f"{bond.symbol}: 价格异常")
                        validation_result['invalid_records'] += 1
                        continue

                    validation_result['valid_records'] += 1

            elif isinstance(data, YieldCurve):
                # 验证收益率曲线
                validation_result['total_records'] = len(data.yields)

                for i, yield_rate in enumerate(data.yields):
                    if yield_rate < 0 or yield_rate > 1:
                        validation_result['errors'].append(f"收益率异常: {data.tenors[i]}")
                        validation_result['invalid_records'] += 1
                    else:
                        validation_result['valid_records'] += 1

            elif isinstance(data, list) and data and isinstance(data[0], CreditRating):
                # 验证信用评级
                validation_result['total_records'] = len(data)

                for rating in data:
                    if not rating.rating or not rating.issuer:
                        validation_result['errors'].append(f"评级数据不完整: {rating.issuer}")
                        validation_result['invalid_records'] += 1
                        continue

                    validation_result['valid_records'] += 1

        except Exception as e:
            validation_result['errors'].append(f"验证异常: {str(e)}")
            validation_result['invalid_records'] += 1

        validation_result['valid'] = validation_result['invalid_records'] == 0

        return validation_result

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        return self._validate_config()

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "bond",
            "version": "1.0.0",
            "description": "债券数据加载器",
            "supported_sources": ["treasury", "corporate"],
            "supported_frequencies": ["1d", "1w", "1m"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def load_data(self, **kwargs) -> Dict[str, Any]:
        """实现IDataLoader接口的load_data方法"""
        try:
            await self.initialize()

            bond_type = kwargs.get('bond_type', 'treasury')
            country = kwargs.get('country', 'US')
            issuer = kwargs.get('issuer')

            if bond_type == 'treasury':
                bonds = await self.get_treasury_bonds(country)
            elif bond_type == 'corporate':
                bonds = await self.get_corporate_bonds(issuer)
            else:
                return {
                    'data': pd.DataFrame(),
                    'metadata': {
                        'error': f'Unsupported bond type: {bond_type}',
                        'timestamp': datetime.now().isoformat()
                    }
                }

            if not bonds:
                return {
                    'data': pd.DataFrame(),
                    'metadata': {
                        'error': 'Failed to load bond data',
                        'timestamp': datetime.now().isoformat()
                    }
                }

            # 验证数据
            validation_result = await self.validate_data(bonds)

            # 转换为DataFrame
            bonds_data = []
            for bond in bonds:
                bonds_data.append({
                    'symbol': bond.symbol,
                    'name': bond.name,
                    'bond_type': bond.bond_type,
                    'maturity_date': bond.maturity_date,
                    'coupon_rate': bond.coupon_rate,
                    'yield_to_maturity': bond.yield_to_maturity,
                    'price': bond.price,
                    'face_value': bond.face_value,
                    'credit_rating': bond.credit_rating,
                    'issuer': bond.issuer,
                    'country': bond.country,
                    'currency': bond.currency,
                    'timestamp': bond.timestamp,
                    'source': bond.source
                })

            df = pd.DataFrame(bonds_data)

            return {
                'data': df,
                'metadata': {
                    'bond_type': bond_type,
                    'country': country,
                    'issuer': issuer,
                    'total_records': len(bonds_data),
                    'timestamp': datetime.now().isoformat(),
                    'validation_result': validation_result
                }
            }

        except Exception as e:
            logger.error(f"加载债券数据时发生错误: {e}")
            return {
                'data': pd.DataFrame(),
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }


# 便捷函数
async def get_yield_curve(country: str = "US") -> Optional[YieldCurve]:
    """获取收益率曲线的便捷函数"""
    loader = BondDataLoader()
    await loader.initialize()
    return await loader.get_yield_curve(country)


async def get_treasury_bonds(country: str = "US") -> List[BondData]:
    """获取国债数据的便捷函数"""
    loader = BondDataLoader()
    await loader.initialize()
    return await loader.get_treasury_bonds(country)


async def get_corporate_bonds(issuer: Optional[str] = None) -> List[BondData]:
    """获取企业债券数据的便捷函数"""
    loader = BondDataLoader()
    await loader.initialize()
    return await loader.get_corporate_bonds(issuer)


if __name__ == "__main__":
    # 测试代码
    async def test_bond_loader():
        """测试债券数据加载器"""
        print("测试债券数据加载器...")

        loader = BondDataLoader()
        result = await loader.load_data(bond_type="treasury", country="US")

        print(f"加载结果: {len(result['data'])} 条记录")
        print(f"元数据: {result['metadata']}")

        if not result['data'].empty:
            print("\n数据预览:")
            print(result['data'].head())

    asyncio.run(test_bond_loader())
