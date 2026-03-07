"""
incremental_calculator.py

增量特征计算器模块

提供增量特征计算功能，支持：
- 只计算新增数据，避免重复计算
- 特征结果缓存和合并
- 计算历史记录管理
- 自动检测数据更新

适用于A股市场大规模特征计算场景，显著提高计算效率。

作者: RQA2025 Team
日期: 2026-02-13
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class CalculationRecord:
    """计算记录数据类"""
    symbol: str
    feature_type: str
    last_calc_date: str
    data_hash: str
    feature_count: int
    calc_time_ms: float
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class IncrementalConfig:
    """增量计算配置"""
    enable_cache: bool = True
    cache_ttl_hours: int = 24
    auto_detect_update: bool = True
    merge_strategy: str = "append"  # append/overwrite
    min_incremental_days: int = 1   # 最少增量天数


class IncrementalFeatureCalculator:
    """
    增量特征计算器
    
    支持增量特征计算，避免重复计算已处理的数据，
    显著提高大规模特征计算效率。
    
    Attributes:
        config: 增量计算配置
        _calculation_records: 计算记录缓存
        _feature_cache: 特征结果缓存
        
    Example:
        >>> calculator = IncrementalFeatureCalculator()
        >>> 
        >>> # 检查是否需要重新计算
        >>> if calculator.needs_recalculation("002837", "2026-02-13"):
        ...     # 加载新增数据
        ...     new_data = calculator.load_incremental_data("002837", last_date)
        ...     # 计算新增特征
        ...     new_features = calculator.calculate_features(new_data)
        ...     # 合并特征
        ...     all_features = calculator.merge_features("002837", new_features)
    """
    
    def __init__(self, config: Optional[IncrementalConfig] = None):
        """
        初始化增量特征计算器
        
        Args:
            config: 增量计算配置
        """
        self.config = config or IncrementalConfig()
        self._calculation_records: Dict[str, CalculationRecord] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        
        # 初始化数据加载器
        try:
            from src.data_management.loaders import PostgreSQLDataLoader, DataLoaderConfig
            loader_config = DataLoaderConfig(source_type="postgresql")
            self.data_loader = PostgreSQLDataLoader(loader_config)
            logger.info("IncrementalFeatureCalculator 初始化完成")
        except Exception as e:
            logger.error(f"初始化数据加载器失败: {e}")
            self.data_loader = None
        
        # 加载历史计算记录
        self._load_calculation_records()
    
    def _get_record_key(self, symbol: str, feature_type: str) -> str:
        """生成记录键"""
        return f"{symbol}_{feature_type}"
    
    def _load_calculation_records(self):
        """从持久化存储加载计算记录"""
        try:
            # 从PostgreSQL加载计算记录
            if self.data_loader:
                query = """
                    SELECT symbol, feature_type, last_calc_date, data_hash, 
                           feature_count, calc_time_ms, created_at, updated_at
                    FROM feature_calculation_records
                    ORDER BY updated_at DESC
                """
                result = self.data_loader.load(query)
                
                if result.success and result.data is not None:
                    for _, row in result.data.iterrows():
                        key = self._get_record_key(row['symbol'], row['feature_type'])
                        self._calculation_records[key] = CalculationRecord(
                            symbol=row['symbol'],
                            feature_type=row['feature_type'],
                            last_calc_date=row['last_calc_date'],
                            data_hash=row['data_hash'],
                            feature_count=row['feature_count'],
                            calc_time_ms=row['calc_time_ms'],
                            created_at=row['created_at'],
                            updated_at=row['updated_at']
                        )
                    
                    logger.info(f"加载了 {len(self._calculation_records)} 条计算记录")
        
        except Exception as e:
            logger.warning(f"加载计算记录失败: {e}")
    
    def _save_calculation_record(self, record: CalculationRecord):
        """保存计算记录到持久化存储"""
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # 使用UPSERT语法
            cursor.execute("""
                INSERT INTO feature_calculation_records 
                (symbol, feature_type, last_calc_date, data_hash, feature_count, calc_time_ms, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, feature_type) 
                DO UPDATE SET
                    last_calc_date = EXCLUDED.last_calc_date,
                    data_hash = EXCLUDED.data_hash,
                    feature_count = EXCLUDED.feature_count,
                    calc_time_ms = EXCLUDED.calc_time_ms,
                    updated_at = EXCLUDED.updated_at
            """, (
                record.symbol,
                record.feature_type,
                record.last_calc_date,
                record.data_hash,
                record.feature_count,
                record.calc_time_ms,
                record.created_at,
                record.updated_at
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
        except Exception as e:
            logger.error(f"保存计算记录失败: {e}")
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """计算数据哈希值"""
        # 使用数据的统计特征计算哈希
        hash_input = f"{len(df)}_{df['date'].min()}_{df['date'].max()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_last_calculation(self, symbol: str, feature_type: str) -> Optional[CalculationRecord]:
        """
        获取上次计算记录
        
        Args:
            symbol: 股票代码
            feature_type: 特征类型
            
        Returns:
            计算记录，如果没有则返回None
        """
        key = self._get_record_key(symbol, feature_type)
        return self._calculation_records.get(key)
    
    def needs_recalculation(
        self,
        symbol: str,
        end_date: str,
        feature_type: str = "technical"
    ) -> Tuple[bool, Optional[str]]:
        """
        检查是否需要重新计算
        
        Args:
            symbol: 股票代码
            end_date: 数据截止日期
            feature_type: 特征类型
            
        Returns:
            (是否需要重新计算, 上次计算日期)
        """
        record = self.get_last_calculation(symbol, feature_type)
        
        if not record:
            logger.info(f"{symbol} 没有历史计算记录，需要完整计算")
            return True, None
        
        # 检查缓存是否过期
        cache_age = datetime.now() - record.updated_at
        if cache_age > timedelta(hours=self.config.cache_ttl_hours):
            logger.info(f"{symbol} 缓存已过期，需要重新计算")
            return True, record.last_calc_date
        
        # 检查是否有新数据
        if end_date > record.last_calc_date:
            days_diff = (datetime.strptime(end_date, "%Y-%m-%d") - 
                        datetime.strptime(record.last_calc_date, "%Y-%m-%d")).days
            
            if days_diff >= self.config.min_incremental_days:
                logger.info(f"{symbol} 有 {days_diff} 天新数据，需要增量计算")
                return True, record.last_calc_date
        
        logger.info(f"{symbol} 数据未更新，无需重新计算")
        return False, record.last_calc_date
    
    def load_incremental_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        加载增量数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            增量数据DataFrame
        """
        if not self.data_loader:
            logger.error("数据加载器不可用")
            return None
        
        try:
            result = self.data_loader.load_stock_data(symbol, start_date, end_date)
            
            if result.success and result.data is not None:
                logger.info(f"加载 {symbol} 增量数据: {len(result.data)} 条记录")
                return result.data
            else:
                logger.warning(f"加载 {symbol} 增量数据失败: {result.message}")
                return None
        
        except Exception as e:
            logger.error(f"加载增量数据失败: {e}")
            return None
    
    def calculate_features(
        self,
        data: pd.DataFrame,
        feature_type: str = "technical"
    ) -> Optional[pd.DataFrame]:
        """
        计算特征
        
        Args:
            data: 股票数据
            feature_type: 特征类型
            
        Returns:
            特征DataFrame
        """
        try:
            from src.gateway.web.feature_engineering_service import get_feature_engine
            
            engine = get_feature_engine()
            if engine and hasattr(engine, 'process_features'):
                features = engine.process_features(data)
                logger.info(f"计算特征完成: {len(features.columns)} 个特征")
                return features
            else:
                logger.warning("特征引擎不可用")
                return None
        
        except Exception as e:
            logger.error(f"计算特征失败: {e}")
            return None
    
    def get_cached_features(self, symbol: str, feature_type: str) -> Optional[pd.DataFrame]:
        """
        获取缓存的特征
        
        Args:
            symbol: 股票代码
            feature_type: 特征类型
            
        Returns:
            缓存的特征DataFrame
        """
        if not self.config.enable_cache:
            return None
        
        key = self._get_record_key(symbol, feature_type)
        
        # 检查内存缓存
        if key in self._feature_cache:
            logger.debug(f"从内存缓存获取 {symbol} 特征")
            return self._feature_cache[key]
        
        # 从PostgreSQL加载
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return None
            
            # 检查特征表是否存在
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information.tables 
                    WHERE table_name = 'cached_features'
                )
            """)
            
            if not cursor.fetchone()[0]:
                cursor.close()
                return_db_connection(conn)
                return None
            
            # 加载特征
            query = """
                SELECT feature_data
                FROM cached_features
                WHERE symbol = %s AND feature_type = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            cursor.execute(query, (symbol, feature_type))
            row = cursor.fetchone()
            cursor.close()
            return_db_connection(conn)
            
            if row:
                # 解析JSON数据
                import json
                feature_data = json.loads(row[0])
                df = pd.DataFrame(feature_data)
                
                # 缓存到内存
                self._feature_cache[key] = df
                
                logger.debug(f"从数据库加载 {symbol} 特征: {len(df)} 条")
                return df
        
        except Exception as e:
            logger.warning(f"加载缓存特征失败: {e}")
        
        return None
    
    def cache_features(
        self,
        symbol: str,
        feature_type: str,
        features: pd.DataFrame
    ):
        """
        缓存特征结果
        
        Args:
            symbol: 股票代码
            feature_type: 特征类型
            features: 特征DataFrame
        """
        if not self.config.enable_cache:
            return
        
        key = self._get_record_key(symbol, feature_type)
        
        # 缓存到内存
        self._feature_cache[key] = features
        
        # 保存到PostgreSQL
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # 确保表存在
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cached_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    feature_type VARCHAR(50) NOT NULL,
                    feature_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, feature_type)
                )
            """)
            
            # 转换为JSON
            feature_json = features.to_json(orient='records', date_format='iso')
            
            cursor.execute("""
                INSERT INTO cached_features (symbol, feature_type, feature_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol, feature_type)
                DO UPDATE SET
                    feature_data = EXCLUDED.feature_data,
                    created_at = CURRENT_TIMESTAMP
            """, (symbol, feature_type, feature_json))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.debug(f"缓存 {symbol} 特征到数据库")
        
        except Exception as e:
            logger.warning(f"缓存特征失败: {e}")
    
    def merge_features(
        self,
        symbol: str,
        new_features: pd.DataFrame,
        feature_type: str = "technical"
    ) -> pd.DataFrame:
        """
        合并新旧特征
        
        Args:
            symbol: 股票代码
            new_features: 新计算的特征
            feature_type: 特征类型
            
        Returns:
            合并后的特征
        """
        # 获取缓存的特征
        cached_features = self.get_cached_features(symbol, feature_type)
        
        if cached_features is None or self.config.merge_strategy == "overwrite":
            logger.info(f"{symbol} 使用新特征覆盖")
            return new_features
        
        # 合并特征
        try:
            # 假设都有date列
            if 'date' in cached_features.columns and 'date' in new_features.columns:
                # 去重合并
                combined = pd.concat([cached_features, new_features], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date'], keep='last')
                combined = combined.sort_values('date')
                
                logger.info(f"{symbol} 合并特征: 缓存 {len(cached_features)} 条 + 新增 {len(new_features)} 条 = {len(combined)} 条")
                return combined
            else:
                logger.warning(f"{symbol} 特征缺少date列，使用新特征覆盖")
                return new_features
        
        except Exception as e:
            logger.error(f"合并特征失败: {e}")
            return new_features
    
    def calculate_incremental(
        self,
        symbol: str,
        end_date: str,
        feature_type: str = "technical"
    ) -> Optional[pd.DataFrame]:
        """
        执行增量计算（主入口）
        
        Args:
            symbol: 股票代码
            end_date: 数据截止日期
            feature_type: 特征类型
            
        Returns:
            特征DataFrame
        """
        import time
        start_time = time.time()
        
        # 检查是否需要重新计算
        needs_calc, last_calc_date = self.needs_recalculation(symbol, end_date, feature_type)
        
        if not needs_calc:
            # 返回缓存的特征
            cached = self.get_cached_features(symbol, feature_type)
            if cached is not None:
                logger.info(f"{symbol} 返回缓存特征，无需计算")
                return cached
        
        # 确定计算范围
        if last_calc_date:
            # 增量计算：从上次的下一天开始
            start_date = (datetime.strptime(last_calc_date, "%Y-%m-%d") + 
                         timedelta(days=1)).strftime("%Y-%m-%d")
            calc_mode = "增量"
        else:
            # 完整计算：从历史数据开始
            start_date = "2020-01-01"
            calc_mode = "完整"
        
        logger.info(f"{symbol} 开始{calc_mode}计算: {start_date} ~ {end_date}")
        
        # 加载数据
        data = self.load_incremental_data(symbol, start_date, end_date)
        if data is None or data.empty:
            logger.warning(f"{symbol} 没有新数据")
            return self.get_cached_features(symbol, feature_type)
        
        # 计算特征
        new_features = self.calculate_features(data, feature_type)
        if new_features is None:
            logger.error(f"{symbol} 特征计算失败")
            return None
        
        # 合并特征
        if calc_mode == "增量":
            all_features = self.merge_features(symbol, new_features, feature_type)
        else:
            all_features = new_features
        
        # 缓存结果
        self.cache_features(symbol, feature_type, all_features)
        
        # 更新计算记录
        calc_time_ms = (time.time() - start_time) * 1000
        record = CalculationRecord(
            symbol=symbol,
            feature_type=feature_type,
            last_calc_date=end_date,
            data_hash=self._compute_data_hash(data),
            feature_count=len(all_features.columns),
            calc_time_ms=calc_time_ms
        )
        self._calculation_records[self._get_record_key(symbol, feature_type)] = record
        self._save_calculation_record(record)
        
        logger.info(f"{symbol} {calc_mode}计算完成: {len(all_features.columns)} 个特征，"
                   f"耗时 {calc_time_ms:.2f}ms")
        
        return all_features
    
    def batch_calculate_incremental(
        self,
        symbols: List[str],
        end_date: str,
        feature_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        批量增量计算
        
        Args:
            symbols: 股票代码列表
            end_date: 数据截止日期
            feature_type: 特征类型
            
        Returns:
            计算结果统计
        """
        results = {
            "total": len(symbols),
            "calculated": 0,
            "cached": 0,
            "failed": 0,
            "details": []
        }
        
        for symbol in symbols:
            try:
                needs_calc, _ = self.needs_recalculation(symbol, end_date, feature_type)
                
                if needs_calc:
                    features = self.calculate_incremental(symbol, end_date, feature_type)
                    if features is not None:
                        results["calculated"] += 1
                        results["details"].append({
                            "symbol": symbol,
                            "status": "calculated",
                            "features": len(features.columns)
                        })
                    else:
                        results["failed"] += 1
                        results["details"].append({
                            "symbol": symbol,
                            "status": "failed"
                        })
                else:
                    results["cached"] += 1
                    results["details"].append({
                        "symbol": symbol,
                        "status": "cached"
                    })
            
            except Exception as e:
                logger.error(f"批量计算 {symbol} 失败: {e}")
                results["failed"] += 1
                results["details"].append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"批量计算完成: 总计 {results['total']}, "
                   f"计算 {results['calculated']}, "
                   f"缓存 {results['cached']}, "
                   f"失败 {results['failed']}")
        
        return results
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """
        获取计算统计信息
        
        Returns:
            统计信息
        """
        total_records = len(self._calculation_records)
        
        if total_records == 0:
            return {"total_records": 0}
        
        # 计算平均特征数
        avg_features = sum(r.feature_count for r in self._calculation_records.values()) / total_records
        
        # 计算平均计算时间
        avg_calc_time = sum(r.calc_time_ms for r in self._calculation_records.values()) / total_records
        
        # 按特征类型分组
        type_distribution = {}
        for record in self._calculation_records.values():
            feature_type = record.feature_type
            if feature_type not in type_distribution:
                type_distribution[feature_type] = 0
            type_distribution[feature_type] += 1
        
        return {
            "total_records": total_records,
            "avg_features": round(avg_features, 2),
            "avg_calc_time_ms": round(avg_calc_time, 2),
            "type_distribution": type_distribution,
            "memory_cached_features": len(self._feature_cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._feature_cache.clear()
        logger.info("特征缓存已清除")
    
    def close(self):
        """关闭计算器"""
        if self.data_loader:
            self.data_loader.close()
        logger.info("IncrementalFeatureCalculator 已关闭")


# 全局计算器实例（单例模式）
_global_calculator: Optional[IncrementalFeatureCalculator] = None


def get_incremental_calculator(config: Optional[IncrementalConfig] = None) -> IncrementalFeatureCalculator:
    """
    获取全局增量特征计算器实例
    
    Args:
        config: 计算器配置
        
    Returns:
        增量特征计算器实例
    """
    global _global_calculator
    
    if _global_calculator is None:
        _global_calculator = IncrementalFeatureCalculator(config)
    
    return _global_calculator


def close_incremental_calculator():
    """关闭全局增量特征计算器实例"""
    global _global_calculator
    
    if _global_calculator:
        _global_calculator.close()
        _global_calculator = None
