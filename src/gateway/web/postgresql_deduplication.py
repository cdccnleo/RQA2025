"""
PostgreSQL数据去重和冲突处理模块
实现智能的数据去重和冲突解决策略
"""

import time
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import psycopg2
import psycopg2.extras

from .postgresql_persistence import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)


class DataDeduplicationManager:
    """数据去重管理器"""

    def __init__(self):
        self.deduplication_strategies = {
            'stock_data': StockDataDeduplicationStrategy(),
            'index_data': IndexDataDeduplicationStrategy(),
            'macro_data': MacroDataDeduplicationStrategy(),
            'news_data': NewsDataDeduplicationStrategy(),
        }

    def deduplicate_data(self, data_type: str, data: List[Dict[str, Any]],
                        source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        对数据进行去重处理

        Args:
            data_type: 数据类型
            data: 原始数据列表
            source_id: 数据源ID

        Returns:
            (去重后的数据, 统计信息)
        """
        strategy = self.deduplication_strategies.get(data_type)
        if not strategy:
            logger.warning(f"未找到{data_type}的去重策略，使用默认策略")
            strategy = DefaultDeduplicationStrategy()

        return strategy.deduplicate(data, source_id)


class BaseDeduplicationStrategy:
    """基础去重策略"""

    def deduplicate(self, data: List[Dict[str, Any]], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        数据去重接口

        Returns:
            (去重后的数据, 统计信息)
        """
        raise NotImplementedError

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """获取记录的唯一键"""
        raise NotImplementedError

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """验证记录的有效性"""
        # 默认验证：检查关键字段是否存在
        return bool(record.get('source_id') and record.get('collected_at'))


class StockDataDeduplicationStrategy(BaseDeduplicationStrategy):
    """股票数据去重策略"""

    def deduplicate(self, data: List[Dict[str, Any]], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """股票数据去重"""
        seen_keys = set()
        deduplicated_data = []
        stats = {
            'original_count': len(data),
            'deduplicated_count': 0,
            'duplicates_removed': 0,
            'invalid_records': 0
        }

        for record in data:
            if not self._validate_record(record):
                stats['invalid_records'] += 1
                continue

            unique_key = self._get_unique_key(record)

            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated_data.append(record)
            else:
                stats['duplicates_removed'] += 1

        stats['deduplicated_count'] = len(deduplicated_data)

        logger.info(f"股票数据去重完成: {stats}")
        return deduplicated_data, stats

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """股票数据的唯一键：(source_id, symbol, date, data_type)"""
        return (
            record.get('source_id'),
            record.get('symbol'),
            record.get('date'),
            record.get('data_type', 'daily')
        )

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """验证股票记录"""
        required_fields = ['source_id', 'symbol', 'date']
        has_required = all(record.get(field) for field in required_fields)

        if not has_required:
            return False

        # 验证价格数据：至少要有收盘价
        has_price_data = record.get('close_price') is not None
        if not has_price_data:
            return False

        # 验证日期格式
        date_value = record.get('date')
        if isinstance(date_value, str):
            try:
                datetime.strptime(date_value, '%Y-%m-%d')
            except ValueError:
                return False

        return True


class IndexDataDeduplicationStrategy(StockDataDeduplicationStrategy):
    """指数数据去重策略（继承股票策略）"""

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """指数数据的唯一键：与股票数据相同"""
        return super()._get_unique_key(record)


class MacroDataDeduplicationStrategy(BaseDeduplicationStrategy):
    """宏观经济数据去重策略"""

    def deduplicate(self, data: List[Dict[str, Any]], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """宏观数据去重"""
        seen_keys = set()
        deduplicated_data = []
        stats = {
            'original_count': len(data),
            'deduplicated_count': 0,
            'duplicates_removed': 0,
            'invalid_records': 0
        }

        for record in data:
            if not self._validate_record(record):
                stats['invalid_records'] += 1
                continue

            unique_key = self._get_unique_key(record)

            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated_data.append(record)
            else:
                stats['duplicates_removed'] += 1

        stats['deduplicated_count'] = len(deduplicated_data)

        logger.info(f"宏观数据去重完成: {stats}")
        return deduplicated_data, stats

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """宏观数据的唯一键：(source_id, indicator_name, date)"""
        return (
            record.get('source_id'),
            record.get('indicator_name'),
            record.get('date')
        )

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """验证宏观记录"""
        required_fields = ['source_id', 'indicator_name', 'date', 'value']
        has_required = all(record.get(field) for field in required_fields)

        if not has_required:
            return False

        # 验证数值
        try:
            float(record.get('value', 0))
        except (ValueError, TypeError):
            return False

        return True


class NewsDataDeduplicationStrategy(BaseDeduplicationStrategy):
    """新闻数据去重策略"""

    def deduplicate(self, data: List[Dict[str, Any]], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """新闻数据去重（基于内容相似度）"""
        deduplicated_data = []
        stats = {
            'original_count': len(data),
            'deduplicated_count': 0,
            'duplicates_removed': 0,
            'invalid_records': 0
        }

        # 简单的基于标题和内容的去重
        seen_content_hashes = set()

        for record in data:
            if not self._validate_record(record):
                stats['invalid_records'] += 1
                continue

            content_hash = self._get_content_hash(record)

            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                deduplicated_data.append(record)
            else:
                stats['duplicates_removed'] += 1

        stats['deduplicated_count'] = len(deduplicated_data)

        logger.info(f"新闻数据去重完成: {stats}")
        return deduplicated_data, stats

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """新闻数据的唯一键：基于内容哈希"""
        return (self._get_content_hash(record),)

    def _get_content_hash(self, record: Dict[str, Any]) -> str:
        """生成内容哈希"""
        title = record.get('title', '').strip()
        content = record.get('content', '').strip()[:200]  # 只取前200字符
        return f"{title}|{content}"

    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """验证新闻记录"""
        has_title = bool(record.get('title', '').strip())
        has_content = bool(record.get('content', '').strip())

        return has_title or has_content


class DefaultDeduplicationStrategy(BaseDeduplicationStrategy):
    """默认去重策略"""

    def deduplicate(self, data: List[Dict[str, Any]], source_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """默认去重：基于source_id和timestamp"""
        seen_keys = set()
        deduplicated_data = []
        stats = {
            'original_count': len(data),
            'deduplicated_count': 0,
            'duplicates_removed': 0,
            'invalid_records': 0
        }

        for record in data:
            if not self._validate_record(record):
                stats['invalid_records'] += 1
                continue

            unique_key = self._get_unique_key(record)

            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                deduplicated_data.append(record)
            else:
                stats['duplicates_removed'] += 1

        stats['deduplicated_count'] = len(deduplicated_data)

        return deduplicated_data, stats

    def _get_unique_key(self, record: Dict[str, Any]) -> Tuple:
        """默认唯一键"""
        return (
            record.get('source_id'),
            record.get('collected_at'),
            str(record.get('data', {}))
        )


class ConflictResolutionManager:
    """冲突解决管理器"""

    def __init__(self):
        self.resolution_strategies = {
            'latest_wins': self._resolve_latest_wins,
            'highest_quality': self._resolve_highest_quality,
            'merge_data': self._resolve_merge_data,
        }

    def resolve_conflicts(self, conflicts: List[Dict[str, Any]],
                         strategy: str = 'latest_wins') -> List[Dict[str, Any]]:
        """
        解决数据冲突

        Args:
            conflicts: 冲突数据列表
            strategy: 解决策略 ('latest_wins', 'highest_quality', 'merge_data')

        Returns:
            解决后的数据列表
        """
        resolver = self.resolution_strategies.get(strategy, self._resolve_latest_wins)
        return resolver(conflicts)

    def _resolve_latest_wins(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最新优先策略"""
        # 按collected_at排序，取最新的
        sorted_conflicts = sorted(conflicts,
                                key=lambda x: x.get('collected_at', datetime.min),
                                reverse=True)
        return [sorted_conflicts[0]] if sorted_conflicts else []

    def _resolve_highest_quality(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最高质量优先策略"""
        # 计算每条记录的质量评分，取最高分
        def calculate_quality_score(record: Dict[str, Any]) -> float:
            score = 0.0
            # 检查关键字段完整性
            key_fields = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
            for field in key_fields:
                if record.get(field) is not None:
                    score += 1.0
            return score / len(key_fields)

        scored_conflicts = [(record, calculate_quality_score(record)) for record in conflicts]
        scored_conflicts.sort(key=lambda x: x[1], reverse=True)

        return [scored_conflicts[0][0]] if scored_conflicts else []

    def _resolve_merge_data(self, conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据合并策略"""
        if not conflicts:
            return []

        # 合并所有记录的数据，取非空值优先
        merged_record = {}
        for conflict in conflicts:
            for key, value in conflict.items():
                if value is not None and merged_record.get(key) is None:
                    merged_record[key] = value

        # 设置合并时间戳
        merged_record['merged_at'] = datetime.now()
        merged_record['merge_count'] = len(conflicts)

        return [merged_record]


class DataQualityValidator:
    """数据质量验证器"""

    def __init__(self):
        self.quality_rules = {
            'stock_data': self._validate_stock_data_quality,
            'index_data': self._validate_index_data_quality,
            'macro_data': self._validate_macro_data_quality,
        }

    def validate_quality(self, data_type: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证数据质量

        Returns:
            {
                'is_valid': bool,
                'quality_score': float,
                'issues': List[str],
                'warnings': List[str]
            }
        """
        validator = self.quality_rules.get(data_type, self._validate_generic_quality)
        return validator(record)

    def _validate_stock_data_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """验证股票数据质量"""
        issues = []
        warnings = []
        score = 1.0

        # 检查必需字段
        required_fields = ['symbol', 'date', 'close_price']
        for field in required_fields:
            if not record.get(field):
                issues.append(f"缺少必需字段: {field}")
                score -= 0.3

        # 检查价格逻辑
        open_price = record.get('open_price')
        high_price = record.get('high_price')
        low_price = record.get('low_price')
        close_price = record.get('close_price')

        if all([open_price, high_price, low_price, close_price]):
            if not (low_price <= open_price <= high_price and
                    low_price <= close_price <= high_price):
                issues.append("价格数据逻辑错误")
                score -= 0.4

        # 检查异常价格
        for price_field in ['open_price', 'high_price', 'low_price', 'close_price']:
            price = record.get(price_field)
            if price and (price <= 0 or price > 1000000):  # 假设股价不会超过100万
                warnings.append(f"{price_field}价格异常: {price}")

        # 检查成交量
        volume = record.get('volume')
        if volume and volume < 0:
            issues.append("成交量不能为负数")
            score -= 0.2

        return {
            'is_valid': len(issues) == 0,
            'quality_score': max(0.0, score),
            'issues': issues,
            'warnings': warnings
        }

    def _validate_index_data_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """验证指数数据质量（类似股票数据）"""
        return self._validate_stock_data_quality(record)

    def _validate_macro_data_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """验证宏观数据质量"""
        issues = []
        warnings = []
        score = 1.0

        # 检查必需字段
        required_fields = ['indicator_name', 'date', 'value']
        for field in required_fields:
            if not record.get(field):
                issues.append(f"缺少必需字段: {field}")
                score -= 0.3

        # 检查数值合理性
        value = record.get('value')
        if value is not None:
            try:
                float_value = float(value)
                # 检查异常值（可以根据具体指标设置更精确的规则）
                if abs(float_value) > 1000000000:  # 超过10亿的异常值
                    warnings.append(f"数值异常: {float_value}")
            except (ValueError, TypeError):
                issues.append("数值格式错误")
                score -= 0.2

        return {
            'is_valid': len(issues) == 0,
            'quality_score': max(0.0, score),
            'issues': issues,
            'warnings': warnings
        }

    def _validate_generic_quality(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """通用质量验证"""
        return {
            'is_valid': True,
            'quality_score': 0.8,
            'issues': [],
            'warnings': ['使用通用质量验证']
        }


# 全局实例
_deduplication_manager = None
_conflict_resolution_manager = None
_quality_validator = None


def get_deduplication_manager() -> DataDeduplicationManager:
    """获取去重管理器实例"""
    global _deduplication_manager
    if _deduplication_manager is None:
        _deduplication_manager = DataDeduplicationManager()
    return _deduplication_manager


def get_conflict_resolution_manager() -> ConflictResolutionManager:
    """获取冲突解决管理器实例"""
    global _conflict_resolution_manager
    if _conflict_resolution_manager is None:
        _conflict_resolution_manager = ConflictResolutionManager()
    return _conflict_resolution_manager


def get_quality_validator() -> DataQualityValidator:
    """获取质量验证器实例"""
    global _quality_validator
    if _quality_validator is None:
        _quality_validator = DataQualityValidator()
    return _quality_validator