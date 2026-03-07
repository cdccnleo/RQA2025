"""
增量数据合并策略优化器
实现高效的数据合并和冲突解决
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, date
from decimal import Decimal
from collections import defaultdict

from .postgresql_deduplication import (
    get_deduplication_manager,
    get_conflict_resolution_manager,
    get_quality_validator
)

logger = logging.getLogger(__name__)


class DataMergeOptimizer:
    """数据合并优化器"""

    def __init__(self):
        self.deduplication_manager = get_deduplication_manager()
        self.conflict_resolver = get_conflict_resolution_manager()
        self.quality_validator = get_quality_validator()

    def optimize_incremental_merge(self, source_id: str, new_data: List[Dict[str, Any]],
                                 data_type: str, merge_strategy: str = 'latest_wins') -> Dict[str, Any]:
        """
        优化增量数据合并

        Args:
            source_id: 数据源ID
            new_data: 新采集的数据
            data_type: 数据类型
            merge_strategy: 合并策略 ('latest_wins', 'highest_quality', 'merge_data')

        Returns:
            合并结果统计
        """
        start_time = datetime.now()

        try:
            # 1. 数据去重
            deduplicated_data, dedup_stats = self.deduplication_manager.deduplicate_data(
                data_type, new_data, source_id
            )

            if not deduplicated_data:
                return {
                    'success': True,
                    'merged_count': 0,
                    'skipped_count': len(new_data),
                    'quality_score': 0.0,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'message': '所有数据都被去重过滤'
                }

            # 2. 数据质量验证
            quality_validated_data, quality_stats = self._validate_data_quality(
                deduplicated_data, data_type
            )

            # 3. 数据标准化
            standardized_data = self._standardize_data(quality_validated_data, data_type)

            # 4. 冲突检测和解决
            final_data, conflict_stats = self._resolve_data_conflicts(
                standardized_data, data_type, merge_strategy
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'original_count': len(new_data),
                'deduplicated_count': len(deduplicated_data),
                'quality_validated_count': len(quality_validated_data),
                'final_count': len(final_data),
                'processing_time': processing_time,
                'merge_stats': {
                    'deduplication': dedup_stats,
                    'quality_validation': quality_stats,
                    'conflict_resolution': conflict_stats
                },
                'data': final_data
            }

            logger.info(
                f"增量数据合并优化完成: {source_id}, "
                f"原始{len(new_data)} -> 去重{len(deduplicated_data)} -> "
                f"质量验证{len(quality_validated_data)} -> 最终{len(final_data)}, "
                f"耗时{processing_time:.2f}秒"
            )

            return result

        except Exception as e:
            error_msg = f"增量数据合并优化异常: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    def _validate_data_quality(self, data: List[Dict[str, Any]], data_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """数据质量验证"""
        validated_data = []
        stats = {
            'total_validated': 0,
            'quality_issues': 0,
            'warnings': 0,
            'avg_quality_score': 0.0
        }

        total_score = 0.0

        for record in data:
            quality_result = self.quality_validator.validate_quality(data_type, record)

            if quality_result['is_valid']:
                validated_data.append(record)
                stats['total_validated'] += 1
            else:
                stats['quality_issues'] += 1
                logger.debug(f"数据质量问题，跳过记录: {record.get('symbol', 'unknown')}")

            if quality_result.get('warnings'):
                stats['warnings'] += len(quality_result['warnings'])

            total_score += quality_result.get('quality_score', 0)

        if validated_data:
            stats['avg_quality_score'] = total_score / len(data)

        return validated_data, stats

    def _standardize_data(self, data: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """数据标准化"""
        standardized = []

        for record in data:
            try:
                if data_type in ['stock_data', 'index_data']:
                    standardized_record = self._standardize_price_data(record)
                elif data_type == 'macro_data':
                    standardized_record = self._standardize_macro_data(record)
                elif data_type == 'news_data':
                    standardized_record = self._standardize_news_data(record)
                else:
                    standardized_record = record.copy()

                # 添加标准化时间戳
                standardized_record['standardized_at'] = datetime.now()
                standardized.append(standardized_record)

            except Exception as e:
                logger.warning(f"数据标准化失败: {e}, 记录: {record.get('symbol', 'unknown')}")
                continue

        return standardized

    def _standardize_price_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """标准化价格数据"""
        standardized = record.copy()

        # 确保价格字段为Decimal类型
        price_fields = ['open_price', 'high_price', 'low_price', 'close_price',
                       'volume', 'amount', 'pct_change', 'change',
                       'turnover_rate', 'amplitude']

        for field in price_fields:
            if field in standardized and standardized[field] is not None:
                try:
                    if isinstance(standardized[field], str):
                        # 移除可能的逗号分隔符
                        value_str = str(standardized[field]).replace(',', '')
                        standardized[field] = Decimal(value_str)
                    else:
                        standardized[field] = Decimal(str(standardized[field]))
                except Exception:
                    standardized[field] = None

        # 确保日期字段为date类型
        if 'date' in standardized:
            date_value = standardized['date']
            if isinstance(date_value, str):
                try:
                    standardized['date'] = datetime.strptime(date_value, '%Y-%m-%d').date()
                except ValueError:
                    try:
                        standardized['date'] = datetime.fromisoformat(date_value.split('T')[0]).date()
                    except Exception:
                        pass

        return standardized

    def _standardize_macro_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """标准化宏观数据"""
        standardized = record.copy()

        # 确保数值字段为Decimal类型
        if 'value' in standardized and standardized['value'] is not None:
            try:
                standardized['value'] = Decimal(str(standardized['value']))
            except Exception:
                standardized['value'] = None

        return standardized

    def _standardize_news_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """标准化新闻数据"""
        standardized = record.copy()

        # 确保日期时间字段正确
        if 'publish_time' in standardized and standardized['publish_time']:
            try:
                if isinstance(standardized['publish_time'], str):
                    standardized['publish_time'] = datetime.fromisoformat(standardized['publish_time'])
            except Exception:
                standardized['publish_time'] = None

        # 清理HTML标签（如果有）
        if 'content' in standardized and standardized['content']:
            import re
            standardized['content'] = re.sub(r'<[^>]+>', '', str(standardized['content']))

        return standardized

    def _resolve_data_conflicts(self, data: List[Dict[str, Any]], data_type: str,
                              merge_strategy: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """解决数据冲突"""
        # 按唯一键分组数据
        grouped_data = defaultdict(list)

        for record in data:
            unique_key = self._get_record_unique_key(record, data_type)
            grouped_data[unique_key].append(record)

        resolved_data = []
        conflict_stats = {
            'total_groups': len(grouped_data),
            'conflict_groups': 0,
            'resolved_conflicts': 0,
            'merge_operations': 0
        }

        for unique_key, records in grouped_data.items():
            if len(records) == 1:
                # 无冲突，直接添加
                resolved_data.append(records[0])
            else:
                # 有冲突，需要解决
                conflict_stats['conflict_groups'] += 1

                resolved_record = self.conflict_resolver.resolve_conflicts(
                    records, merge_strategy
                )

                if resolved_record:
                    resolved_data.append(resolved_record[0])
                    conflict_stats['resolved_conflicts'] += 1

                    if merge_strategy == 'merge_data':
                        conflict_stats['merge_operations'] += 1

        return resolved_data, conflict_stats

    def _get_record_unique_key(self, record: Dict[str, Any], data_type: str) -> Tuple:
        """获取记录的唯一键"""
        if data_type in ['stock_data', 'index_data']:
            return (
                record.get('source_id'),
                record.get('symbol'),
                record.get('date'),
                record.get('data_type', 'daily')
            )
        elif data_type == 'macro_data':
            return (
                record.get('source_id'),
                record.get('indicator_name'),
                record.get('date')
            )
        elif data_type == 'news_data':
            # 新闻数据使用标题哈希作为唯一键的一部分
            title_hash = hash(record.get('title', ''))
            return (record.get('source_id'), title_hash, record.get('publish_time'))
        else:
            # 默认唯一键
            return (
                record.get('source_id'),
                record.get('collected_at'),
                str(record.get('data', {}))
            )


class IncrementalDataProcessor:
    """增量数据处理器"""

    def __init__(self):
        self.merge_optimizer = DataMergeOptimizer()

    def process_incremental_data(self, source_id: str, incremental_data: List[Dict[str, Any]],
                               data_type: str, existing_data_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理增量数据

        Args:
            source_id: 数据源ID
            incremental_data: 增量采集的数据
            data_type: 数据类型
            existing_data_info: 现有数据信息（用于优化处理）

        Returns:
            处理结果
        """
        start_time = datetime.now()

        try:
            # 1. 数据合并优化
            merge_result = self.merge_optimizer.optimize_incremental_merge(
                source_id, incremental_data, data_type, 'latest_wins'
            )

            if not merge_result['success']:
                return merge_result

            # 2. 获取最终数据
            final_data = merge_result.get('data', [])

            # 3. 数据一致性检查（如果有现有数据信息）
            consistency_issues = []
            if existing_data_info:
                consistency_issues = self._check_data_consistency(final_data, existing_data_info, data_type)

            # 4. 生成处理报告
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'source_id': source_id,
                'data_type': data_type,
                'original_count': len(incremental_data),
                'processed_count': len(final_data),
                'processing_time': processing_time,
                'merge_stats': merge_result.get('merge_stats', {}),
                'consistency_issues': consistency_issues,
                'data_quality_score': merge_result.get('data_quality_score', 0),
                'data': final_data
            }

            logger.info(
                f"增量数据处理完成: {source_id}, "
                f"原始{len(incremental_data)} -> 处理后{len(final_data)}, "
                f"质量评分{merge_result.get('data_quality_score', 0):.1f}%, "
                f"耗时{processing_time:.2f}秒"
            )

            return result

        except Exception as e:
            error_msg = f"增量数据处理异常: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    def _check_data_consistency(self, new_data: List[Dict[str, Any]],
                              existing_info: Dict[str, Any], data_type: str) -> List[str]:
        """检查数据一致性"""
        issues = []

        try:
            if data_type in ['stock_data', 'index_data']:
                issues.extend(self._check_price_data_consistency(new_data, existing_info))
            elif data_type == 'macro_data':
                issues.extend(self._check_macro_data_consistency(new_data, existing_info))

        except Exception as e:
            issues.append(f"一致性检查异常: {str(e)}")

        return issues

    def _check_price_data_consistency(self, new_data: List[Dict[str, Any]],
                                    existing_info: Dict[str, Any]) -> List[str]:
        """检查价格数据一致性"""
        issues = []

        # 检查价格波动异常
        for record in new_data:
            close_price = record.get('close_price')
            if close_price:
                # 这里可以添加更复杂的波动性检查逻辑
                # 例如：与前一天收盘价比较，检查异常波动
                pass

        return issues

    def _check_macro_data_consistency(self, new_data: List[Dict[str, Any]],
                                    existing_info: Dict[str, Any]) -> List[str]:
        """检查宏观数据一致性"""
        issues = []

        # 检查数值合理性
        for record in new_data:
            value = record.get('value')
            indicator = record.get('indicator_name', '')

            if value and abs(float(value)) > 1000000000:  # 超过10亿的异常值
                issues.append(f"宏观指标 {indicator} 数值异常: {value}")

        return issues


# 全局实例
_merge_optimizer = None
_incremental_processor = None


def get_data_merge_optimizer() -> DataMergeOptimizer:
    """获取数据合并优化器实例"""
    global _merge_optimizer
    if _merge_optimizer is None:
        _merge_optimizer = DataMergeOptimizer()
    return _merge_optimizer


def get_incremental_data_processor() -> IncrementalDataProcessor:
    """获取增量数据处理器实例"""
    global _incremental_processor
    if _incremental_processor is None:
        _incremental_processor = IncrementalDataProcessor()
    return _incremental_processor