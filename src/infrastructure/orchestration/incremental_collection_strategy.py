"""
增量采集策略控制器
实现不超过10天的增量采集时间窗口控制
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta, date
from dataclasses import dataclass

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class IncrementalCollectionConfig:
    """增量采集配置"""
    max_incremental_days: int = 10  # 最大增量采集天数
    complement_period_days: int = 90  # 补全周期（天）
    enable_history_complement: bool = True  # 是否启用历史补全
    collection_mode: str = 'incremental'  # 'incremental', 'complement', 'full'

    # 数据优先级配置
    priority_config: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.priority_config is None:
            self.priority_config = {
                'core_stocks': {
                    'max_incremental_days': 5,
                    'complement_period_days': 30,
                    'priority': 'critical'
                },
                'major_indices': {
                    'max_incremental_days': 7,
                    'complement_period_days': 7,
                    'priority': 'high'
                },
                'all_stocks': {
                    'max_incremental_days': 10,
                    'complement_period_days': 90,
                    'priority': 'medium'
                },
                'macro_data': {
                    'max_incremental_days': 30,
                    'complement_period_days': 180,
                    'priority': 'low'
                }
            }


@dataclass
class CollectionWindow:
    """采集时间窗口"""
    start_date: datetime
    end_date: datetime
    mode: str  # 'incremental', 'complement', 'full'
    priority: str
    description: str = ""


class IncrementalCollectionStrategy:
    """增量采集策略控制器"""

    def __init__(self, config: Optional[IncrementalCollectionConfig] = None):
        self.config = config or IncrementalCollectionConfig()
        self._last_collection_times: Dict[str, datetime] = {}
        self._collection_stats: Dict[str, Dict[str, Any]] = {}

    def determine_collection_strategy(self, source_id: str, data_type: str,
                                    requested_start: Optional[datetime] = None,
                                    requested_end: Optional[datetime] = None) -> CollectionWindow:
        """
        确定采集策略和时间窗口

        Args:
            source_id: 数据源ID
            data_type: 数据类型 ('stock', 'index', 'macro', 'news')
            requested_start: 请求的开始日期
            requested_end: 请求的结束日期

        Returns:
            CollectionWindow: 采集时间窗口
        """
        current_time = datetime.now()

        # 获取数据源的优先级配置
        priority_config = self._get_priority_config(data_type)

        # 获取最后采集时间
        last_collection = self._get_last_collection_time(source_id)

        # 确定采集模式
        collection_mode = self._determine_collection_mode(
            source_id, priority_config, last_collection, current_time
        )

        # 计算采集时间窗口
        window = self._calculate_collection_window(
            collection_mode, priority_config, last_collection, current_time,
            requested_start, requested_end
        )

        # 更新统计信息
        self._update_collection_stats(source_id, collection_mode, window)

        logger.info(
            f"数据源 {source_id} 采集策略确定: "
            f"模式={collection_mode}, 窗口={window.start_date.date()}至{window.end_date.date()}, "
            f"优先级={window.priority}"
        )

        return window

    def _get_priority_config(self, data_type: str) -> Dict[str, Any]:
        """获取数据类型的优先级配置"""
        # 根据数据类型映射到优先级配置
        type_mapping = {
            'stock': 'all_stocks',
            'index': 'major_indices',
            'macro': 'macro_data',
            'news': 'macro_data'  # 新闻数据使用宏观数据配置
        }

        config_key = type_mapping.get(data_type, 'all_stocks')
        return self.config.priority_config.get(config_key, self.config.priority_config['all_stocks'])

    def _get_last_collection_time(self, source_id: str) -> Optional[datetime]:
        """获取数据源最后采集时间"""
        # 这里可以从数据库或缓存中获取真实的最后采集时间
        # 暂时使用内存缓存，实际实现中应该从数据库查询
        return self._last_collection_times.get(source_id)

    def _determine_collection_mode(self, source_id: str, priority_config: Dict[str, Any],
                                 last_collection: Optional[datetime], current_time: datetime) -> str:
        """
        确定采集模式

        逻辑：
        1. 如果从未采集过 -> 增量模式（采集最近N天）
        2. 如果距离上次采集超过补全周期 -> 补全模式（采集历史数据）
        3. 否则 -> 增量模式（不超过最大增量天数）
        """

        if last_collection is None:
            # 首次采集，使用增量模式
            return 'incremental'

        time_since_last = current_time - last_collection
        complement_period = timedelta(days=priority_config['complement_period_days'])

        if time_since_last > complement_period:
            # 距离上次采集太久，需要补全历史数据
            return 'complement'
        else:
            # 正常增量采集
            return 'incremental'

    def _calculate_collection_window(self, mode: str, priority_config: Dict[str, Any],
                                   last_collection: Optional[datetime], current_time: datetime,
                                   requested_start: Optional[datetime],
                                   requested_end: Optional[datetime]) -> CollectionWindow:
        """计算采集时间窗口"""

        if mode == 'incremental':
            return self._calculate_incremental_window(priority_config, last_collection, current_time)
        elif mode == 'complement':
            return self._calculate_complement_window(priority_config, last_collection, current_time)
        elif mode == 'full':
            return self._calculate_full_window(requested_start, requested_end, current_time)
        else:
            # 默认使用增量模式
            return self._calculate_incremental_window(priority_config, last_collection, current_time)

    def _calculate_incremental_window(self, priority_config: Dict[str, Any],
                                    last_collection: Optional[datetime],
                                    current_time: datetime) -> CollectionWindow:
        """计算增量采集窗口（不超过最大天数限制）"""

        max_days = priority_config['max_incremental_days']

        if last_collection is None:
            # 首次采集：采集最近N天的数据
            start_date = current_time - timedelta(days=max_days)
            end_date = current_time
            description = f"首次采集，获取最近{max_days}天数据"
        else:
            # 增量采集：从上次采集时间开始，但不超过最大天数
            start_date = last_collection

            # 计算理论结束时间（当前时间）
            theoretical_end = current_time

            # 检查时间窗口是否超过最大限制
            window_days = (theoretical_end - start_date).days

            if window_days > max_days:
                # 超过限制，只采集最近N天
                start_date = theoretical_end - timedelta(days=max_days)
                description = f"增量采集超过{max_days}天限制，调整为最近{max_days}天"
            else:
                description = f"增量采集{window_days}天数据"

            end_date = theoretical_end

        return CollectionWindow(
            start_date=start_date,
            end_date=end_date,
            mode='incremental',
            priority=priority_config['priority'],
            description=description
        )

    def _calculate_complement_window(self, priority_config: Dict[str, Any],
                                   last_collection: Optional[datetime],
                                   current_time: datetime) -> CollectionWindow:
        """计算补全采集窗口"""

        complement_days = priority_config['complement_period_days']

        if last_collection is None:
            # 不应该进入补全模式
            start_date = current_time - timedelta(days=complement_days)
        else:
            # 补全模式：采集从上次采集到当前时间的数据
            start_date = last_collection

        end_date = current_time
        actual_days = (end_date - start_date).days

        return CollectionWindow(
            start_date=start_date,
            end_date=end_date,
            mode='complement',
            priority=priority_config['priority'],
            description=f"补全采集{actual_days}天历史数据"
        )

    def _calculate_full_window(self, requested_start: Optional[datetime],
                             requested_end: Optional[datetime],
                             current_time: datetime) -> CollectionWindow:
        """计算全量采集窗口"""

        start_date = requested_start or (current_time - timedelta(days=365))
        end_date = requested_end or current_time

        return CollectionWindow(
            start_date=start_date,
            end_date=end_date,
            mode='full',
            priority='low',
            description="全量数据采集"
        )

    def update_last_collection_time(self, source_id: str, collection_time: datetime):
        """更新最后采集时间"""
        self._last_collection_times[source_id] = collection_time

        # 在实际实现中，这里应该持久化到数据库
        logger.debug(f"更新数据源 {source_id} 最后采集时间: {collection_time}")

    def _update_collection_stats(self, source_id: str, mode: str, window: CollectionWindow):
        """更新采集统计信息"""
        if source_id not in self._collection_stats:
            self._collection_stats[source_id] = {
                'total_collections': 0,
                'modes_used': {},
                'last_window_days': 0,
                'avg_window_days': 0.0
            }

        stats = self._collection_stats[source_id]
        stats['total_collections'] += 1

        if mode not in stats['modes_used']:
            stats['modes_used'][mode] = 0
        stats['modes_used'][mode] += 1

        window_days = (window.end_date - window.start_date).days
        stats['last_window_days'] = window_days

        # 计算平均窗口天数
        total_days = stats['avg_window_days'] * (stats['total_collections'] - 1) + window_days
        stats['avg_window_days'] = total_days / stats['total_collections']

    def get_collection_stats(self, source_id: str) -> Dict[str, Any]:
        """获取采集统计信息"""
        return self._collection_stats.get(source_id, {})

    def reset_collection_stats(self, source_id: str):
        """重置采集统计信息"""
        if source_id in self._collection_stats:
            del self._collection_stats[source_id]
            logger.info(f"重置数据源 {source_id} 的采集统计信息")


class SmartMissingDataDetector:
    """智能缺失数据检测器"""

    def __init__(self):
        self.trading_days_cache: Dict[int, Set[date]] = {}  # 按年份缓存交易日

    def detect_missing_ranges(self, symbol: str, start_date: date, end_date: date,
                            existing_dates: Set[date], data_type: str = 'daily') -> List[Tuple[date, date]]:
        """
        检测缺失的数据日期范围

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            existing_dates: 已存在数据的日期集合
            data_type: 数据类型 ('daily', 'weekly', 'monthly')

        Returns:
            缺失日期范围列表 [(start, end), ...]
        """
        # 获取应该存在的日期集合
        expected_dates = self._get_expected_dates(start_date, end_date, data_type, symbol)

        # 计算缺失的日期
        missing_dates = expected_dates - existing_dates

        if not missing_dates:
            return []

        # 将缺失日期合并为连续的范围
        return self._merge_consecutive_dates(sorted(missing_dates))

    def _get_expected_dates(self, start_date: date, end_date: date,
                          data_type: str, symbol: str) -> Set[date]:
        """获取应该存在数据的日期集合"""

        if data_type == 'daily':
            # 对于日线数据，需要考虑交易日历
            return self._get_trading_days(start_date, end_date, symbol)
        elif data_type == 'weekly':
            # 周线数据：每周最后一个交易日
            return self._get_weekly_dates(start_date, end_date)
        elif data_type == 'monthly':
            # 月线数据：每月最后一个交易日
            return self._get_monthly_dates(start_date, end_date, symbol)
        else:
            # 其他类型：简单按日历日
            return self._get_calendar_days(start_date, end_date)

    def _get_trading_days(self, start_date: date, end_date: date, symbol: str) -> Set[date]:
        """获取交易日（简化为工作日，实际应该使用真实的交易日历）"""
        trading_days = set()

        current = start_date
        while current <= end_date:
            # 简化的交易日判断：周一到周五
            if current.weekday() < 5:  # 0-4: 周一到周五
                trading_days.add(current)
            current += timedelta(days=1)

        return trading_days

    def _get_weekly_dates(self, start_date: date, end_date: date) -> Set[date]:
        """获取周线日期（每周最后一个交易日）"""
        weekly_dates = set()
        current = start_date

        while current <= end_date:
            # 找到本周的周五（或最后一个交易日）
            week_end = current + timedelta(days=(4 - current.weekday()))  # 周五
            if week_end <= end_date:
                weekly_dates.add(week_end)
            current += timedelta(days=7)

        return weekly_dates

    def _get_monthly_dates(self, start_date: date, end_date: date, symbol: str) -> Set[date]:
        """获取月线日期（每月最后一个交易日）"""
        monthly_dates = set()
        current = start_date.replace(day=1)  # 月初

        while current <= end_date:
            # 下个月初减去1天得到本月最后一天
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)

            month_end = next_month - timedelta(days=1)

            if month_end >= start_date and month_end <= end_date:
                # 简化为月最后一天（实际应该使用交易日历找到最后一个交易日）
                monthly_dates.add(month_end)

            current = next_month

        return monthly_dates

    def _get_calendar_days(self, start_date: date, end_date: date) -> Set[date]:
        """获取日历日"""
        calendar_days = set()
        current = start_date

        while current <= end_date:
            calendar_days.add(current)
            current += timedelta(days=1)

        return calendar_days

    def _merge_consecutive_dates(self, dates: List[date]) -> List[Tuple[date, date]]:
        """将连续的日期合并为范围"""
        if not dates:
            return []

        ranges = []
        start = dates[0]
        prev = dates[0]

        for current in dates[1:]:
            if (current - prev).days > 1:
                # 不连续，结束当前范围
                ranges.append((start, prev))
                start = current
            prev = current

        # 添加最后一个范围
        ranges.append((start, prev))

        return ranges


# 全局实例
_incremental_strategy = None


def get_incremental_collection_strategy() -> IncrementalCollectionStrategy:
    """获取增量采集策略控制器实例"""
    global _incremental_strategy
    if _incremental_strategy is None:
        _incremental_strategy = IncrementalCollectionStrategy()
    return _incremental_strategy


def get_smart_missing_data_detector() -> SmartMissingDataDetector:
    """获取智能缺失数据检测器实例"""
    return SmartMissingDataDetector()