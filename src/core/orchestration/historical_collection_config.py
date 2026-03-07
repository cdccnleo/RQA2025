#!/usr/bin/env python3
"""
历史数据采集配置管理器

负责管理历史数据采集的配置规则，包括：
- 采集时间窗口配置
- 股票列表管理
- 数据类型配置
- 采集策略配置
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class CollectionTimeWindow:
    """采集时间窗口配置"""
    start_hour: int = 2  # 开始小时 (0-23)
    end_hour: int = 6    # 结束小时 (0-23)
    enable_weekend: bool = False  # 是否在周末采集
    timezone: str = "Asia/Shanghai"  # 时区

    def is_within_window(self, dt: Optional[datetime] = None) -> bool:
        """检查指定时间是否在采集窗口内"""
        if dt is None:
            dt = datetime.now()

        current_hour = dt.hour

        # 检查小时范围
        if not (self.start_hour <= current_hour < self.end_hour):
            return False

        # 检查周末限制
        if not self.enable_weekend and dt.weekday() >= 5:  # 周六=5，日=6
            return False

        return True


@dataclass
class CollectionRule:
    """采集规则配置"""
    name: str
    symbols: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=lambda: ['price', 'volume'])
    priority: str = 'normal'  # low, normal, high, urgent
    max_history_days: int = 365  # 最大历史天数
    min_interval_days: int = 7  # 最小采集间隔（天）
    enabled: bool = True
    last_collection: Optional[str] = None  # 最后采集时间

    def needs_collection(self, current_date: Optional[str] = None) -> bool:
        """检查是否需要采集"""
        if not self.enabled:
            return False

        if self.last_collection is None:
            return True

        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        # 计算距离上次采集的天数
        last_dt = datetime.strptime(self.last_collection.split(' ')[0], '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        days_since_last = (current_dt - last_dt).days

        return days_since_last >= self.min_interval_days


@dataclass
class HistoricalCollectionConfig:
    """历史数据采集全局配置"""
    # 时间窗口配置
    time_window: CollectionTimeWindow = field(default_factory=CollectionTimeWindow)

    # 采集控制
    max_daily_tasks: int = 50
    batch_size: int = 10
    check_interval_minutes: int = 10
    enabled: bool = True

    # 历史数据采集期配置（独立于日常采集）
    collection_period_type: str = 'quarterly'  # 'quarterly'(90天) 或 'annual'(365天)
    collection_period_days: int = 90  # 每段采集天数，用于分批；根据collection_period_type设置

    # 双轨边界：历史轨只采「日常采集周期之前」的数据，与日常轨不重叠
    daily_period_days: int = 30  # 日常采集周期（与数据源 default_days 对齐），历史 end = today - daily_period_days - 1
    max_history_days: int = 3650  # 历史最大回溯天数，如 10 年；历史 start = end - max_history_days

    # 分段采集配置（用于大量历史数据）
    enable_segmented_collection: bool = True
    segment_years: int = 1  # 每个分段的年数
    max_concurrent_segments: int = 3  # 最大并发分段数

    # 采集规则
    rules: List[CollectionRule] = field(default_factory=list)

    # 默认配置
    default_data_types: List[str] = field(default_factory=lambda: ['price', 'volume', 'fundamental'])
    default_priority: str = 'normal'

    def _get_collection_period_days(self, period_type: str) -> int:
        """根据采集期类型获取对应的天数"""
        if period_type == 'annual':
            return 365
        elif period_type == 'quarterly':
            return 90
        else:
            logger.warning(f"未知的采集期类型: {period_type}，使用默认季度采集")
            return 90

    def __post_init__(self):
        """初始化后处理"""
        # 确保collection_period_days与collection_period_type保持一致
        self.collection_period_days = self._get_collection_period_days(self.collection_period_type)


class HistoricalCollectionConfigManager:
    """历史数据采集配置管理器"""

    def __init__(self, config_file: str = None):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径，默认使用环境变量或默认路径
        """
        if config_file is None:
            # 默认配置文件路径
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "historical_collection_config.json"

        self.config_file = Path(config_file)
        self.config = self._load_config()
        self._initialize_default_rules()

        logger.info(f"历史数据采集配置管理器已初始化，配置文件: {self.config_file}")

    def _load_config(self) -> HistoricalCollectionConfig:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 反序列化配置
                config = HistoricalCollectionConfig()

                # 时间窗口
                if 'time_window' in data:
                    tw_data = data['time_window']
                    config.time_window = CollectionTimeWindow(
                        start_hour=tw_data.get('start_hour', 2),
                        end_hour=tw_data.get('end_hour', 6),
                        enable_weekend=tw_data.get('enable_weekend', False),
                        timezone=tw_data.get('timezone', 'Asia/Shanghai')
                    )

                # 基本配置
                config.max_daily_tasks = data.get('max_daily_tasks', 50)
                config.batch_size = data.get('batch_size', 10)
                config.check_interval_minutes = data.get('check_interval_minutes', 10)
                config.enabled = data.get('enabled', True)
                config.default_data_types = data.get('default_data_types', ['price', 'volume', 'fundamental'])
                config.default_priority = data.get('default_priority', 'normal')

                # 历史采集期配置
                config.collection_period_type = data.get('collection_period_type', 'quarterly')
                config.collection_period_days = config._get_collection_period_days(config.collection_period_type)
                config.daily_period_days = data.get('daily_period_days', 30)
                config.max_history_days = data.get('max_history_days', 3650)

                # 采集规则
                if 'rules' in data:
                    config.rules = []
                    for rule_data in data['rules']:
                        rule = CollectionRule(
                            name=rule_data['name'],
                            symbols=rule_data.get('symbols', []),
                            data_types=rule_data.get('data_types', config.default_data_types.copy()),
                            priority=rule_data.get('priority', config.default_priority),
                            max_history_days=rule_data.get('max_history_days', 365),
                            min_interval_days=rule_data.get('min_interval_days', 7),
                            enabled=rule_data.get('enabled', True),
                            last_collection=rule_data.get('last_collection')
                        )
                        config.rules.append(rule)

                logger.info(f"配置文件已加载: {len(config.rules)} 个采集规则")
                return config

            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")

        # 返回默认配置
        return HistoricalCollectionConfig()

    def _initialize_default_rules(self):
        """初始化默认采集规则

        注意：默认规则中不再包含硬编码的股票代码，
        而是通过数据源配置管理器动态获取活跃的股票代码
        """
        if not self.config.rules:
            # 创建基于数据源的动态规则
            try:
                # 尝试从数据源配置管理器获取活跃股票代码
                from src.gateway.web.data_source_config_manager import get_data_source_config_manager

                config_manager = get_data_source_config_manager()
                active_symbols = config_manager.get_active_symbols()

                if active_symbols:
                    # 创建基于活跃数据源的规则
                    default_rule = CollectionRule(
                        name="active_data_sources",
                        symbols=active_symbols[:50],  # 限制为前50个，避免过多
                        data_types=['price', 'volume', 'fundamental'],
                        priority='normal',
                        max_history_days=self.config.collection_period_days,  # 使用配置的采集期天数
                        min_interval_days=7
                    )

                    self.config.rules = [default_rule]
                    logger.info(f"已基于活跃数据源初始化采集规则，包含 {len(active_symbols)} 个股票代码")
                else:
                    # 如果无法获取活跃数据源，创建空的默认规则
                    empty_rule = CollectionRule(
                        name="empty_default",
                        symbols=[],
                        data_types=['price', 'volume', 'fundamental'],
                        priority='normal',
                        max_history_days=self.config.collection_period_days,  # 使用配置的采集期天数
                        min_interval_days=7
                    )

                    self.config.rules = [empty_rule]
                    logger.warning("无法获取活跃数据源，创建空的默认规则。请检查数据源配置。")

            except Exception as e:
                logger.error(f"初始化默认采集规则失败: {e}")

                # 创建空的备用规则
                fallback_rule = CollectionRule(
                    name="fallback_empty",
                    symbols=[],
                    data_types=['price', 'volume', 'fundamental'],
                    priority='normal',
                    max_history_days=self.config.collection_period_days,  # 使用配置的采集期天数
                    min_interval_days=7
                )

                self.config.rules = [fallback_rule]
                logger.warning("使用空的备用规则。请手动配置采集规则。")

    def save_config(self):
        """保存配置到文件"""
        try:
            # 序列化配置
            data = {
                'time_window': {
                    'start_hour': self.config.time_window.start_hour,
                    'end_hour': self.config.time_window.end_hour,
                    'enable_weekend': self.config.time_window.enable_weekend,
                    'timezone': self.config.time_window.timezone
                },
                'max_daily_tasks': self.config.max_daily_tasks,
                'batch_size': self.config.batch_size,
                'check_interval_minutes': self.config.check_interval_minutes,
                'enabled': self.config.enabled,
                'default_data_types': self.config.default_data_types,
                'default_priority': self.config.default_priority,
                'collection_period_type': self.config.collection_period_type,
                'collection_period_days': self.config.collection_period_days,
                'daily_period_days': self.config.daily_period_days,
                'max_history_days': self.config.max_history_days,
                'rules': [
                    {
                        'name': rule.name,
                        'symbols': rule.symbols,
                        'data_types': rule.data_types,
                        'priority': rule.priority,
                        'max_history_days': rule.max_history_days,
                        'min_interval_days': rule.min_interval_days,
                        'enabled': rule.enabled,
                        'last_collection': rule.last_collection
                    }
                    for rule in self.config.rules
                ]
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已保存到: {self.config_file}")

        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def get_active_rules(self) -> List[CollectionRule]:
        """获取活跃的采集规则"""
        return [rule for rule in self.config.rules if rule.enabled]

    def get_symbols_to_collect(self, refresh_from_data_sources: bool = False) -> List[str]:
        """获取需要采集的股票列表

        Args:
            refresh_from_data_sources: 是否从数据源配置重新刷新股票列表

        Returns:
            List[str]: 需要采集的股票代码列表
        """
        symbols = set()

        # 如果需要刷新，从数据源配置重新获取
        if refresh_from_data_sources:
            try:
                from src.gateway.web.data_source_config_manager import get_data_source_config_manager
                config_manager = get_data_source_config_manager()
                active_symbols = config_manager.get_active_symbols()

                if active_symbols:
                    # 更新规则中的股票代码
                    for rule in self.config.rules:
                        if rule.name == "active_data_sources":
                            rule.symbols = active_symbols[:100]  # 限制数量
                            logger.info(f"更新活跃数据源规则，包含 {len(active_symbols)} 个股票代码")

                    # 保存配置
                    self.save_config()

                symbols.update(active_symbols)
                logger.info(f"从数据源配置刷新获取到 {len(active_symbols)} 个股票代码")

            except Exception as e:
                logger.warning(f"从数据源配置刷新股票代码失败: {e}")

        # 从规则中收集股票代码
        for rule in self.get_active_rules():
            symbols.update(rule.symbols)

        # 去重并排序
        symbol_list = sorted(list(symbols))

        if not symbol_list:
            logger.warning("没有找到任何需要采集的股票代码")

        return symbol_list

    def update_rule_last_collection(self, rule_name: str, collection_time: Optional[str] = None):
        """更新规则的最后采集时间"""
        if collection_time is None:
            collection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for rule in self.config.rules:
            if rule.name == rule_name:
                rule.last_collection = collection_time
                logger.info(f"更新规则 {rule_name} 的最后采集时间: {collection_time}")
                break

    def add_rule(self, rule: CollectionRule):
        """添加新的采集规则"""
        # 检查是否已存在同名规则
        for existing_rule in self.config.rules:
            if existing_rule.name == rule.name:
                logger.warning(f"规则 {rule.name} 已存在，将被替换")
                self.config.rules.remove(existing_rule)
                break

        self.config.rules.append(rule)
        logger.info(f"添加采集规则: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """删除采集规则"""
        for rule in self.config.rules:
            if rule.name == rule_name:
                self.config.rules.remove(rule)
                logger.info(f"删除采集规则: {rule_name}")
                return True

        logger.warning(f"未找到采集规则: {rule_name}")
        return False

    def get_collection_schedule_info(self) -> Dict[str, Any]:
        """获取采集调度信息"""
        active_rules = self.get_active_rules()
        total_symbols = len(self.get_symbols_to_collect())

        return {
            'enabled': self.config.enabled,
            'time_window': {
                'start_hour': self.config.time_window.start_hour,
                'end_hour': self.config.time_window.end_hour,
                'enable_weekend': self.config.time_window.enable_weekend,
                'timezone': self.config.time_window.timezone
            },
            'limits': {
                'max_daily_tasks': self.config.max_daily_tasks,
                'batch_size': self.config.batch_size,
                'check_interval_minutes': self.config.check_interval_minutes
            },
            'historical': {
                'collection_period_type': self.config.collection_period_type,
                'collection_period_days': self.config.collection_period_days,
                'daily_period_days': getattr(self.config, 'daily_period_days', 30),
                'max_history_days': getattr(self.config, 'max_history_days', 3650)
            },
            'rules': {
                'total': len(self.config.rules),
                'active': len(active_rules),
                'total_symbols': total_symbols
            },
            'next_collection_check': self._calculate_next_check_time()
        }

    def _calculate_next_check_time(self) -> Optional[str]:
        """计算下次检查时间"""
        if not self.config.enabled:
            return None

        now = datetime.now()
        current_hour = now.hour

        # 如果当前在采集窗口内，下次检查时间为当前时间 + 检查间隔
        if self.config.time_window.is_within_window(now):
            next_check = now + timedelta(minutes=self.config.check_interval_minutes)
            return next_check.strftime('%Y-%m-%d %H:%M:%S')

        # 如果当前不在采集窗口内，计算下一个采集窗口的开始时间
        if current_hour < self.config.time_window.start_hour:
            # 今天还没到采集时间
            next_window = now.replace(hour=self.config.time_window.start_hour, minute=0, second=0, microsecond=0)
        else:
            # 今天采集时间已过，计算明天
            tomorrow = now + timedelta(days=1)
            next_window = tomorrow.replace(hour=self.config.time_window.start_hour, minute=0, second=0, microsecond=0)

        return next_window.strftime('%Y-%m-%d %H:%M:%S')

    def set_collection_period_type(self, period_type: str) -> bool:
        """
        设置采集期类型

        Args:
            period_type: 'quarterly' (季度, 90天) 或 'annual' (年度, 365天)

        Returns:
            bool: 设置是否成功
        """
        if period_type not in ['quarterly', 'annual']:
            logger.error(f"无效的采集期类型: {period_type}，支持: quarterly, annual")
            return False

        self.config.collection_period_type = period_type
        self.config.collection_period_days = self.config._get_collection_period_days(period_type)

        # 更新所有现有规则的max_history_days
        for rule in self.config.rules:
            rule.max_history_days = self.config.collection_period_days

        logger.info(f"采集期类型已设置为: {period_type} ({self.config.collection_period_days}天)")
        return True


# 全局配置管理器实例
_config_manager_instance: Optional[HistoricalCollectionConfigManager] = None


def get_historical_collection_config_manager(config_file: str = None) -> HistoricalCollectionConfigManager:
    """
    获取历史数据采集配置管理器实例（单例模式）

    Args:
        config_file: 配置文件路径

    Returns:
        HistoricalCollectionConfigManager: 配置管理器实例
    """
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = HistoricalCollectionConfigManager(config_file)
    return _config_manager_instance