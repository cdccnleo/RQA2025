"""
数据优先级管理器
基于业务需求和市场状态管理数据采集优先级
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


@dataclass
class DataPriorityConfig:
    """数据优先级配置"""
    priority_level: str  # 'critical', 'high', 'medium', 'low'
    max_incremental_days: int  # 增量采集最大天数
    complement_period_days: int  # 补全周期（天）
    base_collection_frequency_days: int  # 基础采集频率（天）
    description: str = ""
    enabled: bool = True


class DataPriorityManager:
    """
    数据优先级管理器

    基于业务需求和数据重要性定义采集优先级：
    - 核心股票：沪深300、上证50等关键指数成分股
    - 主要指数：上证指数、深证成指等主要市场指数
    - 全市场股票：全部A股市场股票
    - 宏观数据：经济指标、政策数据等
    """

    def __init__(self):
        # 数据源优先级配置
        self.priority_configs = self._initialize_priority_configs()

        # 数据源类型映射
        self.data_source_mappings = self._initialize_data_source_mappings()

        logger.info("数据优先级管理器初始化完成")

    def _initialize_priority_configs(self) -> Dict[str, DataPriorityConfig]:
        """初始化优先级配置"""
        return {
            'core_stocks': DataPriorityConfig(
                priority_level='critical',
                max_incremental_days=5,  # 增量不超过5天
                complement_period_days=30,  # 每月补全
                base_collection_frequency_days=1,  # 每日采集
                description='核心交易股票，优先级最高',
                enabled=True
            ),
            'major_indices': DataPriorityConfig(
                priority_level='high',
                max_incremental_days=7,  # 增量不超过7天
                complement_period_days=7,  # 每周补全
                base_collection_frequency_days=1,  # 每日采集
                description='主要市场指数，策略重要参考',
                enabled=True
            ),
            'all_stocks': DataPriorityConfig(
                priority_level='medium',
                max_incremental_days=10,  # 增量不超过10天 ⭐业务需求
                complement_period_days=90,  # 季度补全
                base_collection_frequency_days=7,  # 每周采集
                description='全市场股票，数据完整性重要',
                enabled=True
            ),
            'macro_data': DataPriorityConfig(
                priority_level='low',
                max_incremental_days=30,  # 增量不超过30天
                complement_period_days=180,  # 半年补全
                base_collection_frequency_days=30,  # 每月采集
                description='宏观经济指标，辅助分析',
                enabled=True
            ),
            'news_data': DataPriorityConfig(
                priority_level='low',
                max_incremental_days=7,  # 新闻数据保留较短
                complement_period_days=30,  # 每月补全
                base_collection_frequency_days=1,  # 每日采集
                description='财经新闻数据，实时性要求高',
                enabled=True
            )
        }

    def _initialize_data_source_mappings(self) -> Dict[str, str]:
        """初始化数据源类型映射"""
        return {
            # 核心股票映射
            'core_stocks': 'core_stocks',
            'critical_stocks': 'core_stocks',
            'important_stocks': 'core_stocks',

            # 主要指数映射
            'major_indices': 'major_indices',
            'market_indices': 'major_indices',
            'benchmark_indices': 'major_indices',

            # 全市场股票映射
            'all_stocks': 'all_stocks',
            'market_stocks': 'all_stocks',
            'a_stocks': 'all_stocks',
            'stock_data': 'all_stocks',

            # 宏观数据映射
            'macro_data': 'macro_data',
            'economic_data': 'macro_data',
            'macroeconomic': 'macro_data',

            # 新闻数据映射
            'news_data': 'news_data',
            'financial_news': 'news_data',
            'market_news': 'news_data',
        }

    def get_data_priority(self, source_id: str, data_type: Optional[str] = None) -> DataPriorityConfig:
        """
        获取数据源的优先级配置

        Args:
            source_id: 数据源ID
            data_type: 数据类型（可选，用于辅助判断）

        Returns:
            DataPriorityConfig: 优先级配置
        """
        # 首先尝试通过source_id直接匹配
        priority_key = self._classify_source_by_id(source_id)

        # 如果无法通过ID确定，尝试通过数据类型辅助判断
        if priority_key == 'all_stocks' and data_type:
            priority_key = self._classify_source_by_type(data_type)

        # 获取配置，如果不存在返回默认配置
        config = self.priority_configs.get(priority_key)
        if not config:
            logger.warning(f"未找到优先级配置: {priority_key}，使用默认配置")
            config = self.priority_configs['all_stocks']

        logger.debug(f"数据源 {source_id} 优先级: {config.priority_level}")
        return config

    def _classify_source_by_id(self, source_id: str) -> str:
        """
        通过数据源ID分类优先级

        核心股票判断逻辑：
        - 上证50成分股
        - 沪深300成分股
        - 创业板指数成分股
        - 其他重要股票
        """
        source_id_lower = source_id.lower()

        # 核心股票判断（基于股票代码特征）
        if self._is_core_stock_code(source_id):
            return 'core_stocks'

        # 主要指数判断
        if self._is_major_index(source_id):
            return 'major_indices'

        # 宏观数据判断
        if self._is_macro_data(source_id):
            return 'macro_data'

        # 新闻数据判断
        if self._is_news_data(source_id):
            return 'news_data'

        # 默认全市场股票
        return 'all_stocks'

    def _is_core_stock_code(self, source_id: str) -> bool:
        """判断是否为核心股票代码"""
        # 上证50主要成分股代码
        sz50_codes = {
            '600000', '600036', '600519', '600276', '600887', '600000', '600016', '600028',
            '600030', '600031', '600036', '600048', '600050', '600104', '600196', '600276',
            '600309', '600340', '600346', '600352', '600362', '600383', '600390', '600398',
            '600406', '600436', '600438', '600519', '600547', '600570', '600583', '600585',
            '600588', '600606', '600637', '600690', '600703', '600732', '600745', '600754',
            '600795', '600803', '600809', '600837', '600887', '600893', '600900', '600909',
            '600919', '600926', '600928', '600958', '600989', '600999', '601006', '601088',
            '601166', '601211', '601288', '601318', '601319', '601328', '601336', '601360',
            '601377', '601390', '601398', '601600', '601601', '601628', '601633', '601668',
            '601669', '601688', '601698', '601727', '601766', '601800', '601818', '601857',
            '601866', '601872', '601877', '601878', '601881', '601888', '601898', '601899',
            '601901', '601916', '601918', '601919', '601933', '601939', '601949', '601952',
            '601958', '601965', '601969', '601975', '601985', '601988', '601989', '601992',
            '601995', '601998', '603019', '603156', '603160', '603259', '603260', '603288',
            '603369', '603501', '603658', '603799', '603806', '603833', '603899', '603986',
            '603993'
        }

        # 沪深300主要成分股（部分）
        hs300_codes = {
            '000001', '000002', '000063', '000069', '000100', '000157', '000166', '000301',
            '000338', '000402', '000408', '000425', '000538', '000568', '000596', '000617',
            '000625', '000627', '000629', '000630', '000651', '000661', '000671', '000703',
            '000708', '000723', '000725', '000728', '000738', '000750', '000768', '000776',
            '000783', '000786', '000800', '000807', '000829', '000830', '000831', '000858',
            '000876', '000883', '000895', '000898', '000938', '000961', '000963', '000977',
            '001979', '002007', '002008', '002024', '002027', '002032', '002044', '002049',
            '002050', '002064', '002081', '002085', '002120', '002142', '002146', '002152',
            '002157', '002179', '002202', '002230', '002236', '002241', '002252', '002271',
            '002294', '002304', '002310', '002352', '002371', '002410', '002414', '002415',
            '002422', '002424', '002426', '002450', '002456', '002460', '002463', '002466',
            '002468', '002475', '002493', '002508', '002555', '002558', '002572', '002594',
            '002600', '002601', '002602', '002607', '002624', '002625', '002648', '002709',
            '002714', '002736', '002739', '002773', '002812', '002821', '002841', '002916',
            '002920', '002938', '002939', '002945', '002958'
        }

        return source_id in sz50_codes or source_id in hs300_codes

    def _is_major_index(self, source_id: str) -> bool:
        """判断是否为主要指数"""
        major_indices = {
            'sh000001', 'sz399001', 'sz399006', 'sz399300', 'sh000300',  # 主要市场指数
            'sh000016', 'sh000905', 'sz399005', 'sz399678',  # 其他重要指数
        }

        # 指数代码通常以市场代码开头
        if source_id.lower() in major_indices:
            return True

        # 或者以特定模式命名
        if source_id.lower().startswith(('sh000', 'sz399', 'csi')):
            return True

        return False

    def _is_macro_data(self, source_id: str) -> bool:
        """判断是否为宏观数据"""
        macro_keywords = [
            'gdp', 'cpi', 'ppi', 'pmi', 'macro', 'economic', '央行', '利率',
            '外汇', '汇率', '贸易', '就业', '通胀', 'm2', '社融'
        ]

        source_lower = source_id.lower()
        return any(keyword in source_lower for keyword in macro_keywords)

    def _is_news_data(self, source_id: str) -> bool:
        """判断是否为新闻数据"""
        news_keywords = ['news', '新闻', '资讯', 'announcement', '公告']

        source_lower = source_id.lower()
        return any(keyword in source_lower for keyword in news_keywords)

    def _classify_source_by_type(self, data_type: str) -> str:
        """通过数据类型辅助分类"""
        type_mapping = {
            'index': 'major_indices',
            'macro': 'macro_data',
            'news': 'news_data',
            'stock': 'all_stocks',
        }

        return type_mapping.get(data_type.lower(), 'all_stocks')

    def get_all_priority_configs(self) -> Dict[str, DataPriorityConfig]:
        """获取所有优先级配置"""
        return self.priority_configs.copy()

    def update_priority_config(self, priority_key: str, config: DataPriorityConfig):
        """更新优先级配置"""
        if priority_key in self.priority_configs:
            self.priority_configs[priority_key] = config
            logger.info(f"更新优先级配置: {priority_key}")
        else:
            logger.warning(f"优先级配置不存在: {priority_key}")

    def get_priority_order(self) -> List[str]:
        """获取优先级排序"""
        priority_order = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3
        }

        sorted_configs = sorted(
            self.priority_configs.items(),
            key=lambda x: priority_order.get(x[1].priority_level, 99)
        )

        return [key for key, _ in sorted_configs]

    def get_sources_by_priority(self, priority_level: str) -> List[str]:
        """获取指定优先级的数据源列表"""
        # 这里返回该优先级对应的典型数据源示例
        # 实际使用时应该从配置或数据库中获取
        priority_sources = {
            'critical': ['core_stocks', '上证50', '沪深300'],
            'high': ['major_indices', '上证指数', '深证成指'],
            'medium': ['all_stocks', 'A股市场'],
            'low': ['macro_data', '宏观经济', '财经新闻']
        }

        return priority_sources.get(priority_level, [])

    def calculate_task_priority_score(self, source_id: str, data_type: Optional[str] = None) -> float:
        """
        计算任务优先级得分

        Returns:
            float: 优先级得分 (0-100)，越高优先级越低
        """
        config = self.get_data_priority(source_id, data_type)

        base_scores = {
            'critical': 10,
            'high': 30,
            'medium': 60,
            'low': 90
        }

        return base_scores.get(config.priority_level, 60)


# 全局实例
_priority_manager = None


def get_data_priority_manager() -> DataPriorityManager:
    """获取数据优先级管理器实例"""
    global _priority_manager
    if _priority_manager is None:
        _priority_manager = DataPriorityManager()
    return _priority_manager