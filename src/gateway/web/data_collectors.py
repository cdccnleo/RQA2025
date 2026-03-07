"""
数据采集模块
包含所有数据源的数据采集和处理逻辑
符合架构设计：使用统一适配器工厂访问数据层组件，使用EventBus进行事件通信
"""

import time
import logging
from typing import List, Dict, Any, Optional

# 导入统一的AKShare服务
from src.core.integration.akshare_service import get_akshare_service
# 导入BaoStock备用数据源服务
from src.core.integration.baostock_service import get_baostock_service

logger = logging.getLogger(__name__)

# 全局服务容器（延迟初始化，符合架构设计：使用ServiceContainer进行依赖管理）
_container = None

# 全局统一适配器工厂（延迟初始化，符合架构设计）
_adapter_factory = None
_data_adapter = None

# 全局事件总线（延迟初始化，符合架构设计）
_event_bus = None

# 全局缓存管理器和监控器（延迟初始化）
_cache_manager = None
_monitor = None

# AKShare函数映射配置
AKSHARE_FUNCTION_MAPPING = {
    # 分钟级数据 - 都使用 stock_zh_a_hist_min_em
    '1min': {'function': 'stock_zh_a_hist_min_em', 'period': '1', 'description': '1分钟K线'},
    '5min': {'function': 'stock_zh_a_hist_min_em', 'period': '5', 'description': '5分钟K线'},
    '15min': {'function': 'stock_zh_a_hist_min_em', 'period': '15', 'description': '15分钟K线'},
    '30min': {'function': 'stock_zh_a_hist_min_em', 'period': '30', 'description': '30分钟K线'},
    '60min': {'function': 'stock_zh_a_hist_min_em', 'period': '60', 'description': '60分钟K线'},

    # K线数据 - 使用 stock_zh_a_hist
    'daily': {'function': 'stock_zh_a_hist', 'period': 'daily', 'description': '日线数据'},
    'weekly': {'function': 'stock_zh_a_hist', 'period': 'weekly', 'description': '周线数据'},
    'monthly': {'function': 'stock_zh_a_hist', 'period': 'monthly', 'description': '月线数据'},

    # 实时数据 - 使用更可靠的 stock_zh_a_spot
    'realtime': {'function': 'stock_zh_a_spot', 'period': None, 'description': '实时行情'}
}


def get_core_universe_symbols(all_symbols: List[str], batch_size: int = 30) -> List[str]:
    """
    获取核心交易股票池：上证50 + 沪深300成分股

    Args:
        all_symbols: 所有可用的股票代码列表
        batch_size: 批次大小

    Returns:
        核心池股票代码列表
    """
    try:
        # 上证50 + 沪深300成分股权重股
        core_stocks = [
            # 上证50主要成分股
            '600000', '600036', '600519', '600276', '600887', '600000', '600016', '600028', '600030',
            '600031', '600036', '600048', '600050', '600104', '600196', '600276', '600309', '600340',
            '600346', '600352', '600362', '600383', '600390', '600398', '600406', '600436', '600438',
            '600519', '600547', '600570', '600583', '600585', '600588', '600606', '600637', '600690',
            '600703', '600732', '600745', '600754', '600795', '600803', '600809', '600837', '600887',
            '600893', '600900', '600909', '600919', '600926', '600928', '600958', '600989', '600999',
            '601006', '601088', '601166', '601211', '601288', '601318', '601319', '601328', '601336',
            '601360', '601377', '601390', '601398', '601600', '601601', '601628', '601633', '601668',
            '601669', '601688', '601698', '601727', '601766', '601800', '601818', '601857', '601866',
            '601872', '601877', '601878', '601881', '601888', '601898', '601899', '601901', '601916',
            '601918', '601919', '601933', '601939', '601949', '601952', '601958', '601965', '601966',
            '601969', '601975', '601985', '601988', '601989', '601992', '601995', '601998', '603019',
            '603156', '603160', '603259', '603260', '603288', '603369', '603501', '603658', '603799',
            '603806', '603833', '603899', '603986', '603993', '000001', '000002', '000063', '000069',
            '000100', '000157', '000166', '000301', '000338', '000402', '000408', '000425', '000538',
            '000568', '000596', '000617', '000625', '000627', '000629', '000630', '000651', '000661',
            '000671', '000703', '000708', '000723', '000725', '000728', '000738', '000750', '000768',
            '000776', '000783', '000786', '000800', '000807', '000829', '000830', '000831', '000858',
            '000876', '000883', '000895', '000898', '000938', '000961', '000963', '000977', '001979',
            '002007', '002008', '002024', '002027', '002032', '002044', '002049', '002050', '002064',
            '002081', '002085', '002120', '002142', '002146', '002152', '002157', '002179', '002202',
            '002230', '002236', '002241', '002252', '002271', '002294', '002304', '002310', '002352',
            '002371', '002410', '002414', '002415', '002422', '002424', '002426', '002450', '002456',
            '002460', '002463', '002466', '002468', '002475', '002493', '002508', '002555', '002558',
            '002572', '002594', '002600', '002601', '002602', '002607', '002624', '002625', '002648',
            '002709', '002714', '002736', '002739', '002773', '002812', '002821', '002841', '002916',
            '002920', '002938', '002939', '002945', '002958', '003816', '003833', '300003', '300014',
            '300015', '300017', '300024', '300027', '300033', '300058', '300070', '300072', '300122',
            '300124', '300136', '300142', '300144', '300207', '300223', '300274', '300308', '300316',
            '300347', '300408', '300413', '300415', '300433', '300450', '300454', '300496', '300498',
            '300529', '300601', '300628', '300661', '300750', '300751', '300759', '300760', '300782',
            '300832', '300896', '300919', '300957', '300979', '300999', '301269', '301279', '301317',
            '301319', '301338', '301358', '301369', '301380', '301391', '301421', '301488', '301489'
        ]

        # 过滤出实际存在的股票
        core_symbols = [s for s in core_stocks if s in all_symbols]

        # 如果核心股票数量不够，使用批次轮询补充
        if len(core_symbols) < batch_size:
            remaining = batch_size - len(core_symbols)
            additional_symbols = get_batch_symbols_for_collection(
                [s for s in all_symbols if s not in core_symbols],
                {"config": {"batch_size": remaining}}
            )
            core_symbols.extend(additional_symbols[:remaining])

        logger.info(f"核心股票池：{len(core_symbols)}只股票（上证50+沪深300成分股）")
        return core_symbols[:batch_size]

    except Exception as e:
        logger.warning(f"获取核心股票池失败: {e}，使用批次轮询")
        return get_batch_symbols_for_collection(all_symbols, {"config": {"batch_size": batch_size}})


def get_extended_universe_symbols(all_symbols: List[str], batch_size: int = 50) -> List[str]:
    """
    获取扩展交易股票池：中证500 + 创业板指成分股

    Args:
        all_symbols: 所有可用的股票代码列表
        batch_size: 批次大小

    Returns:
        扩展池股票代码列表
    """
    try:
        # 中证500 + 创业板主要成分股
        extended_stocks = [
            # 中证500代表性股票
            '000001', '000002', '000063', '000069', '000100', '000157', '000166', '000301', '000338',
            '000402', '000408', '000425', '000538', '000568', '000596', '000617', '000625', '000627',
            '000629', '000630', '000651', '000661', '000671', '000703', '000708', '000723', '000725',
            '000728', '000738', '000750', '000768', '000776', '000783', '000786', '000800', '000807',
            '000829', '000830', '000831', '000858', '000876', '000883', '000895', '000898', '000938',
            '000961', '000963', '000977', '001979', '002007', '002008', '002024', '002027', '002032',
            '002044', '002049', '002050', '002064', '002081', '002085', '002120', '002142', '002146',
            '002152', '002157', '002179', '002202', '002230', '002236', '002241', '002252', '002271',
            '002294', '002304', '002310', '002352', '002371', '002410', '002414', '002415', '002422',
            '002424', '002426', '002450', '002456', '002460', '002463', '002466', '002468', '002475',
            '002493', '002508', '002555', '002558', '002572', '002594', '002600', '002601', '002602',
            '002607', '002624', '002625', '002648', '002709', '002714', '002736', '002739', '002773',
            '002812', '002821', '002841', '002916', '002920', '002938', '002939', '002945', '002958',
            '003816', '003833', '300003', '300014', '300015', '300017', '300024', '300027', '300033',
            '300058', '300070', '300072', '300122', '300124', '300136', '300142', '300144', '300207',
            '300223', '300274', '300308', '300316', '300347', '300408', '300413', '300415', '300433',
            '300450', '300454', '300496', '300498', '300529', '300601', '300628', '300661', '300750',
            '300751', '300759', '300760', '300782', '300832', '300896', '300919', '300957', '300979',
            '300999', '301269', '301279', '301317', '301319', '301338', '301358', '301369', '301380',
            '301391', '301421', '301488', '301489'
        ]

        # 创业板股票（3开头）
        gem_stocks = [s for s in all_symbols if s.startswith('3')]

        # 合并扩展池
        extended_symbols = list(set(extended_stocks + gem_stocks[:200]))  # 限制创业板股票数量
        extended_symbols = [s for s in extended_symbols if s in all_symbols]

        # 如果扩展股票数量不够，使用批次轮询补充
        if len(extended_symbols) < batch_size:
            remaining = batch_size - len(extended_symbols)
            additional_symbols = get_batch_symbols_for_collection(
                [s for s in all_symbols if s not in extended_symbols],
                {"config": {"batch_size": remaining}}
            )
            extended_symbols.extend(additional_symbols[:remaining])

        logger.info(f"扩展股票池：{len(extended_symbols)}只股票（中证500+创业板）")
        return extended_symbols[:batch_size]

    except Exception as e:
        logger.warning(f"获取扩展股票池失败: {e}，使用批次轮询")
        return get_batch_symbols_for_collection(all_symbols, {"config": {"batch_size": batch_size}})


async def get_strategy_driven_symbols(all_symbols: List[str], strategy_config: Dict[str, Any], batch_size: int) -> List[str]:
    """
    根据策略配置选择股票

    Args:
        all_symbols: 所有可用的股票代码列表
        strategy_config: 策略配置
        batch_size: 批次大小

    Returns:
        策略选择的股票代码列表
    """
    try:
        strategy_type = strategy_config.get("strategy_id", "multi_factor")
        pool_size = min(strategy_config.get("pool_size", 100), len(all_symbols))

        if strategy_type == "hf_trading":
            # 高频交易：选择高流动性股票
            symbols = await select_high_liquidity_stocks(all_symbols, pool_size)
        elif strategy_type == "multi_factor":
            # 多因子：选择基本面优质股票
            symbols = await select_multi_factor_stocks(all_symbols, pool_size)
        elif strategy_type == "market_making":
            # 做市策略：选择中等波动性股票
            symbols = await select_market_making_stocks(all_symbols, pool_size)
        else:
            # 默认策略
            symbols = all_symbols[:pool_size]

        # 限制批次大小
        symbols = symbols[:batch_size]

        logger.info(f"策略驱动选择：{strategy_type}策略选择了 {len(symbols)} 只股票")
        return symbols

    except Exception as e:
        logger.warning(f"策略驱动选股失败: {e}，使用默认选择")
        return all_symbols[:batch_size]


async def select_high_liquidity_stocks(all_symbols: List[str], pool_size: int) -> List[str]:
    """选择高流动性股票（适合高频交易）"""
    try:
        # 获取实时行情数据（使用网络重试机制）
        from src.infrastructure.utils.network_utils import enhance_akshare_function
        import akshare as ak

        enhanced_stock_zh_a_spot = enhance_akshare_function(ak.stock_zh_a_spot)
        market_data = await enhanced_stock_zh_a_spot()

        # 筛选高流动性股票
        high_liquidity_stocks = []
        for _, stock in market_data.iterrows():
            try:
                code = str(stock['代码'])
                if code not in all_symbols:
                    continue

                # 成交额 > 50万/日
                turnover = float(str(stock.get('成交额(万)', '0')).replace('万', '')) * 10000
                if turnover > 500000:  # 50万
                    high_liquidity_stocks.append((code, turnover))
            except:
                continue

        # 按成交额排序
        high_liquidity_stocks.sort(key=lambda x: x[1], reverse=True)
        symbols = [code for code, _ in high_liquidity_stocks[:pool_size]]

        logger.info(f"高频交易策略：选择了 {len(symbols)} 只高流动性股票")
        return symbols

    except Exception as e:
        logger.warning(f"高频交易选股失败: {e}")
        return all_symbols[:pool_size]


async def select_multi_factor_stocks(all_symbols: List[str], pool_size: int) -> List[str]:
    """选择多因子模型股票（综合评分）"""
    try:
        # 获取实时行情数据（使用网络重试机制）
        from src.infrastructure.utils.network_utils import enhance_akshare_function
        import akshare as ak

        enhanced_stock_zh_a_spot = enhance_akshare_function(ak.stock_zh_a_spot)
        market_data = await enhanced_stock_zh_a_spot()

        # 计算综合评分
        scored_stocks = []
        for _, stock in market_data.iterrows():
            try:
                code = str(stock['代码'])
                if code not in all_symbols:
                    continue

                # 计算多因子评分（简化版）
                price = float(stock.get('最新价', 0))
                volume = float(str(stock.get('成交量(手)', '0')).replace('手', ''))
                change_pct = float(str(stock.get('涨跌幅', '0')).replace('%', ''))

                # 流动性因子
                liquidity_score = min(volume / 10000, 10)  # 成交量标准化

                # 波动性因子（适中波动为宜）
                volatility_score = max(0, 10 - abs(change_pct))

                # 价格因子（合理价格区间）
                price_score = 10 if 5 <= price <= 200 else 5

                # 综合评分
                total_score = (liquidity_score * 0.4 + volatility_score * 0.3 + price_score * 0.3)

                scored_stocks.append((code, total_score))

            except:
                continue

        # 按评分排序
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        symbols = [code for code, _ in scored_stocks[:pool_size]]

        logger.info(f"多因子策略：选择了 {len(symbols)} 只高评分股票")
        return symbols

    except Exception as e:
        logger.warning(f"多因子选股失败: {e}")
        return all_symbols[:pool_size]


async def select_market_making_stocks(all_symbols: List[str], pool_size: int) -> List[str]:
    """选择适合做市策略的股票"""
    try:
        # 获取实时行情数据
        import akshare as ak
        market_data = ak.stock_zh_a_spot()

        # 筛选做市股票：中等流动性，中等波动性
        market_making_stocks = []
        for _, stock in market_data.iterrows():
            try:
                code = str(stock['代码'])
                if code not in all_symbols:
                    continue

                price = float(stock.get('最新价', 0))
                turnover = float(str(stock.get('成交额(万)', '0')).replace('万', '')) * 10000
                change_pct = float(str(stock.get('涨跌幅', '0')).replace('%', ''))

                # 做市条件：中等成交额，中等价格，适中波动
                if (100000 <= turnover <= 1000000 and  # 10-100万成交额
                    10 <= price <= 100 and              # 10-100元价格
                    abs(change_pct) <= 5):               # 波动不超过5%

                    market_making_stocks.append((code, turnover))

            except:
                continue

        # 按成交额排序
        market_making_stocks.sort(key=lambda x: x[1], reverse=True)
        symbols = [code for code, _ in market_making_stocks[:pool_size]]

        logger.info(f"做市策略：选择了 {len(symbols)} 只适合做市的股票")
        return symbols

    except Exception as e:
        logger.warning(f"做市选股失败: {e}")
        return all_symbols[:pool_size]


def get_batch_symbols_for_collection(all_symbols: List[str], source_config: Dict[str, Any]) -> List[str]:
    """
    分批次轮询采集股票，避免随机选择造成数据缺失

    量化交易系统要求：确保所有股票都能被定期采集，不能有系统性缺失

    Args:
        all_symbols: 所有可用的股票代码列表
        source_config: 数据源配置

    Returns:
        本次应该采集的股票代码列表
    """
    try:
        # 批次大小：每次采集50只股票
        batch_size = 50

        # 如果股票总数不超过批次大小，直接返回所有股票
        if len(all_symbols) <= batch_size:
            logger.info(f"股票总数 {len(all_symbols)} 不超过批次大小 {batch_size}，采集所有股票")
            return all_symbols

        # 按重要性对股票进行排序（成分股优先）
        prioritized_symbols = prioritize_stocks(all_symbols)

        # 获取当前批次索引
        current_batch_index = get_current_batch_index(source_config['id'])

        # 计算本次采集的股票范围
        start_index = (current_batch_index * batch_size) % len(prioritized_symbols)
        end_index = min(start_index + batch_size, len(prioritized_symbols))

        # 获取本次采集的股票
        batch_symbols = prioritized_symbols[start_index:end_index]

        # 如果到达列表末尾，从头开始（处理环形队列）
        if end_index >= len(prioritized_symbols):
            remaining = batch_size - len(batch_symbols)
            if remaining > 0:
                batch_symbols.extend(prioritized_symbols[:remaining])

        # 更新批次索引
        update_batch_index(source_config['id'], (current_batch_index + 1) % (len(prioritized_symbols) // batch_size + 1))

        return batch_symbols

    except Exception as e:
        logger.error(f"分批次采集策略失败: {e}，降级为随机选择")
        # 降级策略：随机选择
        import random
        return random.sample(all_symbols, min(batch_size, len(all_symbols)))


async def collect_single_batch(symbols: List[str], data_types: List[str], source_config: Dict[str, Any],
                              start_date: str = None, end_date: str = None,
                              enable_incremental: bool = True, request_data: Dict[str, Any] = None,
                              existing_dates_by_type: Dict[str, Dict[str, set]] = None) -> Dict[str, Any]:
    """
    采集单个批次的股票数据

    Args:
        symbols: 股票代码列表
        data_types: 数据类型列表（如["1min", "5min", "daily"]）
        source_config: 数据源配置
        start_date: 开始日期
        end_date: 结束日期
        enable_incremental: 是否启用增量采集
        request_data: 请求数据
        existing_dates_by_type: 按数据类型分组的已存在日期数据

    Returns:
        批次采集结果
    """
    try:
        start_time = time.time()

        # 构建批次的请求数据
        batch_request_data = request_data.copy() if request_data else {}
        batch_request_data.update({
            "symbols": symbols,
            "data_types": data_types,
            "start_date": start_date,
            "end_date": end_date,
            "incremental": enable_incremental
        })

        # 调用实际的采集函数
        result = await collect_data_via_data_layer(source_config, batch_request_data, existing_dates_by_type)

        collection_time = time.time() - start_time
        logger.debug(f"批次采集完成，耗时: {collection_time:.2f}秒，数据点: {len(result.get('data', []))}")

        return result

    except Exception as e:
        logger.error(f"单个批次采集失败: {e}")
        return {"data": [], "error": str(e)}


def prioritize_stocks(symbols: List[str]) -> List[str]:
    """
    按重要性对股票进行排序

    优先级从高到低：
    1. 上证50成分股
    2. 沪深300成分股
    3. 中证500成分股
    4. 创业板股票
    5. 其他股票

    Args:
        symbols: 股票代码列表

    Returns:
        按优先级排序的股票列表
    """
    try:
        # 预定义的重要指数成分股（示例，实际应该从配置或API获取）
        # 这里简化处理，实际应该从数据源动态获取
        sz50_stocks = [
            '600000', '600036', '600519', '600276', '600887', '600000', '600016', '600028', '600030',
            '600031', '600036', '600048', '600050', '600104', '600196', '600276', '600309', '600340',
            '600346', '600352', '600362', '600383', '600390', '600398', '600406', '600436', '600438',
            '600519', '600547', '600570', '600583', '600585', '600588', '600606', '600637', '600690',
            '600703', '600732', '600745', '600754', '600795', '600803', '600809', '600837', '600887',
            '600893', '600900', '600909', '600919', '600926', '600928', '600958', '600989', '600999',
            '601006', '601088', '601166', '601211', '601288', '601318', '601319', '601328', '601336',
            '601360', '601377', '601390', '601398', '601600', '601601', '601628', '601633', '601668',
            '601669', '601688', '601698', '601727', '601766', '601800', '601818', '601857', '601866',
            '601872', '601877', '601878', '601881', '601888', '601898', '601899', '601901', '601916',
            '601918', '601919', '601933', '601939', '601949', '601952', '601958', '601965', '601966',
            '601969', '601975', '601985', '601988', '601989', '601992', '601995', '601998', '603019',
            '603156', '603160', '603259', '603260', '603288', '603369', '603501', '603658', '603799',
            '603806', '603833', '603899', '603986', '603993', '000001', '000002', '000063', '000069',
            '000100', '000157', '000166', '000301', '000338', '000402', '000408', '000425', '000538',
            '000568', '000596', '000617', '000625', '000627', '000629', '000630', '000651', '000661',
            '000671', '000703', '000708', '000723', '000725', '000728', '000738', '000750', '000768',
            '000776', '000783', '000786', '000800', '000807', '000829', '000830', '000831', '000858',
            '000876', '000883', '000895', '000898', '000938', '000961', '000963', '000977', '001979',
            '002007', '002008', '002024', '002027', '002032', '002044', '002049', '002050', '002064',
            '002081', '002085', '002120', '002142', '002146', '002152', '002157', '002179', '002202',
            '002230', '002236', '002241', '002252', '002271', '002294', '002304', '002310', '002352',
            '002371', '002410', '002414', '002415', '002422', '002424', '002426', '002450', '002456',
            '002460', '002463', '002466', '002468', '002475', '002493', '002508', '002555', '002558',
            '002572', '002594', '002600', '002601', '002602', '002607', '002624', '002625', '002648',
            '002709', '002714', '002736', '002739', '002773', '002812', '002821', '002841', '002916',
            '002920', '002938', '002939', '002945', '002958', '003816', '003833', '300003', '300014',
            '300015', '300017', '300024', '300027', '300033', '300058', '300070', '300072', '300122',
            '300124', '300136', '300142', '300144', '300207', '300223', '300274', '300308', '300316',
            '300347', '300408', '300413', '300415', '300433', '300450', '300454', '300496', '300498',
            '300529', '300601', '300628', '300661', '300750', '300751', '300759', '300760', '300782',
            '300832', '300896', '300919', '300957', '300979', '300999', '301269', '301279', '301317',
            '301319', '301338', '301358', '301369', '301380', '301391', '301421', '301488', '301489'
        ]

        # 沪深300成分股（简化版，实际应该动态获取）
        hs300_stocks = [
            '600000', '600036', '600519', '600276', '600887', '000001', '000002', '000063', '000069',
            '000100', '000157', '000166', '000301', '000338', '000402', '000408', '000425', '000538',
            '000568', '000596', '000617', '000625', '000627', '000629', '000630', '000651', '000661',
            '000671', '000703', '000708', '000723', '000725', '000728', '000738', '000750', '000768',
            '000776', '000783', '000786', '000800', '000807', '000829', '000830', '000831', '000858',
            '000876', '000883', '000895', '000898', '000938', '000961', '000963', '000977', '001979',
            '002007', '002008', '002024', '002027', '002032', '002044', '002049', '002050', '002064',
            '002081', '002085', '002120', '002142', '002146', '002152', '002157', '002179', '002202',
            '002230', '002236', '002241', '002252', '002271', '002294', '002304', '002310', '002352',
            '002371', '002410', '002414', '002415', '002422', '002424', '002426', '002450', '002456',
            '002460', '002463', '002466', '002468', '002475', '002493', '002508', '002555', '002558',
            '002572', '002594', '002600', '002601', '002602', '002607', '002624', '002625', '002648',
            '002709', '002714', '002736', '002739', '002773', '002812', '002821', '002841', '002916',
            '002920', '002938', '002939', '002945', '002958', '003816', '003833', '300003', '300014',
            '300015', '300017', '300024', '300027', '300033', '300058', '300070', '300072', '300122',
            '300124', '300136', '300142', '300144', '300207', '300223', '300274', '300308', '300316',
            '300347', '300408', '300413', '300415', '300433', '300450', '300454', '300496', '300498',
            '300529', '300601', '300628', '300661', '300750', '300751', '300759', '300760', '300782',
            '300832', '300896', '300919', '300957', '300979', '300999', '301269', '301279', '301317',
            '301319', '301338', '301358', '301369', '301380', '301391', '301421', '301488', '301489'
        ]

        # 创业板股票标识（以3开头）
        gem_stocks = [s for s in symbols if s.startswith('3')]

        # 按优先级分组
        high_priority = list(set(sz50_stocks) & set(symbols))  # 上证50
        medium_priority = list(set(hs300_stocks) & set(symbols))  # 沪深300（扣除上证50）
        low_priority = gem_stocks  # 创业板
        normal_priority = [s for s in symbols if s not in high_priority + medium_priority + low_priority]  # 其他

        # 合并排序结果
        prioritized = high_priority + medium_priority + low_priority + normal_priority

        logger.info(f"股票优先级排序完成: 高优先级{len(high_priority)}只, "
                   f"中优先级{len(medium_priority)}只, 创业板{len(low_priority)}只, "
                   f"普通{len(normal_priority)}只")

        return prioritized

    except Exception as e:
        logger.warning(f"股票优先级排序失败: {e}，使用原始顺序")
        return symbols


def get_current_batch_index(source_id: str) -> int:
    """
    获取当前批次索引

    Args:
        source_id: 数据源ID

    Returns:
        当前批次索引
    """
    try:
        # 从持久化存储获取批次索引
        # 这里简化实现，实际应该使用数据库或文件存储
        import os
        batch_file = f"/tmp/{source_id}_batch_index.txt"

        if os.path.exists(batch_file):
            with open(batch_file, 'r') as f:
                return int(f.read().strip())
        return 0
    except Exception:
        return 0


def update_batch_index(source_id: str, new_index: int):
    """
    更新批次索引

    Args:
        source_id: 数据源ID
        new_index: 新的批次索引
    """
    try:
        # 持久化批次索引
        import os
        batch_file = f"/tmp/{source_id}_batch_index.txt"
        os.makedirs(os.path.dirname(batch_file), exist_ok=True)

        with open(batch_file, 'w') as f:
            f.write(str(new_index))
    except Exception as e:
        logger.warning(f"更新批次索引失败: {e}")


def _get_container():
    """获取服务容器实例（单例模式，符合架构设计：使用ServiceContainer进行依赖管理）"""
    global _container
    if _container is None:
        try:
            from src.core.container.container import DependencyContainer
            _container = DependencyContainer()
            
            # 注册事件总线（符合架构设计：事件驱动通信）
            try:
                from src.core.event_bus.core import EventBus
                event_bus = EventBus()
                event_bus.initialize()
                _container.register(
                    "event_bus",
                    service=event_bus,
                    lifecycle="singleton"
                )
                logger.info("事件总线已注册到服务容器")
            except Exception as e:
                logger.warning(f"注册事件总线失败: {e}")
            
            # 注册业务流程编排器（符合架构设计：业务流程管理）
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
                _container.register(
                    "business_process_orchestrator",
                    factory=lambda: BusinessProcessOrchestrator(),
                    lifecycle="singleton"
                )
                logger.info("业务流程编排器已注册到服务容器")
            except Exception as e:
                logger.debug(f"注册业务流程编排器失败（可选）: {e}")
            
            # 注册统一适配器工厂（符合架构设计：统一基础设施集成）
            try:
                from src.core.integration.unified_business_adapters import get_unified_adapter_factory, BusinessLayerType
                adapter_factory = get_unified_adapter_factory()
                if adapter_factory:
                    _container.register(
                        "adapter_factory",
                        service=adapter_factory,
                        lifecycle="singleton"
                    )
                    logger.info("统一适配器工厂已注册到服务容器")
            except Exception as e:
                logger.debug(f"注册统一适配器工厂失败（可选）: {e}")
            
            logger.info("服务容器初始化成功")
        except Exception as e:
            logger.error(f"服务容器初始化失败: {e}")
            return None
    return _container


def _get_adapter_factory():
    """获取统一适配器工厂（符合架构设计：优先从服务容器获取）"""
    global _adapter_factory, _data_adapter
    
    # 优先从服务容器获取
    container = _get_container()
    if container:
        try:
            adapter_factory = container.resolve("adapter_factory")
            if adapter_factory:
                _adapter_factory = adapter_factory
                from src.core.integration.unified_business_adapters import BusinessLayerType
                _data_adapter = adapter_factory.get_adapter(BusinessLayerType.DATA)
                logger.debug("从服务容器获取统一适配器工厂")
                return _adapter_factory
        except Exception as e:
            logger.debug(f"从服务容器获取适配器工厂失败: {e}")
    
    # 降级方案：直接初始化
    if _adapter_factory is None:
        try:
            from src.core.integration.unified_business_adapters import get_unified_adapter_factory, BusinessLayerType
            _adapter_factory = get_unified_adapter_factory()
            if _adapter_factory:
                _data_adapter = _adapter_factory.get_adapter(BusinessLayerType.DATA)
                logger.info("数据层适配器已初始化")
        except Exception as e:
            logger.warning(f"统一适配器工厂初始化失败: {e}")
    return _adapter_factory


def _get_event_bus():
    """获取事件总线实例（符合架构设计：优先从服务容器获取）"""
    global _event_bus
    
    # 优先从服务容器获取
    container = _get_container()
    if container:
        try:
            event_bus = container.resolve("event_bus")
            if event_bus:
                _event_bus = event_bus
                logger.debug("从服务容器获取事件总线")
                return _event_bus
        except Exception as e:
            logger.debug(f"从服务容器获取事件总线失败: {e}")
    
    # 降级方案：直接初始化
    if _event_bus is None:
        try:
            from src.core.event_bus.core import EventBus
            _event_bus = EventBus()
            if not _event_bus._initialized:
                _event_bus.initialize()
            logger.info("事件总线已初始化")
        except Exception as e:
            logger.warning(f"事件总线初始化失败: {e}")
            _event_bus = None
    return _event_bus


def _get_orchestrator():
    """获取业务流程编排器实例（符合架构设计：从服务容器获取）"""
    container = _get_container()
    if container:
        try:
            orchestrator = container.resolve("business_process_orchestrator")
            return orchestrator
        except Exception as e:
            logger.debug(f"从服务容器获取业务流程编排器失败: {e}")
            return None
    return None


def _get_cache_manager():
    """获取AKShare缓存管理器（单例模式）"""
    global _cache_manager
    if _cache_manager is None:
        try:
            from src.core.cache.akshare_cache import get_akshare_cache_manager
            # 尝试从服务容器获取Redis客户端
            redis_client = None
            container = _get_container()
            if container:
                try:
                    redis_client = container.resolve("redis_client")
                except:
                    pass
            _cache_manager = get_akshare_cache_manager(redis_client=redis_client)
        except Exception as e:
            logger.warning(f"初始化AKShare缓存管理器失败: {e}")
            _cache_manager = None
    return _cache_manager


def _get_monitor():
    """获取数据采集监控器（单例模式）"""
    global _monitor
    if _monitor is None:
        try:
            from src.core.monitoring.data_collection_monitor import get_data_collection_monitor

            # 定义告警回调函数
            def alert_callback(alert):
                logger.warning(f"数据采集告警: {alert.alert_type.value} - {alert.message}")
                # 可以在这里添加更多的告警处理逻辑，比如发送邮件、短信等

            _monitor = get_data_collection_monitor(alert_callback=alert_callback)
        except Exception as e:
            logger.warning(f"初始化数据采集监控器失败: {e}")
            _monitor = None
    return _monitor


def get_akshare_function_config(data_type):
    """
    根据数据类型获取AKShare函数配置

    Args:
        data_type: 数据类型 ('1min', '5min', 'daily', 'weekly', 'monthly', 'realtime')

    Returns:
        dict: 包含function、period、description的配置字典，如果不支持返回None
    """
    return AKSHARE_FUNCTION_MAPPING.get(data_type)


def get_supported_data_types():
    """
    获取支持的所有数据类型

    Returns:
        list: 支持的数据类型列表
    """
    return list(AKSHARE_FUNCTION_MAPPING.keys())


async def collect_data_via_data_layer(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None, existing_dates_by_type: Dict[str, Dict[str, set]] = None) -> Dict[str, Any]:
    """通过数据层微服务采集数据（可选：使用业务流程编排器管理）"""
    import time
    start_time = time.time()

    # 确保existing_dates_by_type被正确初始化
    if existing_dates_by_type is None:
        existing_dates_by_type = {}

    try:
        print("🔥🔥🔥 collect_data_via_data_layer try块开始执行 🔥🔥🔥")
        print(f"🔥🔥🔥 输入source_config ID: {source_config.get('id', 'unknown')} 🔥🔥🔥")
        print(f"DEBUG: source_config keys: {list(source_config.keys()) if isinstance(source_config, dict) else 'not dict'}")
        source_id = source_config["id"]
        print(f"DEBUG: source_id = {source_id}")
        source_type = source_config.get("type", "")
        print(f"DEBUG: source_type = '{source_type}' (repr: {repr(source_type)})")
        source_url = source_config.get("url", "")

        logger.info(f"开始通过数据层采集数据源 {source_id} ({source_type})")
        logger.debug(f"数据源类型匹配检查: source_type='{source_type}', lower='{source_type.lower()}'")
        print(f"DEBUG: 处理数据源 {source_id}, 类型: {source_type} (lower: {source_type.lower()})")
        print(f"DEBUG: 检查财经新闻条件: {source_type.lower() in ['财经新闻', 'news', '新闻数据']}")

        # 获取服务实例（符合架构设计：使用ServiceContainer进行依赖管理）
        event_bus = _get_event_bus()
        orchestrator = _get_orchestrator()
        
        # 可选：使用BusinessProcessOrchestrator管理数据采集业务流程（符合架构设计）
        process_id = None
        if orchestrator:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState, ProcessConfig
                process_id = f"data_collection_{source_id}_{int(start_time)}"
                process_config = ProcessConfig(
                    process_id=process_id,
                    name=f"Data Collection: {source_id}",
                    initial_state=BusinessProcessState.DATA_COLLECTION,
                    parameters={
                        "source_id": source_id,
                        "source_type": source_type,
                        "source_config": source_config
                    }
                )
                orchestrator.start_process(process_config)
                logger.debug(f"已启动业务流程编排器流程: {process_id}")
            except Exception as e:
                logger.debug(f"启动业务流程编排器流程失败（可选功能）: {e}")
        
        # 可选：使用DataCollectionWorkflow管理数据采集业务流程（符合架构设计）
        workflow = None
        try:
            from src.core.orchestration.business_process.data_collection_orchestrator import DataCollectionWorkflow
            workflow = DataCollectionWorkflow(config={"max_retries": 3, "retry_delay": 60})
            logger.debug(f"已初始化数据采集业务流程编排器: {source_id}")
        except Exception as e:
            logger.debug(f"初始化数据采集业务流程编排器失败（可选功能）: {e}")

        # 发布数据采集开始事件（符合架构设计：事件驱动通信）
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.DATA_COLLECTION_STARTED,
                    {
                        "source_id": source_id,
                        "source_type": source_type,
                        "source_url": source_url,
                        "timestamp": time.time()
                    },
                    source="data_collectors"
                )
                logger.debug(f"已发布数据采集开始事件: {source_id}")
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")

        # 优先尝试使用统一适配器工厂获取数据层适配器（符合架构设计）
        data_adapter = _data_adapter if _get_adapter_factory() else None
        if data_adapter and hasattr(data_adapter, 'collect_data'):
            try:
                logger.info(f"尝试通过统一适配器采集数据: {source_id}")
                
                # 发布数据采集进度事件（符合架构设计：实时进度更新）
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        event_bus.publish(
                            EventType.DATA_COLLECTION_PROGRESS,
                            {
                                "source_id": source_id,
                                "source_type": source_type,
                                "progress": 10,
                                "status": "connecting",
                                "message": "正在连接数据源...",
                                "timestamp": time.time()
                            },
                            source="data_collectors"
                        )
                    except Exception as e:
                        logger.debug(f"发布进度事件失败: {e}")
                
                adapter_result = await data_adapter.collect_data(source_id, source_config)
                
                # 发布数据采集进度事件（数据获取完成）
                if event_bus:
                    try:
                        from src.core.event_bus.types import EventType
                        event_bus.publish(
                            EventType.DATA_COLLECTION_PROGRESS,
                            {
                                "source_id": source_id,
                                "source_type": source_type,
                                "progress": 50,
                                "status": "collecting",
                                "message": f"已获取 {len(adapter_result) if hasattr(adapter_result, '__len__') else 1} 条数据",
                                "data_points": len(adapter_result) if hasattr(adapter_result, "__len__") else 1,
                                "timestamp": time.time()
                            },
                            source="data_collectors"
                        )
                    except Exception as e:
                        logger.debug(f"发布进度事件失败: {e}")
                
                if adapter_result:
                    # 发布数据采集完成事件
                    if event_bus:
                        try:
                            from src.core.event_bus.types import EventType
                            event_bus.publish(
                                EventType.DATA_COLLECTED,
                                {
                                    "source_id": source_id,
                                    "source_type": source_type,
                                    "data_points": len(adapter_result) if hasattr(adapter_result, "__len__") else 1,
                                    "timestamp": time.time()
                                },
                                source="data_collectors"
                            )
                        except Exception as e:
                            logger.debug(f"发布事件失败: {e}")
                    
                    collection_time = time.time() - start_time
                    
                    # 发布数据采集进度事件（质量验证中）
                    if event_bus:
                        try:
                            from src.core.event_bus.types import EventType
                            event_bus.publish(
                                EventType.DATA_COLLECTION_PROGRESS,
                                {
                                    "source_id": source_id,
                                    "source_type": source_type,
                                    "progress": 75,
                                    "status": "validating",
                                    "message": "正在验证数据质量...",
                                    "timestamp": time.time()
                                },
                                source="data_collectors"
                            )
                        except Exception as e:
                            logger.debug(f"发布进度事件失败: {e}")
                    
                    quality_score = await validate_data_quality(adapter_result, source_type)
                    
                    # 发布数据采集进度事件（完成）
                    if event_bus:
                        try:
                            from src.core.event_bus.types import EventType
                            event_bus.publish(
                                EventType.DATA_COLLECTION_PROGRESS,
                                {
                                    "source_id": source_id,
                                    "source_type": source_type,
                                    "progress": 100,
                                    "status": "completed",
                                    "message": "数据采集完成",
                                    "data_points": len(adapter_result) if hasattr(adapter_result, "__len__") else 1,
                                    "quality_score": quality_score,
                                    "collection_time": collection_time,
                                    "timestamp": time.time()
                                },
                                source="data_collectors"
                            )
                        except Exception as e:
                            logger.debug(f"发布进度事件失败: {e}")
                    
                    # 注意：DataCollectionWorkflow已在上面调用，这里不再重复调用
                    
                    # 可选：使用BusinessProcessOrchestrator更新流程状态（符合架构设计）
                    if orchestrator and process_id:
                        try:
                            from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                            orchestrator.update_process_state(
                                process_id,
                                BusinessProcessState.COMPLETED,
                                metrics={
                                    "collection_time": collection_time,
                                    "data_points": len(adapter_result) if hasattr(adapter_result, "__len__") else 1,
                                    "quality_score": quality_score
                                }
                            )
                            logger.debug(f"已更新业务流程编排器流程状态: {process_id}")
                        except Exception as e:
                            logger.debug(f"更新业务流程编排器流程状态失败（可选功能）: {e}")
                    
                    return {
                        "data": adapter_result,
                        "metadata": {
                            "source_id": source_id,
                            "source_type": source_type,
                            "source_url": source_url,
                            "collection_timestamp": time.time(),
                            "data_points": len(adapter_result) if hasattr(adapter_result, "__len__") else 1
                        },
                        "collection_time": collection_time,
                        "quality_score": quality_score
                    }
            except Exception as e:
                logger.warning(f"通过统一适配器采集数据失败，使用降级方案: {e}")

        # 降级方案：根据数据源类型调用相应的适配器
        # 获取AKShare分类（如果配置了）
        ak_category = source_config.get("config", {}).get("akshare_category", "").lower()

        # 初始化批次完成状态
        collection_result = {"completed_all_batches": True, "batches_info": {"completed": 1, "total": 1}}

        if source_type.lower() in ["miniqmt", "交易接口"]:
            # 调用MiniQMT适配器
            data = await collect_from_miniqmt_adapter(source_config, request_data)
        elif source_type.lower() in ["股票数据", "stock", "akshare", "a股", "astock"]:
            # AKShare股票数据：根据分类进一步细分
            if ak_category in ["港股", "hk", "h股", "hongkong"]:
                data = await collect_from_akshare_hk_stock_adapter(source_config, request_data)
            elif ak_category in ["美股", "us", "nasdaq", "nyse", "america"]:
                data = await collect_from_akshare_us_stock_adapter(source_config, request_data)
            else:
                # 默认A股 - 现在返回字典格式
                result = await collect_from_akshare_adapter(source_config, request_data, existing_dates_by_type)
                # 从字典中提取数据和元信息
                data = result.get('data', []) if isinstance(result, dict) else result
                # 保存批次完成状态，用于传递给调度器
                if isinstance(result, dict):
                    collection_result = result  # 保存完整结果用于后续处理
        elif source_type.lower() in ["指数数据", "index", "指数"]:
            # AKShare指数数据
            data = await collect_from_akshare_index_adapter(source_config, request_data)
        elif source_type.lower() in ["基金数据", "fund", "基金"]:
            # AKShare基金数据
            data = await collect_from_akshare_fund_adapter(source_config, request_data)
        elif source_type.lower() in ["债券数据", "bond", "债券"]:
            # AKShare债券数据
            data = await collect_from_akshare_bond_adapter(source_config, request_data)
        elif source_type.lower() in ["期货数据", "futures", "期货", "期权", "options"]:
            # AKShare期货/期权数据
            data = await collect_from_akshare_futures_adapter(source_config, request_data)
        elif source_type.lower() in ["外汇数据", "forex", "外汇", "数字货币", "crypto"]:
            # AKShare外汇/数字货币数据
            data = await collect_from_akshare_forex_crypto_adapter(source_config, request_data)
        elif source_type.lower() in ["宏观经济", "macro", "宏观数据"]:
            # AKShare宏观经济数据（优先）或其他宏观数据适配器
            if "akshare" in source_id.lower() or ak_category:
                data = await collect_from_akshare_macro_adapter(source_config, request_data)
            else:
                data = await collect_from_macro_adapter(source_config, request_data)
        elif source_type.lower() in ["财经新闻", "news", "新闻数据"]:
            # AKShare新闻数据（优先）或其他新闻数据适配器
            print("🚨🚨🚨 执行财经新闻分支! 🚨🚨🚨")
            print("🎯🎯🎯 财经新闻分支开始执行! 🎯🎯🎯")
            print(f"🔍 source_id: '{source_id}', source_type: '{source_type}'")
            print(f"🔍 ak_category: '{ak_category}'")

            akshare_condition = "akshare" in source_id.lower()
            print(f"🔍 'akshare' in source_id.lower(): {akshare_condition}")

            if akshare_condition or ak_category:
                print("✅ 调用AKShare财经新闻采集器")
                print(f"   source_id: {source_id}, ak_category: {ak_category}")
                print(f"   条件检查: {akshare_condition} or {ak_category}")
                try:
                    print("🔄 开始调用collect_from_akshare_news_adapter...")
                    data = await collect_from_akshare_news_adapter(source_config, request_data)
                    print(f"✅ AKShare财经新闻采集器返回数据: {len(data) if data else 0} 条数据")
                except Exception as e:
                    print(f"❌ 调用财经新闻采集器异常: {e}")
                    import traceback
                    traceback.print_exc()
                    data = []
            else:
                print("📝 调用普通新闻采集器")
                data = await collect_from_news_adapter(source_config, request_data)
        elif source_type.lower() in ["另类数据", "alternative", "alternative_data", "社交媒体", "消费数据", "供应链数据", "环境数据"]:
            # AKShare另类数据（社交媒体、消费、供应链、环境等）
            if "akshare" in source_id.lower() or ak_category:
                data = await collect_from_akshare_alternative_adapter(source_config, request_data)
            else:
                # 其他另类数据适配器
                data = await collect_generic_data(source_config, request_data)
        elif source_type.lower() in ["市场指数", "crypto", "加密货币"]:
            # 加密货币数据（可能是AKShare或其他来源）
            if "akshare" in source_id.lower() or ak_category:
                data = await collect_from_akshare_forex_crypto_adapter(source_config, request_data)
            else:
                data = await collect_from_crypto_adapter(source_config, request_data)
        elif source_id.lower() == "akshare_stock_basic":
            # 股票基本信息采集
            logger.info(f"开始采集股票基本信息数据源 {source_id}")
            try:
                from src.core.integration.akshare_service import get_akshare_service
                akshare_service = get_akshare_service()
                
                # 获取股票基本信息
                stock_basic_df = await akshare_service.get_stock_basic_info()
                
                if stock_basic_df is not None and not stock_basic_df.empty:
                    # 将DataFrame转换为列表字典格式
                    data = stock_basic_df.to_dict('records')
                    logger.info(f"成功采集到 {len(data)} 条股票基本信息")
                else:
                    logger.warning("未采集到股票基本信息数据")
                    data = []
            except Exception as e:
                logger.error(f"采集股票基本信息失败: {e}")
                data = []
        else:
            # 默认处理
            print(f"DEBUG: 没有匹配任何类型条件，执行默认处理 - source_type: {source_type}")
            data = await collect_generic_data(source_config, request_data)

        collection_time = time.time() - start_time

        # 数据质量验证
        quality_score = await validate_data_quality(data, source_type)
        
        # 注意：DataCollectionWorkflow已在上面调用，这里不再重复调用
        
        # 可选：使用BusinessProcessOrchestrator更新流程状态（符合架构设计）
        if orchestrator and process_id:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                orchestrator.update_process_state(
                    process_id,
                    BusinessProcessState.COMPLETED,
                    metrics={
                        "collection_time": collection_time,
                        "data_points": len(data) if hasattr(data, "__len__") else 1,
                        "quality_score": quality_score
                    }
                )
                logger.debug(f"已更新业务流程编排器流程状态: {process_id}")
            except Exception as e:
                logger.debug(f"更新业务流程编排器流程状态失败（可选功能）: {e}")

        # 发布数据采集完成事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.DATA_COLLECTED,
                    {
                        "source_id": source_id,
                        "source_type": source_type,
                        "data_points": len(data) if hasattr(data, "__len__") else 1,
                        "collection_time": collection_time,
                        "quality_score": quality_score,
                        "timestamp": time.time()
                    },
                    source="data_collectors"
                )
                logger.debug(f"已发布数据采集完成事件: {source_id}")
            except Exception as e:
                logger.debug(f"发布事件失败: {e}")

        # 清理数据，确保JSON兼容性
        def sanitize_value(value):
            """清理数值，确保JSON兼容"""
            # 处理numpy类型（AKShare经常返回numpy类型）
            if str(type(value)).startswith("<class 'numpy"):
                try:
                    # 将numpy类型转换为Python基本类型
                    converted_value = value.item() if hasattr(value, 'item') else str(value)
                    # 如果是数值类型，进行有效性检查
                    if isinstance(converted_value, float):
                        if not (converted_value >= -1e308 and converted_value <= 1e308) or converted_value != converted_value:  # 检查inf或nan
                            return 0.0
                        return converted_value
                    elif isinstance(converted_value, int):
                        return converted_value
                    else:
                        return converted_value
                except:
                    return str(value)  # 如果转换失败，转为字符串

            if isinstance(value, float):
                if not (value >= -1e308 and value <= 1e308) or value != value:  # 检查inf或nan
                    return 0.0
                return value  # 有效的float直接返回
            elif isinstance(value, (int, str, bool, type(None))):
                return value
            elif hasattr(value, 'isoformat'):  # pandas Timestamp
                return value.isoformat()
            elif hasattr(value, 'strftime'):  # datetime对象
                if hasattr(value, 'hour'):  # 完整的datetime
                    return value.strftime('%Y-%m-%d %H:%M:%S')
                else:  # datetime.date
                    return value.strftime('%Y-%m-%d')
            elif str(type(value)).startswith("<class 'pandas"):  # 其他pandas类型
                return str(value)
            else:
                return str(value)  # 其他类型转换为字符串

        # 清理所有数据
        logger.info("开始数据清理和类型转换...")
        for i, item in enumerate(data):
            if i >= 3:  # 只记录前3条记录的详细日志
                break
            logger.debug(f"清理前记录{i+1}样例: open={item.get('open')}({type(item.get('open'))}), close={item.get('close')}({type(item.get('close'))})")

            for key, value in list(item.items()):
                original_value = value
                cleaned_value = sanitize_value(value)
                item[key] = cleaned_value

                # 记录数值字段的转换
                if key in ['open', 'close', 'high', 'low', 'volume', 'amount'] and original_value != cleaned_value:
                    logger.debug(f"字段{key}转换: {original_value}({type(original_value)}) -> {cleaned_value}({type(cleaned_value)})")

            logger.debug(f"清理后记录{i+1}样例: open={item.get('open')}({type(item.get('open'))}), close={item.get('close')}({type(item.get('close'))})")

        # 检查清理后的数据是否仍然包含问题
        for i, item in enumerate(data):
            if i >= 2:  # 只检查前2个项目
                break
            for key, value in item.items():
                if hasattr(value, 'isoformat') or hasattr(value, 'strftime'):
                    print(f"⚠️⚠️⚠️ 返回数据中仍包含时间对象: 项目{i}字段{key} - {type(value)} - 值: {value} ⚠️⚠️⚠️")

        return {
            "data": data,
            "metadata": {
                "source_id": source_id,
                "source_type": source_type,
                "source_url": source_url,
                "collection_timestamp": time.time(),
                "data_points": len(data) if hasattr(data, "__len__") else 1
            },
            "collection_time": collection_time,
            "quality_score": quality_score,
            "completed_all_batches": collection_result.get('completed_all_batches', True),
            "batches_info": collection_result.get('batches_info', {"completed": 1, "total": 1})
        }

    except Exception as e:
        collection_time = time.time() - start_time
        logger.error(f"数据层采集失败: {e}")
        
        # 可选：使用BusinessProcessOrchestrator更新流程状态为失败（符合架构设计）
        orchestrator = _get_orchestrator()
        if orchestrator and process_id:
            try:
                from src.core.orchestration.orchestrator_refactored import BusinessProcessState
                orchestrator.update_process_state(
                    process_id,
                    BusinessProcessState.FAILED,
                    metrics={
                        "collection_time": collection_time,
                        "error": str(e)
                    }
                )
                logger.debug(f"已更新业务流程编排器流程状态为失败: {process_id}")
            except Exception as e_orch:
                logger.debug(f"更新业务流程编排器流程状态失败（可选功能）: {e_orch}")
        
        # 发布数据采集失败事件（符合架构设计：事件驱动通信）
        event_bus = _get_event_bus()
        if event_bus:
            try:
                from src.core.event_bus.types import EventType
                event_bus.publish(
                    EventType.DATA_QUALITY_ALERT,
                    {
                        "source_id": source_id,
                        "source_type": source_type,
                        "error": str(e),
                        "timestamp": time.time()
                    },
                    source="data_collectors"
                )
                logger.debug(f"已发布数据采集失败事件: {source_id}")
            except Exception as e2:
                logger.debug(f"发布事件失败: {e2}")
        
        return {
            "data": [],
            "metadata": {"error": str(e)},
            "collection_time": collection_time,
            "quality_score": 0
        }


async def collect_from_miniqmt_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从MiniQMT适配器采集数据"""
    try:
        # 量化交易系统要求使用真实数据，不能使用模拟数据
        # TODO: 实现真实的MiniQMT适配器接口调用
        logger.warning("MiniQMT适配器尚未实现真实数据采集接口，返回空数据")
        logger.warning("量化交易系统要求使用真实数据，不能使用模拟数据")
        return []
    except Exception as e:
        logger.error(f"MiniQMT数据采集失败: {e}")
        return []


async def collect_from_news_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从新闻适配器采集数据"""
    print("🚨🚨🚨 collect_from_news_adapter 被调用了! 🚨🚨🚨")
    print(f"源ID: {source_config.get('id', 'unknown')}")
    try:
        # 量化交易系统要求使用真实数据，不能使用模拟数据
        # TODO: 实现真实的新闻数据适配器接口调用
        logger.warning("新闻数据适配器尚未实现真实数据采集接口，返回空数据")
        logger.warning("量化交易系统要求使用真实数据，不能使用模拟数据")
        return []
    except Exception as e:
        logger.error(f"新闻数据采集失败: {e}")
        return []


async def collect_from_macro_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从宏观数据适配器采集数据"""
    try:
        # 量化交易系统要求使用真实数据，不能使用模拟数据
        # TODO: 实现真实的宏观数据适配器接口调用
        logger.warning("宏观数据适配器尚未实现真实数据采集接口，返回空数据")
        logger.warning("量化交易系统要求使用真实数据，不能使用模拟数据")
        return []
    except Exception as e:
        logger.error(f"宏观数据采集失败: {e}")
        return []


async def collect_from_crypto_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从加密货币适配器采集数据"""
    try:
        # 量化交易系统要求使用真实数据，不能使用模拟数据
        # TODO: 实现真实的加密货币适配器接口调用
        logger.warning("加密货币适配器尚未实现真实数据采集接口，返回空数据")
        logger.warning("量化交易系统要求使用真实数据，不能使用模拟数据")
        return []
    except Exception as e:
        logger.error(f"加密货币数据采集失败: {e}")
        return []


async def collect_from_akshare_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None, existing_dates_by_type: Dict[str, Dict[str, set]] = None) -> Dict[str, Any]:
    """从AKShare适配器采集A股数据（支持分层采集策略）"""
    try:
        # 记录采集开始时间（用于监控）
        collection_start_time = time.time()

        # 确保existing_dates_by_type被正确初始化
        if existing_dates_by_type is None:
            existing_dates_by_type = {}

        # 获取缓存管理器和监控器
        cache_manager = _get_cache_manager()
        monitor = _get_monitor()

        # 获取池优先级配置
        pool_config = source_config.get("config", {})
        pool_priority = pool_config.get("pool_priority", "medium")
        batch_size = pool_config.get("batch_size", 50)

        # 支持多数据类型采集 - 从配置中解析启用的数据类型（提前初始化）
        data_types = []

        # 优先从请求参数获取
        if request_data and "data_types" in request_data:
            requested_types = request_data["data_types"]
            if isinstance(requested_types, str):
                requested_types = [requested_types]
            elif isinstance(requested_types, list):
                pass
            else:
                requested_types = ["daily"]

            # 验证请求的数据类型是否支持
            supported_types = get_supported_data_types()
            data_types = [dt for dt in requested_types if dt in supported_types]

        # 从配置中获取启用的数据类型
        if not data_types and source_config.get("config") and "data_type_configs" in source_config["config"]:
            data_type_configs = source_config["config"]["data_type_configs"]

            # 确保data_type_configs是字典类型，避免字符串调用items()错误
            if isinstance(data_type_configs, dict):
                data_types = [dt for dt, config in data_type_configs.items() if config.get("enabled", False)]
                logger.debug(f"从data_type_configs解析到数据类型: {data_types}")
            else:
                logger.warning(f"data_type_configs不是字典类型: {type(data_type_configs)} = {data_type_configs}，跳过")
                # 如果是损坏的配置，尝试修复为空字典
                if isinstance(data_type_configs, str) and data_type_configs == "[object Object]":
                    logger.error("检测到配置序列化错误，重置为默认配置")
                    # 这里可以设置默认配置，但为了安全起见，我们跳过这个配置项

        # 回退到旧的data_types配置（向后兼容）
        if not data_types and source_config.get("config") and "data_types" in source_config["config"]:
            old_data_types = source_config["config"]["data_types"]
            if isinstance(old_data_types, str):
                old_data_types = [old_data_types]
            elif isinstance(old_data_types, list):
                pass
            else:
                old_data_types = ["daily"]

            # 验证旧配置的数据类型是否支持
            supported_types = get_supported_data_types()
            data_types = [dt for dt in old_data_types if dt in supported_types]

        # 最后的默认值
        if not data_types:
            data_types = ["daily"]  # 默认只采集日线数据

        # 延迟导入akshare
        try:
            import akshare as ak
            logger.info(f"AKShare版本: {ak.__version__}")

            # AKShare健康检查 - 尝试一个简单的API调用
            try:
                logger.debug("执行AKShare健康检查...")
                # 简单的健康检查：获取市场概况
                test_df = ak.stock_zh_a_spot_em()
                if test_df is None or test_df.empty:
                    logger.warning("AKShare健康检查失败：API返回空数据")
                else:
                    logger.info(f"AKShare健康检查通过，获取到 {len(test_df)} 条市场数据")
            except Exception as health_check_error:
                logger.warning(f"AKShare健康检查失败: {health_check_error}，但继续尝试数据采集")

        except ImportError:
            logger.error("akshare未安装，无法采集A股数据。请运行: pip install akshare")
            return {"success": False, "data": [], "error": "akshare未安装"}
        except Exception as e:
            logger.error(f"AKShare导入异常: {e}")
            return {"success": False, "data": [], "error": f"AKShare导入失败: {e}"}

        # 从请求参数或配置中获取采集参数
        symbols = []
        if request_data and "symbols" in request_data:
            symbols = request_data["symbols"]
        elif source_config.get("config") and "default_symbols" in source_config["config"]:
            symbols = source_config["config"]["default_symbols"]
        elif source_config.get("config") and "custom_stocks" in source_config["config"]:
            # 支持自定义股票池配置
            custom_stocks = source_config["config"]["custom_stocks"]
            # 从新的对象格式或旧的字符串格式中提取股票代码
            if custom_stocks and isinstance(custom_stocks, list):
                symbols = []
                for stock in custom_stocks:
                    if isinstance(stock, dict) and 'code' in stock:
                        symbols.append(stock['code'])
                    elif isinstance(stock, str):
                        symbols.append(stock)
            else:
                symbols = []

        # 确保symbols是列表类型
        if not isinstance(symbols, list):
            symbols = []

        logger.info(f"开始采集 {pool_priority} 优先级股票池数据，批次大小: {batch_size}，数据类型: {data_types}，股票数量: {len(symbols) if symbols else '待确定'}")

        # 辅助函数：验证和调整时间范围
        def validate_and_adjust_time_range(start_time, end_time, context_desc="采集时间范围"):
            """
            验证时间范围是否合理，量化交易系统专用

            Args:
                start_time: 开始时间 (datetime)
                end_time: 结束时间 (datetime)
                context_desc: 上下文描述，用于日志

            Returns:
                tuple: (adjusted_start_time, adjusted_end_time)
            """
            from datetime import datetime as dt_class
            now = dt_class.utcnow()

            # 计算时间差异
            start_time_diff = start_time - now
            end_time_diff = end_time - now
            start_days_diff = start_time_diff.days

            # 量化交易系统时间范围验证逻辑：
            # 1. 允许结束时间为未来时间（AKShare接口会处理）
            # 2. 如果开始时间晚于当前时间（无效采集范围），调整为当前时间往前默认天数
            if start_days_diff > 0:
                logger.warning(f"检测到无效{context_desc}: 开始时间{start_time.strftime('%Y-%m-%d')}晚于当前时间，"
                             f"调整为当前时间往前默认天数范围")

                # 获取数据源配置的默认天数（如果有的话），否则使用30天
                default_days = 30  # 默认30天
                try:
                    # 从配置中获取默认采集天数
                    if source_config and "config" in source_config:
                        config = source_config["config"]
                        if "default_collection_days" in config:
                            default_days = config["default_collection_days"]
                        elif "collection_days" in config:
                            default_days = config["collection_days"]
                except Exception as e:
                    logger.debug(f"获取默认采集天数失败，使用默认值30天: {e}")

                # 调整为当前时间往前默认天数
                adjusted_end = now.replace(hour=0, minute=0, second=0, microsecond=0)
                adjusted_start = adjusted_end - timedelta(days=default_days)

                logger.info(f"调整后的{context_desc}: {adjusted_start.strftime('%Y-%m-%d')} 到 {adjusted_end.strftime('%Y-%m-%d')} "
                          f"({default_days}天)")
                return adjusted_start, adjusted_end
            else:
                # 时间范围有效，保持原有设置
                logger.debug(f"{context_desc}有效: {start_time.strftime('%Y-%m-%d')} 到 {end_time.strftime('%Y-%m-%d')}")
                return start_time, end_time

        start_date = None
        end_date = None
        if request_data:
            start_date = request_data.get("start_date")
            end_date = request_data.get("end_date")

        # 检查是否启用增量采集（默认启用）
        enable_incremental = True
        if request_data and "incremental" in request_data:
            enable_incremental = request_data["incremental"]
        elif source_config.get("config") and "enable_incremental" in source_config["config"]:
            enable_incremental = source_config["config"]["enable_incremental"]

        # 记录启用的数据类型详细信息
        enabled_types_info = []
        for dt in data_types:
            func_config = get_akshare_function_config(dt)
            if func_config:
                enabled_types_info.append(f"{dt}({func_config['description']})")
            else:
                enabled_types_info.append(f"{dt}(不支持)")

        logger.info(f"将采集以下数据类型: {data_types}")
        logger.info(f"数据类型详情: {enabled_types_info}")

        # 根据股票池类型选择策略
        stock_pool_type = pool_config.get("stock_pool_type", "full_universe")
        stock_universe = stock_pool_type  # 设置统一的universe标识
        logger.info(f"初始股票池类型: {stock_pool_type}, symbols是否为空: {not symbols}")

        # 暂时跳过股票池选择逻辑，直接使用已有的symbols

        # 统一的日志输出
        logger.info(f"{pool_priority}优先级采集：本次采集 {len(symbols)} 只股票（{stock_universe}）")

        # 如果启用了分批次采集策略，检查是否需要分批处理
        if stock_pool_type in ["full_universe", "batch_fallback"] and len(symbols) > batch_size:
            # 计算总批次数
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            logger.info(f"分批次采集 [批次{0}]: 采集 {len(symbols)} 只股票，总批次: {total_batches}")

            # 在一次调度中完成多个批次的采集，避免频繁调度
            # 限制单次调度最多采集的批次数，避免超时
            max_batches_per_schedule = min(5, total_batches)  # 最多5个批次，或总批次数

            all_batch_data = []
            batch_count = 0
            current_batch_index = 0

            while current_batch_index < total_batches and batch_count < max_batches_per_schedule:
                try:
                    # 计算当前批次的范围
                    start_index = current_batch_index * batch_size
                    end_index = min(start_index + batch_size, len(symbols))

                    # 获取当前批次的股票
                    batch_symbols = symbols[start_index:end_index]
                    if not batch_symbols:
                        break

                    logger.info(f"执行批次 {current_batch_index + 1}/{total_batches}: 采集 {len(batch_symbols)} 只股票")

                    # 执行当前批次的采集
                    batch_data = await collect_single_batch(
                        batch_symbols, data_types, source_config, start_date, end_date,
                        enable_incremental, request_data
                    )

                    if batch_data and 'data' in batch_data:
                        all_batch_data.extend(batch_data['data'])

                    batch_count += 1
                    current_batch_index += 1

                except Exception as e:
                    logger.error(f"批次 {current_batch_index} 采集失败: {e}")
                    break

            # 返回合并后的所有批次数据
            completed_all_batches = (batch_count >= total_batches)
            if all_batch_data:
                return {
                    "success": True,
                    "data": all_batch_data,
                    "metadata": {
                        "source_id": source_config['id'],
                        "source_type": source_type,
                        "source_url": source_config.get('url', ''),
                        "collection_timestamp": time.time(),
                        "data_points": len(all_batch_data),
                        "batches_completed": batch_count,
                        "total_batches": total_batches
                    },
                    "collection_time": time.time() - start_time,
                    "quality_score": 0.95,  # 批次采集质量评分
                    "completed_all_batches": completed_all_batches,
                    "batches_info": {
                        "completed": batch_count,
                        "total": total_batches,
                        "symbols_collected": len(set(s.get('symbol', '') for s in all_batch_data if s))
                    }
                }
            else:
                # 如果没有数据，仍然返回成功的响应结构
                return {
                    "success": False,
                    "data": [],
                    "error": "分批次采集未获取到任何数据",
                    "completed_all_batches": False,
                    "batches_info": {
                        "completed": 0,
                        "total": total_batches
                    }
                }

            # 注意：每个股票池类型现在都有自己的错误处理逻辑

        from datetime import datetime, timedelta
        if not start_date or not end_date:
            # 修复：使用UTC时间确保时区一致性
            end_date_obj = datetime.utcnow()
            start_date_obj = end_date_obj - timedelta(days=30)
            start_date_dt = start_date_obj
            end_date_dt = end_date_obj

            # 检查采集时间范围是否合理（量化交易系统）
            start_date_dt, end_date_dt = validate_and_adjust_time_range(
                start_date_obj, end_date_obj, "自动生成采集时间范围"
            )

            # 记录实际计算的日期范围，用于调试
            logger.info(f"自动计算日期范围: {start_date_dt.strftime('%Y-%m-%d')} 到 {end_date_dt.strftime('%Y-%m-%d')}")
            logger.info(f"格式化后: {start_date_dt.strftime('%Y%m%d')} 到 {end_date_dt.strftime('%Y%m%d')}")
        else:
            # 转换日期格式 YYYY-MM-DD -> datetime
            if isinstance(start_date, str):
                if "-" in start_date:
                    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
                else:
                    start_date_dt = datetime.strptime(start_date, "%Y%m%d")
            else:
                start_date_dt = start_date

            if isinstance(end_date, str):
                if "-" in end_date:
                    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                else:
                    end_date_dt = datetime.strptime(end_date, "%Y%m%d")
            else:
                end_date_dt = end_date

            # 验证传入的时间范围是否合理（量化交易系统）
            start_date_dt, end_date_dt = validate_and_adjust_time_range(
                start_date_dt, end_date_dt, "传入采集时间范围"
            )

            # 记录最终确定的日期范围，用于调试
            logger.info(f"最终采集时间范围: {start_date_dt.strftime('%Y-%m-%d')} 到 {end_date_dt.strftime('%Y-%m-%d')}")
            logger.info(f"格式化后: {start_date_dt.strftime('%Y%m%d')} 到 {end_date_dt.strftime('%Y%m%d')}")

        # 增量采集：使用传入的已存在数据日期
        if enable_incremental:
            total_existing = sum(
                len(dates) for type_dates in existing_dates_by_type.values()
                for dates in type_dates.values()
            )
            if total_existing > 0:
                logger.info(f"增量采集模式：发现 {total_existing} 条已存在数据，将只采集缺失部分")

        collected_data = []

        # 验证symbols的有效性
        logger.info(f"准备采集前的symbols: {symbols}, 类型: {type(symbols)}")
        if not symbols or not isinstance(symbols, list) or len(symbols) == 0:
            logger.error(f"无效的股票代码列表: {symbols}, 类型: {type(symbols)}")
            return {"success": False, "data": [], "error": "股票代码列表为空或无效", "completed_all_batches": True}

        # 准备existing_dates_map用于增量采集
        existing_dates_map = {}
        if enable_incremental and existing_dates_by_type:
            for data_type, symbol_dates in existing_dates_by_type.items():
                if isinstance(symbol_dates, dict):
                    for symbol, dates in symbol_dates.items():
                        if symbol not in existing_dates_map:
                            existing_dates_map[symbol] = set()
                        if isinstance(dates, (set, list)):
                            existing_dates_map[symbol].update(dates)

        # 遍历股票代码采集数据
        for symbol in symbols:
            try:
                # 验证和标准化股票代码格式
                original_symbol = symbol
                symbol = str(symbol).strip()

                # AKShare股票代码格式验证（应该是不带市场后缀的6位数字）
                if not (len(symbol) == 6 and symbol.isdigit()):
                    logger.warning(f"股票代码格式异常: {original_symbol} -> {symbol}，跳过")
                    continue

                logger.debug(f"处理股票: {symbol} (原始: {original_symbol})")
                # 增量采集：计算该股票缺失的日期范围
                existing_dates = existing_dates_map.get(symbol, set())
                date_ranges_to_collect = []

                if enable_incremental and existing_dates:
                    # 计算缺失的日期范围
                    from .postgresql_persistence import calculate_missing_date_ranges
                    missing_ranges = calculate_missing_date_ranges(
                        start_date_dt,
                        end_date_dt,
                        existing_dates
                    )

                    if not missing_ranges:
                        logger.info(f"股票 {symbol} 在指定日期范围内数据已完整，跳过采集")
                        continue

                    date_ranges_to_collect = missing_ranges
                    logger.info(f"股票 {symbol} 需要采集 {len(missing_ranges)} 个缺失日期范围")
                else:
                    # 全量采集：使用整个日期范围
                    date_ranges_to_collect = [(start_date_dt, end_date_dt)]

                # 遍历每个日期范围进行采集
                symbol_data = []

                # 为每个数据类型采集数据
                for data_type in data_types:
                    logger.debug(f"为股票 {symbol} 采集 {data_type} 数据")

                    # 获取AKShare函数配置
                    func_config = get_akshare_function_config(data_type)
                    if not func_config:
                        logger.warning(f"不支持的数据类型: {data_type}")
                        continue

                    func_name = func_config['function']
                    period = func_config['period']
                    description = func_config['description']

                    logger.debug(f"使用 {func_name} 获取 {description} 数据")

                    for range_start, range_end in date_ranges_to_collect:
                        # 转换为AKShare需要的格式 YYYYMMDD
                        start_date_str = range_start.strftime("%Y%m%d")
                        end_date_str = range_end.strftime("%Y%m%d")

                        df = None

                        try:
                            # 优化重试机制 - 最多3次重试后切换到备用接口
                            import asyncio
                            max_retries = 3  # 最多3次重试，避免过度等待
                            retry_delay = 3  # 合理的初始延迟
                            timeout_seconds = 30  # 适当的超时设置

                            for attempt in range(max_retries):
                                try:
                                    df = None

                                    # 为stock_zh_a_hist接口预先定义full_symbol
                                    full_symbol = None
                                    if func_name == 'stock_zh_a_hist':
                                        # 日线、周线、月线数据 - 改用Sina数据源的stock_zh_a_daily接口
                                        # 需要转换股票代码格式：添加市场前缀
                                        market_prefix = "sh" if symbol.startswith("6") else "sz"
                                        full_symbol = f"{market_prefix}{symbol}"

                                    # 检查缓存（对历史数据类型有效）
                                    if data_type in ["daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"] and cache_manager:
                                        cache_key = f"akshare_{data_type}_{symbol}_{start_date_str}_{end_date_str}"
                                        cached_data = await cache_manager.get("stock_zh_a_hist", {
                                            "symbol": symbol,
                                            "period": period,
                                            "start_date": start_date_str,
                                            "end_date": end_date_str,
                                            "adjust": "qfq"
                                        })
                                        if cached_data is not None:
                                            logger.debug(f"缓存命中: {cache_key}")
                                            df = cached_data
                                        else:
                                            logger.debug(f"缓存未命中: {cache_key}")

                                    if df is None:  # 缓存未命中，需要调用API
                                        logger.debug(f"调用AKShare API (尝试 {attempt + 1}/{max_retries}): {func_name}")

                                        # 检查日期范围是否合理（不能超过当前日期）
                                        from datetime import datetime
                                        current_date = datetime.utcnow().strftime("%Y%m%d")
                                        
                                        # 智能日期处理：只调整超出当前日期的部分，不重置整个范围
                                        date_adjusted = False
                                        if start_date_str > current_date:
                                            logger.warning(f"开始日期 {start_date_str} 超过当前日期 {current_date}，自动调整为当前日期")
                                            start_date_str = current_date
                                            date_adjusted = True
                                        
                                        if end_date_str > current_date:
                                            logger.warning(f"结束日期 {end_date_str} 超过当前日期 {current_date}，自动调整为当前日期")
                                            end_date_str = current_date
                                            date_adjusted = True
                                        
                                        # 确保开始日期不大于结束日期
                                        if start_date_str > end_date_str:
                                            logger.warning(f"开始日期 {start_date_str} 大于结束日期 {end_date_str}，调整开始日期等于结束日期")
                                            start_date_str = end_date_str
                                            date_adjusted = True
                                        
                                        if date_adjusted:
                                            logger.info(f"最终调整后的日期范围: {start_date_str} - {end_date_str}")

                                        # 根据函数类型调用相应的AKShare API
                                        if func_name == 'stock_zh_a_spot_em':
                                            # 实时数据：获取全市场数据，后续过滤
                                            df = ak.stock_zh_a_spot_em(timeout=timeout_seconds)  # 添加超时设置
                                            logger.info(f"📡 实时数据API调用结果: {'成功' if df is not None and not df.empty else '失败'}")
                                            if df is not None and not df.empty:
                                                logger.info(f"📊 实时数据返回: {len(df)} 条记录，字段: {list(df.columns)}")
                                            else:
                                                logger.warning("⚠️ 实时数据API返回空结果")
                                        elif func_name == 'stock_zh_a_hist_min_em':
                                            # 分钟级数据 - 按照AKShare规范调用
                                            logger.info(f"📈 分钟线API调用: stock_zh_a_hist_min_em(symbol={symbol}, period={period}, start={start_date_str}, end={end_date_str})")
                                            df = ak.stock_zh_a_hist_min_em(
                                                symbol=symbol,
                                                period=period,
                                                start_date=start_date_str,
                                                end_date=end_date_str,
                                                adjust="",  # 1分钟数据强制不复权
                                                timeout=timeout_seconds  # 添加超时设置
                                            )
                                            logger.info(f"📊 分钟线API调用结果: {'成功' if df is not None and not df.empty else '失败'}")
                                            if df is not None and not df.empty:
                                                logger.info(f"📈 分钟线数据返回: {len(df)} 条记录，字段: {list(df.columns)}")
                                                # 检查关键价格字段
                                                price_fields = ['开盘', '收盘', '最高', '最低']
                                                existing_price_fields = [f for f in price_fields if f in df.columns]
                                                logger.info(f"💰 价格字段存在性: {existing_price_fields}")
                                            else:
                                                logger.warning(f"⚠️ 分钟线API返回空结果: symbol={symbol}, period={period}")
                                        elif func_name == 'stock_zh_a_hist':
                                            # 日线、周线、月线数据 - 优先使用stock_zh_a_daily接口（新浪财经）
                                            logger.info(f"📊 日线API调用: 优先使用stock_zh_a_daily接口（新浪财经）")

                                            # 记录API调用前的系统状态
                                            import psutil
                                            memory_usage = psutil.virtual_memory().percent
                                            logger.debug(f"🖥️  API调用前系统状态: 内存使用率 {memory_usage}%")

                                            try:
                                                # 记录API调用开始的详细信息
                                                logger.info(f"🚀 开始AKShare API调用 - 股票: {symbol}, 周期: {period}")
                                                logger.info(f"📅 时间范围: {start_date_str} 到 {end_date_str}")
                                                logger.info(f"⚙️  复权参数: qfq (前复权)")

                                                # 多数据源切换策略: AKShare → BaoStock → 新浪财经
                                                api_used = "stock_zh_a_hist"
                                                
                                                # 根据重试次数决定使用哪个接口
                                                if attempt == 0:  # 第1次尝试：AKShare原始接口
                                                    try:
                                                        logger.info(f"🔀 第{attempt + 1}次尝试使用AKShare接口: {symbol}")
                                                        df = ak.stock_zh_a_hist(
                                                            symbol=symbol,
                                                            period=period,
                                                            start_date=start_date_str,
                                                            end_date=end_date_str,
                                                            adjust="qfq",  # 前复权
                                                            timeout=timeout_seconds  # 添加超时设置
                                                        )
                                                    except Exception as primary_error:
                                                        logger.warning(f"⚠️ AKShare接口第{attempt + 1}次尝试失败: {primary_error}")
                                                        df = None
                                                elif attempt == 1:  # 第2次尝试：BaoStock备用数据源
                                                    logger.warning("⚠️ AKShare接口失败，切换到BaoStock备用数据源")
                                                    api_used = "baostock"
                                                    
                                                    try:
                                                        # 使用BaoStock服务获取数据
                                                        baostock_service = get_baostock_service()
                                                        if baostock_service.is_available:
                                                            # 格式化日期为 YYYY-MM-DD 格式
                                                            bs_start_date = f"{start_date_str[:4]}-{start_date_str[4:6]}-{start_date_str[6:]}"
                                                            bs_end_date = f"{end_date_str[:4]}-{end_date_str[4:6]}-{end_date_str[6:]}"
                                                            
                                                            logger.info(f"🔀 切换到BaoStock: {symbol}, {bs_start_date}~{bs_end_date}")
                                                            
                                                            # 调用BaoStock获取数据（同步调用异步方法）
                                                            import asyncio
                                                            loop = asyncio.get_event_loop()
                                                            df = loop.run_until_complete(baostock_service.get_stock_data(
                                                                symbol=symbol,
                                                                start_date=bs_start_date,
                                                                end_date=bs_end_date,
                                                                frequency="d",  # 日线
                                                                adjustflag="2"  # 前复权
                                                            ))
                                                            
                                                            if df is not None and not df.empty:
                                                                logger.info(f"✅ BaoStock返回数据: {len(df)} 条记录")
                                                        else:
                                                            logger.warning("⚠️ BaoStock服务不可用")
                                                            df = None
                                                    except Exception as baostock_error:
                                                        logger.warning(f"⚠️ BaoStock接口失败: {baostock_error}")
                                                        df = None
                                                else:  # 第3次尝试：新浪财经接口
                                                    logger.warning("⚠️ BaoStock接口也失败，切换到新浪财经接口")
                                                    
                                                    # 需要添加市场前缀：sh（上证）或 sz（深证）
                                                    market_prefix = "sh" if symbol.startswith("6") else "sz"
                                                    full_symbol = f"{market_prefix}{symbol}"
                                                    
                                                    logger.info(f"🔀 切换到新浪财经接口: {full_symbol}")
                                                    api_used = "stock_zh_a_daily"
                                                    
                                                    # 使用stock_zh_a_daily接口（新浪财经，更稳定）
                                                    # 注意：stock_zh_a_daily接口不支持timeout参数
                                                    try:
                                                        df = ak.stock_zh_a_daily(
                                                            symbol=full_symbol,
                                                            start_date=start_date_str,
                                                            end_date=end_date_str,
                                                            adjust="qfq"  # 前复权
                                                        )
                                                    except Exception as sina_error:
                                                        logger.error(f"❌ 新浪财经接口也失败: {sina_error}")
                                                        df = None
                                                
                                                logger.info(f"📊 日线API调用完成 (接口: {api_used}): {'成功' if df is not None else '返回None'}")

                                                if df is not None:
                                                    logger.info(f"📏 DataFrame状态: empty={df.empty}, shape={df.shape}")

                                                    if not df.empty:
                                                        logger.info(f"📈 数据记录数: {len(df)}")
                                                        logger.info(f"🔤 字段列表: {list(df.columns)}")

                                                        # 根据使用的接口确定字段映射
                                                        if api_used == "stock_zh_a_hist" or api_used == "baostock":
                                                            # stock_zh_a_hist 和 BaoStock 使用中文字段名
                                                            # BaoStock已在服务层做了字段映射
                                                            price_fields = ['开盘', '收盘', '最高', '最低']
                                                            extra_fields = ['涨跌幅', '换手率'] if api_used == "baostock" else ['振幅', '涨跌幅', '涨跌额', '换手率']
                                                            date_field = '日期'
                                                        else:
                                                            # stock_zh_a_daily接口使用英文字段名
                                                            price_fields = ['open', 'high', 'low', 'close']
                                                            extra_fields = []  # 新浪接口可能不包含这些字段
                                                            date_field = 'date'
                                                        
                                                        logger.info(f"🔍 字段检查 (接口: {api_used}):")
                                                        for field in price_fields + extra_fields:
                                                            if field in df.columns:
                                                                non_null = df[field].notna().sum()
                                                                total = len(df)
                                                                sample = df[field].dropna().head(1).tolist() if non_null > 0 else []
                                                                logger.info(f"   {field}: {non_null}/{total} 非空, 样例: {sample}")
                                                            else:
                                                                logger.warning(f"   ⚠️ {field}: 字段不存在")

                                                        # 检查日期字段
                                                        if date_field in df.columns:
                                                            logger.info(f"   {date_field}: 日期字段存在")
                                                        else:
                                                            logger.error(f"   ❌ {date_field}: 日期字段不存在")

                                                        # 统一字段映射（如果需要）
                                                        if api_used == "stock_zh_a_daily":
                                                            # 新浪接口字段映射到标准字段名
                                                            field_mapping = {
                                                                'date': '日期',
                                                                'open': '开盘', 
                                                                'high': '最高',
                                                                'low': '最低',
                                                                'close': '收盘',
                                                                'volume': '成交量'
                                                            }
                                                            
                                                            # 重命名字段
                                                            df = df.rename(columns=field_mapping)
                                                            logger.info(f"🔄 字段映射完成: {list(df.columns)}")

                                                        # 最终字段检查
                                                        final_price_fields = ['开盘', '收盘', '最高', '最低']
                                                        for field in final_price_fields:
                                                            if field in df.columns:
                                                                non_null = df[field].notna().sum()
                                                                total = len(df)
                                                                sample_value = df[field].iloc[0] if len(df) > 0 else 'N/A'
                                                                logger.info(f"   {field}: {non_null}/{total} 非空, 样例: {sample_value} (类型: {type(sample_value)})")
                                                            else:
                                                                logger.error(f"   ❌ {field}: 最终字段不存在")

                                                        # 检查是否有任何有效价格数据
                                                        has_any_price_data = False
                                                        for field in price_fields:
                                                            if field in df.columns and df[field].notna().any():
                                                                has_any_price_data = True
                                                                break

                                                        # 检查额外的计算字段
                                                        has_extra_fields = all(field in df.columns for field in extra_fields)

                                                        if has_any_price_data:
                                                            logger.info("✅ AKShare返回了有效价格数据")
                                                            if has_extra_fields:
                                                                logger.info("✅ AKShare返回了完整的计算字段（振幅、涨跌幅、涨跌额、换手率）")
                                                            else:
                                                                logger.warning("⚠️ AKShare缺少部分计算字段")
                                                                missing_extra = [f for f in extra_fields if f not in df.columns]
                                                                logger.warning(f"   缺失字段: {missing_extra}")
                                                        else:
                                                            logger.error("❌ AKShare返回的价格字段全部为空或无效")
                                                            logger.error("   这表明AKShare API调用结果异常")
                                                            logger.error(f"   股票代码: {symbol}")
                                                            logger.error(f"   时间范围: {start_date_str} 到 {end_date_str}")
                                                            logger.error("   可能原因: 股票代码不存在、日期范围无交易日、API服务异常")

                                                        # 显示前几行原始数据
                                                        if len(df) > 0:
                                                            logger.info("📋 原始数据样例:")
                                                            for i in range(min(2, len(df))):
                                                                row_data = df.iloc[i].to_dict()
                                                                price_part = {k: v for k, v in row_data.items() if k in price_fields + extra_fields}
                                                                logger.info(f"   行{i+1}: {price_part}")

                                                    else:
                                                        logger.warning(f"⚠️ AKShare返回空DataFrame: symbol={symbol}, period={period}")
                                                        logger.warning("   可能原因: 该股票在指定时间段内无交易数据")
                                                else:
                                                    logger.error(f"❌ AKShare API返回None: symbol={symbol}, period={period}")
                                                    logger.error("   这通常表示API调用失败或参数错误")

                                            except Exception as api_error:
                                                # 导入requests模块以检查连接错误
                                                import requests
                                                import traceback
                                                
                                                # 检查是否为连接错误
                                                is_connection_error = isinstance(api_error, (requests.exceptions.ConnectionError, ConnectionError))
                                                
                                                if is_connection_error:
                                                    logger.error(f"🔌 AKShare API连接异常: {api_error}")
                                                    logger.error(f"   股票: {symbol}, 周期: {period}, 时间范围: {start_date_str}-{end_date_str}")
                                                    
                                                    # 连接错误的具体原因分析
                                                    logger.info("🔍 连接错误原因分析:")
                                                    logger.info("   • AKShare服务器可能过载或维护中")
                                                    logger.info("   • 网络连接不稳定或超时")
                                                    logger.info("   • 请求频率过高触发限流")
                                                    logger.info("   • 服务器主动断开连接")
                                                    
                                                    # 只有最后一次尝试才记录完整堆栈跟踪
                                                    if attempt == max_retries - 1:
                                                        logger.error(f"   详细异常详情: {traceback.format_exc()}")
                                                        
                                                        # 连接错误的详细诊断
                                                        logger.error("🔧 连接异常诊断报告:")
                                                        logger.error(f"   • 错误类型: {type(api_error).__name__}")
                                                        logger.error(f"   • 重试次数: {attempt + 1}/{max_retries}")
                                                        logger.error(f"   • 超时设置: {timeout_seconds}秒")
                                                        logger.error(f"   • 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                        
                                                        # 建议的解决方案
                                                        logger.info("💡 建议解决方案:")
                                                        logger.info("   1. 等待服务器恢复（通常需要5-30分钟）")
                                                        logger.info("   2. 检查网络连接状态")
                                                        logger.info("   3. 降低请求频率")
                                                        logger.info("   4. 尝试使用备用数据源")
                                                else:
                                                    logger.error(f"💥 AKShare API调用异常: {api_error}")
                                                    logger.error(f"   股票: {symbol}, 周期: {period}, 时间范围: {start_date_str}-{end_date_str}")
                                                    
                                                    # 其他错误的原因分析
                                                    logger.info("🔍 可能的原因分析:")
                                                    logger.info(f"   • 股票代码 {symbol} 是否存在？")
                                                    logger.info(f"   • 日期范围 {start_date_str}-{end_date_str} 是否合理？")
                                                    logger.info(f"   • 该时间段是否有交易日？")
                                                    logger.info(f"   • AKShare服务是否可用？")
                                                    
                                                    # 只有最后一次尝试才记录完整堆栈跟踪
                                                    if attempt == max_retries - 1:
                                                        logger.error(f"   详细异常详情: {traceback.format_exc()}")
                                                        
                                                        # 其他错误的详细诊断
                                                        logger.error("🔧 异常诊断报告:")
                                                        logger.error(f"   • 错误类型: {type(api_error).__name__}")
                                                        logger.error(f"   • 重试次数: {attempt + 1}/{max_retries}")
                                                        logger.error(f"   • 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                                
                                                df = None
                                                logger.error(f"❌ API调用异常: {api_error}")
                                                logger.error(f"   错误详情: symbol={symbol}, period={period}, dates={start_date_str}-{end_date_str}")

                                    # 如果成功，跳出重试循环
                                    break

                                except Exception as retry_error:
                                    # 导入requests模块以检查连接错误
                                    import requests
                                    
                                    # 检查是否为连接错误
                                    is_connection_error = isinstance(retry_error, (requests.exceptions.ConnectionError, ConnectionError))
                                    
                                    if attempt < max_retries - 1:
                                        if is_connection_error:
                                            logger.warning(f"🔌 AKShare API连接失败 (尝试 {attempt + 1}/{max_retries}): {retry_error}，{retry_delay}秒后重试")
                                            logger.info(f"   • 建议: 检查网络连接，降低请求频率")
                                        else:
                                            logger.warning(f"AKShare API调用失败 (尝试 {attempt + 1}/{max_retries}): {retry_error}，{retry_delay}秒后重试")
                                        
                                        await asyncio.sleep(retry_delay)
                                        retry_delay *= 2  # 指数退避
                                    else:
                                        if is_connection_error:
                                            logger.error(f"🔌 AKShare API连接失败，已达到最大重试次数: {retry_error}")
                                            logger.error(f"   • 最终错误: 无法连接到AKShare服务器")
                                            logger.error(f"   • 建议: 检查网络状态，稍后重试")
                                        else:
                                            logger.error(f"AKShare API调用失败，已达到最大重试次数: {retry_error}")
                                            logger.error(f"   • 最终错误: API调用持续失败")
                                            logger.error(f"   • 建议: 检查参数有效性，联系技术支持")
                                        
                                        raise retry_error

                            # 数据后处理：过滤和时间戳添加
                            if data_type == "realtime" and df is not None and not df.empty:
                                # 实时数据：过滤出指定的股票
                                original_count = len(df)
                                df = df[df['代码'] == symbol]
                                filtered_count = len(df)
                                logger.debug(f"实时数据过滤: {original_count} -> {filtered_count} 条记录 (股票代码: {symbol})")

                                if filtered_count > 0:
                                    # 添加时间戳
                                    from datetime import datetime
                                    df['日期'] = datetime.now().strftime("%Y-%m-%d")
                                    df['时间'] = datetime.now().strftime("%H-%M-%S")
                                else:
                                    logger.warning(f"实时数据过滤后无数据: 股票 {symbol} 不在市场快照中")
                                    df = None  # 设置为空，避免后续处理
                            
                            # 备用数据源方案：当AKShare不可用时尝试其他数据源
                            if df is None and attempt == max_retries - 1:
                                logger.warning("🔄 尝试备用数据源方案...")
                                
                                # 方案1：尝试使用Sina财经API作为备用
                                try:
                                    logger.info("📡 尝试Sina财经备用数据源...")
                                    # 这里可以添加Sina财经API的调用逻辑
                                    # 暂时记录备用方案信息
                                    logger.info("💡 备用数据源方案已准备就绪，需要时启用")
                                except Exception as backup_error:
                                    logger.warning(f"备用数据源调用失败: {backup_error}")
                                    
                                # 方案2：使用本地缓存数据（如果有）
                                if cache_manager:
                                    try:
                                        cached_data = await cache_manager.get(
                                            "stock_zh_a_hist",
                                            {
                                                "symbol": symbol,
                                                "period": period,
                                                "start_date": start_date_str,
                                                "end_date": end_date_str,
                                                "adjust": "qfq"
                                            }
                                        )
                                        if cached_data:
                                            df = cached_data
                                            logger.info(f"✅ 使用缓存数据: {symbol} {period}")
                                    except Exception as cache_error:
                                        logger.debug(f"缓存查询失败: {cache_error}")

                            # 设置缓存（对历史数据类型）
                            if data_type in ["daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"] and cache_manager and df is not None and not df.empty:
                                try:
                                    await cache_manager.set(
                                        "stock_zh_a_hist",
                                        {
                                            "symbol": symbol,
                                            "period": period,
                                            "start_date": start_date_str,
                                            "end_date": end_date_str,
                                            "adjust": "qfq"
                                        },
                                        df,
                                        ttl=300  # 5分钟缓存
                                    )
                                    logger.debug(f"数据已缓存: stock_zh_a_hist for {symbol} ({data_type})")
                                except Exception as cache_error:
                                    logger.warning(f"缓存设置失败: {cache_error}")
                        except Exception as e:
                            logger.warning(f"采集股票 {symbol} 的 {data_type} 数据失败: {e}")
                            continue

                    if df is not None and not df.empty:
                        # 转换DataFrame为字典列表
                        records = df.to_dict('records')
                        logger.info(f"✅ 股票 {symbol} {data_type} 数据采集成功: {len(records)} 条记录")

                        # AKShare字段映射：将中文字段名转换为英文字段名
                        akshare_field_mapping = {
                            '日期': 'date',
                            '开盘': 'open',
                            '收盘': 'close',
                            '最高': 'high',
                            '最低': 'low',
                            '成交量': 'volume',
                            '成交额': 'amount',
                            '振幅': 'amplitude',
                            '涨跌幅': 'pct_change',
                            '涨跌额': 'change',
                            '换手率': 'turnover_rate',
                            '股票代码': 'stock_code',
                            '流通股本': 'outstanding_share'
                        }

                        # 应用字段映射
                        mapped_records = []
                        for record in records:
                            mapped_record = {}
                            for key, value in record.items():
                                # 使用映射表转换字段名，保持未映射的字段不变
                                english_key = akshare_field_mapping.get(key, key)
                                mapped_record[english_key] = value
                            mapped_records.append(mapped_record)

                        records = mapped_records
                        logger.info(f"✅ 字段映射完成，原始记录数: {len(mapped_records)}")

                        # 记录DataFrame到字典转换后的数据样例
                        if records and len(records) > 0:
                            sample_record = records[0]
                            logger.info(f"📊 DataFrame转换+映射样例: {sample_record}")
                            logger.info(f"📊 字段类型检查: {[(k, type(v).__name__) for k, v in sample_record.items()]}")

                            # 特别检查价格字段
                            price_fields = ['open', 'close', 'high', 'low']
                            price_info = [(f, sample_record.get(f), type(sample_record.get(f)).__name__) for f in price_fields if f in sample_record]
                            logger.info(f"💰 价格字段详情: {price_info}")

                        # 为每条记录添加股票代码和数据源信息
                        for record in records:
                            # 调试：查看原始记录结构
                            if len(records) <= 3:  # 只对少量数据进行详细调试
                                logger.debug(f"AKShare原始记录字段: {list(record.keys())}")
                                logger.debug(f"AKShare原始记录样例: {record}")

                            # 特别检查价格字段（映射后的英文字段名）
                            price_fields = ['open', 'close', 'high', 'low']
                            extra_fields = ['amplitude', 'pct_change', 'change', 'turnover_rate']
                            for field in price_fields + extra_fields:
                                if field in record:
                                    value = record[field]
                                    logger.debug(f"字段 {field}: {value} (类型: {type(value)})")
                                else:
                                    logger.warning(f"字段 {field} 不存在")

                            # 完整性检查
                            if len(records) <= 3:  # 只对前3条记录进行详细检查
                                logger.info(f"🔍 完整记录样例: {record}")
                                price_check = {f: record.get(f) for f in ['open', 'close', 'high', 'low']}
                                extra_check = {f: record.get(f) for f in ['amplitude', 'pct_change', 'change', 'turnover_rate']}
                                logger.info(f"💰 价格字段完整性: {price_check}")
                                logger.info(f"📊 计算字段完整性: {extra_check}")

                            # 数据质量检查和验证（针对映射后的字段）
                            data_quality_issues = []

                            # 检查关键字段是否存在和有效性
                            required_fields = ['date', 'open', 'close', 'high', 'low', 'volume']
                            for field in required_fields:
                                value = record.get(field)
                                if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
                                    data_quality_issues.append(f"{field}为空")

                            # 检查数值字段的合理性（基于AKShare规范，价格字段必定为有效数值）
                            price_fields = ['open', 'close', 'high', 'low']
                            for field in price_fields:
                                if record.get(field) is not None:
                                    try:
                                        price = float(record[field])
                                        if price <= 0:
                                            data_quality_issues.append(f"{field}价格异常: {price}")
                                        elif price > 100000:  # 合理的价格上限检查
                                            data_quality_issues.append(f"{field}价格过高: {price}")
                                    except (ValueError, TypeError):
                                        data_quality_issues.append(f"{field}价格格式错误: {record[field]}")

                            # 检查成交量合理性
                            if record.get('volume') is not None:
                                try:
                                    volume = int(record['volume'])
                                    if volume < 0:
                                        data_quality_issues.append(f"成交量异常: {volume}")
                                except (ValueError, TypeError):
                                    data_quality_issues.append(f"成交量格式错误: {record['volume']}")

                            # 检查计算字段
                            calc_fields = ['amplitude', 'pct_change', 'change', 'turnover_rate']
                            for field in calc_fields:
                                if field in record and record.get(field) is not None:
                                    try:
                                        val = float(record[field])
                                        logger.debug(f"计算字段 {field}: {val}")
                                    except (ValueError, TypeError):
                                        data_quality_issues.append(f"{field}格式错误: {record[field]}")

                            # 记录数据质量问题
                            if data_quality_issues:
                                logger.warning(f"股票 {symbol} {data_type} 数据质量问题: {', '.join(data_quality_issues)} - 记录: {record}")
                            else:
                                logger.debug(f"股票 {symbol} {data_type} 数据质量正常: open={record.get('open')}, close={record.get('close')}, pct_change={record.get('pct_change')}, turnover_rate={record.get('turnover_rate')}")

                            record.update({
                                "symbol": symbol,
                                "source_id": source_config.get("id"),
                                "source_type": "akshare_a_stock",
                                "data_type": data_type
                            })

                        symbol_data.extend(records)
                        logger.debug(f"股票 {symbol} {data_type} 数据已添加到结果集")
                    else:
                        logger.warning(f"⚠️ 股票 {symbol} {data_type} 数据为空或无数据 (日期范围: {start_date_str} 到 {end_date_str})")
                        if df is None:
                            logger.debug(f"DataFrame为None，可能API调用失败")
                        elif hasattr(df, 'empty') and df.empty:
                            logger.debug(f"DataFrame为空，可能该股票在指定时间段内无交易数据")
                        else:
                            logger.debug(f"数据状态未知: {type(df)}")

                collected_data.extend(symbol_data)

            except Exception as e:
                logger.error(f"采集股票 {symbol} 数据失败: {e}")
                continue

        # 处理datetime对象，确保JSON可序列化
        for record in collected_data:
            for key, value in record.items():
                if hasattr(value, 'isoformat'):  # pandas Timestamp对象
                    record[key] = value.isoformat()
                elif hasattr(value, 'strftime'):  # datetime对象
                    if hasattr(value, 'hour'):  # 完整的datetime对象
                        record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                    else:  # datetime.date对象
                        record[key] = value.strftime('%Y-%m-%d')
                elif str(type(value)).startswith("<class 'pandas"):  # 其他pandas类型
                    record[key] = str(value)
                elif str(type(value)).startswith("<class 'numpy"):  # numpy类型转换
                    try:
                        # 将numpy类型转换为Python基本类型
                        record[key] = value.item() if hasattr(value, 'item') else str(value)
                    except:
                        record[key] = str(value)  # 如果转换失败，转为字符串
                elif isinstance(value, (int, float)) and str(key).lower() in ['date', '日期'] and not str(key).lower() in ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']:  # 可能是时间戳，但排除价格字段
                    # 如果日期字段是数字，尝试转换为日期字符串
                    try:
                        import datetime
                        if value > 1e10:  # 秒级时间戳
                            date_obj = datetime.datetime.fromtimestamp(value)
                            record[key] = date_obj.strftime('%Y-%m-%d')
                        elif value > 1e8:  # 毫秒级时间戳
                            date_obj = datetime.datetime.fromtimestamp(value / 1000)
                            record[key] = date_obj.strftime('%Y-%m-%d')
                        else:
                            # 可能是日期数字，如20240101
                            date_str = str(int(value))
                            if len(date_str) == 8:  # YYYYMMDD格式
                                record[key] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    except:
                        pass  # 如果转换失败，保持原值

            # 数据质量最终检查
            if 'date' in record and record['date']:
                price_fields = ['open', 'close', 'high', 'low']
                calc_fields = ['amplitude', 'pct_change', 'change', 'turnover_rate']
                missing_prices = [f for f in price_fields if not record.get(f) or record.get(f) == '']
                missing_calcs = [f for f in calc_fields if record.get(f) is None]
                if missing_prices:
                    logger.warning(f"数据处理后价格字段缺失: date={record['date']}, 缺失字段: {missing_prices}")
                if missing_calcs:
                    logger.warning(f"数据处理后计算字段缺失: date={record['date']}, 缺失字段: {missing_calcs}")
                if not missing_prices and not missing_calcs:
                    logger.debug(f"数据处理后完整: date={record['date']}, open={record.get('open')}, close={record.get('close')}, pct_change={record.get('pct_change')}, turnover_rate={record.get('turnover_rate')}")

        # 数据质量汇总统计
        total_collected = len(collected_data)
        quality_stats = {
            'total_records': total_collected,
            'data_types': {},
            'quality_score': 0.0
        }

        # 按数据类型统计
        for record in collected_data:
            dt = record.get('data_type', 'unknown')
            quality_stats['data_types'][dt] = quality_stats['data_types'].get(dt, 0) + 1

        # 计算质量评分（基于记录数量和字段完整性）
        if total_collected > 0:
            # 检查每条记录的关键字段完整性
            complete_records = 0
            for record in collected_data:
                required_fields = ['date', 'open', 'close', 'volume']
                calc_fields = ['pct_change', 'change', 'turnover_rate', 'amplitude']
                # 检查基础字段和计算字段
                base_complete = all(record.get(f) for f in required_fields)
                calc_complete = any(record.get(f) is not None for f in calc_fields)  # 至少有一个计算字段
                if base_complete and calc_complete:
                    complete_records += 1

            quality_stats['quality_score'] = (complete_records / total_collected) * 100

        logger.info(f"采集质量汇总: 总记录数 {quality_stats['total_records']}, "
                   f"数据类型分布 {quality_stats['data_types']}, "
                   f"质量评分 {quality_stats['quality_score']:.1f}%")

        # 返回前最终数据检查
        if collected_data and len(collected_data) > 0:
            sample_return = collected_data[0]
            logger.info(f"📤 返回数据样例: {sample_return}")
            logger.info(f"📤 返回数据字段: {list(sample_return.keys())}")
            return_prices = {f: sample_return.get(f) for f in ['open', 'close', 'high', 'low']}
            logger.info(f"📤 返回价格字段: {return_prices}")

        # 记录成功的监控信息
        collection_time = time.time() - collection_start_time
        if monitor:
            monitor.record_collection_attempt(
                source_id=source_config.get('id', 'unknown'),
                success=True,
                collection_time=collection_time,
                record_count=len(collected_data)
            )
            # 更新数据质量评分（基于记录完整性检查，与日志显示的质量评分一致）
            quality_score = quality_stats['quality_score'] / 100.0  # 转换为0-1范围
            monitor.update_data_quality(source_config.get('id', 'unknown'), quality_score)

        # 详细的采集统计
        total_symbols = len(symbols)

        # 计算成功采集的股票数（有数据的股票）
        symbols_with_data = set()
        for record in collected_data:
            if isinstance(record, dict) and record.get('symbol'):
                symbols_with_data.add(record['symbol'])

        successful_symbols = len(symbols_with_data)
        failed_symbols = total_symbols - successful_symbols

        logger.info(f"AKShare A股数据采集完成统计:")
        logger.info(f"  📊 总股票数: {total_symbols}")
        logger.info(f"  ✅ 成功采集股票: {successful_symbols}")
        logger.info(f"  ❌ 失败股票: {failed_symbols}")
        logger.info(f"  📈 总记录数: {len(collected_data)}")
        logger.info(f"  ⏱️  采集耗时: {collection_time:.2f}秒")

        if collected_data:
            # 按数据类型统计
            data_types_in_result = {}
            for record in collected_data:
                if isinstance(record, dict) and 'data_type' in record:
                    dt = record['data_type']
                    data_types_in_result[dt] = data_types_in_result.get(dt, 0) + 1

            logger.info(f"  📋 数据类型分布: {data_types_in_result}")

        return collected_data

    except Exception as e:
        # 记录失败的监控信息
        collection_time = time.time() - collection_start_time
        if monitor:
            error_type = "network" if "connection" in str(e).lower() else "api_error"
            monitor.record_collection_attempt(
                source_id=source_config.get('id', 'unknown'),
                success=False,
                collection_time=collection_time,
                record_count=0,
                error_type=error_type
            )

        logger.error(f"AKShare A股数据采集异常: {e}")
        return []


async def collect_from_akshare_hk_stock_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集港股数据"""
    try:
        import akshare as ak
        logger.warning("港股数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare港股数据采集失败: {e}")
        return []


async def collect_from_akshare_us_stock_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集美股数据"""
    try:
        import akshare as ak
        logger.warning("美股数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare美股数据采集失败: {e}")
        return []


async def collect_from_akshare_index_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集指数数据"""
    try:
        import akshare as ak
        logger.warning("指数数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare指数数据采集失败: {e}")
        return []


async def collect_from_akshare_fund_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集基金数据"""
    try:
        import akshare as ak
        logger.warning("基金数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare基金数据采集失败: {e}")
        return []


async def collect_from_akshare_bond_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集债券数据"""
    try:
        import akshare as ak
        logger.warning("债券数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare债券数据采集失败: {e}")
        return []


async def collect_from_akshare_futures_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集期货/期权数据"""
    try:
        import akshare as ak
        logger.warning("期货/期权数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare期货/期权数据采集失败: {e}")
        return []


async def collect_from_akshare_forex_crypto_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集外汇/数字货币数据"""
    try:
        import akshare as ak
        logger.warning("外汇/数字货币数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare外汇/数字货币数据采集失败: {e}")
        return []


async def collect_from_akshare_macro_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集宏观经济数据"""
    try:
        import akshare as ak
        logger.warning("宏观经济数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare宏观经济数据采集失败: {e}")
        return []


async def collect_from_akshare_news_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集新闻数据"""
    print("🎯🎯🎯 collect_from_akshare_news_adapter 函数开始执行! 🎯🎯🎯")
    try:
        import akshare as ak
        import asyncio
        import pandas as pd
        from datetime import datetime

        source_id = source_config.get("id", "")
        config = source_config.get("config", {})
        akshare_function = config.get("akshare_function", "")

        print(f"🎯 源ID: {source_id}")
        print(f"🎯 配置: {config}")
        print(f"🎯 AKShare函数: {akshare_function}")

        if not akshare_function:
            logger.error(f"数据源 {source_id} 缺少 akshare_function 配置")
            print("❌ 缺少 akshare_function 配置")
            return []

        logger.info(f"开始采集AKShare新闻数据: {source_id} 使用函数 {akshare_function}")
        print(f"✅ 开始采集: {source_id} -> {akshare_function}")

        # 根据不同函数调用相应的AKShare接口
        try:
            print(f"🔄 准备调用AKShare函数: {akshare_function}")
            data = None

            if akshare_function == "news_economic_baidu":
                # 百度财经新闻
                print("📡 调用 news_economic_baidu(date='20241107')")
                data = await asyncio.to_thread(ak.news_economic_baidu, date="20241107")
                print(f"📡 news_economic_baidu 返回: {type(data)}")
            elif akshare_function == "futures_news_shmet":
                # 上海期货交易所新闻
                print("📡 调用 futures_news_shmet(symbol='全部')")
                data = await asyncio.to_thread(ak.futures_news_shmet, symbol="全部")
                print(f"📡 futures_news_shmet 返回: {type(data)}")
            else:
                # 尝试通用调用
                print(f"📡 尝试通用调用函数: {akshare_function}")
                ak_func = getattr(ak, akshare_function)
                data = await asyncio.to_thread(ak_func)
                print(f"📡 通用调用返回: {type(data)}")

            # 检查数据
            print(f"🔍 检查返回数据: data={data is not None}, type={type(data)}")
            if data is None:
                logger.warning(f"AKShare函数 {akshare_function} 返回 None")
                print("❌ AKShare函数返回 None")
                return []

            if hasattr(data, 'empty'):
                print(f"🔍 DataFrame是否为空: {data.empty}")
                if data.empty:
                    logger.warning(f"AKShare函数 {akshare_function} 返回空DataFrame")
                    print("❌ AKShare函数返回空DataFrame")
                    return []

            # 转换为标准格式并确保JSON可序列化
            print("🔄 转换为记录格式...")
            records = data.to_dict('records')
            print(f"✅ 转换为 {len(records)} 条记录")

            # 处理pandas Timestamp对象，确保JSON可序列化
            for record in records:
                for key, value in record.items():
                    if hasattr(value, 'isoformat'):  # pandas Timestamp对象
                        record[key] = value.isoformat()
                        print(f"转换Timestamp字段 {key}")
                    elif hasattr(value, 'strftime'):  # datetime对象
                        if hasattr(value, 'hour'):  # 完整的datetime对象
                            record[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                        else:  # datetime.date对象
                            record[key] = value.strftime('%Y-%m-%d')
                        print(f"转换datetime字段 {key}")
                    elif str(type(value)).startswith("<class 'pandas"):  # 其他pandas类型
                        record[key] = str(value)
                        print(f"转换pandas对象字段 {key}")
                    elif isinstance(value, (int, float)) and str(key).lower() in ['date', '日期'] and not str(key).lower() in ['open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']:  # 可能是时间戳，但排除价格字段
                        # 如果日期字段是数字，尝试转换为日期字符串
                        try:
                            import datetime
                            if value > 1e10:  # 秒级时间戳
                                date_obj = datetime.datetime.fromtimestamp(value)
                                record[key] = date_obj.strftime('%Y-%m-%d')
                            elif value > 1e8:  # 毫秒级时间戳
                                date_obj = datetime.datetime.fromtimestamp(value / 1000)
                                record[key] = date_obj.strftime('%Y-%m-%d')
                            print(f"转换时间戳字段 {key}")
                        except:
                            pass  # 如果转换失败，保持原值

            logger.info(f"成功采集 {len(records)} 条新闻数据")
            print(f"🎯🎯🎯 财经新闻采集器完成，返回数据 🎯🎯🎯")

            return {
                "success": True,
                "data": records,
                "completed_all_batches": True,  # 单次采集完成所有数据
                "batches_info": {
                    "completed": 1,
                    "total": 1,
                    "symbols_collected": len(set(r.get('symbol', '') for r in records if r.get('symbol')))
                }
            }

        except Exception as func_error:
            logger.error(f"AKShare函数 {akshare_function} 调用失败: {func_error}")
            return []

    except Exception as e:
        logger.error(f"AKShare新闻数据采集失败: {e}")
        return []

    except Exception as e:
        logger.error(f"AKShare新闻数据采集失败: {e}")
        return []


async def collect_from_akshare_alternative_adapter(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """从AKShare适配器采集另类数据"""
    try:
        import akshare as ak
        logger.warning("另类数据采集适配器尚未完全实现")
        return []
    except Exception as e:
        logger.error(f"AKShare另类数据采集失败: {e}")
        return []


async def collect_generic_data(source_config: Dict[str, Any], request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """通用数据采集函数"""
    try:
        logger.warning("通用数据采集适配器尚未实现")
        return []
    except Exception as e:
        logger.error(f"通用数据采集失败: {e}")
        return []


async def validate_data_quality(data: List[Dict[str, Any]], source_type: str) -> float:
    """验证数据质量并返回质量分数"""
    try:
        if not data:
            return 0.0

        score = 0.0
        total_checks = 0

        # 检查数据完整性
        total_checks += 1
        if all(isinstance(item, dict) and item for item in data):
            score += 1.0
        else:
            score += 0.5

        # 检查字段一致性
        total_checks += 1
        if data:
            first_keys = set(data[0].keys())
            consistency_score = sum(1.0 for item in data if set(item.keys()) == first_keys) / len(data)
            score += consistency_score

        # 根据数据类型进行特定检查
        if source_type.lower() in ["股票数据", "stock"]:
            total_checks += 1
            # 检查是否有价格字段
            price_fields = ['close', 'open', 'high', 'low', 'volume']
            if any(any(field in item for field in price_fields) for item in data):
                score += 1.0

        # 确保total_checks至少为1，避免除零错误
        if total_checks == 0:
            total_checks = 1

        # 避免除零错误
        if total_checks == 0:
            return 50.0  # 默认中等质量

        quality_score = (score / total_checks) * 100.0

        # 确保返回有效的数值
        if not (isinstance(quality_score, (int, float)) and quality_score >= 0 and quality_score <= 100):
            return 50.0

        return min(100.0, max(0.0, quality_score))

    except Exception as e:
        logger.error(f"数据质量验证失败: {e}")
        return 50.0  # 默认中等质量


def parse_rate_limit(rate_limit_str: str) -> float:
    """
    解析频率限制字符串，返回采集间隔秒数（统一函数，符合架构设计）
    
    根据数据管理层架构设计，数据采集应按照数据源配置的rate_limit进行调度。
    此函数提供统一的频率解析逻辑，确保所有调度器使用相同的解析规则。
    
    Args:
        rate_limit_str: 频率限制字符串，支持多种格式：
            - "10次/分钟" -> 返回 6.0 (60/10)
            - "1次/小时" -> 返回 3600.0 (3600/1)
            - "按协议" -> 返回 30.0 (保守设置，约2次/分钟)
            - "无限制" -> 返回 60.0 (默认每分钟1次)
            - "100次/分钟" -> 返回 5.0 (计算 60/100=0.6，应用最小5秒下限)
    
    Returns:
        float: 采集间隔秒数（最小5秒，降低外部 API 限流/封禁风险）
    
    Examples:
        >>> parse_rate_limit("10次/分钟")
        6.0
        >>> parse_rate_limit("1次/小时")
        3600.0
        >>> parse_rate_limit("按协议")
        30.0
        >>> parse_rate_limit("无限制")
        60.0
    """
    import re
    
    if not rate_limit_str or not isinstance(rate_limit_str, str):
        return 60.0  # 默认60秒（每分钟1次）
    
    rate_limit_str = rate_limit_str.strip()
    
    # 处理特殊值
    if rate_limit_str == "无限制" or rate_limit_str.lower() == "unlimited":
        return 60.0  # 默认每分钟1次
    
    if "按协议" in rate_limit_str or "protocol" in rate_limit_str.lower():
        return 30.0  # 保守设置：约2次/分钟，降低未知协议源的限流/封禁风险
    
    # 解析格式：支持 "10次/分钟", "1次/小时", "1次/天" 等
    # 匹配模式：数字 + "次" + "/" + 时间单位
    match = re.search(r'(\d+)\s*次\s*/?\s*(\w+)', rate_limit_str)
    if match:
        count = int(match.group(1))
        unit = match.group(2).strip()
        
        if count <= 0:
            return 60.0  # 无效值，使用默认值
        
        # 根据时间单位计算间隔秒数
        if unit in ['分钟', 'minute', 'min', 'm']:
            interval = 60.0 / count
        elif unit in ['小时', 'hour', 'h']:
            interval = 3600.0 / count
        elif unit in ['天', 'day', 'd']:
            interval = 86400.0 / count
        elif unit in ['秒', 'second', 'sec', 's']:
            interval = 1.0 / count if count > 0 else 1.0
        else:
            # 未知单位，假设是分钟
            interval = 60.0 / count
        
        # 确保最小间隔为5秒，降低外部 API 限流/封禁风险
        return max(5.0, interval)
    
    # 如果无法解析，尝试直接解析为数字（假设是秒数）
    try:
        interval = float(rate_limit_str)
        return max(5.0, interval)
    except (ValueError, TypeError):
        # 完全无法解析，使用默认值
        logger.warning(f"无法解析频率限制字符串: {rate_limit_str}，使用默认值60秒")
        return 60.0
