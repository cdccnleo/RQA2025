"""
交易信号服务层
封装实际的交易信号生成组件，为API提供统一接口
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# 导入交易层组件
try:
    from src.trading.signal.signal_generator import SimpleSignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入信号生成器: {e}")
    SIGNAL_GENERATOR_AVAILABLE = False

# 导入实时数据集成
try:
    from .realtime_data_integration import get_realtime_data_integration
    REALTIME_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入实时数据集成: {e}")
    REALTIME_INTEGRATION_AVAILABLE = False


# 单例实例
_signal_generator: Optional[Any] = None
_realtime_integration = None


def get_signal_generator() -> Optional[Any]:
    """获取信号生成器实例"""
    global _signal_generator
    if _signal_generator is None and SIGNAL_GENERATOR_AVAILABLE:
        try:
            _signal_generator = SimpleSignalGenerator()
            logger.info("信号生成器初始化成功")
        except Exception as e:
            logger.error(f"初始化信号生成器失败: {e}")
    return _signal_generator


def get_realtime_integration():
    """获取实时数据集成实例"""
    global _realtime_integration
    if _realtime_integration is None and REALTIME_INTEGRATION_AVAILABLE:
        try:
            _realtime_integration = get_realtime_data_integration()
            logger.info("实时数据集成初始化成功")
        except Exception as e:
            logger.error(f"初始化实时数据集成失败: {e}")
    return _realtime_integration


# ==================== 交易信号服务 ====================

def get_realtime_signals(strategy_id: str = None, strategy_name: str = None) -> List[Dict[str, Any]]:
    """
    获取实时交易信号 - 从真实信号生成器获取，不使用模拟数据
    
    Args:
        strategy_id: 策略ID（可选）
        strategy_name: 策略名称（可选）
    
    Returns:
        信号列表，每个信号包含策略关联信息
    """
    signal_generator = get_signal_generator()
    
    if not signal_generator:
        # 量化交易系统要求：不使用模拟数据，返回空列表
        logger.debug("信号生成器不可用，返回空信号列表")
        return []
    
    try:
        # 尝试从信号生成器获取实时信号
        signals = []

        # 尝试不同的方法名
        if hasattr(signal_generator, 'get_realtime_signals'):
            signals = signal_generator.get_realtime_signals()
        elif hasattr(signal_generator, 'get_current_signals'):
            signals = signal_generator.get_current_signals()
        elif hasattr(signal_generator, 'generate_signals'):
            # generate_signals 需要传入市场数据参数
            market_data = _get_market_data_for_signal_generation(strategy_id)
            if market_data is not None and not market_data.empty:
                try:
                    # 获取symbol
                    symbol = _get_symbol_for_strategy(strategy_id)
                    signals = signal_generator.generate_signals(market_data, strategy_id=strategy_id, symbol=symbol)
                    logger.info(f"信号生成成功，生成 {len(signals) if signals else 0} 个信号")
                except Exception as e:
                    logger.error(f"信号生成失败: {e}")
                    signals = []
            else:
                logger.warning("无法获取市场数据，跳过信号生成")
                signals = []
        elif hasattr(signal_generator, 'get_signals'):
            signals = signal_generator.get_signals()
        
        # 转换信号格式（如果需要）
        if signals:
            formatted_signals = []
            for signal in signals:
                # 如果信号是对象，转换为字典
                if not isinstance(signal, dict):
                    if hasattr(signal, '__dict__'):
                        signal_dict = signal.__dict__
                    elif hasattr(signal, 'to_dict'):
                        signal_dict = signal.to_dict()
                    else:
                        continue
                else:
                    signal_dict = signal
                
                # 从metadata中获取策略信息
                metadata = signal_dict.get('metadata', {})
                signal_strategy_id = strategy_id or metadata.get('strategy_id', signal_dict.get('strategy_id', ''))
                signal_strategy_name = strategy_name or metadata.get('strategy_name', signal_dict.get('strategy_name', ''))
                
                # 如果没有策略名称，尝试从策略ID获取
                if not signal_strategy_name and signal_strategy_id:
                    signal_strategy_name = _get_strategy_name_by_id(signal_strategy_id)
                
                signal_data = {
                    "id": signal_dict.get('id', signal_dict.get('signal_id', '')),
                    "strategy_id": signal_strategy_id,
                    "strategy_name": signal_strategy_name or signal_strategy_id or '未知策略',
                    "symbol": signal_dict.get('symbol', metadata.get('symbol', 'UNKNOWN')),
                    "type": signal_dict.get('type', signal_dict.get('signal_type', 'unknown')),
                    "strength": signal_dict.get('strength', 0),
                    "price": signal_dict.get('price', metadata.get('price', 0)),
                    "status": signal_dict.get('status', 'pending'),
                    "timestamp": signal_dict.get('timestamp', int(datetime.now().timestamp())),
                    "accuracy": signal_dict.get('accuracy', 0),
                    "latency": signal_dict.get('latency', 0),
                    "quality": signal_dict.get('quality', 0),
                    "source": metadata.get('signal_source', 'generator')
                }
                formatted_signals.append(signal_data)
                
                # 保存到持久化存储
                try:
                    from .signal_persistence import save_signal
                    save_signal(signal_data)
                except Exception as e:
                    logger.debug(f"保存信号到持久化存储失败: {e}")
            
            return formatted_signals
        
        # 如果没有获取到信号，返回空列表
        return []
        
    except Exception as e:
        logger.error(f"从信号生成器获取实时信号失败: {e}")
        # 量化交易系统要求：不使用模拟数据，返回空列表
        return []


def _get_market_data_for_signal_generation(strategy_id: str = None):
    """
    获取信号生成所需的市场数据
    
    Args:
        strategy_id: 策略ID，如果提供则从策略配置获取股票代码
        
    Returns:
        pandas DataFrame with market data
    """
    try:
        from .market_data_service import get_market_data_service
        from datetime import datetime, timedelta

        service = get_market_data_service()
        
        # 确定要使用的股票代码
        symbols = []
        
        if strategy_id:
            # 从策略配置获取股票代码
            symbols = _get_symbols_from_strategy(strategy_id)
            if symbols:
                logger.info(f"从策略 {strategy_id} 获取股票代码: {symbols}")
        
        if not symbols:
            # 从数据库获取默认股票代码（数据最多的股票）
            default_symbol = service.get_default_symbol()
            if default_symbol:
                symbols = [default_symbol]
                logger.info(f"使用数据库默认股票代码: {default_symbol}")
        
        if not symbols:
            logger.warning("没有可用的股票代码")
            return None

        # 获取最近30天的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # 获取第一个股票的数据（后续可扩展为多股票）
        symbol = symbols[0]
        df = service.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )

        if df.empty:
            logger.warning(f"获取到的市场数据为空: {symbol}")
            return None

        logger.info(f"获取市场数据成功: {symbol}, 记录数: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        return None


def _get_symbols_from_strategy(strategy_id: str) -> list:
    """
    从策略配置获取股票代码列表
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        股票代码列表
    """
    try:
        # 从执行状态获取策略信息
        from .execution_persistence import load_execution_state
        
        state = load_execution_state(strategy_id)
        if not state:
            logger.warning(f"策略执行状态不存在: {strategy_id}")
            return []
        
        # 从策略配置获取股票代码
        strategy_config = state.get('strategy_config', {})
        symbols = strategy_config.get('symbols', [])
        
        if symbols:
            logger.info(f"从策略 {strategy_id} 配置获取股票代码: {symbols}")
            return symbols
        
        # 如果没有配置symbols，尝试从其他字段获取
        universe = strategy_config.get('universe', [])
        if universe:
            logger.info(f"从策略 {strategy_id} 配置获取universe: {universe}")
            return universe
        
        instruments = strategy_config.get('instruments', [])
        if instruments:
            logger.info(f"从策略 {strategy_id} 配置获取instruments: {instruments}")
            return instruments
        
        logger.warning(f"策略 {strategy_id} 没有配置股票代码")
        return []
        
    except Exception as e:
        logger.error(f"从策略获取股票代码失败 {strategy_id}: {e}")
        return []


def _get_symbol_for_strategy(strategy_id: str = None) -> str:
    """
    获取策略对应的股票代码
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        股票代码
    """
    if not strategy_id:
        return "UNKNOWN"
    
    try:
        symbols = _get_symbols_from_strategy(strategy_id)
        if symbols:
            return symbols[0]  # 返回第一个股票代码
    except Exception as e:
        logger.debug(f"获取策略 {strategy_id} 的股票代码失败: {e}")
    
    return "UNKNOWN"


def _get_strategy_name_by_id(strategy_id: str) -> str:
    """
    根据策略ID获取策略名称
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        策略名称
    """
    if not strategy_id:
        return ""
    
    try:
        # 从执行状态获取策略名称
        from .execution_persistence import load_execution_state
        state = load_execution_state(strategy_id)
        if state:
            return state.get('name', strategy_id)
        
        # 从生命周期获取策略名称
        from .strategy_lifecycle import get_strategy_lifecycle
        lifecycle = get_strategy_lifecycle(strategy_id)
        if lifecycle:
            return lifecycle.strategy_name
        
    except Exception as e:
        logger.debug(f"获取策略 {strategy_id} 名称失败: {e}")
    
    return strategy_id


async def get_realtime_signals_with_live_data() -> List[Dict[str, Any]]:
    """
    获取实时交易信号 - 使用实时数据流
    
    Returns:
        实时信号列表
    """
    signal_generator = get_signal_generator()
    realtime_integration = get_realtime_integration()
    
    if not signal_generator:
        logger.debug("信号生成器不可用，返回空信号列表")
        return []
    
    try:
        # 获取市场数据服务
        from .market_data_service import get_market_data_service
        service = get_market_data_service()
        
        # 获取默认股票代码
        symbol = service.get_default_symbol()
        if not symbol:
            logger.warning("没有可用的股票代码")
            return []
        
        # 获取历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        historical_df = service.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )
        
        # 如果实时数据集成可用，合并实时数据
        if realtime_integration and realtime_integration._is_running:
            combined_df = realtime_integration.get_combined_dataframe(symbol, historical_df)
            if combined_df is not None and not combined_df.empty:
                market_data = combined_df
                logger.info(f"使用组合数据（历史+实时）生成信号: {symbol}, 记录数: {len(market_data)}")
            else:
                market_data = historical_df
                logger.info(f"使用历史数据生成信号: {symbol}, 记录数: {len(market_data)}")
        else:
            market_data = historical_df
            logger.info(f"使用历史数据生成信号: {symbol}, 记录数: {len(market_data)}")
        
        if market_data is None or market_data.empty:
            logger.warning("无法获取市场数据，跳过信号生成")
            return []
        
        # 生成信号
        signals = []
        if hasattr(signal_generator, 'generate_signals'):
            signals = signal_generator.generate_signals(market_data)
            logger.info(f"信号生成成功，生成 {len(signals) if signals else 0} 个信号")
        
        # 转换信号格式
        if signals:
            formatted_signals = []
            for signal in signals:
                if not isinstance(signal, dict):
                    if hasattr(signal, '__dict__'):
                        signal_dict = signal.__dict__
                    elif hasattr(signal, 'to_dict'):
                        signal_dict = signal.to_dict()
                    else:
                        continue
                else:
                    signal_dict = signal
                
                signal_data = {
                    "id": signal_dict.get('id', signal_dict.get('signal_id', '')),
                    "symbol": symbol,  # 使用实际股票代码
                    "type": signal_dict.get('type', signal_dict.get('signal_type', 'unknown')),
                    "strength": signal_dict.get('strength', 0),
                    "price": signal_dict.get('price', 0),
                    "status": signal_dict.get('status', 'pending'),
                    "timestamp": signal_dict.get('timestamp', int(datetime.now().timestamp())),
                    "accuracy": signal_dict.get('accuracy', 0),
                    "latency": signal_dict.get('latency', 0),
                    "quality": signal_dict.get('quality', 0),
                    "data_source": "realtime" if realtime_integration and realtime_integration._is_running else "historical"
                }
                formatted_signals.append(signal_data)
                
                # 保存到持久化存储
                try:
                    from .signal_persistence import save_signal
                    save_signal(signal_data)
                except Exception as e:
                    logger.debug(f"保存信号到持久化存储失败: {e}")
            
            return formatted_signals
        
        return []
        
    except Exception as e:
        logger.error(f"获取实时信号失败: {e}")
        return []


def get_signal_stats() -> Dict[str, Any]:
    """获取信号统计"""
    signals = get_realtime_signals()
    
    today = datetime.now().date()
    today_signals = [s for s in signals if s.get('timestamp') and 
                     datetime.fromtimestamp(s['timestamp']).date() == today]
    
    accuracy_scores = [s.get('accuracy', 0) for s in signals if s.get('accuracy')]
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    
    latencies = [s.get('latency', 0) for s in signals if s.get('latency')]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    strengths = [s.get('strength', 0) for s in signals if s.get('strength')]
    avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
    
    return {
        "today_signals": len(today_signals),
        "accuracy": avg_accuracy,
        "avg_latency": avg_latency,
        "avg_strength": avg_strength
    }


def get_signal_distribution() -> Dict[str, Any]:
    """获取信号分布"""
    signals = get_realtime_signals()
    
    type_distribution = {}
    for signal in signals:
        signal_type = signal.get('type', 'unknown')
        type_distribution[signal_type] = type_distribution.get(signal_type, 0) + 1
    
    # 生成质量趋势
    quality_trend = []
    for i in range(24):
        hour_ago = datetime.now() - timedelta(hours=i)
        hour_signals = [s for s in signals if s.get('timestamp') and 
                       datetime.fromtimestamp(s['timestamp']) >= hour_ago - timedelta(hours=1)]
        if hour_signals:
            avg_quality = sum(s.get('quality', 0) for s in hour_signals) / len(hour_signals)
            quality_trend.append({"quality": avg_quality, "timestamp": int(hour_ago.timestamp())})
    
    quality_trend.reverse()
    
    # 从实际信号执行结果计算有效性（不使用硬编码）
    effectiveness = {}
    try:
        from .signal_persistence import list_signals
        
        # 获取已执行的信号
        executed_signals = list_signals(status="executed", limit=1000)
        
        # 按类型统计有效性
        type_effectiveness = {}
        for signal in executed_signals:
            signal_type = signal.get('type', 'unknown')
            if signal_type not in type_effectiveness:
                type_effectiveness[signal_type] = {"total": 0, "accurate": 0}
            
            type_effectiveness[signal_type]["total"] += 1
            if signal.get('accuracy', 0) > 0.5:  # 准确率超过50%认为是有效的
                type_effectiveness[signal_type]["accurate"] += 1
        
        # 计算有效性
        for signal_type, stats in type_effectiveness.items():
            if stats["total"] > 0:
                effectiveness[signal_type] = stats["accurate"] / stats["total"]
        
        # 如果没有数据，返回空字典
        if not effectiveness:
            effectiveness = {}
    except Exception as e:
        logger.debug(f"计算信号有效性失败: {e}")
        effectiveness = {}
    
    return {
        "type_distribution": type_distribution,
        "quality_trend": quality_trend,
        "effectiveness": effectiveness
    }


# 注意：已移除_get_mock_signals()函数，系统要求不使用模拟数据

