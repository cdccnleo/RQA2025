"""
策略执行服务层
封装RealTimeStrategyEngine，提供策略执行管理功能
符合架构设计：使用EventBus进行事件驱动监控，通过统一适配器工厂访问策略层和交易层服务
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# 使用统一日志系统（符合架构设计：基础设施层统一日志接口）
try:
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
    logger = get_unified_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# 全局事件总线（符合架构设计：事件驱动监控）
_event_bus = None

def _get_event_bus():
    """获取事件总线实例（符合架构设计）"""
    global _event_bus
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

# 全局实时引擎实例
_realtime_engine = None
_engine_lock = asyncio.Lock()

# 全局适配器工厂实例（符合架构设计：统一适配器工厂）
_adapter_factory = None
_strategy_adapter = None
_trading_adapter = None

def _get_adapter_factory():
    """获取统一适配器工厂实例（符合架构设计）"""
    global _adapter_factory
    if _adapter_factory is None:
        try:
            from src.infrastructure.integration.business_adapters import get_unified_adapter_factory
            _adapter_factory = get_unified_adapter_factory()
            logger.info("统一适配器工厂已获取")
        except Exception as e:
            logger.warning(f"获取统一适配器工厂失败: {e}")
            _adapter_factory = None
    return _adapter_factory

def _get_strategy_adapter():
    """获取策略层适配器（符合架构设计：通过统一适配器工厂访问策略层）"""
    global _strategy_adapter
    if _strategy_adapter is None:
        try:
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            factory = _get_adapter_factory()
            if factory:
                _strategy_adapter = factory.get_adapter(BusinessLayerType.STRATEGY)
                if _strategy_adapter:
                    logger.info("策略层适配器已获取")
                else:
                    logger.warning("策略层适配器获取失败，将使用降级方案")
            else:
                logger.warning("统一适配器工厂不可用，将使用降级方案")
        except Exception as e:
            logger.warning(f"获取策略层适配器失败: {e}")
            _strategy_adapter = None
    return _strategy_adapter

def _get_trading_adapter():
    """获取交易层适配器（符合架构设计：通过统一适配器工厂访问交易层）"""
    global _trading_adapter
    if _trading_adapter is None:
        try:
            from src.infrastructure.integration.unified_business_adapters import BusinessLayerType
            factory = _get_adapter_factory()
            if factory:
                _trading_adapter = factory.get_adapter(BusinessLayerType.TRADING)
                if _trading_adapter:
                    logger.info("交易层适配器已获取")
                else:
                    logger.warning("交易层适配器获取失败，将使用降级方案")
            else:
                logger.warning("统一适配器工厂不可用，将使用降级方案")
        except Exception as e:
            logger.warning(f"获取交易层适配器失败: {e}")
            _trading_adapter = None
    return _trading_adapter

async def get_realtime_engine():
    """获取或创建实时策略引擎实例（符合架构设计：优先通过策略层适配器访问）"""
    global _realtime_engine
    
    async with _engine_lock:
        if _realtime_engine is None:
            # 优先通过策略层适配器获取（符合架构设计）
            strategy_adapter = _get_strategy_adapter()
            if strategy_adapter:
                try:
                    # 尝试通过适配器获取实时策略引擎
                    if hasattr(strategy_adapter, 'get_realtime_strategy_engine'):
                        _realtime_engine = strategy_adapter.get_realtime_strategy_engine()
                        if _realtime_engine:
                            logger.info("通过策略层适配器获取实时策略引擎成功")
                            if not _realtime_engine.is_running() if hasattr(_realtime_engine, 'is_running') else False:
                                await _realtime_engine.start()
                            return _realtime_engine
                except Exception as e:
                    logger.debug(f"通过策略层适配器获取实时策略引擎失败: {e}")
            
            # 降级方案：直接实例化（符合架构设计：降级处理）
            try:
                from src.strategy.realtime.real_time_processor import RealTimeStrategyEngine, RealTimeConfig
                _realtime_engine = RealTimeStrategyEngine(RealTimeConfig())
                await _realtime_engine.start()
                logger.info("实时策略引擎已启动（降级方案）")
            except Exception as e:
                logger.error(f"初始化实时策略引擎失败: {e}")
                # 返回None，API层会处理
        return _realtime_engine


async def get_strategy_execution_status() -> Dict[str, Any]:
    """获取策略执行状态 - 使用真实数据，优先从缓存加载，其次从持久化存储加载"""
    # 辅助函数：检查策略是否已退市或已停止
    def _is_strategy_retired_or_stopped(strategy_id: str, exec_status: str = None) -> bool:
        """
        检查策略是否已退市（archived）或已停止（stopped）
        
        Args:
            strategy_id: 策略ID
            exec_status: 执行状态（可选），如果为 'stopped' 则视为已停止
        """
        # 如果执行状态为 stopped，视为已停止
        if exec_status == 'stopped':
            return True
        
        # 检查生命周期状态
        try:
            from .strategy_lifecycle import get_strategy_lifecycle
            lifecycle = get_strategy_lifecycle(strategy_id)
            if lifecycle and lifecycle.current_status.value == 'archived':
                return True
        except Exception as e:
            logger.debug(f"检查策略生命周期状态失败 {strategy_id}: {e}")
        
        # 如果生命周期不存在，检查执行状态
        if exec_status is None:
            try:
                from .execution_persistence import load_execution_state
                exec_state = load_execution_state(strategy_id)
                if exec_state and exec_state.get('status') == 'stopped':
                    return True
            except Exception as e:
                logger.debug(f"检查策略执行状态失败 {strategy_id}: {e}")
        
        return False
    
    # 1. 优先从Redis缓存加载
    try:
        from .redis_cache import get_execution_status_cache
        cached_data = get_execution_status_cache()
        if cached_data:
            logger.debug("从Redis缓存获取执行状态成功")
            # 过滤掉已停止的策略
            if 'strategies' in cached_data:
                cached_data['strategies'] = [
                    s for s in cached_data['strategies'] 
                    if not _is_strategy_retired_or_stopped(s.get('id'), s.get('status'))
                ]
            return cached_data
    except Exception as e:
        logger.debug(f"从缓存获取执行状态失败: {e}")
    
    # 2. 发布执行状态查询事件
    event_bus = _get_event_bus()
    if event_bus:
        try:
            from src.core.event_bus.types import EventType
            event_bus.publish(
                EventType.EXECUTION_STARTED,
                {"source": "strategy_execution_service", "action": "get_execution_status"},
                source="strategy_execution_service"
            )
        except Exception as e:
            logger.debug(f"发布事件失败: {e}")
    
    try:
        # 3. 从实时引擎获取（优先，因为实时引擎包含最新状态）
        engine = await get_realtime_engine()
        if engine and hasattr(engine, 'strategies') and engine.strategies:
            strategies = []
            for strategy_id, strategy in engine.strategies.items():
                # 从持久化存储获取策略状态（以获取准确的 paused/stopped 状态）
                strategy_status = "running" if getattr(strategy, 'is_active', True) else "stopped"
                try:
                    from .execution_persistence import load_execution_state
                    persisted_state = load_execution_state(strategy_id)
                    if persisted_state and 'status' in persisted_state:
                        # 使用持久化存储中的状态（更准确的 paused/stopped 区分）
                        strategy_status = persisted_state['status']
                        logger.debug(f"策略 {strategy_id} 使用持久化状态: {strategy_status}")
                except Exception as e:
                    logger.debug(f"从持久化存储获取策略状态失败 {strategy_id}: {e}")
                
                # 过滤已退市或已停止的策略
                if _is_strategy_retired_or_stopped(strategy_id, strategy_status):
                    logger.debug(f"策略 {strategy_id} 已退市或已停止，从执行列表中过滤")
                    continue
                
                metrics = strategy.get_performance_metrics() if hasattr(strategy, 'get_performance_metrics') else {}
                strategy_data = {
                    "id": strategy_id,
                    "name": getattr(strategy, 'name', strategy_id),
                    "type": getattr(strategy, 'strategy_type', 'unknown'),
                    "status": strategy_status,
                    "latency": metrics.get('latency', 0),
                    "throughput": metrics.get('throughput', 0),
                    "signals_count": len(getattr(strategy, 'signals', [])),
                    "positions_count": getattr(strategy, 'position', 0)
                }
                strategies.append(strategy_data)
                
                # 保存到持久化存储
                try:
                    from .execution_persistence import save_execution_state
                    save_execution_state(strategy_id, {
                        "name": strategy_data["name"],
                        "type": strategy_data["type"],
                        "status": strategy_data["status"],
                        "latency": strategy_data["latency"],
                        "throughput": strategy_data["throughput"],
                        "signals_count": strategy_data["signals_count"],
                        "positions_count": strategy_data["positions_count"],
                        "metrics": metrics
                    })
                except Exception as e:
                    logger.debug(f"保存执行状态到持久化存储失败: {e}")
            
            if strategies:
                result = {
                    "strategies": strategies,
                    "running_count": len([s for s in strategies if s["status"] == "running"]),
                    "paused_count": 0,
                    "stopped_count": len([s for s in strategies if s["status"] == "stopped"]),
                    "total_count": len(strategies)
                }
                # 缓存结果
                try:
                    from .redis_cache import set_execution_status_cache
                    set_execution_status_cache(result)
                except Exception as e:
                    logger.debug(f"缓存执行状态失败: {e}")
                return result
        
        # 4. 从持久化存储加载（降级方案）
        try:
            from .execution_persistence import list_execution_states
            persisted_states = list_execution_states(limit=100)
            if persisted_states:
                strategies = []
                for state in persisted_states:
                    strategy_id = state.get("strategy_id")
                    strategy_status = state.get("status", "stopped")
                    
                    # 过滤已退市或已停止的策略
                    if _is_strategy_retired_or_stopped(strategy_id, strategy_status):
                        logger.debug(f"策略 {strategy_id} 已退市或已停止，从执行列表中过滤")
                        continue
                    
                    strategies.append({
                        "id": strategy_id,
                        "name": state.get("name", strategy_id),
                        "type": state.get("type", "unknown"),
                        "status": strategy_status,
                        "latency": state.get("latency", 0),
                        "throughput": state.get("throughput", 0),
                        "signals_count": state.get("signals_count", 0),
                        "positions_count": state.get("positions_count", 0)
                    })
                
                if strategies:
                    result = {
                        "strategies": strategies,
                        "running_count": len([s for s in strategies if s["status"] == "running"]),
                        "paused_count": len([s for s in strategies if s["status"] == "paused"]),
                        "stopped_count": len([s for s in strategies if s["status"] == "stopped"]),
                        "total_count": len(strategies)
                    }
                    # 缓存结果
                    try:
                        from .redis_cache import set_execution_status_cache
                        set_execution_status_cache(result)
                    except Exception as e:
                        logger.debug(f"缓存执行状态失败: {e}")
                    return result
        except Exception as e:
            logger.debug(f"从持久化存储加载失败: {e}")
        
        # 4. 从实时引擎获取
        engine = await get_realtime_engine()
        if engine is None:
            result = {
                "strategies": [],
                "running_count": 0,
                "paused_count": 0,
                "stopped_count": 0,
                "total_count": 0
            }
            return result
        
        strategies = []
        for strategy_id, strategy in engine.strategies.items():
            metrics = strategy.get_performance_metrics() if hasattr(strategy, 'get_performance_metrics') else {}
            strategy_data = {
                "id": strategy_id,
                "name": getattr(strategy, 'name', strategy_id),
                "type": getattr(strategy, 'strategy_type', 'unknown'),
                "status": "running" if getattr(strategy, 'is_active', True) else "stopped",
                "latency": metrics.get('latency', 0),
                "throughput": metrics.get('throughput', 0),
                "signals_count": len(getattr(strategy, 'signals', [])),
                "positions_count": getattr(strategy, 'position', 0)
            }
            strategies.append(strategy_data)
            
            # 保存到持久化存储
            try:
                from .execution_persistence import save_execution_state
                save_execution_state(strategy_id, {
                    "name": strategy_data["name"],
                    "type": strategy_data["type"],
                    "status": strategy_data["status"],
                    "latency": strategy_data["latency"],
                    "throughput": strategy_data["throughput"],
                    "signals_count": strategy_data["signals_count"],
                    "positions_count": strategy_data["positions_count"],
                    "metrics": metrics
                })
            except Exception as e:
                logger.debug(f"保存执行状态到持久化存储失败: {e}")
        
        result = {
            "strategies": strategies,
            "running_count": len([s for s in strategies if s["status"] == "running"]),
            "paused_count": 0,
            "stopped_count": len([s for s in strategies if s["status"] == "stopped"]),
            "total_count": len(strategies)
        }
        
        # 缓存结果
        try:
            from .redis_cache import set_execution_status_cache
            set_execution_status_cache(result)
        except Exception as e:
            logger.debug(f"缓存执行状态失败: {e}")
        
        return result
    except Exception as e:
        logger.error(f"获取策略执行状态失败: {e}")
        return {
            "strategies": [],
            "running_count": 0,
            "paused_count": 0,
            "stopped_count": 0,
            "total_count": 0
        }


async def get_execution_metrics() -> Dict[str, Any]:
    """获取执行性能指标 - 优先从缓存加载"""
    # 1. 优先从Redis缓存加载
    try:
        from .redis_cache import get_execution_metrics_cache
        cached_data = get_execution_metrics_cache()
        if cached_data:
            logger.debug("从Redis缓存获取执行指标成功")
            return cached_data
    except Exception as e:
        logger.debug(f"从缓存获取执行指标失败: {e}")
    
    try:
        # 2. 从实时引擎获取
        engine = await get_realtime_engine()
        if engine is None:
            result = {
                "avg_latency": 0,
                "today_signals": 0,
                "total_trades": 0,
                "latency_history": [],
                "throughput_history": []
            }
            return result
        
        metrics = engine.get_performance_metrics()
        stream_metrics = metrics.get('stream_metrics', {})
        strategy_metrics = metrics.get('strategy_metrics', {})
        
        result = {
            "avg_latency": stream_metrics.get('processing_latency', 0),
            "today_signals": strategy_metrics.get('total_signals', 0),
            "total_trades": strategy_metrics.get('total_trades', 0),
            "latency_history": [],  # 需要从监控器获取历史数据
            "throughput_history": []
        }
        
        # 3. 缓存结果
        try:
            from .redis_cache import set_execution_metrics_cache
            set_execution_metrics_cache(result)
        except Exception as e:
            logger.debug(f"缓存执行指标失败: {e}")
        
        return result
    except Exception as e:
        logger.error(f"获取执行指标失败: {e}")
        return {
            "avg_latency": 0,
            "today_signals": 0,
            "total_trades": 0,
            "latency_history": [],
            "throughput_history": []
        }


async def get_realtime_metrics() -> Dict[str, Any]:
    """获取实时处理指标 - 优先从缓存加载"""
    # 1. 优先从Redis缓存加载
    try:
        from .redis_cache import get_realtime_signals_cache
        cached_data = get_realtime_signals_cache()
        if cached_data:
            logger.debug("从Redis缓存获取实时指标成功")
            return cached_data
    except Exception as e:
        logger.debug(f"从缓存获取实时指标失败: {e}")
    
    try:
        # 2. 从实时引擎获取
        engine = await get_realtime_engine()
        if engine is None:
            result = {
                "metrics": {},
                "stream_metrics": {},
                "strategies": [],
                "history": {"latency": [], "throughput": []}
            }
            return result
        
        metrics = engine.get_performance_metrics()
        stream_metrics = metrics.get('stream_metrics', {})
        
        strategies = []
        for strategy_id, strategy in engine.strategies.items():
            strategy_metrics = strategy.get_performance_metrics() if hasattr(strategy, 'get_performance_metrics') else {}
            strategies.append({
                "id": strategy_id,
                "name": getattr(strategy, 'name', strategy_id),
                "type": getattr(strategy, 'strategy_type', 'unknown'),
                "metrics": {
                    "latency": strategy_metrics.get('latency', 0),
                    "signals_count": len(getattr(strategy, 'signals', [])),
                    "positions_count": getattr(strategy, 'position', 0),
                    "trades_count": len(getattr(strategy, 'trades', []))
                }
            })
        
        result = {
            "metrics": stream_metrics,
            "stream_metrics": stream_metrics,
            "strategies": strategies,
            "history": {
                "latency": [],
                "throughput": []
            }
        }
        
        # 3. 缓存结果
        try:
            from .redis_cache import set_realtime_signals_cache
            set_realtime_signals_cache(result)
        except Exception as e:
            logger.debug(f"缓存实时指标失败: {e}")
        
        return result
    except Exception as e:
        logger.error(f"获取实时指标失败: {e}")
        return {
            "metrics": {},
            "stream_metrics": {},
            "strategies": [],
            "history": {"latency": [], "throughput": []}
        }


async def start_strategy(strategy_id: str) -> bool:
    """
    启动策略执行
    
    支持从以下位置加载策略配置：
    1. 策略构思目录（strategy_conceptions）
    2. 执行状态持久化存储（execution_states）- 用于策略构思已被删除但执行状态仍存在的情况
    """
    try:
        strategy_config = None
        
        # 1. 首先尝试从策略构思目录加载
        try:
            from .strategy_routes import load_strategy_conceptions
            strategies = load_strategy_conceptions()
            strategy_config = next((s for s in strategies if s.get("id") == strategy_id), None)
        except Exception as e:
            logger.debug(f"从策略构思目录加载失败: {e}")
        
        # 2. 如果策略构思不存在，尝试从执行状态加载
        if not strategy_config:
            try:
                from .execution_persistence import load_execution_state
                execution_state = load_execution_state(strategy_id)
                if execution_state:
                    # 从执行状态构建策略配置
                    strategy_config = {
                        "id": strategy_id,
                        "name": execution_state.get("name", strategy_id),
                        "type": execution_state.get("type", "unknown"),
                        "parameters": execution_state.get("parameters", {})
                    }
                    logger.info(f"从执行状态加载策略配置: {strategy_id}")
            except Exception as e:
                logger.debug(f"从执行状态加载失败: {e}")
        
        # 3. 如果仍然找不到策略配置，创建默认配置
        if not strategy_config:
            logger.warning(f"策略 {strategy_id} 未找到配置，使用默认配置")
            strategy_config = {
                "id": strategy_id,
                "name": strategy_id,
                "type": "unknown",
                "parameters": {}
            }
        
        engine = await get_realtime_engine()
        if engine is None:
            logger.error("实时引擎未初始化")
            return False
        
        # 创建策略配置对象
        from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategyType
        
        # 解析策略类型
        strategy_type_str = strategy_config.get("type", "unknown")
        try:
            strategy_type = StrategyType(strategy_type_str)
        except ValueError:
            # 如果类型不匹配，使用 QUANTITATIVE 作为默认值
            strategy_type = StrategyType.QUANTITATIVE
        
        config = StrategyConfig(
            strategy_id=strategy_id,
            strategy_name=strategy_config.get("name", strategy_id),
            strategy_type=strategy_type,
            parameters=strategy_config.get("parameters", {}),
            symbols=[],
            risk_limits={}
        )
        
        engine.register_strategy(config)
        
        # 保存执行状态到持久化存储
        try:
            from .execution_persistence import save_execution_state
            save_execution_state(strategy_id, {
                "name": strategy_config.get("name", strategy_id),
                "type": strategy_config.get("type", "unknown"),
                "status": "running",
                "latency": 0,
                "throughput": 0,
                "signals_count": 0,
                "positions_count": 0,
                "metrics": {}
            })
            logger.info(f"策略 {strategy_id} 执行状态已保存到持久化存储")
        except Exception as e:
            logger.warning(f"保存策略执行状态失败 {strategy_id}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"启动策略失败: {e}")
        return False


async def pause_strategy(strategy_id: str) -> bool:
    """
    暂停策略执行
    
    支持以下场景：
    1. 策略存在于实时引擎中 - 直接暂停
    2. 策略存在于持久化存储中 - 更新状态
    3. 策略仅存在于实时引擎中（无持久化状态）- 创建暂停状态
    """
    try:
        paused_in_engine = False
        
        # 1. 尝试从实时引擎暂停
        engine = await get_realtime_engine()
        if engine and strategy_id in engine.strategies:
            strategy = engine.strategies[strategy_id]
            if hasattr(strategy, 'is_active'):
                strategy.is_active = False
            paused_in_engine = True
            logger.info(f"策略 {strategy_id} 已从实时引擎暂停")
        
        # 2. 更新持久化存储中的状态
        try:
            from .execution_persistence import load_execution_state, save_execution_state
            state = load_execution_state(strategy_id)
            if state:
                state['status'] = 'paused'
                state['paused_at'] = time.time()
                save_execution_state(strategy_id, state)
                logger.info(f"策略 {strategy_id} 执行状态已更新为 paused")
                return True
            elif paused_in_engine:
                # 策略在实时引擎中但不在持久化存储中，创建新的状态记录
                save_execution_state(strategy_id, {
                    "strategy_id": strategy_id,
                    "name": getattr(engine.strategies.get(strategy_id), 'name', strategy_id),
                    "type": "unknown",
                    "status": "paused",
                    "paused_at": time.time(),
                    "latency": 0,
                    "throughput": 0,
                    "signals_count": 0,
                    "positions_count": 0
                })
                logger.info(f"策略 {strategy_id} 已创建暂停状态（仅存在于实时引擎）")
                return True
            else:
                logger.warning(f"策略 {strategy_id} 在实时引擎和持久化存储中都不存在")
                return False
        except Exception as e:
            logger.error(f"更新持久化存储失败: {e}")
            # 如果已经在引擎中暂停成功，仍然返回True
            if paused_in_engine:
                return True
            return False
            
    except Exception as e:
        logger.error(f"暂停策略失败: {e}")
        return False

