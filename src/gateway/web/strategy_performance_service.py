"""
策略性能评估服务层
封装实际的策略回测和性能分析组件，为API提供统一接口
符合量化交易系统安全要求
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# 量化交易系统安全要求：计算结果验证函数
def validate_metrics(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证指标计算结果合理性
    
    Returns:
        (is_valid, warnings): 是否有效，警告信息列表
    """
    warnings = []
    
    # 检查夏普比率
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 10:
        warnings.append(f"夏普比率异常: {sharpe:.2f}（超过10）")
    elif sharpe < -5:
        warnings.append(f"夏普比率异常: {sharpe:.2f}（低于-5）")
    
    # 检查最大回撤
    max_dd = metrics.get('max_drawdown', 0)
    if max_dd > 0.99:
        warnings.append(f"最大回撤异常: {max_dd:.2%}（超过99%）")
    
    # 检查年化收益
    annual_return = metrics.get('annual_return', 0)
    if annual_return > 10:  # 超过1000%
        warnings.append(f"年化收益异常: {annual_return:.2%}（超过1000%）")
    elif annual_return < -1:  # 低于-100%
        warnings.append(f"年化收益异常: {annual_return:.2%}（低于-100%）")
    
    # 检查胜率
    win_rate = metrics.get('win_rate', 0)
    if win_rate > 1.0:  # 胜率不能超过100%
        warnings.append(f"胜率异常: {win_rate:.2%}（超过100%）")
    
    return len(warnings) == 0, warnings

# 导入Redis缓存模块
try:
    from .redis_cache import (
        get_strategy_comparison_cache,
        set_strategy_comparison_cache,
        get_performance_metrics_cache,
        set_performance_metrics_cache,
        get_strategy_detail_cache,
        set_strategy_detail_cache,
        clear_strategy_caches
    )
    REDIS_AVAILABLE = True
    logger.info("Redis缓存模块导入成功")
except Exception as e:
    logger.warning(f"Redis缓存模块导入失败: {e}")
    REDIS_AVAILABLE = False
    
    # 定义空函数作为降级
    def get_strategy_comparison_cache(): return None
    def set_strategy_comparison_cache(data): return False
    def get_performance_metrics_cache(): return None
    def set_performance_metrics_cache(data): return False
    def get_strategy_detail_cache(strategy_id): return None
    def set_strategy_detail_cache(strategy_id, data): return False
    def clear_strategy_caches(): return False

# 导入InfluxDB持久化模块
try:
    from .influxdb_persistence import (
        save_performance_metrics,
        batch_save_performance_metrics,
        query_performance_metrics,
        query_aggregate_metrics,
        test_influxdb_connection
    )
    INFLUXDB_AVAILABLE = True
    logger.info("InfluxDB持久化模块导入成功")
    # 测试InfluxDB连接
    if test_influxdb_connection():
        logger.info("InfluxDB连接测试成功")
    else:
        logger.warning("InfluxDB连接测试失败")
except Exception as e:
    logger.warning(f"InfluxDB持久化模块导入失败: {e}")
    INFLUXDB_AVAILABLE = False
    
    # 定义空函数作为降级
    def save_performance_metrics(strategy_id, metrics): return False
    def batch_save_performance_metrics(metrics_list): return {"success": False, "total_processed": 0, "success_count": 0, "failed_count": 0}
    def query_performance_metrics(strategy_id, start_time=None, end_time=None, limit=1000): return []
    def query_aggregate_metrics(start_time=None, end_time=None): return {}

# 确保backtest_results表存在适当的索引
try:
    from .postgresql_persistence import ensure_backtest_table_indexes
    # 在模块初始化时创建索引
    if ensure_backtest_table_indexes():
        logger.info("backtest_results表索引创建成功")
    else:
        logger.info("backtest_results表索引创建跳过或失败")
except Exception as e:
    logger.warning(f"创建backtest_results表索引失败: {e}")

# 导入策略层组件
try:
    from src.strategy.backtest.backtest_engine import BacktestEngine
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入回测引擎: {e}")
    BACKTEST_ENGINE_AVAILABLE = False

try:
    from src.strategy.backtest.analyzer import PerformanceAnalyzer
    PERFORMANCE_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入性能分析器: {e}")
    PERFORMANCE_ANALYZER_AVAILABLE = False


# 单例实例
_backtest_engine: Optional[Any] = None
_performance_analyzer: Optional[Any] = None


def get_backtest_engine() -> Optional[Any]:
    """获取回测引擎实例"""
    global _backtest_engine
    if _backtest_engine is None and BACKTEST_ENGINE_AVAILABLE:
        try:
            _backtest_engine = BacktestEngine()
            logger.info("回测引擎初始化成功")
        except Exception as e:
            logger.error(f"初始化回测引擎失败: {e}")
    return _backtest_engine


def get_performance_analyzer() -> Optional[Any]:
    """获取性能分析器实例"""
    global _performance_analyzer
    if _performance_analyzer is None and PERFORMANCE_ANALYZER_AVAILABLE:
        try:
            _performance_analyzer = PerformanceAnalyzer()
            logger.info("性能分析器初始化成功")
        except Exception as e:
            logger.error(f"初始化性能分析器失败: {e}")
    return _performance_analyzer


# ==================== 策略性能对比服务 ====================

def get_strategy_comparison(cache_only: bool = False) -> List[Dict[str, Any]]:
    """
    获取策略对比数据 - 优先从缓存加载，然后从回测持久化存储加载，不使用模拟数据
    
    Args:
        cache_only: 是否只从缓存加载数据，不进行实时计算
    """
    # 优先从缓存加载
    cached_data = get_strategy_comparison_cache()
    if cached_data:
        logger.debug("从Redis缓存加载策略对比数据")
        return cached_data
    
    # 如果设置了只从缓存加载且缓存为空，返回空列表
    if cache_only:
        return []
    
    strategies = []
    
    # 优先从回测持久化存储加载
    try:
        from .backtest_persistence import list_backtest_results
        
        backtest_results = list_backtest_results(limit=100)
        
        # 按策略ID分组，取最新的回测结果
        strategy_backtests = {}
        for backtest in backtest_results:
            strategy_id = backtest.get('strategy_id')
            if not strategy_id:
                continue
            
            # 如果策略ID不存在或当前回测更新，则使用当前回测结果
            if strategy_id not in strategy_backtests:
                strategy_backtests[strategy_id] = backtest
            else:
                # 比较时间戳，使用最新的
                current_time = backtest.get('created_at') or backtest.get('saved_at', 0)
                existing_time = strategy_backtests[strategy_id].get('created_at') or strategy_backtests[strategy_id].get('saved_at', 0)
                if current_time > existing_time:
                    strategy_backtests[strategy_id] = backtest
        
        # 尝试从策略配置中获取策略名称
        strategy_names = {}
        try:
            from .strategy_routes import load_strategy_conceptions
            all_strategies = load_strategy_conceptions()
            strategy_names = {s.get('id'): s.get('name', s.get('id')) for s in all_strategies}
        except Exception as e:
            logger.debug(f"加载策略配置失败: {e}")
        
        # 转换为策略对比格式
        for strategy_id, backtest in strategy_backtests.items():
            strategies.append({
                "id": strategy_id,
                "name": strategy_names.get(strategy_id, backtest.get('strategy_id', strategy_id)),
                "type": backtest.get('strategy_type', '未知'),
                "status": "active" if backtest.get('status') == 'completed' else "inactive",
                "total_return": backtest.get('total_return', 0),
                "sharpe_ratio": backtest.get('sharpe_ratio', 0),
                "max_drawdown": backtest.get('max_drawdown', 0),
                "annual_return": backtest.get('annualized_return', backtest.get('annual_return', 0)),
                "win_rate": backtest.get('win_rate', 0),
                "backtest_date": backtest.get('end_date', backtest.get('created_at', ''))
            })
        
        logger.debug(f"从回测持久化存储加载了 {len(strategies)} 个策略对比数据")
    except Exception as e:
        logger.debug(f"从回测持久化存储加载策略对比数据失败: {e}")
    
    # 如果持久化存储中没有数据，尝试从回测引擎获取
    if not strategies:
        backtest_engine = get_backtest_engine()
        performance_analyzer = get_performance_analyzer()
        
        if backtest_engine:
            try:
                # 尝试获取已完成的回测结果
                if hasattr(backtest_engine, 'get_completed_backtests'):
                    completed_backtests = backtest_engine.get_completed_backtests()
                    for backtest in completed_backtests:
                        strategy_id = backtest.get('strategy_id', '')
                        strategy_name = backtest.get('strategy_name', strategy_id)
                        
                        # 获取回测结果
                        result = backtest.get('result', {})
                        
                        strategies.append({
                            "id": strategy_id,
                            "name": strategy_name,
                            "type": backtest.get('strategy_type', '未知'),
                            "status": "active" if backtest.get('status') == 'completed' else "inactive",
                            "total_return": result.get('total_return', 0),
                            "sharpe_ratio": result.get('sharpe_ratio', 0),
                            "max_drawdown": result.get('max_drawdown', 0),
                            "annual_return": result.get('annual_return', 0),
                            "win_rate": result.get('win_rate', 0),
                            "backtest_date": backtest.get('end_date', '')
                        })
                elif hasattr(backtest_engine, 'list_strategies'):
                    # 尝试获取策略列表
                    strategy_list = backtest_engine.list_strategies()
                    for strategy_info in strategy_list:
                        strategies.append({
                            "id": strategy_info.get('id', ''),
                            "name": strategy_info.get('name', ''),
                            "type": strategy_info.get('type', '未知'),
                            "status": strategy_info.get('status', 'inactive'),
                            "total_return": 0,
                            "sharpe_ratio": 0,
                            "max_drawdown": 0,
                            "annual_return": 0,
                            "win_rate": 0
                        })
            except Exception as e:
                logger.debug(f"从回测引擎获取策略对比数据失败: {e}")
        
        # 如果性能分析器可用，尝试获取性能分析结果
        if performance_analyzer and not strategies:
            try:
                if hasattr(performance_analyzer, 'get_all_strategies_performance'):
                    performance_data = performance_analyzer.get_all_strategies_performance()
                    for perf in performance_data:
                        strategies.append({
                            "id": perf.get('strategy_id', ''),
                            "name": perf.get('strategy_name', ''),
                            "type": perf.get('strategy_type', '未知'),
                            "status": "active",
                            "total_return": perf.get('total_return', 0),
                            "sharpe_ratio": perf.get('sharpe_ratio', 0),
                            "max_drawdown": perf.get('max_drawdown', 0),
                            "annual_return": perf.get('annual_return', 0),
                            "win_rate": perf.get('win_rate', 0)
                        })
            except Exception as e:
                logger.debug(f"从性能分析器获取策略对比数据失败: {e}")
    
    # 缓存结果
    if strategies:
        set_strategy_comparison_cache(strategies)
        
        # 保存性能指标到InfluxDB
        if INFLUXDB_AVAILABLE:
            try:
                metrics_list = []
                for strategy in strategies:
                    strategy_id = strategy.get('id')
                    if not strategy_id:
                        continue
                    
                    # 提取数值型性能指标
                    metrics = {
                        'strategy_id': strategy_id,
                        'total_return': strategy.get('total_return', 0),
                        'sharpe_ratio': strategy.get('sharpe_ratio', 0),
                        'max_drawdown': strategy.get('max_drawdown', 0),
                        'annual_return': strategy.get('annual_return', 0),
                        'win_rate': strategy.get('win_rate', 0)
                    }
                    metrics_list.append(metrics)
                
                # 批量保存
                if metrics_list:
                    batch_save_performance_metrics(metrics_list)
            except Exception as e:
                logger.warning(f"保存性能指标到InfluxDB失败: {e}")
    
    # 量化交易系统要求：如果没有真实数据，返回空列表而不是模拟数据
    return strategies


async def get_strategy_comparison_data(cache_only: bool = False) -> List[Dict[str, Any]]:
    """
    获取策略对比数据的异步接口，用于WebSocket广播
    
    Args:
        cache_only: 是否只从缓存加载数据，不进行实时计算
    """
    return get_strategy_comparison(cache_only=cache_only)


def get_performance_metrics(cache_only: bool = False) -> Dict[str, Any]:
    """
    获取性能指标 - 优先从缓存加载，然后从回测持久化存储加载，不使用模拟数据
    
    Args:
        cache_only: 是否只从缓存加载数据，不进行实时计算
    """
    # 优先从缓存加载
    cached_data = get_performance_metrics_cache()
    if cached_data:
        logger.debug("从Redis缓存加载性能指标数据")
        return cached_data
    
    # 如果设置了只从缓存加载且缓存为空，返回空结果
    if cache_only:
        return {
            "metrics": {},
            "return_curves": [],
            "risk_return": [],
            "rankings": []
        }
    
    strategies = get_strategy_comparison(cache_only=False)
    
    if not strategies:
        result = {
            "metrics": {},
            "return_curves": [],
            "risk_return": [],
            "rankings": []
        }
        # 缓存空结果
        set_performance_metrics_cache(result)
        return result
    
    # 计算平均指标
    sharpe_ratios = [s.get('sharpe_ratio', 0) for s in strategies if s.get('sharpe_ratio')]
    max_drawdowns = [s.get('max_drawdown', 0) for s in strategies if s.get('max_drawdown')]
    annual_returns = [s.get('annual_return', 0) for s in strategies if s.get('annual_return')]
    win_rates = [s.get('win_rate', 0) for s in strategies if s.get('win_rate')]
    
    # 构建收益曲线数据（从回测结果的equity_curve）
    return_curves = []
    try:
        from .backtest_persistence import list_backtest_results
        
        backtest_results = list_backtest_results(limit=10)
        for backtest in backtest_results[:5]:  # 最多取5个策略
            equity_curve = backtest.get('equity_curve', [])
            if equity_curve:
                strategy_id = backtest.get('strategy_id', '')
                return_curves.append({
                    "name": strategy_id,
                    "labels": [f"Day {i}" for i in range(len(equity_curve))],
                    "values": equity_curve
                })
    except Exception as e:
        logger.debug(f"构建收益曲线数据失败: {e}")
    
    # 构建风险收益散点数据
    risk_return = [
        {
            "risk": s.get('max_drawdown', 0),
            "return": s.get('annual_return', 0)
        }
        for s in strategies if s.get('max_drawdown') and s.get('annual_return')
    ]
    
    # 构建排名数据
    rankings = sorted(strategies, key=lambda x: x.get('total_return', 0), reverse=True)[:10]
    rankings = [
        {
            "name": s.get('name', s.get('id', '')),
            "score": s.get('total_return', 0) * 100
        }
        for s in rankings
    ]
    
    result = {
        "metrics": {
            "avg_sharpe_ratio": sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0.0,
            "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0.0,
            "avg_annual_return": sum(annual_returns) / len(annual_returns) if annual_returns else 0.0,
            "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0
        },
        "return_curves": return_curves,
        "risk_return": risk_return,
        "rankings": rankings
    }
    
    # 缓存结果
    set_performance_metrics_cache(result)
    
    return result


async def get_performance_metrics_data(cache_only: bool = False) -> Dict[str, Any]:
    """
    获取性能指标的异步接口，用于WebSocket广播
    
    Args:
        cache_only: 是否只从缓存加载数据，不进行实时计算
    """
    return get_performance_metrics(cache_only=cache_only)


# 注：已移除模拟数据函数 _get_mock_strategies() 和 _get_mock_performance_metrics()
# 量化交易系统要求：不使用模拟数据，所有数据来自真实回测引擎或持久化存储

