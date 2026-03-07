#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一策略服务实现
Unified Strategy Service Implementation

基于业务流程驱动架构，实现统一的策略服务，整合策略框架和回测框架的功能。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import pandas as pd
from ..interfaces.strategy_interfaces import (
    IStrategyService, StrategyConfig, StrategySignal,
    StrategyResult, StrategyStatus, StrategyType
)
from strategy.interfaces.backtest_interfaces import (
    IBacktestService
)
from strategy.interfaces.optimization_interfaces import (
    IOptimizationService
)
from strategy.interfaces.monitoring_interfaces import (
    IMonitoringService, MetricData
)
from strategy.core.integration.business_adapters import get_unified_adapter_factory
from strategy.core.high_concurrency_optimizer import get_high_concurrency_optimizer
from ...core.interfaces.ml_strategy_interfaces import (
    IStrategyService, IStrategyDataPreparation
)

logger = logging.getLogger(__name__)


class UnifiedStrategyService(IStrategyService, IStrategyDataPreparation):

    """
    统一策略服务
    Unified Strategy Service

    整合策略管理、回测、优化、监控等功能的统一服务实现。
    """

    def __init__(self):
        """初始化统一策略服务"""
        self.adapter_factory = get_unified_adapter_factory()

        # 获取基础设施服务
        self.config_adapter = self.adapter_factory.get_adapter("config")
        self.cache_adapter = self.adapter_factory.get_adapter("cache")
        self.monitoring_adapter = self.adapter_factory.get_adapter("monitoring")
        self.event_bus_adapter = self.adapter_factory.get_adapter("event_bus")

        # 获取高并发优化器
        self.concurrency_optimizer = get_high_concurrency_optimizer()

        # 策略存储
        self.strategies: Dict[str, StrategyConfig] = {}
        self.strategy_states: Dict[str, Dict[str, Any]] = {}

        # 依赖服务
        self.backtest_service: Optional[IBacktestService] = None
        self.optimization_service: Optional[IOptimizationService] = None
        self.monitoring_service: Optional[IMonitoringService] = None

        logger.info("统一策略服务已初始化")

    def register_backtest_service(self, backtest_service: IBacktestService):
        """注册回测服务"""
        self.backtest_service = backtest_service
        logger.info("回测服务已注册")

    def register_optimization_service(self, optimization_service: IOptimizationService):
        """注册优化服务"""
        self.optimization_service = optimization_service
        logger.info("优化服务已注册")

    def register_monitoring_service(self, monitoring_service: IMonitoringService):
        """注册监控服务"""
        self.monitoring_service = monitoring_service
        logger.info("监控服务已注册")

    def create_strategy(self, config: StrategyConfig) -> bool:
        """
        创建策略

        Args:
            config: 策略配置

        Returns:
            bool: 创建是否成功
        """
        try:
            if config.strategy_id in self.strategies:
                logger.warning(f"策略已存在: {config.strategy_id}")
                return False

            # 验证配置
            if not self._validate_strategy_config(config):
                logger.error(f"策略配置验证失败: {config.strategy_id}")
                return False

            # 保存配置
            self.strategies[config.strategy_id] = config
            self.strategy_states[config.strategy_id] = {
                "status": StrategyStatus.CREATED,
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }

            # 持久化存储
            success = self._persist_strategy_config(config)
            if not success:
                logger.error(f"策略配置持久化失败: {config.strategy_id}")
                return False

            # 发布事件
            self._publish_event("strategy_created", {
                "strategy_id": config.strategy_id,
                "strategy_type": config.strategy_type.value,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略创建成功: {config.strategy_id}")
            return True

        except Exception as e:
            logger.error(f"策略创建异常: {e}")
            return False

    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]:
        """
        获取策略配置

        Args:
            strategy_id: 策略ID

        Returns:
            Optional[StrategyConfig]: 策略配置
        """
        return self.strategies.get(strategy_id)

    def update_strategy(self, strategy_id: str, config: StrategyConfig) -> bool:
        """
        更新策略配置

        Args:
            strategy_id: 策略ID
            config: 新的策略配置

        Returns:
            bool: 更新是否成功
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"策略不存在: {strategy_id}")
                return False

            # 验证配置
            if not self._validate_strategy_config(config):
                logger.error(f"策略配置验证失败: {strategy_id}")
                return False

            # 更新配置
            self.strategies[strategy_id] = config
            self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 持久化存储
            success = self._persist_strategy_config(config)
            if not success:
                logger.error(f"策略配置更新持久化失败: {strategy_id}")
                return False

            # 发布事件
            self._publish_event("strategy_updated", {
                "strategy_id": strategy_id,
                "strategy_type": config.strategy_type.value,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略更新成功: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"策略更新异常: {e}")
            return False

    def delete_strategy(self, strategy_id: str) -> bool:
        """
        删除策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 删除是否成功
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"策略不存在: {strategy_id}")
                return False

            # 删除配置
            del self.strategies[strategy_id]
            del self.strategy_states[strategy_id]

            # 删除持久化存储
            success = self._delete_strategy_config(strategy_id)
            if not success:
                logger.error(f"策略配置删除持久化失败: {strategy_id}")
                return False

            # 发布事件
            self._publish_event("strategy_deleted", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略删除成功: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"策略删除异常: {e}")
            return False

    def list_strategies(self, strategy_type: Optional[StrategyType] = None,


                        tags: Optional[List[str]] = None) -> List[StrategyConfig]:
        """
        列出策略

        Args:
            strategy_type: 策略类型过滤器
            tags: 标签过滤器

        Returns:
            List[StrategyConfig]: 策略配置列表
        """
        strategies = list(self.strategies.values())

        # 按类型过滤
        if strategy_type:
            strategies = [s for s in strategies if s.strategy_type == strategy_type]

        # 按标签过滤
        if tags:
            strategies = [s for s in strategies if any(tag in s.tags for tag in tags)]

        return strategies

    def batch_execute_strategies(self, strategy_requests: List[Dict[str, Any]],


                                 market_data: Dict[str, Any]) -> List[StrategyResult]:
        """
        批量执行多个策略 - 高并发优化

        Args:
            strategy_requests: 策略执行请求列表，每个包含strategy_id和execution_context
            market_data: 市场数据

        Returns:
            List[StrategyResult]: 策略执行结果列表
        """
        try:
            # 验证策略是否存在
            valid_requests = []
            for request in strategy_requests:
                strategy_id = request.get('strategy_id')
                if strategy_id and strategy_id in self.strategies:
                    valid_requests.append(request)
                else:
                    logger.warning(f"跳过无效策略: {strategy_id}")

            if not valid_requests:
                logger.warning("没有有效的策略执行请求")
                return []

            # 使用高并发优化器批量执行
            strategy_configs = []
            for request in valid_requests:
                strategy_id = request['strategy_id']
                config = self.strategies[strategy_id]
                strategy_configs.append({
                    'id': strategy_id,
                    'type': config.strategy_type.value,
                    'params': config.parameters
                })

            # 提交批量任务
            batch_results = self.concurrency_optimizer.optimize_strategy_execution(strategy_configs)

            # 处理结果
            results = []
            for i, result in enumerate(batch_results):
                if result['status'] == 'submitted':
                    # 异步执行，立即返回提交状态
                    request = valid_requests[i]
                    strategy_id = request['strategy_id']
                    results.append(StrategyResult(
                        strategy_id=strategy_id,
                        signals=[],
                        performance_metrics={},
                        execution_time=0.0,
                        timestamp=datetime.now(),
                        error_message="异步执行中"
                    ))
                else:
                    # 处理失败的情况
                    results.append(StrategyResult(
                        strategy_id=valid_requests[i]['strategy_id'],
                        signals=[],
                        performance_metrics={},
                        execution_time=0.0,
                        timestamp=datetime.now(),
                        error_message=f"批量提交失败: {result.get('error', '未知错误')}"
                    ))

            logger.info(f"批量策略执行已提交: {len(results)} 个策略")
            return results

        except Exception as e:
            logger.error(f"批量策略执行异常: {e}")
            return [StrategyResult(
                strategy_id="batch_error",
                signals=[],
                performance_metrics={},
                execution_time=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )]

    def execute_strategy(self, strategy_id: str, market_data: Dict[str, Any],


                         execution_context: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """
        执行策略

        Args:
            strategy_id: 策略ID
            market_data: 市场数据
            execution_context: 执行上下文

        Returns:
            StrategyResult: 策略执行结果
        """
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"策略不存在: {strategy_id}")

            strategy_config = self.strategies[strategy_id]

            # 更新策略状态
            self.strategy_states[strategy_id]["status"] = StrategyStatus.RUNNING
            self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 执行策略逻辑
            start_time = datetime.now()
            signals = self._execute_strategy_logic(strategy_config, market_data, execution_context)
            execution_time = (datetime.now() - start_time).total_seconds()

            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(signals, market_data)

            # 创建执行结果
            result = StrategyResult(
                strategy_id=strategy_id,
                signals=signals,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                timestamp=datetime.now()
            )

            # 更新策略状态
            self.strategy_states[strategy_id]["status"] = StrategyStatus.RUNNING
            self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 记录性能指标
            if self.monitoring_service:
                self._record_performance_metrics(strategy_id, performance_metrics)

            # 发布事件
            self._publish_event("strategy_executed", {
                "strategy_id": strategy_id,
                "signal_count": len(signals),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略执行完成: {strategy_id}, 信号数量: {len(signals)}")
            return result

        except Exception as e:
            logger.error(f"策略执行异常: {strategy_id}, 错误: {e}")

            # 更新策略状态为错误
            if strategy_id in self.strategy_states:
                self.strategy_states[strategy_id]["status"] = StrategyStatus.ERROR
                self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 返回错误结果
            return StrategyResult(
                strategy_id=strategy_id,
                signals=[],
                performance_metrics={},
                execution_time=0.0,
                error_message=str(e),
                timestamp=datetime.now()
            )

    def _execute_strategy_logic(self, config: StrategyConfig, market_data: Dict[str, Any],


                                execution_context: Optional[Dict[str, Any]] = None) -> List[StrategySignal]:
        """
        执行策略逻辑

        Args:
            config: 策略配置
            market_data: 市场数据
            execution_context: 执行上下文

        Returns:
            List[StrategySignal]: 策略信号列表
        """
        signals = []

        try:
            if config.strategy_type == StrategyType.MOMENTUM:
                signals = self._execute_momentum_strategy(config, market_data)

            elif config.strategy_type == StrategyType.MEAN_REVERSION:
                signals = self._execute_mean_reversion_strategy(config, market_data)

            elif config.strategy_type == StrategyType.ARBITRAGE:
                signals = self._execute_arbitrage_strategy(config, market_data)

            elif config.strategy_type == StrategyType.MACHINE_LEARNING:
                signals = self._execute_ml_strategy(config, market_data)

            elif config.strategy_type == StrategyType.REINFORCEMENT_LEARNING:
                signals = self._execute_rl_strategy(config, market_data)

            else:
                logger.warning(f"不支持的策略类型: {config.strategy_type}")

        except Exception as e:
            logger.error(f"策略逻辑执行异常: {config.strategy_id}, 错误: {e}")

        return signals

    def _execute_momentum_strategy(self, config: StrategyConfig,


                                   market_data: Dict[str, Any]) -> List[StrategySignal]:
        """执行动量策略"""
        signals = []
        lookback_period = config.parameters.get('lookback_period', 20)
        momentum_threshold = config.parameters.get('momentum_threshold', 0.05)

        for symbol, data in market_data.items():
            if len(data) < lookback_period:
                continue

            # 计算动量
            prices = [d.get('close', d.get('price', 0)) for d in data[-lookback_period:]]
            if len(prices) < 2:
                continue

            momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0

            if momentum > momentum_threshold:
                signals.append(StrategySignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=config.parameters.get('position_size', 100),
                    confidence=float(abs(momentum)),
                    price=prices[-1],
                    strategy_id=config.strategy_id,
                    metadata={'momentum': momentum, 'lookback': lookback_period}
                ))
            elif momentum < -momentum_threshold:
                signals.append(StrategySignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=config.parameters.get('position_size', 100),
                    confidence=float(abs(momentum)),
                    price=prices[-1],
                    strategy_id=config.strategy_id,
                    metadata={'momentum': momentum, 'lookback': lookback_period}
                ))

        return signals

    def _execute_mean_reversion_strategy(self, config: StrategyConfig,


                                         market_data: Dict[str, Any]) -> List[StrategySignal]:
        """执行均值回归策略"""
        signals = []
        lookback_period = config.parameters.get('lookback_period', 50)
        std_threshold = config.parameters.get('std_threshold', 2.0)

        for symbol, data in market_data.items():
            if len(data) < lookback_period:
                continue

            # 计算价格和均值
            prices = [d.get('close', d.get('price', 0)) for d in data[-lookback_period:]]
            if not prices:
                continue

            current_price = prices[-1]
            mean_price = sum(prices) / len(prices)
            std_price = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5

            if std_price == 0:
                continue

            z_score = (current_price - mean_price) / std_price

            if z_score > std_threshold:
                signals.append(StrategySignal(
                    symbol=symbol,
                    action='SELL',
                    quantity=config.parameters.get('position_size', 100),
                    confidence=float(abs(z_score) / (std_threshold * 2)),
                    price=current_price,
                    strategy_id=config.strategy_id,
                    metadata={'z_score': z_score, 'mean': mean_price, 'std': std_price}
                ))
            elif z_score < -std_threshold:
                signals.append(StrategySignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=config.parameters.get('position_size', 100),
                    confidence=float(abs(z_score) / (std_threshold * 2)),
                    price=current_price,
                    strategy_id=config.strategy_id,
                    metadata={'z_score': z_score, 'mean': mean_price, 'std': std_price}
                ))

        return signals

    def _execute_arbitrage_strategy(self, config: StrategyConfig,


                                    market_data: Dict[str, Any]) -> List[StrategySignal]:
        """执行套利策略"""
        signals = []
        min_spread = config.parameters.get('min_spread', 0.01)

        # 这里简化实现，实际套利策略需要更复杂的逻辑
        # 比较不同市场的价格差异

        return signals

    def _execute_ml_strategy(self, config: StrategyConfig,


                             market_data: Dict[str, Any]) -> List[StrategySignal]:
        """执行机器学习策略"""
        signals = []

        # 这里需要集成ML模型推理
        # 暂时返回空信号

        return signals

    def _execute_rl_strategy(self, config: StrategyConfig,


                             market_data: Dict[str, Any]) -> List[StrategySignal]:
        """执行强化学习策略"""
        signals = []

        # 这里需要集成RL模型推理
        # 暂时返回空信号

        return signals

    def _calculate_performance_metrics(self, signals: List[StrategySignal],


                                       market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算性能指标"""
        if not signals:
            return {}

        # 基础指标计算
        buy_signals = [s for s in signals if s.action == 'BUY']
        sell_signals = [s for s in signals if s.action == 'SELL']

        metrics = {
            'total_signals': len(signals),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': sum(s.confidence for s in signals) / len(signals) if signals else 0,
            'total_volume': sum(s.quantity for s in signals)
        }

        return metrics

    def _record_performance_metrics(self, strategy_id: str, metrics: Dict[str, float]):
        """记录性能指标"""
        if not self.monitoring_service:
            return

        try:
            for metric_name, value in metrics.items():
                metric_data = MetricData(
                    metric_name=f"strategy_{metric_name}",
                    value=value,
                    timestamp=datetime.now(),
                    strategy_id=strategy_id,
                    metric_type="performance"
                )
                self.monitoring_service.record_metric(metric_data)
        except Exception as e:
            logger.error(f"性能指标记录异常: {e}")

    def start_strategy(self, strategy_id: str, execution_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        启动策略

        Args:
            strategy_id: 策略ID
            execution_params: 执行参数

        Returns:
            bool: 启动是否成功
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"策略不存在: {strategy_id}")
                return False

            self.strategy_states[strategy_id]["status"] = StrategyStatus.RUNNING
            self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 发布事件
            self._publish_event("strategy_started", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略启动成功: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"策略启动异常: {e}")
            return False

    def stop_strategy(self, strategy_id: str) -> bool:
        """
        停止策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 停止是否成功
        """
        try:
            if strategy_id not in self.strategies:
                logger.warning(f"策略不存在: {strategy_id}")
                return False

            self.strategy_states[strategy_id]["status"] = StrategyStatus.STOPPED
            self.strategy_states[strategy_id]["last_updated"] = datetime.now()

            # 发布事件
            self._publish_event("strategy_stopped", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略停止成功: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"策略停止异常: {e}")
            return False

    def get_strategy_status(self, strategy_id: str) -> StrategyStatus:
        """
        获取策略状态

        Args:
            strategy_id: 策略ID

        Returns:
            StrategyStatus: 策略状态
        """
        if strategy_id in self.strategy_states:
            return self.strategy_states[strategy_id]["status"]
        return StrategyStatus.CREATED

    def get_strategy_performance(self, strategy_id: str,


                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        获取策略性能

        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[str, Any]: 性能数据
        """
        # 这里可以实现更复杂的性能分析逻辑
        # 暂时返回基础性能指标

        if strategy_id not in self.strategies:
            return {}

        # 从监控服务获取历史指标
        if self.monitoring_service:
            try:
                metrics = self.monitoring_service.get_current_metrics(strategy_id)
                return {k: v.value for k, v in metrics.items()}
            except Exception as e:
                logger.error(f"获取策略性能异常: {e}")

        return {}

    def _validate_strategy_config(self, config: StrategyConfig) -> bool:
        """验证策略配置"""
        if not config.strategy_id or not config.strategy_name:
            return False

        if not isinstance(config.strategy_type, StrategyType):
            return False

        if not config.parameters:
            return False

        return True

    def _persist_strategy_config(self, config: StrategyConfig) -> bool:
        """持久化策略配置"""
        try:
            # 这里实现持久化逻辑
            # 可以使用config_adapter或直接数据库操作

            if self.config_adapter:
                # 使用配置适配器
                config_key = f"strategy_config_{config.strategy_id}"
                config_data = {
                    "strategy_id": config.strategy_id,
                    "strategy_name": config.strategy_name,
                    "strategy_type": config.strategy_type.value,
                    "parameters": config.parameters,
                    "risk_limits": config.risk_limits,
                    "enabled": config.enabled,
                    "description": config.description,
                    "created_at": config.created_at.isoformat() if config.created_at else None,
                    "updated_at": config.updated_at.isoformat() if config.updated_at else None,
                    "version": config.version,
                    "author": config.author,
                    "tags": config.tags
                }
                return self.config_adapter.set_config(config_key, config_data)

            # 如果没有配置适配器，直接存储在内存中
            return True

        except Exception as e:
            logger.error(f"策略配置持久化异常: {e}")
            return False

    def _delete_strategy_config(self, strategy_id: str) -> bool:
        """删除策略配置"""
        try:
            if self.config_adapter:
                config_key = f"strategy_config_{strategy_id}"
                return self.config_adapter.delete_config(config_key)
            return True
        except Exception as e:
            logger.error(f"策略配置删除异常: {e}")
            return False

    def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """发布事件"""
        try:
            if self.event_bus_adapter:
                event = {
                    "event_type": f"strategy_{event_type}",
                    "data": event_data,
                    "source": "unified_strategy_service",
                    "timestamp": datetime.now().isoformat()
                }
                self.event_bus_adapter.publish_event(event)
        except Exception as e:
            logger.error(f"事件发布异常: {e}")

    # 实现IStrategyDataPreparation接口的方法

    def prepare_market_data(self, market_data: pd.DataFrame,


                            preparation_config: Dict[str, Any]) -> pd.DataFrame:
        """
        准备市场数据

        Args:
            market_data: 原始市场数据
            preparation_config: 数据准备配置

        Returns:
            pd.DataFrame: 准备后的市场数据
        """
        try:
            prepared_data = market_data.copy()

            # 数据验证
            validation_result = self.validate_market_data(prepared_data)
            if not validation_result['valid']:
                logger.warning(f"市场数据验证失败: {validation_result['issues']}")
                prepared_data = self.handle_data_anomalies(prepared_data, {})

            # 数据清洗
            if preparation_config.get('clean_data', True):
                prepared_data = self._clean_market_data(prepared_data)

            # 数据转换
            if preparation_config.get('transform_data', True):
                prepared_data = self._transform_market_data(prepared_data)

            # 数据标准化
            if preparation_config.get('normalize_data', False):
                prepared_data = self._normalize_market_data(prepared_data)

            logger.info("市场数据准备完成")
            return prepared_data

        except Exception as e:
            logger.error(f"市场数据准备异常: {e}")
            return market_data

    def validate_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证市场数据质量

        Args:
            market_data: 市场数据

        Returns:
            Dict[str, Any]: 验证结果
        """
        issues = []

        # 检查必需列
        required_columns = ['timestamp', 'symbol', 'close', 'volume']
        for col in required_columns:
            if col not in market_data.columns:
                issues.append(f"缺少必需列: {col}")

        # 检查数据类型
        if 'timestamp' in market_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
                issues.append("timestamp列类型不正确")

        # 检查缺失值
        missing_data = market_data.isnull().sum()
        if missing_data.any():
            issues.append(f"存在缺失值: {missing_data[missing_data > 0].to_dict()}")

        # 检查数据范围
        if 'close' in market_data.columns:
            negative_prices = (market_data['close'] <= 0).sum()
        if negative_prices > 0:
            issues.append(f"存在{negative_prices}个非正价格")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_quality_score': max(0, 100 - len(issues) * 10)
        }

    def handle_data_anomalies(self, market_data: pd.DataFrame,


                              anomaly_config: Dict[str, Any]) -> pd.DataFrame:
        """
        处理数据异常

        Args:
            market_data: 市场数据
            anomaly_config: 异常处理配置

        Returns:
            pd.DataFrame: 处理后的市场数据
        """
        try:
            cleaned_data = market_data.copy()

            # 处理缺失值
            if anomaly_config.get('handle_missing', True):
                cleaned_data = cleaned_data.fillna(method='ffill')  # 前向填充
                cleaned_data = cleaned_data.fillna(method='bfill')  # 后向填充

            # 处理异常值
            if anomaly_config.get('handle_outliers', True):
                for col in ['close', 'volume']:
                    if col in cleaned_data.columns:
                        cleaned_data = self._remove_outliers(cleaned_data, col)

            # 处理重复数据
            if anomaly_config.get('remove_duplicates', True):
                cleaned_data = cleaned_data.drop_duplicates()

            logger.info("数据异常处理完成")
            return cleaned_data

        except Exception as e:
            logger.error(f"数据异常处理异常: {e}")
            return market_data

    # 辅助方法

    def _clean_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗市场数据"""
        # 移除全为空的行
        cleaned = data.dropna(how='all')

        # 移除重复行
        cleaned = cleaned.drop_duplicates()

        return cleaned

    def _transform_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换市场数据"""
        transformed = data.copy()

        # 确保时间戳格式
        if 'timestamp' in transformed.columns:
            transformed['timestamp'] = pd.to_datetime(transformed['timestamp'])

        # 计算收益率
        if 'close' in transformed.columns:
            transformed['returns'] = transformed['close'].pct_change()

        # 计算成交量变化
        if 'volume' in transformed.columns:
            transformed['volume_change'] = transformed['volume'].pct_change()

        return transformed

    def _normalize_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化市场数据"""
        normalized = data.copy()

        # 对价格数据进行标准化
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in normalized.columns:
                normalized[f'{col}_normalized'] = (
                    normalized[col] - normalized[col].rolling(20).mean()
                ) / normalized[col].rolling(20).std()

        # 对成交量数据进行标准化
        if 'volume' in normalized.columns:
            normalized['volume_normalized'] = (
                normalized['volume'] - normalized['volume'].rolling(20).mean()
            ) / normalized['volume'].rolling(20).std()

        return normalized

    def _remove_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """移除异常值"""
        if column not in data.columns:
            return data

        # 使用IQR方法检测异常值
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 将异常值替换为边界值
        cleaned_data = data.copy()
        cleaned_data[column] = cleaned_data[column].clip(lower_bound, upper_bound)

        return cleaned_data


# 导出类
__all__ = [
    'UnifiedStrategyService'
]
