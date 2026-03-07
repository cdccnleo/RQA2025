"""
信号验证和监控服务
实现信号质量评分、历史回测验证、准确性统计
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SignalQualityLevel(Enum):
    """信号质量等级"""
    EXCELLENT = "excellent"  # 优秀 (>= 0.8)
    GOOD = "good"            # 良好 (>= 0.6)
    AVERAGE = "average"      # 一般 (>= 0.4)
    POOR = "poor"            # 较差 (>= 0.2)
    BAD = "bad"              # 差 (< 0.2)


@dataclass
class SignalValidationResult:
    """信号验证结果"""
    signal_id: str
    symbol: str
    signal_type: str
    timestamp: datetime
    
    # 质量评分
    overall_score: float = 0.0
    accuracy_score: float = 0.0
    risk_score: float = 0.0
    profit_score: float = 0.0
    consistency_score: float = 0.0
    
    # 质量等级
    quality_level: SignalQualityLevel = SignalQualityLevel.BAD
    
    # 是否有效
    is_valid: bool = False
    
    # 验证详情
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # 回测结果
    backtest_result: Optional[Dict[str, Any]] = None
    
    # 统计信息
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0


class SignalValidator:
    """
    信号验证器
    
    职责：
    1. 信号质量评分
    2. 历史回测验证
    3. 准确性统计
    4. 风险评估
    """
    
    def __init__(self):
        self._signal_history: Dict[str, List[Dict]] = {}
        self._validation_cache: Dict[str, SignalValidationResult] = {}
        self._cache_ttl = 300  # 5分钟缓存
        
        logger.info("信号验证器初始化完成")
    
    def validate_signal(
        self,
        signal: Dict[str, Any],
        historical_data: pd.DataFrame,
        lookback_periods: int = 20
    ) -> SignalValidationResult:
        """
        验证信号质量
        
        Args:
            signal: 信号数据
            historical_data: 历史市场数据
            lookback_periods: 回测周期数
            
        Returns:
            信号验证结果
        """
        try:
            signal_id = signal.get('id', '')
            symbol = signal.get('symbol', '')
            signal_type = signal.get('type', 'unknown')
            timestamp = signal.get('timestamp', datetime.now())
            
            # 检查缓存
            cache_key = f"{signal_id}_{symbol}_{signal_type}"
            if cache_key in self._validation_cache:
                cached_result = self._validation_cache[cache_key]
                if (datetime.now() - cached_result.timestamp).seconds < self._cache_ttl:
                    logger.debug(f"从缓存获取验证结果: {signal_id}")
                    return cached_result
            
            # 创建验证结果
            result = SignalValidationResult(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now()
            )
            
            # 1. 历史回测验证
            backtest_result = self._backtest_signal(signal, historical_data, lookback_periods)
            result.backtest_result = backtest_result
            
            # 2. 计算准确性评分
            result.accuracy_score = self._calculate_accuracy_score(backtest_result)
            
            # 3. 计算风险评分
            result.risk_score = self._calculate_risk_score(backtest_result)
            
            # 4. 计算收益评分
            result.profit_score = self._calculate_profit_score(backtest_result)
            
            # 5. 计算一致性评分
            result.consistency_score = self._calculate_consistency_score(backtest_result)
            
            # 6. 综合评分
            result.overall_score = self._calculate_overall_score(
                result.accuracy_score,
                result.risk_score,
                result.profit_score,
                result.consistency_score
            )
            
            # 7. 确定质量等级
            result.quality_level = self._determine_quality_level(result.overall_score)
            
            # 8. 判断是否有效
            result.is_valid = result.overall_score >= 0.4  # 分数 >= 0.4 认为有效
            
            # 9. 填充统计信息
            if backtest_result:
                result.total_trades = backtest_result.get('total_trades', 0)
                result.winning_trades = backtest_result.get('winning_trades', 0)
                result.losing_trades = backtest_result.get('losing_trades', 0)
                result.win_rate = backtest_result.get('win_rate', 0.0)
                result.avg_profit = backtest_result.get('avg_profit', 0.0)
                result.max_drawdown = backtest_result.get('max_drawdown', 0.0)
                result.sharpe_ratio = backtest_result.get('sharpe_ratio', 0.0)
            
            # 10. 填充验证详情
            result.validation_details = {
                'validation_timestamp': datetime.now().isoformat(),
                'lookback_periods': lookback_periods,
                'historical_data_points': len(historical_data),
                'scoring_weights': {
                    'accuracy': 0.35,
                    'risk': 0.25,
                    'profit': 0.25,
                    'consistency': 0.15
                }
            }
            
            # 缓存结果
            self._validation_cache[cache_key] = result
            
            logger.info(f"信号验证完成: {signal_id}, 综合评分: {result.overall_score:.2f}, 等级: {result.quality_level.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"验证信号失败: {e}")
            return SignalValidationResult(
                signal_id=signal.get('id', ''),
                symbol=signal.get('symbol', ''),
                signal_type=signal.get('type', 'unknown'),
                timestamp=datetime.now(),
                overall_score=0.0,
                is_valid=False
            )
    
    def _backtest_signal(
        self,
        signal: Dict[str, Any],
        historical_data: pd.DataFrame,
        lookback_periods: int = 20
    ) -> Dict[str, Any]:
        """
        回测信号
        
        Args:
            signal: 信号数据
            historical_data: 历史市场数据
            lookback_periods: 回测周期数
            
        Returns:
            回测结果
        """
        try:
            if historical_data.empty or len(historical_data) < lookback_periods:
                return {}
            
            signal_type = signal.get('type', 'unknown')
            
            # 使用最近的数据进行回测
            backtest_data = historical_data.tail(lookback_periods)
            
            # 计算收益率
            returns = backtest_data['close'].pct_change().dropna()
            
            # 模拟交易
            trades = []
            position = 0  # 0: 无仓位, 1: 多头, -1: 空头
            entry_price = 0.0
            
            for i in range(1, len(backtest_data)):
                current_price = backtest_data['close'].iloc[i]
                prev_price = backtest_data['close'].iloc[i-1]
                
                # 简单的信号逻辑
                if signal_type == 'buy':
                    if position == 0:
                        position = 1
                        entry_price = current_price
                    elif position == 1:
                        # 持有多头
                        pass
                        
                elif signal_type == 'sell':
                    if position == 1:
                        # 平仓
                        profit = (current_price - entry_price) / entry_price
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit,
                            'type': 'long'
                        })
                        position = 0
                        entry_price = 0.0
            
            # 计算统计指标
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            profits = [t['profit'] for t in trades]
            avg_profit = np.mean(profits) if profits else 0.0
            
            # 计算最大回撤
            cumulative_returns = (1 + pd.Series(profits)).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
            
            # 计算夏普比率（简化版）
            if len(returns) > 1 and returns.std() != 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'returns': returns.tolist() if len(returns) > 0 else []
            }
            
        except Exception as e:
            logger.error(f"回测信号失败: {e}")
            return {}
    
    def _calculate_accuracy_score(self, backtest_result: Dict) -> float:
        """计算准确性评分"""
        if not backtest_result:
            return 0.0
        
        win_rate = backtest_result.get('win_rate', 0.0)
        total_trades = backtest_result.get('total_trades', 0)
        
        # 基于胜率和交易次数计算准确性
        if total_trades < 5:
            return win_rate * 0.5  # 交易次数少，降低权重
        
        return min(win_rate * 1.2, 1.0)  # 最高1.0
    
    def _calculate_risk_score(self, backtest_result: Dict) -> float:
        """计算风险评分"""
        if not backtest_result:
            return 0.0
        
        max_drawdown = backtest_result.get('max_drawdown', 0.0)
        sharpe_ratio = backtest_result.get('sharpe_ratio', 0.0)
        
        # 回撤越小越好，夏普比率越高越好
        drawdown_score = max(0, 1 + max_drawdown) if max_drawdown < 0 else 1.0
        sharpe_score = min(max(sharpe_ratio / 2, 0), 1.0)  # 归一化到 0-1
        
        return (drawdown_score * 0.6 + sharpe_score * 0.4)
    
    def _calculate_profit_score(self, backtest_result: Dict) -> float:
        """计算收益评分"""
        if not backtest_result:
            return 0.0
        
        avg_profit = backtest_result.get('avg_profit', 0.0)
        total_trades = backtest_result.get('total_trades', 0)
        
        # 基于平均收益和交易次数
        if total_trades < 3:
            return max(0, avg_profit * 5)  # 放大收益影响
        
        return min(max(avg_profit * 10, 0), 1.0)
    
    def _calculate_consistency_score(self, backtest_result: Dict) -> float:
        """计算一致性评分"""
        if not backtest_result:
            return 0.0
        
        returns = backtest_result.get('returns', [])
        if len(returns) < 2:
            return 0.5
        
        # 计算收益的稳定性（标准差越小越好）
        returns_std = np.std(returns)
        returns_mean = np.mean(returns)
        
        if returns_mean == 0:
            return 0.5
        
        # 变异系数（Coefficient of Variation）
        cv = abs(returns_std / returns_mean)
        consistency = max(0, 1 - cv)
        
        return min(consistency, 1.0)
    
    def _calculate_overall_score(
        self,
        accuracy_score: float,
        risk_score: float,
        profit_score: float,
        consistency_score: float
    ) -> float:
        """计算综合评分"""
        weights = {
            'accuracy': 0.35,
            'risk': 0.25,
            'profit': 0.25,
            'consistency': 0.15
        }
        
        overall = (
            accuracy_score * weights['accuracy'] +
            risk_score * weights['risk'] +
            profit_score * weights['profit'] +
            consistency_score * weights['consistency']
        )
        
        return min(max(overall, 0.0), 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> SignalQualityLevel:
        """确定质量等级"""
        if overall_score >= 0.8:
            return SignalQualityLevel.EXCELLENT
        elif overall_score >= 0.6:
            return SignalQualityLevel.GOOD
        elif overall_score >= 0.4:
            return SignalQualityLevel.AVERAGE
        elif overall_score >= 0.2:
            return SignalQualityLevel.POOR
        else:
            return SignalQualityLevel.BAD
    
    def get_signal_statistics(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取信号统计信息
        
        Args:
            symbol: 股票代码（可选）
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            
        Returns:
            统计信息
        """
        try:
            # 过滤验证结果
            filtered_results = []
            for result in self._validation_cache.values():
                if symbol and result.symbol != symbol:
                    continue
                if start_time and result.timestamp < start_time:
                    continue
                if end_time and result.timestamp > end_time:
                    continue
                filtered_results.append(result)
            
            if not filtered_results:
                return {
                    'total_signals': 0,
                    'valid_signals': 0,
                    'invalid_signals': 0,
                    'avg_score': 0.0,
                    'quality_distribution': {}
                }
            
            # 计算统计指标
            total_signals = len(filtered_results)
            valid_signals = sum(1 for r in filtered_results if r.is_valid)
            invalid_signals = total_signals - valid_signals
            
            scores = [r.overall_score for r in filtered_results]
            avg_score = np.mean(scores) if scores else 0.0
            
            # 质量分布
            quality_distribution = {}
            for level in SignalQualityLevel:
                count = sum(1 for r in filtered_results if r.quality_level == level)
                quality_distribution[level.value] = count
            
            return {
                'total_signals': total_signals,
                'valid_signals': valid_signals,
                'invalid_signals': invalid_signals,
                'avg_score': round(avg_score, 4),
                'quality_distribution': quality_distribution,
                'win_rate_avg': np.mean([r.win_rate for r in filtered_results]),
                'sharpe_ratio_avg': np.mean([r.sharpe_ratio for r in filtered_results]),
                'max_drawdown_avg': np.mean([r.max_drawdown for r in filtered_results])
            }
            
        except Exception as e:
            logger.error(f"获取信号统计信息失败: {e}")
            return {}
    
    def clear_cache(self):
        """清除缓存"""
        self._validation_cache.clear()
        logger.info("信号验证缓存已清除")


# 单例实例
_signal_validator: Optional[SignalValidator] = None


def get_signal_validator() -> SignalValidator:
    """获取信号验证器实例"""
    global _signal_validator
    if _signal_validator is None:
        _signal_validator = SignalValidator()
    return _signal_validator
