#!/usr/bin/env python3
"""
动态风险限额管理系统

构建智能的动态风险限额管理和调整系统
    创建时间: 2025年3月
"""

import sys
import os
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 配置日志记录器
logger = logging.getLogger(__name__)

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from automation.trade_adjustment_engine import (
        RiskLimit
    )
    print("✅ 交易调整引擎导入成功")
except ImportError as e:
    print(f"❌ 交易调整引擎导入失败: {e}")
    # 创建简化的替代类用于演示

    class RiskLimit:

        def __init__(self, **kwargs):

            for k, v in kwargs.items():
                setattr(self, k, v)


class LimitAdjustmentStrategy(Enum):

    """限额调整策略枚举"""
    CONSERVATIVE = "conservative"     # 保守策略
    MODERATE = "moderate"            # 适中策略
    AGGRESSIVE = "aggressive"        # 激进策略
    ADAPTIVE = "adaptive"            # 自适应策略
    MACHINE_LEARNING = "machine_learning"  # 机器学习策略


class MarketRegime(Enum):

    """市场状态枚举"""
    NORMAL = "normal"                # 正常市场
    VOLATILE = "volatile"            # 波动市场
    STRESSED = "stressed"            # 压力市场
    CRISIS = "crisis"                # 危机市场
    RECOVERY = "recovery"            # 恢复市场


@dataclass
class LimitAdjustment:

    """限额调整"""
    adjustment_id: str
    timestamp: datetime
    strategy: LimitAdjustmentStrategy
    market_regime: MarketRegime
    asset_symbol: str
    original_limits: Dict[str, float]
    adjusted_limits: Dict[str, float]
    adjustment_factors: Dict[str, float]
    reason: str
    expected_impact: Dict[str, float]
    status: str = "proposed"  # proposed, approved, rejected, applied

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'adjustment_id': self.adjustment_id,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy.value,
            'market_regime': self.market_regime.value,
            'asset_symbol': self.asset_symbol,
            'original_limits': self.original_limits,
            'adjusted_limits': self.adjusted_limits,
            'adjustment_factors': self.adjustment_factors,
            'reason': self.reason,
            'expected_impact': self.expected_impact,
            'status': self.status
        }


class MarketRegimeDetector:

    """市场状态检测器"""

    def __init__(self):

        self.regime_history = []
        self.current_regime = MarketRegime.NORMAL
        self.regime_confidence = 0.0

        # 市场状态阈值
        self.thresholds = {
            'volatility_normal': 0.15,
            'volatility_volatile': 0.25,
            'volatility_crisis': 0.40,
            'stress_normal': 0.3,
            'stress_crisis': 0.7,
            'correlation_normal': 0.5,
            'correlation_crisis': 0.8
        }

    def detect_regime(self, market_data: Dict[str, Any]) -> Tuple[MarketRegime, float]:
        """检测市场状态"""
        # 提取关键指标
        volatility = market_data.get('market_volatility', 0.15)
        stress_level = market_data.get('market_stress_level', 0.3)
        correlation = market_data.get('average_correlation', 0.3)
        price_momentum = market_data.get('price_momentum', 0.0)
        volume_spike = market_data.get('volume_spike', False)

        # 计算市场状态得分
        volatility_score = self._calculate_volatility_score(volatility)
        stress_score = self._calculate_stress_score(stress_level)
        correlation_score = self._calculate_correlation_score(correlation)

        # 综合评分
        total_score = (volatility_score + stress_score + correlation_score) / 3

        # 确定市场状态
        if total_score >= 0.8 or volume_spike:
            regime = MarketRegime.CRISIS
            confidence = min(total_score, 0.95)
        elif total_score >= 0.6:
            regime = MarketRegime.STRESSED
            confidence = total_score
        elif total_score >= 0.4:
            regime = MarketRegime.VOLATILE
            confidence = total_score
        elif price_momentum > 0.1:
            regime = MarketRegime.RECOVERY
            confidence = 0.7
        else:
            regime = MarketRegime.NORMAL
            confidence = 1.0 - total_score

        self.current_regime = regime
        self.regime_confidence = confidence

        # 记录历史
        self.regime_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': regime.value,
            'confidence': confidence,
            'scores': {
                'volatility': volatility_score,
                'stress': stress_score,
                'correlation': correlation_score
            }
        })

        # 限制历史记录数量
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

        return regime, confidence

    def _calculate_volatility_score(self, volatility: float) -> float:
        """计算波动率得分"""
        if volatility >= self.thresholds['volatility_crisis']:
            return 1.0
        elif volatility >= self.thresholds['volatility_volatile']:
            return 0.7 + (volatility - self.thresholds['volatility_volatile']) / \
                         (self.thresholds['volatility_crisis'] -
                          self.thresholds['volatility_volatile']) * 0.3
        elif volatility >= self.thresholds['volatility_normal']:
            return 0.4 + (volatility - self.thresholds['volatility_normal']) / \
                         (self.thresholds['volatility_volatile'] -
                          self.thresholds['volatility_normal']) * 0.3
        else:
            return volatility / self.thresholds['volatility_normal'] * 0.4

    def _calculate_stress_score(self, stress_level: float) -> float:
        """计算压力得分"""
        if stress_level >= self.thresholds['stress_crisis']:
            return 1.0
        elif stress_level >= self.thresholds['stress_normal']:
            return 0.5 + (stress_level - self.thresholds['stress_normal']) / \
                         (self.thresholds['stress_crisis'] - self.thresholds['stress_normal']) * 0.5
        else:
            return stress_level / self.thresholds['stress_normal'] * 0.5

    def _calculate_correlation_score(self, correlation: float) -> float:
        """计算相关性得分"""
        if correlation >= self.thresholds['correlation_crisis']:
            return 1.0
        elif correlation >= self.thresholds['correlation_normal']:
            return 0.6 + (correlation - self.thresholds['correlation_normal']) / \
                         (self.thresholds['correlation_crisis'] -
                          self.thresholds['correlation_normal']) * 0.4
        else:
            return correlation / self.thresholds['correlation_normal'] * 0.6

    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取状态统计"""
        if not self.regime_history:
            return {}

        recent_regimes = self.regime_history[-100:]  # 最近100个记录

        # 计算状态分布
        regime_counts = {}
        for record in recent_regimes:
            regime = record['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # 计算状态持续时间
        current_regime_start = None
        regime_durations = []

        for i, record in enumerate(recent_regimes):
            if i == 0 or record['regime'] != recent_regimes[i - 1]['regime']:
                if current_regime_start is not None:
                    duration = datetime.fromisoformat(record['timestamp']) - current_regime_start
                    regime_durations.append({
                        'regime': recent_regimes[i - 1]['regime'],
                        'duration_minutes': duration.total_seconds() / 60
                    })
                current_regime_start = datetime.fromisoformat(record['timestamp'])

        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_distribution': regime_counts,
            'recent_transitions': regime_durations[-5:],  # 最近5个状态转换
            'total_records': len(self.regime_history)
        }


class DynamicLimitAdjuster:

    """动态限额调整器"""

    def __init__(self, strategy: LimitAdjustmentStrategy = LimitAdjustmentStrategy.ADAPTIVE):

        self.strategy = strategy
        self.market_detector = MarketRegimeDetector()
        self.adjustments: List[LimitAdjustment] = []
        self.baseline_limits = {}

        # 策略参数
        self.strategy_params = {
            LimitAdjustmentStrategy.CONSERVATIVE: {
                'volatility_factor': 0.7,
                'stress_factor': 0.8,
                'correlation_factor': 0.9,
                'recovery_factor': 1.1
            },
            LimitAdjustmentStrategy.MODERATE: {
                'volatility_factor': 0.8,
                'stress_factor': 0.9,
                'correlation_factor': 0.95,
                'recovery_factor': 1.15
            },
            LimitAdjustmentStrategy.AGGRESSIVE: {
                'volatility_factor': 0.9,
                'stress_factor': 0.95,
                'correlation_factor': 0.98,
                'recovery_factor': 1.2
            },
            LimitAdjustmentStrategy.ADAPTIVE: {
                'base_factor': 0.85,
                'regime_sensitivity': 0.3,
                'momentum_factor': 0.1
            }
        }

    def calculate_adjustment(self, asset_symbol: str, current_limits: Dict[str, float],


                             market_data: Dict[str, Any]) -> LimitAdjustment:
        """计算限额调整"""
        # 检测市场状态
        market_regime, confidence = self.market_detector.detect_regime(market_data)

        # 根据策略计算调整因子
        if self.strategy == LimitAdjustmentStrategy.ADAPTIVE:
            adjustment_factors = self._calculate_adaptive_factors(market_regime, market_data)
        else:
            adjustment_factors = self._calculate_strategy_factors(market_regime)

        # 计算调整后的限额
        adjusted_limits = {}
        for limit_name, current_value in current_limits.items():
            if limit_name in adjustment_factors:
                factor = adjustment_factors[limit_name]
                adjusted_limits[limit_name] = current_value * factor
            else:
                adjusted_limits[limit_name] = current_value

        # 确定调整原因
        reason = self._generate_adjustment_reason(market_regime, confidence, market_data)

        # 计算预期影响
        expected_impact = self._calculate_expected_impact(
            current_limits, adjusted_limits, market_data
        )

        # 创建调整对象
        adjustment = LimitAdjustment(
            adjustment_id=f"limit_adj_{asset_symbol}_{int(time.time())}",
            timestamp=datetime.now(),
            strategy=self.strategy,
            market_regime=market_regime,
            asset_symbol=asset_symbol,
            original_limits=current_limits.copy(),
            adjusted_limits=adjusted_limits,
            adjustment_factors=adjustment_factors,
            reason=reason,
            expected_impact=expected_impact
        )

        self.adjustments.append(adjustment)
        return adjustment

    def _calculate_adaptive_factors(self, market_regime: MarketRegime,


                                    market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算自适应调整因子"""
        base_factor = self.strategy_params[LimitAdjustmentStrategy.ADAPTIVE]['base_factor']
        regime_sensitivity = self.strategy_params[LimitAdjustmentStrategy.ADAPTIVE]['regime_sensitivity']

        # 根据市场状态调整基础因子
        if market_regime == MarketRegime.CRISIS:
            regime_multiplier = 0.5
        elif market_regime == MarketRegime.STRESSED:
            regime_multiplier = 0.7
        elif market_regime == MarketRegime.VOLATILE:
            regime_multiplier = 0.85
        elif market_regime == MarketRegime.RECOVERY:
            regime_multiplier = 1.1
        else:  # NORMAL
            regime_multiplier = 1.0

        # 考虑动量因素
        momentum = market_data.get('price_momentum', 0.0)
        momentum_factor = 1.0 + momentum * 0.1

        # 计算最终因子
        final_factor = base_factor * regime_multiplier * momentum_factor

        return {
            'max_position_size': final_factor,
            'max_var_limit': final_factor,
            'max_drawdown_limit': final_factor * 0.9,  # 回撤限额更保守
            'daily_loss_limit': final_factor * 0.8,    # 每日损失限额最保守
            'volatility_limit': final_factor
        }

    def _calculate_strategy_factors(self, market_regime: MarketRegime) -> Dict[str, float]:
        """计算策略调整因子"""
        params = self.strategy_params[self.strategy]

        if market_regime == MarketRegime.CRISIS:
            base_factor = 0.5
        elif market_regime == MarketRegime.STRESSED:
            base_factor = 0.7
        elif market_regime == MarketRegime.VOLATILE:
            base_factor = 0.85
        elif market_regime == MarketRegime.RECOVERY:
            base_factor = 1.1
        else:  # NORMAL
            base_factor = 1.0

        return {
            'max_position_size': base_factor,
            'max_var_limit': base_factor,
            'max_drawdown_limit': base_factor * 0.9,
            'daily_loss_limit': base_factor * 0.8,
            'volatility_limit': base_factor
        }

    def _generate_adjustment_reason(self, market_regime: MarketRegime, confidence: float,


                                    market_data: Dict[str, Any]) -> str:
        """生成调整原因"""
        reasons = []

        if market_regime != MarketRegime.NORMAL:
            reasons.append(f"市场状态: {market_regime.value} (置信度: {confidence:.1%})")

        volatility = market_data.get('market_volatility', 0)
        if volatility > 0.25:
            reasons.append(f"市场波动率过高: {volatility:.2%}")

        stress_level = market_data.get('market_stress_level', 0)
        if stress_level > 0.5:
            reasons.append(f"市场压力水平较高: {stress_level:.2%}")

        if not reasons:
            reasons.append("基于当前市场条件的最优调整")

        return " | ".join(reasons)

    def _calculate_expected_impact(self, original_limits: Dict[str, float],


                                   adjusted_limits: Dict[str, float],
                                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """计算预期影响"""
        impact = {}

        for limit_name in original_limits:
            if limit_name in adjusted_limits:
                original = original_limits[limit_name]
                adjusted = adjusted_limits[limit_name]

                if original > 0:
                    change_pct = (adjusted - original) / original
                    impact[f"{limit_name}_change"] = change_pct

                    # 估算风险影响
                    if 'position' in limit_name.lower():
                        impact['expected_position_impact'] = -change_pct * 0.5  # 简化的影响估算
                    elif 'var' in limit_name.lower():
                        impact['expected_var_impact'] = -change_pct * 0.3
                    elif 'loss' in limit_name.lower():
                        impact['expected_loss_impact'] = -change_pct * 0.8

        # 整体风险影响
        avg_change = np.mean([v for k, v in impact.items() if k.endswith('_change')])
        impact['overall_risk_impact'] = avg_change if not np.isnan(avg_change) else 0

        return impact

    def get_adjustment_history(self, asset_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取调整历史"""
        if asset_symbol:
            history = [adj for adj in self.adjustments if adj.asset_symbol == asset_symbol]
        else:
            history = self.adjustments

        return [adj.to_dict() for adj in history[-100:]]  # 最近100个调整


class DynamicRiskLimitsManager:

    """动态风险限额管理器"""

    def __init__(self, strategy: LimitAdjustmentStrategy = LimitAdjustmentStrategy.ADAPTIVE):

        self.adjuster = DynamicLimitAdjuster(strategy)
        self.current_limits = {}
        self.applied_adjustments = []
        self.is_auto_adjustment_enabled = True

        # 监控统计
        self.stats = {
            'total_adjustments': 0,
            'applied_adjustments': 0,
            'rejected_adjustments': 0,
            'avg_adjustment_impact': 0.0,
            'regime_distribution': {}
        }

    def set_baseline_limits(self, asset_symbol: str, limits: Dict[str, float]):
        """设置基准限额"""
        self.current_limits[asset_symbol] = limits.copy()
        logger.info(f"设置基准限额: {asset_symbol} - {limits}")

    def evaluate_and_adjust(self, asset_symbol: str, market_data: Dict[str, Any]) -> Optional[LimitAdjustment]:
        """评估并调整限额"""
        if not self.is_auto_adjustment_enabled:
            return None

        if asset_symbol not in self.current_limits:
            logger.warning(f"未找到资产限额: {asset_symbol}")
            return None

        current_limits = self.current_limits[asset_symbol]

        # 计算调整
        adjustment = self.adjuster.calculate_adjustment(
            asset_symbol, current_limits, market_data
        )

        # 验证调整合理性
        if self._validate_adjustment(adjustment):
            # 应用调整
            self._apply_adjustment(adjustment)
            self.applied_adjustments.append(adjustment)
            self.stats['applied_adjustments'] += 1

            logger.info(f"限额调整已应用: {asset_symbol} - {adjustment.reason}")
            return adjustment
        else:
            self.stats['rejected_adjustments'] += 1
            adjustment.status = "rejected"
            logger.warning(f"限额调整被拒绝: {asset_symbol} - 验证失败")

        return None

    def _validate_adjustment(self, adjustment: LimitAdjustment) -> bool:
        """验证调整合理性"""
        # 检查调整幅度是否过大
        for limit_name, factor in adjustment.adjustment_factors.items():
            if factor < 0.3:  # 减少超过70%
                logger.warning(f"调整幅度过大: {limit_name} - {factor}")
                return False
            if factor > 2.0:  # 增加超过100%
                logger.warning(f"调整幅度过大: {limit_name} - {factor}")
                return False

        # 检查调整后的限额是否合理
        for limit_name, new_value in adjustment.adjusted_limits.items():
            if new_value <= 0:
                logger.warning(f"调整后限额无效: {limit_name} - {new_value}")
                return False

        return True

    def _apply_adjustment(self, adjustment: LimitAdjustment):
        """应用调整"""
        asset_symbol = adjustment.asset_symbol
        self.current_limits[asset_symbol] = adjustment.adjusted_limits.copy()
        adjustment.status = "applied"

        self.stats['total_adjustments'] += 1

        # 更新统计
        impact = adjustment.expected_impact.get('overall_risk_impact', 0)
        self.stats['avg_adjustment_impact'] = (
            (self.stats['avg_adjustment_impact'] * (self.stats['total_adjustments'] - 1))
            + impact
        ) / self.stats['total_adjustments']

    def get_current_limits(self, asset_symbol: str) -> Optional[Dict[str, float]]:
        """获取当前限额"""
        return self.current_limits.get(asset_symbol)

    def get_limits_summary(self) -> Dict[str, Any]:
        """获取限额摘要"""
        return {
            'total_assets': len(self.current_limits),
            'current_limits': self.current_limits.copy(),
            'applied_adjustments': len(self.applied_adjustments),
            'auto_adjustment_enabled': self.is_auto_adjustment_enabled,
            'stats': self.stats.copy(),
            'regime_stats': self.adjuster.market_detector.get_regime_statistics()
        }

    def enable_auto_adjustment(self):
        """启用自动调整"""
        self.is_auto_adjustment_enabled = True
        logger.info("自动限额调整已启用")

    def disable_auto_adjustment(self):
        """禁用自动调整"""
        self.is_auto_adjustment_enabled = False
        logger.info("自动限额调整已禁用")


def create_sample_limits() -> Dict[str, Dict[str, float]]:
    """创建示例限额"""
    return {
        'AAPL': {
            'max_position_size': 100000,
            'max_var_limit': 0.05,
            'max_drawdown_limit': 0.15,
            'daily_loss_limit': 0.02,
            'volatility_limit': 0.25
        },
        'GOOGL': {
            'max_position_size': 80000,
            'max_var_limit': 0.04,
            'max_drawdown_limit': 0.12,
            'daily_loss_limit': 0.015,
            'volatility_limit': 0.22
        },
        'MSFT': {
            'max_position_size': 120000,
            'max_var_limit': 0.06,
            'max_drawdown_limit': 0.18,
            'daily_loss_limit': 0.025,
            'volatility_limit': 0.28
        }
    }


def create_sample_market_data(regime: str = "normal") -> Dict[str, Any]:
    """创建示例市场数据"""
    np.random.seed(int(time.time()) % 10000)

    if regime == "crisis":
        volatility = 0.45 + np.secrets.normal(0, 0.05)
        stress_level = 0.8 + np.secrets.normal(0, 0.1)
        correlation = 0.85 + np.secrets.normal(0, 0.05)
        price_momentum = np.secrets.normal(-0.2, 0.1)
    elif regime == "stressed":
        volatility = 0.30 + np.secrets.normal(0, 0.05)
        stress_level = 0.6 + np.secrets.normal(0, 0.1)
        correlation = 0.7 + np.secrets.normal(0, 0.1)
        price_momentum = np.secrets.normal(-0.1, 0.1)
    elif regime == "volatile":
        volatility = 0.25 + np.secrets.normal(0, 0.05)
        stress_level = 0.4 + np.secrets.normal(0, 0.1)
        correlation = 0.6 + np.secrets.normal(0, 0.1)
        price_momentum = np.secrets.normal(0, 0.1)
    else:  # normal
        volatility = 0.15 + np.secrets.normal(0, 0.05)
        stress_level = 0.2 + np.secrets.normal(0, 0.1)
        correlation = 0.3 + np.secrets.normal(0, 0.1)
        price_momentum = np.secrets.normal(0.05, 0.1)

    return {
        'timestamp': datetime.now().isoformat(),
        'market_volatility': max(0.1, volatility),
        'market_stress_level': np.clip(stress_level, 0, 1),
        'average_correlation': np.clip(correlation, 0, 1),
        'price_momentum': price_momentum,
        'volume_spike': np.secrets.random() < 0.1,
        'regime': regime
    }


def main():
    """主函数 - 动态风险限额管理系统演示"""
    print("🎛️ RQA2025动态风险限额管理系统")
    print("=" * 60)

    # 创建动态限额管理器
    manager = DynamicRiskLimitsManager(LimitAdjustmentStrategy.ADAPTIVE)

    # 设置基准限额
    sample_limits = create_sample_limits()
    for asset, limits in sample_limits.items():
        manager.set_baseline_limits(asset, limits)

    print("✅ 动态风险限额管理系统创建完成")
    print(f"   配置资产数量: {len(sample_limits)}")
    print("   调整策略: 自适应策略")
    print("   市场状态检测器: 已启用")

    # 显示初始限额
    print("\n📋 初始风险限额:")
    for asset, limits in sample_limits.items():
        print(f"   {asset}:")
        for limit_name, value in limits.items():
            print(f"     {limit_name}: {value}")

    try:
        # 模拟不同市场状态
        market_scenarios = [
            ("normal", "正常市场"),
            ("volatile", "波动市场"),
            ("stressed", "压力市场"),
            ("crisis", "危机市场"),
            ("recovery", "恢复市场")
        ]

        print("\n" + "=" * 60)
        print("🧪 市场情景模拟测试")

        for regime, description in market_scenarios:
            print(f"\n🌟 情景: {description}")
            print("-" * 40)

            # 创建市场数据
            market_data = create_sample_market_data(regime)

            print("市场数据:")
            print(f"   波动率: {market_data['market_volatility']:.2%}")
            print(f"   压力水平: {market_data['market_stress_level']:.2%}")
            print(f"   相关性: {market_data['average_correlation']:.2%}")

            # 检测市场状态
            detector = manager.adjuster.market_detector
            detected_regime, confidence = detector.detect_regime(market_data)
            print(f"   检测状态: {detected_regime.value} (置信度: {confidence:.1%})")

            # 对每个资产评估限额调整
            for asset in sample_limits.keys():
                adjustment = manager.evaluate_and_adjust(asset, market_data)

                if adjustment:
                    print(f"\n{asset}限额调整:")
                    print(f"   原因: {adjustment.reason}")
                    print(f"   调整因子: {adjustment.adjustment_factors}")

                    print("   调整前后对比:")
                    for limit_name in adjustment.original_limits:
                        original = adjustment.original_limits[limit_name]
                        adjusted = adjustment.adjusted_limits[limit_name]
                        change_pct = (adjusted - original) / original * 100
                        print(f"     {limit_name}: {original} -> {adjusted:.4f} ({change_pct:+.1f}%)")

                    # 显示预期影响
                    impact = adjustment.expected_impact
                    if 'overall_risk_impact' in impact:
                        print(f"   预期风险影响: {impact['overall_risk_impact']:.2%}")
                else:
                    print(f"\n{asset}: 无需调整")

            print("\n" + "-" * 40)

        # 显示最终统计
        print("\n" + "=" * 60)
        print("📊 最终统计")

        summary = manager.get_limits_summary()
        stats = summary['stats']

        print(f"总调整次数: {stats['total_adjustments']}")
        print(f"成功调整次数: {stats['applied_adjustments']}")
        print(f"拒绝调整次数: {stats['rejected_adjustments']}")
        print(f"平均调整影响: {stats['avg_adjustment_impact']:.2%}")

        # 显示当前限额
        print("\n📋 当前风险限额:")
        for asset, limits in summary['current_limits'].items():
            print(f"   {asset}:")
            for limit_name, value in limits.items():
                original = sample_limits[asset][limit_name]
                change_pct = (value - original) / original * 100
                print(f"     {limit_name}: {value:.4f} ({change_pct:+.1f}%)")

        # 显示市场状态统计
        regime_stats = summary['regime_stats']
        if regime_stats:
            print("📈 市场状态统计:")
            print(f"   当前状态: {regime_stats['current_regime']}")
            print(f"   状态置信度: {regime_stats['regime_confidence']:.1%}")
            if 'regime_distribution' in regime_stats:
                print(f"   状态分布: {regime_stats['regime_distribution']}")

        print("\n🎉 动态风险限额管理系统演示完成！")
        print("   系统已成功根据不同市场状态动态调整风险限额")
        print("   自适应策略能够有效平衡风险控制和投资机会")

        return manager

    except Exception as e:
        print(f"\n❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    manager = main()

# Logger setup
logger = logging.getLogger(__name__)
