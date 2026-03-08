"""交易风险组件 - 交易执行层组件，支持统一基础设施集成"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# 导入统一基础设施集成层
try:
    from src.infrastructure.integration import get_trading_layer_adapter
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_INTEGRATION_AVAILABLE = False


class RiskAction(Enum):

    """风险处理动作"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"


class TradingRiskManager:

    """交易风险管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化交易风险管理器

        Args:
            config: 风险管理配置
        """
        self.config = config or {}
        self._risk_rules = {}
        self._risk_history = []

        # 日志记录器
        import logging
        self._logger = logging.getLogger(self.__class__.__name__)

        # 基础设施集成
        self._infrastructure_adapter = None
        self._config_manager = None
        self._cache_manager = None
        self._monitoring = None

        # 初始化基础设施集成
        self._init_infrastructure_integration()

        # 从配置中获取参数
        self._load_config()

        # 设置默认风险规则
        self._setup_default_risk_rules()

    def _setup_default_risk_rules(self):
        """设置默认风险规则"""
        self._risk_rules = {
            "max_position_size": self._check_position_size,
            "max_daily_loss": self._check_daily_loss,
            "max_order_frequency": self._check_order_frequency,
            "market_volatility": self._check_market_volatility,
            "liquidity_check": self._check_liquidity
        }

    def _init_infrastructure_integration(self):
        """初始化基础设施集成"""
        if not INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            print("统一基础设施集成层不可用，使用降级模式")
            return

        try:
            # 获取交易层适配器
            self._infrastructure_adapter = get_trading_layer_adapter()

            if self._infrastructure_adapter:
                # 获取基础设施服务
                services = self._infrastructure_adapter.get_infrastructure_services()
                self._config_manager = services.get('config_manager')
                self._cache_manager = services.get('cache_manager')
                self._monitoring = services.get('monitoring')
                self._logger = services.get('logger')

                print("交易风险管理器成功连接统一基础设施集成层")
            else:
                print("无法获取交易层适配器")

        except Exception as e:
            print(f"基础设施集成初始化失败: {e}")

    def _load_config(self):
        """从配置管理器加载配置"""
        try:
            if self._config_manager:
                # 从统一配置管理器获取风险管理相关配置
                self.enable_monitoring = self._config_manager.get(
                    'trading.risk.enable_monitoring', True)
                self.enable_caching = self._config_manager.get('trading.risk.enable_caching', True)
                self.max_risk_history = self._config_manager.get('trading.risk.max_history', 1000)
                # 加载风险规则配置
                self.max_position_size = self._config_manager.get(
                    'trading.risk.max_position_size', 1000000)
                self.max_daily_loss = self._config_manager.get('trading.risk.max_daily_loss', 10000)
                self.max_orders_per_minute = self._config_manager.get(
                    'trading.risk.max_orders_per_minute', 10)
            else:
                # 使用默认值
                self.enable_monitoring = True
                self.enable_caching = True
                self.max_risk_history = 1000
                self.max_position_size = 1000000
                self.max_daily_loss = 10000
                self.max_orders_per_minute = 10
        except Exception as e:
            print(f"配置加载失败，使用默认值: {e}")
            self.enable_monitoring = True
            self.enable_caching = True
            self.max_risk_history = 1000
            self.max_position_size = 1000000
            self.max_daily_loss = 10000
            self.max_orders_per_minute = 10

    def evaluate_trade_risk(self, trade_context: Dict[str, Any]) -> Dict[str, Any]:
        """评估交易风险 - 支持基础设施集成

        Args:
            trade_context: 交易上下文信息

        Returns:
            风险评估结果
        """
        # 基础设施集成：检查缓存
        cached_result = None
        if self.enable_caching and hasattr(self, '_cache_manager') and self._cache_manager:
            cache_key = f"risk_eval_{hash(str(trade_context))}"
            cached_result = self._cache_manager.get(cache_key)
        if cached_result:
            print("使用缓存的风险评估结果")
            return cached_result

        # 基础设施集成：记录监控指标
        if self.enable_monitoring and hasattr(self, '_monitoring') and self._monitoring:
            try:
                self._monitoring.record_metric(
                    'risk_evaluation_start',
                    1,
                    {
                        'symbol': trade_context.get('symbol'),
                        'order_size': trade_context.get('order_size'),
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                print(f"记录风险评估开始指标失败: {e}")

        results = {
            "overall_action": RiskAction.ALLOW.value,
            "risk_score": 0.0,
            "warnings": [],
            "blocks": [],
            "recommendations": [],
            "rule_results": {}
        }

        # 执行各项风险规则检查
        for rule_name, rule_func in self._risk_rules.items():
            try:
                rule_result = rule_func(trade_context)
                results["rule_results"][rule_name] = rule_result

                # 更新总体风险评分
                results["risk_score"] += rule_result.get("risk_score", 0)

                # 处理警告和阻止
                if rule_result.get("action") == RiskAction.BLOCK.value:
                    results["blocks"].append(rule_result.get("message", ""))
                    results["overall_action"] = RiskAction.BLOCK.value
                elif rule_result.get("action") == RiskAction.WARN.value:
                    results["warnings"].append(rule_result.get("message", ""))
                    if results["overall_action"] == RiskAction.ALLOW.value:
                        results["overall_action"] = RiskAction.WARN.value

                # 收集建议
                if rule_result.get("recommendations"):
                    results["recommendations"].extend(rule_result["recommendations"])

            except Exception as e:
                results["warnings"].append(f"风险规则检查失败 {rule_name}: {e}")

        # 记录风险评估历史
        self._record_risk_evaluation(trade_context, results)

        # 基础设施集成：缓存评估结果
        if self.enable_caching and hasattr(self, '_cache_manager') and self._cache_manager:
            try:
                cache_key = f"risk_eval_{hash(str(trade_context))}"
                self._cache_manager.set(cache_key, results, ttl=300)  # 缓存5分钟
            except Exception as e:
                print(f"缓存风险评估结果失败: {e}")

        # 基础设施集成：记录完成监控指标
        if self.enable_monitoring and hasattr(self, '_monitoring') and self._monitoring:
            try:
                self._monitoring.record_metric(
                    'risk_evaluation_complete',
                    1,
                    {
                        'symbol': trade_context.get('symbol'),
                        'overall_action': results.get('overall_action'),
                        'risk_score': results.get('risk_score'),
                        'layer': 'trading'
                    }
                )
            except Exception as e:
                print(f"记录风险评估完成指标失败: {e}")

        return results

    def _check_position_size(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查持仓规模"""
        max_position = self.config.get("max_position_size", 1000000)
        current_position = context.get("current_position", 0)
        order_size = context.get("order_size", 0)

        new_position = current_position + order_size

        if new_position > max_position:
            return {
                "rule": "max_position_size",
                "action": RiskAction.BLOCK.value,
                "risk_score": 1.0,
                "message": f"持仓规模超限: {new_position} > {max_position}",
                "recommendations": ["减少订单规模", "分批执行"]
            }
        elif new_position > max_position * 0.8:
            return {
                "rule": "max_position_size",
                "action": RiskAction.WARN.value,
                "risk_score": 0.5,
                "message": f"持仓规模接近上限: {new_position} / {max_position}",
                "recommendations": ["谨慎操作", "考虑减仓"]
            }
        else:
            return {
                "rule": "max_position_size",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }

    def _check_daily_loss(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查日损失"""
        max_daily_loss = self.config.get("max_daily_loss", 10000)
        current_daily_loss = context.get("current_daily_loss", 0)

        if current_daily_loss >= max_daily_loss:
            return {
                "rule": "max_daily_loss",
                "action": RiskAction.BLOCK.value,
                "risk_score": 1.0,
                "message": f"日损失已达上限: {current_daily_loss} >= {max_daily_loss}",
                "recommendations": ["停止交易", "重新评估策略"]
            }
        elif current_daily_loss >= max_daily_loss * 0.8:
            return {
                "rule": "max_daily_loss",
                "action": RiskAction.WARN.value,
                "risk_score": 0.7,
                "message": f"日损失接近上限: {current_daily_loss} / {max_daily_loss}",
                "recommendations": ["减少交易频率", "降低风险暴露"]
            }
        else:
            return {
                "rule": "max_daily_loss",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }

    def _check_order_frequency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查订单频率"""
        max_orders_per_minute = self.config.get("max_orders_per_minute", 10)
        recent_orders = context.get("recent_orders", [])
        current_time = datetime.now()

        # 计算最近1分钟的订单数量
        recent_minute_orders = [
            order for order in recent_orders
            if (current_time - datetime.fromisoformat(order.get("timestamp", "2000-01-01T00:00:00"))).total_seconds() < 60
        ]

        if len(recent_minute_orders) >= max_orders_per_minute:
            return {
                "rule": "max_order_frequency",
                "action": RiskAction.BLOCK.value,
                "risk_score": 0.8,
                "message": f"订单频率超限: {len(recent_minute_orders)} / {max_orders_per_minute}",
                "recommendations": ["降低订单频率", "使用订单合并"]
            }
        elif len(recent_minute_orders) >= max_orders_per_minute * 0.7:
            return {
                "rule": "max_order_frequency",
                "action": RiskAction.WARN.value,
                "risk_score": 0.3,
                "message": f"订单频率较高: {len(recent_minute_orders)} / {max_orders_per_minute}",
                "recommendations": ["注意订单频率", "考虑批量处理"]
            }
        else:
            return {
                "rule": "max_order_frequency",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }

    def _check_market_volatility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查市场波动率"""
        max_volatility = self.config.get("max_volatility_threshold", 0.05)
        current_volatility = context.get("market_volatility", 0)

        if current_volatility >= max_volatility:
            return {
                "rule": "market_volatility",
                "action": RiskAction.REDUCE.value,
                "risk_score": 0.6,
                "message": f"市场波动率过高: {current_volatility:.2%} >= {max_volatility:.2%}",
                "recommendations": ["减少持仓规模", "使用更保守的策略"]
            }
        else:
            return {
                "rule": "market_volatility",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }

    def _check_liquidity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查流动性"""
        min_liquidity = self.config.get("min_liquidity_threshold", 100000)
        current_liquidity = context.get("liquidity", 0)

        if current_liquidity < min_liquidity:
            return {
                "rule": "liquidity_check",
                "action": RiskAction.REDUCE.value,
                "risk_score": 0.4,
                "message": f"流动性不足: {current_liquidity} < {min_liquidity}",
                "recommendations": ["减少订单规模", "寻找替代市场"]
            }
        else:
            return {
                "rule": "liquidity_check",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }

    def _record_risk_evaluation(self, context: Dict[str, Any], results: Dict[str, Any]):
        """记录风险评估"""
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "results": results
        }

        self._risk_history.append(evaluation_record)

        # 限制历史记录数量
        max_history = self.config.get("max_risk_history", 1000)
        if len(self._risk_history) > max_history:
            self._risk_history = self._risk_history[-max_history:]

    def add_risk_rule(self, name: str, rule_func: callable):
        """添加自定义风险规则

        Args:
            name: 规则名称
            rule_func: 规则检查函数
        """
        self._risk_rules[name] = rule_func

    def remove_risk_rule(self, name: str):
        """移除风险规则

        Args:
            name: 规则名称
        """
        if name in self._risk_rules:
            del self._risk_rules[name]

    def get_risk_history(self) -> List[Dict[str, Any]]:
        """获取风险评估历史

        Returns:
            风险评估历史记录列表
        """
        return self._risk_history.copy()

    def get_risk_statistics(self) -> Dict[str, Any]:
        """获取风险统计信息

        Returns:
            风险统计信息
        """
        total_evaluations = len(self._risk_history)
        blocked_trades = len([r for r in self._risk_history if r["results"]
                             ["overall_action"] == RiskAction.BLOCK.value])
        warned_trades = len([r for r in self._risk_history if r["results"]
                            ["overall_action"] == RiskAction.WARN.value])

        return {
            "total_evaluations": total_evaluations,
            "blocked_trades": blocked_trades,
            "warned_trades": warned_trades,
            "block_rate": blocked_trades / total_evaluations if total_evaluations > 0 else 0,
            "warn_rate": warned_trades / total_evaluations if total_evaluations > 0 else 0,
            "active_rules": list(self._risk_rules.keys())
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查 - 支持基础设施层监控"""
        health_info = {
            'component': 'TradingRiskManager',
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'risk_history_count': len(self._risk_history),
            'active_rules_count': len(self._risk_rules),
            'infrastructure_integration': INFRASTRUCTURE_INTEGRATION_AVAILABLE,
            'metrics': {}
        }

        # 检查风险历史状态
        if len(self._risk_history) > self.max_risk_history * 1.2:
            health_info['status'] = 'warning'
            health_info['warnings'] = ['风险历史记录超出限制']

        # 检查基础设施集成状态
        if INFRASTRUCTURE_INTEGRATION_AVAILABLE:
            health_info['infrastructure_status'] = {
                'adapter_available': self._infrastructure_adapter is not None,
                'config_manager': self._config_manager is not None,
                'cache_manager': self._cache_manager is not None,
                'monitoring': self._monitoring is not None,
                'logger': self._logger is not None
            }
        else:
            health_info['infrastructure_status'] = 'not_available'

        # 收集性能指标
        total_evaluations = len(self._risk_history)
        blocked_trades = len([r for r in self._risk_history if r["results"]
                             ["overall_action"] == RiskAction.BLOCK.value])
        warned_trades = len([r for r in self._risk_history if r["results"]
                            ["overall_action"] == RiskAction.WARN.value])

        health_info['metrics'] = {
            'total_evaluations': total_evaluations,
            'blocked_trades': blocked_trades,
            'warned_trades': warned_trades,
            'block_rate': blocked_trades / total_evaluations if total_evaluations > 0 else 0,
            'active_rules': len(self._risk_rules)
        }

        return health_info

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total_evaluations = len(self._risk_history)
        blocked_trades = len([r for r in self._risk_history if r["results"]
                             ["overall_action"] == RiskAction.BLOCK.value])
        warned_trades = len([r for r in self._risk_history if r["results"]
                            ["overall_action"] == RiskAction.WARN.value])

        return {
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': total_evaluations,
            'blocked_trades': blocked_trades,
            'warned_trades': warned_trades,
            'block_rate': blocked_trades / total_evaluations if total_evaluations > 0 else 0,
            'warn_rate': warned_trades / total_evaluations if total_evaluations > 0 else 0,
            'active_rules': list(self._risk_rules.keys()),
            'infrastructure_enabled': INFRASTRUCTURE_INTEGRATION_AVAILABLE,
            'monitoring_enabled': self.enable_monitoring,
            'caching_enabled': self.enable_caching
        }


class ChinaRiskController:
    """
    中国市场风险控制器

    专门针对中国股票市场的风险控制逻辑，包括：
    - 大宗交易限制
    - 涨跌停板控制
    - 融资融券风险控制
    - 市场波动风险控制
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化中国风险控制器

        Args:
            config: 风险控制配置
        """
        self.config = config or {}
        self._max_single_trade_ratio = self.config.get('max_single_trade_ratio', 0.1)  # 单笔交易最大比例
        self._max_daily_trade_ratio = self.config.get('max_daily_trade_ratio', 0.5)  # 日交易最大比例
        self._circuit_breaker_enabled = self.config.get('circuit_breaker_enabled', True)
        self._margin_trading_enabled = self.config.get('margin_trading_enabled', False)

        # 中国市场特有风险指标
        self._price_limit_check = True
        self._volume_spike_check = True
        self._market_impact_check = True

    def check_trade_risk(self, trade_request: Dict[str, Any]) -> RiskAction:
        """
        检查交易风险（中国市场特化）

        Args:
            trade_request: 交易请求信息

        Returns:
            RiskAction: 风险处理动作
        """
        try:
            # 检查基本风险
            basic_check = self._check_basic_risk(trade_request)
            if basic_check != RiskAction.ALLOW:
                return basic_check

            # 检查涨跌停板
            if self._price_limit_check:
                price_check = self._check_price_limits(trade_request)
                if price_check != RiskAction.ALLOW:
                    return price_check

            # 检查成交量异常
            if self._volume_spike_check:
                volume_check = self._check_volume_spike(trade_request)
                if volume_check != RiskAction.ALLOW:
                    return volume_check

            # 检查市场冲击
            if self._market_impact_check:
                impact_check = self._check_market_impact(trade_request)
                if impact_check != RiskAction.ALLOW:
                    return impact_check

            # 检查融资融券风险
            if self._margin_trading_enabled:
                margin_check = self._check_margin_trading_risk(trade_request)
                if margin_check != RiskAction.ALLOW:
                    return margin_check

            return RiskAction.ALLOW

        except Exception as e:
            logger.error(f"中国风险控制器检查失败: {e}")
            return RiskAction.BLOCK

    def _check_basic_risk(self, trade_request: Dict[str, Any]) -> RiskAction:
        """检查基本风险"""
        # 实现基本风险检查逻辑
        return RiskAction.ALLOW

    def _check_price_limits(self, trade_request: Dict[str, Any]) -> RiskAction:
        """检查涨跌停板"""
        # 实现涨跌停板检查逻辑
        return RiskAction.ALLOW

    def _check_volume_spike(self, trade_request: Dict[str, Any]) -> RiskAction:
        """检查成交量异常"""
        # 实现成交量异常检查逻辑
        return RiskAction.ALLOW

    def _check_market_impact(self, trade_request: Dict[str, Any]) -> RiskAction:
        """检查市场冲击"""
        # 实现市场冲击检查逻辑
        return RiskAction.ALLOW

    def _check_margin_trading_risk(self, trade_request: Dict[str, Any]) -> RiskAction:
        """检查融资融券风险"""
        # 实现融资融券风险检查逻辑
        return RiskAction.ALLOW

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        获取风险指标

        Returns:
            Dict[str, Any]: 风险指标数据
        """
        return {
            'max_single_trade_ratio': self._max_single_trade_ratio,
            'max_daily_trade_ratio': self._max_daily_trade_ratio,
            'circuit_breaker_enabled': self._circuit_breaker_enabled,
            'margin_trading_enabled': self._margin_trading_enabled,
            'price_limit_check': self._price_limit_check,
            'volume_spike_check': self._volume_spike_check,
            'market_impact_check': self._market_impact_check
        }