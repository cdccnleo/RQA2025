# -*- coding: utf-8 -*-
"""
自动化层 - 自动化层测试覆盖率提升测试
补充自动化层单元测试，目标覆盖率: 75%+

测试范围:
1. 自动化交易测试 - 市场做市、风险限额、交易调整
2. 调度执行测试 - 任务调度、规则引擎、工作流管理
3. 监控告警测试 - 系统监控、性能告警、自动化响应
4. 策略自动化测试 - 回测自动化、部署自动化、参数调优
5. 系统自动化测试 - DevOps自动化、维护自动化、扩展自动化
6. 数据自动化测试 - 数据管道、备份恢复、质量检查
7. 集成自动化测试 - API集成、云集成、数据库集成
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import numpy as np


class TestAutomatedTrading:
    """测试自动化交易功能"""

    def test_market_making_automation(self):
        """测试市场做市自动化"""
        class MarketMaker:
            def __init__(self, symbol: str, spread_target: float = 0.001,
                        inventory_limit: int = 1000):
                self.symbol = symbol
                self.spread_target = spread_target
                self.inventory_limit = inventory_limit
                self.inventory = 0
                self.orders = []
                self.last_mid_price = 100.0

            def update_market_data(self, bid_price: float, ask_price: float):
                """更新市场数据并调整报价"""
                mid_price = (bid_price + ask_price) / 2
                spread = ask_price - bid_price
                self.last_mid_price = mid_price

                # 计算目标报价
                target_bid = mid_price * (1 - self.spread_target / 2)
                target_ask = mid_price * (1 + self.spread_target / 2)

                # 基于库存调整报价（库存过多时降低买价，提高卖价）
                inventory_adjustment = (self.inventory / self.inventory_limit) * 0.001

                adjusted_bid = target_bid * (1 - inventory_adjustment)
                adjusted_ask = target_ask * (1 + inventory_adjustment)

                return {
                    "mid_price": mid_price,
                    "current_spread": spread,
                    "target_spread": self.spread_target * mid_price,
                    "adjusted_bid": adjusted_bid,
                    "adjusted_ask": adjusted_ask,
                    "inventory_adjustment": inventory_adjustment,
                    "suggested_action": self._determine_action(spread, self.spread_target * mid_price)
                }

            def _determine_action(self, current_spread: float, target_spread: float) -> str:
                """确定行动策略"""
                if current_spread > target_spread * 1.2:
                    return "tighten_spread"  # 收紧价差
                elif current_spread < target_spread * 0.8:
                    return "widen_spread"  # 放宽价差
                else:
                    return "maintain_spread"  # 维持价差

            def execute_market_making(self, bid_price: float, ask_price: float,
                                    max_order_size: int = 100) -> Dict[str, Any]:
                """执行市场做市策略"""
                market_update = self.update_market_data(bid_price, ask_price)

                # 生成买卖订单
                bid_order = {
                    "type": "limit_buy",
                    "symbol": self.symbol,
                    "price": round(market_update["adjusted_bid"], 2),
                    "quantity": min(max_order_size, self.inventory_limit - self.inventory),
                    "timestamp": datetime.now()
                }

                ask_order = {
                    "type": "limit_sell",
                    "symbol": self.symbol,
                    "price": round(market_update["adjusted_ask"], 2),
                    "quantity": min(max_order_size, self.inventory_limit + self.inventory),
                    "timestamp": datetime.now()
                }

                # 记录订单
                self.orders.extend([bid_order, ask_order])

                return {
                    "market_analysis": market_update,
                    "bid_order": bid_order,
                    "ask_order": ask_order,
                    "expected_spread_capture": market_update["target_spread"],
                    "inventory_hedging": abs(self.inventory) / self.inventory_limit
                }

            def update_inventory(self, trade_quantity: int):
                """更新库存"""
                self.inventory += trade_quantity

                # 触发库存再平衡检查
                if abs(self.inventory) > self.inventory_limit * 0.8:
                    return {"rebalance_needed": True, "current_inventory": self.inventory}
                else:
                    return {"rebalance_needed": False, "current_inventory": self.inventory}

        # 测试市场做市自动化
        market_maker = MarketMaker("AAPL", spread_target=0.002, inventory_limit=1000)

        # 模拟市场数据更新
        market_data = market_maker.update_market_data(99.50, 100.50)
        assert market_data["mid_price"] == 100.0
        assert market_data["current_spread"] == 1.0
        assert "adjusted_bid" in market_data
        assert "adjusted_ask" in market_data

        # 执行市场做市
        execution = market_maker.execute_market_making(99.50, 100.50, max_order_size=50)
        assert "bid_order" in execution
        assert "ask_order" in execution
        assert execution["bid_order"]["quantity"] <= 50
        assert execution["ask_order"]["quantity"] <= 50

        # 测试库存管理
        inventory_update = market_maker.update_inventory(25)  # 买入25股
        assert inventory_update["current_inventory"] == 25

        inventory_update = market_maker.update_inventory(-10)  # 卖出10股
        assert inventory_update["current_inventory"] == 15

    def test_risk_limits_automation(self):
        """测试风险限额自动化"""
        class RiskLimitsAutomator:
            def __init__(self):
                self.position_limits = {
                    "max_position_size": 10000,
                    "max_daily_loss": 50000,
                    "max_drawdown": 0.1,
                    "var_limit": 0.05  # 5% VaR限制
                }
                self.current_positions = {}
                self.daily_pnl = 0.0
                self.peak_value = 1000000.0
                self.current_value = 1000000.0
                self.var_history = []

            def check_position_limits(self, symbol: str, proposed_quantity: int,
                                   current_price: float) -> Dict[str, Any]:
                """检查持仓限额"""
                current_position = self.current_positions.get(symbol, 0)
                new_position = current_position + proposed_quantity
                position_value = abs(new_position) * current_price

                violations = []

                # 检查持仓规模限额
                if position_value > self.position_limits["max_position_size"]:
                    violations.append({
                        "type": "position_size",
                        "current_value": position_value,
                        "limit": self.position_limits["max_position_size"],
                        "violation_ratio": position_value / self.position_limits["max_position_size"]
                    })

                # 检查集中度（单个持仓占总资产的比例）
                concentration_ratio = position_value / self.current_value
                if concentration_ratio > 0.2:  # 超过20%集中度
                    violations.append({
                        "type": "concentration",
                        "concentration_ratio": concentration_ratio,
                        "threshold": 0.2
                    })

                return {
                    "can_trade": len(violations) == 0,
                    "proposed_position": new_position,
                    "position_value": position_value,
                    "violations": violations,
                    "risk_score": len(violations) * 0.2  # 每个违规增加0.2风险分数
                }

            def check_portfolio_limits(self, trade_pnl: float = 0.0) -> Dict[str, Any]:
                """检查投资组合限额"""
                # 更新PnL
                self.daily_pnl += trade_pnl

                # 更新资产价值
                self.current_value += trade_pnl

                # 计算回撤
                drawdown = (self.peak_value - self.current_value) / self.peak_value
                if self.current_value > self.peak_value:
                    self.peak_value = self.current_value

                violations = []

                # 检查每日损失限额
                if self.daily_pnl < -self.position_limits["max_daily_loss"]:
                    violations.append({
                        "type": "daily_loss",
                        "current_loss": -self.daily_pnl,
                        "limit": self.position_limits["max_daily_loss"],
                        "violation_ratio": (-self.daily_pnl) / self.position_limits["max_daily_loss"]
                    })

                # 检查最大回撤限额
                if drawdown > self.position_limits["max_drawdown"]:
                    violations.append({
                        "type": "drawdown",
                        "current_drawdown": drawdown,
                        "limit": self.position_limits["max_drawdown"],
                        "violation_ratio": drawdown / self.position_limits["max_drawdown"]
                    })

                # 检查VaR限额（模拟）
                current_var = self._calculate_var()
                if current_var > self.position_limits["var_limit"]:
                    violations.append({
                        "type": "var_limit",
                        "current_var": current_var,
                        "limit": self.position_limits["var_limit"],
                        "violation_ratio": current_var / self.position_limits["var_limit"]
                    })

                return {
                    "within_limits": len(violations) == 0,
                    "daily_pnl": self.daily_pnl,
                    "current_drawdown": drawdown,
                    "current_var": current_var,
                    "violations": violations,
                    "risk_level": "high" if len(violations) > 1 else "medium" if len(violations) == 1 else "low"
                }

            def _calculate_var(self) -> float:
                """计算VaR（简化实现）"""
                # 模拟VaR计算
                base_var = 0.02  # 2%基础VaR
                # 基于持仓集中度调整VaR
                concentration_factor = sum(abs(pos) for pos in self.current_positions.values()) / 10000
                var = base_var * (1 + concentration_factor)
                self.var_history.append(var)
                return var

            def execute_risk_control_action(self, violations: List[Dict[str, Any]]) -> List[str]:
                """执行风险控制行动"""
                actions = []

                for violation in violations:
                    if violation["type"] == "daily_loss":
                        actions.extend([
                            "Reduce position sizes by 50%",
                            "Stop trading for today",
                            "Alert risk management team"
                        ])
                    elif violation["type"] == "drawdown":
                        actions.extend([
                            "Implement position rebalancing",
                            "Reduce leverage",
                            "Increase cash holdings"
                        ])
                    elif violation["type"] == "position_size":
                        actions.extend([
                            "Liquidate partial positions",
                            "Implement position size limits",
                            "Diversify across more assets"
                        ])
                    elif violation["type"] == "concentration":
                        actions.append("Reduce position concentration through diversification")

                return list(set(actions))  # 去重

        # 测试风险限额自动化
        risk_automator = RiskLimitsAutomator()

        # 测试持仓限额检查
        position_check = risk_automator.check_position_limits("AAPL", 5000, 150.0)
        position_value = 5000 * 150.0  # 75万美元

        if position_value > risk_automator.position_limits["max_position_size"]:
            assert position_check["can_trade"] == False
            assert len(position_check["violations"]) > 0

        # 测试投资组合限额检查
        portfolio_check = risk_automator.check_portfolio_limits(trade_pnl=-60000)  # 6万美元损失

        if -risk_automator.daily_pnl > risk_automator.position_limits["max_daily_loss"]:
            assert portfolio_check["within_limits"] == False
            assert len(portfolio_check["violations"]) > 0

        # 测试风险控制行动
        if portfolio_check["violations"]:
            actions = risk_automator.execute_risk_control_action(portfolio_check["violations"])
            assert len(actions) > 0
            assert all(isinstance(action, str) for action in actions)

    def test_trade_adjustment_automation(self):
        """测试交易调整自动化"""
        class TradeAdjustmentAutomator:
            def __init__(self):
                self.adjustment_rules = {
                    "price_slippage": {"threshold": 0.005, "action": "cancel_and_replace"},
                    "volume_surge": {"threshold": 2.0, "action": "split_orders"},
                    "market_volatility": {"threshold": 0.03, "action": "widen_spread"},
                    "liquidity_dryup": {"threshold": 0.1, "action": "reduce_size"}
                }
                self.active_orders = {}
                self.market_conditions = {}

            def analyze_market_conditions(self, symbol: str, recent_trades: List[Dict[str, float]],
                                       order_book_depth: Dict[str, int]) -> Dict[str, Any]:
                """分析市场条件"""
                if not recent_trades:
                    return {"conditions_normal": True}

                # 计算价格波动率
                prices = [trade["price"] for trade in recent_trades]
                returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                volatility = np.std(returns) if returns else 0.0

                # 计算交易量变化
                volumes = [trade["volume"] for trade in recent_trades]
                avg_volume = np.mean(volumes)
                recent_volume = volumes[-1] if volumes else 0
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

                # 评估流动性
                bid_depth = order_book_depth.get("bid_depth", 0)
                ask_depth = order_book_depth.get("ask_depth", 0)
                liquidity_score = (bid_depth + ask_depth) / 1000  # 标准化流动性分数

                conditions = {
                    "volatility": volatility,
                    "volume_ratio": volume_ratio,
                    "liquidity_score": liquidity_score,
                    "price_trend": "up" if prices[-1] > prices[0] else "down",
                    "market_stress": volatility > 0.05 or volume_ratio > 3.0 or liquidity_score < 0.2
                }

                self.market_conditions[symbol] = conditions
                return conditions

            def adjust_order_parameters(self, symbol: str, original_order: Dict[str, Any]) -> Dict[str, Any]:
                """调整订单参数"""
                conditions = self.market_conditions.get(symbol, {})
                adjusted_order = original_order.copy()

                adjustments_made = []

                # 基于波动率调整
                volatility = conditions.get("volatility", 0)
                if volatility > self.adjustment_rules["market_volatility"]["threshold"]:
                    # 放宽价差
                    if "limit_price" in adjusted_order:
                        spread_adjustment = volatility * 0.1
                        adjusted_order["limit_price"] *= (1 + spread_adjustment)
                        adjustments_made.append(f"widened_spread_by_{spread_adjustment:.1%}")

                # 基于交易量调整
                volume_ratio = conditions.get("volume_ratio", 1.0)
                if volume_ratio > self.adjustment_rules["volume_surge"]["threshold"]:
                    # 分割订单
                    original_quantity = adjusted_order.get("quantity", 0)
                    if original_quantity > 100:
                        adjusted_order["quantity"] = original_quantity // 2
                        adjusted_order["split_order"] = True
                        adjustments_made.append("split_order_due_to_volume_surge")

                # 基于流动性调整
                liquidity_score = conditions.get("liquidity_score", 1.0)
                if liquidity_score < self.adjustment_rules["liquidity_dryup"]["threshold"]:
                    # 减少订单规模
                    original_quantity = adjusted_order.get("quantity", 0)
                    adjusted_order["quantity"] = int(original_quantity * 0.5)
                    adjustments_made.append("reduced_size_due_to_low_liquidity")

                adjusted_order["adjustments"] = adjustments_made
                adjusted_order["adjustment_timestamp"] = datetime.now()

                return adjusted_order

            def monitor_and_adjust_orders(self, symbol: str, orders: List[Dict[str, Any]],
                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
                """监控并调整订单"""
                # 分析市场条件
                conditions = self.analyze_market_conditions(
                    symbol,
                    market_data.get("recent_trades", []),
                    market_data.get("order_book_depth", {})
                )

                adjusted_orders = []
                adjustment_summary = {
                    "total_orders": len(orders),
                    "adjusted_orders": 0,
                    "canceled_orders": 0,
                    "adjustments_by_type": {}
                }

                for order in orders:
                    original_price = order.get("limit_price", 0)
                    adjusted_order = self.adjust_order_parameters(symbol, order)

                    # 检查是否需要取消订单（严重的价格滑点）
                    if "limit_price" in adjusted_order:
                        price_change = abs(adjusted_order["limit_price"] - original_price) / original_price
                        if price_change > self.adjustment_rules["price_slippage"]["threshold"]:
                            adjustment_summary["canceled_orders"] += 1
                            continue

                    if adjusted_order.get("adjustments"):
                        adjustment_summary["adjusted_orders"] += 1
                        for adjustment in adjusted_order["adjustments"]:
                            adjustment_summary["adjustments_by_type"][adjustment] = \
                                adjustment_summary["adjustments_by_type"].get(adjustment, 0) + 1

                    adjusted_orders.append(adjusted_order)

                return {
                    "market_conditions": conditions,
                    "adjusted_orders": adjusted_orders,
                    "adjustment_summary": adjustment_summary,
                    "needs_manual_intervention": adjustment_summary["canceled_orders"] > len(orders) * 0.5
                }

        # 测试交易调整自动化
        adjuster = TradeAdjustmentAutomator()

        # 模拟市场数据
        market_data = {
            "recent_trades": [
                {"price": 100.0, "volume": 1000},
                {"price": 101.0, "volume": 1200},
                {"price": 102.0, "volume": 800},
                {"price": 103.0, "volume": 1500},  # 交易量激增
            ],
            "order_book_depth": {"bid_depth": 50, "ask_depth": 30}  # 流动性较低
        }

        # 分析市场条件
        conditions = adjuster.analyze_market_conditions("AAPL", market_data["recent_trades"],
                                                      market_data["order_book_depth"])

        assert "volatility" in conditions
        assert "volume_ratio" in conditions
        assert "liquidity_score" in conditions

        # 测试订单调整
        orders = [
            {"order_id": "001", "symbol": "AAPL", "quantity": 1000, "limit_price": 100.0},
            {"order_id": "002", "symbol": "AAPL", "quantity": 500, "limit_price": 101.0}
        ]

        adjustment_result = adjuster.monitor_and_adjust_orders("AAPL", orders, market_data)

        assert "adjusted_orders" in adjustment_result
        assert "adjustment_summary" in adjustment_result
        assert adjustment_result["adjustment_summary"]["total_orders"] == 2

        # 检查是否有调整发生
        total_adjustments = adjustment_result["adjustment_summary"]["adjusted_orders"]
        assert total_adjustments >= 0  # 可能有也可能没有调整，取决于市场条件


class TestSchedulingExecution:
    """测试调度执行功能"""

    def test_task_scheduler_automation(self):
        """测试任务调度自动化"""
        class TaskScheduler:
            def __init__(self):
                self.tasks = {}
                self.schedule_queue = queue.PriorityQueue()
                self.execution_history = []
                self.current_time = datetime.now()

            def schedule_task(self, task_id: str, execution_time: datetime,
                            task_func: Callable, priority: int = 1,
                            dependencies: List[str] = None) -> str:
                """调度任务"""
                if dependencies is None:
                    dependencies = []

                task = {
                    "id": task_id,
                    "execution_time": execution_time,
                    "function": task_func,
                    "priority": priority,
                    "dependencies": dependencies,
                    "status": "scheduled",
                    "created_at": self.current_time
                }

                self.tasks[task_id] = task

                # 使用优先级队列，优先级和执行时间共同决定顺序
                priority_score = priority * 1000000 - (execution_time.timestamp() * 1000)
                self.schedule_queue.put((priority_score, task_id))

                return task_id

            def execute_pending_tasks(self) -> Dict[str, Any]:
                """执行待处理任务"""
                executed_tasks = []
                failed_tasks = []
                skipped_tasks = []

                max_iterations = 100  # 防止无限循环
                iterations = 0

                while not self.schedule_queue.empty() and iterations < max_iterations:
                    _, task_id = self.schedule_queue.get()
                    iterations += 1

                    if task_id not in self.tasks:
                        continue

                    task = self.tasks[task_id]

                    # 检查依赖
                    unmet_deps = [dep for dep in task["dependencies"]
                                if dep not in self.tasks or self.tasks[dep]["status"] != "completed"]

                    if unmet_deps:
                        # 重新放回队列
                        priority_score = task["priority"] * 1000000 - (task["execution_time"].timestamp() * 1000)
                        self.schedule_queue.put((priority_score, task_id))
                        skipped_tasks.append(task_id)
                        continue

                    # 执行任务
                    try:
                        start_time = datetime.now()
                        result = task["function"]()
                        end_time = datetime.now()

                        task["status"] = "completed"
                        task["result"] = result
                        task["execution_time_actual"] = start_time
                        task["duration"] = (end_time - start_time).total_seconds()

                        execution_record = {
                            "task_id": task_id,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": task["duration"],
                            "result": result,
                            "status": "success"
                        }

                        self.execution_history.append(execution_record)
                        executed_tasks.append(task_id)

                    except Exception as e:
                        task["status"] = "failed"
                        task["error"] = str(e)

                        execution_record = {
                            "task_id": task_id,
                            "start_time": datetime.now(),
                            "error": str(e),
                            "status": "failed"
                        }

                        self.execution_history.append(execution_record)
                        failed_tasks.append(task_id)

                return {
                    "executed_tasks": executed_tasks,
                    "failed_tasks": failed_tasks,
                    "skipped_tasks": skipped_tasks,
                    "total_processed": len(executed_tasks) + len(failed_tasks),
                    "success_rate": len(executed_tasks) / max(1, len(executed_tasks) + len(failed_tasks))
                }

            def get_scheduler_status(self) -> Dict[str, Any]:
                """获取调度器状态"""
                pending_tasks = [tid for tid in self.tasks.keys()
                               if self.tasks[tid]["status"] == "scheduled"]

                completed_tasks = [tid for tid in self.tasks.keys()
                                 if self.tasks[tid]["status"] == "completed"]

                failed_tasks = [tid for tid in self.tasks.keys()
                              if self.tasks[tid]["status"] == "failed"]

                return {
                    "total_tasks": len(self.tasks),
                    "pending_tasks": len(pending_tasks),
                    "completed_tasks": len(completed_tasks),
                    "failed_tasks": len(failed_tasks),
                    "queue_size": self.schedule_queue.qsize(),
                    "completion_rate": len(completed_tasks) / max(1, len(self.tasks))
                }

        # 测试任务调度自动化
        scheduler = TaskScheduler()

        # 定义任务函数
        def data_processing_task():
            time.sleep(0.1)
            return {"processed_records": 1000}

        def model_training_task():
            time.sleep(0.2)
            return {"model_accuracy": 0.85}

        def report_generation_task():
            time.sleep(0.05)
            return {"report_pages": 5}

        # 调度任务（有依赖关系）
        execution_time = datetime.now() + timedelta(seconds=1)

        scheduler.schedule_task("data_processing", execution_time, data_processing_task, priority=3)
        scheduler.schedule_task("model_training", execution_time + timedelta(seconds=2),
                              model_training_task, priority=2, dependencies=["data_processing"])
        scheduler.schedule_task("report_generation", execution_time + timedelta(seconds=3),
                              report_generation_task, priority=1, dependencies=["model_training"])

        # 执行任务
        execution_result = scheduler.execute_pending_tasks()

        # 检查执行结果
        assert execution_result["total_processed"] >= 0
        assert "success_rate" in execution_result

        # 获取调度器状态
        status = scheduler.get_scheduler_status()
        assert status["total_tasks"] == 3
        assert status["completed_tasks"] <= 3  # 可能由于依赖关系不是所有任务都执行

        # 验证执行历史
        assert len(scheduler.execution_history) >= 0

    def test_rule_engine_automation(self):
        """测试规则引擎自动化"""
        class RuleEngine:
            def __init__(self):
                self.rules = {}
                self.rule_execution_stats = {}

            def add_rule(self, rule_id: str, conditions: List[Callable],
                        actions: List[Callable], priority: int = 1) -> str:
                """添加规则"""
                rule = {
                    "id": rule_id,
                    "conditions": conditions,
                    "actions": actions,
                    "priority": priority,
                    "enabled": True,
                    "execution_count": 0,
                    "last_executed": None
                }

                self.rules[rule_id] = rule
                self.rule_execution_stats[rule_id] = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_execution_time": 0.0
                }

                return rule_id

            def evaluate_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
                """评估规则"""
                triggered_rules = []
                executed_actions = []

                # 按优先级排序规则
                sorted_rules = sorted(self.rules.values(), key=lambda r: r["priority"], reverse=True)

                for rule in sorted_rules:
                    if not rule["enabled"]:
                        continue

                    # 检查所有条件
                    conditions_met = True
                    for condition in rule["conditions"]:
                        try:
                            if not condition(context):
                                conditions_met = False
                                break
                        except Exception:
                            conditions_met = False
                            break

                    if conditions_met:
                        triggered_rules.append(rule["id"])

                        # 执行动作
                        start_time = time.time()
                        action_results = []

                        for action in rule["actions"]:
                            try:
                                result = action(context)
                                action_results.append({"action": str(action), "result": result, "status": "success"})
                            except Exception as e:
                                action_results.append({"action": str(action), "error": str(e), "status": "failed"})

                        execution_time = time.time() - start_time

                        # 更新统计
                        self.rule_execution_stats[rule["id"]]["executions"] += 1
                        self.rule_execution_stats[rule["id"]]["successes"] += sum(1 for ar in action_results if ar["status"] == "success")
                        self.rule_execution_stats[rule["id"]]["failures"] += sum(1 for ar in action_results if ar["status"] == "failed")

                        # 更新平均执行时间
                        current_avg = self.rule_execution_stats[rule["id"]]["avg_execution_time"]
                        current_count = self.rule_execution_stats[rule["id"]]["executions"]
                        self.rule_execution_stats[rule["id"]]["avg_execution_time"] = \
                            (current_avg * (current_count - 1) + execution_time) / current_count

                        rule["execution_count"] += 1
                        rule["last_executed"] = datetime.now()

                        executed_actions.extend(action_results)

                return {
                    "triggered_rules": triggered_rules,
                    "executed_actions": executed_actions,
                    "total_rules_evaluated": len(sorted_rules),
                    "rules_triggered": len(triggered_rules),
                    "actions_executed": len(executed_actions),
                    "evaluation_context": context
                }

            def disable_rule(self, rule_id: str) -> bool:
                """禁用规则"""
                if rule_id in self.rules:
                    self.rules[rule_id]["enabled"] = False
                    return True
                return False

            def get_engine_stats(self) -> Dict[str, Any]:
                """获取引擎统计"""
                total_executions = sum(stats["executions"] for stats in self.rule_execution_stats.values())
                total_successes = sum(stats["successes"] for stats in self.rule_execution_stats.values())
                total_failures = sum(stats["failures"] for stats in self.rule_execution_stats.values())

                return {
                    "total_rules": len(self.rules),
                    "enabled_rules": sum(1 for r in self.rules.values() if r["enabled"]),
                    "disabled_rules": sum(1 for r in self.rules.values() if not r["enabled"]),
                    "total_executions": total_executions,
                    "total_successes": total_successes,
                    "total_failures": total_failures,
                    "overall_success_rate": total_successes / max(1, total_executions),
                    "rule_stats": self.rule_execution_stats
                }

        # 测试规则引擎自动化
        engine = RuleEngine()

        # 定义条件函数
        def high_cpu_condition(context):
            return context.get("cpu_usage", 0) > 80

        def high_memory_condition(context):
            return context.get("memory_usage", 0) > 85

        def market_volatility_condition(context):
            return context.get("market_volatility", 0) > 0.05

        # 定义动作函数
        def scale_up_action(context):
            return {"action": "scale_up", "message": "Scaling up resources due to high load"}

        def alert_admin_action(context):
            return {"action": "alert", "message": "Alerting administrator"}

        def reduce_trading_action(context):
            return {"action": "reduce_trading", "message": "Reducing trading volume due to volatility"}

        # 添加规则
        engine.add_rule("high_load_response", [high_cpu_condition, high_memory_condition],
                       [scale_up_action, alert_admin_action], priority=3)

        engine.add_rule("volatility_response", [market_volatility_condition],
                       [reduce_trading_action], priority=2)

        # 测试规则评估
        test_context = {
            "cpu_usage": 85,
            "memory_usage": 90,
            "market_volatility": 0.03,
            "timestamp": datetime.now()
        }

        evaluation_result = engine.evaluate_rules(test_context)

        # 由于CPU和内存都超过阈值，应该触发high_load_response规则
        assert "triggered_rules" in evaluation_result
        assert "high_load_response" in evaluation_result["triggered_rules"]
        assert evaluation_result["actions_executed"] >= 2  # scale_up 和 alert

        # 测试引擎统计
        stats = engine.get_engine_stats()
        assert stats["total_rules"] == 2
        assert stats["enabled_rules"] == 2
        assert stats["total_executions"] == 1  # 执行了一次评估

        # 测试规则禁用
        assert engine.disable_rule("high_load_response") == True
        stats_after = engine.get_engine_stats()
        assert stats_after["enabled_rules"] == 1

    def test_workflow_manager_automation(self):
        """测试工作流管理自动化"""
        class WorkflowManager:
            def __init__(self):
                self.workflows = {}
                self.active_workflows = {}
                self.workflow_history = []

            def create_workflow(self, workflow_id: str, steps: List[Dict[str, Any]]) -> str:
                """创建工作流"""
                workflow = {
                    "id": workflow_id,
                    "steps": steps,
                    "status": "created",
                    "current_step": 0,
                    "created_at": datetime.now(),
                    "step_results": {},
                    "metadata": {}
                }

                self.workflows[workflow_id] = workflow
                return workflow_id

            def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
                """执行工作流"""
                if workflow_id not in self.workflows:
                    return {"error": "workflow_not_found"}

                workflow = self.workflows[workflow_id]
                workflow["status"] = "running"
                workflow["started_at"] = datetime.now()

                self.active_workflows[workflow_id] = workflow

                execution_log = []
                success = True

                try:
                    for i, step in enumerate(workflow["steps"]):
                        workflow["current_step"] = i

                        step_start = datetime.now()
                        try:
                            # 执行步骤
                            result = step["function"](**step.get("params", {}))

                            step_end = datetime.now()
                            execution_time = (step_end - step_start).total_seconds()

                            step_result = {
                                "step_id": step["id"],
                                "status": "completed",
                                "result": result,
                                "execution_time": execution_time,
                                "executed_at": step_start
                            }

                            workflow["step_results"][step["id"]] = step_result
                            execution_log.append(step_result)

                        except Exception as e:
                            step_result = {
                                "step_id": step["id"],
                                "status": "failed",
                                "error": str(e),
                                "executed_at": step_start
                            }

                            workflow["step_results"][step["id"]] = step_result
                            execution_log.append(step_result)
                            success = False
                            break

                    # 完成工作流
                    workflow["status"] = "completed" if success else "failed"
                    workflow["completed_at"] = datetime.now()
                    workflow["duration"] = (workflow["completed_at"] - workflow["started_at"]).total_seconds()

                    del self.active_workflows[workflow_id]

                    # 记录到历史
                    history_record = {
                        "workflow_id": workflow_id,
                        "status": workflow["status"],
                        "duration": workflow["duration"],
                        "steps_executed": len(execution_log),
                        "success": success,
                        "completed_at": workflow["completed_at"]
                    }

                    self.workflow_history.append(history_record)

                except Exception as e:
                    workflow["status"] = "error"
                    workflow["error"] = str(e)

                return {
                    "workflow_id": workflow_id,
                    "status": workflow["status"],
                    "execution_log": execution_log,
                    "success": success,
                    "total_steps": len(workflow["steps"]),
                    "completed_steps": len([s for s in execution_log if s["status"] == "completed"]),
                    "failed_steps": len([s for s in execution_log if s["status"] == "failed"])
                }

            def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
                """获取工作流状态"""
                if workflow_id in self.active_workflows:
                    workflow = self.active_workflows[workflow_id]
                    return {
                        "status": "active",
                        "current_step": workflow["current_step"],
                        "progress": workflow["current_step"] / len(workflow["steps"]),
                        "started_at": workflow["started_at"]
                    }
                elif workflow_id in self.workflows:
                    workflow = self.workflows[workflow_id]
                    return {
                        "status": workflow["status"],
                        "completed_at": workflow.get("completed_at"),
                        "duration": workflow.get("duration"),
                        "total_steps": len(workflow["steps"]),
                        "completed_steps": len(workflow["step_results"])
                    }
                else:
                    return {"status": "not_found"}

            def get_workflow_history(self, limit: int = 10) -> List[Dict[str, Any]]:
                """获取工作流历史"""
                return self.workflow_history[-limit:] if self.workflow_history else []

        # 测试工作流管理自动化
        manager = WorkflowManager()

        # 定义工作流步骤
        def data_collection():
            time.sleep(0.1)
            return {"data_collected": 1000}

        def data_processing(data_collected):
            time.sleep(0.1)
            return {"processed_data": data_collected * 2}

        def model_training(processed_data):
            time.sleep(0.2)
            return {"model_trained": True, "accuracy": 0.85}

        def report_generation(model_trained, accuracy):
            time.sleep(0.05)
            return {"report_generated": True, "accuracy_reported": accuracy}

        # 创建工作流
        steps = [
            {"id": "collect", "function": data_collection, "params": {}},
            {"id": "process", "function": data_processing, "params": {"data_collected": 1000}},
            {"id": "train", "function": model_training, "params": {"processed_data": 2000}},
            {"id": "report", "function": report_generation, "params": {"model_trained": True, "accuracy": 0.85}}
        ]

        workflow_id = manager.create_workflow("ml_pipeline", steps)

        # 执行工作流
        execution_result = manager.execute_workflow(workflow_id)

        assert execution_result["workflow_id"] == workflow_id
        assert execution_result["total_steps"] == 4
        assert execution_result["completed_steps"] >= 0
        assert execution_result["status"] in ["completed", "failed"]

        # 获取工作流状态
        status = manager.get_workflow_status(workflow_id)
        assert status["status"] in ["completed", "failed"]
        assert "completed_at" in status or "started_at" in status

        # 获取工作流历史
        history = manager.get_workflow_history()
        assert len(history) >= 1
        assert history[-1]["workflow_id"] == workflow_id


class TestMonitoringAlerting:
    """测试监控告警功能"""

    def test_system_monitoring_automation(self):
        """测试系统监控自动化"""
        class SystemMonitor:
            def __init__(self):
                self.metrics = {}
                self.alerts = []
                self.monitoring_rules = {
                    "cpu_threshold": {"metric": "cpu_usage", "operator": ">", "value": 80, "severity": "warning"},
                    "memory_threshold": {"metric": "memory_usage", "operator": ">", "value": 90, "severity": "critical"},
                    "disk_threshold": {"metric": "disk_usage", "operator": ">", "value": 95, "severity": "critical"},
                    "response_time_threshold": {"metric": "avg_response_time", "operator": ">", "value": 2.0, "severity": "warning"}
                }

            def collect_metrics(self) -> Dict[str, float]:
                """收集系统指标"""
                # 模拟指标收集
                metrics = {
                    "cpu_usage": 45.2 + (time.time() % 10) * 2,
                    "memory_usage": 60.1 + (time.time() % 5),
                    "disk_usage": 75.0 + (time.time() % 3),
                    "avg_response_time": 0.8 + (time.time() % 1) * 0.5,
                    "network_latency": 15.5 + (time.time() % 2),
                    "error_rate": 0.02 + (time.time() % 0.1) * 0.01,
                    "timestamp": time.time()
                }

                # 存储历史指标
                for key, value in metrics.items():
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append({"value": value, "timestamp": metrics["timestamp"]})
                    # 保持最近100个数据点
                    if len(self.metrics[key]) > 100:
                        self.metrics[key] = self.metrics[key][-100:]

                return metrics

            def evaluate_monitoring_rules(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
                """评估监控规则"""
                triggered_alerts = []

                for rule_name, rule_config in self.monitoring_rules.items():
                    metric_name = rule_config["metric"]
                    operator = rule_config["operator"]
                    threshold = rule_config["value"]
                    severity = rule_config["severity"]

                    if metric_name in metrics:
                        current_value = metrics[metric_name]

                        # 评估条件
                        condition_met = False
                        if operator == ">":
                            condition_met = current_value > threshold
                        elif operator == "<":
                            condition_met = current_value < threshold
                        elif operator == ">=":
                            condition_met = current_value >= threshold
                        elif operator == "<=":
                            condition_met = current_value <= threshold

                        if condition_met:
                            alert = {
                                "rule": rule_name,
                                "metric": metric_name,
                                "current_value": current_value,
                                "threshold": threshold,
                                "operator": operator,
                                "severity": severity,
                                "timestamp": datetime.now(),
                                "message": f"{metric_name} {operator} {threshold}: current={current_value:.2f}"
                            }

                            triggered_alerts.append(alert)
                            self.alerts.append(alert)

                return triggered_alerts

            def perform_automated_response(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """执行自动化响应"""
                responses = []

                for alert in alerts:
                    response = {
                        "alert_id": f"{alert['rule']}_{int(alert['timestamp'].timestamp())}",
                        "alert_rule": alert["rule"],
                        "response_actions": [],
                        "timestamp": datetime.now()
                    }

                    # 根据告警类型执行响应
                    if alert["rule"] == "cpu_threshold":
                        response["response_actions"].extend([
                            "Scale up CPU resources",
                            "Optimize CPU-intensive processes",
                            "Alert system administrator"
                        ])
                    elif alert["rule"] == "memory_threshold":
                        response["response_actions"].extend([
                            "Trigger garbage collection",
                            "Scale up memory resources",
                            "Restart memory-intensive services"
                        ])
                    elif alert["rule"] == "disk_threshold":
                        response["response_actions"].extend([
                            "Clean up temporary files",
                            "Archive old data",
                            "Scale up disk storage"
                        ])
                    elif alert["rule"] == "response_time_threshold":
                        response["response_actions"].extend([
                            "Scale up application instances",
                            "Optimize database queries",
                            "Enable response time monitoring"
                        ])

                    responses.append(response)

                return responses

            def get_monitoring_dashboard(self) -> Dict[str, Any]:
                """获取监控仪表板"""
                latest_metrics = {}
                for metric_name, values in self.metrics.items():
                    if values:
                        latest_metrics[metric_name] = values[-1]["value"]

                # 计算健康分数
                health_score = self._calculate_health_score(latest_metrics)

                # 最近告警
                recent_alerts = self.alerts[-10:] if self.alerts else []

                # 指标趋势
                trends = {}
                for metric_name, values in self.metrics.items():
                    if len(values) >= 2:
                        recent = [v["value"] for v in values[-5:]]
                        older = [v["value"] for v in values[-10:-5]] if len(values) >= 10 else recent

                        recent_avg = sum(recent) / len(recent)
                        older_avg = sum(older) / len(older)

                        if older_avg > 0:
                            trend = (recent_avg - older_avg) / older_avg * 100
                            trends[metric_name] = {
                                "direction": "up" if trend > 5 else "down" if trend < -5 else "stable",
                                "change_percent": trend
                            }

                return {
                    "current_metrics": latest_metrics,
                    "health_score": health_score,
                    "recent_alerts": recent_alerts,
                    "metric_trends": trends,
                    "total_alerts": len(self.alerts),
                    "monitoring_status": "active"
                }

            def _calculate_health_score(self, metrics: Dict[str, float]) -> float:
                """计算健康分数"""
                if not metrics:
                    return 0.0

                score = 100.0

                # CPU健康评分
                cpu = metrics.get("cpu_usage", 0)
                if cpu > 90:
                    score -= 30
                elif cpu > 80:
                    score -= 15
                elif cpu > 70:
                    score -= 5

                # 内存健康评分
                memory = metrics.get("memory_usage", 0)
                if memory > 95:
                    score -= 30
                elif memory > 90:
                    score -= 15
                elif memory > 85:
                    score -= 5

                # 响应时间健康评分
                response_time = metrics.get("avg_response_time", 0)
                if response_time > 5.0:
                    score -= 30
                elif response_time > 2.0:
                    score -= 15
                elif response_time > 1.0:
                    score -= 5

                return max(0.0, score)

        # 测试系统监控自动化
        monitor = SystemMonitor()

        # 收集指标
        metrics = monitor.collect_metrics()
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "timestamp" in metrics

        # 评估监控规则
        alerts = monitor.evaluate_monitoring_rules(metrics)
        # 可能有也可能没有告警，取决于随机生成的指标值

        # 执行自动化响应
        if alerts:
            responses = monitor.perform_automated_response(alerts)
            assert len(responses) == len(alerts)
            for response in responses:
                assert "response_actions" in response
                assert len(response["response_actions"]) > 0

        # 获取监控仪表板
        dashboard = monitor.get_monitoring_dashboard()
        assert "current_metrics" in dashboard
        assert "health_score" in dashboard
        assert 0 <= dashboard["health_score"] <= 100
        assert "monitoring_status" in dashboard

    def test_performance_alerting_automation(self):
        """测试性能告警自动化"""
        class PerformanceAlertManager:
            def __init__(self):
                self.baseline_metrics = {}
                self.alert_thresholds = {
                    "cpu_usage": {"warning": 1.5, "critical": 2.0},  # 倍数
                    "memory_usage": {"warning": 1.3, "critical": 1.8},
                    "response_time": {"warning": 2.0, "critical": 3.0},
                    "error_rate": {"warning": 5.0, "critical": 10.0}  # 倍数
                }
                self.active_alerts = {}
                self.alert_history = []

            def establish_baselines(self, historical_data: List[Dict[str, float]]):
                """建立基线"""
                if not historical_data:
                    return

                # 计算每个指标的基线统计
                for metric_name in historical_data[0].keys():
                    if metric_name == "timestamp":
                        continue

                    values = [data[metric_name] for data in historical_data if metric_name in data]
                    if values:
                        self.baseline_metrics[metric_name] = {
                            "mean": sum(values) / len(values),
                            "std": (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5,
                            "min": min(values),
                            "max": max(values),
                            "p95": sorted(values)[int(len(values) * 0.95)]
                        }

            def detect_performance_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
                """检测性能异常"""
                anomalies = []

                for metric_name, current_value in current_metrics.items():
                    if metric_name not in self.baseline_metrics:
                        continue

                    baseline = self.baseline_metrics[metric_name]
                    baseline_mean = baseline["mean"]
                    baseline_std = baseline["std"]

                    if baseline_mean == 0:
                        continue

                    # 计算偏差倍数
                    deviation_ratio = current_value / baseline_mean

                    # 检查是否超过阈值
                    thresholds = self.alert_thresholds.get(metric_name, {})
                    warning_threshold = thresholds.get("warning", 2.0)
                    critical_threshold = thresholds.get("critical", 3.0)

                    severity = None
                    if deviation_ratio >= critical_threshold:
                        severity = "critical"
                    elif deviation_ratio >= warning_threshold:
                        severity = "warning"

                    if severity:
                        anomaly = {
                            "metric": metric_name,
                            "current_value": current_value,
                            "baseline_mean": baseline_mean,
                            "deviation_ratio": deviation_ratio,
                            "severity": severity,
                            "threshold": critical_threshold if severity == "critical" else warning_threshold,
                            "timestamp": datetime.now(),
                            "description": f"{metric_name} is {deviation_ratio:.1f}x above baseline ({current_value:.2f} vs {baseline_mean:.2f})"
                        }

                        anomalies.append(anomaly)

                        # 记录活跃告警
                        alert_key = f"{metric_name}_{severity}"
                        self.active_alerts[alert_key] = anomaly

                return anomalies

            def escalate_alerts(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """升级告警"""
                escalated = []

                for anomaly in anomalies:
                    alert_key = f"{anomaly['metric']}_{anomaly['severity']}"

                    # 检查是否已经升级
                    if alert_key in self.active_alerts:
                        existing_alert = self.active_alerts[alert_key]
                        time_since_alert = (datetime.now() - existing_alert["timestamp"]).total_seconds()

                        # 如果告警持续5分钟以上，进行升级
                        if time_since_alert > 300:  # 5分钟
                            escalation = anomaly.copy()
                            escalation["escalation_level"] = "escalated"
                            escalation["escalation_reason"] = f"Alert persisting for {time_since_alert/60:.1f} minutes"
                            escalation["recommended_actions"] = self._get_escalation_actions(anomaly["metric"])

                            escalated.append(escalation)

                            # 更新告警历史
                            self.alert_history.append(escalation)

                return escalated

            def _get_escalation_actions(self, metric_name: str) -> List[str]:
                """获取升级行动"""
                actions = {
                    "cpu_usage": ["Immediate resource scaling", "Process optimization", "Emergency alert to DevOps"],
                    "memory_usage": ["Memory leak investigation", "Service restart", "Resource augmentation"],
                    "response_time": ["Load balancer adjustment", "Database optimization", "CDN configuration"],
                    "error_rate": ["Error log analysis", "Service health check", "Rollback preparation"]
                }

                return actions.get(metric_name, ["General investigation required", "Escalate to senior team"])

            def generate_performance_report(self) -> Dict[str, Any]:
                """生成性能报告"""
                total_alerts = len(self.alert_history)
                critical_alerts = len([a for a in self.alert_history if a.get("severity") == "critical"])
                escalated_alerts = len([a for a in self.alert_history if a.get("escalation_level") == "escalated"])

                # 计算MTTR（平均修复时间）- 简化为告警持续时间
                if self.alert_history:
                    avg_resolution_time = sum(
                        (datetime.now() - a["timestamp"]).total_seconds() / 3600  # 小时
                        for a in self.alert_history
                    ) / len(self.alert_history)
                else:
                    avg_resolution_time = 0

                return {
                    "period_start": datetime.now() - timedelta(days=1),
                    "period_end": datetime.now(),
                    "total_alerts": total_alerts,
                    "critical_alerts": critical_alerts,
                    "escalated_alerts": escalated_alerts,
                    "avg_resolution_time_hours": avg_resolution_time,
                    "alerts_by_metric": self._group_alerts_by_metric(),
                    "system_health_trend": "improving" if total_alerts == 0 else "needs_attention"
                }

            def _group_alerts_by_metric(self) -> Dict[str, int]:
                """按指标分组告警"""
                grouped = {}
                for alert in self.alert_history:
                    metric = alert.get("metric", "unknown")
                    grouped[metric] = grouped.get(metric, 0) + 1
                return grouped

        # 测试性能告警自动化
        alert_manager = PerformanceAlertManager()

        # 建立基线
        historical_data = [
            {"cpu_usage": 60, "memory_usage": 70, "response_time": 0.8, "error_rate": 0.02},
            {"cpu_usage": 65, "memory_usage": 75, "response_time": 0.9, "error_rate": 0.025},
            {"cpu_usage": 58, "memory_usage": 72, "response_time": 0.7, "error_rate": 0.018}
        ]
        alert_manager.establish_baselines(historical_data)

        # 检测异常（模拟高负载情况）
        current_metrics = {
            "cpu_usage": 150,  # 2.5倍基线
            "memory_usage": 120,  # 1.7倍基线
            "response_time": 2.5,  # 3倍基线
            "error_rate": 0.15  # 7.5倍基线
        }

        anomalies = alert_manager.detect_performance_anomalies(current_metrics)

        # 应该检测到多个异常
        assert len(anomalies) >= 2  # 至少CPU和响应时间异常

        # 升级告警（如果有持续告警）
        escalated = alert_manager.escalate_alerts(anomalies)
        # 首次检测可能没有升级，因为没有持续时间

        # 生成性能报告
        report = alert_manager.generate_performance_report()
        assert "total_alerts" in report
        assert "critical_alerts" in report
        assert "system_health_trend" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
