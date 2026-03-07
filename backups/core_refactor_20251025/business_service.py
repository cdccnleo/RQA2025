#!/usr/bin/env python3
"""
RQA2025核心业务服务

整合各个业务模块，提供完整的业务流程处理能力。
包括策略执行、风险控制、订单管理、数据处理等核心业务逻辑。
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import uuid

from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
from ...database_service import get_database_service
# Strategy engine import (optional)
try:
    from src.strategy.core.strategy_service import UnifiedStrategyService as StrategyEngine
    STRATEGY_ENGINE_AVAILABLE = True
except ImportError:
    StrategyEngine = None
    STRATEGY_ENGINE_AVAILABLE = False
from src.trading.order_manager import OrderManager
from src.trading.execution_engine import ExecutionEngine
from src.risk.risk_manager import RiskManager
from src.monitoring.monitoring_system import MonitoringSystem
from src.data.data_manager import DataManager

logger = get_logger(__name__)


class BusinessProcessStatus(Enum):
    """业务流程状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BusinessProcessType(Enum):
    """业务流程类型"""
    STRATEGY_EXECUTION = "strategy_execution"
    ORDER_PROCESSING = "order_processing"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_ASSESSMENT = "risk_assessment"
    DATA_PROCESSING = "data_processing"
    MARKET_ANALYSIS = "market_analysis"


@dataclass
class BusinessProcess:
    """业务流程"""
    process_id: str
    process_type: BusinessProcessType
    user_id: int
    status: BusinessProcessStatus = BusinessProcessStatus.PENDING
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TradingStrategy:
    """交易策略"""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    user_id: int
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class BusinessService:
    """核心业务服务"""

    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        self.db_service = None

        # 业务组件
        self.strategy_engine = None
        self.order_manager = None
        self.execution_engine = None
        self.risk_manager = None
        self.monitoring_system = None
        self.data_manager = None
        self.strategy_available = STRATEGY_ENGINE_AVAILABLE

        # 业务流程存储
        self.active_processes: Dict[str, BusinessProcess] = {}

        # 策略存储
        self.active_strategies: Dict[str, TradingStrategy] = {}

    async def initialize(self):
        """初始化业务服务"""
        try:
            # 初始化数据库服务
            self.db_service = await get_database_service()

            # 初始化业务组件（带错误处理）
            try:
                if self.strategy_available and StrategyEngine:
                    self.strategy_engine = StrategyEngine()
                else:
                    self.strategy_engine = None
                    logger.info("策略引擎不可用，使用模拟模式")
            except Exception as e:
                logger.warning(f"策略引擎初始化失败: {e}")
                self.strategy_engine = None

            try:
                self.order_manager = OrderManager()
            except Exception as e:
                logger.warning(f"订单管理器初始化失败: {e}")
                self.order_manager = None

            try:
                self.execution_engine = ExecutionEngine()
            except Exception as e:
                logger.warning(f"执行引擎初始化失败: {e}")
                self.execution_engine = None

            try:
                self.risk_manager = RiskManager()
            except Exception as e:
                logger.warning(f"风险管理器初始化失败: {e}")
                self.risk_manager = None

            try:
                self.monitoring_system = MonitoringSystem()
            except Exception as e:
                logger.warning(f"监控系统初始化失败: {e}")
                self.monitoring_system = None

            try:
                self.data_manager = DataManager()
            except Exception as e:
                logger.warning(f"数据管理器初始化失败: {e}")
                self.data_manager = None

            # 加载活跃策略
            await self._load_active_strategies()

            logger.info("业务服务初始化完成")

        except Exception as e:
            logger.error(f"业务服务初始化失败: {e}")
            raise

    async def _load_active_strategies(self):
        """加载活跃策略"""
        try:
            # 从数据库加载策略（如果有的话）
            # 这里使用内存存储作为示例
            self.active_strategies = {}

        except Exception as e:
            logger.warning(f"加载活跃策略失败: {e}")

    # ==================== 策略管理 ====================

    async def create_strategy(self, user_id: int, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建交易策略"""
        try:
            strategy = TradingStrategy(
                strategy_id=str(uuid.uuid4()),
                name=strategy_data["name"],
                description=strategy_data.get("description", ""),
                parameters=strategy_data.get("parameters", {}),
                user_id=user_id
            )

            # 存储策略
            self.active_strategies[strategy.strategy_id] = strategy

            # 如果有策略引擎，注册策略
            if self.strategy_engine:
                try:
                    await self.strategy_engine.register_strategy(strategy.strategy_id, strategy.parameters)
                except Exception as e:
                    logger.warning(f"策略引擎注册失败: {e}")

            return {
                "success": True,
                "strategy_id": strategy.strategy_id,
                "strategy": {
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "description": strategy.description,
                    "parameters": strategy.parameters,
                    "is_active": strategy.is_active,
                    "created_at": strategy.created_at.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"创建策略失败: {e}")
            return {"success": False, "error": str(e)}

    async def execute_strategy(self, strategy_id: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易策略"""
        try:
            if strategy_id not in self.active_strategies:
                return {"success": False, "error": "策略不存在或未激活"}

            strategy = self.active_strategies[strategy_id]

            # 创建业务流程
            process = BusinessProcess(
                process_id=str(uuid.uuid4()),
                process_type=BusinessProcessType.STRATEGY_EXECUTION,
                user_id=strategy.user_id,
                parameters={"strategy_id": strategy_id, "market_data": market_data}
            )

            self.active_processes[process.process_id] = process

            # 异步执行策略
            asyncio.create_task(self._execute_strategy_async(process, strategy, market_data))

            return {
                "success": True,
                "process_id": process.process_id,
                "message": "策略执行已启动"
            }

        except Exception as e:
            logger.error(f"执行策略失败: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_strategy_async(self, process: BusinessProcess, strategy: TradingStrategy, market_data: Dict[str, Any]):
        """异步执行策略"""
        try:
            process.status = BusinessProcessStatus.RUNNING
            process.started_at = datetime.now()

            # 更新进度
            process.progress = 0.1
            process.steps.append({"step": "初始化", "status": "completed",
                                 "timestamp": datetime.now().isoformat()})

            # 执行策略逻辑
            if self.strategy_engine:
                try:
                    # 调用策略引擎
                    signals = await self.strategy_engine.generate_signals(strategy.strategy_id, market_data)

                    process.progress = 0.5
                    process.steps.append({"step": "信号生成", "status": "completed",
                                         "signals": signals, "timestamp": datetime.now().isoformat()})

                    # 基于信号生成订单
                    orders = []
                    for signal in signals:
                        if signal.get("action") in ["buy", "sell"]:
                            order = {
                                "symbol": signal["symbol"],
                                "quantity": signal.get("quantity", 100),
                                "price": signal.get("price", market_data.get("price", 50.0)),
                                "order_type": "market",
                                "side": signal["action"]
                            }
                            orders.append(order)

                    process.progress = 0.8
                    process.steps.append({"step": "订单生成", "status": "completed",
                                         "orders": orders, "timestamp": datetime.now().isoformat()})

                    # 执行订单
                    executed_orders = []
                    for order in orders:
                        if self.execution_engine:
                            try:
                                result = await self.execution_engine.execute_order(order)
                                executed_orders.append(result)
                            except Exception as e:
                                logger.warning(f"订单执行失败: {e}")
                                executed_orders.append({"order": order, "error": str(e)})

                    process.progress = 1.0
                    process.steps.append({"step": "订单执行", "status": "completed",
                                         "executed_orders": executed_orders, "timestamp": datetime.now().isoformat()})

                    # 完成流程
                    process.status = BusinessProcessStatus.COMPLETED
                    process.completed_at = datetime.now()
                    process.results = {
                        "signals": signals,
                        "orders": orders,
                        "executed_orders": executed_orders,
                        "total_signals": len(signals),
                        "total_orders": len(orders),
                        "successful_executions": len([o for o in executed_orders if not o.get("error")])
                    }

                except Exception as e:
                    logger.error(f"策略引擎执行失败: {e}")
                    process.status = BusinessProcessStatus.FAILED
                    process.error_message = str(e)
            else:
                # 模拟策略执行
                process.progress = 0.5
                process.steps.append({"step": "信号生成", "status": "completed",
                                     "signals": [], "timestamp": datetime.now().isoformat()})

                process.progress = 1.0
                process.steps.append({"step": "订单生成", "status": "completed",
                                     "orders": [], "timestamp": datetime.now().isoformat()})

                process.status = BusinessProcessStatus.COMPLETED
                process.completed_at = datetime.now()
                process.results = {
                    "signals": [],
                    "orders": [],
                    "executed_orders": [],
                    "message": "模拟执行完成（无真实策略引擎）"
                }

            # 更新策略最后执行时间
            strategy.last_executed = datetime.now()

        except Exception as e:
            logger.error(f"策略异步执行失败: {e}")
            process.status = BusinessProcessStatus.FAILED
            process.error_message = str(e)
            process.completed_at = datetime.now()

    # ==================== 订单处理 ====================

    async def process_order(self, user_id: int, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单"""
        try:
            # 创建业务流程
            process = BusinessProcess(
                process_id=str(uuid.uuid4()),
                process_type=BusinessProcessType.ORDER_PROCESSING,
                user_id=user_id,
                parameters=order_data
            )

            self.active_processes[process.process_id] = process

            # 异步处理订单
            asyncio.create_task(self._process_order_async(process))

            return {
                "success": True,
                "process_id": process.process_id,
                "message": "订单处理已启动"
            }

        except Exception as e:
            logger.error(f"处理订单失败: {e}")
            return {"success": False, "error": str(e)}

    async def _process_order_async(self, process: BusinessProcess):
        """异步处理订单"""
        try:
            process.status = BusinessProcessStatus.RUNNING
            process.started_at = datetime.now()

            order_data = process.parameters

            # 步骤1: 订单验证
            process.progress = 0.2
            process.steps.append({"step": "订单验证", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            # 风险检查
            if self.risk_manager:
                try:
                    risk_check = await self.risk_manager.check_order_risk(process.user_id, order_data)
                    if not risk_check.get("approved", True):
                        process.status = BusinessProcessStatus.FAILED
                        process.error_message = risk_check.get("reason", "风险检查未通过")
                        process.completed_at = datetime.now()
                        return
                except Exception as e:
                    logger.warning(f"风险检查失败: {e}")

            process.steps[-1]["status"] = "completed"

            # 步骤2: 创建订单
            process.progress = 0.5
            process.steps.append({"step": "创建订单", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            if self.db_service:
                create_result = await self.db_service.create_order(process.user_id, order_data)
                if not create_result["success"]:
                    process.status = BusinessProcessStatus.FAILED
                    process.error_message = create_result.get("error", "订单创建失败")
                    process.completed_at = datetime.now()
                    return

                order_info = create_result["order"]
            else:
                order_info = {"order_id": str(uuid.uuid4()), **order_data}

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["order_info"] = order_info

            # 步骤3: 执行订单
            process.progress = 0.8
            process.steps.append({"step": "执行订单", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            if self.execution_engine:
                try:
                    execution_result = await self.execution_engine.execute_order(order_data)
                    process.steps[-1]["execution_result"] = execution_result
                except Exception as e:
                    logger.warning(f"订单执行失败: {e}")
                    process.steps[-1]["execution_error"] = str(e)

            process.steps[-1]["status"] = "completed"

            # 步骤4: 更新持仓
            process.progress = 1.0
            process.steps.append({"step": "更新持仓", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            if self.db_service:
                try:
                    quantity = order_data["quantity"] if order_data["side"] == "buy" else - \
                        order_data["quantity"]
                    await self.db_service.update_position(
                        process.user_id,
                        order_data["symbol"],
                        quantity,
                        order_data["price"]
                    )
                except Exception as e:
                    logger.warning(f"持仓更新失败: {e}")

            process.steps[-1]["status"] = "completed"

            # 完成流程
            process.status = BusinessProcessStatus.COMPLETED
            process.completed_at = datetime.now()
            process.results = {
                "order_info": order_info,
                "execution_status": "completed",
                "message": "订单处理完成"
            }

        except Exception as e:
            logger.error(f"订单异步处理失败: {e}")
            process.status = BusinessProcessStatus.FAILED
            process.error_message = str(e)
            process.completed_at = datetime.now()

    # ==================== 投资组合管理 ====================

    async def rebalance_portfolio(self, user_id: int, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """投资组合再平衡"""
        try:
            # 创建业务流程
            process = BusinessProcess(
                process_id=str(uuid.uuid4()),
                process_type=BusinessProcessType.PORTFOLIO_REBALANCE,
                user_id=user_id,
                parameters={"target_allocation": target_allocation}
            )

            self.active_processes[process.process_id] = process

            # 异步执行再平衡
            asyncio.create_task(self._rebalance_portfolio_async(process))

            return {
                "success": True,
                "process_id": process.process_id,
                "message": "组合再平衡已启动"
            }

        except Exception as e:
            logger.error(f"组合再平衡失败: {e}")
            return {"success": False, "error": str(e)}

    async def _rebalance_portfolio_async(self, process: BusinessProcess):
        """异步执行组合再平衡"""
        try:
            process.status = BusinessProcessStatus.RUNNING
            process.started_at = datetime.now()

            target_allocation = process.parameters["target_allocation"]

            # 步骤1: 获取当前持仓
            process.progress = 0.2
            process.steps.append({"step": "获取持仓", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            if self.db_service:
                current_positions = await self.db_service.get_user_positions(process.user_id)
            else:
                current_positions = []

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["current_positions"] = current_positions

            # 步骤2: 计算目标持仓
            process.progress = 0.5
            process.steps.append({"step": "计算目标", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            # 获取用户总资产
            if self.db_service:
                user = await self.db_service.get_user(process.user_id)
                total_assets = user["balance"] + sum(p["quantity"] * p.get(
                    "current_price", p["avg_price"]) for p in current_positions)
            else:
                total_assets = 10000.0  # 模拟数据

            target_positions = {}
            for symbol, weight in target_allocation.items():
                target_value = total_assets * weight
                # 模拟价格
                price = 50.0 + (hash(symbol) % 100)
                target_positions[symbol] = {
                    "quantity": int(target_value / price),
                    "price": price,
                    "target_value": target_value
                }

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["target_positions"] = target_positions

            # 步骤3: 生成调仓订单
            process.progress = 0.8
            process.steps.append({"step": "生成订单", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            rebalance_orders = []

            # 计算需要买入和卖出的股票
            current_allocation = {}
            for position in current_positions:
                symbol = position["symbol"]
                value = position["quantity"] * position.get("current_price", position["avg_price"])
                current_allocation[symbol] = value / total_assets if total_assets > 0 else 0

            for symbol, target_info in target_positions.items():
                current_weight = current_allocation.get(symbol, 0)
                target_weight = target_allocation[symbol]

                if target_weight > current_weight:
                    # 需要买入
                    buy_quantity = target_info["quantity"] - \
                        sum(p["quantity"] for p in current_positions if p["symbol"] == symbol)
                    if buy_quantity > 0:
                        rebalance_orders.append({
                            "symbol": symbol,
                            "quantity": buy_quantity,
                            "price": target_info["price"],
                            "order_type": "market",
                            "side": "buy",
                            "reason": "rebalance"
                        })
                elif target_weight < current_weight:
                    # 需要卖出
                    current_quantity = sum(p["quantity"]
                                           for p in current_positions if p["symbol"] == symbol)
                    sell_quantity = current_quantity - target_info["quantity"]
                    if sell_quantity > 0:
                        rebalance_orders.append({
                            "symbol": symbol,
                            "quantity": sell_quantity,
                            "price": target_info["price"],
                            "order_type": "market",
                            "side": "sell",
                            "reason": "rebalance"
                        })

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["rebalance_orders"] = rebalance_orders

            # 步骤4: 执行调仓订单
            process.progress = 1.0
            process.steps.append({"step": "执行订单", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            executed_orders = []
            for order in rebalance_orders:
                try:
                    if self.execution_engine:
                        result = await self.execution_engine.execute_order(order)
                        executed_orders.append(result)
                    else:
                        # 模拟执行
                        executed_orders.append({"order": order, "status": "simulated_success"})
                except Exception as e:
                    logger.warning(f"调仓订单执行失败: {e}")
                    executed_orders.append({"order": order, "error": str(e)})

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["executed_orders"] = executed_orders

            # 完成流程
            process.status = BusinessProcessStatus.COMPLETED
            process.completed_at = datetime.now()
            process.results = {
                "current_allocation": current_allocation,
                "target_allocation": target_allocation,
                "rebalance_orders": rebalance_orders,
                "executed_orders": executed_orders,
                "total_assets": total_assets
            }

        except Exception as e:
            logger.error(f"组合再平衡异步执行失败: {e}")
            process.status = BusinessProcessStatus.FAILED
            process.error_message = str(e)
            process.completed_at = datetime.now()

    # ==================== 流程管理 ====================

    async def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """获取流程状态"""
        try:
            if process_id not in self.active_processes:
                return None

            process = self.active_processes[process_id]

            return {
                "process_id": process.process_id,
                "process_type": process.process_type.value,
                "status": process.status.value,
                "progress": process.progress,
                "created_at": process.created_at.isoformat(),
                "started_at": process.started_at.isoformat() if process.started_at else None,
                "completed_at": process.completed_at.isoformat() if process.completed_at else None,
                "error_message": process.error_message,
                "steps": process.steps,
                "results": process.results
            }

        except Exception as e:
            logger.error(f"获取流程状态失败: {e}")
            return None

    async def cancel_process(self, process_id: str) -> bool:
        """取消流程"""
        try:
            if process_id in self.active_processes:
                process = self.active_processes[process_id]
                if process.status == BusinessProcessStatus.RUNNING:
                    process.status = BusinessProcessStatus.CANCELLED
                    process.completed_at = datetime.now()
                    return True
            return False

        except Exception as e:
            logger.error(f"取消流程失败: {e}")
            return False

    async def get_user_processes(self, user_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取用户流程"""
        try:
            processes = []
            for process in self.active_processes.values():
                if process.user_id == user_id:
                    if status is None or process.status.value == status:
                        processes.append(await self.get_process_status(process.process_id))

            # 按创建时间倒序
            processes.sort(key=lambda x: x["created_at"], reverse=True)
            return processes

        except Exception as e:
            logger.error(f"获取用户流程失败: {e}")
            return []

    # ==================== 数据分析 ====================

    async def analyze_market_data(self, symbols: List[str], analysis_type: str = "technical") -> Dict[str, Any]:
        """市场数据分析"""
        try:
            # 创建业务流程
            process = BusinessProcess(
                process_id=str(uuid.uuid4()),
                process_type=BusinessProcessType.MARKET_ANALYSIS,
                user_id=0,  # 系统级流程
                parameters={"symbols": symbols, "analysis_type": analysis_type}
            )

            self.active_processes[process.process_id] = process

            # 异步执行分析
            asyncio.create_task(self._analyze_market_data_async(process))

            return {
                "success": True,
                "process_id": process.process_id,
                "message": "市场分析已启动"
            }

        except Exception as e:
            logger.error(f"市场数据分析失败: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_market_data_async(self, process: BusinessProcess):
        """异步执行市场数据分析"""
        try:
            process.status = BusinessProcessStatus.RUNNING
            process.started_at = datetime.now()

            symbols = process.parameters["symbols"]
            analysis_type = process.parameters["analysis_type"]

            # 步骤1: 数据获取
            process.progress = 0.3
            process.steps.append({"step": "数据获取", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            market_data = {}
            if self.data_manager:
                try:
                    for symbol in symbols:
                        data = await self.data_manager.get_market_data(symbol, days=30)
                        market_data[symbol] = data
                except Exception as e:
                    logger.warning(f"数据获取失败: {e}")

            # 模拟数据
            if not market_data:
                for symbol in symbols:
                    market_data[symbol] = {
                        "prices": [50.0 + i * 0.1 + (time.time() % 10 - 5) for i in range(30)],
                        "volumes": [100000 + i * 1000 for i in range(30)],
                        "dates": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(30)]
                    }

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["data_points"] = sum(len(d.get("prices", []))
                                                   for d in market_data.values())

            # 步骤2: 数据分析
            process.progress = 0.7
            process.steps.append({"step": "数据分析", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            analysis_results = {}

            if analysis_type == "technical":
                for symbol, data in market_data.items():
                    prices = data.get("prices", [])
                    if len(prices) >= 20:
                        # 简单的技术指标计算
                        sma_20 = sum(prices[-20:]) / 20
                        current_price = prices[-1]
                        prev_price = prices[-2] if len(prices) > 1 else prices[0]

                        analysis_results[symbol] = {
                            "current_price": current_price,
                            "price_change": current_price - prev_price,
                            "price_change_pct": (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0,
                            "sma_20": sma_20,
                            "signal": "buy" if current_price > sma_20 else "sell",
                            "confidence": 0.7
                        }
            else:
                # 基本面分析（模拟）
                for symbol in symbols:
                    analysis_results[symbol] = {
                        "pe_ratio": 15.0 + (hash(symbol) % 20),
                        "pb_ratio": 1.5 + (hash(symbol) % 5) / 10,
                        "roe": 0.12 + (hash(symbol) % 20) / 100,
                        "recommendation": "hold",
                        "confidence": 0.6
                    }

            process.steps[-1]["status"] = "completed"
            process.steps[-1]["analysis_results"] = analysis_results

            # 步骤3: 生成报告
            process.progress = 1.0
            process.steps.append({"step": "生成报告", "status": "running",
                                 "timestamp": datetime.now().isoformat()})

            report = {
                "analysis_type": analysis_type,
                "symbols_analyzed": len(symbols),
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_symbols": len(symbols),
                    "analyzed_symbols": len(analysis_results),
                    "buy_signals": sum(1 for r in analysis_results.values() if r.get("signal") == "buy"),
                    "sell_signals": sum(1 for r in analysis_results.values() if r.get("signal") == "sell")
                },
                "results": analysis_results
            }

            process.steps[-1]["status"] = "completed"

            # 完成流程
            process.status = BusinessProcessStatus.COMPLETED
            process.completed_at = datetime.now()
            process.results = report

        except Exception as e:
            logger.error(f"市场数据分析异步执行失败: {e}")
            process.status = BusinessProcessStatus.FAILED
            process.error_message = str(e)
            process.completed_at = datetime.now()

    async def health_check(self) -> Dict[str, Any]:
        """业务服务健康检查"""
        try:
            components_status = {
                "database_service": self.db_service is not None,
                "strategy_engine": self.strategy_engine is not None,
                "order_manager": self.order_manager is not None,
                "execution_engine": self.execution_engine is not None,
                "risk_manager": self.risk_manager is not None,
                "monitoring_system": self.monitoring_system is not None,
                "data_manager": self.data_manager is not None
            }

            active_processes = len(self.active_processes)
            active_strategies = len(self.active_strategies)

            return {
                "status": "healthy" if self.db_service else "degraded",
                "timestamp": datetime.now().isoformat(),
                "components": components_status,
                "active_processes": active_processes,
                "active_strategies": active_strategies,
                "service": "BusinessService"
            }

        except Exception as e:
            logger.error(f"业务服务健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "service": "BusinessService"
            }


# 全局服务实例
_business_service = None


async def get_business_service() -> BusinessService:
    """获取业务服务实例"""
    global _business_service

    if _business_service is None:
        _business_service = BusinessService()
        await _business_service.initialize()

    return _business_service
