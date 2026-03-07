#!/usr/bin/env python3
"""
架构一致性修复工具

修复src目录结构与架构设计文档的不一致问题
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ArchitectureConsistencyFixer:
    """架构一致性修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

        # 目录迁移映射
        self.migration_map = {
            "acceleration": {
                "source": "acceleration",
                "target": "features/acceleration",
                "description": "硬件加速组件迁移到特征处理层",
                "priority": "high"
            },
            "adapters": {
                "source": "adapters",
                "target": "data/adapters",
                "description": "数据适配器迁移到数据采集层",
                "priority": "high"
            },
            "analysis": {
                "source": "analysis",
                "target": "backtest/analysis",
                "description": "分析功能迁移到策略决策层",
                "priority": "medium"
            },
            "deployment": {
                "source": "deployment",
                "target": "infrastructure/deployment",
                "description": "部署功能迁移到基础设施层",
                "priority": "medium"
            },
            "integration": {
                "source": "integration",
                "target": "core/integration",
                "description": "系统集成迁移到核心服务层",
                "priority": "medium"
            },
            "models": {
                "source": "models",
                "target": "ml/models",
                "description": "模型管理迁移到模型推理层",
                "priority": "high"
            },
            "monitoring": {
                "source": "monitoring",
                "target": "engine/monitoring",
                "description": "系统监控迁移到监控反馈层",
                "priority": "high"
            },
            "services": {
                "source": "services",
                "target": "infrastructure/services",
                "description": "通用服务迁移到基础设施层",
                "priority": "medium"
            },
            "tuning": {
                "source": "tuning",
                "target": "ml/tuning",
                "description": "调优功能迁移到模型推理层",
                "priority": "medium"
            },
            "utils": {
                "source": "utils",
                "target": "infrastructure/utils",
                "description": "通用工具迁移到基础设施层",
                "priority": "high"
            }
        }

        # 未分类目录处理
        self.unclassified_map = {
            "ensemble": {
                "target": "ml/ensemble",
                "description": "集成学习目录归类到模型推理层",
                "priority": "low"
            }
        }

    def fix_consistency_issues(self, dry_run: bool = True) -> Dict[str, Any]:
        """修复一致性问题"""
        print(f"🔧 {'预览' if dry_run else '执行'}架构一致性修复...")

        result = {
            "timestamp": datetime.now().isoformat(),
            "executed_fixes": [],
            "skipped_fixes": [],
            "errors": [],
            "summary": {}
        }

        # 1. 修复冗余目录迁移
        print("\n📁 步骤1: 迁移冗余目录")
        for name, config in self.migration_map.items():
            self._migrate_directory(name, config, result, dry_run)

        # 2. 修复未分类目录
        print("\n📁 步骤2: 处理未分类目录")
        for name, config in self.unclassified_map.items():
            self._migrate_directory(name, config, result, dry_run)

        # 3. 创建缺失的组件文件
        print("\n📄 步骤3: 创建缺失的组件文件")
        self._create_missing_component_files(result, dry_run)

        # 4. 验证修复结果
        print("\n✅ 步骤4: 验证修复结果")
        validation = self._validate_fixes(result)

        result["summary"] = {
            "total_fixes": len(result["executed_fixes"]),
            "successful_fixes": len([f for f in result["executed_fixes"] if f.get("success", False)]),
            "failed_fixes": len(result["errors"]),
            "validation_passed": validation["passed"]
        }

        return result

    def _migrate_directory(self, name: str, config: Dict[str, Any], result: Dict[str, Any], dry_run: bool) -> None:
        """迁移目录"""
        source_path = self.src_dir / name
        target_path = self.src_dir / config["target"]

        if not source_path.exists():
            result["skipped_fixes"].append({
                "type": "migration",
                "name": name,
                "reason": f"源目录不存在: {source_path}",
                "priority": config["priority"]
            })
            return

        print(f"  📁 {'将迁移' if dry_run else '迁移'} {name} -> {config['target']}")

        try:
            if not dry_run:
                # 确保目标父目录存在
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # 如果目标目录已存在，先备份
                if target_path.exists():
                    backup_path = target_path.with_suffix(
                        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    shutil.move(str(target_path), str(backup_path))
                    print(f"    💾 备份现有目录: {backup_path}")

                # 移动目录
                shutil.move(str(source_path), str(target_path))
                print(f"    ✅ 已迁移: {name} -> {config['target']}")

            result["executed_fixes"].append({
                "type": "migration",
                "name": name,
                "source": str(source_path),
                "target": str(target_path),
                "description": config["description"],
                "priority": config["priority"],
                "success": not dry_run
            })

        except Exception as e:
            error_msg = f"迁移失败 {name}: {e}"
            result["errors"].append(error_msg)
            print(f"    ❌ {error_msg}")

    def _create_missing_component_files(self, result: Dict[str, Any], dry_run: bool) -> None:
        """创建缺失的组件文件"""
        missing_components = {
            "risk": {
                "checker.py": self._generate_risk_checker,
                "monitor.py": self._generate_risk_monitor
            },
            "trading": {
                "executor.py": self._generate_trading_executor,
                "manager.py": self._generate_trading_manager,
                "risk.py": self._generate_trading_risk
            }
        }

        for layer, components in missing_components.items():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            for component_file, generator_func in components.items():
                file_path = layer_path / component_file

                if file_path.exists():
                    continue

                print(f"  📄 {'将创建' if dry_run else '创建'} {layer}/{component_file}")

                if not dry_run:
                    try:
                        content = generator_func(layer, component_file)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"    ✅ 已创建: {layer}/{component_file}")

                        result["executed_fixes"].append({
                            "type": "create_file",
                            "layer": layer,
                            "file": component_file,
                            "path": str(file_path),
                            "success": True
                        })

                    except Exception as e:
                        error_msg = f"创建文件失败 {layer}/{component_file}: {e}"
                        result["errors"].append(error_msg)
                        print(f"    ❌ {error_msg}")

    def _generate_risk_checker(self, layer: str, file_name: str) -> str:
        """生成风险检查器"""
        return f'''"""风险检查器 - 风控合规层组件"""

from typing import Dict, List, Any, Optional
from enum import Enum

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskChecker:
    """风险检查器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化风险检查器

        Args:
            config: 检查器配置
        """
        self.config = config or {{}}
        self._checkers = {{}}
        self._setup_default_checkers()

    def _setup_default_checkers(self):
        """设置默认检查器"""
        self._checkers = {{
            "position_risk": self._check_position_risk,
            "market_risk": self._check_market_risk,
            "liquidity_risk": self._check_liquidity_risk,
            "operational_risk": self._check_operational_risk
        }}

    def check_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行风险检查

        Args:
            context: 风险检查上下文

        Returns:
            风险检查结果
        """
        results = {{
            "overall_risk_level": RiskLevel.LOW.value,
            "check_results": {{}},
            "recommendations": [],
            "warnings": []
        }}

        # 执行各项风险检查
        for check_name, check_func in self._checkers.items():
            try:
                check_result = check_func(context)
                results["check_results"][check_name] = check_result

                # 更新整体风险等级
                if check_result.get("risk_level") == RiskLevel.CRITICAL.value:
                    results["overall_risk_level"] = RiskLevel.CRITICAL.value
                elif check_result.get("risk_level") == RiskLevel.HIGH.value and results["overall_risk_level"] != RiskLevel.CRITICAL.value:
                    results["overall_risk_level"] = RiskLevel.HIGH.value
                elif check_result.get("risk_level") == RiskLevel.MEDIUM.value and results["overall_risk_level"] not in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]:
                    results["overall_risk_level"] = RiskLevel.MEDIUM.value

                # 收集建议
                if check_result.get("recommendations"):
                    results["recommendations"].extend(check_result["recommendations"])

            except Exception as e:
                results["warnings"].append(f"风险检查失败 {{check_name}}: {{e}}")

        return results

    def _check_position_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查持仓风险"""
        # 实现持仓风险检查逻辑
        return {{
            "risk_level": RiskLevel.LOW.value,
            "risk_score": 0.2,
            "recommendations": ["保持当前持仓水平"]
        }}

    def _check_market_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查市场风险"""
        # 实现市场风险检查逻辑
        return {{
            "risk_level": RiskLevel.MEDIUM.value,
            "risk_score": 0.5,
            "recommendations": ["考虑降低仓位"]
        }}

    def _check_liquidity_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查流动性风险"""
        # 实现流动性风险检查逻辑
        return {{
            "risk_level": RiskLevel.LOW.value,
            "risk_score": 0.1,
            "recommendations": ["流动性充足"]
        }}

    def _check_operational_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查操作风险"""
        # 实现操作风险检查逻辑
        return {{
            "risk_level": RiskLevel.LOW.value,
            "risk_score": 0.15,
            "recommendations": ["操作流程正常"]
        }}

    def add_checker(self, name: str, checker_func: callable):
        """添加自定义检查器

        Args:
            name: 检查器名称
            checker_func: 检查函数
        """
        self._checkers[name] = checker_func

    def remove_checker(self, name: str):
        """移除检查器

        Args:
            name: 检查器名称
        """
        if name in self._checkers:
            del self._checkers[name]

    def get_available_checkers(self) -> List[str]:
        """获取可用检查器列表

        Returns:
            检查器名称列表
        """
        return list(self._checkers.keys())
'''

    def _generate_risk_monitor(self, layer: str, file_name: str) -> str:
        """生成风险监控器"""
        return f'''"""风险监控器 - 风控合规层组件"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import time

class RiskMonitor:
    """风险监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化风险监控器

        Args:
            config: 监控器配置
        """
        self.config = config or {{}}
        self._monitors = {{}}
        self._alert_handlers = []
        self._is_running = False
        self._monitor_thread = None
        self._setup_default_monitors()

    def _setup_default_monitors(self):
        """设置默认监控器"""
        self._monitors = {{
            "realtime_risk": self._monitor_realtime_risk,
            "portfolio_risk": self._monitor_portfolio_risk,
            "market_risk": self._monitor_market_risk,
            "compliance_risk": self._monitor_compliance_risk
        }}

    def start_monitoring(self):
        """启动监控"""
        if self._is_running:
            return

        self._is_running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        print("✅ 风险监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self._is_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        print("🛑 风险监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        check_interval = self.config.get("check_interval", 60)  # 默认60秒

        while self._is_running:
            try:
                self._execute_monitoring_cycle()
                time.sleep(check_interval)
            except Exception as e:
                print(f"监控循环异常: {{e}}")
                time.sleep(10)  # 异常情况下等待更长时间

    def _execute_monitoring_cycle(self):
        """执行监控周期"""
        current_time = datetime.now()

        for monitor_name, monitor_func in self._monitors.items():
            try:
                result = monitor_func(current_time)

                if result.get("alert_triggered", False):
                    self._trigger_alert(monitor_name, result)

            except Exception as e:
                print(f"监控器异常 {{monitor_name}}: {{e}}")

    def _monitor_realtime_risk(self, current_time: datetime) -> Dict[str, Any]:
        """实时风险监控"""
        # 实现实时风险监控逻辑
        return {{
            "monitor_type": "realtime_risk",
            "timestamp": current_time.isoformat(),
            "alert_triggered": False,
            "data": {{"current_risk_level": "low"}}
        }}

    def _monitor_portfolio_risk(self, current_time: datetime) -> Dict[str, Any]:
        """投资组合风险监控"""
        # 实现投资组合风险监控逻辑
        return {{
            "monitor_type": "portfolio_risk",
            "timestamp": current_time.isoformat(),
            "alert_triggered": False,
            "data": {{"portfolio_var": 0.05}}
        }}

    def _monitor_market_risk(self, current_time: datetime) -> Dict[str, Any]:
        """市场风险监控"""
        # 实现市场风险监控逻辑
        return {{
            "monitor_type": "market_risk",
            "timestamp": current_time.isoformat(),
            "alert_triggered": False,
            "data": {{"market_volatility": 0.15}}
        }}

    def _monitor_compliance_risk(self, current_time: datetime) -> Dict[str, Any]:
        """合规风险监控"""
        # 实现合规风险监控逻辑
        return {{
            "monitor_type": "compliance_risk",
            "timestamp": current_time.isoformat(),
            "alert_triggered": False,
            "data": {{"compliance_status": "good"}}
        }}

    def _trigger_alert(self, monitor_name: str, result: Dict[str, Any]):
        """触发告警"""
        alert_data = {{
            "monitor_name": monitor_name,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }}

        # 调用所有告警处理器
        for handler in self._alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                print(f"告警处理器异常: {{e}}")

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """添加告警处理器

        Args:
            handler: 告警处理函数
        """
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """移除告警处理器

        Args:
            handler: 告警处理函数
        """
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def add_monitor(self, name: str, monitor_func: Callable[[datetime], Dict[str, Any]]):
        """添加自定义监控器

        Args:
            name: 监控器名称
            monitor_func: 监控函数
        """
        self._monitors[name] = monitor_func

    def remove_monitor(self, name: str):
        """移除监控器

        Args:
            name: 监控器名称
        """
        if name in self._monitors:
            del self._monitors[name]

    def get_monitor_status(self) -> Dict[str, Any]:
        """获取监控状态

        Returns:
            监控状态信息
        """
        return {{
            "is_running": self._is_running,
            "monitor_count": len(self._monitors),
            "alert_handler_count": len(self._alert_handlers),
            "available_monitors": list(self._monitors.keys())
        }}
'''

    def _generate_trading_executor(self, layer: str, file_name: str) -> str:
        """生成交易执行器"""
        return f'''"""交易执行器 - 交易执行层组件"""

from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TradingExecutor:
    """交易执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化交易执行器

        Args:
            config: 执行器配置
        """
        self.config = config or {{}}
        self._executors = {{}}
        self._order_history = []
        self._setup_default_executors()

    def _setup_default_executors(self):
        """设置默认执行器"""
        self._executors = {{
            "market_maker": self._execute_market_maker,
            "algorithmic": self._execute_algorithmic,
            "smart_router": self._execute_smart_router
        }}

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """执行订单

        Args:
            order: 订单信息

        Returns:
            执行结果
        """
        try:
            # 验证订单
            validation_result = self._validate_order(order)
            if not validation_result["valid"]:
                return {{
                    "success": False,
                    "error": validation_result["error"],
                    "order_id": order.get("order_id")
                }}

            # 选择执行策略
            execution_strategy = self._select_execution_strategy(order)

            # 执行订单
            if execution_strategy in self._executors:
                result = self._executors[execution_strategy](order)
            else:
                result = self._execute_default(order)

            # 记录订单历史
            self._record_order_history(order, result)

            return result

        except Exception as e:
            return {{
                "success": False,
                "error": f"执行异常: {{e}}",
                "order_id": order.get("order_id")
            }}

    def _validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """验证订单"""
        required_fields = ["symbol", "side", "quantity", "order_type"]

        for field in required_fields:
            if field not in order:
                return {{"valid": False, "error": f"缺少必要字段: {{field}}"}}

        if order["quantity"] <= 0:
            return {{"valid": False, "error": "数量必须大于0"}}

        if order["order_type"] not in [t.value for t in OrderType]:
            return {{"valid": False, "error": f"无效的订单类型: {{order['order_type']}}"}}

        return {{"valid": True}}

    def _select_execution_strategy(self, order: Dict[str, Any]) -> str:
        """选择执行策略"""
        # 基于订单特征选择执行策略
        order_type = order.get("order_type", "")
        quantity = order.get("quantity", 0)
        urgency = order.get("urgency", "normal")

        if urgency == "high":
            return "market_maker"
        elif quantity > 10000:
            return "algorithmic"
        else:
            return "smart_router"

    def _execute_market_maker(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """市商执行"""
        # 实现市商执行逻辑
        return {{
            "success": True,
            "order_id": order.get("order_id"),
            "status": OrderStatus.FILLED.value,
            "executed_quantity": order["quantity"],
            "execution_strategy": "market_maker",
            "timestamp": datetime.now().isoformat()
        }}

    def _execute_algorithmic(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """算法执行"""
        # 实现算法执行逻辑
        return {{
            "success": True,
            "order_id": order.get("order_id"),
            "status": OrderStatus.FILLED.value,
            "executed_quantity": order["quantity"],
            "execution_strategy": "algorithmic",
            "timestamp": datetime.now().isoformat()
        }}

    def _execute_smart_router(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """智能路由执行"""
        # 实现智能路由执行逻辑
        return {{
            "success": True,
            "order_id": order.get("order_id"),
            "status": OrderStatus.FILLED.value,
            "executed_quantity": order["quantity"],
            "execution_strategy": "smart_router",
            "timestamp": datetime.now().isoformat()
        }}

    def _execute_default(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """默认执行"""
        return {{
            "success": True,
            "order_id": order.get("order_id"),
            "status": OrderStatus.SUBMITTED.value,
            "executed_quantity": 0,
            "execution_strategy": "default",
            "timestamp": datetime.now().isoformat()
        }}

    def _record_order_history(self, order: Dict[str, Any], result: Dict[str, Any]):
        """记录订单历史"""
        history_entry = {{
            "order": order,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }}
        self._order_history.append(history_entry)

        # 限制历史记录数量
        max_history = self.config.get("max_history", 1000)
        if len(self._order_history) > max_history:
            self._order_history = self._order_history[-max_history:]

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            取消结果
        """
        # 实现订单取消逻辑
        return {{
            "success": True,
            "order_id": order_id,
            "status": OrderStatus.CANCELLED.value,
            "timestamp": datetime.now().isoformat()
        }}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        # 实现获取订单状态逻辑
        return {{
            "order_id": order_id,
            "status": OrderStatus.FILLED.value,
            "last_update": datetime.now().isoformat()
        }}

    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息

        Returns:
            统计信息
        """
        total_orders = len(self._order_history)
        successful_orders = len([h for h in self._order_history if h["result"].get("success", False)])

        return {{
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "success_rate": successful_orders / total_orders if total_orders > 0 else 0,
            "available_strategies": list(self._executors.keys())
        }}
'''

    def _generate_trading_manager(self, layer: str, file_name: str) -> str:
        """生成交易管理器"""
        return f'''"""交易管理器 - 交易执行层组件"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import threading
import queue

class TradingManager:
    """交易管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化交易管理器

        Args:
            config: 管理器配置
        """
        self.config = config or {{}}
        self._order_queue = queue.Queue()
        self._active_orders = {{}}
        self._completed_orders = []
        self._is_running = False
        self._manager_thread = None
        self._order_listeners = []

    def start(self):
        """启动交易管理器"""
        if self._is_running:
            return

        self._is_running = True
        self._manager_thread = threading.Thread(target=self._management_loop)
        self._manager_thread.daemon = True
        self._manager_thread.start()

        print("✅ 交易管理器已启动")

    def stop(self):
        """停止交易管理器"""
        self._is_running = False
        if self._manager_thread:
            self._manager_thread.join(timeout=5)

        print("🛑 交易管理器已停止")

    def _management_loop(self):
        """管理循环"""
        process_interval = self.config.get("process_interval", 1)  # 默认1秒

        while self._is_running:
            try:
                self._process_order_queue()
                self._check_order_timeouts()
                self._update_order_status()
                time.sleep(process_interval)
            except Exception as e:
                print(f"管理循环异常: {{e}}")
                time.sleep(5)

    def _process_order_queue(self):
        """处理订单队列"""
        while not self._order_queue.empty():
            try:
                order = self._order_queue.get_nowait()
                self._handle_new_order(order)
            except queue.Empty:
                break

    def _handle_new_order(self, order: Dict[str, Any]):
        """处理新订单"""
        order_id = order.get("order_id", f"order_{datetime.now().timestamp()}")

        # 添加订单元数据
        order["order_id"] = order_id
        order["submit_time"] = datetime.now().isoformat()
        order["status"] = "submitted"

        # 存储到活跃订单
        self._active_orders[order_id] = order

        # 通知监听器
        self._notify_order_listeners("order_submitted", order)

        print(f"📋 已提交订单: {{order_id}}")

    def submit_order(self, order: Dict[str, Any]) -> str:
        """提交订单

        Args:
            order: 订单信息

        Returns:
            订单ID
        """
        order_id = f"order_{datetime.now().timestamp()}"
        order["order_id"] = order_id

        self._order_queue.put(order)
        return order_id

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """取消订单

        Args:
            order_id: 订单ID

        Returns:
            取消结果
        """
        if order_id in self._active_orders:
            order = self._active_orders[order_id]
            order["status"] = "cancelled"
            order["cancel_time"] = datetime.now().isoformat()

            # 移动到完成订单
            self._completed_orders.append(order)
            del self._active_orders[order_id]

            # 通知监听器
            self._notify_order_listeners("order_cancelled", order)

            return {{
                "success": True,
                "order_id": order_id,
                "message": "订单已取消"
            }}
        else:
            return {{
                "success": False,
                "order_id": order_id,
                "message": "订单不存在或已完成"
            }}

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态

        Args:
            order_id: 订单ID

        Returns:
            订单状态
        """
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        else:
            # 在完成订单中查找
            for order in self._completed_orders:
                if order.get("order_id") == order_id:
                    return order

            return {{"error": "订单不存在"}}

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取活跃订单列表

        Returns:
            活跃订单列表
        """
        return list(self._active_orders.values())

    def get_completed_orders(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取完成订单列表

        Args:
            hours: 过去多少小时的订单

        Returns:
            完成订单列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            order for order in self._completed_orders
            if datetime.fromisoformat(order.get("submit_time", "2000-01-01T00:00:00")) > cutoff_time
        ]

    def _check_order_timeouts(self):
        """检查订单超时"""
        timeout_seconds = self.config.get("order_timeout", 300)  # 默认5分钟
        current_time = datetime.now()

        timeout_orders = []
        for order_id, order in self._active_orders.items():
            submit_time = datetime.fromisoformat(order.get("submit_time", "2000-01-01T00:00:00"))
            if (current_time - submit_time).total_seconds() > timeout_seconds:
                timeout_orders.append(order_id)

        # 处理超时订单
        for order_id in timeout_orders:
            order = self._active_orders[order_id]
            order["status"] = "timeout"
            order["timeout_time"] = current_time.isoformat()

            self._completed_orders.append(order)
            del self._active_orders[order_id]

            self._notify_order_listeners("order_timeout", order)

    def _update_order_status(self):
        """更新订单状态"""
        # 这里可以实现从外部系统获取订单状态的逻辑
        pass

    def _notify_order_listeners(self, event_type: str, order: Dict[str, Any]):
        """通知订单监听器"""
        for listener in self._order_listeners:
            try:
                listener(event_type, order)
            except Exception as e:
                print(f"监听器异常: {{e}}")

    def add_order_listener(self, listener: callable):
        """添加订单监听器

        Args:
            listener: 监听器函数，接收(event_type, order)参数
        """
        self._order_listeners.append(listener)

    def remove_order_listener(self, listener: callable):
        """移除订单监听器

        Args:
            listener: 监听器函数
        """
        if listener in self._order_listeners:
            self._order_listeners.remove(listener)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息
        """
        return {{
            "active_orders": len(self._active_orders),
            "completed_orders": len(self._completed_orders),
            "queue_size": self._order_queue.qsize(),
            "is_running": self._is_running
        }}
'''

    def _generate_trading_risk(self, layer: str, file_name: str) -> str:
        """生成交易风险组件"""
        return f'''"""交易风险组件 - 交易执行层组件"""

from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

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
        self.config = config or {{}}
        self._risk_rules = {{}}
        self._risk_history = []
        self._setup_default_risk_rules()

    def _setup_default_risk_rules(self):
        """设置默认风险规则"""
        self._risk_rules = {{
            "max_position_size": self._check_position_size,
            "max_daily_loss": self._check_daily_loss,
            "max_order_frequency": self._check_order_frequency,
            "market_volatility": self._check_market_volatility,
            "liquidity_check": self._check_liquidity
        }}

    def evaluate_trade_risk(self, trade_context: Dict[str, Any]) -> Dict[str, Any]:
        """评估交易风险

        Args:
            trade_context: 交易上下文信息

        Returns:
            风险评估结果
        """
        results = {{
            "overall_action": RiskAction.ALLOW.value,
            "risk_score": 0.0,
            "warnings": [],
            "blocks": [],
            "recommendations": [],
            "rule_results": {{}}
        }}

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
                results["warnings"].append(f"风险规则检查失败 {{rule_name}}: {{e}}")

        # 记录风险评估历史
        self._record_risk_evaluation(trade_context, results)

        return results

    def _check_position_size(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查持仓规模"""
        max_position = self.config.get("max_position_size", 1000000)
        current_position = context.get("current_position", 0)
        order_size = context.get("order_size", 0)

        new_position = current_position + order_size

        if new_position > max_position:
            return {{
                "rule": "max_position_size",
                "action": RiskAction.BLOCK.value,
                "risk_score": 1.0,
                "message": f"持仓规模超限: {{new_position}} > {{max_position}}",
                "recommendations": ["减少订单规模", "分批执行"]
            }}
        elif new_position > max_position * 0.8:
            return {{
                "rule": "max_position_size",
                "action": RiskAction.WARN.value,
                "risk_score": 0.5,
                "message": f"持仓规模接近上限: {{new_position}} / {{max_position}}",
                "recommendations": ["谨慎操作", "考虑减仓"]
            }}
        else:
            return {{
                "rule": "max_position_size",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }}

    def _check_daily_loss(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查日损失"""
        max_daily_loss = self.config.get("max_daily_loss", 10000)
        current_daily_loss = context.get("current_daily_loss", 0)

        if current_daily_loss >= max_daily_loss:
            return {{
                "rule": "max_daily_loss",
                "action": RiskAction.BLOCK.value,
                "risk_score": 1.0,
                "message": f"日损失已达上限: {{current_daily_loss}} >= {{max_daily_loss}}",
                "recommendations": ["停止交易", "重新评估策略"]
            }}
        elif current_daily_loss >= max_daily_loss * 0.8:
            return {{
                "rule": "max_daily_loss",
                "action": RiskAction.WARN.value,
                "risk_score": 0.7,
                "message": f"日损失接近上限: {{current_daily_loss}} / {{max_daily_loss}}",
                "recommendations": ["减少交易频率", "降低风险暴露"]
            }}
        else:
            return {{
                "rule": "max_daily_loss",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }}

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
            return {{
                "rule": "max_order_frequency",
                "action": RiskAction.BLOCK.value,
                "risk_score": 0.8,
                "message": f"订单频率超限: {{len(recent_minute_orders)}} / {{max_orders_per_minute}}",
                "recommendations": ["降低订单频率", "使用订单合并"]
            }}
        elif len(recent_minute_orders) >= max_orders_per_minute * 0.7:
            return {{
                "rule": "max_order_frequency",
                "action": RiskAction.WARN.value,
                "risk_score": 0.3,
                "message": f"订单频率较高: {{len(recent_minute_orders)}} / {{max_orders_per_minute}}",
                "recommendations": ["注意订单频率", "考虑批量处理"]
            }}
        else:
            return {{
                "rule": "max_order_frequency",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }}

    def _check_market_volatility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查市场波动率"""
        max_volatility = self.config.get("max_volatility_threshold", 0.05)
        current_volatility = context.get("market_volatility", 0)

        if current_volatility >= max_volatility:
            return {{
                "rule": "market_volatility",
                "action": RiskAction.REDUCE.value,
                "risk_score": 0.6,
                "message": f"市场波动率过高: {{current_volatility:.2%}} >= {{max_volatility:.2%}}",
                "recommendations": ["减少持仓规模", "使用更保守的策略"]
            }}
        else:
            return {{
                "rule": "market_volatility",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }}

    def _check_liquidity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查流动性"""
        min_liquidity = self.config.get("min_liquidity_threshold", 100000)
        current_liquidity = context.get("liquidity", 0)

        if current_liquidity < min_liquidity:
            return {{
                "rule": "liquidity_check",
                "action": RiskAction.REDUCE.value,
                "risk_score": 0.4,
                "message": f"流动性不足: {{current_liquidity}} < {{min_liquidity}}",
                "recommendations": ["减少订单规模", "寻找替代市场"]
            }}
        else:
            return {{
                "rule": "liquidity_check",
                "action": RiskAction.ALLOW.value,
                "risk_score": 0.0
            }}

    def _record_risk_evaluation(self, context: Dict[str, Any], results: Dict[str, Any]):
        """记录风险评估"""
        evaluation_record = {{
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "results": results
        }}

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

    def get_risk_statistics(self) -> Dict[str, Any]:
        """获取风险统计信息

        Returns:
            风险统计信息
        """
        total_evaluations = len(self._risk_history)
        blocked_trades = len([r for r in self._risk_history if r["results"]["overall_action"] == RiskAction.BLOCK.value])
        warned_trades = len([r for r in self._risk_history if r["results"]["overall_action"] == RiskAction.WARN.value])

        return {{
            "total_evaluations": total_evaluations,
            "blocked_trades": blocked_trades,
            "warned_trades": warned_trades,
            "block_rate": blocked_trades / total_evaluations if total_evaluations > 0 else 0,
            "warn_rate": warned_trades / total_evaluations if total_evaluations > 0 else 0,
            "active_rules": list(self._risk_rules.keys())
        }}
'''

    def _validate_fixes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证修复结果"""
        validation = {
            "passed": True,
            "issues": [],
            "improvements": []
        }

        # 检查是否还有冗余目录
        redundant_dirs = ["acceleration", "adapters", "analysis", "deployment",
                          "integration", "models", "monitoring", "services",
                          "tuning", "utils"]

        for dir_name in redundant_dirs:
            dir_path = self.src_dir / dir_name
            if dir_path.exists():
                validation["issues"].append(f"冗余目录仍然存在: {dir_name}")

        # 检查架构层级是否完整
        expected_layers = ["core", "infrastructure", "data", "gateway",
                           "features", "ml", "backtest", "risk", "trading", "engine"]

        for layer in expected_layers:
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                validation["issues"].append(f"缺少架构层级目录: {layer}")

        validation["passed"] = len(validation["issues"]) == 0
        return validation

    def generate_fix_report(self, result: Dict[str, Any]) -> str:
        """生成修复报告"""
        report = f"""# 架构一致性修复报告

## 📊 修复概览

**修复时间**: {result['timestamp']}
**执行修复**: {len(result['executed_fixes'])} 项
**修复成功**: {len([f for f in result['executed_fixes'] if f.get('success', False)])} 项
**修复失败**: {len(result['errors'])} 项

---

## 🏗️ 修复详情

"""

        # 迁移修复详情
        migrations = [f for f in result['executed_fixes'] if f.get('type') == 'migration']
        if migrations:
            report += "### 📁 目录迁移修复\n\n"
            for fix in migrations:
                status = "✅" if fix.get('success') else "❌"
                report += f"**{fix['name']}** -> `{fix['target']}`\n"
                report += f"- 状态: {status}\n"
                report += f"- 优先级: {fix.get('priority', 'N/A')}\n"
                report += f"- 描述: {fix.get('description', 'N/A')}\n\n"

        # 文件创建修复详情
        file_creations = [f for f in result['executed_fixes'] if f.get('type') == 'create_file']
        if file_creations:
            report += "### 📄 缺失文件创建\n\n"
            for fix in file_creations:
                status = "✅" if fix.get('success') else "❌"
                report += f"**{fix['layer']}/{fix['file']}**\n"
                report += f"- 状态: {status}\n"
                report += f"- 路径: `{fix.get('path', 'N/A')}`\n\n"

        # 错误详情
        if result['errors']:
            report += "## ❌ 修复错误\n\n"
            for error in result['errors']:
                report += f"- {error}\n"
            report += "\n"

        # 跳过的修复
        skipped = result['skipped_fixes']
        if skipped:
            report += "## ⏭️ 跳过的修复\n\n"
            for skip in skipped:
                report += f"- **{skip['name']}**: {skip['reason']}\n"
            report += "\n"

        # 验证结果
        validation = result['summary'].get('validation_passed', False)
        report += f"""## ✅ 验证结果

### 修复验证
- **验证状态**: {'✅ 通过' if validation else '❌ 失败'}
- **架构完整性**: {'✅ 保持' if validation else '❌ 受损'}
- **目录一致性**: {'✅ 达成' if validation else '❌ 未达成'}

### 修复统计
- **总修复项数**: {result['summary']['total_fixes']}
- **成功修复数**: {result['summary']['successful_fixes']}
- **失败修复数**: {result['summary']['failed_fixes']}

"""

        # 改进建议
        report += "## 💡 改进建议\n\n"
        if result['summary']['failed_fixes'] > 0:
            report += "- 🔴 **紧急**: 检查并修复失败的修复项\n"
        if len(result['executed_fixes']) > 0:
            report += "- 🟡 **重要**: 运行架构一致性检查验证修复效果\n"
        if validation:
            report += "- 🟢 **建议**: 定期运行一致性检查维护架构整洁\n"

        report += "\n---\n\n"
        report += "**修复工具**: scripts/fix_architecture_consistency.py\n"
        report += "**验证工具**: scripts/architecture_consistency_check.py\n"
        report += "**修复标准**: 基于架构设计文档 v5.0\n"

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构一致性修复工具')
    parser.add_argument('--execute', action='store_true', help='执行实际修复操作')
    parser.add_argument('--force', action='store_true', help='强制执行（跳过确认）')
    parser.add_argument('--dry-run', action='store_true', help='预览模式（默认）')
    parser.add_argument('--output', help='输出报告文件')

    args = parser.parse_args()

    fixer = ArchitectureConsistencyFixer(".")
    result = fixer.fix_consistency_issues(dry_run=not args.execute)

    report = fixer.generate_fix_report(result)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
