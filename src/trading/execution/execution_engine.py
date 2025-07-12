"""交易执行引擎主模块"""

from src.infrastructure.monitoring.performance_monitor import PerformanceMonitor
from src.infrastructure.config.config_manager import ConfigManager
from .order_router import OrderRouter
from .execution_algorithm import get_algorithm, AlgorithmType
from src.trading.risk import RiskController
import time
from enum import Enum

class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = 1
    PARTIAL = 2
    COMPLETED = 3
    REJECTED = 4
    CANCELLED = 5

class ExecutionEngine:
    """交易执行引擎主类"""

    def __init__(self, config=None, metrics=None):
        self.config = config or ConfigManager()
        self.metrics = metrics or MetricsCollector('execution_engine')
        self.router = OrderRouter(config, metrics)
        self.risk_controller = RiskController()

        # 执行算法缓存
        self.algorithms = {
            algo_type: get_algorithm(algo_type, config, metrics)
            for algo_type in AlgorithmType
        }

        # 执行状态跟踪
        self.execution_status = {}

    def execute_order(self, order):
        """
        执行订单
        Args:
            order: 订单对象，包含:
                - symbol: 股票代码
                - quantity: 数量
                - price: 价格(可选)
                - algo_type: 算法类型
                - algo_params: 算法参数
        Returns:
            dict: 执行结果
        """
        start_time = time.time()

        # 1. 风控检查
        risk_check = self.risk_controller.check_order(order)
        if not risk_check['allowed']:
            self._record_rejection(order, 'risk_rejected')
            return {
                'status': ExecutionStatus.REJECTED,
                'reason': risk_check['reason']
            }

        # 2. 获取执行算法
        algo_type = AlgorithmType[order.get('algo_type', 'TWAP')]
        algorithm = self.algorithms[algo_type]

        # 3. 执行订单
        execution_result = algorithm.execute(order)

        # 4. 处理执行结果
        final_status = self._process_execution_result(order, execution_result)

        # 记录执行指标
        self.metrics.record_execution_time(time.time() - start_time)
        self.metrics.record_order_status(final_status)

        return {
            'status': final_status,
            'execution_id': order.get('order_id'),
            'details': execution_result
        }

    def _process_execution_result(self, order, execution_result):
        """处理执行结果"""
        # 检查是否有拒绝的订单
        rejected = any(r['status'] == 'rejected' for r in execution_result)

        # 检查是否全部成交
        filled_qty = sum(
            r['quantity'] for r in execution_result
            if r['status'] == 'filled'
        )

        original_qty = order['quantity']

        if rejected and filled_qty == 0:
            return ExecutionStatus.REJECTED
        elif filled_qty == original_qty:
            return ExecutionStatus.COMPLETED
        elif filled_qty > 0:
            return ExecutionStatus.PARTIAL
        else:
            return ExecutionStatus.REJECTED

    def _record_rejection(self, order, reason):
        """记录订单拒绝"""
        self.execution_status[order.get('order_id')] = ExecutionStatus.REJECTED
        self.metrics.record_rejection(reason)

    def cancel_order(self, order_id):
        """取消订单"""
        if order_id in self.execution_status:
            self.execution_status[order_id] = ExecutionStatus.CANCELLED
            self.metrics.record_cancellation()
            return True
        return False

    def get_execution_status(self, order_id):
        """获取订单执行状态"""
        return self.execution_status.get(order_id, ExecutionStatus.PENDING)
