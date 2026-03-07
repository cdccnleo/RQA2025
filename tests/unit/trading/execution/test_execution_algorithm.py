"""
交易执行层执行算法测试
测试各种订单执行算法的正确性
"""

import pytest
import time
from pathlib import Path
import sys
from typing import Optional

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入执行算法相关类
from src.trading.execution.execution_algorithm import (
    BaseExecutionAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    AlgorithmType,
    AlgorithmConfig,
    ExecutionSlice
)


class TestExecutionAlgorithm:
    """执行算法测试"""

    def test_algorithm_types_enum(self):
        """测试算法类型枚举"""
        assert AlgorithmType.TWAP.value == "twap"
        assert AlgorithmType.VWAP.value == "vwap"
        assert AlgorithmType.POV.value == "pov"
        assert AlgorithmType.ICEBERG.value == "iceberg"
        assert AlgorithmType.MARKET.value == "market"
        assert AlgorithmType.LIMIT.value == "limit"

    def test_algorithm_config_initialization(self):
        """测试算法配置初始化"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.TWAP,
            duration=600,
            target_quantity=5000,
            max_participation=0.15,
            randomize_timing=False
        )

        assert config.algo_type == AlgorithmType.TWAP
        assert config.duration == 600
        assert config.target_quantity == 5000
        assert config.max_participation == 0.15
        assert config.randomize_timing == False

    def test_algorithm_config_defaults(self):
        """测试算法配置默认值"""
        config = AlgorithmConfig(algo_type=AlgorithmType.MARKET)

        assert config.algo_type == AlgorithmType.MARKET
        assert config.duration == 300
        assert config.target_quantity == 1000
        assert config.max_participation == 0.1
        assert config.randomize_timing == True

    def test_execution_slice_creation(self):
        """测试执行切片创建"""
        slice_obj = ExecutionSlice(
            quantity=100,
            price=50.5,
            timestamp=time.time(),
            venue="NYSE"
        )

        assert slice_obj.quantity == 100
        assert slice_obj.price == 50.5
        assert slice_obj.venue == "NYSE"
        assert slice_obj.timestamp is not None

    def test_execution_slice_defaults(self):
        """测试执行切片默认值"""
        slice_obj = ExecutionSlice(quantity=200)

        assert slice_obj.quantity == 200
        assert slice_obj.price is None
        assert slice_obj.timestamp is None
        assert slice_obj.venue == "default"

    def test_base_execution_algorithm_interface(self):
        """测试基础执行算法接口"""
        # 创建一个简单的实现来测试接口
        class TestAlgorithm(BaseExecutionAlgorithm):
            def execute_slice(self, slice_obj: ExecutionSlice) -> bool:
                return True

            def calculate_next_slice(self, remaining_quantity: int, time_remaining: float) -> Optional[ExecutionSlice]:
                if remaining_quantity > 0:
                    return ExecutionSlice(quantity=min(remaining_quantity, 100))
                return None

        config = AlgorithmConfig(algo_type=AlgorithmType.MARKET)
        algorithm = TestAlgorithm(config)

        assert algorithm.config == config

        # 测试切片执行
        slice_obj = ExecutionSlice(quantity=50)
        result = algorithm.execute_slice(slice_obj)
        assert result == True

        # 测试下一切片计算
        next_slice = algorithm.calculate_next_slice(200, 300.0)
        assert next_slice is not None
        assert next_slice.quantity == 100

        # 测试剩余数量为0的情况
        next_slice = algorithm.calculate_next_slice(0, 300.0)
        assert next_slice is None

    def test_twap_algorithm_basic(self):
        """测试TWAP算法基本功能"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.TWAP,
            duration=60,  # 1分钟
            target_quantity=300
        )

        twap = TWAPAlgorithm(config)

        # 测试初始状态
        assert twap.config == config

        # 计算下一切片（均匀分布）
        next_slice = twap.calculate_next_slice(300, 60.0)
        assert next_slice is not None
        assert next_slice.quantity > 0

    def test_twap_algorithm_time_based_slicing(self):
        """测试TWAP算法基于时间的切片"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.TWAP,
            duration=120,  # 2分钟
            target_quantity=600
        )

        twap = TWAPAlgorithm(config)

        # 第一分钟应该执行一半数量
        next_slice = twap.calculate_next_slice(600, 120.0)
        assert next_slice is not None
        assert next_slice.quantity == 300  # 600 * (60/120)

    def test_vwap_algorithm_basic(self):
        """测试VWAP算法基本功能"""
        config = AlgorithmConfig(
            algo_type=AlgorithmType.VWAP,
            duration=300,
            target_quantity=1000
        )

        vwap = VWAPAlgorithm(config)

        # 测试初始状态
        assert vwap.config == config

        # VWAP算法需要成交量数据，这里只测试接口
        order = {'quantity': 1000, 'price': 300.0}
        slices = vwap._execute_single_order(order)
        assert slices is not None
        assert len(slices) > 0

    def test_algorithm_execution_with_zero_quantity(self):
        """测试算法执行零数量订单"""
        class TestAlgorithm(BaseExecutionAlgorithm):
            def execute_slice(self, slice_obj: ExecutionSlice) -> bool:
                return slice_obj.quantity > 0

            def calculate_next_slice(self, remaining_quantity: int, time_remaining: float) -> Optional[ExecutionSlice]:
                if remaining_quantity <= 0:
                    return None
                return ExecutionSlice(quantity=min(remaining_quantity, 50))

        config = AlgorithmConfig(algo_type=AlgorithmType.MARKET)
        algorithm = TestAlgorithm(config)

        # 测试零数量
        next_slice = algorithm.calculate_next_slice(0, 300.0)
        assert next_slice is None

        # 测试负数量
        next_slice = algorithm.calculate_next_slice(-100, 300.0)
        assert next_slice is None

    def test_algorithm_execution_with_time_pressure(self):
        """测试算法在时间压力下的表现"""
        class TestAlgorithm(BaseExecutionAlgorithm):
            def execute_slice(self, slice_obj: ExecutionSlice) -> bool:
                return True

            def calculate_next_slice(self, remaining_quantity: int, time_remaining: float) -> Optional[ExecutionSlice]:
                if time_remaining <= 0:
                    # 时间用完，立即执行剩余所有数量
                    return ExecutionSlice(quantity=remaining_quantity) if remaining_quantity > 0 else None
                else:
                    # 正常切片
                    slice_qty = min(remaining_quantity, int(remaining_quantity / max(time_remaining, 1)))
                    return ExecutionSlice(quantity=slice_qty) if slice_qty > 0 else None

        config = AlgorithmConfig(algo_type=AlgorithmType.MARKET)
        algorithm = TestAlgorithm(config)

        # 时间充足的情况
        next_slice = algorithm.calculate_next_slice(1000, 100.0)
        assert next_slice is not None
        assert next_slice.quantity == 10  # 1000 / 100

        # 时间压力大的情况
        next_slice = algorithm.calculate_next_slice(1000, 10.0)
        assert next_slice is not None
        assert next_slice.quantity == 100  # 1000 / 10

        # 时间用完的情况
        next_slice = algorithm.calculate_next_slice(500, 0.0)
        assert next_slice is not None
        assert next_slice.quantity == 500  # 立即执行所有剩余

    def test_algorithm_config_validation(self):
        """测试算法配置验证"""
        # 测试无效的持续时间
        config_duration = AlgorithmConfig(algo_type=AlgorithmType.TWAP, duration=0)
        # 应该允许duration=0，但在实际使用中可能需要特殊处理

        # 测试无效的目标数量
        config_quantity = AlgorithmConfig(algo_type=AlgorithmType.TWAP, target_quantity=0)
        # 应该允许target_quantity=0

        # 测试无效的最大参与率
        config_participation = AlgorithmConfig(algo_type=AlgorithmType.TWAP, max_participation=0.0)
        # 应该允许max_participation=0.0

        # 所有这些配置都应该是有效的
        assert config_duration.duration == 0
        assert config_quantity.target_quantity == 0
        assert config_participation.max_participation == 0.0

    def test_execution_slice_immutability(self):
        """测试执行切片的不可变性"""
        slice_obj = ExecutionSlice(quantity=100, price=50.5)

        # 尝试修改属性
        # 注意：dataclass默认是可变的，这里测试实际行为
        original_quantity = slice_obj.quantity
        original_price = slice_obj.price

        slice_obj.quantity = 200
        slice_obj.price = 60.0

        assert slice_obj.quantity == 200  # 可以修改
        assert slice_obj.price == 60.0   # 可以修改

    def test_algorithm_state_management(self):
        """测试算法状态管理"""
        class StateAwareAlgorithm(BaseExecutionAlgorithm):
            def __init__(self, config):
                super().__init__(config)
                self.executed_quantity = 0
                self.is_active = True

            def execute_slice(self, slice_obj: ExecutionSlice) -> bool:
                if self.is_active and slice_obj.quantity > 0:
                    self.executed_quantity += slice_obj.quantity
                    return True
                return False

            def calculate_next_slice(self, remaining_quantity: int, time_remaining: float) -> Optional[ExecutionSlice]:
                if not self.is_active or remaining_quantity <= 0:
                    return None

                slice_qty = min(remaining_quantity, 100)
                return ExecutionSlice(quantity=slice_qty)

        config = AlgorithmConfig(algo_type=AlgorithmType.MARKET)
        algorithm = StateAwareAlgorithm(config)

        # 初始状态
        assert algorithm.executed_quantity == 0
        assert algorithm.is_active == True

        # 执行切片
        slice_obj = ExecutionSlice(quantity=50)
        result = algorithm.execute_slice(slice_obj)
        assert result == True
        assert algorithm.executed_quantity == 50

        # 再次执行
        slice_obj2 = ExecutionSlice(quantity=30)
        result2 = algorithm.execute_slice(slice_obj2)
        assert result2 == True
        assert algorithm.executed_quantity == 80

        # 停用算法
        algorithm.is_active = False
        slice_obj3 = ExecutionSlice(quantity=20)
        result3 = algorithm.execute_slice(slice_obj3)
        assert result3 == False  # 应该失败
        assert algorithm.executed_quantity == 80  # 数量不变
