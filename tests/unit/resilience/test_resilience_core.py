"""
弹性层核心功能测试
测试弹性机制的接口和枚举定义
"""

import pytest
from pathlib import Path
import sys

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入弹性层相关类
from src.resilience.core.unified_resilience_interface import (
    ResilienceLevel,
    FailureType,
    CircuitBreakerState,
    DegradationStrategy,
    RecoveryAction
)


class TestResilienceCore:
    """弹性层核心测试"""

    def test_resilience_level_enum(self):
        """测试弹性级别枚举"""
        assert ResilienceLevel.NONE.value == "none"
        assert ResilienceLevel.BASIC.value == "basic"
        assert ResilienceLevel.ADVANCED.value == "advanced"
        assert ResilienceLevel.ENTERPRISE.value == "enterprise"

    def test_failure_type_enum(self):
        """测试故障类型枚举"""
        assert FailureType.NETWORK.value == "network"
        assert FailureType.DATABASE.value == "database"
        assert FailureType.SERVICE.value == "service"
        assert FailureType.RESOURCE.value == "resource"
        assert FailureType.CONFIGURATION.value == "configuration"
        assert FailureType.EXTERNAL.value == "external"

    def test_circuit_breaker_state_enum(self):
        """测试熔断器状态枚举"""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_degradation_strategy_enum(self):
        """测试降级策略枚举"""
        assert DegradationStrategy.DISABLE_FEATURE.value == "disable_feature"
        assert DegradationStrategy.REDUCE_FREQUENCY.value == "reduce_frequency"
        assert DegradationStrategy.LIMIT_CONCURRENCY.value == "limit_concurrency"
        assert DegradationStrategy.USE_CACHE.value == "use_cache"
        assert DegradationStrategy.FALLBACK.value == "fallback"

    def test_recovery_action_enum(self):
        """测试恢复动作枚举"""
        # 验证枚举值存在
        assert hasattr(RecoveryAction, 'RESTART')
        assert hasattr(RecoveryAction, 'SCALE_UP')
        assert hasattr(RecoveryAction, 'FAILOVER')
        assert hasattr(RecoveryAction, 'ROLLBACK')
        assert hasattr(RecoveryAction, 'RELOAD_CONFIG')

    def test_enum_uniqueness(self):
        """测试枚举值的唯一性"""
        # 验证所有枚举值都是唯一的
        resilience_values = [level.value for level in ResilienceLevel]
        assert len(resilience_values) == len(set(resilience_values))

        failure_values = [failure.value for failure in FailureType]
        assert len(failure_values) == len(set(failure_values))

        circuit_values = [state.value for state in CircuitBreakerState]
        assert len(circuit_values) == len(set(circuit_values))

        degradation_values = [strategy.value for strategy in DegradationStrategy]
        assert len(degradation_values) == len(set(degradation_values))

    def test_enum_string_representation(self):
        """测试枚举的字符串表示"""
        # 验证枚举的字符串表示正确
        assert str(ResilienceLevel.BASIC) == "ResilienceLevel.BASIC"
        assert str(FailureType.NETWORK) == "FailureType.NETWORK"
        assert str(CircuitBreakerState.CLOSED) == "CircuitBreakerState.CLOSED"
        assert str(DegradationStrategy.USE_CACHE) == "DegradationStrategy.USE_CACHE"

    def test_enum_iteration(self):
        """测试枚举的可迭代性"""
        # 验证可以迭代枚举成员
        resilience_levels = list(ResilienceLevel)
        assert len(resilience_levels) == 4
        assert ResilienceLevel.NONE in resilience_levels
        assert ResilienceLevel.ENTERPRISE in resilience_levels

        failure_types = list(FailureType)
        assert len(failure_types) == 6
        assert FailureType.NETWORK in failure_types
        assert FailureType.EXTERNAL in failure_types

    def test_enum_member_access(self):
        """测试枚举成员访问"""
        # 验证可以通过名称访问枚举成员
        assert ResilienceLevel['BASIC'] == ResilienceLevel.BASIC
        assert FailureType['DATABASE'] == FailureType.DATABASE
        assert CircuitBreakerState['OPEN'] == CircuitBreakerState.OPEN

        # 验证可以通过值访问枚举成员
        assert ResilienceLevel('basic') == ResilienceLevel.BASIC
        assert FailureType('network') == FailureType.NETWORK
        assert CircuitBreakerState('closed') == CircuitBreakerState.CLOSED

    def test_enum_comparison(self):
        """测试枚举比较"""
        # 验证枚举成员的相等性
        assert ResilienceLevel.BASIC == ResilienceLevel.BASIC
        assert ResilienceLevel.BASIC != ResilienceLevel.ADVANCED

        assert FailureType.NETWORK == FailureType.NETWORK
        assert FailureType.NETWORK != FailureType.DATABASE

        # 验证不同枚举类型的比较（Python允许不同枚举类型比较）
        # 不同枚举类型的成员不相等
        assert ResilienceLevel.BASIC != FailureType.NETWORK

    def test_enum_hashability(self):
        """测试枚举的可哈希性"""
        # 验证枚举成员可以用作字典键
        resilience_dict = {ResilienceLevel.BASIC: "basic_level"}
        assert resilience_dict[ResilienceLevel.BASIC] == "basic_level"

        failure_dict = {FailureType.NETWORK: "network_failure"}
        assert failure_dict[FailureType.NETWORK] == "network_failure"

        # 验证枚举成员可以用作集合元素
        resilience_set = {ResilienceLevel.NONE, ResilienceLevel.BASIC}
        assert len(resilience_set) == 2
        assert ResilienceLevel.BASIC in resilience_set

    def test_enum_member_count(self):
        """测试枚举成员数量"""
        # 验证枚举成员数量符合预期
        assert len(ResilienceLevel) == 4
        assert len(FailureType) == 6
        assert len(CircuitBreakerState) == 3
        assert len(DegradationStrategy) == 5
        assert len(RecoveryAction) == 5

    def test_enum_immutability(self):
        """测试枚举的不可变性"""
        # 枚举值应该是不可变的
        with pytest.raises(AttributeError):
            ResilienceLevel.BASIC.value = "modified"

        with pytest.raises(AttributeError):
            FailureType.NETWORK.value = "modified"

    def test_enum_member_attributes(self):
        """测试枚举成员属性"""
        # 验证枚举成员有name和value属性
        member = ResilienceLevel.BASIC
        assert hasattr(member, 'name')
        assert hasattr(member, 'value')
        assert member.name == 'BASIC'
        assert member.value == 'basic'

        member = FailureType.DATABASE
        assert member.name == 'DATABASE'
        assert member.value == 'database'

    def test_enum_usage_in_data_structures(self):
        """测试枚举在数据结构中的使用"""
        # 测试在列表中的使用
        resilience_list = [ResilienceLevel.NONE, ResilienceLevel.BASIC, ResilienceLevel.ADVANCED]
        assert len(resilience_list) == 3
        assert ResilienceLevel.BASIC in resilience_list

        # 测试在元组中的使用
        failure_tuple = (FailureType.NETWORK, FailureType.DATABASE)
        assert len(failure_tuple) == 2
        assert FailureType.NETWORK in failure_tuple

        # 测试在字典中的使用（键和值）
        strategy_map = {
            DegradationStrategy.DISABLE_FEATURE: "disable",
            DegradationStrategy.USE_CACHE: "cache"
        }
        assert strategy_map[DegradationStrategy.USE_CACHE] == "cache"

        # 测试枚举作为字典值
        state_config = {
            "circuit_breaker": CircuitBreakerState.CLOSED,
            "degradation": DegradationStrategy.FALLBACK
        }
        assert state_config["circuit_breaker"] == CircuitBreakerState.CLOSED
        assert state_config["degradation"] == DegradationStrategy.FALLBACK

    def test_enum_json_serialization(self):
        """测试枚举的JSON序列化"""
        import json

        # 测试枚举值的JSON序列化
        data = {
            "resilience_level": ResilienceLevel.BASIC.value,
            "failure_type": FailureType.NETWORK.value,
            "circuit_state": CircuitBreakerState.CLOSED.value
        }

        json_str = json.dumps(data)
        parsed_data = json.loads(json_str)

        assert parsed_data["resilience_level"] == "basic"
        assert parsed_data["failure_type"] == "network"
        assert parsed_data["circuit_state"] == "closed"

    def test_enum_error_handling(self):
        """测试枚举错误处理"""
        # 测试访问不存在的枚举成员
        with pytest.raises(KeyError):
            ResilienceLevel['NON_EXISTENT']

        with pytest.raises(ValueError):
            ResilienceLevel('non_existent_value')

        # 测试不同枚举类型的比较
        with pytest.raises(TypeError):
            ResilienceLevel.BASIC < FailureType.NETWORK

    def test_enum_documentation(self):
        """测试枚举文档"""
        # 验证枚举有适当的文档字符串
        assert ResilienceLevel.__doc__ is not None
        assert FailureType.__doc__ is not None
        assert CircuitBreakerState.__doc__ is not None
        assert DegradationStrategy.__doc__ is not None

        # 验证文档字符串包含有意义的描述
        assert "弹性级别" in ResilienceLevel.__doc__
        assert "故障类型" in FailureType.__doc__
        assert "熔断器状态" in CircuitBreakerState.__doc__
        assert "降级策略" in DegradationStrategy.__doc__
