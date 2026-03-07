"""
性能测试 - 参数增强器缓存优化

测试ParameterEnhancer的缓存机制和性能优化
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from src.infrastructure.api.documentation_enhancement.parameter_enhancer import (
    ParameterEnhancer,
    APIParameterDocumentation
)


class TestParameterEnhancerPerformance:
    """测试参数增强器的性能优化"""

    @pytest.fixture
    def enhancer(self):
        """创建增强器实例"""
        return ParameterEnhancer()

    @pytest.fixture
    def sample_params(self):
        """创建示例参数列表"""
        return [
            APIParameterDocumentation("user_id", "string", True, "用户ID"),
            APIParameterDocumentation("email", "string", True, "邮箱地址"),
            APIParameterDocumentation("age", "integer", False, "年龄"),
            APIParameterDocumentation("price", "number", False, "价格"),
            APIParameterDocumentation("is_active", "boolean", False, "是否激活"),
            APIParameterDocumentation("tags", "array", False, "标签列表"),
            APIParameterDocumentation("config", "object", False, "配置对象"),
        ]

    def test_cache_initialization(self, enhancer):
        """测试缓存初始化"""
        assert hasattr(enhancer, 'example_cache')
        assert hasattr(enhancer, 'rules_cache')
        assert hasattr(enhancer, 'cache_max_size')
        assert enhancer.cache_max_size == 1000
        assert enhancer.cache_hit_count == 0
        assert enhancer.cache_miss_count == 0

    def test_cache_functionality(self, enhancer, sample_params):
        """测试缓存功能"""
        # 第一次调用 - 缓存未命中
        param = sample_params[0]  # user_id string
        result1 = enhancer._generate_example_value(param)

        assert result1 == "abc123"  # user_id生成的值
        assert enhancer.cache_hit_count == 0
        assert enhancer.cache_miss_count == 1
        assert len(enhancer.example_cache) == 1

        # 第二次调用相同参数 - 缓存命中
        result2 = enhancer._generate_example_value(param)

        assert result1 == result2
        assert enhancer.cache_hit_count == 1
        assert enhancer.cache_miss_count == 1
        assert len(enhancer.example_cache) == 1

    def test_cache_performance(self, enhancer, sample_params):
        """测试缓存性能提升"""
        # 预热缓存
        for param in sample_params:
            enhancer._generate_example_value(param)

        # 测试大量重复请求的性能
        iterations = 1000

        start_time = time.time()
        for _ in range(iterations):
            for param in sample_params:
                enhancer._generate_example_value(param)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_call = total_time / (iterations * len(sample_params))

        # 验证缓存命中率
        stats = enhancer.get_cache_stats()
        assert stats['hit_rate'] > 0.8  # 应该有80%以上的命中率
        assert stats['cache_size'] == len(sample_params)

        # 性能应该很快（每个调用应该小于1ms）
        assert avg_time_per_call < 0.001

    def test_cache_stats(self, enhancer, sample_params):
        """测试缓存统计信息"""
        # 初始状态
        stats = enhancer.get_cache_stats()
        assert stats['cache_hit_count'] == 0
        assert stats['cache_miss_count'] == 0
        assert stats['total_requests'] == 0
        assert stats['hit_rate'] == 0.0

        # 执行一些操作
        for param in sample_params[:3]:
            enhancer._generate_example_value(param)

        stats = enhancer.get_cache_stats()
        assert stats['cache_miss_count'] == 3
        assert stats['total_requests'] == 3
        assert stats['hit_rate'] == 0.0

        # 再次执行相同的操作
        for param in sample_params[:3]:
            enhancer._generate_example_value(param)

        stats = enhancer.get_cache_stats()
        assert stats['cache_hit_count'] == 3
        assert stats['cache_miss_count'] == 3
        assert stats['total_requests'] == 6
        assert stats['hit_rate'] == 0.5

    def test_cache_size_limit(self, enhancer):
        """测试缓存大小限制"""
        # 创建超过缓存限制的参数
        params = []
        for i in range(1100):  # 超过默认的1000限制
            param = APIParameterDocumentation(f"param_{i}", "string", False, f"参数{i}")
            params.append(param)

        # 生成示例值
        for param in params[:1050]:  # 多生成一些来测试LRU
            enhancer._generate_example_value(param)

        stats = enhancer.get_cache_stats()
        # 缓存大小应该不超过最大限制
        assert stats['cache_size'] <= stats['max_cache_size']

    def test_cache_clear(self, enhancer, sample_params):
        """测试缓存清空功能"""
        # 填充缓存
        for param in sample_params:
            enhancer._generate_example_value(param)

        assert len(enhancer.example_cache) > 0
        assert enhancer.cache_hit_count > 0
        assert enhancer.cache_miss_count > 0

        # 清空缓存
        enhancer.clear_cache()

        assert len(enhancer.example_cache) == 0
        assert len(enhancer.rules_cache) == 0
        assert enhancer.cache_hit_count == 0
        assert enhancer.cache_miss_count == 0

    def test_different_param_types_performance(self, enhancer):
        """测试不同参数类型的性能"""
        param_types = [
            ("string", "username", "testuser"),
            ("integer", "count", 0),
            ("number", "price", 0.0),
            ("boolean", "enabled", True),
            ("array", "items", []),
            ("object", "config", {}),
        ]

        # 测试每种类型的性能
        for param_type, param_name, expected_value in param_types:
            param = APIParameterDocumentation(param_name, param_type, False, f"测试{param_type}")

            # 第一次调用
            result1 = enhancer._generate_example_value(param)
            # 第二次调用（缓存命中）
            result2 = enhancer._generate_example_value(param)

            assert result1 == result2
            # 对于简单类型，验证结果符合预期
            if param_type in ["boolean", "array", "object"]:
                assert result1 == expected_value

    @pytest.mark.parametrize("iterations", [10, 100, 1000])
    def test_scaling_performance(self, enhancer, sample_params, iterations):
        """测试不同规模下的性能表现"""
        # 预热
        for param in sample_params:
            enhancer._generate_example_value(param)

        # 测试大规模调用
        start_time = time.time()
        for _ in range(iterations):
            for param in sample_params:
                enhancer._generate_example_value(param)
        end_time = time.time()

        total_calls = iterations * len(sample_params)
        total_time = end_time - start_time
        avg_time_per_call = total_time / total_calls

        # 即使在大规模测试下，性能也应该保持良好
        assert avg_time_per_call < 0.01  # 每调用小于10ms

        # 缓存命中率应该很高
        stats = enhancer.get_cache_stats()
        if iterations > 1:
            assert stats['hit_rate'] > 0.5


if __name__ == "__main__":
    pytest.main([__file__])
