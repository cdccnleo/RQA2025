"""
测试缓存工具类

覆盖 cache_utils.py 中的 CacheUtils 类
"""

import pytest
from src.infrastructure.cache_utils import CacheUtils


class TestCacheUtils:
    """CacheUtils 类测试"""

    def test_generate_cache_key_no_args(self):
        """测试生成缓存键（无参数）"""
        key = CacheUtils.generate_cache_key()

        assert isinstance(key, str)
        assert key == ""

    def test_generate_cache_key_single_arg(self):
        """测试生成缓存键（单个参数）"""
        key = CacheUtils.generate_cache_key("test")

        assert key == "test"

    def test_generate_cache_key_multiple_args(self):
        """测试生成缓存键（多个参数）"""
        key = CacheUtils.generate_cache_key("user", 123, "profile")

        assert key == "user_123_profile"

    def test_generate_cache_key_mixed_types(self):
        """测试生成缓存键（混合类型）"""
        key = CacheUtils.generate_cache_key("user", 123, True, None)

        assert key == "user_123_True_None"

    def test_generate_cache_key_with_kwargs(self):
        """测试生成缓存键（包含kwargs）"""
        # generate_cache_key 不处理 kwargs，这里测试忽略 kwargs 的行为
        key = CacheUtils.generate_cache_key("user", id=123)

        assert key == "user"

    def test_is_cacheable_none(self):
        """测试检查可缓存性（None值）"""
        result = CacheUtils.is_cacheable(None)

        assert result == True

    def test_is_cacheable_string(self):
        """测试检查可缓存性（字符串）"""
        result = CacheUtils.is_cacheable("test string")

        assert result == True

    def test_is_cacheable_number(self):
        """测试检查可缓存性（数字）"""
        result = CacheUtils.is_cacheable(123)

        assert result == True

    def test_is_cacheable_list(self):
        """测试检查可缓存性（列表）"""
        result = CacheUtils.is_cacheable([1, 2, 3])

        assert result == True

    def test_is_cacheable_dict(self):
        """测试检查可缓存性（字典）"""
        result = CacheUtils.is_cacheable({"key": "value"})

        assert result == True

    def test_is_cacheable_boolean(self):
        """测试检查可缓存性（布尔值）"""
        result = CacheUtils.is_cacheable(True)

        assert result == True

    def test_calculate_hash_string(self):
        """测试计算哈希（字符串）"""
        hash_value = CacheUtils.calculate_hash("test")

        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

    def test_calculate_hash_number(self):
        """测试计算哈希（数字）"""
        hash_value = CacheUtils.calculate_hash(123)

        assert isinstance(hash_value, str)

    def test_calculate_hash_dict(self):
        """测试计算哈希（字典）"""
        data = {"key": "value", "number": 123}
        hash_value = CacheUtils.calculate_hash(data)

        assert isinstance(hash_value, str)

    def test_calculate_hash_list(self):
        """测试计算哈希（列表）"""
        data = [1, 2, 3, "test"]
        hash_value = CacheUtils.calculate_hash(data)

        assert isinstance(hash_value, str)

    def test_calculate_hash_same_data_same_hash(self):
        """测试相同数据产生相同哈希"""
        data = "test data"
        hash1 = CacheUtils.calculate_hash(data)
        hash2 = CacheUtils.calculate_hash(data)

        assert hash1 == hash2

    def test_calculate_hash_different_data_different_hash(self):
        """测试不同数据产生不同哈希"""
        hash1 = CacheUtils.calculate_hash("data1")
        hash2 = CacheUtils.calculate_hash("data2")

        assert hash1 != hash2

    def test_generate_cache_key_complex_objects(self):
        """测试生成缓存键（复杂对象）"""
        obj = {"key": "value"}
        key = CacheUtils.generate_cache_key("prefix", obj, "suffix")

        assert key == "prefix_{'key': 'value'}_suffix"

    def test_is_cacheable_complex_objects(self):
        """测试检查可缓存性（复杂对象）"""
        complex_obj = {
            "nested": {"data": [1, 2, 3]},
            "list": ["a", "b", "c"],
            "number": 42
        }

        result = CacheUtils.is_cacheable(complex_obj)

        assert result == True

    def test_calculate_hash_consistency(self):
        """测试哈希计算的一致性"""
        # 相同的输入应该产生相同的哈希
        data = ["consistent", "data", 123]
        hash1 = CacheUtils.calculate_hash(data)
        hash2 = CacheUtils.calculate_hash(data.copy())

        assert hash1 == hash2

    def test_generate_cache_key_empty_string(self):
        """测试生成缓存键（空字符串）"""
        key = CacheUtils.generate_cache_key("")

        assert key == ""

    def test_generate_cache_key_special_characters(self):
        """测试生成缓存键（特殊字符）"""
        key = CacheUtils.generate_cache_key("user@domain.com", "path/to/resource")

        assert key == "user@domain.com_path/to/resource"

    def test_calculate_hash_empty_data(self):
        """测试计算哈希（空数据）"""
        hash_value = CacheUtils.calculate_hash("")

        assert isinstance(hash_value, str)

    def test_calculate_hash_none_data(self):
        """测试计算哈希（None数据）"""
        hash_value = CacheUtils.calculate_hash(None)

        assert isinstance(hash_value, str)
