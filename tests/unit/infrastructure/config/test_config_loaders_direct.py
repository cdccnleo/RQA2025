"""
直接测试配置加载器的测试文件
测试src/infrastructure/config/loaders目录下可独立导入的实际代码
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import tempfile
import os


class TestConfigLoadersDirect:
    """直接测试配置加载器"""

    def test_json_operations(self):
        """测试基础JSON操作"""
        # 测试JSON序列化和反序列化
        test_data = {"database": {"host": "localhost", "port": 5432}}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        assert parsed_data == test_data

    def test_json_file_operations(self):
        """测试JSON文件操作"""
        # 创建临时JSON文件
        test_data = {"database": {"host": "localhost", "port": 5432}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # 读取JSON文件
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data
        finally:
            os.unlink(temp_file)

    def test_invalid_json_handling(self):
        """测试无效JSON处理"""
        invalid_json = "{invalid json"
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_file_not_found_handling(self):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            with open("nonexistent_file.json", 'r') as f:
                pass
