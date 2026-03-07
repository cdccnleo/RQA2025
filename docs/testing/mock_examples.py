#!/usr/bin/env python3
"""
Mock 使用示例

本文件展示了正确的 Mock 使用方法，作为 mock_guidelines.md 的补充示例。
"""

from unittest.mock import patch, MagicMock, call
from unittest.mock import mock_open


class TestCorrectMockUsage:
    """正确的 Mock 使用示例"""

    @patch('requests.get')
    def test_api_call_with_mock(self, mock_get):
        """示例：Mock 外部 API 调用"""
        # 配置 mock 响应
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": [1, 2, 3]}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # 执行测试（这里假设有一个 api_client 模块）
        # result = api_client.get_data()

        # 验证调用
        mock_get.assert_called_once()
        # assert result == {"status": "success", "data": [1, 2, 3]}

    @patch('builtins.open', mock_open(read_data='file content'))
    def test_file_operation_with_mock(self):
        """示例：Mock 文件操作"""
        # 执行测试
        # content = file_service.read_file("test.txt")

        # 验证
        # assert content == "file content"

    @patch.object(SomeClass, 'some_method')
    def test_partial_mock(self, mock_method):
        """示例：部分 Mock 类方法"""
        mock_method.return_value = "mocked_result"

        # 测试代码
        # instance = SomeClass()
        # result = instance.some_method()

        # assert result == "mocked_result"

    def test_context_manager_mock(self):
        """示例：使用上下文管理器进行 Mock"""
        with patch('external.module') as mock_module:
            mock_module.some_function.return_value = "mocked_result"

            # 执行测试
            # result = some_function_under_test()

            # 验证
            # assert result == "expected_result"


class TestMockVerification:
    """Mock 验证示例"""

    @patch('target.module')
    def test_mock_verification(self, mock_module):
        """示例：验证 Mock 调用"""
        # 配置 mock
        mock_module.some_method.return_value = "result"

        # 执行测试
        # result = function_under_test()

        # 验证调用
        mock_module.some_method.assert_called_once()
        mock_module.some_method.assert_called_with("expected_arg")

        # 验证调用历史
        expected_calls = [
            call("first_call"),
            call("second_call")
        ]
        # mock_module.some_method.assert_has_calls(expected_calls)

    @patch('target.module')
    def test_mock_side_effects(self, mock_module):
        """示例：Mock 副作用"""
        # 抛出异常
        mock_module.error_method.side_effect = ValueError("Error")

        # 多次调用返回不同值
        mock_module.multi_method.side_effect = ["first", "second", "third"]

        # 自定义函数
        def custom_side_effect(*args, **kwargs):
            return f"called with {args}, {kwargs}"

        mock_module.custom_method.side_effect = custom_side_effect


class TestIsolatedTests:
    """测试隔离示例"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.mock_data = {"test": "data"}

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理 mock 状态

    @patch('external.module')
    def test_isolated_method(self, mock_module):
        """示例：独立的测试方法"""
        # 每个测试都是独立的，不依赖其他测试的状态
        mock_module.some_method.return_value = "isolated_result"

        # 测试代码


# 错误示例（注释掉，仅作参考）
class TestIncorrectMockUsage:
    """错误的 Mock 使用示例（仅作参考）"""

    def test_global_mock_wrong(self):
        """❌ 错误示例：全局 Mock"""
        # 以下代码会导致测试污染，应该避免
        # import sys
        # from unittest.mock import MagicMock
        # sys.modules['some_module'] = MagicMock()  # ❌ 全局 Mock

        # 正确做法：使用 @patch 装饰器

    def test_shared_mock_wrong(self):
        """❌ 错误示例：测试间共享 Mock 状态"""
        # 以下代码会导致测试间依赖，应该避免
        # class TestSharedState:
        #     shared_mock = None  # ❌ 共享状态

        # 正确做法：每个测试独立创建 Mock


# 辅助类（用于示例）
class SomeClass:
    """示例类，用于演示部分 Mock"""

    def some_method(self):
        return "original_result"


if __name__ == "__main__":
    print("Mock 使用示例")
    print("请参考 mock_guidelines.md 获取完整的规范说明")
