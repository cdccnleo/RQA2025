"""
测试格式化和显示相关常量定义

覆盖 FormatConstants 类的所有常量值
"""

import pytest
from src.infrastructure.constants.format_constants import FormatConstants


class TestFormatConstants:
    """FormatConstants 单元测试"""

    def test_separator_length_constants(self):
        """测试分隔符长度相关常量"""
        assert FormatConstants.SEPARATOR_LENGTH_SHORT == 40
        assert FormatConstants.SEPARATOR_LENGTH_MEDIUM == 50
        assert FormatConstants.SEPARATOR_LENGTH_LONG == 60
        assert FormatConstants.SEPARATOR_LENGTH_FULL == 80
        assert FormatConstants.SEPARATOR_LENGTH_WIDE == 100

    def test_separator_char_constants(self):
        """测试分隔符字符相关常量"""
        assert FormatConstants.SEPARATOR_CHAR_DASH == '-'
        assert FormatConstants.SEPARATOR_CHAR_EQUAL == '='
        assert FormatConstants.SEPARATOR_CHAR_STAR == '*'
        assert FormatConstants.SEPARATOR_CHAR_HASH == '#'

    def test_indent_level_constants(self):
        """测试缩进级别相关常量"""
        assert FormatConstants.INDENT_LEVEL_0 == 0
        assert FormatConstants.INDENT_LEVEL_1 == 2
        assert FormatConstants.INDENT_LEVEL_2 == 4
        assert FormatConstants.INDENT_LEVEL_3 == 6
        assert FormatConstants.INDENT_LEVEL_4 == 8

    def test_json_format_constants(self):
        """测试JSON格式化相关常量"""
        assert FormatConstants.JSON_INDENT == 2
        assert FormatConstants.JSON_SEPARATORS == (',', ': ')
        assert FormatConstants.JSON_ENSURE_ASCII is False

    def test_log_format_constants(self):
        """测试日志格式相关常量"""
        assert FormatConstants.LOG_MAX_MESSAGE_LENGTH == 1000
        assert FormatConstants.LOG_MAX_STACKTRACE_DEPTH == 10

    def test_table_display_constants(self):
        """测试表格显示相关常量"""
        assert FormatConstants.TABLE_COLUMN_WIDTH_SMALL == 10
        assert FormatConstants.TABLE_COLUMN_WIDTH_MEDIUM == 20
        assert FormatConstants.TABLE_COLUMN_WIDTH_LARGE == 30
        assert FormatConstants.TABLE_COLUMN_WIDTH_XLARGE == 50

    def test_truncate_constants(self):
        """测试截断相关常量"""
        assert FormatConstants.TRUNCATE_LENGTH_SHORT == 50
        assert FormatConstants.TRUNCATE_LENGTH_MEDIUM == 100
        assert FormatConstants.TRUNCATE_LENGTH_LONG == 200
        assert FormatConstants.TRUNCATE_SUFFIX == '...'

    def test_encoding_constants(self):
        """测试编码相关常量"""
        assert FormatConstants.DEFAULT_ENCODING == 'utf-8'
        assert FormatConstants.FALLBACK_ENCODING == 'latin-1'

    def test_separator_length_relationships(self):
        """测试分隔符长度关系"""
        assert (FormatConstants.SEPARATOR_LENGTH_SHORT <
                FormatConstants.SEPARATOR_LENGTH_MEDIUM <
                FormatConstants.SEPARATOR_LENGTH_LONG <
                FormatConstants.SEPARATOR_LENGTH_FULL <
                FormatConstants.SEPARATOR_LENGTH_WIDE)

    def test_indent_level_relationships(self):
        """测试缩进级别关系"""
        assert (FormatConstants.INDENT_LEVEL_0 <
                FormatConstants.INDENT_LEVEL_1 <
                FormatConstants.INDENT_LEVEL_2 <
                FormatConstants.INDENT_LEVEL_3 <
                FormatConstants.INDENT_LEVEL_4)

    def test_table_column_width_relationships(self):
        """测试表格列宽关系"""
        assert (FormatConstants.TABLE_COLUMN_WIDTH_SMALL <
                FormatConstants.TABLE_COLUMN_WIDTH_MEDIUM <
                FormatConstants.TABLE_COLUMN_WIDTH_LARGE <
                FormatConstants.TABLE_COLUMN_WIDTH_XLARGE)

    def test_truncate_length_relationships(self):
        """测试截断长度关系"""
        assert (FormatConstants.TRUNCATE_LENGTH_SHORT <
                FormatConstants.TRUNCATE_LENGTH_MEDIUM <
                FormatConstants.TRUNCATE_LENGTH_LONG)

    def test_indent_progression(self):
        """测试缩进递增规律"""
        # 每个级别应该比前一个多2个空格
        expected_indents = [0, 2, 4, 6, 8]
        actual_indents = [
            FormatConstants.INDENT_LEVEL_0,
            FormatConstants.INDENT_LEVEL_1,
            FormatConstants.INDENT_LEVEL_2,
            FormatConstants.INDENT_LEVEL_3,
            FormatConstants.INDENT_LEVEL_4
        ]
        assert actual_indents == expected_indents

    def test_positive_values(self):
        """测试所有数值常量都是正值"""
        numeric_constants = [
            FormatConstants.SEPARATOR_LENGTH_SHORT,
            FormatConstants.SEPARATOR_LENGTH_MEDIUM,
            FormatConstants.SEPARATOR_LENGTH_LONG,
            FormatConstants.SEPARATOR_LENGTH_FULL,
            FormatConstants.SEPARATOR_LENGTH_WIDE,
            FormatConstants.INDENT_LEVEL_1,
            FormatConstants.INDENT_LEVEL_2,
            FormatConstants.INDENT_LEVEL_3,
            FormatConstants.INDENT_LEVEL_4,
            FormatConstants.JSON_INDENT,
            FormatConstants.LOG_MAX_MESSAGE_LENGTH,
            FormatConstants.LOG_MAX_STACKTRACE_DEPTH,
            FormatConstants.TABLE_COLUMN_WIDTH_SMALL,
            FormatConstants.TABLE_COLUMN_WIDTH_MEDIUM,
            FormatConstants.TABLE_COLUMN_WIDTH_LARGE,
            FormatConstants.TABLE_COLUMN_WIDTH_XLARGE,
            FormatConstants.TRUNCATE_LENGTH_SHORT,
            FormatConstants.TRUNCATE_LENGTH_MEDIUM,
            FormatConstants.TRUNCATE_LENGTH_LONG
        ]

        for constant in numeric_constants:
            assert constant > 0, f"Constant {constant} should be positive"

    def test_string_values(self):
        """测试字符串常量"""
        string_constants = [
            FormatConstants.SEPARATOR_CHAR_DASH,
            FormatConstants.SEPARATOR_CHAR_EQUAL,
            FormatConstants.SEPARATOR_CHAR_STAR,
            FormatConstants.SEPARATOR_CHAR_HASH,
            FormatConstants.TRUNCATE_SUFFIX,
            FormatConstants.DEFAULT_ENCODING,
            FormatConstants.FALLBACK_ENCODING
        ]

        for constant in string_constants:
            assert isinstance(constant, str), f"Constant {constant} should be a string"
            assert len(constant) > 0, f"Constant {constant} should not be empty"

    def test_boolean_values(self):
        """测试布尔常量"""
        assert isinstance(FormatConstants.JSON_ENSURE_ASCII, bool)

    def test_tuple_values(self):
        """测试元组常量"""
        assert isinstance(FormatConstants.JSON_SEPARATORS, tuple)
        assert len(FormatConstants.JSON_SEPARATORS) == 2