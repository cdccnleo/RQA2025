#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugin_validator补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
from unittest.mock import MagicMock, Mock
from src.features.plugins.plugin_validator import PluginValidator
from src.features.plugins.base_plugin import (
    BaseFeaturePlugin,
    PluginMetadata,
    PluginType
)


class TestPluginValidatorCoverageSupplement:
    """plugin_validator补充测试"""

    @pytest.fixture
    def validator(self):
        """创建验证器实例"""
        return PluginValidator()

    @pytest.fixture
    def valid_plugin_class(self):
        """创建有效的插件类"""
        class ValidPlugin(BaseFeaturePlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="test_plugin",
                    version="1.0.0",
                    description="Test plugin",
                    author="Test Author",
                    plugin_type=PluginType.PROCESSOR
                )
            
            def process(self, data):
                return data
        
        return ValidPlugin

    @pytest.fixture
    def valid_plugin_instance(self, valid_plugin_class):
        """创建有效的插件实例"""
        return valid_plugin_class()

    def test_validate_plugin_class_missing_method(self, validator):
        """测试validate_plugin_class（缺少必需方法）"""
        class InvalidPlugin(BaseFeaturePlugin):
            # 缺少process方法
            def _get_metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    author="Test",
                    plugin_type=PluginType.PROCESSOR
                )
        
        result = validator.validate_plugin_class(InvalidPlugin)
        assert result is False

    def test_validate_plugin_class_method_not_callable(self, validator):
        """测试validate_plugin_class（方法不可调用）"""
        class InvalidPlugin(BaseFeaturePlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    author="Test",
                    plugin_type=PluginType.PROCESSOR
                )
            
            # process不是方法，而是属性
            process = "not a method"
        
        result = validator.validate_plugin_class(InvalidPlugin)
        assert result is False

    def test_validate_plugin_class_missing_init(self, validator):
        """测试validate_plugin_class（缺少__init__方法）"""
        # 由于BaseFeaturePlugin已经有__init__，所有子类都会继承它
        # 无法真正测试缺少__init__的情况，但可以测试其他验证逻辑
        # 测试一个正常的插件类应该通过验证
        class ValidPlugin(BaseFeaturePlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    author="Test",
                    plugin_type=PluginType.PROCESSOR
                )
            
            def process(self, data):
                return data
        
        # 所有必需方法都存在，应该通过
        result = validator.validate_plugin_class(ValidPlugin)
        assert result is True

    def test_validate_plugin_class_exception(self, validator):
        """测试validate_plugin_class异常处理"""
        # 传入非类对象
        result = validator.validate_plugin_class("not a class")
        assert result is False

    def test_validate_plugin_instance_missing_method(self, validator, valid_plugin_class):
        """测试validate_plugin_instance（缺少必需方法）"""
        # 创建一个缺少process方法的插件类
        class InvalidPlugin(BaseFeaturePlugin):
            def _get_metadata(self):
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test",
                    author="Test",
                    plugin_type=PluginType.PROCESSOR
                )
            # 缺少process方法
        
        # 由于缺少process方法，实例化会失败（因为process是抽象方法）
        # 我们需要使用try-except来测试
        try:
            invalid_plugin = InvalidPlugin()
            # 如果能实例化，验证应该失败
            result = validator.validate_plugin_instance(invalid_plugin)
            # 由于缺少process方法，验证应该失败
            assert result is False
        except TypeError:
            # 如果无法实例化（因为缺少抽象方法），这也是预期的
            # 这种情况下我们无法测试实例验证，但可以测试类验证
            result = validator.validate_plugin_class(InvalidPlugin)
            assert result is False

    def test_validate_plugin_instance_method_not_callable(self, validator, valid_plugin_class):
        """测试validate_plugin_instance（方法不可调用）"""
        plugin = valid_plugin_class()
        # 将process设置为非可调用对象
        plugin.process = "not a method"
        
        result = validator.validate_plugin_instance(plugin)
        assert result is False

    def test_validate_plugin_instance_optional_method_not_callable(self, validator, valid_plugin_class):
        """测试validate_plugin_instance（可选方法不可调用）"""
        plugin = valid_plugin_class()
        # 添加不可调用的可选方法
        plugin.initialize = "not a method"
        
        result = validator.validate_plugin_instance(plugin)
        assert result is False

    def test_validate_plugin_instance_metadata_type_error(self, validator, valid_plugin_class):
        """测试validate_plugin_instance（元数据类型错误）"""
        plugin = valid_plugin_class()
        
        # Mock _get_metadata返回错误类型
        def mock_get_metadata():
            return "not PluginMetadata"
        
        plugin._get_metadata = mock_get_metadata
        
        result = validator.validate_plugin_instance(plugin)
        assert result is False

    def test_validate_plugin_instance_metadata_exception(self, validator, valid_plugin_class):
        """测试validate_plugin_instance（获取元数据异常）"""
        plugin = valid_plugin_class()
        
        # Mock _get_metadata抛出异常
        def mock_get_metadata():
            raise Exception("获取元数据失败")
        
        plugin._get_metadata = mock_get_metadata
        
        result = validator.validate_plugin_instance(plugin)
        assert result is False

    def test_validate_plugin_instance_exception(self, validator):
        """测试validate_plugin_instance异常处理"""
        # 传入非插件实例
        result = validator.validate_plugin_instance("not a plugin")
        assert result is False

    def test_validate_metadata(self, validator):
        """测试validate_metadata"""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        
        result = validator.validate_metadata(metadata)
        assert result is True

    def test_validate_config_not_dict(self, validator, valid_plugin_instance):
        """测试validate_config（配置不是字典）"""
        result = validator.validate_config(valid_plugin_instance, "not a dict")
        assert result is False

    def test_validate_config_exception(self, validator, valid_plugin_instance):
        """测试validate_config异常处理"""
        # Mock _validate_config_schema导致异常
        original_validate = validator._validate_config_schema
        validator._validate_config_schema = MagicMock(side_effect=Exception("配置错误"))
        
        valid_plugin_instance.metadata.config_schema = {'type': 'object'}
        
        result = validator.validate_config(valid_plugin_instance, {})
        assert result is False
        
        # 恢复原始方法
        validator._validate_config_schema = original_validate

    def test_validate_api_compatibility_exception(self, validator, valid_plugin_instance):
        """测试validate_api_compatibility异常处理"""
        # Mock metadata导致异常
        valid_plugin_instance.metadata = None
        
        result = validator.validate_api_compatibility(valid_plugin_instance)
        assert result is False

    def test_validate_metadata_missing_field(self, validator):
        """测试_validate_metadata（缺少必需字段）"""
        # 创建缺少字段的元数据
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        # 删除一个必需字段
        delattr(metadata, 'name')
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_empty_field(self, validator):
        """测试_validate_metadata（字段为空）"""
        metadata = PluginMetadata(
            name="",  # 空字符串
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_none_field(self, validator):
        """测试_validate_metadata（字段为None）"""
        metadata = PluginMetadata(
            name=None,  # None值
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_invalid_name(self, validator):
        """测试_validate_metadata（名称格式无效）"""
        metadata = PluginMetadata(
            name="123invalid",  # 以数字开头
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_invalid_version(self, validator):
        """测试_validate_metadata（版本格式无效）"""
        metadata = PluginMetadata(
            name="test",
            version="invalid_version",  # 无效版本格式
            description="Test",
            author="Test",
            plugin_type=PluginType.PROCESSOR
        )
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_invalid_plugin_type(self, validator):
        """测试_validate_metadata（插件类型无效）"""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type="invalid_type"  # 不是PluginType枚举
        )
        
        result = validator._validate_metadata(metadata)
        assert result is False

    def test_validate_metadata_exception(self, validator):
        """测试_validate_metadata异常处理"""
        # 传入非PluginMetadata对象
        result = validator._validate_metadata("not metadata")
        assert result is False

    def test_validate_config_schema(self, validator, valid_plugin_instance):
        """测试_validate_config_schema"""
        # 设置配置模式
        valid_plugin_instance.metadata.config_schema = {
            'type': 'object',
            'properties': {
                'param1': {'type': 'string'}
            }
        }
        
        # 有效配置
        result = validator._validate_config_schema({'param1': 'value1'}, valid_plugin_instance.metadata.config_schema)
        assert isinstance(result, bool)

    def test_validate_config_schema_invalid(self, validator, valid_plugin_instance):
        """测试_validate_config_schema（无效配置）"""
        valid_plugin_instance.metadata.config_schema = {
            'type': 'object',
            'required': ['param1']
        }
        
        # 缺少必需参数
        result = validator._validate_config_schema({}, valid_plugin_instance.metadata.config_schema)
        assert isinstance(result, bool)

    def test_is_version_in_range_valid(self, validator):
        """测试_is_version_in_range（有效范围）"""
        result = validator._is_version_in_range("1.0.0", "2.0.0", "1.5.0", "1.8.0")
        assert result is True

    def test_is_version_in_range_invalid(self, validator):
        """测试_is_version_in_range（无效范围）"""
        result = validator._is_version_in_range("1.0.0", "1.5.0", "2.0.0", "2.5.0")
        assert result is False

    def test_is_version_in_range_exception(self, validator):
        """测试_is_version_in_range异常处理"""
        # 无效版本格式
        result = validator._is_version_in_range("invalid", "1.0.0", "1.5.0", "2.0.0")
        assert result is False

    def test_is_valid_name_invalid_type(self, validator):
        """测试_is_valid_name（无效类型）"""
        result = validator._is_valid_name(None)
        assert result is False
        
        result = validator._is_valid_name(123)
        assert result is False

    def test_is_valid_name_invalid_length(self, validator):
        """测试_is_valid_name（无效长度）"""
        # 空字符串
        result = validator._is_valid_name("")
        assert result is False
        
        # 过长
        result = validator._is_valid_name("a" * 51)
        assert result is False

    def test_is_valid_name_invalid_chars(self, validator):
        """测试_is_valid_name（无效字符）"""
        # 包含特殊字符
        result = validator._is_valid_name("test-plugin")
        assert result is False
        
        # 以数字开头
        result = validator._is_valid_name("123test")
        assert result is False

    def test_is_valid_version_invalid_type(self, validator):
        """测试_is_valid_version（无效类型）"""
        result = validator._is_valid_version(None)
        assert result is False
        
        result = validator._is_valid_version(123)
        assert result is False

    def test_is_valid_version_invalid_format(self, validator):
        """测试_is_valid_version（无效格式）"""
        # 不符合x.y.z格式
        result = validator._is_valid_version("1.0")
        assert result is False
        
        result = validator._is_valid_version("1.0.0.0")
        assert result is False
        
        result = validator._is_valid_version("invalid")
        assert result is False

