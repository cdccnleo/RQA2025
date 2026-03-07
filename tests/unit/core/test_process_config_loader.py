# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试业务流程配置加载器

测试process_config_loader.py中的配置加载和验证功能
"""

import unittest
import tempfile
from pathlib import Path

from src.core.orchestration.configs.process_config_loader import (
    ProcessConfigLoader,
    ProcessConfiguration,
    ProcessState,
    StateTransition,
    EventSchema,
    ProcessStateType,
    ConfigValidationError,
    ConfigVersionError
)


class TestProcessConfigLoader(unittest.TestCase):
    """测试业务流程配置加载器"""

    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = ProcessConfigLoader(self.temp_dir)

    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.loader.config_dir, Path)
        self.assertEqual(str(self.loader.config_dir), self.temp_dir)
        self.assertIsInstance(self.loader.loaded_configs, dict)
        self.assertIsInstance(self.loader.config_cache, dict)
        self.assertTrue(self.loader.config_dir.exists())

    def test_list_available_processes(self):
        """测试列出可用流程"""
        processes = self.loader.list_available_processes()
        self.assertIsInstance(processes, list)
        # 初始状态下应该没有配置文件
        self.assertEqual(len(processes), 0)


class TestDataClasses(unittest.TestCase):
    """测试数据类"""

    def test_state_transition_creation(self):
        """测试状态转换创建"""
        transition = StateTransition(
            to='target_state',
            condition='some_condition',
            event='some_event',
            description='Test transition'
        )

        self.assertEqual(transition.to, 'target_state')
        self.assertEqual(transition.condition, 'some_condition')
        self.assertEqual(transition.event, 'some_event')
        self.assertEqual(transition.description, 'Test transition')

    def test_process_state_creation(self):
        """测试流程状态创建"""
        transition = StateTransition(to='state2', condition='cond', event='evt')
        state = ProcessState(
            description='Test state',
            actions=['action1', 'action2'],
            transitions=[transition],
            final=True
        )

        self.assertEqual(state.description, 'Test state')
        self.assertEqual(state.actions, ['action1', 'action2'])
        self.assertEqual(len(state.transitions), 1)
        self.assertTrue(state.final)

    def test_event_schema_creation(self):
        """测试事件模式创建"""
        schema = EventSchema(
            description='Test event',
            data_schema={'field1': 'string', 'field2': 'number'}
        )

        self.assertEqual(schema.description, 'Test event')
        self.assertEqual(schema.data_schema['field1'], 'string')

    def test_process_configuration_creation(self):
        """测试流程配置创建"""
        config = ProcessConfiguration(
            process_name='test_process',
            version='1.0.0',
            description='Test process',
            workflow={'key': 'value'},
            events={},
            configuration={'setting': 'value'},
            dependencies={'dep': 'version'},
            compatibility={'version': '1.0.0'}
        )

        self.assertEqual(config.process_name, 'test_process')
        self.assertEqual(config.version, '1.0.0')
        self.assertEqual(config.workflow, {'key': 'value'})


class TestEnums(unittest.TestCase):
    """测试枚举"""

    def test_process_state_type_values(self):
        """测试流程状态类型值"""
        self.assertEqual(ProcessStateType.INITIAL.value, 'initial')
        self.assertEqual(ProcessStateType.INTERMEDIATE.value, 'intermediate')
        self.assertEqual(ProcessStateType.FINAL.value, 'final')


if __name__ == '__main__':
    unittest.main()