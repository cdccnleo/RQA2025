#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程配置加载器
用于加载和解析YAML格式的业务流程配置文件，支持配置验证和版本管理
"""

import yaml
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):

    """配置验证错误"""


class ConfigVersionError(Exception):

    """配置版本错误"""


class ProcessStateType(Enum):

    """流程状态类型"""
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


@dataclass
class StateTransition:

    """状态转换定义"""
    to: str
    condition: str
    event: str
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ProcessState:

    """流程状态定义"""
    description: str
    actions: List[str]
    transitions: List[StateTransition]
    final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'description': self.description,
            'actions': self.actions,
            'transitions': [t.to_dict() for t in self.transitions],
            'final': self.final
        }


@dataclass
class EventSchema:

    """事件模式定义"""
    description: str
    data_schema: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ProcessConfiguration:

    """流程配置定义"""
    process_name: str
    version: str
    description: str
    workflow: Dict[str, Any]
    events: Dict[str, EventSchema]
    configuration: Dict[str, Any]
    dependencies: Dict[str, Any]
    compatibility: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'process_name': self.process_name,
            'version': self.version,
            'description': self.description,
            'workflow': self.workflow,
            'events': {k: v.to_dict() for k, v in self.events.items()},
            'configuration': self.configuration,
            'dependencies': self.dependencies,
            'compatibility': self.compatibility
        }


class ProcessConfigLoader:

    """业务流程配置加载器"""

    def __init__(self, config_dir: str = "config / processes"):
        """
        初始化配置加载器

        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.loaded_configs: Dict[str, ProcessConfiguration] = {}
        self.config_cache: Dict[str, Dict[str, Any]] = {}

        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"业务流程配置加载器初始化完成，配置目录: {self.config_dir}")

    def load_process_config(self, process_name: str) -> ProcessConfiguration:
        """
        加载指定的流程配置

        Args:
            process_name: 流程名称

        Returns:
            流程配置对象

        Raises:
            ConfigValidationError: 配置验证失败
            ConfigVersionError: 配置版本不兼容
            FileNotFoundError: 配置文件不存在
        """
        # 检查缓存
        if process_name in self.loaded_configs:
            logger.debug(f"从缓存加载流程配置: {process_name}")
            return self.loaded_configs[process_name]

        # 构建配置文件路径
        config_file = self.config_dir / f"{process_name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"流程配置文件不存在: {config_file}")

        try:
            # 加载YAML配置
            logger.info(f"加载流程配置文件: {config_file}")
            with open(config_file, 'r', encoding='utf - 8') as f:
                config_data = yaml.safe_load(f)

            # 验证配置
            self._validate_config(config_data)

            # 解析配置
            process_config = self._parse_config(config_data)

            # 缓存配置
            self.loaded_configs[process_name] = process_config
            self.config_cache[process_name] = config_data

            logger.info(f"流程配置加载成功: {process_name} v{process_config.version}")
            return process_config

        except yaml.YAMLError as e:
            raise ConfigValidationError(f"YAML解析错误: {e}")
        except Exception as e:
            raise ConfigValidationError(f"配置加载失败: {e}")

    def load_all_process_configs(self) -> Dict[str, ProcessConfiguration]:
        """
        加载所有流程配置

        Returns:
            所有流程配置的字典
        """
        configs = {}

        # 查找所有YAML配置文件
        yaml_files = list(self.config_dir.glob("*.yaml"))

        for config_file in yaml_files:
            try:
                process_name = config_file.stem
                config = self.load_process_config(process_name)
                configs[process_name] = config
            except Exception as e:
                logger.warning(f"加载配置文件失败 {config_file}: {e}")
                continue

        logger.info(f"成功加载 {len(configs)} 个流程配置")
        return configs

    def reload_process_config(self, process_name: str) -> ProcessConfiguration:
        """
        重新加载流程配置

        Args:
            process_name: 流程名称

        Returns:
            重新加载的流程配置对象
        """
        # 清除缓存
        if process_name in self.loaded_configs:
            del self.loaded_configs[process_name]
        if process_name in self.config_cache:
            del self.config_cache[process_name]

        # 重新加载
        return self.load_process_config(process_name)

    def get_process_states(self, process_name: str) -> Dict[str, ProcessState]:
        """
        获取流程的所有状态定义

        Args:
            process_name: 流程名称

        Returns:
            状态定义字典
        """
        config = self.load_process_config(process_name)
        return self._parse_states(config.workflow)

    def get_process_events(self, process_name: str) -> Dict[str, EventSchema]:
        """
        获取流程的所有事件定义

        Args:
            process_name: 流程名称

        Returns:
            事件定义字典
        """
        config = self.load_process_config(process_name)
        return config.events

    def get_process_configuration(self, process_name: str) -> Dict[str, Any]:
        """
        获取流程的配置参数

        Args:
            process_name: 流程名称

        Returns:
            配置参数字典
        """
        config = self.load_process_config(process_name)
        return config.configuration

    def validate_process_config(self, process_name: str) -> bool:
        """
        验证流程配置

        Args:
            process_name: 流程名称

        Returns:
            验证是否通过
        """
        try:
            config = self.load_process_config(process_name)
            self._validate_config_integrity(config)
            return True
        except Exception as e:
            logger.error(f"流程配置验证失败 {process_name}: {e}")
            return False

    def export_config_as_json(self, process_name: str, output_file: str) -> bool:
        """
        将流程配置导出为JSON格式

        Args:
            process_name: 流程名称
            output_file: 输出文件路径

        Returns:
            导出是否成功
        """
        try:
            config = self.load_process_config(process_name)
            config_dict = config.to_dict()

            with open(output_file, 'w', encoding='utf - 8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"流程配置导出成功: {output_file}")
            return True

        except Exception as e:
            logger.error(f"流程配置导出失败: {e}")
            return False

    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """
        验证配置数据

        Args:
            config_data: 配置数据

        Raises:
            ConfigValidationError: 验证失败
        """
        required_fields = [
            'process_name', 'version', 'description', 'workflow',
            'events', 'configuration', 'dependencies', 'compatibility'
        ]

        for field in required_fields:
            if field not in config_data:
                raise ConfigValidationError(f"缺少必需字段: {field}")

        # 验证版本格式
        version = config_data['version']
        if not self._is_valid_version(version):
            raise ConfigValidationError(f"无效的版本格式: {version}")

        # 验证工作流
        workflow = config_data['workflow']
        if not isinstance(workflow, dict):
            raise ConfigValidationError("工作流必须是字典格式")

        required_workflow_fields = ['name', 'description',
                                    'initial_state', 'final_states', 'states']
        for field in required_workflow_fields:
            if field not in workflow:
                raise ConfigValidationError(f"工作流缺少必需字段: {field}")

    def _validate_config_integrity(self, config: ProcessConfiguration) -> None:
        """
        验证配置完整性

        Args:
            config: 流程配置对象

        Raises:
            ConfigValidationError: 验证失败
        """
        # 验证状态转换的一致性
        states = self._parse_states(config.workflow)

        for state_name, state in states.items():
            for transition in state.transitions:
                # 检查目标状态是否存在
                if transition.to not in states:
                    raise ConfigValidationError(
                        f"状态 {state_name} 的转换目标 {transition.to} 不存在"
                    )

                # 检查事件是否已定义
                if transition.event not in config.events:
                    raise ConfigValidationError(
                        f"状态 {state_name} 的转换事件 {transition.event} 未定义"
                    )

        # 验证初始状态存在
        initial_state = config.workflow['initial_state']
        if initial_state not in states:
            raise ConfigValidationError(f"初始状态 {initial_state} 不存在")

        # 验证最终状态存在
        final_states = config.workflow['final_states']
        for final_state in final_states:
            if final_state not in states:
                raise ConfigValidationError(f"最终状态 {final_state} 不存在")

    def _parse_config(self, config_data: Dict[str, Any]) -> ProcessConfiguration:
        """
        解析配置数据

        Args:
            config_data: 配置数据

        Returns:
            流程配置对象
        """
        # 解析事件
        events = {}
        for event_name, event_data in config_data['events'].items():
            events[event_name] = EventSchema(
                description=event_data['description'],
                data_schema=event_data.get('data_schema', {})
            )

        # 创建流程配置对象
        return ProcessConfiguration(
            process_name=config_data['process_name'],
            version=config_data['version'],
            description=config_data['description'],
            workflow=config_data['workflow'],
            events=events,
            configuration=config_data['configuration'],
            dependencies=config_data['dependencies'],
            compatibility=config_data['compatibility']
        )

    def _parse_states(self, workflow: Dict[str, Any]) -> Dict[str, ProcessState]:
        """
        解析工作流状态

        Args:
            workflow: 工作流定义

        Returns:
            状态定义字典
        """
        states = {}

        for state_name, state_data in workflow['states'].items():
            # 解析转换
            transitions = []
            for trans_data in state_data.get('transitions', []):
                transition = StateTransition(
                    to=trans_data['to'],
                    condition=trans_data['condition'],
                    event=trans_data['event'],
                    description=trans_data.get('description')
                )
                transitions.append(transition)

            # 创建状态对象
            state = ProcessState(
                description=state_data['description'],
                actions=state_data.get('actions', []),
                transitions=transitions,
                final=state_data.get('final', False)
            )

            states[state_name] = state

        return states

    def _is_valid_version(self, version: str) -> bool:
        """
        验证版本格式

        Args:
            version: 版本字符串

        Returns:
            版本格式是否有效
        """
        try:
            # 简单的版本格式验证 (x.y.z)
            parts = version.split('.')
            if len(parts) != 3:
                return False

            for part in parts:
                int(part)

            return True
        except ValueError:
            return False

    def get_config_summary(self, process_name: str) -> Dict[str, Any]:
        """
        获取配置摘要信息

        Args:
            process_name: 流程名称

        Returns:
            配置摘要
        """
        try:
            config = self.load_process_config(process_name)
            states = self._parse_states(config.workflow)

            return {
                'process_name': config.process_name,
                'version': config.version,
                'description': config.description,
                'state_count': len(states),
                'event_count': len(config.events),
                'initial_state': config.workflow['initial_state'],
                'final_states': config.workflow['final_states'],
                'dependencies': config.dependencies,
                'compatibility': config.compatibility
            }
        except Exception as e:
            logger.error(f"获取配置摘要失败: {e}")
            return {}

    def list_available_processes(self) -> List[str]:
        """
        列出可用的流程配置

        Returns:
            流程名称列表
        """
        yaml_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]

    def clear_cache(self) -> None:
        """清除配置缓存"""
        self.loaded_configs.clear()
        self.config_cache.clear()
        logger.info("配置缓存已清除")

# 便捷函数


def load_process_config(process_name: str, config_dir: str = "config / processes") -> ProcessConfiguration:
    """
    便捷函数：加载流程配置

    Args:
        process_name: 流程名称
        config_dir: 配置目录

    Returns:
        流程配置对象
    """
    loader = ProcessConfigLoader(config_dir)
    return loader.load_process_config(process_name)


def validate_process_config(process_name: str, config_dir: str = "config / processes") -> bool:
    """
    便捷函数：验证流程配置

    Args:
        process_name: 流程名称
        config_dir: 配置目录

    Returns:
        验证是否通过
    """
    loader = ProcessConfigLoader(config_dir)
    return loader.validate_process_config(process_name)


if __name__ == "__main__":
    # 测试配置加载器
    loader = ProcessConfigLoader()

    # 列出可用流程
    processes = loader.list_available_processes()
    print(f"可用流程: {processes}")

    # 加载并验证配置
    for process_name in processes:
        try:
            config = loader.load_process_config(process_name)
            summary = loader.get_config_summary(process_name)
            print(f"\n流程: {process_name}")
            print(f"版本: {summary.get('version')}")
            print(f"状态数: {summary.get('state_count')}")
            print(f"事件数: {summary.get('event_count')}")
        except Exception as e:
            print(f"加载流程 {process_name} 失败: {e}")
