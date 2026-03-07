#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 安全配置管理器

负责安全模块的配置管理和持久化
分离了AccessControlManager的配置职责
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from copy import deepcopy
from src.infrastructure.security.core.types import ConfigOperationParams


class SecurityConfigManager:
    """
    安全配置管理器

    职责：管理安全模块的所有配置，包括用户、角色、策略等
    分离了配置相关的复杂逻辑，使主类更专注于业务逻辑
    """

    def __init__(self, config_path: str = "data/security/access_control"):
        """
        初始化配置管理器

        Args:
            config_path: 配置存储路径
        """
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

        # 配置验证规则
        self.validation_rules = {
            'users': self._validate_user_config,
            'roles': self._validate_role_config,
            'policies': self._validate_policy_config
        }

    def load_config(self, params: ConfigOperationParams) -> Dict[str, Any]:
        """
        加载配置

        Args:
            params: 配置操作参数

        Returns:
            加载的配置数据
        """
        try:
            sections = self._resolve_sections(params)
            config_data: Dict[str, Any] = {}

            if 'users' in sections:
                users_data = self._load_users_config()
                if users_data is not None:
                    config_data['users'] = users_data

            if 'roles' in sections:
                roles_data = self._load_roles_config()
                if roles_data is not None:
                    config_data['roles'] = roles_data

            if 'policies' in sections:
                policies_data = self._load_policies_config()
                if policies_data is not None:
                    config_data['policies'] = policies_data

            if params.validate_before_save and config_data:
                self._validate_config(config_data)

            if len(sections) == 1:
                section = next(iter(sections))
                return _SectionView(section, config_data.get(section, {}))

            return config_data

        except Exception as e:
            logging.error(f"加载配置失败: {e}")
            raise

    def save_config(self, config_data: Dict[str, Any], params: ConfigOperationParams) -> bool:
        """
        保存配置

        Args:
            config_data: 要保存的配置数据
            params: 配置操作参数

        Returns:
            是否保存成功
        """
        try:
            sections = self._resolve_sections(params)
            normalized_data = self._normalize_config_input(config_data, sections)

            if params.validate_before_save and normalized_data:
                self._validate_config(normalized_data)

            if params.create_backup:
                self._create_backup()

            if 'users' in normalized_data:
                self._save_users_config(normalized_data['users'])
            if 'roles' in normalized_data:
                self._save_roles_config(normalized_data['roles'])
            if 'policies' in normalized_data:
                self._save_policies_config(normalized_data['policies'])

            logging.info("配置保存成功")
            return True

        except Exception as e:
            logging.error(f"保存配置失败: {e}")
            raise

    def _load_users_config(self) -> Optional[Dict[str, Any]]:
        """加载用户配置"""
        users_file = self.config_path / "users.json"
        if not users_file.exists():
            return None

        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载用户配置失败: {e}")
            return None

    def _load_roles_config(self) -> Optional[Dict[str, Any]]:
        """加载角色配置"""
        roles_file = self.config_path / "roles.json"
        if not roles_file.exists():
            return None

        try:
            with open(roles_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载角色配置失败: {e}")
            return None

    def _load_policies_config(self) -> Optional[Dict[str, Any]]:
        """加载策略配置"""
        policies_file = self.config_path / "policies.json"
        if not policies_file.exists():
            return None

        try:
            with open(policies_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载策略配置失败: {e}")
            return None

    def _save_users_config(self, users_data: Dict[str, Any]) -> None:
        """保存用户配置"""
        users_file = self.config_path / "users.json"
        with open(users_file, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False)

    def _save_roles_config(self, roles_data: Dict[str, Any]) -> None:
        """保存角色配置"""
        roles_file = self.config_path / "roles.json"
        with open(roles_file, 'w', encoding='utf-8') as f:
            json.dump(roles_data, f, indent=2, ensure_ascii=False)

    def _save_policies_config(self, policies_data: Dict[str, Any]) -> None:
        """保存策略配置"""
        policies_file = self.config_path / "policies.json"
        with open(policies_file, 'w', encoding='utf-8') as f:
            json.dump(policies_data, f, indent=2, ensure_ascii=False)

    def _create_backup(self) -> None:
        """创建配置备份"""
        import shutil
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config_path / "backups"
        backup_dir.mkdir(exist_ok=True)

        for config_file in self.config_path.glob("*.json"):
            if config_file.is_file():
                backup_file = backup_dir / f"{config_file.name}.{timestamp}.backup"
                shutil.copy2(config_file, backup_file)

    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """验证配置数据"""
        for section, data in config_data.items():
            if section in self.validation_rules:
                self.validation_rules[section](data)

    def _validate_user_config(self, users_data: Dict[str, Any]) -> None:
        """验证用户配置"""
        if not isinstance(users_data, dict):
            raise ValueError("用户配置必须是字典")
        for user_id, user in users_data.items():
            if not isinstance(user, dict):
                raise ValueError(f"用户 {user_id} 配置必须是字典")
            if "roles" in user and not isinstance(user["roles"], list):
                raise ValueError(f"用户 {user_id} 的 roles 必须为列表")

    def _validate_role_config(self, roles_data: Dict[str, Any]) -> None:
        """验证角色配置"""
        if not isinstance(roles_data, dict):
            raise ValueError("角色配置必须是字典")
        for role_id, role in roles_data.items():
            if not isinstance(role, dict):
                raise ValueError(f"角色 {role_id} 配置必须是字典")
            if "permissions" in role and not isinstance(role["permissions"], list):
                raise ValueError(f"角色 {role_id} 的 permissions 必须为列表")

    def _validate_policy_config(self, policies_data: Dict[str, Any]) -> None:
        """验证策略配置"""
        if not isinstance(policies_data, dict):
            raise ValueError("策略配置必须是字典")
        for policy_id, policy in policies_data.items():
            if not isinstance(policy, dict):
                raise ValueError(f"策略 {policy_id} 配置必须是字典")
            if "conditions" in policy and not isinstance(policy["conditions"], list):
                raise ValueError(f"策略 {policy_id} 的 conditions 必须为列表")

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #

    def _resolve_sections(self, params: ConfigOperationParams) -> Set[str]:
        sections = set(params.config_sections or [])
        config_type = getattr(params, "config_type", None)
        if config_type:
            sections = {config_type}
        if not sections:
            return {"users", "roles", "policies"}
        return sections

    def _normalize_config_input(self, config_data: Dict[str, Any], sections: Set[str]) -> Dict[str, Any]:
        if not isinstance(config_data, dict):
            raise ValueError("配置数据必须是字典")

        normalized: Dict[str, Any] = {}
        known_sections = {"users", "roles", "policies"}
        intersect = set(config_data.keys()) & known_sections

        if intersect:
            for section in intersect:
                if section in sections:
                    normalized[section] = deepcopy(config_data[section])
        elif len(sections) == 1:
            section = next(iter(sections))
            normalized[section] = deepcopy(config_data)
        else:
            for section in sections:
                normalized[section] = deepcopy(config_data.get(section, {}))

        return normalized


class AuditConfigManager:
    """
    审计配置管理器

    专门负责审计日志的配置管理
    """

    def __init__(self, config_path: str = "data/security/audit"):
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

    def load_audit_rules(self) -> Dict[str, Any]:
        """加载审计规则"""
        rules_file = self.config_path / "audit_rules.json"
        if not rules_file.exists():
            return self._get_default_rules()

        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载审计规则失败: {e}")
            return self._get_default_rules()

    def save_audit_rules(self, rules: Dict[str, Any]) -> None:
        """保存审计规则"""
        rules_file = self.config_path / "audit_rules.json"
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        return True

    def _get_default_rules(self) -> Dict[str, Any]:
        """获取默认审计规则"""
        return {
            'enabled': True,
            'log_level': 'INFO',
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'retention_days': 90,
            'compression_enabled': True,
            'encryption_enabled': False,
            'version': '1.0',
            'rules': [
                {
                    'event_type': 'security',
                    'severity': 'high',
                    'action': 'block_and_log'
                },
                {
                    'event_type': 'access',
                    'severity': 'medium',
                    'action': 'log_only'
                }
            ]
        }


class _SectionView(dict):
    """单个配置节的视图，既支持扁平访问又支持通过节名访问"""

    def __init__(self, section_name: str, data: Optional[Dict[str, Any]]):
        super().__init__(deepcopy(data or {}))
        self._section_name = section_name

    def __contains__(self, key: object) -> bool:
        return key == self._section_name or super().__contains__(key)

    def __getitem__(self, key: str) -> Any:
        if key == self._section_name:
            return self
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == self._section_name:
            return self
        return super().get(key, default)

