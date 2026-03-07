#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制组件 - 配置管理器

负责访问控制相关配置的持久化和管理
"""

import logging
import json
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime
import hashlib


class ConfigManager:
    """
    配置管理器

    负责访问控制系统配置的加载、保存和管理
    """

    def __init__(self, config_path: Optional[Path] = None, enable_hot_reload: bool = True):
        """
        初始化配置管理器

        Args:
            config_path: 配置存储路径
            enable_hot_reload: 是否启用热更新
        """
        self.config_path = config_path or Path("data/security/access")
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.config_file = self._resolve_config_file(self.config_path)
        self.backup_dir = self.config_path / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # 默认配置
        self._config = self._get_default_config()
        self._load_config()

        # 热更新相关属性
        self.enable_hot_reload = enable_hot_reload
        self._config_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._config_hash = self._calculate_config_hash()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        if enable_hot_reload:
            self._start_hot_reload_monitor()

        logging.info(f"配置管理器初始化完成，热更新: {'启用' if enable_hot_reload else '禁用'}")

    def _resolve_config_file(self, base_path: Path) -> Path:
        """确定配置文件路径"""
        preferred = base_path / "access_config.json"
        if preferred.exists():
            return preferred

        candidates = sorted(base_path.glob("*.json"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return preferred

    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置

        Returns:
            默认配置字典
        """
        now = datetime.now().isoformat()

        return {
            "version": "1.0",
            "created_at": now,
            "updated_at": now,

            # 各模块默认配置
            "cache": self._get_default_cache_config(),
            "audit": self._get_default_audit_config(),
            "security": self._get_default_security_config(),
            "policies": self._get_default_policies_config(),
            "monitoring": self._get_default_monitoring_config()
        }

    def _get_default_cache_config(self) -> Dict[str, Any]:
        """
        获取默认缓存配置

        Returns:
            缓存配置字典
        """
        return {
            "enabled": True,
            "max_size": 1000,
            "ttl_seconds": 3600,
            "cleanup_interval": 300
        }

    def _get_default_audit_config(self) -> Dict[str, Any]:
        """
        获取默认审计配置

        Returns:
            审计配置字典
        """
        return {
            "enabled": True,
            "log_path": "data/security/audit",
            "max_log_files": 30,
            "async_writing": False,
            "log_level": "INFO"
        }

    def _get_default_security_config(self) -> Dict[str, Any]:
        """
        获取默认安全配置

        Returns:
            安全配置字典
        """
        return {
            "password_min_length": 8,
            "password_require_uppercase": True,
            "password_require_lowercase": True,
            "password_require_digits": True,
            "password_require_special": False,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "session_timeout_minutes": 60
        }

    def _get_default_policies_config(self) -> Dict[str, Any]:
        """
        获取默认策略配置

        Returns:
            策略配置字典
        """
        return {
            "default_deny": True,
            "policy_evaluation_order": "allow_override",
            "max_policies_per_resource": 10
        }

    def _get_default_monitoring_config(self) -> Dict[str, Any]:
        """
        获取默认监控配置

        Returns:
            监控配置字典
        """
        return {
            "enabled": True,
            "metrics_interval_seconds": 60,
            "alert_thresholds": {
                "failed_access_attempts": 10,
                "suspicious_activities": 5
            }
        }

    def _read_config_file(self) -> Optional[Dict[str, Any]]:
        """读取配置文件内容"""
        try:
            if not self.config_file.exists():
                return {}
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as exc:
            logging.error(f"读取配置文件失败: {exc}")
            return None

    def get_config(self, key: Optional[str] = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点分隔的嵌套键，如 "cache.enabled"

        Returns:
            配置值
        """
        if key is None:
            from copy import deepcopy
            return deepcopy(self._config)

        # 支持嵌套键访问
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None

    def set_config(self, key: str, value: Any) -> bool:
        """
        设置配置值

        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值

        Returns:
            是否设置成功
        """
        keys = key.split('.')
        config = self._config

        # 导航到父级字典
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value
        self._config["updated_at"] = datetime.now().isoformat()

        # 保存配置
        saved = self._save_config()
        if saved:
            self._notify_config_callbacks()
        return saved

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        批量更新配置

        Args:
            updates: 配置更新字典

        Returns:
            是否更新成功
        """
        def update_nested_dict(target: Dict[str, Any], source: Dict[str, Any]):
            """递归更新嵌套字典"""
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested_dict(target[key], value)
                else:
                    target[key] = value

        update_nested_dict(self._config, updates)
        self._config["updated_at"] = datetime.now().isoformat()

        saved = self._save_config()
        if saved:
            self._notify_config_callbacks()
        return saved

    def reset_config(self, section: Optional[str] = None) -> bool:
        """
        重置配置到默认值

        Args:
            section: 要重置的配置节，如果为None则重置所有配置

        Returns:
            是否重置成功
        """
        default_config = self._get_default_config()

        if section is None:
            self._config = default_config
        elif section in default_config:
            self._config[section] = default_config[section]
        else:
            logging.warning(f"未知的配置节: {section}")
            return False

        self._config["updated_at"] = datetime.now().isoformat()
        saved = self._save_config()
        if saved:
            self._notify_config_callbacks()
        return saved

    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        验证配置有效性

        Args:
            config: 可选的配置字典，如果提供则仅对该配置进行快速验证并返回布尔值
        """
        target_config = config or self._config

        def _is_positive_int(value: Any) -> bool:
            return isinstance(value, int) and value > 0

        # 提供兼容模式：当传入外部配置时返回布尔值
        if config is not None:
            cache_cfg = config.get("cache", {})
            if not _is_positive_int(cache_cfg.get("max_size", 0)):
                return False
            return True

        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        cache_config = target_config.get("cache", {})
        if not _is_positive_int(cache_config.get("max_size", 0)):
            result["errors"].append("cache.max_size 必须大于0")
            result["valid"] = False

        ttl_value = cache_config.get("ttl_seconds", 0)
        if isinstance(ttl_value, int) and ttl_value < 0:
            result["warnings"].append("cache.ttl_seconds 不应为负数")

        security_config = target_config.get("security", {})
        min_length = security_config.get("password_min_length", 0)
        if isinstance(min_length, int) and min_length < 4:
            result["warnings"].append("password_min_length 建议至少为4")

        if not _is_positive_int(security_config.get("max_login_attempts", 0)):
            result["errors"].append("max_login_attempts 必须大于0")
            result["valid"] = False

        audit_config = target_config.get("audit", {})
        if audit_config.get("enabled") and not audit_config.get("log_path"):
            result["warnings"].append("启用审计时建议设置log_path")

        return result

    def export_config(self, file_path: Path) -> bool:
        """
        导出配置到文件

        Args:
            file_path: 导出文件路径

        Returns:
            是否导出成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logging.info(f"配置导出成功: {file_path}")
            return True
        except Exception as e:
            logging.error(f"配置导出失败: {e}")
            return False

    def import_config(self, file_path: Path, merge: bool = True) -> bool:
        """
        从文件导入配置

        Args:
            file_path: 导入文件路径
            merge: 是否合并到现有配置，False则完全替换

        Returns:
            是否导入成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            if merge:
                self.update_config(imported_config)
            else:
                # 先备份当前配置
                self._create_backup()
                self._config = imported_config

            # 验证导入的配置
            validation = self.validate_config()
            if not validation["valid"]:
                logging.warning(f"导入的配置存在错误: {validation['errors']}")

            self._save_config()
            logging.info(f"配置导入成功: {file_path} (merge={merge})")
            return True

        except Exception as e:
            logging.error(f"配置导入失败: {e}")
            return False

    def _load_config(self) -> bool:
        """从文件加载配置"""
        if not self.config_file.exists():
            # 配置文件不存在，使用默认配置并保存
            self._config = self._get_default_config()
            self._save_config()
            return True

        loaded_config = self._read_config_file()
        if loaded_config is None:
            return False
        merged_config = self._get_default_config()
        if loaded_config:
            self._merge_configs(merged_config, loaded_config)
        self._config = merged_config
        logging.info("配置加载成功")
        return True

    def _save_config(self) -> bool:
        """
        保存配置到文件

        Returns:
            是否保存成功
        """
        # 先创建备份
        self._create_backup()

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logging.info("配置保存成功")
            return True
        except Exception as e:
            logging.error(f"配置保存失败: {e}")
            return False

    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """
        递归合并配置字典

        Args:
            base_config: 基础配置
            override_config: 覆盖配置
        """
        for key, value in override_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value

    def _create_backup(self):
        """创建配置备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"config_backup_{timestamp}.json"

            if self.config_file.exists():
                import shutil
                shutil.copy2(self.config_file, backup_file)

                # 清理旧备份，保留最新的10个
                backup_files = sorted(self.backup_dir.glob("config_backup_*.json"),
                                    key=lambda x: x.stat().st_mtime, reverse=True)
                if len(backup_files) > 10:
                    for old_file in backup_files[10:]:
                        old_file.unlink()

        except Exception as e:
            logging.warning(f"创建配置备份失败: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要信息

        Returns:
            配置摘要
        """
        from copy import deepcopy
        return {
            "version": self._config.get("version", "unknown"),
            "created_at": self._config.get("created_at"),
            "updated_at": self._config.get("updated_at"),
            "cache_enabled": self._config.get("cache", {}).get("enabled", False),
            "audit_enabled": self._config.get("audit", {}).get("enabled", False),
            "monitoring_enabled": self._config.get("monitoring", {}).get("enabled", False),
            "config_file_path": str(self.config_file),
            "backup_dir": str(self.backup_dir),
            "hot_reload_enabled": self.enable_hot_reload,
            "callback_count": len(self._config_callbacks)
        }

    # =========================================================================
    # 热更新功能
    # =========================================================================

    def _calculate_config_hash(self) -> str:
        """
        计算配置文件的哈希值

        Returns:
            配置文件的SHA256哈希值
        """
        try:
            if not self.config_file.exists():
                return ""

            with open(self.config_file, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logging.error(f"计算配置文件哈希失败: {e}")
            return ""

    def _start_hot_reload_monitor(self):
        """
        启动热更新监控线程
        """
        if not self.enable_hot_reload:
            return

        self._monitor_thread = threading.Thread(
            target=self._hot_reload_monitor_loop,
            daemon=True,
            name="ConfigHotReloadMonitor"
        )
        self._monitor_thread.start()
        logging.info("配置热更新监控已启动")

    def _hot_reload_monitor_loop(self):
        """
        热更新监控循环
        """
        while not self._stop_monitoring.is_set():
            try:
                # 检查配置文件是否有变化
                if self._check_config_file_changed():
                    logging.info("检测到配置文件变更，开始热更新...")
                    self._perform_hot_reload()

                # 每2秒检查一次
                time.sleep(2)

            except Exception as e:
                logging.error(f"配置热更新监控异常: {e}")
                time.sleep(5)  # 出错后等待更长时间

    def _check_config_file_changed(self) -> bool:
        """
        检查配置文件是否发生变化

        Returns:
            如果文件有变化返回True
        """
        current_hash = self._calculate_config_hash()
        if current_hash != self._config_hash:
            self._config_hash = current_hash
            return True
        return False

    def _perform_hot_reload(self):
        """
        执行热更新
        """
        try:
            # 重新加载配置
            from copy import deepcopy
            old_config = deepcopy(self._config)
            load_success = self._load_config()
            if not load_success:
                logging.error("热更新失败：无法加载新的配置文件")
                self._config = old_config
                self._config_hash = self._calculate_config_hash()
                return

            # 验证新配置
            if not self.validate_config()["valid"]:
                logging.error("热更新失败：新配置验证不通过，回滚到旧配置")
                self._config = old_config
                self._config_hash = self._calculate_config_hash()
                return

            # 通知所有回调函数
            self._notify_config_callbacks(old_config)
            self._config_hash = self._calculate_config_hash()

            logging.info("配置热更新成功完成")

        except Exception as e:
            logging.error(f"配置热更新执行失败: {e}")

    def add_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        添加配置变更回调函数

        Args:
            callback: 回调函数，参数为新的配置字典
        """
        if callback not in self._config_callbacks:
            self._config_callbacks.append(callback)
        logging.debug("配置变更回调函数已添加")

    def remove_config_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        移除配置变更回调函数

        Args:
            callback: 要移除的回调函数
        """
        if callback in self._config_callbacks:
            self._config_callbacks.remove(callback)
            logging.debug("配置变更回调函数已移除")
            return True
        return False

    def _notify_config_callbacks(self, old_config: Optional[Dict[str, Any]] = None):
        """
        通知所有配置变更回调函数

        Args:
            old_config: 旧配置
        """
        from copy import deepcopy
        for callback in self._config_callbacks:
            try:
                callback(deepcopy(self._config))
            except Exception as e:
                logging.error(f"配置变更回调执行失败: {e}")

    def trigger_manual_reload(self) -> bool:
        """
        手动触发配置重新加载

        Returns:
            重新加载是否成功
        """
        from copy import deepcopy
        old_config = deepcopy(self._config)
        try:
            raw_config = self._read_config_file()
            if raw_config is None:
                logging.error("手动配置重新加载失败：配置文件无法读取")
                self._config = old_config
                self._config_hash = self._calculate_config_hash()
                return False

            if not raw_config:
                logging.error("手动配置重新加载失败：配置文件为空或无法读取")
                self._config = old_config
                self._config_hash = self._calculate_config_hash()
                return False

            if not self.validate_config(raw_config):
                logging.error("手动配置重新加载失败：配置验证不通过")
                self._config = old_config
                self._config_hash = self._calculate_config_hash()
                return False

            merged_config = self._get_default_config()
            self._merge_configs(merged_config, raw_config)
            merged_config["updated_at"] = datetime.now().isoformat()

            self._config = merged_config
            self._config_hash = self._calculate_config_hash()
            self._notify_config_callbacks(old_config)
            logging.info("手动配置重新加载成功")
            return True

        except Exception as e:
            logging.error(f"手动配置重新加载失败: {e}")
            self._config = old_config  # 回滚到原始配置
            self._config_hash = self._calculate_config_hash()
            return False

    def shutdown(self):
        """
        关闭配置管理器
        """
        if self.enable_hot_reload:
            self._stop_monitoring.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)

        logging.info("配置管理器已关闭")
