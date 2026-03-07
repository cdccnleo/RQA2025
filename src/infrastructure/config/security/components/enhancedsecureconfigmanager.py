
from typing import Dict, Any, Optional, List
import threading
from pathlib import Path
import json
import logging
from .accessrecord import AccessRecord
from .configaccesscontrol import ConfigAccessControl
from .configauditlog import ConfigAuditLog
from .configauditmanager import ConfigAuditManager
from .configencryptionmanager import ConfigEncryptionManager
from .hotreloadmanager import HotReloadManager
from .securityconfig import SecurityConfig

"""安全配置相关类"""
logger = logging.getLogger(__name__)


class EnhancedSecureConfigManager:
    """增强版安全配置管理器"""

    def __init__(self, config_dir: str = "config", security_config: Optional[SecurityConfig] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.security_config = security_config or SecurityConfig()

        # 初始化组件
        self.encryption = ConfigEncryptionManager()
        self.access_control = ConfigAccessControl(self.security_config)
        self.audit = ConfigAuditManager()
        self.hot_reload = HotReloadManager()

        # 配置缓存
        self._config_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()

        # 敏感配置项
        self.sensitive_keys = {
            "password", "secret", "key", "token", "api_key",
            "database_password", "redis_password", "email_password"
        }

    def load_config(self, config_file: str, user: str = "system") -> Dict[str, Any]:
        """加载配置文件"""
        if not self.access_control.check_access(user, "read", config_file):
            raise PermissionError(f"用户 {user} 无权读取配置 {config_file}")

        file_path = self.config_dir / config_file

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试解密
            if self._is_encrypted(content):
                content = self.encryption.decrypt(content, f"file:{config_file}")

            config = json.loads(content)

            # 缓存配置
            with self._cache_lock:
                self._config_cache[config_file] = config

            # 设置热重载监视
            self.hot_reload.watch_file(str(file_path), lambda p: self._on_config_changed(p, user))

            self.audit.log_change("load", config_file, user=user)
            return config

        except Exception as e:
            logger.error(f"加载配置文件失败 {config_file}: {e}")
            raise

    def save_config(self, config: Dict[str, Any], config_file: str,
                    user: str = "system", reason: str = ""):
        """保存配置文件"""
        # 兼容调用顺序：保存时可能传入 (config_file, config)
        if isinstance(config, str) and isinstance(config_file, dict):
            config, config_file = config_file, config

        if not self.access_control.check_access(user, "write", config_file):
            raise PermissionError(f"用户 {user} 无权写入配置 {config_file}")

        file_path = self.config_dir / config_file

        # 获取旧配置用于审计
        old_config = None
        if file_path.exists():
            try:
                old_config = self.load_config(config_file, "system")
            except Exception as e:
                pass

        try:
            # 处理敏感信息
            processed_config = self._process_sensitive_data(config)

            # 转换为JSON
            content = json.dumps(processed_config, indent=2, ensure_ascii=False)

            # 加密敏感配置
            if self.security_config.encryption_enabled:
                content = self.encryption.encrypt(content, f"file:{config_file}")

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 更新缓存
            with self._cache_lock:
                self._config_cache[config_file] = processed_config

            # 审计日志
            self.audit.log_change("save", config_file,
                                  old_value=old_config,
                                  new_value=processed_config,
                                  user=user,
                                  reason=reason)

        except Exception as e:
            logger.error(f"保存配置文件失败 {config_file}: {e}")
            raise

    def get_value(self, config_file: str, key: str, default: Any = None,
                  user: str = "system") -> Any:
        """获取配置值"""
        if not self.access_control.check_access(user, "read", f"{config_file}.{key}"):
            raise PermissionError(f"用户 {user} 无权读取配置 {config_file}.{key}")

        with self._cache_lock:
            if config_file in self._config_cache:
                config = self._config_cache[config_file]
            else:
                config = self.load_config(config_file, user)

        value = self._get_nested_value(config, key.split('.'))
        return value if value is not None else default

    def set_value(self, *args, user: str = "system", reason: str = ""):
        """设置配置值"""
        if len(args) == 4:
            config, config_file, key, value = args
        elif len(args) == 3:
            config = None
            config_file, key, value = args
        else:
            raise TypeError("set_value() 接受 (config, config_file, key, value) 或 (config_file, key, value) 两种调用方式")

        if not self.access_control.check_access(user, "write", f"{config_file}.{key}"):
            raise PermissionError(f"用户 {user} 无权写入配置 {config_file}.{key}")

        with self._cache_lock:
            if config is None:
                if config_file in self._config_cache:
                    working_config = json.loads(json.dumps(self._config_cache[config_file]))
                else:
                    working_config = self.load_config(config_file, user)
            else:
                working_config = json.loads(json.dumps(config))

        self._set_nested_value(working_config, key.split('.'), value)

        # 保存配置
        self.save_config(working_config, config_file, user, reason)

    def enable_hot_reload(self):
        """启用热重载"""
        self.hot_reload.start_monitoring()

    def disable_hot_reload(self):
        """禁用热重载"""
        self.hot_reload.stop_monitoring()

    def get_audit_logs(self, **filters) -> List[ConfigAuditLog]:
        """获取审计日志"""
        return self.audit.get_audit_logs(**filters)

    def get_access_logs(self, **filters) -> List[AccessRecord]:
        """获取访问日志"""
        return self.access_control.get_access_logs(**filters)

    def _is_encrypted(self, content: str) -> bool:
        """检查内容是否已加密"""
        try:
            # 尝试解析为JSON，如果失败则可能是加密内容
            json.loads(content)
            return False
        except Exception as e:
            return True

    def _process_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理敏感数据"""
        processed = config.copy()

        def process_dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    process_dict(v)
                elif isinstance(v, str) and any(sensitive in k.lower() for sensitive in self.sensitive_keys):
                    # 这里可以添加额外的加密处理
                    pass

        process_dict(processed)
        return processed

    def _get_nested_value(self, config: Dict[str, Any], keys: List[str]) -> Any:
        """获取嵌套配置值"""
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _set_nested_value(self, config: Dict[str, Any], keys: List[str], value: Any):
        """设置嵌套配置值"""
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _on_config_changed(self, file_path: str, user: str):
        """配置文件变更回调"""
        config_file = Path(file_path).name

        # 重新加载配置
        try:
            new_config = self.load_config(config_file, user)
            logger.info(f"配置文件热重载成功: {config_file}")

            # 这里可以添加配置变更通知机制
            self._notify_config_change(config_file, new_config)

        except Exception as e:
            logger.error(f"配置文件热重载失败 {config_file}: {e}")

    def _notify_config_change(self, config_file: str, new_config: Dict[str, Any]):
        """通知配置变更"""
        # 这里可以集成配置变更通知机制
        logger.info(f"配置已更新: {config_file}")

# 全局实例




