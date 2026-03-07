from types import SimpleNamespace, ModuleType
import sys

# 为缺失的服务模块注入桩实现
security_module = ModuleType("src.infrastructure.security.services.security")
class StubSecurityService:
    pass
security_module.SecurityService = StubSecurityService
sys.modules.setdefault("src.infrastructure.security.services.security", security_module)

config_enc_module = ModuleType("src.infrastructure.security.services.config_encryption_service")
class StubConfigEncryptionService:
    def encrypt_config(self, config):
        return {"encrypted": config}

    def decrypt_config(self, config):
        return {"decrypted": config}
config_enc_module.ConfigEncryptionService = StubConfigEncryptionService
sys.modules.setdefault("src.infrastructure.security.services.config_encryption_service", config_enc_module)

config_sync_module = ModuleType("src.infrastructure.security.services.config_sync_service")
class StubConfigSyncService:
    def get_sync_status(self):
        return {"nodes": []}

    def sync_config(self, config, nodes=None):
        return {"success": True}

    def get_sync_history(self, limit=20):
        return []

    def get_conflicts(self):
        return []

    def resolve_conflicts(self, conflicts, strategy="merge"):
        return {"success": True}
config_sync_module.ConfigSyncService = StubConfigSyncService
sys.modules.setdefault("src.infrastructure.security.services.config_sync_service", config_sync_module)

web_auth_module = ModuleType("src.infrastructure.security.services.web_auth_manager")
class StubWebAuthManager:
    def authenticate_user(self, *args, **kwargs):
        return None

    def check_permission(self, *args, **kwargs):
        return False

    def create_session(self, username):
        return "session"

    def validate_session(self, session_id):
        return None

    def invalidate_session(self, session_id):
        return False

    def get_user_info(self, username):
        return None

    def add_user(self, *args, **kwargs):
        return False

    def update_user(self, *args, **kwargs):
        return False

    def delete_user(self, *args, **kwargs):
        return False

    def list_users(self):
        return []

    def list_sessions(self):
        return []

    def get_permissions(self):
        return {}

    def cleanup_expired_sessions(self):
        return 0
web_auth_module.WebAuthManager = StubWebAuthManager
sys.modules.setdefault("src.infrastructure.security.services.web_auth_manager", web_auth_module)

import pytest

import src.infrastructure.security.services.web_management_service as web_module
from src.infrastructure.security.services.web_management_service import WebManagementService, WebConfig


def test_lazy_initialization_creates_services(monkeypatch):
    fake_security = SimpleNamespace()
    fake_encryption = SimpleNamespace()
    fake_sync = SimpleNamespace()

    monkeypatch.setattr(web_module, "SecurityService", lambda: fake_security)
    monkeypatch.setattr(web_module, "ConfigEncryptionService", lambda: fake_encryption)
    monkeypatch.setattr(web_module, "ConfigSyncService", lambda: fake_sync)

    service = WebManagementService()

    assert service._get_security_service() is fake_security
    assert service._get_encryption_service() is fake_encryption
    assert service._get_sync_service() is fake_sync


def test_encrypt_decrypt_config_on_failure_returns_original(monkeypatch):
    service = WebManagementService()
    failing_encryptor = SimpleNamespace(
        encrypt_config=lambda config: (_ for _ in ()).throw(RuntimeError("encrypt fail")),
        decrypt_config=lambda config: (_ for _ in ()).throw(RuntimeError("decrypt fail")),
    )
    service._encryption_service = failing_encryptor

    original = {"secret": "value"}
    assert service.encrypt_sensitive_config(original) is original
    assert service.decrypt_config(original) is original


def test_get_dashboard_data_handles_exception(monkeypatch):
    service = WebManagementService()
    def raise_sync():
        raise RuntimeError("sync down")

    monkeypatch.setattr(service, "_get_sync_service", raise_sync)

    result = service.get_dashboard_data()
    assert "error" in result and "sync down" in result["error"]


def test_sync_operations_failure(monkeypatch):
    service = WebManagementService()
    failing_sync = SimpleNamespace(
        get_sync_status=lambda: (_ for _ in ()).throw(RuntimeError("status fail")),
        sync_config=lambda config, nodes: (_ for _ in ()).throw(RuntimeError("sync fail")),
        get_sync_history=lambda limit: (_ for _ in ()).throw(RuntimeError("history fail")),
        get_conflicts=lambda: (_ for _ in ()).throw(RuntimeError("conflict fail")),
        resolve_conflicts=lambda conflicts, strategy: (_ for _ in ()).throw(RuntimeError("resolve fail")),
    )
    monkeypatch.setattr(service, "_get_sync_service", lambda: failing_sync)

    assert service.get_sync_nodes() == []
    assert service.get_sync_history() == []
    assert service.get_conflicts() == []

    sync_result = service.sync_config_to_nodes({}, None)
    assert sync_result == {"success": False, "error": "sync fail"}

    resolve_result = service.resolve_conflicts([])
    assert resolve_result == {"success": False, "error": "resolve fail"}


def test_get_security_defaults():
    config = WebConfig()
    service = WebManagementService(web_config=config)
    assert isinstance(service._config, WebConfig)
