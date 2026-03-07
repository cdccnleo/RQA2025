import json
import os
from datetime import datetime, timedelta

import pytest

import src.infrastructure.security.crypto.encryption_service as service_module
from src.infrastructure.security.crypto.encryption_service import EncryptionService


@pytest.fixture
def service(monkeypatch):
    svc = EncryptionService()
    monkeypatch.setattr(service_module, "_encryption_service", svc, raising=False)
    return svc


def test_encrypt_file_handles_io_errors(tmp_path, service, monkeypatch):
    # 读取失败
    assert service.encrypt_file("/nonexistent.txt", "unused.json", "file") is False

    input_path = tmp_path / "plain.bin"
    input_path.write_bytes(b"binary-data")
    monkeypatch.setattr(service.encryptor, "encrypt", lambda *args, **kwargs: None)
    assert service.encrypt_file(str(input_path), str(tmp_path / "cipher.json"), "file") is False

    def fake_encrypt(data, key_id="file"):
        return "{}"  # valid json

    monkeypatch.setattr(service.encryptor, "encrypt", fake_encrypt)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service_module.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("write fail")))
    assert service.encrypt_file(str(input_path), str(tmp_path / "cipher.json"), "file") is False


def test_decrypt_file_integrity_and_errors(tmp_path, service):
    cipher_path = tmp_path / "cipher.json"
    cipher_path.write_text(json.dumps({"metadata": {}, "data": "not encrypted"}), encoding="utf-8")
    assert service.decrypt_file(str(cipher_path), str(tmp_path / "out.bin"), "file") is False


def test_health_check_emits_warnings_and_critical(monkeypatch, service):
    base_health = service.health_check()
    assert base_health["status"] in {"healthy", "warning", "critical"}

    monkeypatch.setattr(service.key_manager, "list_keys", lambda: {})
    monkeypatch.setattr(service, "encrypt", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "decrypt", lambda *args, **kwargs: None)

    health = service.health_check()
    assert health["status"] == "critical"
    assert any("无可用密钥" in issue for issue in health["critical_issues"])


def test_health_check_warns_on_expired_rotation(monkeypatch, service):
    rotated_time = (datetime.now() - timedelta(days=120)).isoformat()
    monkeypatch.setattr(service.key_manager, "list_keys", lambda: {
        "old_key": {"rotated_at": rotated_time}
    })

    health = service.health_check()
    assert health["status"] == "warning"
    assert any("需要轮换" in warning for warning in health["warnings"])


def test_global_helpers_use_singleton(monkeypatch, service):
    encrypted = service_module.encrypt_data("payload")
    assert service_module.decrypt_data(encrypted) == "payload"

    token = service_module.create_secure_token({"user": "alice"}, expiration_minutes=1)
    assert service_module.verify_secure_token(token)["user"] == "alice"
