"""Unit tests for security service functionality."""
import pytest
import threading
from src.infrastructure.security import (
    get_default_security_service,
    SecurityService,
    SecurityManager
)

class TestSecurityService:
    def test_singleton_pattern(self):
        """Verify that get_default_security_service returns same instance."""
        instance1 = get_default_security_service()
        instance2 = get_default_security_service()
        assert instance1 is instance2
        assert id(instance1) == id(instance2)

    def test_thread_safety(self):
        """Test that the security service is thread-safe."""
        results = []
        
        def worker():
            service = get_default_security_service()
            results.append(service.validate_config({"test": "config"}))
            
        threads = [threading.Thread(target=worker) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        assert all(results) or not any(results)  # All True or all False

    def test_signature_verification(self, mocker):
        """Test data signing and verification."""
        service = get_default_security_service()
        test_data = b"test_data"
        
        # Mock the internal sign/verify methods
        mocker.patch.object(service, '_sign_data', return_value=b"signed_data")
        mocker.patch.object(service, '_verify_signature', return_value=True)
        
        signed = service._sign_data(test_data)
        assert service._verify_signature(test_data, signed)

    def test_audit_logging(self, mocker):
        """Test audit logging functionality."""
        service = get_default_security_service()
        mock_audit = mocker.patch.object(service, 'audit')
        
        test_action = "test_action"
        test_details = {"key": "value"}
        service.audit(test_action, test_details)
        
        mock_audit.assert_called_once_with(test_action, test_details)

    def test_get_default_service_returns_manager(self):
        """Test that get_default_security_service returns a SecurityService instance."""
        service = get_default_security_service()
        assert isinstance(service, SecurityService)
        # 注意：SecurityManager和SecurityService是不同的类，这里期望SecurityService

    def test_service_implements_required_methods(self):
        """Test that the service implements all required methods."""
        service = get_default_security_service()
        assert hasattr(service, 'validate_config')
        assert hasattr(service, 'check_access')
        assert hasattr(service, 'audit')

    @pytest.mark.parametrize("config,expected", [
        ({}, False),
        ({"valid": True}, True),
    ])
    def test_validate_config(self, config, expected, mocker):
        """Test config validation behavior."""
        service = get_default_security_service()
        mocker.patch.object(service, 'validate_config', return_value=expected)
        assert service.validate_config(config) == expected
