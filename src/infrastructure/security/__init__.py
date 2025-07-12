"""Security module for configuration management system.

This module provides:
- Core security services (authentication, authorization, encryption)
- Security configuration management
- Default security service implementation
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .security import SecurityManager, SecurityService

from .security import (
    SecurityConfig,
    KeyManager,
    EncryptionService,
    DataSanitizer,
    SecurityManager,
    SecurityService
)

def get_default_security_service() -> 'SecurityService':
    """Get the default singleton security service instance.

    Returns:
        SecurityService: The default security service instance
    """
    return SecurityService()

__all__ = [
    'SecurityConfig',
    'KeyManager',
    'EncryptionService',
    'DataSanitizer',
    'SecurityManager',
    'SecurityService',
    'get_default_security_service'
]
