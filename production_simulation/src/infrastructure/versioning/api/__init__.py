"""
版本管理API模块

提供RESTful API接口来管理各种版本资源。
"""

from .version_api import VersionAPI, create_version_api

__all__ = [
    "VersionAPI",
    "create_version_api",
]
