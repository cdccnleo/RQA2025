#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义认证插件示例

展示如何创建安全插件来扩展认证功能
"""

import logging
from typing import Dict, Any, Optional
from .plugin_system import SecurityPlugin, PluginInfo


class CustomAuthPlugin(SecurityPlugin):
    """自定义认证插件"""

    def __init__(self):
        self.config = {}
        self.auth_attempts = {}
        self.blocked_users = set()
        self.shutdown_called = False

    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name="custom_auth",
            version="1.0.0",
            description="自定义认证插件，提供额外的认证检查",
            author="Security Team",
            dependencies=[],
            capabilities=[
                "pre_auth_check",
                "post_auth_check",
                "get_auth_stats",
                "block_user",
                "unblock_user"
            ],
            config_schema={
                "max_attempts": {
                    "type": "integer",
                    "default": 5,
                    "description": "最大认证尝试次数"
                },
                "block_duration": {
                    "type": "integer",
                    "default": 300,
                    "description": "用户阻塞时长（秒）"
                },
                "enable_logging": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否启用详细日志"
                }
            }
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self.config = {
                "max_attempts": config.get("max_attempts", 5),
                "block_duration": config.get("block_duration", 300),
                "enable_logging": config.get("enable_logging", True)
            }

            logging.info(f"自定义认证插件已初始化: {self.config}")
            return True

        except Exception as e:
            logging.error(f"自定义认证插件初始化失败: {e}")
            return False

    def shutdown(self) -> None:
        """关闭插件"""
        self.auth_attempts.clear()
        self.blocked_users.clear()
        self.shutdown_called = True
        logging.info("自定义认证插件已关闭")

    def pre_auth_check(self, username: str, **kwargs) -> Dict[str, Any]:
        """预认证检查"""
        result = {
            "allowed": True,
            "reason": "",
            "risk_score": 0.0
        }

        # 检查用户是否被阻塞
        if username in self.blocked_users:
            result["allowed"] = False
            result["reason"] = "用户已被临时阻塞"
            result["risk_score"] = 1.0

            if self.config["enable_logging"]:
                logging.warning(f"阻止用户 {username} 的认证尝试")
            return result

        # 检查认证尝试次数
        attempts = self.auth_attempts.get(username, 0)
        if attempts >= self.config["max_attempts"]:
            # 阻塞用户
            self.blocked_users.add(username)

            result["allowed"] = False
            result["reason"] = f"认证尝试次数过多 ({attempts})"
            result["risk_score"] = 0.8

            if self.config["enable_logging"]:
                logging.warning(f"因尝试次数过多阻塞用户 {username}")
            return result

        # 计算风险分数
        if attempts > 2:
            result["risk_score"] = 0.3 + (attempts - 2) * 0.1

        return result

    def post_auth_check(self, username: str, success: bool, **kwargs) -> None:
        """后认证检查"""
        if success:
            # 认证成功，重置尝试次数
            self.auth_attempts[username] = 0
            if self.config["enable_logging"]:
                logging.info(f"用户 {username} 认证成功")
        else:
            # 认证失败，增加尝试次数
            self.auth_attempts[username] = self.auth_attempts.get(username, 0) + 1
            attempts = self.auth_attempts[username]

            if self.config["enable_logging"]:
                logging.warning(f"用户 {username} 认证失败 (尝试次数: {attempts})")

            # 检查是否需要阻塞
            if attempts >= self.config["max_attempts"]:
                self.blocked_users.add(username)
                logging.error(f"用户 {username} 因多次失败被阻塞")

    def get_auth_stats(self) -> Dict[str, Any]:
        """获取认证统计信息"""
        return {
            "total_users_tracked": len(self.auth_attempts),
            "blocked_users": len(self.blocked_users),
            "blocked_users_list": list(self.blocked_users),
            "auth_attempts": dict(self.auth_attempts),
            "config": self.config
        }

    def block_user(self, username: str) -> bool:
        """手动阻塞用户"""
        self.blocked_users.add(username)
        logging.info(f"手动阻塞用户: {username}")
        return True

    def unblock_user(self, username: str) -> bool:
        """解除用户阻塞"""
        if username in self.blocked_users:
            self.blocked_users.remove(username)
            # 重置尝试次数
            self.auth_attempts[username] = 0
            logging.info(f"解除用户阻塞: {username}")
            return True
        return False


# 插件工厂函数
def create_plugin():
    """创建插件实例"""
    return CustomAuthPlugin()
