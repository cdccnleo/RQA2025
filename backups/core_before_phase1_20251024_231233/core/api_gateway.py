#!/usr/bin/env python3
"""
API网关别名文件

提供对services.api_gateway模块的别名导入
"""

# 导入services模块中的ApiGateway
from .services.api_gateway import ApiGateway as APIGateway, LoadBalancer as GatewayRouter

__all__ = ['APIGateway', 'GatewayRouter']
