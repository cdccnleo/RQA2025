"""
FastAPI应用工厂
创建和配置FastAPI应用实例
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import app as main_app


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    
    返回:
        FastAPI: 配置好的应用实例
    """
    # 直接返回已配置好的应用实例
    return main_app
