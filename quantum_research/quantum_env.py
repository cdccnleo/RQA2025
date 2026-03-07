# -*- coding: utf-8 -*-
"""
RQA2026量子计算开发环境配置

此文件包含量子计算开发环境的基本配置和工具函数。
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class QuantumConfig:
    """量子计算环境配置类"""

    def __init__(self):
        # 项目路径配置
        self.project_root = Path(__file__).parent.parent
        self.quantum_dir = self.project_root / "quantum_research"

        # 量子计算提供商配置
        self.providers = {
            "ibm": {
                "token": os.getenv("IBM_QUANTUM_TOKEN", ""),
                "hub": os.getenv("IBM_QUANTUM_HUB", "ibm-q"),
                "group": os.getenv("IBM_QUANTUM_GROUP", "open"),
                "project": os.getenv("IBM_QUANTUM_PROJECT", "main")
            },
            "aws": {
                "region": os.getenv("AWS_REGION", "us-east-1"),
                "bucket": os.getenv("AWS_S3_BUCKET", "")
            },
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "")
            }
        }

        # 计算资源配置
        self.compute_resources = {
            "max_shots": int(os.getenv("QUANTUM_MAX_SHOTS", "1000")),
            "optimization_level": int(os.getenv("QUANTUM_OPT_LEVEL", "1")),
            "timeout": int(os.getenv("QUANTUM_TIMEOUT", "300"))
        }

        # 开发环境配置
        self.development = {
            "debug_mode": os.getenv("QUANTUM_DEBUG", "false").lower() == "true",
            "log_level": os.getenv("QUANTUM_LOG_LEVEL", "INFO"),
            "cache_dir": self.quantum_dir / "cache",
            "results_dir": self.quantum_dir / "results"
        }

        # 创建必要的目录
        self.development["cache_dir"].mkdir(exist_ok=True)
        self.development["results_dir"].mkdir(exist_ok=True)

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """获取指定提供商的配置"""
        return self.providers.get(provider_name, {})

    def is_provider_configured(self, provider_name: str) -> bool:
        """检查提供商是否已配置"""
        config = self.get_provider_config(provider_name)
        if provider_name == "ibm":
            return bool(config.get("token"))
        elif provider_name == "aws":
            return bool(config.get("region"))
        elif provider_name == "azure":
            return bool(config.get("subscription_id"))
        return False

    def get_compute_config(self) -> Dict[str, Any]:
        """获取计算资源配置"""
        return self.compute_resources

    def get_development_config(self) -> Dict[str, Any]:
        """获取开发环境配置"""
        return self.development


# 全局配置实例
quantum_config = QuantumConfig()


def setup_quantum_logging():
    """设置量子计算相关的日志配置"""
    import logging

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, quantum_config.development["log_level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(quantum_config.development["cache_dir"] / "quantum.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 创建量子计算专用日志器
    quantum_logger = logging.getLogger("quantum")
    quantum_logger.setLevel(logging.DEBUG if quantum_config.development["debug_mode"] else logging.INFO)

    return quantum_logger


def check_quantum_environment():
    """检查量子计算环境是否正确配置"""
    print("🔍 检查量子计算环境配置...")

    issues = []

    # 检查Qiskit
    try:
        import qiskit
        print(f"✅ Qiskit版本: {qiskit.__version__}")
    except ImportError:
        issues.append("❌ Qiskit未安装")

    # 检查提供商配置
    configured_providers = []
    for provider in ["ibm", "aws", "azure"]:
        if quantum_config.is_provider_configured(provider):
            configured_providers.append(provider)

    if configured_providers:
        print(f"✅ 已配置提供商: {', '.join(configured_providers)}")
    else:
        print("⚠️  警告: 未配置任何量子计算提供商")

    # 检查目录权限
    dirs_to_check = [
        quantum_config.development["cache_dir"],
        quantum_config.development["results_dir"]
    ]

    for directory in dirs_to_check:
        if directory.exists() and os.access(directory, os.W_OK):
            print(f"✅ 目录权限正常: {directory}")
        else:
            issues.append(f"❌ 目录权限问题: {directory}")

    if issues:
        print("
⚠️  发现以下问题:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("
🎉 量子计算环境配置检查通过!")
        return True


if __name__ == "__main__":
    # 初始化日志
    logger = setup_quantum_logging()
    logger.info("量子计算环境配置模块加载完成")

    # 运行环境检查
    check_quantum_environment()
