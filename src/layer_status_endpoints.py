"""
RQA2025 层级状态端点
为新创建的层级提供统一的状态查询接口
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from src.streaming import get_streaming_manager
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logger.warning("Streaming module not available")

try:
    from src.optimization import get_optimization_manager
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.warning("Optimization module not available")

try:
    from src.automation import get_automation_manager
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    logger.warning("Automation module not available")

try:
    from src.resilience import get_resilience_manager
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False
    logger.warning("Resilience module not available")

try:
    from src.utils import get_utils_manager
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logger.warning("Utils module not available")


def get_streaming_status() -> Dict[str, Any]:
    """获取流处理层状态"""
    try:
        if STREAMING_AVAILABLE:
            manager = get_streaming_manager()
            return manager.get_status()
        else:
            return {
                "layer_id": "streaming",
                "layer_name": "流处理层",
                "status": "unknown",
                "components": {},
                "metrics": {},
                "last_updated": datetime.now().isoformat(),
                "message": "Module not initialized"
            }
    except Exception as e:
        logger.error(f"Failed to get streaming status: {e}")
        return {
            "layer_id": "streaming",
            "layer_name": "流处理层",
            "status": "error",
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


def get_optimization_status() -> Dict[str, Any]:
    """获取优化层状态"""
    try:
        if OPTIMIZATION_AVAILABLE:
            manager = get_optimization_manager()
            return manager.get_status()
        else:
            return {
                "layer_id": "optimization",
                "layer_name": "优化层",
                "status": "unknown",
                "components": {},
                "metrics": {},
                "last_updated": datetime.now().isoformat(),
                "message": "Module not initialized"
            }
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        return {
            "layer_id": "optimization",
            "layer_name": "优化层",
            "status": "error",
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


def get_automation_status() -> Dict[str, Any]:
    """获取自动化层状态"""
    try:
        if AUTOMATION_AVAILABLE:
            manager = get_automation_manager()
            return manager.get_status()
        else:
            return {
                "layer_id": "automation",
                "layer_name": "自动化层",
                "status": "unknown",
                "components": {},
                "metrics": {},
                "last_updated": datetime.now().isoformat(),
                "message": "Module not initialized"
            }
    except Exception as e:
        logger.error(f"Failed to get automation status: {e}")
        return {
            "layer_id": "automation",
            "layer_name": "自动化层",
            "status": "error",
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


def get_resilience_status() -> Dict[str, Any]:
    """获取弹性层状态"""
    try:
        if RESILIENCE_AVAILABLE:
            manager = get_resilience_manager()
            return manager.get_status()
        else:
            return {
                "layer_id": "resilience",
                "layer_name": "弹性层",
                "status": "unknown",
                "components": {},
                "metrics": {},
                "last_updated": datetime.now().isoformat(),
                "message": "Module not initialized"
            }
    except Exception as e:
        logger.error(f"Failed to get resilience status: {e}")
        return {
            "layer_id": "resilience",
            "layer_name": "弹性层",
            "status": "error",
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


def get_utils_status() -> Dict[str, Any]:
    """获取工具层状态"""
    try:
        if UTILS_AVAILABLE:
            manager = get_utils_manager()
            return manager.get_status()
        else:
            return {
                "layer_id": "utils",
                "layer_name": "工具层",
                "status": "unknown",
                "components": {},
                "metrics": {},
                "last_updated": datetime.now().isoformat(),
                "message": "Module not initialized"
            }
    except Exception as e:
        logger.error(f"Failed to get utils status: {e}")
        return {
            "layer_id": "utils",
            "layer_name": "工具层",
            "status": "error",
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


def register_layer_status_routes(app):
    """注册层级状态路由到FastAPI应用"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/api/v1", tags=["layer-status"])
    
    @router.get("/streaming/status")
    async def get_streaming_status_endpoint():
        """获取流处理层状态"""
        return get_streaming_status()
    
    @router.get("/optimization/status")
    async def get_optimization_status_endpoint():
        """获取优化层状态"""
        return get_optimization_status()
    
    @router.get("/automation/status")
    async def get_automation_status_endpoint():
        """获取自动化层状态"""
        return get_automation_status()
    
    @router.get("/resilience/status")
    async def get_resilience_status_endpoint():
        """获取弹性层状态"""
        return get_resilience_status()
    
    @router.get("/utils/status")
    async def get_utils_status_endpoint():
        """获取工具层状态"""
        return get_utils_status()
    
    # 注册路由
    app.include_router(router)
    
    logger.info("Layer status routes registered successfully")
    
    return router
