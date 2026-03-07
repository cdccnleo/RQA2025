"""
模型管理API路由
提供模型查询、加载、删除等接口
符合架构设计：API网关层
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", tags=["models"])
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = 'active',
    min_accuracy: Optional[float] = None,
    limit: int = 100
):
    """列出所有模型
    
    Args:
        model_type: 模型类型筛选
        status: 状态筛选，'all'表示所有状态，None表示所有状态
        min_accuracy: 最小准确率筛选
        limit: 返回数量限制
    """
    try:
        from .model_persistence_service import get_model_persistence_service

        service = get_model_persistence_service()
        
        # 处理 'all' 或空值，表示查询所有状态
        if status == 'all' or status is None:
            status = None  # None 表示不筛选状态
        
        models = service.list_available_models(
            model_type=model_type,
            status=status,
            min_accuracy=min_accuracy,
            limit=limit
        )

        return {
            "success": True,
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出模型失败: {str(e)}")


@router.get("/models/{model_id}", tags=["models"])
async def get_model_details(model_id: str):
    """获取模型详情"""
    try:
        from .model_persistence_service import get_model_persistence_service

        service = get_model_persistence_service()
        metadata = service.get_model_metadata(model_id)

        if not metadata:
            raise HTTPException(status_code=404, detail="模型未找到")

        return {
            "success": True,
            "model": metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")


@router.post("/models/{model_id}/load", tags=["models"])
async def load_model(model_id: str):
    """加载模型"""
    try:
        from .model_persistence_service import get_model_persistence_service

        service = get_model_persistence_service()
        model = service.load_model(model_id)

        if not model:
            raise HTTPException(status_code=404, detail="模型加载失败")

        return {
            "success": True,
            "message": "模型加载成功",
            "model_id": model_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@router.delete("/models/{model_id}", tags=["models"])
async def delete_model(model_id: str, permanent: bool = False):
    """删除模型
    
    如果模型状态为 'deleted' 或 permanent=True，则彻底删除（硬删除）
    否则执行软删除（将状态更新为 'deleted'）
    
    Args:
        model_id: 模型ID
        permanent: 是否强制彻底删除
    """
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection
        import shutil
        from pathlib import Path

        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="数据库连接失败")

        cursor = conn.cursor()

        # 查询模型当前状态
        cursor.execute(
            "SELECT status FROM trained_models WHERE model_id = %s",
            (model_id,)
        )
        row = cursor.fetchone()

        if not row:
            cursor.close()
            return_db_connection(conn)
            raise HTTPException(status_code=404, detail="模型不存在")

        current_status = row[0]

        # 判断是否需要彻底删除
        if current_status == 'deleted' or permanent:
            # 彻底删除：从数据库删除记录
            cursor.execute(
                "DELETE FROM trained_models WHERE model_id = %s",
                (model_id,)
            )
            conn.commit()
            cursor.close()
            return_db_connection(conn)

            # 删除文件系统中的模型文件
            models_dir = Path("/app/models") / model_id
            if models_dir.exists():
                try:
                    shutil.rmtree(models_dir)
                    logger.info(f"模型文件已彻底删除: {models_dir}")
                except Exception as e:
                    logger.warning(f"删除模型文件失败 {model_id}: {e}")

            logger.info(f"模型已彻底删除: {model_id}")
            return {
                "success": True,
                "message": "模型已彻底删除",
                "permanent": True
            }
        else:
            # 软删除：更新状态为 'deleted'
            cursor.execute(
                "UPDATE trained_models SET status = 'deleted', updated_at = NOW() WHERE model_id = %s",
                (model_id,)
            )
            conn.commit()
            cursor.close()
            return_db_connection(conn)

            logger.info(f"模型已软删除: {model_id}")
            return {
                "success": True,
                "message": "模型已删除（可在已删除列表中查看）",
                "permanent": False
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.post("/models/{model_id}/deploy", tags=["models"])
async def deploy_model(model_id: str):
    """部署模型"""
    try:
        from .postgresql_persistence import get_db_connection, return_db_connection

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE trained_models SET is_deployed = TRUE WHERE model_id = %s",
                (model_id,)
            )
            conn.commit()
            cursor.close()
            return_db_connection(conn)

            logger.info(f"模型已部署: {model_id}")

        return {
            "success": True,
            "message": "模型已部署"
        }

    except Exception as e:
        logger.error(f"部署模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署模型失败: {str(e)}")
