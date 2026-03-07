"""
策略版本管理路由模块
提供策略版本管理API端点
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# 导入版本管理器
from .strategy_version_manager import (
    version_manager,
    create_strategy_version,
    get_strategy_version,
    list_strategy_versions,
    compare_strategy_versions,
    rollback_strategy_version
)


@router.post("/api/v1/strategy/{strategy_id}/version/create")
async def create_version_api(strategy_id: str, request: Dict[str, Any]):
    """创建策略版本"""
    try:
        import os
        import json
        
        # 加载当前策略数据
        strategy_path = os.path.join("/app/data/strategies", f"{strategy_id}.json")
        if not os.path.exists(strategy_path):
            raise HTTPException(status_code=404, detail="策略不存在")
        
        with open(strategy_path, 'r', encoding='utf-8') as f:
            strategy_data = json.load(f)
        
        comment = request.get("comment", "")
        created_by = request.get("created_by", "user")
        tags = request.get("tags", [])
        
        version = create_strategy_version(
            strategy_id=strategy_id,
            strategy_data=strategy_data,
            comment=comment,
            created_by=created_by
        )
        
        # 添加标签
        if tags:
            version_manager.tag_version(strategy_id, version.version_id, tags)
        
        return {
            "success": True,
            "version_id": version.version_id,
            "version_number": version.version_number,
            "created_at": version.created_at,
            "message": "版本创建成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/versions")
async def list_versions_api(
    strategy_id: str,
    include_inactive: bool = Query(False, description="包含已删除版本")
):
    """列出策略版本"""
    try:
        versions = version_manager.list_versions(strategy_id, include_inactive)
        
        return {
            "strategy_id": strategy_id,
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "created_at": v.created_at,
                    "created_by": v.created_by,
                    "comment": v.comment,
                    "tags": v.tags,
                    "is_active": v.is_active
                }
                for v in versions
            ],
            "total": len(versions)
        }
        
    except Exception as e:
        logger.error(f"列出策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/version/{version_id}")
async def get_version_api(strategy_id: str, version_id: str):
    """获取策略版本详情"""
    try:
        version = get_strategy_version(strategy_id, version_id)
        
        if not version:
            raise HTTPException(status_code=404, detail="版本不存在")
        
        return {
            "version_id": version.version_id,
            "version_number": version.version_number,
            "created_at": version.created_at,
            "created_by": version.created_by,
            "comment": version.comment,
            "tags": version.tags,
            "is_active": version.is_active,
            "change_summary": version.change_summary,
            "strategy_data": version.strategy_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/version/compare")
async def compare_versions_api(strategy_id: str, request: Dict[str, Any]):
    """对比两个版本"""
    try:
        version_id1 = request.get("version_id1")
        version_id2 = request.get("version_id2")
        
        if not version_id1 or not version_id2:
            raise HTTPException(status_code=400, detail="需要提供两个版本ID")
        
        result = compare_strategy_versions(strategy_id, version_id1, version_id2)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对比策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"对比失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/version/rollback")
async def rollback_version_api(strategy_id: str, request: Dict[str, Any]):
    """回滚到指定版本"""
    try:
        version_id = request.get("version_id")
        comment = request.get("comment", "")
        
        if not version_id:
            raise HTTPException(status_code=400, detail="需要提供版本ID")
        
        new_version = rollback_strategy_version(strategy_id, version_id, comment)
        
        if not new_version:
            raise HTTPException(status_code=400, detail="回滚失败")
        
        # 更新策略文件
        import os
        import json
        strategy_path = os.path.join("/app/data/strategies", f"{strategy_id}.json")
        if os.path.exists(strategy_path):
            with open(strategy_path, 'w', encoding='utf-8') as f:
                json.dump(new_version.strategy_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": "回滚成功",
            "new_version_id": new_version.version_id,
            "new_version_number": new_version.version_number,
            "rolled_back_to": version_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"回滚策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"回滚失败: {str(e)}")


@router.delete("/api/v1/strategy/{strategy_id}/version/{version_id}")
async def delete_version_api(
    strategy_id: str, 
    version_id: str,
    hard_delete: bool = Query(False, description="是否硬删除")
):
    """删除版本"""
    try:
        success = version_manager.delete_version(strategy_id, version_id, hard_delete)
        
        if not success:
            raise HTTPException(status_code=404, detail="版本不存在或删除失败")
        
        return {
            "success": True,
            "message": "版本已删除" if hard_delete else "版本已标记为删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除策略版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/version/statistics")
async def get_version_statistics_api(strategy_id: str):
    """获取版本统计信息"""
    try:
        stats = version_manager.get_version_statistics(strategy_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取版本统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/version/{version_id}/tag")
async def tag_version_api(strategy_id: str, version_id: str, request: Dict[str, Any]):
    """为版本添加标签"""
    try:
        tags = request.get("tags", [])
        
        if not tags:
            raise HTTPException(status_code=400, detail="需要提供标签")
        
        success = version_manager.tag_version(strategy_id, version_id, tags)
        
        if not success:
            raise HTTPException(status_code=404, detail="版本不存在")
        
        return {
            "success": True,
            "message": "标签添加成功",
            "tags": tags
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加版本标签失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/version/search")
async def search_versions_api(strategy_id: str, request: Dict[str, Any]):
    """搜索版本"""
    try:
        keyword = request.get("keyword")
        tags = request.get("tags")
        created_by = request.get("created_by")
        start_time = request.get("start_time")
        end_time = request.get("end_time")
        
        results = version_manager.search_versions(
            strategy_id=strategy_id,
            keyword=keyword,
            tags=tags,
            created_by=created_by,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "strategy_id": strategy_id,
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "created_at": v.created_at,
                    "created_by": v.created_by,
                    "comment": v.comment,
                    "tags": v.tags
                }
                for v in results
            ],
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"搜索版本失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
