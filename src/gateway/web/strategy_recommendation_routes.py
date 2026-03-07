"""
策略推荐系统路由模块
提供策略智能推荐API端点
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# 导入推荐引擎
from .strategy_recommendation_engine import (
    recommendation_engine,
    analyze_backtest_and_recommend,
    get_strategy_recommendations,
    generate_performance_alert
)


@router.post("/api/v1/strategy/{strategy_id}/recommendations/analyze")
async def analyze_backtest_api(strategy_id: str, request: Dict[str, Any]):
    """分析回测结果并生成推荐"""
    try:
        backtest_result = request.get("backtest_result")
        
        if not backtest_result:
            raise HTTPException(status_code=400, detail="缺少回测结果")
        
        recommendations = analyze_backtest_and_recommend(strategy_id, backtest_result)
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "recommendations_generated": len(recommendations),
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "type": r.recommendation_type,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "priority": r.priority,
                    "created_at": r.created_at
                }
                for r in recommendations
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析回测结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/recommendations")
async def get_recommendations_api(
    strategy_id: str,
    recommendation_type: Optional[str] = Query(None, description="推荐类型筛选"),
    unread_only: bool = Query(False, description="仅未读")
):
    """获取策略推荐列表"""
    try:
        recommendations = recommendation_engine.get_recommendations(
            strategy_id, 
            recommendation_type=recommendation_type,
            unread_only=unread_only
        )
        
        return {
            "strategy_id": strategy_id,
            "total": len(recommendations),
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "type": r.recommendation_type,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "priority": r.priority,
                    "created_at": r.created_at,
                    "is_read": r.is_read,
                    "is_applied": r.is_applied,
                    "metadata": r.metadata
                }
                for r in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"获取推荐列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/{recommendation_id}/read")
async def mark_recommendation_read_api(strategy_id: str, recommendation_id: str):
    """标记推荐为已读"""
    try:
        success = recommendation_engine.mark_recommendation_read(strategy_id, recommendation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="推荐不存在")
        
        return {
            "success": True,
            "message": "已标记为已读"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"标记推荐已读失败: {e}")
        raise HTTPException(status_code=500, detail=f"标记失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/{recommendation_id}/apply")
async def mark_recommendation_applied_api(strategy_id: str, recommendation_id: str):
    """标记推荐为已应用"""
    try:
        success = recommendation_engine.mark_recommendation_applied(strategy_id, recommendation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="推荐不存在")
        
        return {
            "success": True,
            "message": "已标记为已应用"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"标记推荐已应用失败: {e}")
        raise HTTPException(status_code=500, detail=f"标记失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/optimize-direction")
async def recommend_optimize_direction_api(strategy_id: str, request: Dict[str, Any]):
    """推荐优化方向"""
    try:
        backtest_history = request.get("backtest_history", [])
        
        recommendations = recommendation_engine.recommend_optimization_direction(
            strategy_id, backtest_history
        )
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "priority": r.priority
                }
                for r in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"推荐优化方向失败: {e}")
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/parameter-range")
async def recommend_parameter_range_api(strategy_id: str, request: Dict[str, Any]):
    """推荐参数范围"""
    try:
        current_params = request.get("current_params", {})
        optimization_history = request.get("optimization_history", [])
        
        recommendations = recommendation_engine.recommend_parameter_range(
            strategy_id, current_params, optimization_history
        )
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "title": r.title,
                    "description": r.description,
                    "confidence": r.confidence,
                    "priority": r.priority,
                    "metadata": r.metadata
                }
                for r in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"推荐参数范围失败: {e}")
        raise HTTPException(status_code=500, detail=f"推荐失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/performance-alert")
async def generate_performance_alert_api(strategy_id: str, request: Dict[str, Any]):
    """生成性能预警"""
    try:
        current_metrics = request.get("current_metrics", {})
        baseline_metrics = request.get("baseline_metrics", {})
        
        alert = generate_performance_alert(strategy_id, current_metrics, baseline_metrics)
        
        if alert:
            return {
                "success": True,
                "alert_generated": True,
                "alert": {
                    "recommendation_id": alert.recommendation_id,
                    "title": alert.title,
                    "description": alert.description,
                    "priority": alert.priority,
                    "confidence": alert.confidence
                }
            }
        else:
            return {
                "success": True,
                "alert_generated": False,
                "message": "未生成预警，策略表现正常"
            }
        
    except Exception as e:
        logger.error(f"生成性能预警失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/recommendations/unread-count")
async def get_unread_recommendations_count_api(strategy_id: str):
    """获取未读推荐数量"""
    try:
        recommendations = recommendation_engine.get_recommendations(
            strategy_id, unread_only=True
        )
        
        # 按优先级分组
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in recommendations:
            priority_counts[r.priority] = priority_counts.get(r.priority, 0) + 1
        
        return {
            "strategy_id": strategy_id,
            "total_unread": len(recommendations),
            "priority_distribution": priority_counts,
            "has_high_priority": any(r.priority >= 4 for r in recommendations)
        }
        
    except Exception as e:
        logger.error(f"获取未读推荐数量失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/recommendations/mark-all-read")
async def mark_all_recommendations_read_api(strategy_id: str):
    """标记所有推荐为已读"""
    try:
        recommendations = recommendation_engine.get_recommendations(strategy_id)
        
        marked_count = 0
        for rec in recommendations:
            if not rec.is_read:
                if recommendation_engine.mark_recommendation_read(strategy_id, rec.recommendation_id):
                    marked_count += 1
        
        return {
            "success": True,
            "message": f"已标记 {marked_count} 条推荐为已读",
            "marked_count": marked_count
        }
        
    except Exception as e:
        logger.error(f"标记所有推荐已读失败: {e}")
        raise HTTPException(status_code=500, detail=f"标记失败: {str(e)}")
