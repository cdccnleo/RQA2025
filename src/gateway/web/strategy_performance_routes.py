"""
策略性能监控路由模块
提供策略性能监控API端点
符合量化交易系统安全要求
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

# 量化交易系统安全要求：审计日志函数
def audit_log(action: str, strategy_id: str, user_id: str = None, details: dict = None):
    """记录审计日志"""
    log_entry = f"AUDIT: {action} | strategy={strategy_id}"
    if user_id:
        log_entry += f" | user={user_id}"
    if details:
        log_entry += f" | details={details}"
    logger.info(log_entry)

# 导入性能监控器
from .strategy_performance_monitor import (
    performance_monitor,
    record_strategy_performance,
    get_strategy_performance_history,
    generate_strategy_performance_report,
    calculate_strategy_score
)


@router.post("/api/v1/strategy/{strategy_id}/performance/record")
async def record_performance_api(strategy_id: str, request: Dict[str, Any]):
    """记录策略性能 - 符合量化交易系统安全要求"""
    try:
        metrics = request.get("metrics")
        period = request.get("period", "daily")
        metadata = request.get("metadata", {})
        
        if not metrics:
            raise HTTPException(status_code=400, detail="缺少性能指标")
        
        # 量化交易系统安全要求：记录审计日志
        audit_log("record_performance", strategy_id, details={
            "period": period,
            "metrics_count": len(metrics) if isinstance(metrics, dict) else 0
        })
        
        snapshot = record_strategy_performance(strategy_id, metrics, period, metadata)
        
        return {
            "success": True,
            "snapshot_id": snapshot.snapshot_id,
            "strategy_id": strategy_id,
            "timestamp": snapshot.timestamp,
            "message": "性能记录成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"记录性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"记录失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/history")
async def get_performance_history_api(
    strategy_id: str,
    metric_name: Optional[str] = Query(None, description="指标名称"),
    days: int = Query(30, description="天数")
):
    """获取性能历史 - 符合量化交易系统安全要求"""
    try:
        # 量化交易系统安全要求：记录审计日志
        audit_log("view_performance_history", strategy_id, details={
            "metric": metric_name,
            "days": days
        })
        
        history = get_strategy_performance_history(strategy_id, metric_name, days)
        
        return {
            "strategy_id": strategy_id,
            "metric": metric_name or "all",
            "days": days,
            "data_points": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"获取性能历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/trend")
async def analyze_trend_api(
    strategy_id: str,
    metric_name: str = Query(..., description="指标名称"),
    days: int = Query(30, description="天数")
):
    """分析指标趋势"""
    try:
        trend = performance_monitor.analyze_trend(strategy_id, metric_name, days=days)
        
        if "error" in trend:
            raise HTTPException(status_code=400, detail=trend["error"])
        
        return trend
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析趋势失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/performance/compare")
async def compare_periods_api(strategy_id: str, request: Dict[str, Any]):
    """对比两个时期"""
    try:
        metric_name = request.get("metric_name")
        period1 = request.get("period1", {})
        period2 = request.get("period2", {})
        
        if not metric_name:
            raise HTTPException(status_code=400, detail="缺少指标名称")
        
        result = performance_monitor.compare_periods(
            strategy_id,
            metric_name,
            period1.get("start"),
            period1.get("end"),
            period2.get("start"),
            period2.get("end")
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对比时期失败: {e}")
        raise HTTPException(status_code=500, detail=f"对比失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/report")
async def generate_report_api(
    strategy_id: str,
    days: int = Query(30, description="报告天数")
):
    """生成性能报告"""
    try:
        report = generate_strategy_performance_report(strategy_id, days)
        
        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/score")
async def calculate_score_api(strategy_id: str):
    """计算性能评分"""
    try:
        score = calculate_strategy_score(strategy_id)
        
        if "error" in score:
            raise HTTPException(status_code=400, detail=score["error"])
        
        return score
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算评分失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/latest")
async def get_latest_metrics_api(strategy_id: str):
    """获取最新指标"""
    try:
        metrics = performance_monitor.get_latest_metrics(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "timestamp": time.time(),
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"获取最新指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/{strategy_id}/performance/metrics")
async def get_metric_definitions_api(strategy_id: str):
    """获取指标定义"""
    try:
        definitions = performance_monitor.metric_definitions
        
        return {
            "strategy_id": strategy_id,
            "metrics": definitions
        }
        
    except Exception as e:
        logger.error(f"获取指标定义失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.post("/api/v1/strategy/{strategy_id}/performance/batch-record")
async def batch_record_performance_api(strategy_id: str, request: Dict[str, Any]):
    """批量记录性能"""
    try:
        records = request.get("records", [])
        
        if not records:
            raise HTTPException(status_code=400, detail="缺少记录数据")
        
        success_count = 0
        for record in records:
            try:
                metrics = record.get("metrics")
                period = record.get("period", "daily")
                metadata = record.get("metadata", {})
                
                if metrics:
                    record_strategy_performance(strategy_id, metrics, period, metadata)
                    success_count += 1
            except Exception as e:
                logger.warning(f"批量记录中单项失败: {e}")
        
        return {
            "success": True,
            "message": f"成功记录 {success_count}/{len(records)} 条数据",
            "success_count": success_count,
            "total_count": len(records)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量记录性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量记录失败: {str(e)}")


import time
from typing import List
from datetime import datetime, timedelta


@router.get("/api/v1/strategy/performance/comparison")
async def compare_strategy_performance(
    strategy_ids: str = Query(..., description="策略ID列表，逗号分隔"),
    days: int = Query(30, description="对比天数")
):
    """
    对比多个策略的性能
    
    Args:
        strategy_ids: 策略ID列表，逗号分隔
        days: 对比天数
        
    Returns:
        策略性能对比数据
    """
    try:
        # 解析策略ID列表
        strategy_id_list = [s.strip() for s in strategy_ids.split(",") if s.strip()]
        
        if len(strategy_id_list) < 2:
            raise HTTPException(status_code=400, detail="至少需要两个策略进行对比")
        
        if len(strategy_id_list) > 10:
            raise HTTPException(status_code=400, detail="最多支持10个策略对比")
        
        # 获取每个策略的性能数据
        comparison_data = []
        for strategy_id in strategy_id_list:
            try:
                # 获取性能评分
                score = calculate_strategy_score(strategy_id)
                
                # 获取最新指标
                latest_metrics = performance_monitor.get_latest_metrics(strategy_id)
                
                # 获取历史数据
                history = get_strategy_performance_history(strategy_id, days=days)
                
                comparison_data.append({
                    "strategy_id": strategy_id,
                    "score": score.get("score", 0) if "error" not in score else 0,
                    "grade": score.get("grade", "N/A") if "error" not in score else "N/A",
                    "latest_metrics": latest_metrics,
                    "history_summary": {
                        "data_points": len(history),
                        "avg_return": calculate_avg_return(history),
                        "max_drawdown": calculate_max_drawdown(history),
                        "sharpe_ratio": calculate_sharpe_ratio(history)
                    }
                })
            except Exception as e:
                logger.warning(f"获取策略 {strategy_id} 性能数据失败: {e}")
                comparison_data.append({
                    "strategy_id": strategy_id,
                    "error": str(e)
                })
        
        # 计算排名
        valid_data = [d for d in comparison_data if "error" not in d]
        if valid_data:
            valid_data.sort(key=lambda x: x["score"], reverse=True)
            for i, data in enumerate(valid_data):
                data["rank"] = i + 1
        
        return {
            "comparison_date": datetime.now().isoformat(),
            "days": days,
            "strategy_count": len(strategy_id_list),
            "comparison_data": comparison_data,
            "best_strategy": valid_data[0]["strategy_id"] if valid_data else None,
            "summary": {
                "avg_score": sum(d.get("score", 0) for d in valid_data) / len(valid_data) if valid_data else 0,
                "highest_score": max(d.get("score", 0) for d in valid_data) if valid_data else 0,
                "lowest_score": min(d.get("score", 0) for d in valid_data) if valid_data else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对比策略性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"对比失败: {str(e)}")


@router.get("/api/v1/strategy/performance/metrics")
async def get_strategy_metrics(
    strategy_id: str = Query(..., description="策略ID"),
    days: int = Query(30, description="天数")
):
    """
    获取策略详细性能指标
    
    Args:
        strategy_id: 策略ID
        days: 统计天数
        
    Returns:
        策略详细性能指标
    """
    try:
        # 获取性能评分
        score = calculate_strategy_score(strategy_id)
        
        # 获取最新指标
        latest_metrics = performance_monitor.get_latest_metrics(strategy_id)
        
        # 获取历史数据
        history = get_strategy_performance_history(strategy_id, days=days)
        
        # 计算详细指标
        metrics_detail = {
            "strategy_id": strategy_id,
            "query_date": datetime.now().isoformat(),
            "days": days,
            "score": score if "error" not in score else {"score": 0, "grade": "N/A"},
            "latest_metrics": latest_metrics,
            "historical_analysis": analyze_historical_data(history),
            "risk_metrics": calculate_risk_metrics(history),
            "return_metrics": calculate_return_metrics(history),
            "trading_metrics": calculate_trading_metrics(history)
        }
        
        return metrics_detail
        
    except Exception as e:
        logger.error(f"获取策略指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/api/v1/strategy/performance/{strategy_id}")
async def get_strategy_performance_detail(strategy_id: str):
    """
    获取单个策略性能详情
    
    Args:
        strategy_id: 策略ID
        
    Returns:
        策略性能详情
    """
    try:
        # 获取性能评分
        score = calculate_strategy_score(strategy_id)
        
        # 获取最新指标
        latest_metrics = performance_monitor.get_latest_metrics(strategy_id)
        
        # 获取历史数据（最近90天）
        history = get_strategy_performance_history(strategy_id, days=90)
        
        # 生成趋势分析
        trend_analysis = {}
        for metric_name in ["total_return", "sharpe_ratio", "max_drawdown"]:
            try:
                trend = performance_monitor.analyze_trend(strategy_id, metric_name, days=30)
                trend_analysis[metric_name] = trend
            except Exception as e:
                logger.warning(f"分析趋势失败 {metric_name}: {e}")
        
        # 获取指标定义
        metric_definitions = performance_monitor.metric_definitions
        
        return {
            "strategy_id": strategy_id,
            "query_timestamp": time.time(),
            "score": score if "error" not in score else {"score": 0, "grade": "N/A"},
            "current_metrics": latest_metrics,
            "historical_data": {
                "days": 90,
                "data_points": len(history),
                "data": history[-30:] if len(history) > 30 else history  # 只返回最近30条
            },
            "trend_analysis": trend_analysis,
            "metric_definitions": metric_definitions,
            "performance_summary": {
                "overall_grade": score.get("grade", "N/A") if "error" not in score else "N/A",
                "recommendation": generate_recommendation(score, latest_metrics),
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"获取策略性能详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


# 辅助函数
def calculate_avg_return(history: List[Dict]) -> float:
    """计算平均收益"""
    if not history:
        return 0.0
    returns = [h.get("total_return", 0) for h in history if "total_return" in h]
    return sum(returns) / len(returns) if returns else 0.0


def calculate_max_drawdown(history: List[Dict]) -> float:
    """计算最大回撤"""
    if not history:
        return 0.0
    drawdowns = [h.get("max_drawdown", 0) for h in history if "max_drawdown" in h]
    return min(drawdowns) if drawdowns else 0.0


def calculate_sharpe_ratio(history: List[Dict]) -> float:
    """计算夏普比率"""
    if not history:
        return 0.0
    sharpe_ratios = [h.get("sharpe_ratio", 0) for h in history if "sharpe_ratio" in h]
    return sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0.0


def analyze_historical_data(history: List[Dict]) -> Dict:
    """分析历史数据"""
    if not history:
        return {"error": "无历史数据"}
    
    returns = [h.get("total_return", 0) for h in history if "total_return" in h]
    drawdowns = [h.get("max_drawdown", 0) for h in history if "max_drawdown" in h]
    
    return {
        "data_points": len(history),
        "avg_return": sum(returns) / len(returns) if returns else 0,
        "best_return": max(returns) if returns else 0,
        "worst_return": min(returns) if returns else 0,
        "avg_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        "max_drawdown": min(drawdowns) if drawdowns else 0
    }


def calculate_risk_metrics(history: List[Dict]) -> Dict:
    """计算风险指标"""
    if not history:
        return {"error": "无历史数据"}
    
    drawdowns = [h.get("max_drawdown", 0) for h in history if "max_drawdown" in h]
    volatility = [h.get("volatility", 0) for h in history if "volatility" in h]
    
    return {
        "max_drawdown": min(drawdowns) if drawdowns else 0,
        "avg_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        "volatility": sum(volatility) / len(volatility) if volatility else 0,
        "risk_level": "high" if (min(drawdowns) if drawdowns else 0) < -0.2 else "medium" if (min(drawdowns) if drawdowns else 0) < -0.1 else "low"
    }


def calculate_return_metrics(history: List[Dict]) -> Dict:
    """计算收益指标"""
    if not history:
        return {"error": "无历史数据"}
    
    returns = [h.get("total_return", 0) for h in history if "total_return" in h]
    
    return {
        "total_return": returns[-1] if returns else 0,
        "avg_daily_return": sum(returns) / len(returns) if returns else 0,
        "best_day": max(returns) if returns else 0,
        "worst_day": min(returns) if returns else 0,
        "positive_days": sum(1 for r in returns if r > 0),
        "negative_days": sum(1 for r in returns if r < 0)
    }


def calculate_trading_metrics(history: List[Dict]) -> Dict:
    """计算交易指标"""
    if not history:
        return {"error": "无历史数据"}
    
    win_rates = [h.get("win_rate", 0) for h in history if "win_rate" in h]
    trade_counts = [h.get("trade_count", 0) for h in history if "trade_count" in h]
    
    return {
        "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
        "total_trades": sum(trade_counts),
        "avg_trades_per_day": sum(trade_counts) / len(history) if history else 0
    }


def generate_recommendation(score: Dict, metrics: Dict) -> str:
    """生成建议"""
    if "error" in score:
        return "无法生成建议：评分计算失败"
    
    strategy_score = score.get("score", 0)
    
    if strategy_score >= 80:
        return "策略表现优秀，建议保持当前配置"
    elif strategy_score >= 60:
        return "策略表现良好，可考虑小幅优化"
    elif strategy_score >= 40:
        return "策略表现一般，建议进行优化调整"
    else:
        return "策略表现较差，建议重新评估策略逻辑"
