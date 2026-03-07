"""
信号监控和告警 API
提供信号质量监控、统计信息、告警管理等功能
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/signals", tags=["信号监控"])


class SignalValidationRequest(BaseModel):
    """信号验证请求"""
    signal_id: str = Field(..., description="信号ID")
    symbol: str = Field(..., description="股票代码")
    signal_type: str = Field(..., description="信号类型")
    lookback_periods: int = Field(default=20, description="回测周期数")


class SignalValidationResponse(BaseModel):
    """信号验证响应"""
    signal_id: str
    symbol: str
    signal_type: str
    overall_score: float
    quality_level: str
    is_valid: bool
    accuracy_score: float
    risk_score: float
    profit_score: float
    consistency_score: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    validation_timestamp: str


class SignalStatisticsResponse(BaseModel):
    """信号统计响应"""
    total_signals: int
    valid_signals: int
    invalid_signals: int
    avg_score: float
    quality_distribution: Dict[str, int]
    win_rate_avg: float
    sharpe_ratio_avg: float
    max_drawdown_avg: float


class AlertRule(BaseModel):
    """告警规则"""
    rule_id: str
    name: str
    condition: str  # "score_below", "win_rate_below", "drawdown_above"
    threshold: float
    enabled: bool = True


class Alert(BaseModel):
    """告警"""
    alert_id: str
    rule_id: str
    signal_id: str
    symbol: str
    message: str
    severity: str  # "info", "warning", "critical"
    timestamp: str
    acknowledged: bool = False


# 告警规则存储（实际应用中应该使用数据库）
_alert_rules: Dict[str, AlertRule] = {}
_alerts: List[Alert] = []


@router.post("/validate", response_model=SignalValidationResponse)
async def validate_signal(request: SignalValidationRequest):
    """
    验证信号质量
    
    对指定信号进行质量评分和回测验证
    """
    try:
        from src.trading.signal.signal_validator import get_signal_validator
        from src.gateway.web.market_data_service import get_market_data_service
        
        # 获取市场数据
        market_service = get_market_data_service()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = market_service.get_stock_data(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail=f"找不到股票 {request.symbol} 的历史数据")
        
        # 构建信号数据
        signal = {
            'id': request.signal_id,
            'symbol': request.symbol,
            'type': request.signal_type,
            'timestamp': datetime.now()
        }
        
        # 验证信号
        validator = get_signal_validator()
        result = validator.validate_signal(signal, historical_data, request.lookback_periods)
        
        return SignalValidationResponse(
            signal_id=result.signal_id,
            symbol=result.symbol,
            signal_type=result.signal_type,
            overall_score=round(result.overall_score, 4),
            quality_level=result.quality_level.value,
            is_valid=result.is_valid,
            accuracy_score=round(result.accuracy_score, 4),
            risk_score=round(result.risk_score, 4),
            profit_score=round(result.profit_score, 4),
            consistency_score=round(result.consistency_score, 4),
            win_rate=round(result.win_rate, 4),
            sharpe_ratio=round(result.sharpe_ratio, 4),
            max_drawdown=round(result.max_drawdown, 4),
            total_trades=result.total_trades,
            validation_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"验证信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证信号失败: {str(e)}")


@router.get("/statistics", response_model=SignalStatisticsResponse)
async def get_signal_statistics(
    symbol: Optional[str] = Query(None, description="股票代码"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间")
):
    """
    获取信号统计信息
    
    返回信号的质量分布、胜率、夏普比率等统计指标
    """
    try:
        from src.trading.signal.signal_validator import get_signal_validator
        
        validator = get_signal_validator()
        stats = validator.get_signal_statistics(symbol, start_time, end_time)
        
        return SignalStatisticsResponse(
            total_signals=stats.get('total_signals', 0),
            valid_signals=stats.get('valid_signals', 0),
            invalid_signals=stats.get('invalid_signals', 0),
            avg_score=round(stats.get('avg_score', 0.0), 4),
            quality_distribution=stats.get('quality_distribution', {}),
            win_rate_avg=round(stats.get('win_rate_avg', 0.0), 4),
            sharpe_ratio_avg=round(stats.get('sharpe_ratio_avg', 0.0), 4),
            max_drawdown_avg=round(stats.get('max_drawdown_avg', 0.0), 4)
        )
        
    except Exception as e:
        logger.error(f"获取信号统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/validated", response_model=List[SignalValidationResponse])
async def get_validated_signals(
    symbol: Optional[str] = Query(None, description="股票代码"),
    min_score: float = Query(0.0, description="最低评分"),
    limit: int = Query(100, description="返回数量限制")
):
    """
    获取已验证的信号列表
    
    返回经过验证的信号及其质量评分
    """
    try:
        from src.trading.signal.signal_validator import get_signal_validator
        
        validator = get_signal_validator()
        
        # 获取所有验证结果
        results = []
        for cache_key, result in validator._validation_cache.items():
            if symbol and result.symbol != symbol:
                continue
            if result.overall_score < min_score:
                continue
            
            results.append(SignalValidationResponse(
                signal_id=result.signal_id,
                symbol=result.symbol,
                signal_type=result.signal_type,
                overall_score=round(result.overall_score, 4),
                quality_level=result.quality_level.value,
                is_valid=result.is_valid,
                accuracy_score=round(result.accuracy_score, 4),
                risk_score=round(result.risk_score, 4),
                profit_score=round(result.profit_score, 4),
                consistency_score=round(result.consistency_score, 4),
                win_rate=round(result.win_rate, 4),
                sharpe_ratio=round(result.sharpe_ratio, 4),
                max_drawdown=round(result.max_drawdown, 4),
                total_trades=result.total_trades,
                validation_timestamp=result.timestamp.isoformat()
            ))
        
        # 按评分排序
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        logger.error(f"获取已验证信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")


# ==================== 告警管理 API ====================

@router.post("/alerts/rules", response_model=AlertRule)
async def create_alert_rule(rule: AlertRule):
    """创建告警规则"""
    try:
        _alert_rules[rule.rule_id] = rule
        logger.info(f"创建告警规则: {rule.rule_id}")
        return rule
    except Exception as e:
        logger.error(f"创建告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建规则失败: {str(e)}")


@router.get("/alerts/rules", response_model=List[AlertRule])
async def get_alert_rules():
    """获取所有告警规则"""
    return list(_alert_rules.values())


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """删除告警规则"""
    if rule_id not in _alert_rules:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    del _alert_rules[rule_id]
    logger.info(f"删除告警规则: {rule_id}")
    return {"message": "规则已删除"}


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    acknowledged: Optional[bool] = Query(None, description="是否已确认"),
    severity: Optional[str] = Query(None, description="严重级别")
):
    """获取告警列表"""
    filtered_alerts = _alerts
    
    if acknowledged is not None:
        filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
    
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
    
    return filtered_alerts


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """确认告警"""
    for alert in _alerts:
        if alert.alert_id == alert_id:
            alert.acknowledged = True
            logger.info(f"确认告警: {alert_id}")
            return {"message": "告警已确认"}
    
    raise HTTPException(status_code=404, detail="告警不存在")


@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """删除告警"""
    global _alerts
    original_count = len(_alerts)
    _alerts = [a for a in _alerts if a.alert_id != alert_id]
    
    if len(_alerts) == original_count:
        raise HTTPException(status_code=404, detail="告警不存在")
    
    logger.info(f"删除告警: {alert_id}")
    return {"message": "告警已删除"}


# ==================== 告警检查函数 ====================

def check_signal_alerts(signal_validation_result):
    """
    检查信号是否需要触发告警
    
    Args:
        signal_validation_result: 信号验证结果
    """
    try:
        for rule in _alert_rules.values():
            if not rule.enabled:
                continue
            
            should_alert = False
            message = ""
            severity = "info"
            
            if rule.condition == "score_below":
                if signal_validation_result.overall_score < rule.threshold:
                    should_alert = True
                    message = f"信号 {signal_validation_result.signal_id} 评分 {signal_validation_result.overall_score:.2f} 低于阈值 {rule.threshold}"
                    severity = "warning" if signal_validation_result.overall_score < 0.3 else "info"
            
            elif rule.condition == "win_rate_below":
                if signal_validation_result.win_rate < rule.threshold:
                    should_alert = True
                    message = f"信号 {signal_validation_result.signal_id} 胜率 {signal_validation_result.win_rate:.2f} 低于阈值 {rule.threshold}"
                    severity = "warning"
            
            elif rule.condition == "drawdown_above":
                if abs(signal_validation_result.max_drawdown) > rule.threshold:
                    should_alert = True
                    message = f"信号 {signal_validation_result.signal_id} 最大回撤 {signal_validation_result.max_drawdown:.2f} 超过阈值 {rule.threshold}"
                    severity = "critical"
            
            if should_alert:
                alert = Alert(
                    alert_id=f"alert_{datetime.now().timestamp()}",
                    rule_id=rule.rule_id,
                    signal_id=signal_validation_result.signal_id,
                    symbol=signal_validation_result.symbol,
                    message=message,
                    severity=severity,
                    timestamp=datetime.now().isoformat()
                )
                _alerts.append(alert)
                logger.warning(f"触发告警: {message}")
                
    except Exception as e:
        logger.error(f"检查告警失败: {e}")


def initialize_default_alert_rules():
    """初始化默认告警规则"""
    default_rules = [
        AlertRule(
            rule_id="low_score",
            name="低质量信号告警",
            condition="score_below",
            threshold=0.3,
            enabled=True
        ),
        AlertRule(
            rule_id="low_win_rate",
            name="低胜率告警",
            condition="win_rate_below",
            threshold=0.4,
            enabled=True
        ),
        AlertRule(
            rule_id="high_drawdown",
            name="高回撤告警",
            condition="drawdown_above",
            threshold=0.15,
            enabled=True
        )
    ]
    
    for rule in default_rules:
        _alert_rules[rule.rule_id] = rule
    
    logger.info(f"初始化 {len(default_rules)} 个默认告警规则")


# 初始化默认规则
initialize_default_alert_rules()
