"""
订单路由服务层
封装实际的订单路由组件，为API提供统一接口
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# 导入交易层组件
try:
    from src.trading.execution.smart_execution import SmartExecution
    SMART_EXECUTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入智能执行: {e}")
    SMART_EXECUTION_AVAILABLE = False

try:
    from src.trading.execution.order_manager import OrderManager
    ORDER_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入订单管理器: {e}")
    ORDER_MANAGER_AVAILABLE = False


# 单例实例
_smart_execution: Optional[Any] = None
_order_manager: Optional[Any] = None


def get_smart_execution() -> Optional[Any]:
    """获取智能执行实例"""
    global _smart_execution
    if _smart_execution is None and SMART_EXECUTION_AVAILABLE:
        try:
            _smart_execution = SmartExecution()
            logger.info("智能执行初始化成功")
        except Exception as e:
            logger.error(f"初始化智能执行失败: {e}")
    return _smart_execution


def get_order_manager() -> Optional[Any]:
    """获取订单管理器实例"""
    global _order_manager
    if _order_manager is None and ORDER_MANAGER_AVAILABLE:
        try:
            _order_manager = OrderManager()
            logger.info("订单管理器初始化成功")
        except Exception as e:
            logger.error(f"初始化订单管理器失败: {e}")
    return _order_manager


# ==================== 订单路由服务 ====================

def get_routing_decisions() -> List[Dict[str, Any]]:
    """获取路由决策列表 - 从真实路由系统获取，不使用模拟数据"""
    smart_execution = get_smart_execution()
    order_manager = get_order_manager()
    
    decisions = []
    
    # 尝试从智能执行获取路由决策
    if smart_execution:
        try:
            if hasattr(smart_execution, 'get_routing_decisions'):
                decisions = smart_execution.get_routing_decisions()
            elif hasattr(smart_execution, 'get_recent_routes'):
                decisions = smart_execution.get_recent_routes()
            elif hasattr(smart_execution, 'list_routing_history'):
                decisions = smart_execution.list_routing_history()
        except Exception as e:
            logger.debug(f"从智能执行获取路由决策失败: {e}")
    
    # 尝试从订单管理器获取路由决策
    if not decisions and order_manager:
        try:
            if hasattr(order_manager, 'get_routing_decisions'):
                decisions = order_manager.get_routing_decisions()
            elif hasattr(order_manager, 'get_order_routes'):
                orders = order_manager.get_recent_orders(limit=100)
                # 从订单中提取路由信息
                for order in orders:
                    if hasattr(order, 'routing_decision') or isinstance(order, dict) and 'routing_decision' in order:
                        routing = order.get('routing_decision') if isinstance(order, dict) else getattr(order, 'routing_decision', None)
                        if routing:
                            decisions.append(routing)
        except Exception as e:
            logger.debug(f"从订单管理器获取路由决策失败: {e}")
    
    # 格式化决策数据
    if decisions:
        formatted_decisions = []
        for decision in decisions:
            if not isinstance(decision, dict):
                if hasattr(decision, '__dict__'):
                    decision_dict = decision.__dict__
                elif hasattr(decision, 'to_dict'):
                    decision_dict = decision.to_dict()
                else:
                    continue
            else:
                decision_dict = decision
            
            decision_data = {
                "order_id": decision_dict.get('order_id', ''),
                "routing_strategy": decision_dict.get('routing_strategy', ''),
                "target_route": decision_dict.get('target_route', ''),
                "cost": decision_dict.get('cost', 0),
                "latency": decision_dict.get('latency', 0),
                "status": decision_dict.get('status', 'unknown'),
                "timestamp": decision_dict.get('timestamp', int(datetime.now().timestamp())),
                "failure_reason": decision_dict.get('failure_reason')
            }
            formatted_decisions.append(decision_data)
            
            # 保存到持久化存储
            try:
                from .routing_persistence import save_routing_decision
                save_routing_decision(decision_data)
            except Exception as e:
                logger.debug(f"保存路由决策到持久化存储失败: {e}")
        
        return formatted_decisions
    
    # 量化交易系统要求：不使用模拟数据，返回空列表
    return []


def get_routing_stats() -> Dict[str, Any]:
    """获取路由统计"""
    decisions = get_routing_decisions()
    
    today = datetime.now().date()
    today_routes = [d for d in decisions if d.get('timestamp') and 
                    datetime.fromtimestamp(d['timestamp']).date() == today]
    
    success_routes = [d for d in decisions if d.get('status') == 'success']
    success_rate = len(success_routes) / len(decisions) if decisions else 0.0
    
    costs = [d.get('cost', 0) for d in decisions if d.get('cost')]
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    
    latencies = [d.get('latency', 0) for d in decisions if d.get('latency')]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        "today_routes": len(today_routes),
        "success_rate": success_rate,
        "avg_cost": avg_cost,
        "avg_latency": avg_latency
    }


def get_routing_performance() -> Dict[str, Any]:
    """获取路由性能"""
    decisions = get_routing_decisions()
    
    # 生成性能趋势
    performance_trend = []
    for i in range(24):
        hour_ago = datetime.now() - timedelta(hours=i)
        hour_decisions = [d for d in decisions if d.get('timestamp') and 
                         datetime.fromtimestamp(d['timestamp']) >= hour_ago - timedelta(hours=1)]
        if hour_decisions:
            success_count = len([d for d in hour_decisions if d.get('status') == 'success'])
            success_rate = success_count / len(hour_decisions) if hour_decisions else 0.0
            performance_trend.append({"success_rate": success_rate, "timestamp": int(hour_ago.timestamp())})
    
    performance_trend.reverse()
    
    # 生成策略分布
    strategy_distribution = {}
    for decision in decisions:
        strategy = decision.get('routing_strategy', 'unknown')
        strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
    
    # 生成成本趋势
    cost_trend = []
    for i in range(24):
        hour_ago = datetime.now() - timedelta(hours=i)
        hour_decisions = [d for d in decisions if d.get('timestamp') and 
                         datetime.fromtimestamp(d['timestamp']) >= hour_ago - timedelta(hours=1)]
        if hour_decisions:
            avg_cost = sum(d.get('cost', 0) for d in hour_decisions) / len(hour_decisions)
            cost_trend.append({"cost": avg_cost, "timestamp": int(hour_ago.timestamp())})
    
    cost_trend.reverse()
    
    # 生成失败分析
    failures = [d for d in decisions if d.get('status') == 'failed']
    failure_list = [
        {
            "order_id": f.get('order_id', 'unknown'),
            "reason": f.get('failure_reason', '未知原因'),
            "timestamp": f.get('timestamp', int(datetime.now().timestamp()))
        }
        for f in failures[:10]  # 只返回最近10个失败
    ]
    
    return {
        "performance_trend": performance_trend,
        "strategy_distribution": strategy_distribution,
        "cost_trend": cost_trend,
        "failures": failure_list
    }


# ==================== 路由决策详情 ====================

def get_routing_decision_detail(decision_id: str) -> Optional[Dict[str, Any]]:
    """
    获取路由决策详情
    
    Args:
        decision_id: 决策ID
        
    Returns:
        决策详情字典，不存在则返回None
    """
    decisions = get_routing_decisions()
    
    for decision in decisions:
        if decision.get('decision_id') == decision_id or decision.get('order_id') == decision_id:
            # 添加额外详情信息
            detail = decision.copy()
            detail['query_time'] = datetime.now().isoformat()
            detail['detail_level'] = 'full'
            
            # 尝试从智能执行获取更详细信息
            smart_execution = get_smart_execution()
            if smart_execution and hasattr(smart_execution, 'get_routing_detail'):
                try:
                    extra_detail = smart_execution.get_routing_detail(decision_id)
                    if extra_detail:
                        detail.update(extra_detail)
                except Exception as e:
                    logger.debug(f"获取详细路由信息失败: {e}")
            
            return detail
    
    return None


# ==================== 筛选查询 ====================

def get_filtered_routing_decisions(
    strategy_id: Optional[str] = None,
    status: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    获取筛选后的路由决策列表
    
    Args:
        strategy_id: 策略ID筛选
        status: 状态筛选 (success/failed/pending)
        start_time: 开始时间筛选 (ISO格式字符串)
        end_time: 结束时间筛选 (ISO格式字符串)
        limit: 返回记录数量限制
        
    Returns:
        筛选后的决策列表
    """
    decisions = get_routing_decisions()
    filtered = decisions
    
    # 按策略ID筛选
    if strategy_id:
        filtered = [d for d in filtered if d.get('strategy_id') == strategy_id or 
                    d.get('routing_strategy') == strategy_id]
    
    # 按状态筛选
    if status:
        filtered = [d for d in filtered if d.get('status') == status]
    
    # 按时间范围筛选
    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            start_ts = start_dt.timestamp()
            filtered = [d for d in filtered if d.get('timestamp', 0) >= start_ts]
        except Exception as e:
            logger.warning(f"解析开始时间失败: {e}")
    
    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            end_ts = end_dt.timestamp()
            filtered = [d for d in filtered if d.get('timestamp', 0) <= end_ts]
        except Exception as e:
            logger.warning(f"解析结束时间失败: {e}")
    
    # 按时间倒序排序
    filtered = sorted(filtered, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    # 限制返回数量
    return filtered[:limit]


# ==================== 降级方案 ====================

# 注意：已移除_get_mock_routing_decisions()函数，系统要求不使用模拟数据

