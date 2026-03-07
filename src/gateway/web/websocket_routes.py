"""
WebSocket路由模块
提供实时数据推送的WebSocket端点
"""

import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from .websocket_api import manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_websocket_token(token: str = None) -> bool:
    """
    验证WebSocket连接令牌
    
    Args:
        token: 认证令牌
        
    Returns:
        是否验证通过
    """
    # TODO: 实现实际的token验证逻辑
    # 目前允许无token连接（向后兼容）
    if token:
        # 这里可以添加实际的token验证逻辑
        # 例如：检查token是否有效、是否过期等
        return True
    return True  # 暂时允许无token连接


@router.websocket("/ws/realtime-metrics")
async def websocket_realtime_metrics(websocket: WebSocket):
    """实时指标WebSocket连接（兼容旧版本）"""
    await manager.connect(websocket, "realtime_metrics")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "realtime_metrics")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "realtime_metrics")


@router.websocket("/ws/dashboard-metrics")
async def websocket_dashboard_metrics(websocket: WebSocket, token: str = Query(None)):
    """Dashboard实时指标WebSocket连接（系统性能+数据流）"""
    # 验证token
    if not _validate_websocket_token(token):
        await websocket.close(code=1008, reason="认证失败")
        return
    
    await manager.connect(websocket, "dashboard_metrics")
    try:
        while True:
            data = await websocket.receive_text()
            # 处理pong响应
            try:
                import json
                message = json.loads(data)
                if message.get("type") == "pong":
                    # 更新心跳时间
                    if websocket in manager._connection_metadata:
                        manager._connection_metadata[websocket]['last_heartbeat'] = time.time()
                    continue
            except:
                pass
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "dashboard_metrics")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "dashboard_metrics")


@router.websocket("/ws/dashboard-alerts")
async def websocket_dashboard_alerts(websocket: WebSocket, token: str = Query(None)):
    """Dashboard实时告警和事件WebSocket连接"""
    # 验证token
    if not _validate_websocket_token(token):
        await websocket.close(code=1008, reason="认证失败")
        return
    
    await manager.connect(websocket, "dashboard_alerts")
    try:
        while True:
            data = await websocket.receive_text()
            # 处理pong响应
            try:
                import json
                message = json.loads(data)
                if message.get("type") == "pong":
                    if websocket in manager._connection_metadata:
                        manager._connection_metadata[websocket]['last_heartbeat'] = time.time()
                    continue
            except:
                pass
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "dashboard_alerts")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "dashboard_alerts")


@router.websocket("/ws/execution-status")
async def websocket_execution_status(websocket: WebSocket):
    """执行状态WebSocket连接"""
    await manager.connect(websocket, "execution_status")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "execution_status")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "execution_status")


@router.websocket("/ws/optimization-progress")
async def websocket_optimization_progress(websocket: WebSocket):
    """优化进度WebSocket连接"""
    await manager.connect(websocket, "optimization_progress")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "optimization_progress")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "optimization_progress")


@router.websocket("/ws/backtest-progress")
async def websocket_backtest_progress(websocket: WebSocket):
    """回测进度WebSocket连接"""
    await manager.connect(websocket, "backtest_progress")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "backtest_progress")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "backtest_progress")


@router.websocket("/ws/lifecycle-events")
async def websocket_lifecycle_events(websocket: WebSocket):
    """生命周期事件WebSocket连接"""
    await manager.connect(websocket, "lifecycle_events")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "lifecycle_events")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "lifecycle_events")


@router.websocket("/ws/trading-signals")
async def websocket_trading_signals(websocket: WebSocket, token: str = Query(None)):
    """
    实时交易信号WebSocket连接
    
    推送实时生成的交易信号，支持历史数据和实时数据组合
    """
    # 验证token
    if not _validate_websocket_token(token):
        await websocket.close(code=1008, reason="认证失败")
        return
    
    await manager.connect(websocket, "trading_signals")
    
    try:
        # 发送初始连接成功消息
        await websocket.send_json({
            "type": "connection_established",
            "message": "实时交易信号WebSocket连接已建立",
            "timestamp": time.time()
        })
        
        while True:
            data = await websocket.receive_text()
            
            # 处理客户端消息
            try:
                import json
                message = json.loads(data)
                
                # 处理pong响应
                if message.get("type") == "pong":
                    if websocket in manager._connection_metadata:
                        manager._connection_metadata[websocket]['last_heartbeat'] = time.time()
                    continue
                
                # 处理订阅请求
                if message.get("type") == "subscribe":
                    symbol = message.get("symbol")
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "symbol": symbol,
                        "timestamp": time.time()
                    })
                    continue
                
                # 处理信号请求
                if message.get("type") == "request_signals":
                    # 获取实时信号
                    from .trading_signal_service import get_realtime_signals_with_live_data
                    signals = await get_realtime_signals_with_live_data()
                    
                    await websocket.send_json({
                        "type": "signals_update",
                        "data": signals,
                        "count": len(signals),
                        "timestamp": time.time()
                    })
                    continue
                    
            except json.JSONDecodeError:
                logger.warning(f"收到无效的JSON消息: {data}")
            except Exception as e:
                logger.error(f"处理WebSocket消息失败: {e}")
            
            logger.debug(f"收到WebSocket消息: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, "trading_signals")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "trading_signals")


@router.websocket("/ws/feature-engineering")
async def websocket_feature_engineering(websocket: WebSocket):
    """特征工程监控WebSocket连接"""
    try:
        await manager.connect(websocket, "feature_engineering")
        try:
            while True:
                data = await websocket.receive_text()
                logger.debug(f"收到WebSocket消息: {data}")
        except WebSocketDisconnect:
            manager.disconnect(websocket, "feature_engineering")
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
            manager.disconnect(websocket, "feature_engineering")
    except Exception as e:
        logger.error(f"WebSocket连接建立失败: {e}")
        try:
            await websocket.close(code=1011, reason=f"连接建立失败: {str(e)}")
        except Exception:
            pass


@router.websocket("/ws/model-training")
async def websocket_model_training(websocket: WebSocket):
    """模型训练监控WebSocket连接 - 集成事件总线实现事件驱动更新"""
    await manager.connect(websocket, "model_training")
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_training_event(event):
            """处理模型训练相关事件（同步函数，内部使用异步发送）"""
            try:
                import asyncio
                import json
                # 创建异步任务发送消息
                message = {
                    "type": "model_training",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.MODEL_TRAINING_STARTED,
            EventType.MODEL_TRAINING_COMPLETED,
            EventType.TRAINING_JOB_CREATED,
            EventType.TRAINING_JOB_UPDATED,
            EventType.TRAINING_JOB_STOPPED
        ]
        
        handlers = []  # 在try块外定义，确保finally块可以访问
        try:
            for event_type in event_types:
                try:
                    # 为每个事件类型创建处理函数
                    def create_handler(et):
                        def handler(event):
                            handle_training_event(event)
                        return handler
                    
                    handler_func = create_handler(event_type)
                    # 订阅事件（使用异步处理器）
                    event_bus.subscribe(
                        event_type,
                        handler_func,
                        async_handler=True
                    )
                    handlers.append((event_type, handler_func))
                    logger.debug(f"已订阅模型训练事件: {event_type}")
                except Exception as e:
                    logger.warning(f"订阅事件 {event_type} 失败: {e}")
            
            # 保持连接，等待客户端消息或事件
            while True:
                try:
                    data = await websocket.receive_text()
                    logger.debug(f"收到WebSocket消息: {data}")
                except WebSocketDisconnect:
                    break
        finally:
            # 取消订阅所有事件
            for event_type, handler_func in handlers:
                try:
                    event_bus.unsubscribe(event_type, handler_func)
                    logger.debug(f"已取消订阅事件: {event_type}")
                except Exception as e:
                    logger.warning(f"取消订阅事件 {event_type} 失败: {e}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket, "model_training")


@router.websocket("/ws/trading-signals")
async def websocket_trading_signals(websocket: WebSocket):
    """交易信号监控WebSocket连接"""
    await manager.connect(websocket, "trading_signals")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "trading_signals")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "trading_signals")


@router.websocket("/ws/order-routing")
async def websocket_order_routing(websocket: WebSocket, token: str = Query(None)):
    """
    订单路由监控WebSocket连接
    
    提供实时订单路由数据推送，支持事件驱动更新
    量化交易系统合规要求：QTS-015 权限控制
    """
    # 验证token（权限控制）
    if not _validate_websocket_token(token):
        await websocket.close(code=1008, reason="认证失败")
        return
    
    await manager.connect(websocket, "order_routing")
    
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_order_routing_event(event):
            """处理订单路由相关事件（同步函数，内部使用异步发送）"""
            try:
                import asyncio
                import json
                # 创建异步任务发送消息
                message = {
                    "type": "order_routing_event",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.ORDERS_GENERATED,
            EventType.EXECUTION_COMPLETED,
            EventType.POSITION_UPDATED,
            EventType.RISK_CHECK_COMPLETED,
            EventType.SIGNALS_GENERATED,
            EventType.EXECUTION_STARTED
        ]
        
        handlers = []
        try:
            for event_type in event_types:
                try:
                    # 为每个事件类型创建处理函数
                    def create_handler(et):
                        def handler(event):
                            handle_order_routing_event(event)
                        return handler
                    
                    handler_func = create_handler(event_type)
                    # 订阅事件（使用异步处理器）
                    event_bus.subscribe(
                        event_type,
                        handler_func,
                        async_handler=True
                    )
                    handlers.append((event_type, handler_func))
                    logger.debug(f"已订阅订单路由事件: {event_type}")
                except Exception as e:
                    logger.debug(f"订阅事件 {event_type} 失败: {e}")
            
            # 发送初始连接成功消息
            await websocket.send_json({
                "type": "connection_established",
                "message": "订单路由WebSocket连接已建立",
                "timestamp": time.time()
            })
            
            # 保持连接活跃
            while True:
                try:
                    # 接收客户端消息（心跳或控制消息）
                    data = await websocket.receive_text()
                    
                    # 处理pong响应
                    try:
                        import json
                        message = json.loads(data)
                        if message.get("type") == "pong":
                            if websocket in manager._connection_metadata:
                                manager._connection_metadata[websocket]['last_heartbeat'] = time.time()
                            continue
                    except:
                        pass
                    
                    logger.debug(f"收到WebSocket消息: {data}")
                except WebSocketDisconnect:
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
        finally:
            # 取消订阅
            try:
                for event_type, handler_func in handlers:
                    event_bus.unsubscribe(event_type, handler_func)
            except Exception:
                pass
            manager.disconnect(websocket, "order_routing")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket, "order_routing")


@router.websocket("/ws/trading-execution")
async def websocket_trading_execution(websocket: WebSocket):
    """交易执行流程监控WebSocket连接"""
    await manager.connect(websocket, "trading_execution")
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_execution_event(event):
            """处理交易执行相关事件（同步函数，内部使用异步发送）"""
            try:
                import asyncio
                import json
                # 创建异步任务发送消息
                message = {
                    "type": "execution_event",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.ORDERS_GENERATED,
            EventType.EXECUTION_COMPLETED,
            EventType.POSITION_UPDATED,
            EventType.RISK_CHECK_COMPLETED,
            EventType.SIGNALS_GENERATED,
            EventType.EXECUTION_STARTED
        ]
        
        handlers = []  # 在try块外定义，确保finally块可以访问
        try:
            for event_type in event_types:
                try:
                    # 为每个事件类型创建处理函数
                    def create_handler(et):
                        def handler(event):
                            handle_execution_event(event)
                        return handler
                    
                    handler_func = create_handler(event_type)
                    # 订阅事件（使用异步处理器）
                    event_bus.subscribe(
                        event_type,
                        handler_func,
                        async_handler=True
                    )
                    handlers.append((event_type, handler_func))
                    logger.debug(f"已订阅事件: {event_type}")
                except Exception as e:
                    logger.debug(f"订阅事件 {event_type} 失败: {e}")
            
            # 保持连接活跃
            while True:
                try:
                    # 接收客户端消息（心跳或控制消息）
                    data = await websocket.receive_text()
                    logger.debug(f"收到WebSocket消息: {data}")
                except WebSocketDisconnect:
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
        finally:
            # 取消订阅
            try:
                for event_type, handler_func in handlers:
                    event_bus.unsubscribe(event_type, handler_func)
            except Exception:
                pass
            manager.disconnect(websocket, "trading_execution")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket, "trading_execution")


@router.websocket("/ws/data-quality")
async def websocket_data_quality(websocket: WebSocket):
    """数据质量监控WebSocket连接"""
    await manager.connect(websocket, "data_quality")
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_quality_event(event):
            """处理数据质量相关事件"""
            try:
                import asyncio
                import json
                message = {
                    "type": "data_quality",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.DATA_QUALITY_CHECKED,
            EventType.DATA_QUALITY_ALERT,
            EventType.DATA_QUALITY_UPDATED,
            EventType.DATA_COLLECTED,
            EventType.DATA_COLLECTION_PROGRESS  # 添加数据采集进度事件订阅
        ]
        
        handlers = []
        try:
            for event_type in event_types:
                handler_func = lambda event, et=event_type: handle_quality_event(event)
                event_bus.subscribe(event_type, handler_func, async_handler=True)
                handlers.append((event_type, handler_func))
                logger.debug(f"已订阅数据质量事件: {event_type}")
            
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("数据质量WebSocket连接已断开")
        except Exception as e:
            logger.error(f"数据质量WebSocket错误: {e}")
        finally:
            for event_type, handler_func in handlers:
                try:
                    event_bus.unsubscribe(event_type, handler_func)
                except Exception:
                    pass
            manager.disconnect(websocket, "data_quality")
    except Exception as e:
        logger.error(f"数据质量WebSocket连接建立失败: {e}")
        manager.disconnect(websocket, "data_quality")


@router.websocket("/ws/data-performance")
async def websocket_data_performance(websocket: WebSocket):
    """数据性能监控WebSocket连接"""
    await manager.connect(websocket, "data_performance")
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_performance_event(event):
            """处理数据性能相关事件"""
            try:
                import asyncio
                import json
                message = {
                    "type": "data_performance",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.DATA_PERFORMANCE_UPDATED,
            EventType.DATA_PERFORMANCE_ALERT,
            EventType.DATA_COLLECTED
        ]
        
        handlers = []
        try:
            for event_type in event_types:
                handler_func = lambda event, et=event_type: handle_performance_event(event)
                event_bus.subscribe(event_type, handler_func, async_handler=True)
                handlers.append((event_type, handler_func))
                logger.debug(f"已订阅数据性能事件: {event_type}")
            
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("数据性能WebSocket连接已断开")
        except Exception as e:
            logger.error(f"数据性能WebSocket错误: {e}")
        finally:
            for event_type, handler_func in handlers:
                try:
                    event_bus.unsubscribe(event_type, handler_func)
                except Exception:
                    pass
            manager.disconnect(websocket, "data_performance")
    except Exception as e:
        logger.error(f"数据性能WebSocket连接建立失败: {e}")
        manager.disconnect(websocket, "data_performance")


@router.websocket("/ws/risk-control")
async def websocket_risk_control(websocket: WebSocket):
    """风险控制流程监控WebSocket连接"""
    await manager.connect(websocket, "risk_control")
    try:
        # 订阅事件总线的事件
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        
        event_bus = EventBus()
        if not event_bus._initialized:
            event_bus.initialize()
        
        # 定义事件处理函数
        def handle_risk_control_event(event):
            """处理风险控制相关事件（同步函数，内部使用异步发送）"""
            try:
                import asyncio
                import json
                # 创建异步任务发送消息
                message = {
                    "type": "risk_control_event",
                    "event_type": str(event.event_type) if hasattr(event, 'event_type') else str(event.get('event_type', 'unknown')),
                    "data": event.data if hasattr(event, 'data') else event.get('data', {}),
                    "timestamp": event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', time.time())
                }
                message_str = json.dumps(message, ensure_ascii=False)
                asyncio.create_task(manager.send_personal_message(message_str, websocket))
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
        
        # 订阅相关事件
        event_types = [
            EventType.RISK_CHECK_STARTED,
            EventType.RISK_CHECK_COMPLETED,
            EventType.RISK_ASSESSMENT_COMPLETED,
            EventType.RISK_INTERCEPTED,
            EventType.COMPLIANCE_CHECK_COMPLETED,
            EventType.RISK_REPORT_GENERATED,
            EventType.ALERT_TRIGGERED,
            EventType.ALERT_RESOLVED,
            EventType.REAL_TIME_MONITORING_ALERT
        ]
        
        handlers = []  # 在try块外定义，确保finally块可以访问
        try:
            for event_type in event_types:
                try:
                    # 为每个事件类型创建处理函数
                    def create_handler(et):
                        def handler(event):
                            handle_risk_control_event(event)
                        return handler
                    
                    handler_func = create_handler(event_type)
                    # 订阅事件（使用异步处理器）
                    event_bus.subscribe(
                        event_type,
                        handler_func,
                        async_handler=True
                    )
                    handlers.append((event_type, handler_func))
                    logger.debug(f"已订阅风险控制事件: {event_type}")
                except Exception as e:
                    logger.debug(f"订阅事件 {event_type} 失败: {e}")
            
            # 保持连接活跃
            while True:
                try:
                    # 接收客户端消息（心跳或控制消息）
                    data = await websocket.receive_text()
                    logger.debug(f"收到WebSocket消息: {data}")
                except WebSocketDisconnect:
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
        finally:
            # 取消订阅
            try:
                for event_type, handler_func in handlers:
                    event_bus.unsubscribe(event_type, handler_func)
            except Exception:
                pass
            manager.disconnect(websocket, "risk_control")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        manager.disconnect(websocket, "risk_control")


@router.websocket("/ws/architecture-status")
async def websocket_architecture_status(websocket: WebSocket):
    """架构状态监控WebSocket连接"""
    await manager.connect(websocket, "architecture_status")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"收到WebSocket消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, "architecture_status")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket, "architecture_status")