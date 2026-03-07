#!/usr/bin/env python3
"""
RQA2025 实时数据 WebSocket API

from src.infrastructure.logging.core.unified_logger import get_unified_logger
提供实时数据流推送功能
支持实时行情、质量监控、性能指标等数据推送
"""

from src.data import DataManagerSingleton
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Set

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import HTMLResponse
import numpy as np

from src.data import DataManager  # 合理跨层级导入：数据层数据管理器
from src.data.monitoring import PerformanceMonitor  # 合理跨层级导入：数据层性能监控
from src.data.quality import DataQualityMonitor, AdvancedQualityMonitor  # 合理跨层级导入：数据层质量监控
# 使用基础的DataLoader
from src.data.loader import DataLoader
from src.infrastructure.logging.core.unified_logger import get_unified_logger  # 当前层级内部导入：统一日志器

# 配置日志
logging.basicConfig(level=logging.INFO)

# 创建路由器
router = APIRouter(prefix="/ws", tags=["websocket"])

# 初始化组件
data_manager = DataManagerSingleton()
performance_monitor = PerformanceMonitor()
quality_monitor = DataQualityMonitor()
advanced_quality_monitor = AdvancedQualityMonitor()

# 数据加载器实例
base_config = {
    'cache_dir': 'cache',
    'max_retries': 3,
    'timeout': 30
}

# 数据加载器实例 - 使用通用DataLoader
loaders = {
    "crypto": DataLoader(base_config),
    "macro": DataLoader(base_config),
    "options": DataLoader(base_config),
    "bond": DataLoader(base_config),
    "commodity": DataLoader(base_config),
    "forex": DataLoader(base_config)
}


logger = logging.getLogger(__name__)


class ConnectionManager:

    """WebSocket连接管理器"""

    def __init__(self):

        self.active_connections: Dict[str, Set[WebSocket]] = {
            "market_data": set(),
            "quality_monitor": set(),
            "performance_monitor": set(),
            "alerts": set()
        }
        self.subscriptions: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        """建立连接"""
        await websocket.accept()

        # 动态创建channel如果不存在
        if channel not in self.active_connections:
            self.active_connections[channel] = set()

        self.active_connections[channel].add(websocket)
        self.subscriptions[websocket] = {"channel": channel, "subscribed": True}
        logger.info(f"WebSocket连接建立: {channel}")

    def disconnect(self, websocket: WebSocket):
        """断开连接"""
        for channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info("WebSocket连接断开")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, channel: str):
        """广播消息到指定频道"""
        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.add(connection)

        # 清理断开的连接
        for connection in disconnected:
            self.disconnect(connection)


# 创建连接管理器实例
manager = ConnectionManager()


class RealTimeDataStreamer:

    """实时数据流处理器"""

    def __init__(self):

        self.running = False
        self.tasks = []

    async def start_streaming(self):
        """启动数据流"""
        self.running = True
        self.tasks = [
            asyncio.create_task(self._stream_market_data()),
            asyncio.create_task(self._stream_quality_metrics()),
            asyncio.create_task(self._stream_performance_metrics()),
            asyncio.create_task(self._stream_alerts())
        ]
        logger.info("实时数据流启动")

    async def stop_streaming(self):
        """停止数据流"""
        self.running = False
        for task in self.tasks:
            task.cancel()
        logger.info("实时数据流停止")

    async def _stream_market_data(self):
        """流式推送市场数据"""
        while self.running:
            try:
                # 模拟实时市场数据
                market_data = await self._generate_market_data()

                message = {
                    "type": "market_data",
                    "timestamp": datetime.now().isoformat(),
                    "data": market_data
                }

                await manager.broadcast(json.dumps(message), "market_data")
                await asyncio.sleep(1)  # 1秒间隔

            except Exception as e:
                logger.error(f"市场数据流错误: {e}")
                await asyncio.sleep(5)

    async def _stream_quality_metrics(self):
        """流式推送质量指标"""
        while self.running:
            try:
                # 获取质量指标
                quality_metrics = advanced_quality_monitor.get_current_metrics()

                message = {
                    "type": "quality_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": quality_metrics
                }

                await manager.broadcast(json.dumps(message), "quality_monitor")
                await asyncio.sleep(5)  # 5秒间隔

            except Exception as e:
                logger.error(f"质量指标流错误: {e}")
                await asyncio.sleep(10)

    async def _stream_performance_metrics(self):
        """流式推送性能指标"""
        while self.running:
            try:
                # 获取性能指标
                performance_metrics = performance_monitor.get_metrics()

                message = {
                    "type": "performance_metrics",
                    "timestamp": datetime.now().isoformat(),
                    "data": performance_metrics
                }

                await manager.broadcast(json.dumps(message), "performance_monitor")
                await asyncio.sleep(3)  # 3秒间隔

            except Exception as e:
                logger.error(f"性能指标流错误: {e}")
                await asyncio.sleep(10)

    async def _stream_alerts(self):
        """流式推送告警信息"""
        while self.running:
            try:
                # 获取告警信息
                performance_alerts = performance_monitor.get_alerts()
                quality_alerts = advanced_quality_monitor.get_alerts()

                if performance_alerts or quality_alerts:
                    message = {
                        "type": "alerts",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "performance_alerts": performance_alerts,
                            "quality_alerts": quality_alerts
                        }
                    }

                    await manager.broadcast(json.dumps(message), "alerts")

                await asyncio.sleep(10)  # 10秒间隔

            except Exception as e:
                logger.error(f"告警流错误: {e}")
                await asyncio.sleep(30)

    async def _generate_market_data(self) -> Dict[str, Any]:
        """生成模拟市场数据"""
        # 模拟不同数据源的市场数据
        market_data = {}

        # 加密货币数据
        crypto_data = {
            "BTC": {
                "price": 45000 + np.secrets.normal(0, 100),
                "volume": 1000000 + np.secrets.normal(0, 100000),
                "change": np.secrets.normal(0, 0.02)
            },
            "ETH": {
                "price": 3000 + np.secrets.normal(0, 50),
                "volume": 800000 + np.secrets.normal(0, 80000),
                "change": np.secrets.normal(0, 0.015)
            }
        }

        # 外汇数据
        forex_data = {
            "USD / EUR": {
                "rate": 0.85 + np.secrets.normal(0, 0.001),
                "bid": 0.849 + np.secrets.normal(0, 0.0005),
                "ask": 0.851 + np.secrets.normal(0, 0.0005)
            },
            "USD / JPY": {
                "rate": 110.0 + np.secrets.normal(0, 0.1),
                "bid": 109.9 + np.secrets.normal(0, 0.05),
                "ask": 110.1 + np.secrets.normal(0, 0.05)
            }
        }

        # 商品数据
        commodity_data = {
            "GOLD": {
                "price": 1900 + np.secrets.normal(0, 5),
                "volume": 50000 + np.secrets.normal(0, 5000)
            },
            "OIL": {
                "price": 75 + np.secrets.normal(0, 0.5),
                "volume": 100000 + np.secrets.normal(0, 10000)
            }
        }

        market_data = {
            "crypto": crypto_data,
            "forex": forex_data,
            "commodity": commodity_data,
            "timestamp": datetime.now().isoformat()
        }

        return market_data


# 创建实时数据流处理器
streamer = RealTimeDataStreamer()


# WebSocket 端点定义

@router.get("/")
async def get_websocket_page():
    """WebSocket测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RQA2025 WebSocket 测试</title>
        <style>
            body { font - family: Arial, sans - serif; margin: 20px; }
            .container { max - width: 1200px; margin: 0 auto; }
            .channel { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border - radius: 5px; }
            .message { background: #f5f5f5; padding: 10px; margin: 5px 0; border - radius: 3px; }
            .status { padding: 10px; margin: 10px 0; border - radius: 3px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            button { padding: 10px 20px; margin: 5px; border: none; border - radius: 3px; cursor: pointer; }
            .connect { background: #007bff; color: white; }
            .disconnect { background: #dc3545; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RQA2025 WebSocket 实时数据测试</h1>

            <div>
                <button class="connect" onclick="connectAll()">连接所有频道</button>
                <button class="disconnect" onclick="disconnectAll()">断开所有连接</button>
            </div>

            <div class="channel">
                <h3>市场数据 (market_data)</h3>
                <div id="market - status" class="status disconnected">未连接</div>
                <div id="market - messages"></div>
            </div>

            <div class="channel">
                <h3>质量监控 (quality_monitor)</h3>
                <div id="quality - status" class="status disconnected">未连接</div>
                <div id="quality - messages"></div>
            </div>

            <div class="channel">
                <h3>性能监控 (performance_monitor)</h3>
                <div id="performance - status" class="status disconnected">未连接</div>
                <div id="performance - messages"></div>
            </div>

            <div class="channel">
                <h3>告警信息 (alerts)</h3>
                <div id="alerts - status" class="status disconnected">未连接</div>
                <div id="alerts - messages"></div>
            </div>
        </div>

        <script>
            const channels = ['market_data', 'quality_monitor', 'performance_monitor', 'alerts'];
            const connections = {};

            function connectAll() {
                channels.forEach(channel => {
                    if (!connections[channel]) {
                        connect(channel);
                    }
                });
            }

            function disconnectAll() {
                channels.forEach(channel => {
                    if (connections[channel]) {
                        disconnect(channel);
                    }
                });
            }

            function connect(channel) {
                const ws = new WebSocket(`ws://localhost:8000 / ws/${channel}`);

                ws.onopen = function() {
                    document.getElementById(`${channel}-status`).textContent = '已连接';
                    document.getElementById(`${channel}-status`).className = 'status connected';
                };

                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message';
                    messageDiv.textContent = JSON.stringify(message, null, 2);

                    const container = document.getElementById(`${channel}-messages`);
                    container.appendChild(messageDiv);

                    // 保持最新的10条消息
                    if (container.children.length > 10) {
                        container.removeChild(container.firstChild);
                    }
                };

                ws.onclose = function() {
                    document.getElementById(`${channel}-status`).textContent = '连接断开';
                    document.getElementById(`${channel}-status`).className = 'status disconnected';
                    connections[channel] = null;
                };

                connections[channel] = ws;
            }

            function disconnect(channel) {
                if (connections[channel]) {
                    connections[channel].close();
                    connections[channel] = null;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.websocket("/market_data")
async def websocket_market_data(websocket: WebSocket):
    """市场数据WebSocket连接"""
    await manager.connect(websocket, "market_data")
    try:
        while True:
            # 保持连接活跃
            data = await websocket.receive_text()
            # 处理客户端消息（如果需要）
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/quality_monitor")
async def websocket_quality_monitor(websocket: WebSocket):
    """质量监控WebSocket连接"""
    await manager.connect(websocket, "quality_monitor")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/performance_monitor")
async def websocket_performance_monitor(websocket: WebSocket):
    """性能监控WebSocket连接"""
    await manager.connect(websocket, "performance_monitor")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    """告警信息WebSocket连接"""
    await manager.connect(websocket, "alerts")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """通用WebSocket连接端点"""
    if channel not in manager.active_connections:
        await websocket.close(code=4004, reason="Invalid channel")
        return

    await manager.connect(websocket, channel)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理客户端消息
            try:
                message = json.loads(data)
                # 根据消息类型处理
                if message.get("type") == "subscribe":
                    # 订阅特定数据
                    pass
                elif message.get("type") == "unsubscribe":
                    # 取消订阅
                    pass
            except json.JSONDecodeError:
                logger.warning(f"无效的JSON消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# 启动和停止事件处理
@router.on_event("startup")
async def startup_event():
    """应用启动时启动数据流和任务执行器"""
    await streamer.start_streaming()
    
    # 启动特征任务执行器
    try:
        from .feature_task_executor import start_feature_task_executor
        await start_feature_task_executor()
        logger.info("特征任务执行器已启动")
    except Exception as e:
        logger.error(f"启动特征任务执行器失败: {e}")
    
    # 启动模型训练任务执行器
    try:
        from .training_job_executor import start_training_job_executor
        await start_training_job_executor()
        logger.info("模型训练任务执行器已启动")
    except Exception as e:
        logger.error(f"启动模型训练任务执行器失败: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时停止数据流和任务执行器"""
    await streamer.stop_streaming()
    
    # 停止特征任务执行器
    try:
        from .feature_task_executor import stop_feature_task_executor
        await stop_feature_task_executor()
        logger.info("特征任务执行器已停止")
    except Exception as e:
        logger.error(f"停止特征任务执行器失败: {e}")
    
    # 停止模型训练任务执行器
    try:
        from .training_job_executor import stop_training_job_executor
        await stop_training_job_executor()
        logger.info("模型训练任务执行器已停止")
    except Exception as e:
        logger.error(f"停止模型训练任务执行器失败: {e}")
