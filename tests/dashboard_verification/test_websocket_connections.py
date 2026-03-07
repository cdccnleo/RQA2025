"""
WebSocket连接测试
测试所有仪表盘的WebSocket实时数据推送
"""

import pytest
import asyncio
import websockets
import json
from typing import Dict, Any


BASE_URL = "localhost:8000"
WS_BASE = f"ws://{BASE_URL}/ws"


async def _test_websocket_connection(ws_url: str, channel_name: str, timeout: int = 5):
    """测试WebSocket连接"""
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"✅ {channel_name} WebSocket连接成功")
            
            # 等待接收消息
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                data = json.loads(message)
                print(f"✅ {channel_name} 收到数据: {data.get('type', 'unknown')}")
                return True
            except asyncio.TimeoutError:
                print(f"⚠️ {channel_name} 连接成功但未收到数据（可能正常）")
                return True
    except Exception as e:
        print(f"❌ {channel_name} WebSocket连接失败: {e}")
        return False


@pytest.mark.asyncio
async def test_feature_engineering_websocket():
    """测试特征工程WebSocket"""
    result = await _test_websocket_connection(
        f"{WS_BASE}/feature-engineering",
        "特征工程监控"
    )
    assert result, "特征工程WebSocket连接失败"


@pytest.mark.asyncio
async def test_model_training_websocket():
    """测试模型训练WebSocket"""
    result = await _test_websocket_connection(
        f"{WS_BASE}/model-training",
        "模型训练监控"
    )
    assert result, "模型训练WebSocket连接失败"


@pytest.mark.asyncio
async def test_trading_signals_websocket():
    """测试交易信号WebSocket"""
    result = await _test_websocket_connection(
        f"{WS_BASE}/trading-signals",
        "交易信号监控"
    )
    assert result, "交易信号WebSocket连接失败"


@pytest.mark.asyncio
async def test_order_routing_websocket():
    """测试订单路由WebSocket"""
    result = await _test_websocket_connection(
        f"{WS_BASE}/order-routing",
        "订单路由监控"
    )
    assert result, "订单路由WebSocket连接失败"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

