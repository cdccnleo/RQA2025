#!/usr/bin/env python3
"""
简单的WebSocket服务器测试
"""

import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def websocket_handler(websocket, path):
    """WebSocket处理函数"""
    logger.info(f"WebSocket connection established: {path}")

    try:
        # 接受连接
        await websocket.send(json.dumps({
            'type': 'connected',
            'message': 'WebSocket connection accepted'
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"Received message: {data}")

                if data.get('type') == 'subscribe':
                    await websocket.send(json.dumps({
                        'type': 'subscribed',
                        'topics': data.get('topics', []),
                        'timestamp': asyncio.get_event_loop().time()
                    }))
                    logger.info("Subscription confirmed")
                else:
                    await websocket.send(json.dumps({
                        'type': 'echo',
                        'data': data,
                        'timestamp': asyncio.get_event_loop().time()
                    }))

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message: {message}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }))

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")

async def main():
    """主函数"""
    try:
        # 启动WebSocket服务器
        server = await websockets.serve(
            websocket_handler,
            "0.0.0.0",
            8765,
            ping_interval=None  # 禁用ping以简化测试
        )

        logger.info("WebSocket test server started on ws://0.0.0.0:8765")
        print("WebSocket服务器已启动，请在新终端运行测试客户端")
        print("测试命令: python scripts/test_websocket_client.py")
        print("按 Ctrl+C 停止服务器")

        # 保持服务器运行
        await server.wait_closed()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main())