#!/usr/bin/env python3
"""
最简单的 FastAPI 应用测试
用于排除应用代码干扰，验证 uvicorn 是否能正常启动
"""

from fastapi import FastAPI
import uvicorn

# 创建最简单的应用
app = FastAPI(title="Simple Test App")

@app.get("/")
async def root():
    return {"message": "Hello World", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 启动最简单的 FastAPI 应用测试...")
    print("📍 服务地址: http://0.0.0.0:8000")
    print("❤️  健康检查: http://0.0.0.0:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
