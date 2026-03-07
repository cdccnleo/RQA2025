#!/usr/bin/env python3
"""
RQA2025 简化启动脚本
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from fastapi import FastAPI
    import uvicorn

    # 创建简单的FastAPI应用
    app = FastAPI(
        title="RQA2025 量化交易系统",
        description="企业级量化交易平台",
        version="2.0.0"
    )

    @app.get("/")
    async def root():
        return {"message": "RQA2025 量化交易系统运行中", "version": "2.0.0"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": "2025-01-01T00:00:00Z"}

    @app.get("/docs")
    async def docs():
        return {"message": "API文档", "url": "/docs"}

    if __name__ == "__main__":
        print("🚀 启动RQA2025量化交易系统...")
        print("🌐 访问地址:")
        print("   主应用: http://localhost:8000")
        print("   API文档: http://localhost:8000/docs")
        print("   健康检查: http://localhost:8000/health")
        print()
        print("按 Ctrl+C 停止服务")
        print("=" * 50)

        uvicorn.run(
            "simple_start:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )

except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装必要依赖: pip install fastapi uvicorn")
    sys.exit(1)
